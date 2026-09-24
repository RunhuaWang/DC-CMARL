"""按 dual 分离的历史最佳策略函数保持。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

ADAPTIVE_BEST_CALIBRATION_ANCHOR_KL = "adaptive_best_calibration_anchor_kl"


@dataclass(frozen=True)
class RetentionReferenceUpdate:
    """一次 calibration 后各 dual reference 的更新结果。"""

    updated_duals: tuple[float, ...]
    best_scores: tuple[float, ...]


class PerLambdaPolicyRetention:
    """在固定 anchor observations 上保存每个 dual 的历史最佳策略分布。

    Anchor observations 在首次 calibration 前以随机优先级 reservoir 方式收集，
    首次 calibration 后永久冻结。某个 dual 的 calibration objective 创历史新高
    时，只刷新该 dual 的 reference action distribution；不同 dual 不互相覆盖。
    """

    def __init__(
        self,
        dual_values: Sequence[float],
        *,
        num_agents: int,
        observation_dim: int,
        anchor_capacity: int,
        improvement_tolerance: float,
        seed: int,
        device: str | torch.device,
        adaptive_initial_coefficient: float | None = None,
        adaptive_target_kl: float | None = None,
        adaptive_rate: float = 0.0,
        adaptive_min_coefficient: float = 0.0,
        adaptive_max_coefficient: float = float("inf"),
    ) -> None:
        values = tuple(float(value) for value in dual_values)
        if not values or len(set(values)) != len(values):
            raise ValueError("dual_values must be non-empty and unique")
        if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be a positive integer")
        if (
            isinstance(observation_dim, bool)
            or not isinstance(observation_dim, int)
            or observation_dim <= 0
        ):
            raise ValueError("observation_dim must be a positive integer")
        if (
            isinstance(anchor_capacity, bool)
            or not isinstance(anchor_capacity, int)
            or anchor_capacity <= 0
        ):
            raise ValueError("anchor_capacity must be a positive integer")
        if not np.isfinite(improvement_tolerance) or improvement_tolerance < 0.0:
            raise ValueError("improvement_tolerance must be finite and non-negative")
        adaptive_values = (adaptive_initial_coefficient, adaptive_target_kl)
        if any(value is not None for value in adaptive_values):
            if not all(value is not None for value in adaptive_values):
                raise ValueError(
                    "adaptive_initial_coefficient and adaptive_target_kl must be set together"
                )
            numeric_values = (
                adaptive_initial_coefficient,
                adaptive_target_kl,
                adaptive_rate,
                adaptive_min_coefficient,
                adaptive_max_coefficient,
            )
            if not all(value is not None and np.isfinite(value) for value in numeric_values):
                raise ValueError("adaptive KL settings must be finite")
            assert adaptive_initial_coefficient is not None
            assert adaptive_target_kl is not None
            if (
                adaptive_initial_coefficient <= 0.0
                or adaptive_target_kl <= 0.0
                or adaptive_rate <= 0.0
                or adaptive_min_coefficient <= 0.0
                or adaptive_max_coefficient < adaptive_min_coefficient
                or not adaptive_min_coefficient
                <= adaptive_initial_coefficient
                <= adaptive_max_coefficient
            ):
                raise ValueError("invalid adaptive KL coefficient range or update setting")
        elif (
            adaptive_rate != 0.0
            or adaptive_min_coefficient != 0.0
            or np.isfinite(adaptive_max_coefficient)
        ):
            raise ValueError("adaptive KL settings require an initial coefficient and target")

        self.dual_values = values
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.anchor_capacity = anchor_capacity
        self.improvement_tolerance = float(improvement_tolerance)
        self.device = torch.device(device)
        self.adaptive_kl = adaptive_target_kl is not None
        self.adaptive_target_kl = None if adaptive_target_kl is None else float(adaptive_target_kl)
        self.adaptive_rate = float(adaptive_rate)
        self.adaptive_min_coefficient = float(adaptive_min_coefficient)
        self.adaptive_max_coefficient = float(adaptive_max_coefficient)
        self._initial_coefficient = (
            None if adaptive_initial_coefficient is None else float(adaptive_initial_coefficient)
        )
        self._adaptive_coefficients = (
            None
            if adaptive_initial_coefficient is None
            else np.full(len(values), adaptive_initial_coefficient, dtype=np.float64)
        )
        self._dual_to_index = {value: index for index, value in enumerate(values)}
        self._rng = np.random.default_rng(seed)
        self._anchors: list[np.ndarray] = [
            np.empty((0, num_agents, observation_dim), dtype=np.float32) for _ in values
        ]
        self._priorities: list[np.ndarray] = [np.empty((0,), dtype=np.float64) for _ in values]
        self._reference_log_probs: list[np.ndarray | None] = [None for _ in values]
        self._best_scores = np.full(len(values), -np.inf, dtype=np.float64)
        self._frozen = False
        self._anchor_tensors_by_agent: tuple[torch.Tensor, ...] | None = None
        self._reference_tensors_by_agent: tuple[torch.Tensor, ...] | None = None

    @property
    def active(self) -> bool:
        """所有 dual 都已有可用于 loss 的 reference 时返回 true。"""

        return self._frozen and all(value is not None for value in self._reference_log_probs)

    @property
    def best_scores(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._best_scores)

    def observe(self, observations: np.ndarray, dual_assignments: np.ndarray) -> None:
        """首次 calibration 前从真实 rollout observations 建立均匀 reservoir。"""

        if self._frozen:
            return
        array = np.asarray(observations, dtype=np.float32)
        assignments = np.asarray(dual_assignments, dtype=np.float64)
        if array.ndim != 4 or array.shape[2:] != (self.num_agents, self.observation_dim):
            raise ValueError(
                "observations must have shape [steps, envs, num_agents, observation_dim]"
            )
        if assignments.shape != (array.shape[1],):
            raise ValueError("dual_assignments must contain one value per environment")
        if not np.all(np.isfinite(array)):
            raise ValueError("observations must contain only finite values")

        for dual, index in self._dual_to_index.items():
            environment_mask = assignments == dual
            if not np.any(environment_mask):
                raise ValueError(f"rollout is missing retention samples for dual={dual:g}")
            candidates = array[:, environment_mask].reshape(
                -1,
                self.num_agents,
                self.observation_dim,
            )
            priorities = self._rng.random(candidates.shape[0])
            combined_anchors = np.concatenate((self._anchors[index], candidates), axis=0)
            combined_priorities = np.concatenate((self._priorities[index], priorities), axis=0)
            keep_count = min(self.anchor_capacity, combined_anchors.shape[0])
            if keep_count < combined_anchors.shape[0]:
                keep = np.argpartition(combined_priorities, -keep_count)[-keep_count:]
                order = np.argsort(combined_priorities[keep])[::-1]
                keep = keep[order]
                combined_anchors = combined_anchors[keep]
                combined_priorities = combined_priorities[keep]
            self._anchors[index] = np.ascontiguousarray(combined_anchors, dtype=np.float32)
            self._priorities[index] = np.ascontiguousarray(combined_priorities, dtype=np.float64)

    def update_references(
        self,
        actor,
        calibration_objectives: Mapping[float, float],
    ) -> RetentionReferenceUpdate:
        """以每个 dual 的历史最佳 calibration 策略刷新 reference。"""

        supplied = {float(key): float(value) for key, value in calibration_objectives.items()}
        if set(supplied) != set(self.dual_values):
            raise ValueError(
                "calibration_objectives must contain every configured dual exactly once"
            )
        if not all(np.isfinite(value) for value in supplied.values()):
            raise ValueError("calibration objectives must be finite")
        if not self._frozen:
            sizes = tuple(anchor.shape[0] for anchor in self._anchors)
            if any(size != self.anchor_capacity for size in sizes):
                raise RuntimeError(
                    "cannot activate retention before every dual fills its anchor reservoir"
                )
            self._frozen = True
            self._rebuild_anchor_tensors()

        updated: list[float] = []
        actor_was_training = actor.training
        actor.eval()
        with torch.inference_mode():
            for dual, index in self._dual_to_index.items():
                score = supplied[dual]
                if score <= self._best_scores[index] + self.improvement_tolerance:
                    continue
                anchors = torch.as_tensor(
                    self._anchors[index],
                    dtype=torch.float32,
                    device=self.device,
                )
                reference = functional.log_softmax(actor(anchors), dim=-1)
                self._reference_log_probs[index] = (
                    reference.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                self._best_scores[index] = score
                updated.append(dual)
        actor.train(actor_was_training)
        self._rebuild_reference_tensors()
        return RetentionReferenceUpdate(tuple(updated), self.best_scores)

    def task_tensors(self, agent_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回按 ``dual_values`` 排列的 anchor 与 reference log-probabilities。"""

        if not self.active:
            raise RuntimeError("policy retention reference is not active")
        if not 0 <= agent_index < self.num_agents:
            raise IndexError("agent_index is out of range")
        assert self._anchor_tensors_by_agent is not None
        assert self._reference_tensors_by_agent is not None
        return (
            self._anchor_tensors_by_agent[agent_index],
            self._reference_tensors_by_agent[agent_index],
        )

    def task_coefficients(self, base_coefficient: float) -> torch.Tensor:
        """返回每个 dual 当前使用的 KL 系数。"""

        if not self.active:
            raise RuntimeError("policy retention reference is not active")
        if not 0.0 < base_coefficient < float("inf"):
            raise ValueError("base_coefficient must be finite and positive")
        if self.adaptive_kl:
            assert self._initial_coefficient is not None
            assert self._adaptive_coefficients is not None
            if not np.isclose(base_coefficient, self._initial_coefficient):
                raise ValueError("base_coefficient does not match adaptive initial coefficient")
            return torch.as_tensor(
                self._adaptive_coefficients,
                dtype=torch.float32,
                device=self.device,
            )
        return torch.full(
            (len(self.dual_values),),
            float(base_coefficient),
            dtype=torch.float32,
            device=self.device,
        )

    def adapt_coefficients(self, measured_kls: Sequence[float]) -> tuple[float, ...]:
        """按每个 dual 的 anchor KL 独立调整下一次更新的约束强度。

        乘法控制器把 KL 拉向统一目标；单次最多放大或缩小两倍，避免一次异常
        minibatch 让系数失稳。该更新不读取解析模式或理论切换点。
        """

        if not self.adaptive_kl:
            raise RuntimeError("adaptive KL control is disabled")
        values = np.asarray(tuple(measured_kls), dtype=np.float64)
        if values.shape != (len(self.dual_values),):
            raise ValueError("measured_kls must contain one value per configured dual")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("measured_kls must be finite and non-negative")
        assert self.adaptive_target_kl is not None
        assert self._adaptive_coefficients is not None
        log_step = self.adaptive_rate * (values / self.adaptive_target_kl - 1.0)
        log_step = np.clip(log_step, -np.log(2.0), np.log(2.0))
        self._adaptive_coefficients = np.clip(
            self._adaptive_coefficients * np.exp(log_step),
            self.adaptive_min_coefficient,
            self.adaptive_max_coefficient,
        )
        return tuple(float(value) for value in self._adaptive_coefficients)

    def state_dict(self) -> dict:
        """保存 checkpoint 审计所需的完整 retention 状态。"""

        return {
            "dual_values": self.dual_values,
            "num_agents": self.num_agents,
            "observation_dim": self.observation_dim,
            "anchor_capacity": self.anchor_capacity,
            "improvement_tolerance": self.improvement_tolerance,
            "adaptive_kl": self.adaptive_kl,
            "adaptive_initial_coefficient": self._initial_coefficient,
            "adaptive_target_kl": self.adaptive_target_kl,
            "adaptive_rate": self.adaptive_rate,
            "adaptive_min_coefficient": self.adaptive_min_coefficient,
            "adaptive_max_coefficient": self.adaptive_max_coefficient,
            "adaptive_coefficients": (
                None if self._adaptive_coefficients is None else self._adaptive_coefficients.copy()
            ),
            "anchors": tuple(anchor.copy() for anchor in self._anchors),
            "priorities": tuple(priority.copy() for priority in self._priorities),
            "reference_log_probs": tuple(
                None if value is None else value.copy() for value in self._reference_log_probs
            ),
            "best_scores": self._best_scores.copy(),
            "frozen": self._frozen,
            "rng_state": self._rng.bit_generator.state,
        }

    def _rebuild_anchor_tensors(self) -> None:
        stacked = np.stack(self._anchors, axis=0)
        tensor = torch.as_tensor(stacked, dtype=torch.float32, device=self.device)
        self._anchor_tensors_by_agent = tuple(
            tensor[:, :, agent_index, :] for agent_index in range(self.num_agents)
        )

    def _rebuild_reference_tensors(self) -> None:
        if not all(value is not None for value in self._reference_log_probs):
            self._reference_tensors_by_agent = None
            return
        stacked = np.stack(self._reference_log_probs, axis=0)
        tensor = torch.as_tensor(stacked, dtype=torch.float32, device=self.device)
        self._reference_tensors_by_agent = tuple(
            tensor[:, :, agent_index, :] for agent_index in range(self.num_agents)
        )


__all__ = [
    "ADAPTIVE_BEST_CALIBRATION_ANCHOR_KL",
    "PerLambdaPolicyRetention",
    "RetentionReferenceUpdate",
]
