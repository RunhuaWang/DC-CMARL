"""无多进程的同步 HRMR 向量环境。

该封装只把多个独立 :class:`~hrmr.environment.HRMREnvironment` 按第一个
维度堆叠。每个子环境拥有独立对象和独立 ``numpy.random.Generator``；
rollout 分段不触发 reset，只有显式调用 :meth:`reset` 才会恢复初始状态。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.config import HRMRConfig
from hrmr.constants import Action
from hrmr.environment import HRMREnvironment
from hrmr.rewards_costs import compute_local_costs
from hrmr.transitions import TransitionResult, validate_actions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
Info = dict[str, Any]
SeedValue = int | None
SeedInput = int | Sequence[SeedValue] | None

VectorReset = tuple[FloatArray, FloatArray, list[Info]]
VectorCurrent = VectorReset
VectorStep = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    BoolArray,
    BoolArray,
    list[Info],
]


def _positive_integer(value: int, *, name: str) -> int:
    """验证正整数，避免把布尔值静默当成数量。"""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _seed_value(value: SeedValue, *, name: str) -> SeedValue:
    """验证 NumPy Generator 可接受的非负整数 seed。"""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer or None")
    seed = int(value)
    if seed < 0:
        raise ValueError(f"{name} must be non-negative")
    return seed


def _expand_seeds(seeds: SeedInput, num_envs: int) -> tuple[SeedValue, ...]:
    """把 base seed 或显式 seed 序列展开到每个子环境。

    单个 base seed 按常见 vector-env 约定扩展为
    ``base_seed + env_index``，使各随机流可复现且彼此不同。
    """

    if seeds is None:
        return (None,) * num_envs
    if isinstance(seeds, (bool, np.bool_)):
        raise TypeError("seeds must be an integer, a seed sequence, or None")
    if isinstance(seeds, Integral):
        base_seed = _seed_value(seeds, name="seeds")
        assert base_seed is not None
        return tuple(base_seed + index for index in range(num_envs))
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be an integer, a seed sequence, or None")
    try:
        values = tuple(seeds)
    except TypeError as exc:
        raise TypeError("seeds must be an integer, a seed sequence, or None") from exc
    if len(values) != num_envs:
        raise ValueError(f"expected {num_envs} seeds, got {len(values)}")
    return tuple(_seed_value(seed, name=f"seeds[{index}]") for index, seed in enumerate(values))


def _config_signature(config: HRMRConfig) -> dict[str, Any]:
    """生成足以拒绝不兼容 checkpoint 的纯 Python 配置签名。"""

    return {
        "grid_width": config.grid_width,
        "grid_height": config.grid_height,
        "num_agents": config.num_agents,
        "monitoring_radius": config.monitoring_radius,
        "action_slip_probability": config.action_slip_probability,
        "initial_positions": tuple(config.initial_positions),
        "target_centers": tuple(config.target_centers.items()),
        "target_values": tuple(config.target_values.items()),
        "hazard_intensities": tuple(config.hazard_intensities.items()),
        "reward_normalizer": config.reward_normalizer,
        "dual_upper_bound": config.dual_upper_bound,
        "use_agent_id": config.use_agent_id,
    }


class SyncVectorHRMR:
    """在当前进程中按 env index 顺序同步执行多个 HRMR 环境。"""

    def __init__(
        self,
        num_envs: int,
        config: HRMRConfig | None = None,
        seeds: SeedInput = None,
    ) -> None:
        self._num_envs = _positive_integer(num_envs, name="num_envs")
        self._envs = tuple(HRMREnvironment(config=config) for _ in range(self._num_envs))
        self._default_seeds = _expand_seeds(seeds, self._num_envs)
        self._last_seeds: tuple[SeedValue, ...] | None = None
        self._has_reset = False

    @property
    def num_envs(self) -> int:
        """子环境数量 ``E``。"""

        return self._num_envs

    @property
    def num_agents(self) -> int:
        """每个子环境中的机器人数量 ``N``。"""

        return self._envs[0].config.num_agents

    @property
    def observation_dim(self) -> int:
        """基础 actor observation 维数，默认配置为 48。"""

        config = self._envs[0].config
        return 48 + (config.num_agents if config.use_agent_id else 0)

    @property
    def state_dim(self) -> int:
        """不含 dual 的 centralized environment-state 维数，基础配置为 16。"""

        config = self._envs[0].config
        return 2 * config.num_agents + config.num_targets

    @property
    def num_targets(self) -> int:
        """Coverage/occupancy 向量长度。"""

        return self._envs[0].config.num_targets

    @property
    def envs(self) -> tuple[HRMREnvironment, ...]:
        """返回子环境对象的只读 tuple，主要用于诊断。"""

        return self._envs

    @property
    def positions(self) -> IntArray:
        """按 ``[E,N,2]`` 返回全部当前内部位置的副本。"""

        return np.stack([environment.positions for environment in self._envs])

    @property
    def last_seeds(self) -> tuple[SeedValue, ...] | None:
        """最近一次显式 reset 实际用于各子环境的 seeds。"""

        return self._last_seeds

    def reset(
        self,
        seeds: SeedInput = None,
        *,
        seed: int | None = None,
    ) -> VectorReset:
        """显式 reset 所有子环境并堆叠结果。

        ``seed`` 是单 base seed 的便利别名；它与 ``seeds`` 不能同时传入。
        若构造时提供了默认 seeds，省略两者时会复用该可复现配置。
        """

        if seed is not None and seeds is not None:
            raise ValueError("pass either seed or seeds, not both")
        if seed is not None:
            selected_seeds = _expand_seeds(seed, self._num_envs)
        elif seeds is not None:
            selected_seeds = _expand_seeds(seeds, self._num_envs)
        else:
            selected_seeds = self._default_seeds

        results = [
            environment.reset(seed=environment_seed)
            for environment, environment_seed in zip(
                self._envs,
                selected_seeds,
                strict=True,
            )
        ]
        observations = np.stack([result[0] for result in results]).astype(
            np.float64,
            copy=False,
        )
        states = np.stack([result[1] for result in results]).astype(
            np.float64,
            copy=False,
        )
        infos = [result[2] for result in results]
        self._last_seeds = selected_seeds
        self._has_reset = True
        return observations, states, infos

    def current(self) -> VectorCurrent:
        """只读构造当前 observation、state 与 info，不推进环境或 RNG。

        ``info`` 中的动作与阻塞诊断采用全 Stay/全 False，表示这是当前状态
        快照而非一次新 transition。该方法尤其用于 checkpoint restore 后核对
        冗余保存的 observation/state/occupancy 是否与环境位置一致。
        """

        results: list[tuple[FloatArray, FloatArray, Info]] = []
        for environment in self._envs:
            positions = environment.positions
            coverage = environment._coverage()
            observations = environment._build_observations(coverage)
            state = environment._build_state(coverage)
            local_costs = compute_local_costs(
                positions,
                environment.monitoring_regions,
                environment.config.hazard_intensities,
                environment.config.num_agents,
                environment.config.target_ids,
            )
            stay_actions = np.full(
                environment.config.num_agents,
                int(Action.STAY),
                dtype=np.int64,
            )
            snapshot_transition = TransitionResult(
                requested_actions=stay_actions,
                executed_actions=stay_actions,
                bounded_candidate_positions=positions,
                final_positions=positions,
                boundary_block_flags=np.zeros(
                    environment.config.num_agents,
                    dtype=np.bool_,
                ),
                collision_flags=np.zeros(
                    environment.config.num_agents,
                    dtype=np.bool_,
                ),
            )
            info = environment._build_info(
                coverage,
                local_costs,
                snapshot_transition,
            )
            results.append((observations, state, info))

        observations = np.stack([result[0] for result in results]).astype(
            np.float64,
            copy=False,
        )
        states = np.stack([result[1] for result in results]).astype(
            np.float64,
            copy=False,
        )
        infos = [result[2] for result in results]
        return observations, states, infos

    def _validate_joint_actions(self, actions: ArrayLike) -> IntArray:
        """在任一子环境移动前原子地验证 ``[E,N]`` 联合动作。"""

        array = np.asarray(actions)
        expected_shape = (self._num_envs, self.num_agents)
        if array.ndim != 2 or array.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}")
        if array.dtype.kind not in {"i", "u"}:
            raise TypeError("actions must contain integers")
        checked = array.astype(np.int64, copy=True)
        for env_index in range(self._num_envs):
            validate_actions(checked[env_index], n_agents=self.num_agents)
        return checked

    def step(self, actions: ArrayLike) -> VectorStep:
        """同步执行 ``[E,N]`` 动作并返回八个批量字段。

        返回顺序为 ``observations, states, rewards, local_costs,
        global_costs, terminated, truncated, infos``。基础 continuing task 的
        ``terminated`` 与 ``truncated`` 均为 shape ``[E]`` 的全 False 数组，
        本封装不会据此自动 reset。
        """

        if not self._has_reset:
            raise RuntimeError("reset must be called before the first vector step")
        checked_actions = self._validate_joint_actions(actions)
        results = [
            environment.step(checked_actions[env_index])
            for env_index, environment in enumerate(self._envs)
        ]

        observations = np.stack([result[0] for result in results]).astype(
            np.float64,
            copy=False,
        )
        states = np.stack([result[1] for result in results]).astype(
            np.float64,
            copy=False,
        )
        rewards = np.asarray([result[2] for result in results], dtype=np.float64)
        local_costs = np.stack([result[3] for result in results]).astype(
            np.float64,
            copy=False,
        )
        terminated = np.asarray([result[4] for result in results], dtype=np.bool_)
        truncated = np.asarray([result[5] for result in results], dtype=np.bool_)
        infos = [result[6] for result in results]
        global_costs = np.asarray(
            [info["global_cost"] for info in infos],
            dtype=np.float64,
        )
        return (
            observations,
            states,
            rewards,
            local_costs,
            global_costs,
            terminated,
            truncated,
            infos,
        )

    def state_dict(self) -> dict[str, Any]:
        """导出可精确继续 rollout 的深拷贝状态。

        除联合位置外，还逐环境保存 bit-generator state。Checkpoint 因此不会
        在 resume 时重启 action-slip 随机流或把 rollout 边界误作 reset。
        """

        rng_states = [deepcopy(environment._rng.bit_generator.state) for environment in self._envs]
        return {
            "format_version": 1,
            "config_signature": deepcopy(_config_signature(self._envs[0].config)),
            "num_envs": self._num_envs,
            "positions": self.positions.copy(),
            "rng_states": rng_states,
            "default_seeds": self._default_seeds,
            "last_seeds": self._last_seeds,
            "seeds": self._last_seeds,
            "has_reset": self._has_reset,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """严格校验并原子恢复 :meth:`state_dict` 生成的状态。"""

        if not isinstance(state, Mapping):
            raise TypeError("vector environment state must be a mapping")
        expected_keys = {
            "format_version",
            "config_signature",
            "num_envs",
            "positions",
            "rng_states",
            "default_seeds",
            "last_seeds",
            "seeds",
            "has_reset",
        }
        missing = sorted(expected_keys - set(state))
        extra = sorted(set(state) - expected_keys)
        if missing or extra:
            raise ValueError(
                f"vector environment state fields mismatch; missing={missing}, extra={extra}"
            )
        if state["format_version"] != 1:
            raise ValueError("unsupported vector environment state format_version")
        if (
            isinstance(state["num_envs"], (bool, np.bool_))
            or not isinstance(state["num_envs"], Integral)
            or int(state["num_envs"]) != self._num_envs
        ):
            raise ValueError("checkpoint num_envs does not match this vector environment")
        if state["config_signature"] != _config_signature(self._envs[0].config):
            raise ValueError("checkpoint HRMR config does not match this vector environment")
        if not isinstance(state["has_reset"], (bool, np.bool_)):
            raise TypeError("checkpoint has_reset must be a boolean")
        has_reset = bool(state["has_reset"])

        positions = np.asarray(state["positions"])
        expected_position_shape = (self._num_envs, self.num_agents, 2)
        if positions.shape != expected_position_shape:
            raise ValueError(f"checkpoint positions must have shape {expected_position_shape}")
        if positions.dtype.kind not in {"i", "u"}:
            raise TypeError("checkpoint positions must contain integers")
        checked_positions = positions.astype(np.int64, copy=True)
        config = self._envs[0].config
        if np.any(
            (checked_positions[:, :, 0] < 0)
            | (checked_positions[:, :, 0] >= config.grid_height)
            | (checked_positions[:, :, 1] < 0)
            | (checked_positions[:, :, 1] >= config.grid_width)
        ):
            raise ValueError("checkpoint positions contain an out-of-grid coordinate")
        for env_index, environment_positions in enumerate(checked_positions):
            if len(np.unique(environment_positions, axis=0)) != self.num_agents:
                raise ValueError(f"checkpoint positions overlap in environment {env_index}")

        default_seeds = _expand_seeds(state["default_seeds"], self._num_envs)
        raw_last_seeds = state["last_seeds"]
        last_seeds = (
            None if raw_last_seeds is None else _expand_seeds(raw_last_seeds, self._num_envs)
        )
        raw_seed_alias = state["seeds"]
        seed_alias = (
            None if raw_seed_alias is None else _expand_seeds(raw_seed_alias, self._num_envs)
        )
        if seed_alias != last_seeds:
            raise ValueError("checkpoint seeds and last_seeds disagree")
        if not has_reset and last_seeds is not None:
            raise ValueError("an unreset checkpoint cannot have last_seeds")

        raw_rng_states = state["rng_states"]
        if isinstance(raw_rng_states, (str, bytes)):
            raise TypeError("checkpoint rng_states must be a sequence")
        try:
            rng_states = tuple(raw_rng_states)
        except TypeError as exc:
            raise TypeError("checkpoint rng_states must be a sequence") from exc
        if len(rng_states) != self._num_envs:
            raise ValueError(f"checkpoint must contain {self._num_envs} RNG states")

        # 先在临时 generators 上验证全部状态，确保失败时对象保持原样。
        restored_generators: list[np.random.Generator] = []
        for env_index, rng_state in enumerate(rng_states):
            generator = np.random.default_rng()
            try:
                generator.bit_generator.state = deepcopy(rng_state)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid RNG state for environment {env_index}") from exc
            restored_generators.append(generator)

        for environment, environment_positions, generator in zip(
            self._envs,
            checked_positions,
            restored_generators,
            strict=True,
        ):
            environment._positions = environment_positions.copy()
            environment._rng = generator
        self._default_seeds = default_seeds
        self._last_seeds = last_seeds
        self._has_reset = has_reset

    def close(self) -> None:
        """同步实现不持有外部资源；提供该方法以便统一生命周期调用。"""


__all__ = ["SyncVectorHRMR", "VectorCurrent", "VectorReset", "VectorStep"]
