"""可复现的 rollout-level dual 调度与分 dual 平均率估计。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch

from hrmr.rl.average_rate import AverageRateEstimator


def _dual_grid(values: Sequence[Real]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("dual_values must be a sequence of real numbers")
    checked = []
    for index, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"dual_values[{index}] must be a real number")
        item = float(value)
        if not np.isfinite(item) or item < 0.0:
            raise ValueError("dual_values must be finite and non-negative")
        checked.append(item)
    result = tuple(checked)
    if not result:
        raise ValueError("dual_values must not be empty")
    if len(set(result)) != len(result) or tuple(sorted(result)) != result:
        raise ValueError("dual_values must be unique and strictly increasing")
    return result


class PersistentDualBatchSchedule:
    """训练开始时平衡分配 dual，此后环境槽位始终保持同一 dual。"""

    def __init__(self, dual_values: Sequence[Real], num_envs: int, seed: int) -> None:
        self.dual_values = _dual_grid(dual_values)
        if isinstance(num_envs, (bool, np.bool_)) or not isinstance(num_envs, Integral):
            raise TypeError("num_envs must be an integer")
        self.num_envs = int(num_envs)
        if self.num_envs <= 0 or self.num_envs % len(self.dual_values) != 0:
            raise ValueError("num_envs must be a positive multiple of the dual-grid size")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = int(seed)
        self.environments_per_dual = self.num_envs // len(self.dual_values)
        assignments = np.repeat(
            np.asarray(self.dual_values, dtype=np.float64),
            self.environments_per_dual,
        )
        # 独立 RNG stream 只在训练开始时生成一次平衡绑定。
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, 0x42415443)))
        rng.shuffle(assignments)
        assignments.setflags(write=False)
        self._assignments = assignments
        self._num_draws = 0

    @property
    def num_draws(self) -> int:
        return self._num_draws

    def next(self) -> np.ndarray:
        """返回固定的逐环境 dual 绑定，调用之间不再重新洗牌。"""

        self._num_draws += 1
        return self._assignments

    def state_dict(self) -> dict[str, Any]:
        return {
            "dual_values": list(self.dual_values),
            "num_envs": self.num_envs,
            "seed": self.seed,
            "assignments": self._assignments.tolist(),
            "num_draws": self._num_draws,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        expected = {"dual_values", "num_envs", "seed", "assignments", "num_draws"}
        if set(state) != expected:
            raise ValueError("persistent dual schedule state has unexpected fields")
        if (
            _dual_grid(state["dual_values"]) != self.dual_values
            or state["num_envs"] != self.num_envs
            or state["seed"] != self.seed
        ):
            raise ValueError("persistent dual schedule state does not match this schedule")
        assignments = np.asarray(state["assignments"], dtype=np.float64)
        if assignments.shape != (self.num_envs,) or not np.array_equal(
            assignments, self._assignments
        ):
            raise ValueError("saved persistent assignments do not match this schedule")
        draws = state["num_draws"]
        if type(draws) is not int or draws < 0:
            raise ValueError("saved persistent schedule num_draws is invalid")
        self._num_draws = draws


class ConditionalAverageRateBank:
    """为每个训练 dual 独立维护一个长期平均率 EMA。"""

    def __init__(self, dual_values: Sequence[Real], ema_alpha: float) -> None:
        self.dual_values = _dual_grid(dual_values)
        self._estimators = {dual: AverageRateEstimator(ema_alpha) for dual in self.dual_values}

    def _estimator(self, dual_lambda: Real) -> AverageRateEstimator:
        if isinstance(dual_lambda, (bool, np.bool_)) or not isinstance(dual_lambda, Real):
            raise TypeError("dual_lambda must be a real number")
        value = float(dual_lambda)
        try:
            return self._estimators[value]
        except KeyError as exc:
            raise ValueError("dual_lambda is not in this rate bank") from exc

    def update(
        self,
        dual_lambda: Real,
        values: torch.Tensor | Iterable[float] | float,
    ) -> float:
        return self._estimator(dual_lambda).update(values)

    def value(self, dual_lambda: Real) -> float:
        return self._estimator(dual_lambda).value

    def num_updates(self, dual_lambda: Real) -> int:
        return self._estimator(dual_lambda).num_updates

    def update_by_environment(
        self,
        dual_lambdas: Sequence[Real] | np.ndarray,
        values: torch.Tensor | np.ndarray,
    ) -> np.ndarray:
        """按环境所属 dual 分组更新，并返回与环境顺序一致的 rho。"""

        assignments = np.asarray(dual_lambdas)
        if assignments.ndim != 1 or assignments.size == 0:
            raise ValueError("dual_lambdas must be a non-empty one-dimensional array")
        checked = assignments.astype(np.float64, copy=False)
        raw_values = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
        signals = np.asarray(raw_values)
        if signals.ndim == 0 or signals.shape[-1] != checked.size:
            raise ValueError("values must use environments as its final dimension")
        if not np.all(np.isfinite(signals)):
            raise ValueError("values must contain only finite values")
        rates = np.empty(checked.shape, dtype=np.float64)
        for dual in self.dual_values:
            mask = checked == dual
            if not np.any(mask):
                raise ValueError("every rate-bank dual must appear in each balanced batch")
            rate = self.update(dual, signals[..., mask].reshape(-1))
            rates[mask] = rate
        rates.setflags(write=False)
        return rates

    def state_dict(self) -> dict[str, Any]:
        return {
            "dual_values": list(self.dual_values),
            "estimators": {
                str(dual): estimator.state_dict() for dual, estimator in self._estimators.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {"dual_values", "estimators"}:
            raise ValueError("conditional rate state has unexpected fields")
        if _dual_grid(state["dual_values"]) != self.dual_values:
            raise ValueError("conditional rate state dual grid does not match")
        estimators = state["estimators"]
        if not isinstance(estimators, Mapping) or set(estimators) != {
            str(value) for value in self.dual_values
        }:
            raise ValueError("conditional rate estimator keys do not match")
        for dual, estimator in self._estimators.items():
            estimator.load_state_dict(estimators[str(dual)])


__all__ = [
    "ConditionalAverageRateBank",
    "PersistentDualBatchSchedule",
]
