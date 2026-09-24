"""局部 dual copies 的有限窗口成本估计与投影更新。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .communication import Graph, aggregate_global_costs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DualUpdateResult:
    """一个长度 H 窗口结束后的全部本地计算结果。"""

    local_cost_estimates: FloatArray
    global_cost_estimates: FloatArray
    dual_before: FloatArray
    dual_after: FloatArray


def projected_dual_update(
    dual_values: ArrayLike,
    global_cost_estimates: ArrayLike,
    constraint_threshold: Real,
    step_size: Real,
    dual_min: Real,
    dual_max: Real,
) -> FloatArray:
    """逐 agent 执行 Π_[dual_min,dual_max](λ+η(Ĉ-c))。"""

    duals = np.asarray(dual_values, dtype=np.float64)
    costs = np.asarray(global_cost_estimates, dtype=np.float64)
    if duals.ndim != 1 or costs.shape != duals.shape or duals.size == 0:
        raise ValueError("dual_values and global_cost_estimates must be matching vectors")
    scalars = (constraint_threshold, step_size, dual_min, dual_max)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in scalars):
        raise TypeError("threshold, step size, and dual bounds must be real numbers")
    threshold, eta, lower, upper = (float(value) for value in scalars)
    if not np.all(np.isfinite(duals)) or not np.all(np.isfinite(costs)):
        raise ValueError("dual values and global cost estimates must be finite")
    if not all(np.isfinite(value) for value in (threshold, eta, lower, upper)):
        raise ValueError("threshold, step size, and dual bounds must be finite")
    if threshold < 0.0 or eta <= 0.0 or lower < 0.0 or upper <= lower:
        raise ValueError("dual-update scalars are outside their valid ranges")
    if np.any((duals < lower) | (duals > upper)) or np.any(costs < 0.0):
        raise ValueError("dual values or cost estimates are outside their valid ranges")
    result = np.clip(duals + eta * (costs - threshold), lower, upper)
    result.setflags(write=False)
    return result


class DecentralizedDualController:
    """每个 agent 仅积累 local cost，并在窗口末通过邻居通信更新 λ。"""

    def __init__(
        self,
        *,
        graph: Graph,
        communication_rounds: int,
        initial_dual: float,
        constraint_threshold: float,
        step_size: float,
        dual_min: float,
        dual_max: float,
        horizon: int,
    ) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.graph = graph
        self.communication_rounds = communication_rounds
        self.constraint_threshold = float(constraint_threshold)
        self.step_size = float(step_size)
        self.dual_min = float(dual_min)
        self.dual_max = float(dual_max)
        self.horizon = horizon
        self._duals = np.full(len(graph), float(initial_dual), dtype=np.float64)
        # 构造时复用更新函数执行完整标量/边界校验。
        projected_dual_update(
            self._duals,
            np.zeros(len(graph), dtype=np.float64),
            self.constraint_threshold,
            self.step_size,
            self.dual_min,
            self.dual_max,
        )
        self._local_cost_sums = np.zeros(len(graph), dtype=np.float64)
        self._steps = 0

    @property
    def dual_values(self) -> FloatArray:
        result = self._duals.copy()
        result.setflags(write=False)
        return result

    @property
    def steps_in_window(self) -> int:
        return self._steps

    def observe(self, local_costs: ArrayLike) -> DualUpdateResult | None:
        """记录一步 local costs；恰满 H 步时通信、更新并清空窗口。"""

        raw = np.asarray(local_costs)
        if raw.shape != self._local_cost_sums.shape or raw.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("local_costs must contain one numeric value per agent")
        values = raw.astype(np.float64, copy=False)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("local_costs must be finite and non-negative")
        self._local_cost_sums += values
        self._steps += 1
        if self._steps < self.horizon:
            return None
        if self._steps != self.horizon:
            raise AssertionError("dual controller crossed its finite-time horizon")

        local_estimates = self._local_cost_sums / float(self.horizon)
        global_estimates = aggregate_global_costs(
            local_estimates,
            self.graph,
            self.communication_rounds,
        )
        before = self._duals.copy()
        after = projected_dual_update(
            before,
            global_estimates,
            self.constraint_threshold,
            self.step_size,
            self.dual_min,
            self.dual_max,
        )
        self._duals = after.copy()
        self._local_cost_sums.fill(0.0)
        self._steps = 0
        return DualUpdateResult(
            local_cost_estimates=local_estimates.copy(),
            global_cost_estimates=global_estimates.copy(),
            dual_before=before,
            dual_after=after.copy(),
        )


__all__ = [
    "DecentralizedDualController",
    "DualUpdateResult",
    "projected_dual_update",
]
