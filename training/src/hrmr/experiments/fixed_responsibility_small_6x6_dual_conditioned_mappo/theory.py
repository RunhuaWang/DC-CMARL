"""固定责任区在任意非负 λ 下的解析分配与模式判定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.constants import TARGET_IDS
from hrmr.experiments.assigned_targets_ppo.environment import DEFAULT_ASSIGNMENTS

BoolArray = NDArray[np.bool_]
MODE_LABELS = ("4H", "1S+3H", "2S+2H", "3S+1H", "4S")
MODE_ALLOCATIONS = {
    "4H": ("H1", "H3", "H2", "H4"),
    "1S+3H": ("S1", "H3", "H2", "H4"),
    "2S+2H": ("S1", "H3", "S2", "H4"),
    "3S+1H": ("S1", "S4", "S2", "H4"),
    "4S": ("S1", "S4", "S2", "S3"),
}
_TARGET_REWARDS = {
    "S1": 1.0,
    "S2": 1.0,
    "S3": 1.0,
    "S4": 1.0,
    "H1": 1.8,
    "H2": 2.2,
    "H3": 2.2,
    "H4": 1.8,
}
_TARGET_COSTS = {
    "S1": 0.0,
    "S2": 0.0,
    "S3": 0.0,
    "S4": 0.0,
    "H1": 0.8,
    "H2": 0.6,
    "H3": 0.4,
    "H4": 0.2,
}


@dataclass(frozen=True)
class DualAnalyticReference:
    """一个 λ 的全部最优责任分配；切换点可以同时存在两个模式。"""

    dual_lambda: float
    scalarized_objective: float
    allocations: tuple[tuple[str, ...], ...]
    rewards: tuple[float, ...]
    costs: tuple[float, ...]
    mode_labels: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _checked_lambda(value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("dual_lambda must be a real number")
    checked = float(value)
    if not isfinite(checked) or checked < 0.0:
        raise ValueError("dual_lambda must be finite and non-negative")
    return checked


def _mode_label(allocation: tuple[str, ...]) -> str:
    safe_count = sum(target.startswith("S") for target in allocation)
    hazardous_count = len(allocation) - safe_count
    if safe_count == 0:
        return "4H"
    if hazardous_count == 0:
        return "4S"
    return f"{safe_count}S+{hazardous_count}H"


def analytic_reference(dual_lambda: Real) -> DualAnalyticReference:
    """枚举每个 agent 的 safe/hazard 二选一并返回所有最优分配。"""

    dual = _checked_lambda(dual_lambda)
    candidates = []
    for choices in product((0, 1), repeat=len(DEFAULT_ASSIGNMENTS)):
        allocation = tuple(
            DEFAULT_ASSIGNMENTS[agent_index][choice] for agent_index, choice in enumerate(choices)
        )
        reward = sum(_TARGET_REWARDS[target] for target in allocation)
        cost = sum(_TARGET_COSTS[target] for target in allocation)
        candidates.append((reward - dual * cost, allocation, reward, cost))
    optimum = max(candidate[0] for candidate in candidates)
    selected = [candidate for candidate in candidates if abs(candidate[0] - optimum) <= 1e-12]
    return DualAnalyticReference(
        dual_lambda=dual,
        scalarized_objective=optimum,
        allocations=tuple(candidate[1] for candidate in selected),
        rewards=tuple(candidate[2] for candidate in selected),
        costs=tuple(candidate[3] for candidate in selected),
        mode_labels=tuple(dict.fromkeys(_mode_label(candidate[1]) for candidate in selected)),
    )


def allocation_mask(
    target_occupancies: ArrayLike,
    allocation: tuple[str, ...],
    target_ids: tuple[str, ...] = TARGET_IDS,
) -> BoolArray:
    """判断是否精确覆盖给定四目标 allocation。"""

    values = np.asarray(target_occupancies)
    ids = tuple(target_ids)
    if values.ndim == 0 or values.shape[-1] != len(ids):
        raise ValueError("target_occupancies must end with the target dimension")
    if not np.all(np.isfinite(values)) or not np.all((values == 0) | (values == 1)):
        raise ValueError("target_occupancies must be finite and binary")
    required = tuple(ids.index(target) for target in allocation)
    result = np.sum(values, axis=-1) == len(allocation)
    result &= np.all(values[..., required] == 1, axis=-1)
    return np.asarray(result, dtype=np.bool_)


def optimal_mode_mask(
    target_occupancies: ArrayLike,
    target_ids: tuple[str, ...] = TARGET_IDS,
    *,
    fixed_lambda: Real,
) -> BoolArray:
    """接受该 λ 下任一解析最优分配，包含切换点的并列最优。"""

    reference = analytic_reference(fixed_lambda)
    masks = [
        allocation_mask(target_occupancies, allocation, tuple(target_ids))
        for allocation in reference.allocations
    ]
    return np.logical_or.reduce(masks)


def all_mode_rates(
    target_occupancies: ArrayLike,
    target_ids: tuple[str, ...] = TARGET_IDS,
) -> dict[str, float]:
    """报告五类责任分配的出现率，不依赖当前 λ。"""

    values = np.asarray(target_occupancies)
    return {
        label: float(np.mean(allocation_mask(values, allocation, tuple(target_ids))))
        for label, allocation in MODE_ALLOCATIONS.items()
    }


__all__ = [
    "MODE_ALLOCATIONS",
    "MODE_LABELS",
    "DualAnalyticReference",
    "all_mode_rates",
    "allocation_mask",
    "analytic_reference",
    "optimal_mode_mask",
]
