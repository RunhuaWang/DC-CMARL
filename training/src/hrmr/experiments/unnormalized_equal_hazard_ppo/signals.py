"""未归一化、等危险参数消融的原始 reward/cost 信号。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from hrmr.constants import TARGET_IDS
from hrmr.geometry import compute_hazard_intensities, compute_target_coverage
from hrmr.types import MonitoringRegions, Position, TargetId, TargetScalars

FloatArray = NDArray[np.float64]


def _ordered_finite_scalars(
    values: TargetScalars,
    target_ids: Sequence[TargetId],
    *,
    name: str,
) -> FloatArray:
    """按给定 target order 读取非负有限标量。"""

    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    ids = tuple(target_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("target_ids must be non-empty and contain no duplicates")
    missing = [target_id for target_id in ids if target_id not in values]
    if missing:
        raise KeyError(f"{name} is missing target ids: {missing}")
    raw = []
    for target_id in ids:
        value = values[target_id]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name}[{target_id}] must be a real number")
        checked = float(value)
        if not isfinite(checked) or checked < 0.0:
            raise ValueError(f"{name}[{target_id}] must be finite and non-negative")
        raw.append(checked)
    return np.asarray(raw, dtype=np.float64)


def compute_unnormalized_team_reward(
    positions: Iterable[Position],
    regions: MonitoringRegions,
    target_values: TargetScalars,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> float:
    """返回按 distinct target coverage 求和的未归一化团队 reward。"""

    ids = tuple(target_ids)
    values = _ordered_finite_scalars(target_values, ids, name="target_values")
    coverage = compute_target_coverage(positions, regions, ids).astype(
        np.float64,
        copy=False,
    )
    return float(np.dot(values, coverage))


def compute_unnormalized_local_costs(
    positions: Iterable[Position],
    regions: MonitoringRegions,
    hazard_intensities: TargetScalars,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> FloatArray:
    """返回每个 agent 的原始暴露成本 ``h(x_i)``，不除以团队规模。"""

    ids = tuple(target_ids)
    _ordered_finite_scalars(
        hazard_intensities,
        ids,
        name="hazard_intensities",
    )
    return compute_hazard_intensities(
        positions,
        regions,
        hazard_intensities,
        ids,
    ).astype(np.float64, copy=False)


def aggregate_unnormalized_global_cost(local_costs: Iterable[float]) -> float:
    """直接求和 local costs；只要求有限且非负，不施加单位上界。"""

    if isinstance(local_costs, (str, bytes)):
        raise TypeError("local_costs must be a one-dimensional numeric iterable")
    try:
        values = local_costs if isinstance(local_costs, np.ndarray) else tuple(local_costs)
        costs = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError("local_costs must be a one-dimensional numeric iterable") from exc
    if costs.ndim != 1 or costs.size == 0:
        raise ValueError("local_costs must be a non-empty one-dimensional array")
    if costs.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("local_costs must contain real numbers")
    checked = costs.astype(np.float64, copy=False)
    if not np.all(np.isfinite(checked)):
        raise ValueError("local_costs must be finite")
    if np.any(checked < 0.0):
        raise ValueError("local_costs must be non-negative")
    return float(np.sum(checked, dtype=np.float64))


__all__ = [
    "aggregate_unnormalized_global_cost",
    "compute_unnormalized_local_costs",
    "compute_unnormalized_team_reward",
]
