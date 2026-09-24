"""HRMR 的原始团队奖励与环境暴露成本。

Reward 和 cost 均直接根据冲突解析后的最终位置计算。本模块不读取约束
阈值，也不会用 dual 改写环境信号或状态转移。Phase 1 明确不实现
deployment dual update。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from hrmr.constants import TARGET_IDS
from hrmr.geometry import compute_hazard_intensities, compute_target_coverage
from hrmr.types import MonitoringRegions, Position, TargetId, TargetScalars

FloatArray = NDArray[np.float64]


def _finite_real(value: Real, *, name: str) -> float:
    """验证并返回有限实数，明确拒绝 ``bool``。"""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _ordered_scalars(
    values: TargetScalars,
    target_ids: Sequence[TargetId],
    *,
    name: str,
    lower_bound: float,
    upper_bound: float | None = None,
    strictly_positive: bool = False,
) -> FloatArray:
    """按目标规范顺序验证并读取 scalar mapping。"""

    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    ids = tuple(target_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("target_ids must not contain duplicates")
    missing = [target_id for target_id in ids if target_id not in values]
    if missing:
        raise KeyError(f"{name} is missing target ids: {missing}")

    result = np.asarray(
        [_finite_real(values[target_id], name=f"{name}[{target_id}]") for target_id in ids],
        dtype=np.float64,
    )
    if strictly_positive:
        if np.any(result <= lower_bound):
            raise ValueError(f"all {name} entries must be greater than {lower_bound}")
    elif np.any(result < lower_bound):
        raise ValueError(f"all {name} entries must be at least {lower_bound}")
    if upper_bound is not None and np.any(result > upper_bound):
        raise ValueError(f"all {name} entries must be at most {upper_bound}")
    return result


def _validated_num_agents(num_agents: int | None, observed_count: int) -> int:
    """确定 local-cost 归一化所用的团队规模 ``N``。"""

    if observed_count <= 0:
        raise ValueError("positions must contain at least one agent")
    if num_agents is None:
        return observed_count
    if isinstance(num_agents, (bool, np.bool_)) or not isinstance(num_agents, Integral):
        raise TypeError("num_agents must be an integer")
    result = int(num_agents)
    if result <= 0:
        raise ValueError("num_agents must be positive")
    if result != observed_count:
        raise ValueError(
            f"num_agents must equal the number of final positions ({result} != {observed_count})"
        )
    return result


def compute_team_reward(
    positions: Iterable[Position],
    regions: MonitoringRegions,
    target_values: TargetScalars,
    reward_normalizer: float,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> float:
    """根据最终位置计算 PDF 式 (5.5) 的原始共享 team reward。

    Coverage 在函数内部由 ``positions`` 重新计算，因此同一区域即使有多个
    机器人也只贡献一次价值。调用方应传入同步冲突处理后的最终位置。
    """

    normalizer = _finite_real(reward_normalizer, name="reward_normalizer")
    if normalizer <= 0.0:
        raise ValueError("reward_normalizer must be positive")

    ids = tuple(target_ids)
    values = _ordered_scalars(
        target_values,
        ids,
        name="target_values",
        lower_bound=0.0,
        strictly_positive=True,
    )
    coverage = compute_target_coverage(positions, regions, ids).astype(
        np.float64,
        copy=False,
    )
    reward = float(np.dot(values, coverage) / normalizer)

    # 配置验证通常已保证这个界，但这里在纯函数边界再次防御。
    tolerance = 1e-12
    if reward < -tolerance or reward > 1.0 + tolerance:
        raise ValueError(
            "computed team reward lies outside [0, 1]; check target values and reward_normalizer"
        )
    return float(np.clip(reward, 0.0, 1.0))


def compute_local_costs(
    positions: Iterable[Position],
    regions: MonitoringRegions,
    hazard_intensities: TargetScalars,
    num_agents: int | None = None,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> FloatArray:
    """根据最终位置计算每个机器人的 PDF 式 (6.3) local cost。

    返回顺序与 ``positions`` 的 agent 顺序一致，且 dtype 固定为
    ``float64``。即使多个机器人位于同一危险区域，每个机器人仍独立产生
    ``h(x_i) / N`` 成本。
    """

    # 先物化，避免 generator 被团队规模检查与 hazard 查询消费两次。
    try:
        final_positions = tuple(positions)
    except TypeError as exc:
        raise TypeError("positions must be an iterable of coordinates") from exc
    team_size = _validated_num_agents(num_agents, len(final_positions))
    ids = tuple(target_ids)
    _ordered_scalars(
        hazard_intensities,
        ids,
        name="hazard_intensities",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    hazards = compute_hazard_intensities(
        final_positions,
        regions,
        hazard_intensities,
        ids,
    )
    return np.asarray(hazards / float(team_size), dtype=np.float64)


def aggregate_global_cost(local_costs: Iterable[float]) -> float:
    """按 PDF 式 (6.5) 将 agent-local costs 相加为 global cost。"""

    if isinstance(local_costs, (str, bytes)):
        raise TypeError("local_costs must be a one-dimensional numeric iterable")
    try:
        raw_costs = local_costs if isinstance(local_costs, np.ndarray) else tuple(local_costs)
        costs = np.asarray(raw_costs)
    except (TypeError, ValueError) as exc:
        raise TypeError("local_costs must be a one-dimensional numeric iterable") from exc
    if costs.ndim != 1:
        raise ValueError("local_costs must be one-dimensional")
    if costs.size == 0:
        raise ValueError("local_costs must not be empty")
    if costs.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("local_costs must contain real numbers")
    checked = costs.astype(np.float64, copy=False)
    if not np.all(np.isfinite(checked)):
        raise ValueError("local_costs must be finite")
    if np.any(checked < 0.0):
        raise ValueError("local_costs must be non-negative")
    global_cost = float(np.sum(checked, dtype=np.float64))
    if global_cost > 1.0 + 1e-12:
        raise ValueError("global cost must not exceed 1")
    return global_cost


__all__ = [
    "aggregate_global_cost",
    "compute_local_costs",
    "compute_team_reward",
]
