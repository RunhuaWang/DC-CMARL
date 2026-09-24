"""Phase 2 fixed-lambda 解析 operating-mode 参考值与判定。"""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.constants import TARGET_IDS

REWARD_ONLY_LAMBDA = 0.0
FIXED_LAMBDAS = (REWARD_ONLY_LAMBDA, 0.25, 0.75, 1.25, 1.75, 2.25)
TRAINABLE_FIXED_LAMBDAS = FIXED_LAMBDAS
LARGE_DUAL_DIAGNOSTIC_LAMBDA = 4.0
DIAGNOSTIC_FIXED_LAMBDAS = (LARGE_DUAL_DIAGNOSTIC_LAMBDA,)

MODE_REQUIRED_OCCUPANCY = 0.80
MODE_EXCLUDED_OCCUPANCY = 0.20
MODE_TARGET_COUNT_TOLERANCE = 0.20
MODE_DISTINCT_TARGET_THRESHOLD = 3.80
MODE_RATE_THRESHOLD = 0.75
MODE_REWARD_COST_TOLERANCE = 0.08
MODE_SCALARIZED_GAP_TOLERANCE = 0.08


@dataclass(frozen=True)
class AnalyticMode:
    """一个固定 lambda 对应的解析最优长期分配。"""

    fixed_lambda: float
    reward: float
    cost: float
    scalarized_objective: float
    hazardous_targets: tuple[str, ...]
    safe_target_count: int


@dataclass(frozen=True)
class OperatingModeAssessment:
    """基于长期 occupancy 与 R/C/F 的解析 mode 判定。

    Safe targets 物理身份等价，因此只约束平均 safe 覆盖数，
    不要求固定的 ``S_i`` 组合。
    """

    fixed_lambda: float
    mean_hazardous_targets_covered: float
    mean_safe_targets_covered: float
    required_hazardous_targets_covered: bool
    excluded_hazardous_targets_exited: bool
    safe_target_count_matches: bool
    four_effective_targets: bool
    operating_mode_rate_matches: bool
    reward_matches: bool
    cost_matches: bool
    scalarized_gap_within_tolerance: bool
    success: bool


ANALYTIC_MODES = {
    0.00: AnalyticMode(0.00, 1.00, 0.50, 1.0000, ("H1", "H2", "H3", "H4"), 0),
    0.25: AnalyticMode(0.25, 1.00, 0.50, 0.8750, ("H1", "H2", "H3", "H4"), 0),
    0.75: AnalyticMode(0.75, 0.90, 0.30, 0.6750, ("H2", "H3", "H4"), 1),
    1.25: AnalyticMode(1.25, 0.75, 0.15, 0.5625, ("H3", "H4"), 2),
    1.75: AnalyticMode(1.75, 0.60, 0.05, 0.5125, ("H4",), 3),
    2.25: AnalyticMode(2.25, 0.50, 0.00, 0.5000, (), 4),
    # 该点超出正式 Lambda=2.5，仅用于 all-safe 可学习性诊断。正式
    # ``FIXED_LAMBDAS``/结果曲线故意不包含它。
    4.00: AnalyticMode(4.00, 0.50, 0.00, 0.5000, (), 4),
}


def analytic_mode(fixed_lambda: float) -> AnalyticMode:
    """按精确正式/诊断表读取解析参考，拒绝静默插值。"""

    value = float(fixed_lambda)
    try:
        return ANALYTIC_MODES[value]
    except KeyError as exc:
        raise ValueError(f"unsupported fixed lambda: {value}") from exc


def _canonical_target_values(
    target_ids: Sequence[str],
    target_values: Sequence[Real],
) -> dict[str, float]:
    ids = tuple(target_ids)
    if ids != tuple(TARGET_IDS):
        raise ValueError("target_ids must use canonical HRMR target order")
    if len(target_values) != len(ids):
        raise ValueError("target_values must contain one entry per target")
    result: dict[str, float] = {}
    for target_id, raw in zip(ids, target_values, strict=True):
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError("target_values entries must be real numbers")
        value = float(raw)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("target_values entries must lie in [0, 1]")
        result[target_id] = value
    return result


def assess_operating_mode(
    fixed_lambda: float,
    target_ids: Sequence[str],
    target_occupancy: Sequence[Real],
    *,
    steady_reward: float,
    steady_cost: float,
    steady_mean_num_distinct_targets: float,
    steady_operating_mode_rate: float,
) -> OperatingModeAssessment:
    """结合 target occupancy 与 R/C/F 判定是否命中解析 mode。"""

    mode = analytic_mode(fixed_lambda)
    occupancy = _canonical_target_values(target_ids, target_occupancy)
    reward = float(steady_reward)
    cost = float(steady_cost)
    distinct = float(steady_mean_num_distinct_targets)
    mode_rate = float(steady_operating_mode_rate)
    if not all(np.isfinite(value) for value in (reward, cost, distinct, mode_rate)):
        raise ValueError("reward, cost, distinct-target count, and mode rate must be finite")
    if mode_rate < 0.0 or mode_rate > 1.0:
        raise ValueError("steady_operating_mode_rate must lie in [0, 1]")

    required_hazards = frozenset(mode.hazardous_targets)
    all_hazards = tuple(target_id for target_id in target_ids if target_id.startswith("H"))
    safe_targets = tuple(target_id for target_id in target_ids if target_id.startswith("S"))
    required_ok = all(
        occupancy[target_id] >= MODE_REQUIRED_OCCUPANCY for target_id in required_hazards
    )
    excluded_ok = all(
        occupancy[target_id] <= MODE_EXCLUDED_OCCUPANCY
        for target_id in all_hazards
        if target_id not in required_hazards
    )
    mean_hazardous = float(sum(occupancy[target_id] for target_id in all_hazards))
    mean_safe = float(sum(occupancy[target_id] for target_id in safe_targets))
    safe_count_ok = abs(mean_safe - mode.safe_target_count) <= MODE_TARGET_COUNT_TOLERANCE
    four_effective = distinct >= MODE_DISTINCT_TARGET_THRESHOLD
    mode_rate_ok = mode_rate >= MODE_RATE_THRESHOLD
    reward_ok = abs(reward - mode.reward) <= MODE_REWARD_COST_TOLERANCE
    cost_ok = abs(cost - mode.cost) <= MODE_REWARD_COST_TOLERANCE
    scalarized = reward - float(fixed_lambda) * cost
    gap_ok = abs(mode.scalarized_objective - scalarized) <= MODE_SCALARIZED_GAP_TOLERANCE
    success = all(
        (
            required_ok,
            excluded_ok,
            safe_count_ok,
            four_effective,
            mode_rate_ok,
            reward_ok,
            cost_ok,
            gap_ok,
        )
    )
    return OperatingModeAssessment(
        fixed_lambda=float(fixed_lambda),
        mean_hazardous_targets_covered=mean_hazardous,
        mean_safe_targets_covered=mean_safe,
        required_hazardous_targets_covered=required_ok,
        excluded_hazardous_targets_exited=excluded_ok,
        safe_target_count_matches=safe_count_ok,
        four_effective_targets=four_effective,
        operating_mode_rate_matches=mode_rate_ok,
        reward_matches=reward_ok,
        cost_matches=cost_ok,
        scalarized_gap_within_tolerance=gap_ok,
        success=success,
    )


def instantaneous_optimal_mode_mask(
    occupancies: ArrayLike,
    target_ids: Sequence[str],
    fixed_lambda: float,
) -> NDArray[np.bool_]:
    """返回每个 state 是否精确为该 lambda 的四角色解析 allocation。"""

    ids = tuple(target_ids)
    if ids != tuple(TARGET_IDS):
        raise ValueError("target_ids must use canonical HRMR target order")
    raw = np.asarray(occupancies)
    if raw.ndim < 1 or raw.shape[-1] != len(ids):
        raise ValueError("occupancies must end with the canonical target dimension")
    if raw.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError("occupancies must contain binary numeric values")
    if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("occupancies entries must be finite and binary")

    mode = analytic_mode(fixed_lambda)
    hazard_indices = [ids.index(target_id) for target_id in ("H1", "H2", "H3", "H4")]
    required_indices = [ids.index(target_id) for target_id in mode.hazardous_targets]
    excluded_indices = [index for index in hazard_indices if index not in required_indices]
    safe_indices = [ids.index(target_id) for target_id in ("S1", "S2", "S3", "S4")]
    result = np.ones(raw.shape[:-1], dtype=np.bool_)
    if required_indices:
        result &= np.all(raw[..., required_indices] == 1, axis=-1)
    if excluded_indices:
        result &= np.all(raw[..., excluded_indices] == 0, axis=-1)
    result &= np.sum(raw[..., safe_indices], axis=-1) == mode.safe_target_count
    result &= np.sum(raw, axis=-1) == 4
    return result


__all__ = [
    "ANALYTIC_MODES",
    "DIAGNOSTIC_FIXED_LAMBDAS",
    "FIXED_LAMBDAS",
    "LARGE_DUAL_DIAGNOSTIC_LAMBDA",
    "MODE_DISTINCT_TARGET_THRESHOLD",
    "MODE_EXCLUDED_OCCUPANCY",
    "MODE_RATE_THRESHOLD",
    "MODE_REQUIRED_OCCUPANCY",
    "MODE_REWARD_COST_TOLERANCE",
    "MODE_SCALARIZED_GAP_TOLERANCE",
    "MODE_TARGET_COUNT_TOLERANCE",
    "REWARD_ONLY_LAMBDA",
    "TRAINABLE_FIXED_LAMBDAS",
    "AnalyticMode",
    "OperatingModeAssessment",
    "analytic_mode",
    "assess_operating_mode",
    "instantaneous_optimal_mode_mask",
]
