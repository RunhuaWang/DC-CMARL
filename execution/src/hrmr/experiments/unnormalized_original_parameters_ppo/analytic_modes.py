"""原始异质 target 参数、未归一化信号消融的五级解析模式。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.constants import HAZARDOUS_TARGET_IDS, SAFE_TARGET_IDS, TARGET_IDS

ALLOWED_FIXED_LAMBDAS = (0.0, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0)
LAMBDA9_DIAGNOSTIC_FIXED_LAMBDA = 9.0
LAMBDA10_DIAGNOSTIC_FIXED_LAMBDA = 10.0
SUPPORTED_FIXED_LAMBDAS = (
    *ALLOWED_FIXED_LAMBDAS,
    LAMBDA9_DIAGNOSTIC_FIXED_LAMBDA,
    LAMBDA10_DIAGNOSTIC_FIXED_LAMBDA,
)
TRAINING_SEED = 0

MODE_REQUIRED_OCCUPANCY = 0.80
MODE_EXCLUDED_OCCUPANCY = 0.20
MODE_TARGET_COUNT_TOLERANCE = 0.20
MODE_DISTINCT_TARGET_THRESHOLD = 3.80
MODE_RATE_THRESHOLD = 0.75

# 正式 normalized 判定对 R、C、F 使用 0.08 的绝对容差。这里
# R_raw=8*R_formal、C_raw=4*C_formal、F_raw=8*F_formal，因此按量纲
# 换算，而不是把 0.08 直接施加到 raw 数值。
MODE_REWARD_TOLERANCE = 0.64
MODE_COST_TOLERANCE = 0.32
MODE_SCALARIZED_GAP_TOLERANCE = 0.64

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class UnnormalizedOriginalParametersAnalyticMode:
    """一个 fixed-lambda 的最优异质 target 组成与 raw 信号。"""

    fixed_lambda: float
    reward: float
    cost: float
    scalarized_objective: float
    hazardous_targets: tuple[str, ...]
    safe_target_count: int

    @property
    def required_hazardous_targets(self) -> tuple[str, ...]:
        """返回必须覆盖的危险目标；安全目标只按数量而非身份区分。"""

        return self.hazardous_targets

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的解析参考。"""

        return asdict(self)


ANALYTIC_MODES = {
    0.0: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=0.0,
        reward=8.0,
        cost=2.0,
        scalarized_objective=8.0,
        hazardous_targets=("H1", "H2", "H3", "H4"),
        safe_target_count=0,
    ),
    1.5: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=1.5,
        reward=7.2,
        cost=1.2,
        scalarized_objective=5.4,
        hazardous_targets=("H2", "H3", "H4"),
        safe_target_count=1,
    ),
    2.5: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=2.5,
        reward=6.0,
        cost=0.6,
        scalarized_objective=4.5,
        hazardous_targets=("H3", "H4"),
        safe_target_count=2,
    ),
    3.5: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=3.5,
        reward=4.8,
        cost=0.2,
        scalarized_objective=4.1,
        hazardous_targets=("H4",),
        safe_target_count=3,
    ),
    4.5: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=4.5,
        reward=4.0,
        cost=0.0,
        scalarized_objective=4.0,
        hazardous_targets=(),
        safe_target_count=4,
    ),
    5.0: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=5.0,
        reward=4.0,
        cost=0.0,
        scalarized_objective=4.0,
        hazardous_targets=(),
        safe_target_count=4,
    ),
    5.5: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=5.5,
        reward=4.0,
        cost=0.0,
        scalarized_objective=4.0,
        hazardous_targets=(),
        safe_target_count=4,
    ),
    6.0: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=6.0,
        reward=4.0,
        cost=0.0,
        scalarized_objective=4.0,
        hazardous_targets=(),
        safe_target_count=4,
    ),
    7.0: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=7.0,
        reward=4.0,
        cost=0.0,
        scalarized_objective=4.0,
        hazardous_targets=(),
        safe_target_count=4,
    ),
    8.0: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=8.0,
        reward=4.0,
        cost=0.0,
        scalarized_objective=4.0,
        hazardous_targets=(),
        safe_target_count=4,
    ),
    9.0: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=9.0,
        reward=4.0,
        cost=0.0,
        scalarized_objective=4.0,
        hazardous_targets=(),
        safe_target_count=4,
    ),
    10.0: UnnormalizedOriginalParametersAnalyticMode(
        fixed_lambda=10.0,
        reward=4.0,
        cost=0.0,
        scalarized_objective=4.0,
        hazardous_targets=(),
        safe_target_count=4,
    ),
}


def analytic_mode(fixed_lambda: Real) -> UnnormalizedOriginalParametersAnalyticMode:
    """按精确实验网格读取解析模式，不对切换边界静默插值。"""

    if isinstance(fixed_lambda, (bool, np.bool_)) or not isinstance(fixed_lambda, Real):
        raise TypeError("fixed_lambda must be a real number")
    value = float(fixed_lambda)
    if not isfinite(value):
        raise ValueError("fixed_lambda must be finite")
    try:
        return ANALYTIC_MODES[value]
    except KeyError as exc:
        raise ValueError(f"fixed_lambda must be one of {SUPPORTED_FIXED_LAMBDAS}") from exc


def optimal_mode_mask(
    target_occupancies: ArrayLike,
    target_ids: tuple[str, ...] = TARGET_IDS,
    *,
    fixed_lambda: Real,
) -> BoolArray:
    """判断样本是否精确覆盖该 lambda 的四角色最优 target 组成。

    安全目标物理身份等价。例如 lambda=2.5 时，任意两个不同安全目标与
    H3、H4 的组合都算命中；总 coverage 必须恰为四，以排除 off/duplicate。
    """

    ids = tuple(target_ids)
    if ids != tuple(TARGET_IDS):
        raise ValueError("target_ids must use canonical HRMR target order")
    raw = np.asarray(target_occupancies)
    if raw.ndim == 0 or raw.shape[-1] != len(ids):
        raise ValueError("target_occupancies must end with the eight-target dimension")
    if raw.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError("target_occupancies must contain numeric values")
    if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("target_occupancies must be finite and binary")

    mode = analytic_mode(fixed_lambda)
    safe_indices = tuple(ids.index(target_id) for target_id in SAFE_TARGET_IDS)
    required_hazard_indices = tuple(ids.index(target_id) for target_id in mode.hazardous_targets)
    excluded_hazard_indices = tuple(
        ids.index(target_id)
        for target_id in HAZARDOUS_TARGET_IDS
        if target_id not in mode.hazardous_targets
    )

    result = np.sum(raw, axis=-1) == 4
    result &= np.sum(raw[..., safe_indices], axis=-1) == mode.safe_target_count
    if required_hazard_indices:
        result &= np.all(raw[..., required_hazard_indices] == 1, axis=-1)
    if excluded_hazard_indices:
        result &= np.all(raw[..., excluded_hazard_indices] == 0, axis=-1)
    return np.asarray(result, dtype=np.bool_)


def assess_mode(
    fixed_lambda: Real,
    target_occupancy: ArrayLike,
    *,
    mean_num_distinct_targets: float,
    optimal_mode_rate: float,
    steady_reward: float,
    steady_cost: float,
) -> dict[str, Any]:
    """结合长期 occupancy 与 raw R/C/F 判定是否达到五级解析模式。"""

    mode = analytic_mode(fixed_lambda)
    occupancy = np.asarray(target_occupancy, dtype=np.float64)
    if occupancy.shape != (len(TARGET_IDS),) or not np.all(np.isfinite(occupancy)):
        raise ValueError("target_occupancy must be a finite canonical eight-vector")
    if np.any((occupancy < 0.0) | (occupancy > 1.0)):
        raise ValueError("target_occupancy entries must lie in [0, 1]")

    scalar_metrics = (
        mean_num_distinct_targets,
        optimal_mode_rate,
        steady_reward,
        steady_cost,
    )
    if not all(isfinite(float(value)) for value in scalar_metrics):
        raise ValueError("assessment metrics must be finite")
    if not 0.0 <= float(optimal_mode_rate) <= 1.0:
        raise ValueError("optimal_mode_rate must lie in [0, 1]")

    mapping = dict(zip(TARGET_IDS, occupancy, strict=True))
    required_hazards = frozenset(mode.hazardous_targets)
    excluded_hazards = tuple(
        target_id for target_id in HAZARDOUS_TARGET_IDS if target_id not in required_hazards
    )
    required_ok = all(
        mapping[target_id] >= MODE_REQUIRED_OCCUPANCY for target_id in required_hazards
    )
    excluded_ok = all(
        mapping[target_id] <= MODE_EXCLUDED_OCCUPANCY for target_id in excluded_hazards
    )
    mean_safe = float(sum(mapping[target_id] for target_id in SAFE_TARGET_IDS))
    safe_count_ok = abs(mean_safe - mode.safe_target_count) <= MODE_TARGET_COUNT_TOLERANCE
    distinct_ok = float(mean_num_distinct_targets) >= MODE_DISTINCT_TARGET_THRESHOLD
    rate_ok = float(optimal_mode_rate) >= MODE_RATE_THRESHOLD
    learned_objective = float(steady_reward) - mode.fixed_lambda * float(steady_cost)
    reward_ok = abs(float(steady_reward) - mode.reward) <= MODE_REWARD_TOLERANCE
    cost_ok = abs(float(steady_cost) - mode.cost) <= MODE_COST_TOLERANCE
    gap = mode.scalarized_objective - learned_objective
    gap_ok = abs(gap) <= MODE_SCALARIZED_GAP_TOLERANCE
    success = all(
        (
            required_ok,
            excluded_ok,
            safe_count_ok,
            distinct_ok,
            rate_ok,
            reward_ok,
            cost_ok,
            gap_ok,
        )
    )

    return {
        "required_hazardous_targets": list(mode.hazardous_targets),
        "excluded_hazardous_targets": list(excluded_hazards),
        "safe_target_count": mode.safe_target_count,
        "required_hazardous_target_occupancies": {
            target_id: float(mapping[target_id]) for target_id in mode.hazardous_targets
        },
        "excluded_hazardous_target_occupancies": {
            target_id: float(mapping[target_id]) for target_id in excluded_hazards
        },
        "mean_safe_targets_covered": mean_safe,
        "mean_num_distinct_targets": float(mean_num_distinct_targets),
        "exact_optimal_allocation_rate": float(optimal_mode_rate),
        "learned_scalarized_objective": learned_objective,
        "scalarized_optimality_gap": gap,
        "required_hazardous_targets_covered": required_ok,
        "excluded_hazardous_targets_exited": excluded_ok,
        "safe_target_count_matches": safe_count_ok,
        "four_effective_targets": distinct_ok,
        "optimal_mode_rate_matches": rate_ok,
        "reward_matches": reward_ok,
        "cost_matches": cost_ok,
        "scalarized_gap_within_tolerance": gap_ok,
        "success": success,
    }


__all__ = [
    "ALLOWED_FIXED_LAMBDAS",
    "ANALYTIC_MODES",
    "LAMBDA9_DIAGNOSTIC_FIXED_LAMBDA",
    "LAMBDA10_DIAGNOSTIC_FIXED_LAMBDA",
    "MODE_COST_TOLERANCE",
    "MODE_DISTINCT_TARGET_THRESHOLD",
    "MODE_EXCLUDED_OCCUPANCY",
    "MODE_RATE_THRESHOLD",
    "MODE_REQUIRED_OCCUPANCY",
    "MODE_REWARD_TOLERANCE",
    "MODE_SCALARIZED_GAP_TOLERANCE",
    "MODE_TARGET_COUNT_TOLERANCE",
    "SUPPORTED_FIXED_LAMBDAS",
    "TRAINING_SEED",
    "UnnormalizedOriginalParametersAnalyticMode",
    "analytic_mode",
    "assess_mode",
    "optimal_mode_mask",
]
