"""HRMR rollout 的长期平均 reward、cost 与 occupancy 统计。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RolloutSummary:
    """一个连续 rollout 切片的加法信号汇总。"""

    start_step: int
    end_step: int
    num_steps: int
    mean_reward: float
    mean_global_cost: float
    mean_target_occupancy: FloatArray
    mean_local_costs: FloatArray | None = None

    def __post_init__(self) -> None:
        occupancy = np.asarray(self.mean_target_occupancy, dtype=np.float64).copy()
        occupancy.setflags(write=False)
        object.__setattr__(self, "mean_target_occupancy", occupancy)
        if self.mean_local_costs is not None:
            local_costs = np.asarray(self.mean_local_costs, dtype=np.float64).copy()
            local_costs.setflags(write=False)
            object.__setattr__(self, "mean_local_costs", local_costs)

    @property
    def average_reward(self) -> float:
        """``mean_reward`` 的语义化别名。"""

        return self.mean_reward

    @property
    def average_global_cost(self) -> float:
        """``mean_global_cost`` 的语义化别名。"""

        return self.mean_global_cost

    @property
    def target_occupancy(self) -> FloatArray:
        """``mean_target_occupancy`` 的简短别名。"""

        return self.mean_target_occupancy


def _as_finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (num_steps,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_finite_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (num_steps, num_items)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _slice_bounds(
    num_steps: int,
    start_step: int,
    end_step: int | None,
) -> tuple[int, int]:
    if isinstance(start_step, bool) or not isinstance(start_step, Integral):
        raise TypeError("start_step must be an integer")
    start = int(start_step)
    if end_step is None:
        end = num_steps
    else:
        if isinstance(end_step, bool) or not isinstance(end_step, Integral):
            raise TypeError("end_step must be an integer or None")
        end = int(end_step)
    if not 0 <= start < end <= num_steps:
        raise ValueError(
            f"rollout slice must satisfy 0 <= start < end <= {num_steps}; got ({start}, {end})"
        )
    return start, end


def compute_occupancy_averages(
    target_occupancies: ArrayLike,
    start_step: int = 0,
    end_step: int | None = None,
) -> FloatArray:
    """按 PDF 式 (12.8) 对每个目标的 occupancy 做时间平均。"""

    occupancies = _as_finite_matrix(target_occupancies, name="target_occupancies")
    start, end = _slice_bounds(len(occupancies), start_step, end_step)
    return np.mean(occupancies[start:end], axis=0, dtype=np.float64)


def summarize_rollout(
    rewards: ArrayLike,
    global_costs: ArrayLike,
    target_occupancies: ArrayLike,
    local_costs: ArrayLike | None = None,
    *,
    start_step: int = 0,
    end_step: int | None = None,
) -> RolloutSummary:
    """汇总 rollout 全轨迹或指定的 transient/steady 切片。"""

    reward_array = _as_finite_vector(rewards, name="rewards")
    global_cost_array = _as_finite_vector(global_costs, name="global_costs")
    occupancy_array = _as_finite_matrix(
        target_occupancies,
        name="target_occupancies",
    )
    num_steps = len(reward_array)
    if len(global_cost_array) != num_steps or len(occupancy_array) != num_steps:
        raise ValueError("reward, global-cost and occupancy histories must have equal length")

    local_cost_array: FloatArray | None = None
    if local_costs is not None:
        local_cost_array = _as_finite_matrix(local_costs, name="local_costs")
        if len(local_cost_array) != num_steps:
            raise ValueError("local_costs must have the same number of steps as rewards")

    start, end = _slice_bounds(num_steps, start_step, end_step)
    mean_local_costs = (
        None
        if local_cost_array is None
        else np.mean(local_cost_array[start:end], axis=0, dtype=np.float64)
    )
    return RolloutSummary(
        start_step=start,
        end_step=end,
        num_steps=end - start,
        mean_reward=float(np.mean(reward_array[start:end], dtype=np.float64)),
        mean_global_cost=float(np.mean(global_cost_array[start:end], dtype=np.float64)),
        mean_target_occupancy=np.mean(
            occupancy_array[start:end],
            axis=0,
            dtype=np.float64,
        ),
        mean_local_costs=mean_local_costs,
    )


def split_rollout_summaries(
    rewards: ArrayLike,
    global_costs: ArrayLike,
    target_occupancies: ArrayLike,
    local_costs: ArrayLike | None,
    *,
    steady_start_step: int,
) -> tuple[RolloutSummary | None, RolloutSummary]:
    """按稳态起点分开汇总移动暂态和稳态；零长度暂态返回 ``None``。"""

    if isinstance(steady_start_step, bool) or not isinstance(steady_start_step, Integral):
        raise TypeError("steady_start_step must be an integer")
    steady_start = int(steady_start_step)
    total_steps = len(np.asarray(rewards))
    if not 0 <= steady_start < total_steps:
        raise ValueError("steady_start_step must identify a recorded rollout step")
    transient = (
        None
        if steady_start == 0
        else summarize_rollout(
            rewards,
            global_costs,
            target_occupancies,
            local_costs,
            end_step=steady_start,
        )
    )
    steady = summarize_rollout(
        rewards,
        global_costs,
        target_occupancies,
        local_costs,
        start_step=steady_start,
    )
    return transient, steady


# 便于评价脚本使用的短别名。
occupancy_averages = compute_occupancy_averages
rollout_summary = summarize_rollout


__all__ = [
    "RolloutSummary",
    "compute_occupancy_averages",
    "occupancy_averages",
    "rollout_summary",
    "split_rollout_summaries",
    "summarize_rollout",
]
