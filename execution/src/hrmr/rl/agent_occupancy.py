"""从固定策略 trajectory 计算逐 agent 的目标区域占用诊断。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from hrmr.types import MonitoringRegions, Position, TargetId

FloatArray = NDArray[np.float64]


def _frozen_rate_vector(values: object, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} entries must be finite and lie in [0, 1]")
    frozen = array.copy()
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class AgentOccupancyVectors:
    """逐 agent 三分类向量的只读容器。"""

    non_target_occupancy_rate: FloatArray
    duplicate_target_occupancy_rate: FloatArray
    unique_target_marginal_contribution_rate: FloatArray

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(
                self,
                item.name,
                _frozen_rate_vector(getattr(self, item.name), name=item.name),
            )
        shapes = {getattr(self, item.name).shape for item in fields(self)}
        if len(shapes) != 1:
            raise ValueError("all per-agent occupancy rate vectors must have equal shape")

    @property
    def num_agents(self) -> int:
        return len(self.non_target_occupancy_rate)


@dataclass(frozen=True)
class AgentOccupancyRates(AgentOccupancyVectors):
    """一个 evaluation seed 的逐 agent 三分类占用比例。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        total = (
            self.non_target_occupancy_rate
            + self.duplicate_target_occupancy_rate
            + self.unique_target_marginal_contribution_rate
        )
        if not np.allclose(total, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("agent occupancy categories must partition every evaluated step")


@dataclass(frozen=True)
class AgentOccupancyAggregate:
    """跨 evaluation seeds 的逐 agent mean、sample std 与 95% CI。"""

    num_trajectories: int
    mean: AgentOccupancyRates
    sample_std: AgentOccupancyVectors
    ci95_half_width: AgentOccupancyVectors


def _frozen_nonnegative_vector(values: object, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} entries must be finite and non-negative")
    frozen = array.copy()
    frozen.setflags(write=False)
    return frozen


def _frozen_nonnegative_matrix(values: object, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} entries must be finite and non-negative")
    frozen = array.copy()
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class AgentTargetOccupancyStatistics:
    """逐 agent-target matrix 与逐 agent rates 的只读统计容器。"""

    agent_target_occupancy_matrix: FloatArray
    non_target_occupancy_rate: FloatArray
    duplicate_target_occupancy_rate: FloatArray
    unique_target_marginal_contribution_rate: FloatArray
    hazard_occupancy_rate: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_target_occupancy_matrix",
            _frozen_nonnegative_matrix(
                self.agent_target_occupancy_matrix,
                name="agent_target_occupancy_matrix",
            ),
        )
        for name in (
            "non_target_occupancy_rate",
            "duplicate_target_occupancy_rate",
            "unique_target_marginal_contribution_rate",
            "hazard_occupancy_rate",
        ):
            object.__setattr__(
                self,
                name,
                _frozen_nonnegative_vector(getattr(self, name), name=name),
            )
        num_agents = self.agent_target_occupancy_matrix.shape[0]
        if any(
            len(getattr(self, name)) != num_agents
            for name in (
                "non_target_occupancy_rate",
                "duplicate_target_occupancy_rate",
                "unique_target_marginal_contribution_rate",
                "hazard_occupancy_rate",
            )
        ):
            raise ValueError("all per-agent rate vectors must match the matrix row count")

    @property
    def num_agents(self) -> int:
        return self.agent_target_occupancy_matrix.shape[0]

    @property
    def num_targets(self) -> int:
        return self.agent_target_occupancy_matrix.shape[1]


@dataclass(frozen=True)
class AgentTargetOccupancyRates(AgentTargetOccupancyStatistics):
    """一个 evaluation seed 的完整逐 agent-target 占用比例。"""

    target_ids: tuple[TargetId, ...]
    hazardous_target_ids: tuple[TargetId, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        target_ids = tuple(self.target_ids)
        hazardous_target_ids = tuple(self.hazardous_target_ids)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "hazardous_target_ids", hazardous_target_ids)
        if len(target_ids) != self.num_targets or len(set(target_ids)) != len(target_ids):
            raise ValueError("target_ids must uniquely label every matrix column")
        if len(set(hazardous_target_ids)) != len(hazardous_target_ids):
            raise ValueError("hazardous_target_ids must not contain duplicates")
        missing = set(hazardous_target_ids).difference(target_ids)
        if missing:
            raise ValueError(f"unknown hazardous target ids: {sorted(missing)}")
        for name in (
            "agent_target_occupancy_matrix",
            "non_target_occupancy_rate",
            "duplicate_target_occupancy_rate",
            "unique_target_marginal_contribution_rate",
            "hazard_occupancy_rate",
        ):
            if np.any(getattr(self, name) > 1.0):
                raise ValueError(f"{name} entries must lie in [0, 1]")
        if not np.allclose(
            np.sum(self.agent_target_occupancy_matrix, axis=1) + self.non_target_occupancy_rate,
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("target columns plus non-target rate must partition each agent-step")
        if not np.allclose(
            self.non_target_occupancy_rate
            + self.duplicate_target_occupancy_rate
            + self.unique_target_marginal_contribution_rate,
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("agent occupancy categories must partition every evaluated step")
        hazardous_indices = tuple(target_ids.index(item) for item in hazardous_target_ids)
        expected_hazard = (
            np.sum(self.agent_target_occupancy_matrix[:, hazardous_indices], axis=1)
            if hazardous_indices
            else np.zeros(self.num_agents, dtype=np.float64)
        )
        if not np.allclose(
            self.hazard_occupancy_rate,
            expected_hazard,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("hazard rate must equal the sum of hazardous target columns")


@dataclass(frozen=True)
class AgentTargetOccupancyAggregate:
    """跨 evaluation seeds 聚合的完整逐 agent-target 统计。"""

    num_trajectories: int
    target_ids: tuple[TargetId, ...]
    hazardous_target_ids: tuple[TargetId, ...]
    mean: AgentTargetOccupancyRates
    sample_std: AgentTargetOccupancyStatistics
    ci95_half_width: AgentTargetOccupancyStatistics


def _target_lookup(
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId],
) -> dict[Position, int]:
    if not isinstance(regions, Mapping):
        raise TypeError("regions must be a mapping")
    ordered_ids = tuple(target_ids)
    if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("target_ids must be non-empty and unique")
    lookup: dict[Position, int] = {}
    for target_index, target_id in enumerate(ordered_ids):
        if target_id not in regions:
            raise KeyError(f"regions are missing target {target_id}")
        for raw_position in regions[target_id]:
            position = tuple(raw_position)
            if len(position) != 2:
                raise ValueError("monitoring-region positions must be two-dimensional")
            checked = int(position[0]), int(position[1])
            if checked in lookup:
                raise ValueError(f"monitoring regions overlap at {checked}")
            lookup[checked] = target_index
    return lookup


def _steady_target_memberships(
    position_history: object,
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId],
    *,
    burn_in_steps: int,
    evaluation_steps: int,
) -> NDArray[np.int64]:
    for name, value, allow_zero in (
        ("burn_in_steps", burn_in_steps, True),
        ("evaluation_steps", evaluation_steps, False),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} has an invalid value")

    positions = np.asarray(position_history)
    expected_transitions = burn_in_steps + evaluation_steps
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("position_history must have shape (steps + 1, num_agents, 2)")
    if positions.shape[0] != expected_transitions + 1:
        raise ValueError("position_history must contain reset plus every post-step position")
    if positions.shape[1] == 0:
        raise ValueError("position_history must contain at least one agent")
    if positions.dtype.kind not in {"i", "u"}:
        raise TypeError("position_history must use integer coordinates")

    target_lookup = _target_lookup(regions, target_ids)
    steady_positions = positions[burn_in_steps + 1 :]
    return np.asarray(
        [
            [target_lookup.get(tuple(position), -1) for position in joint_positions]
            for joint_positions in steady_positions
        ],
        dtype=np.int64,
    )


def _occupancy_categories(
    memberships: NDArray[np.int64],
    *,
    num_targets: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    non_target = memberships < 0
    duplicate = np.zeros_like(non_target)
    unique = np.zeros_like(non_target)
    for step_index, step_memberships in enumerate(memberships):
        target_counts = np.bincount(
            step_memberships[step_memberships >= 0],
            minlength=num_targets,
        )
        on_target = step_memberships >= 0
        duplicate[step_index, on_target] = target_counts[step_memberships[on_target]] >= 2
        unique[step_index, on_target] = target_counts[step_memberships[on_target]] == 1
    return non_target, duplicate, unique


def summarize_agent_occupancy(
    position_history: object,
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId],
    *,
    burn_in_steps: int,
    evaluation_steps: int,
) -> AgentOccupancyRates:
    """对 burn-in 后的 post-transition positions 计算三项逐 agent 比例。

    non-target 表示 agent 不在任何监控区域；duplicate 表示其所在目标同时被
    其他 agent 覆盖；unique marginal contribution 表示移除该 agent 会令当步
    distinct target count 减少 1。监控区域互不重叠，因此三类严格构成分割。
    """

    memberships = _steady_target_memberships(
        position_history,
        regions,
        target_ids,
        burn_in_steps=burn_in_steps,
        evaluation_steps=evaluation_steps,
    )
    non_target, duplicate, unique = _occupancy_categories(
        memberships,
        num_targets=len(target_ids),
    )

    return AgentOccupancyRates(
        non_target_occupancy_rate=np.mean(non_target, axis=0),
        duplicate_target_occupancy_rate=np.mean(duplicate, axis=0),
        unique_target_marginal_contribution_rate=np.mean(unique, axis=0),
    )


def summarize_agent_target_occupancy(
    position_history: object,
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId],
    hazardous_target_ids: Sequence[TargetId],
    *,
    burn_in_steps: int,
    evaluation_steps: int,
) -> AgentTargetOccupancyRates:
    """计算 O[i,m] 以及逐 agent unique/duplicate/non-target/hazard rates。"""

    ordered_target_ids = tuple(target_ids)
    ordered_hazardous_ids = tuple(hazardous_target_ids)
    unknown_hazardous = set(ordered_hazardous_ids).difference(ordered_target_ids)
    if unknown_hazardous:
        raise ValueError(f"unknown hazardous target ids: {sorted(unknown_hazardous)}")
    memberships = _steady_target_memberships(
        position_history,
        regions,
        ordered_target_ids,
        burn_in_steps=burn_in_steps,
        evaluation_steps=evaluation_steps,
    )
    non_target, duplicate, unique = _occupancy_categories(
        memberships,
        num_targets=len(ordered_target_ids),
    )
    matrix = np.mean(
        memberships[:, :, None]
        == np.arange(len(ordered_target_ids), dtype=np.int64)[None, None, :],
        axis=0,
    )
    hazardous_indices = np.asarray(
        [ordered_target_ids.index(item) for item in ordered_hazardous_ids],
        dtype=np.int64,
    )
    hazard = (
        np.mean(np.isin(memberships, hazardous_indices), axis=0)
        if len(hazardous_indices)
        else np.zeros(memberships.shape[1], dtype=np.float64)
    )
    return AgentTargetOccupancyRates(
        agent_target_occupancy_matrix=matrix,
        non_target_occupancy_rate=np.mean(non_target, axis=0),
        duplicate_target_occupancy_rate=np.mean(duplicate, axis=0),
        unique_target_marginal_contribution_rate=np.mean(unique, axis=0),
        hazard_occupancy_rate=hazard,
        target_ids=ordered_target_ids,
        hazardous_target_ids=ordered_hazardous_ids,
    )


def aggregate_agent_occupancy(
    trajectories: Sequence[AgentOccupancyRates],
) -> AgentOccupancyAggregate:
    """以 trajectory seed 为独立样本聚合逐 agent rates。"""

    items = tuple(trajectories)
    if not items:
        raise ValueError("at least one trajectory is required")
    num_agents = items[0].num_agents
    if any(item.num_agents != num_agents for item in items):
        raise ValueError("all trajectories must have the same number of agents")

    matrices = {
        item.name: np.stack([getattr(trajectory, item.name) for trajectory in items])
        for item in fields(AgentOccupancyRates)
    }
    means = {name: np.mean(values, axis=0) for name, values in matrices.items()}
    if len(items) == 1:
        sample_stds = {name: np.zeros(num_agents) for name in matrices}
    else:
        sample_stds = {name: np.std(values, axis=0, ddof=1) for name, values in matrices.items()}
    scale = 1.96 / sqrt(len(items)) if len(items) > 1 else 0.0
    return AgentOccupancyAggregate(
        num_trajectories=len(items),
        mean=AgentOccupancyRates(**means),
        sample_std=AgentOccupancyVectors(**sample_stds),
        ci95_half_width=AgentOccupancyVectors(
            **{name: values * scale for name, values in sample_stds.items()}
        ),
    )


def aggregate_agent_target_occupancy(
    trajectories: Sequence[AgentTargetOccupancyRates],
) -> AgentTargetOccupancyAggregate:
    """以 trajectory seed 为独立样本聚合完整逐 agent-target 统计。"""

    items = tuple(trajectories)
    if not items:
        raise ValueError("at least one trajectory is required")
    first = items[0]
    if any(
        item.num_agents != first.num_agents
        or item.target_ids != first.target_ids
        or item.hazardous_target_ids != first.hazardous_target_ids
        for item in items
    ):
        raise ValueError("all trajectories must use matching agents and target definitions")
    names = (
        "agent_target_occupancy_matrix",
        "non_target_occupancy_rate",
        "duplicate_target_occupancy_rate",
        "unique_target_marginal_contribution_rate",
        "hazard_occupancy_rate",
    )
    arrays = {name: np.stack([getattr(trajectory, name) for trajectory in items]) for name in names}
    means = {name: np.mean(values, axis=0) for name, values in arrays.items()}
    sample_stds = {
        name: (np.std(values, axis=0, ddof=1) if len(items) > 1 else np.zeros_like(values[0]))
        for name, values in arrays.items()
    }
    scale = 1.96 / sqrt(len(items)) if len(items) > 1 else 0.0
    return AgentTargetOccupancyAggregate(
        num_trajectories=len(items),
        target_ids=first.target_ids,
        hazardous_target_ids=first.hazardous_target_ids,
        mean=AgentTargetOccupancyRates(
            **means,
            target_ids=first.target_ids,
            hazardous_target_ids=first.hazardous_target_ids,
        ),
        sample_std=AgentTargetOccupancyStatistics(**sample_stds),
        ci95_half_width=AgentTargetOccupancyStatistics(
            **{name: values * scale for name, values in sample_stds.items()}
        ),
    )


__all__ = [
    "AgentOccupancyAggregate",
    "AgentOccupancyRates",
    "AgentOccupancyVectors",
    "AgentTargetOccupancyAggregate",
    "AgentTargetOccupancyRates",
    "AgentTargetOccupancyStatistics",
    "aggregate_agent_occupancy",
    "aggregate_agent_target_occupancy",
    "summarize_agent_occupancy",
    "summarize_agent_target_occupancy",
]
