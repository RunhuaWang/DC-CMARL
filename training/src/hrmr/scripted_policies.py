"""用于 Phase 1 环境验收的确定性 shortest-path scripted allocator。"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import permutations
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.config import HRMRConfig
from hrmr.constants import ACTION_DELTAS, Action
from hrmr.geometry import agent_region_ids, compute_target_occupancy
from hrmr.metrics import RolloutSummary, split_rollout_summaries, summarize_rollout
from hrmr.types import MonitoringRegions, Position

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


# PDF §11.2 的五个规范代表组合；这里仅固定组合结构，R/C 始终由 config 计算。
CANONICAL_ASSIGNMENTS: tuple[tuple[str, ...], ...] = (
    ("H1", "H2", "H3", "H4"),
    ("S1", "H2", "H3", "H4"),
    ("S1", "S2", "H3", "H4"),
    ("S1", "S2", "S3", "H4"),
    ("S1", "S2", "S3", "S4"),
)
CANONICAL_MODE_TARGETS = CANONICAL_ASSIGNMENTS
MIDPOINT_DUALS: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.25)


@dataclass(frozen=True)
class OperatingMode:
    """一个 PDF 规范 operating mode 及其由配置推导的解析信号。"""

    index: int
    target_ids: tuple[str, ...]
    midpoint_lambda: float
    expected_team_reward: float
    expected_global_cost: float
    expected_target_occupancy: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ids", tuple(self.target_ids))
        occupancy = np.asarray(self.expected_target_occupancy, dtype=np.float64).copy()
        occupancy.setflags(write=False)
        object.__setattr__(self, "expected_target_occupancy", occupancy)

    @property
    def mode_id(self) -> int:
        return self.index

    @property
    def assignment(self) -> tuple[str, ...]:
        return self.target_ids

    @property
    def expected_reward(self) -> float:
        return self.expected_team_reward

    @property
    def expected_cost(self) -> float:
        return self.expected_global_cost

    @property
    def target_occupancy(self) -> FloatArray:
        return self.expected_target_occupancy


@dataclass(frozen=True)
class ScriptedModeResult:
    """脚本控制器 rollout 的完整暂态/稳态验收结果。"""

    mode: OperatingMode
    agent_target_ids: tuple[str, ...]
    seed: int | None
    total_steps: int
    transient_steps: int
    steady_steps: int
    arrival_step: int
    full_summary: RolloutSummary
    transient_summary: RolloutSummary | None
    steady_summary: RolloutSummary
    position_history: IntArray
    requested_action_history: IntArray
    reward_history: FloatArray
    global_cost_history: FloatArray
    local_cost_history: FloatArray
    target_occupancy_history: FloatArray
    collision_history: BoolArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_target_ids", tuple(self.agent_target_ids))
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                copied = value.copy()
                copied.setflags(write=False)
                object.__setattr__(self, item.name, copied)

    @property
    def assignment(self) -> tuple[str, ...]:
        return self.agent_target_ids

    @property
    def reached(self) -> bool:
        return True

    @property
    def steady_reward(self) -> float:
        return self.steady_summary.mean_reward

    @property
    def steady_global_cost(self) -> float:
        return self.steady_summary.mean_global_cost

    @property
    def steady_target_occupancy(self) -> FloatArray:
        return self.steady_summary.mean_target_occupancy

    @property
    def final_positions(self) -> IntArray:
        return self.position_history[-1]


def _as_position(position: Sequence[int], *, name: str) -> Position:
    if isinstance(position, (str, bytes)):
        raise TypeError(f"{name} must be a two-element integer coordinate")
    try:
        values = tuple(position)
    except TypeError as exc:
        raise TypeError(f"{name} must be a two-element integer coordinate") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain two components")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise TypeError(f"{name} components must be integers")
    return int(values[0]), int(values[1])


def _as_positions(positions: ArrayLike, *, name: str = "positions") -> IntArray:
    array = np.asarray(positions, dtype=object)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape (num_agents, 2)")
    checked = [
        _as_position(position, name=f"{name}[{index}]") for index, position in enumerate(array)
    ]
    return np.asarray(checked, dtype=np.int64).reshape(array.shape)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def canonical_operating_modes(config: HRMRConfig) -> tuple[OperatingMode, ...]:
    """依据当前配置构造 PDF 的五个规范 operating modes。"""

    if not isinstance(config, HRMRConfig):
        raise TypeError("config must be an HRMRConfig")
    if config.num_agents != 4:
        raise ValueError("the five canonical modes require exactly four agents")

    modes = []
    for mode_index, (target_ids, midpoint_lambda) in enumerate(
        zip(CANONICAL_ASSIGNMENTS, MIDPOINT_DUALS, strict=True),
        start=1,
    ):
        missing = [target_id for target_id in target_ids if target_id not in config.target_ids]
        if missing:
            raise KeyError(f"config is missing canonical target ids: {missing}")
        occupancy = np.asarray(
            [float(target_id in target_ids) for target_id in config.target_ids],
            dtype=np.float64,
        )
        reward = sum(float(config.target_values[target_id]) for target_id in target_ids)
        reward /= float(config.reward_normalizer)
        global_cost = sum(
            float(config.hazard_intensities[target_id]) for target_id in target_ids
        ) / float(config.num_agents)
        modes.append(
            OperatingMode(
                index=mode_index,
                target_ids=target_ids,
                midpoint_lambda=midpoint_lambda,
                expected_team_reward=reward,
                expected_global_cost=global_cost,
                expected_target_occupancy=occupancy,
            )
        )
    return tuple(modes)


build_operating_modes = canonical_operating_modes


def validate_direct_operating_modes(config: HRMRConfig) -> tuple[OperatingMode, ...]:
    """在目标中心直接构造五个状态并交叉验证环境 reward/cost 纯函数。"""

    # 延迟导入可避免 scripted allocator 与环境模块之间形成导入环。
    from hrmr.rewards_costs import (
        aggregate_global_cost,
        compute_local_costs,
        compute_team_reward,
    )

    modes = canonical_operating_modes(config)
    regions = config.monitoring_regions
    for mode in modes:
        positions = np.asarray(
            [config.target_centers[target_id] for target_id in mode.target_ids],
            dtype=np.int64,
        )
        occupancy = compute_target_occupancy(positions, regions, config.target_ids)
        reward = compute_team_reward(
            positions,
            regions,
            config.target_values,
            config.reward_normalizer,
            config.target_ids,
        )
        local_costs = compute_local_costs(
            positions,
            regions,
            config.hazard_intensities,
            config.num_agents,
            config.target_ids,
        )
        global_cost = aggregate_global_cost(local_costs)

        if not np.array_equal(occupancy, mode.expected_target_occupancy):
            raise AssertionError(
                f"mode {mode.index} occupancy mismatch: "
                f"{occupancy.tolist()} != {mode.expected_target_occupancy.tolist()}"
            )
        if not np.isclose(reward, mode.expected_team_reward, rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"mode {mode.index} reward mismatch: {reward} != {mode.expected_team_reward}"
            )
        if not np.isclose(global_cost, mode.expected_global_cost, rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"mode {mode.index} cost mismatch: {global_cost} != {mode.expected_global_cost}"
            )
    return modes


def _distance_to_region(position: Position, region: Iterable[Position]) -> int:
    cells = tuple(region)
    if not cells:
        raise ValueError("target regions must not be empty")
    return min(abs(position[0] - cell[0]) + abs(position[1] - cell[1]) for cell in cells)


def assign_targets_min_manhattan(
    current_positions: ArrayLike,
    target_ids: Sequence[str],
    regions: MonitoringRegions | HRMRConfig,
) -> tuple[str, ...]:
    """求总 target-to-agent Manhattan 距离最小的唯一确定性指派。

    返回元组按 agent ID 排列；总距离相同时用目标 ID 元组的字典序打破
    平局，因此结果不依赖 mapping 的迭代顺序。
    """

    positions = _as_positions(current_positions, name="current_positions")
    ids = tuple(target_ids)
    if len(ids) != len(positions):
        raise ValueError("the number of targets must equal the number of agents")
    if len(set(ids)) != len(ids):
        raise ValueError("assigned target ids must be unique")
    target_regions = regions.monitoring_regions if isinstance(regions, HRMRConfig) else regions
    missing = [target_id for target_id in ids if target_id not in target_regions]
    if missing:
        raise KeyError(f"regions are missing target ids: {missing}")

    best_assignment: tuple[str, ...] | None = None
    best_key: tuple[int, tuple[str, ...]] | None = None
    # 从排序 ID 开始枚举，使 tie-break 与调用方传入顺序无关。
    for assignment in permutations(sorted(ids)):
        total_distance = sum(
            _distance_to_region(
                (int(position[0]), int(position[1])),
                target_regions[target_id],
            )
            for position, target_id in zip(positions, assignment, strict=True)
        )
        key = total_distance, assignment
        if best_key is None or key < best_key:
            best_key = key
            best_assignment = assignment
    if best_assignment is None:  # pragma: no cover - 非空 agent 集合下不可能
        raise RuntimeError("no target assignment could be constructed")
    return best_assignment


minimum_manhattan_assignment = assign_targets_min_manhattan


def shortest_path_bfs(
    start: Sequence[int],
    goal_cells: Iterable[Sequence[int]],
    grid_width: int,
    grid_height: int,
    blocked_cells: Iterable[Sequence[int]] = (),
) -> tuple[Position, ...] | None:
    """在当前占用约束下返回到任一目标格的确定性最短路径。"""

    width = _positive_integer(grid_width, name="grid_width")
    height = _positive_integer(grid_height, name="grid_height")
    source = _as_position(start, name="start")
    goals = frozenset(
        _as_position(cell, name=f"goal_cells[{index}]") for index, cell in enumerate(goal_cells)
    )
    if not goals:
        raise ValueError("goal_cells must not be empty")

    def in_grid(position: Position) -> bool:
        return 0 <= position[0] < height and 0 <= position[1] < width

    if not in_grid(source) or any(not in_grid(goal) for goal in goals):
        raise ValueError("start and goal cells must lie inside the grid")
    blocked = {
        _as_position(cell, name=f"blocked_cells[{index}]")
        for index, cell in enumerate(blocked_cells)
    }
    blocked.discard(source)
    if source in goals:
        return (source,)

    action_order = (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)
    frontier = deque([source])
    parents: dict[Position, Position | None] = {source: None}
    destination: Position | None = None
    while frontier:
        current = frontier.popleft()
        for action in action_order:
            row_delta, column_delta = ACTION_DELTAS[action]
            neighbor = current[0] + row_delta, current[1] + column_delta
            if not in_grid(neighbor) or neighbor in blocked or neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor in goals:
                destination = neighbor
                frontier.clear()
                break
            frontier.append(neighbor)
        if destination is not None:
            break

    if destination is None:
        return None
    reversed_path = [destination]
    while parents[reversed_path[-1]] is not None:
        parent = parents[reversed_path[-1]]
        assert parent is not None
        reversed_path.append(parent)
    return tuple(reversed(reversed_path))


class ShortestPathScriptedController:
    """固定目标指派、每步基于最终位置重规划的 BFS 控制器。"""

    def __init__(
        self,
        config: HRMRConfig,
        target_ids: Sequence[str],
        initial_positions: ArrayLike | None = None,
    ) -> None:
        if not isinstance(config, HRMRConfig):
            raise TypeError("config must be an HRMRConfig")
        positions = (
            np.asarray(config.initial_positions, dtype=np.int64)
            if initial_positions is None
            else _as_positions(initial_positions, name="initial_positions")
        )
        self.config = config
        self.target_ids = tuple(target_ids)
        self.agent_target_ids = assign_targets_min_manhattan(
            positions,
            self.target_ids,
            config.monitoring_regions,
        )
        self._regions = config.monitoring_regions

    @property
    def assignment(self) -> tuple[str, ...]:
        return self.agent_target_ids

    def actions(self, current_positions: ArrayLike) -> IntArray:
        """用当前最终位置和 next-cell reservation 生成联合动作。"""

        positions = _as_positions(current_positions, name="current_positions")
        if len(positions) != len(self.agent_target_ids):
            raise ValueError("current position count does not match the fixed assignment")
        if len(np.unique(positions, axis=0)) != len(positions):
            raise ValueError("current_positions must not overlap")

        actions = np.full(len(positions), int(Action.STAY), dtype=np.int64)
        occupied = {(int(row), int(column)) for row, column in positions}
        reserved: set[Position] = set()

        # 已到达机器人优先预留原位；其后移动机器人不能穿入稳定占用者。
        arrived = [
            (int(position[0]), int(position[1])) in self._regions[target_id]
            for position, target_id in zip(
                positions,
                self.agent_target_ids,
                strict=True,
            )
        ]
        for index, is_arrived in enumerate(arrived):
            if is_arrived:
                reserved.add((int(positions[index, 0]), int(positions[index, 1])))

        delta_to_action = {delta: action for action, delta in ACTION_DELTAS.items()}
        for index, (position, target_id) in enumerate(
            zip(positions, self.agent_target_ids, strict=True)
        ):
            if arrived[index]:
                continue
            start = int(position[0]), int(position[1])
            blocked = (occupied - {start}) | reserved
            path = shortest_path_bfs(
                start,
                self._regions[target_id],
                grid_width=self.config.grid_width,
                grid_height=self.config.grid_height,
                blocked_cells=blocked,
            )
            if path is None or len(path) < 2:
                reserved.add(start)
                continue
            next_cell = path[1]
            delta = next_cell[0] - start[0], next_cell[1] - start[1]
            actions[index] = int(delta_to_action[delta])
            reserved.add(next_cell)
        return actions

    def act(self, current_positions: ArrayLike) -> IntArray:
        return self.actions(current_positions)

    def __call__(self, current_positions: ArrayLike) -> IntArray:
        return self.actions(current_positions)


# 常用短名；两者是同一个实现。
ScriptedController = ShortestPathScriptedController


def _coerce_mode(
    mode: OperatingMode | int | Sequence[str],
    config: HRMRConfig,
) -> OperatingMode:
    modes = canonical_operating_modes(config)
    if isinstance(mode, OperatingMode):
        return mode
    if isinstance(mode, bool):
        raise TypeError("mode must be an OperatingMode, integer, or target-id sequence")
    if isinstance(mode, Integral):
        index = int(mode)
        if index == 0:
            return modes[0]
        if 1 <= index <= len(modes):
            return modes[index - 1]
        raise ValueError(f"mode index must be in 1..{len(modes)}")
    requested_ids = tuple(mode)
    for candidate in modes:
        if frozenset(candidate.target_ids) == frozenset(requested_ids):
            return candidate
    raise ValueError(f"target ids do not form a canonical operating mode: {requested_ids}")


def _positions_from_environment(env: Any, info: Mapping[str, Any]) -> IntArray:
    if hasattr(env, "positions"):
        return _as_positions(env.positions, name="env.positions")
    for key in ("robot_positions", "agent_positions"):
        if key in info:
            return _as_positions(info[key], name=f"info[{key!r}]")
    raise KeyError("environment exposes neither positions nor position info")


def _info_array(
    info: Mapping[str, Any],
    keys: Sequence[str],
    *,
    name: str,
) -> np.ndarray | None:
    for key in keys:
        if key in info:
            return np.asarray(info[key])
    return None


def rollout_scripted_mode(
    env: Any,
    mode: OperatingMode | int | Sequence[str],
    *,
    seed: int | None = 0,
    max_steps: int = 256,
    steady_steps: int = 16,
) -> ScriptedModeResult:
    """运行一个规范模式，直到获得指定长度的连续稳定占据窗口。"""

    maximum_steps = _positive_integer(max_steps, name="max_steps")
    required_steady_steps = _positive_integer(steady_steps, name="steady_steps")
    if required_steady_steps > maximum_steps:
        raise ValueError("steady_steps must not exceed max_steps")
    if not hasattr(env, "config") or not isinstance(env.config, HRMRConfig):
        raise TypeError("env must expose an HRMRConfig as env.config")
    config = env.config
    selected_mode = _coerce_mode(mode, config)

    reset_result = env.reset(seed=seed)
    if not isinstance(reset_result, tuple) or len(reset_result) != 3:
        raise TypeError("env.reset() must return (observations, state, info)")
    reset_info = reset_result[2]
    if not isinstance(reset_info, Mapping):
        raise TypeError("env.reset() info must be a mapping")
    current_positions = _positions_from_environment(env, reset_info)
    controller = ShortestPathScriptedController(
        config,
        selected_mode.target_ids,
        initial_positions=current_positions,
    )

    position_history = [current_positions.copy()]
    action_history = []
    rewards = []
    global_costs = []
    local_costs = []
    occupancies = []
    collisions = []
    steady_run_start: int | None = None

    for _ in range(maximum_steps):
        actions = controller.actions(current_positions)
        step_result = env.step(actions)
        if not isinstance(step_result, tuple) or len(step_result) != 7:
            raise TypeError("env.step() must return the documented seven values")
        (
            _,
            _,
            team_reward,
            step_local_costs,
            terminated,
            truncated,
            step_info,
        ) = step_result
        if bool(terminated):
            raise AssertionError("HRMR continuing environment must never terminate")
        if not isinstance(step_info, Mapping):
            raise TypeError("env.step() info must be a mapping")

        current_positions = _positions_from_environment(env, step_info)
        occupancy = _info_array(
            step_info,
            ("target_occupancy", "target_coverage"),
            name="target occupancy",
        )
        if occupancy is None:
            occupancy = compute_target_occupancy(
                current_positions,
                config.monitoring_regions,
                config.target_ids,
            )
        step_cost_array = np.asarray(step_local_costs, dtype=np.float64)
        if step_cost_array.shape != (config.num_agents,):
            raise ValueError("env.step() local_costs has the wrong shape")
        step_global_cost = float(
            step_info.get("global_cost", np.sum(step_cost_array, dtype=np.float64))
        )
        collision_flags = _info_array(
            step_info,
            ("collision_flags",),
            name="collision flags",
        )
        if collision_flags is None:
            collision_flags = np.zeros(config.num_agents, dtype=np.bool_)

        action_history.append(actions.copy())
        position_history.append(current_positions.copy())
        rewards.append(float(team_reward))
        global_costs.append(step_global_cost)
        local_costs.append(step_cost_array.copy())
        occupancies.append(np.asarray(occupancy, dtype=np.float64).copy())
        collisions.append(np.asarray(collision_flags, dtype=np.bool_).copy())

        recorded_index = len(rewards) - 1
        region_ids = agent_region_ids(
            current_positions,
            config.monitoring_regions,
            config.target_ids,
        )
        if region_ids == controller.agent_target_ids:
            if steady_run_start is None:
                steady_run_start = recorded_index
            if len(rewards) - steady_run_start >= required_steady_steps:
                break
        else:
            steady_run_start = None

        if bool(truncated):
            break

    if steady_run_start is None or len(rewards) - steady_run_start < required_steady_steps:
        raise RuntimeError(
            f"scripted mode {selected_mode.index} did not obtain "
            f"{required_steady_steps} consecutive steady steps within {maximum_steps} steps"
        )

    reward_array = np.asarray(rewards, dtype=np.float64)
    global_cost_array = np.asarray(global_costs, dtype=np.float64)
    local_cost_array = np.asarray(local_costs, dtype=np.float64)
    occupancy_array = np.asarray(occupancies, dtype=np.float64)
    full_summary = summarize_rollout(
        reward_array,
        global_cost_array,
        occupancy_array,
        local_cost_array,
    )
    transient_summary, steady_summary = split_rollout_summaries(
        reward_array,
        global_cost_array,
        occupancy_array,
        local_cost_array,
        steady_start_step=steady_run_start,
    )
    return ScriptedModeResult(
        mode=selected_mode,
        agent_target_ids=controller.agent_target_ids,
        seed=seed,
        total_steps=len(rewards),
        transient_steps=steady_run_start,
        steady_steps=len(rewards) - steady_run_start,
        arrival_step=steady_run_start + 1,
        full_summary=full_summary,
        transient_summary=transient_summary,
        steady_summary=steady_summary,
        position_history=np.asarray(position_history, dtype=np.int64),
        requested_action_history=np.asarray(action_history, dtype=np.int64),
        reward_history=reward_array,
        global_cost_history=global_cost_array,
        local_cost_history=local_cost_array,
        target_occupancy_history=occupancy_array,
        collision_history=np.asarray(collisions, dtype=np.bool_),
    )


__all__ = [
    "CANONICAL_ASSIGNMENTS",
    "CANONICAL_MODE_TARGETS",
    "MIDPOINT_DUALS",
    "OperatingMode",
    "ScriptedController",
    "ScriptedModeResult",
    "ShortestPathScriptedController",
    "assign_targets_min_manhattan",
    "build_operating_modes",
    "canonical_operating_modes",
    "minimum_manhattan_assignment",
    "rollout_scripted_mode",
    "shortest_path_bfs",
    "validate_direct_operating_modes",
]
