"""HRMR 网格动作、边界处理与同步冲突解析。

内部位置统一表示为 ``(row, column)``。其中 row 向北递增，column
向东递增。冲突解析只依赖当前位置和经过边界处理的候选位置，因此是
一个不使用随机数、也不依赖机器人遍历顺序的纯函数。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import ACTION_DELTAS, Action

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class TransitionResult:
    """一次联合位置转移的不可变诊断结果。"""

    requested_actions: IntArray
    executed_actions: IntArray
    bounded_candidate_positions: IntArray
    final_positions: IntArray
    boundary_block_flags: BoolArray
    collision_flags: BoolArray

    def __post_init__(self) -> None:
        """复制并锁定数组，避免环境日志被调用方意外改写。"""

        for item in fields(self):
            value = np.array(getattr(self, item.name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, item.name, value)

    @property
    def candidate_positions(self) -> IntArray:
        """``bounded_candidate_positions`` 的简短兼容别名。"""

        return self.bounded_candidate_positions

    @property
    def boundary_flags(self) -> BoolArray:
        """``boundary_block_flags`` 的简短兼容别名。"""

        return self.boundary_block_flags


def _as_positions(value: ArrayLike, *, name: str) -> IntArray:
    """把输入验证为形状 ``(n, 2)`` 的整数位置数组。"""

    array = np.asarray(value, dtype=object)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape (n_agents, 2)")
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) for item in array.flat
    ):
        raise TypeError(f"{name} must contain integers")
    return np.fromiter((int(item) for item in array.flat), dtype=np.int64).reshape(array.shape)


def _validate_unique_positions(positions: IntArray) -> None:
    """检查多机器人状态不含重叠位置。"""

    if len(positions) != len(np.unique(positions, axis=0)):
        raise ValueError("current_positions must not contain overlapping robots")


def _validate_grid_dimension(value: int, *, name: str) -> int:
    """验证正整数网格维度（布尔值不视为整数维度）。"""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_current_positions_in_grid(
    positions: IntArray,
    *,
    grid_width: int,
    grid_height: int,
) -> None:
    """保证边界回退所使用的当前位置本身合法。"""

    if np.any(positions[:, 0] < 0) or np.any(positions[:, 0] >= grid_height):
        raise ValueError("current position row is outside the grid")
    if np.any(positions[:, 1] < 0) or np.any(positions[:, 1] >= grid_width):
        raise ValueError("current position column is outside the grid")


def validate_actions(actions: ArrayLike, n_agents: int | None = None) -> IntArray:
    """验证联合动作并返回一维整数副本。

    ``Action`` 是 ``IntEnum``，因此枚举成员和对应的 NumPy/Python 整数都可
    接受。浮点数（即便值为整数）、布尔值和越界整数都会被拒绝，以免把
    策略输出错误静默转换为合法动作。
    """

    array = np.asarray(actions, dtype=object)
    if array.ndim != 1:
        raise ValueError("actions must have shape (n_agents,)")
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) for item in array):
        raise TypeError("actions must contain Action members or integers")

    if n_agents is not None:
        expected = _validate_grid_dimension(n_agents, name="n_agents")
        if len(array) != expected:
            raise ValueError(f"expected {expected} actions, got {len(array)}")

    integer_values = [int(item) for item in array]
    valid_values = np.fromiter((int(action) for action in Action), dtype=np.int64)
    valid_value_set = set(valid_values.tolist())
    invalid = sorted({item for item in integer_values if item not in valid_value_set})
    if invalid:
        raise ValueError(f"invalid action value(s): {invalid}")
    return np.asarray(integer_values, dtype=np.int64)


def apply_action_slip(
    requested_actions: ArrayLike,
    slip_probability: float,
    rng: np.random.Generator,
) -> IntArray:
    """独立地向各机器人动作施加 slip。

    发生 slip 时从完整五动作集合均匀重采样，原请求动作也包含在采样集合
    内。概率为零时不会消耗随机数生成器状态。
    """

    actions = validate_actions(requested_actions)
    if isinstance(slip_probability, (bool, np.bool_)) or not isinstance(slip_probability, Real):
        raise TypeError("slip_probability must be a real number")
    probability = float(slip_probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("slip_probability must be in [0, 1]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    executed = actions.copy()
    if probability == 0.0 or len(executed) == 0:
        return executed

    slip_flags = rng.random(len(executed)) < probability
    slip_count = int(np.count_nonzero(slip_flags))
    if slip_count:
        action_values = np.fromiter((int(action) for action in Action), dtype=np.int64)
        executed[slip_flags] = rng.choice(action_values, size=slip_count, replace=True)
    return executed


def apply_displacements(current_positions: ArrayLike, executed_actions: ArrayLike) -> IntArray:
    """按内部坐标约定计算尚未处理边界的原始候选位置。"""

    positions = _as_positions(current_positions, name="current_positions")
    actions = validate_actions(executed_actions, n_agents=len(positions))
    displacements = np.asarray(
        [ACTION_DELTAS[Action(int(action))] for action in actions],
        dtype=np.int64,
    )
    return positions + displacements


def compute_boundary_block_flags(
    candidate_positions: ArrayLike,
    grid_width: int,
    grid_height: int,
) -> BoolArray:
    """标出原始候选位置中越过地图边界的机器人。"""

    candidates = _as_positions(candidate_positions, name="candidate_positions")
    width = _validate_grid_dimension(grid_width, name="grid_width")
    height = _validate_grid_dimension(grid_height, name="grid_height")
    return (
        (candidates[:, 0] < 0)
        | (candidates[:, 0] >= height)
        | (candidates[:, 1] < 0)
        | (candidates[:, 1] >= width)
    )


def apply_boundary_rule(
    candidate_positions: ArrayLike,
    current_positions: ArrayLike,
    grid_width: int,
    grid_height: int,
) -> IntArray:
    """让试图离开地图的机器人回退至各自当前位置。"""

    candidates = _as_positions(candidate_positions, name="candidate_positions")
    current = _as_positions(current_positions, name="current_positions")
    if candidates.shape != current.shape:
        raise ValueError("candidate_positions and current_positions must have the same shape")
    width = _validate_grid_dimension(grid_width, name="grid_width")
    height = _validate_grid_dimension(grid_height, name="grid_height")
    _validate_current_positions_in_grid(current, grid_width=width, grid_height=height)
    flags = compute_boundary_block_flags(candidates, width, height)
    bounded = candidates.copy()
    bounded[flags] = current[flags]
    return bounded


def compute_bounded_candidate_positions(
    current_positions: ArrayLike,
    executed_actions: ArrayLike,
    grid_width: int,
    grid_height: int,
) -> tuple[IntArray, BoolArray]:
    """计算原始位移、边界回退候选位置和边界标志。"""

    current = _as_positions(current_positions, name="current_positions")
    raw_candidates = apply_displacements(current, executed_actions)
    flags = compute_boundary_block_flags(raw_candidates, grid_width, grid_height)
    bounded = apply_boundary_rule(
        raw_candidates,
        current,
        grid_width=grid_width,
        grid_height=grid_height,
    )
    return bounded, flags


def compute_candidates(
    current_positions: ArrayLike,
    executed_actions: ArrayLike,
    grid_width: int,
    grid_height: int,
) -> tuple[IntArray, BoolArray]:
    """``compute_bounded_candidate_positions`` 的简短别名。"""

    return compute_bounded_candidate_positions(
        current_positions,
        executed_actions,
        grid_width,
        grid_height,
    )


def resolve_synchronous_conflicts(
    current_positions: ArrayLike,
    bounded_candidate_positions: ArrayLike,
) -> tuple[IntArray, BoolArray]:
    """确定性地解析一次同步多机器人移动。

    解析过程先阻止同目标与两机器人交换，再把“目标格占用者无法离开”
    沿依赖链反向传播。以空格结尾的链条可以整体移动；长度至少为三的
    置换环也可以同步旋转。算法不按机器人索引选择赢家，因此对输入排列
    等变，最终位置始终保持无重叠。
    """

    current = _as_positions(current_positions, name="current_positions")
    candidates = _as_positions(
        bounded_candidate_positions,
        name="bounded_candidate_positions",
    )
    if candidates.shape != current.shape:
        raise ValueError(
            "bounded_candidate_positions and current_positions must have the same shape"
        )
    _validate_unique_positions(current)

    n_agents = len(current)
    moving = np.any(candidates != current, axis=1)
    blocked = np.zeros(n_agents, dtype=np.bool_)

    # 规则 1：同一候选目标没有基于索引的赢家，相关机器人全部被阻止。
    target_groups: dict[tuple[int, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        target_groups.setdefault((int(candidate[0]), int(candidate[1])), []).append(index)
    for group in target_groups.values():
        if len(group) > 1:
            blocked[group] = True

    origin_owner = {
        (int(position[0]), int(position[1])): index for index, position in enumerate(current)
    }

    # 规则 2：仅二元交换被禁止；更长的同步置换环是合法移动。
    for index in range(n_agents):
        if not moving[index]:
            continue
        owner = origin_owner.get((int(candidates[index, 0]), int(candidates[index, 1])))
        if owner is None or owner == index or not moving[owner]:
            continue
        if np.array_equal(candidates[owner], current[index]):
            blocked[index] = True
            blocked[owner] = True

    # 规则 3：一个阻塞会沿“进入其原位置”的依赖边反向传播至不动点。
    changed = True
    while changed:
        changed = False
        for index in range(n_agents):
            if not moving[index] or blocked[index]:
                continue
            owner = origin_owner.get((int(candidates[index, 0]), int(candidates[index, 1])))
            if owner is not None and (not moving[owner] or blocked[owner]):
                blocked[index] = True
                changed = True

    final = current.copy()
    movable = moving & ~blocked
    final[movable] = candidates[movable]

    # 这也是对未来规则修改的防御性不变量检查，不参与冲突选择。
    if len(final) != len(np.unique(final, axis=0)):
        raise RuntimeError("conflict resolution produced overlapping final positions")
    return final, blocked


def resolve_conflicts(
    current_positions: ArrayLike,
    bounded_candidate_positions: ArrayLike,
) -> tuple[IntArray, BoolArray]:
    """``resolve_synchronous_conflicts`` 的简短别名。"""

    return resolve_synchronous_conflicts(current_positions, bounded_candidate_positions)


def transition_positions(
    current_positions: ArrayLike,
    requested_actions: ArrayLike,
    grid_width: int,
    grid_height: int,
    slip_probability: float,
    rng: np.random.Generator,
) -> TransitionResult:
    """依次执行动作验证、slip、边界处理与同步冲突解析。"""

    current = _as_positions(current_positions, name="current_positions")
    _validate_unique_positions(current)
    width = _validate_grid_dimension(grid_width, name="grid_width")
    height = _validate_grid_dimension(grid_height, name="grid_height")
    _validate_current_positions_in_grid(current, grid_width=width, grid_height=height)

    requested = validate_actions(requested_actions, n_agents=len(current))
    executed = apply_action_slip(requested, slip_probability, rng)
    bounded, boundary_flags = compute_bounded_candidate_positions(
        current,
        executed,
        grid_width=width,
        grid_height=height,
    )
    final, collision_flags = resolve_synchronous_conflicts(current, bounded)
    return TransitionResult(
        requested_actions=requested,
        executed_actions=executed,
        bounded_candidate_positions=bounded,
        final_positions=final,
        boundary_block_flags=boundary_flags,
        collision_flags=collision_flags,
    )


__all__ = [
    "TransitionResult",
    "apply_action_slip",
    "apply_boundary_rule",
    "apply_displacements",
    "compute_boundary_block_flags",
    "compute_bounded_candidate_positions",
    "compute_candidates",
    "resolve_conflicts",
    "resolve_synchronous_conflicts",
    "transition_positions",
    "validate_actions",
]
