"""HRMR actor observation、环境状态与 dual 条件输入编码。

包内位置统一是 ``(row, column)``，但 PDF 的特征顺序是 ``(x, y)``。
因此本模块始终先编码 ``column / (W - 1)``，再编码
``row / (H - 1)``；相对位置也遵循相同顺序。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.constants import TARGET_IDS
from hrmr.types import TargetId, TargetPositions, TargetScalars

FloatArray = NDArray[np.float64]

SELF_POSITION_SIZE = 2
TARGET_FEATURE_SIZE = 5
TEAMMATE_POSITION_SIZE = 2


def _positive_dimension(value: int, *, name: str, minimum: int = 1) -> int:
    """验证整数维度。"""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: Real, *, name: str) -> float:
    """验证并返回有限实数，明确拒绝 ``bool``。"""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positions_array(
    positions: ArrayLike,
    *,
    grid_width: int,
    grid_height: int,
    name: str = "positions",
) -> NDArray[np.int64]:
    """验证内部 ``(row, column)`` 位置数组及地图边界。"""

    array = np.asarray(positions)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape (n_agents, 2)")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one agent")
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integer internal coordinates")
    checked = array.astype(np.int64, copy=True)
    if np.any(checked[:, 0] < 0) or np.any(checked[:, 0] >= grid_height):
        raise ValueError(f"{name} contains a row outside the grid")
    if np.any(checked[:, 1] < 0) or np.any(checked[:, 1] >= grid_width):
        raise ValueError(f"{name} contains a column outside the grid")
    return checked


def _target_centers_array(
    target_centers: TargetPositions,
    target_ids: Sequence[TargetId],
    *,
    grid_width: int,
    grid_height: int,
) -> NDArray[np.int64]:
    """按 canonical target order 读取内部中心坐标。"""

    if not isinstance(target_centers, Mapping):
        raise TypeError("target_centers must be a mapping")
    missing = [target_id for target_id in target_ids if target_id not in target_centers]
    if missing:
        raise KeyError(f"target_centers is missing target ids: {missing}")
    return _positions_array(
        [target_centers[target_id] for target_id in target_ids],
        grid_width=grid_width,
        grid_height=grid_height,
        name="target_centers",
    )


def _target_scalars(
    values: TargetScalars,
    target_ids: Sequence[TargetId],
    *,
    name: str,
    positive: bool,
) -> FloatArray:
    """按 canonical target order 读取 target scalar。"""

    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    missing = [target_id for target_id in target_ids if target_id not in values]
    if missing:
        raise KeyError(f"{name} is missing target ids: {missing}")
    checked = np.asarray(
        [_finite_real(values[target_id], name=f"{name}[{target_id}]") for target_id in target_ids],
        dtype=np.float64,
    )
    if positive:
        if np.any(checked <= 0.0):
            raise ValueError(f"all {name} entries must be positive")
    elif np.any((checked < 0.0) | (checked > 1.0)):
        raise ValueError(f"all {name} entries must lie in [0, 1]")
    return checked


def _coverage_array(coverage: ArrayLike, num_targets: int) -> FloatArray:
    """验证 canonical coverage vector，并转成 ``float64``。"""

    array = np.asarray(coverage)
    if array.ndim != 1 or array.shape != (num_targets,):
        raise ValueError(f"coverage must have shape ({num_targets},)")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError("coverage must contain binary numeric values")
    checked = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(checked)):
        raise ValueError("coverage must be finite")
    if not np.all((checked == 0.0) | (checked == 1.0)):
        raise ValueError("coverage entries must be exactly 0 or 1")
    return checked


def observation_layout(
    use_agent_id: bool = False,
    num_agents: int = 4,
    num_targets: int = 8,
) -> dict[str, slice]:
    """返回单个 actor observation 的连续 feature 区间。

    ``target_features`` 内按 ``S1,S2,S3,S4,H1,H2,H3,H4`` 排列，
    每个五维 block 依次是
    ``delta_x, delta_y, value/max_value, hazard, coverage``。
    ``teammate_relative_positions`` 按 agent index 升序并跳过自身，每个
    teammate 使用 ``delta_x, delta_y``。默认 layout 为 ``2 + 40 + 6 = 48``；
    可选 one-hot ID 位于末尾，使其成为 52 维。
    """

    if not isinstance(use_agent_id, (bool, np.bool_)):
        raise TypeError("use_agent_id must be a boolean")
    agents = _positive_dimension(num_agents, name="num_agents")
    targets = _positive_dimension(num_targets, name="num_targets")

    self_stop = SELF_POSITION_SIZE
    target_stop = self_stop + TARGET_FEATURE_SIZE * targets
    teammate_stop = target_stop + TEAMMATE_POSITION_SIZE * (agents - 1)
    layout = {
        "self_position": slice(0, self_stop),
        "target_features": slice(self_stop, target_stop),
        "teammate_relative_positions": slice(target_stop, teammate_stop),
    }
    if bool(use_agent_id):
        layout["agent_id"] = slice(teammate_stop, teammate_stop + agents)
    return layout


def _normalized_xy(
    positions: NDArray[np.int64],
    *,
    grid_width: int,
    grid_height: int,
) -> FloatArray:
    """把内部 ``(row, col)`` 编码为 PDF 顺序的归一化 ``(x, y)``。"""

    result = np.empty(positions.shape, dtype=np.float64)
    result[:, 0] = positions[:, 1] / float(grid_width - 1)
    result[:, 1] = positions[:, 0] / float(grid_height - 1)
    return result


def build_actor_observations(
    positions: ArrayLike,
    target_centers: TargetPositions,
    target_values: TargetScalars,
    hazard_intensities: TargetScalars,
    coverage: ArrayLike,
    grid_width: int,
    grid_height: int,
    use_agent_id: bool = False,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> FloatArray:
    """为全部机器人构造 PDF 式 (9.5) 的基础 actor observations。

    返回 shape 为 ``(N, 48)``（默认无 ID）或 ``(N, 52)``（带四维
    one-hot ID）。本函数不追加 dual；算法侧可随后调用
    :func:`append_normalized_dual`。
    """

    width = _positive_dimension(grid_width, name="grid_width", minimum=2)
    height = _positive_dimension(grid_height, name="grid_height", minimum=2)
    if not isinstance(use_agent_id, (bool, np.bool_)):
        raise TypeError("use_agent_id must be a boolean")
    ids = tuple(target_ids)
    if not ids:
        raise ValueError("target_ids must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("target_ids must not contain duplicates")

    agents = _positions_array(
        positions,
        grid_width=width,
        grid_height=height,
    )
    centers = _target_centers_array(
        target_centers,
        ids,
        grid_width=width,
        grid_height=height,
    )
    values = _target_scalars(
        target_values,
        ids,
        name="target_values",
        positive=True,
    )
    hazards = _target_scalars(
        hazard_intensities,
        ids,
        name="hazard_intensities",
        positive=False,
    )
    occupancies = _coverage_array(coverage, len(ids))

    n_agents = len(agents)
    layout = observation_layout(bool(use_agent_id), n_agents, len(ids))
    final_stop = max(feature_slice.stop for feature_slice in layout.values())
    observations = np.empty((n_agents, final_stop), dtype=np.float64)
    normalized_agents = _normalized_xy(
        agents,
        grid_width=width,
        grid_height=height,
    )
    max_value = float(np.max(values))

    for agent_index, (agent_row, agent_column) in enumerate(agents):
        observations[agent_index, layout["self_position"]] = normalized_agents[agent_index]

        target_blocks = np.empty((len(ids), TARGET_FEATURE_SIZE), dtype=np.float64)
        target_blocks[:, 0] = (centers[:, 1] - agent_column) / float(width - 1)
        target_blocks[:, 1] = (centers[:, 0] - agent_row) / float(height - 1)
        target_blocks[:, 2] = values / max_value
        target_blocks[:, 3] = hazards
        target_blocks[:, 4] = occupancies
        observations[agent_index, layout["target_features"]] = target_blocks.ravel()

        teammate_indices = [
            teammate_index for teammate_index in range(n_agents) if teammate_index != agent_index
        ]
        teammate_delta = normalized_agents[teammate_indices] - normalized_agents[agent_index]
        observations[
            agent_index,
            layout["teammate_relative_positions"],
        ] = teammate_delta.ravel()

        if bool(use_agent_id):
            observations[agent_index, layout["agent_id"]] = 0.0
            observations[agent_index, layout["agent_id"].start + agent_index] = 1.0

    return observations


def build_actor_observation(
    agent_index: int,
    positions: ArrayLike,
    target_centers: TargetPositions,
    target_values: TargetScalars,
    hazard_intensities: TargetScalars,
    coverage: ArrayLike,
    grid_width: int,
    grid_height: int,
    use_agent_id: bool = False,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> FloatArray:
    """构造一个指定 agent 的 observation，规则与批量函数完全相同。"""

    if isinstance(agent_index, (bool, np.bool_)) or not isinstance(agent_index, Integral):
        raise TypeError("agent_index must be an integer")
    observations = build_actor_observations(
        positions,
        target_centers,
        target_values,
        hazard_intensities,
        coverage,
        grid_width,
        grid_height,
        use_agent_id,
        target_ids,
    )
    index = int(agent_index)
    if not 0 <= index < len(observations):
        raise IndexError("agent_index is outside the agent range")
    return observations[index].copy()


def normalize_dual(dual_lambda: float, dual_upper_bound: float) -> float:
    """严格验证原始 dual 范围并返回 PDF 式 (9.8) 的 ``lambda/Lambda``。

    越界输入会抛出异常而不是静默裁剪；投影只属于独立的 dual-update
    helper。
    """

    dual = _finite_real(dual_lambda, name="dual_lambda")
    upper = _finite_real(dual_upper_bound, name="dual_upper_bound")
    if upper <= 0.0:
        raise ValueError("dual_upper_bound must be positive")
    if not 0.0 <= dual <= upper:
        raise ValueError("dual_lambda must lie in [0, dual_upper_bound]")
    return dual / upper


def append_normalized_dual(
    observation: np.ndarray,
    dual_lambda: float,
    dual_upper_bound: float,
) -> FloatArray:
    """向单个或批量 observation 的末维追加相同的 normalized dual。

     ``(48,) -> (49,)``、``(52,) -> (53,)``；批量 ``(N, D)`` 输入则
    返回 ``(N, D + 1)``。输入数组不会被修改。
    """

    if not isinstance(observation, np.ndarray):
        raise TypeError("observation must be a numpy.ndarray")
    if observation.ndim not in {1, 2}:
        raise ValueError("observation must be one- or two-dimensional")
    if observation.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("observation must contain real numeric values")
    checked = observation.astype(np.float64, copy=True)
    if not np.all(np.isfinite(checked)):
        raise ValueError("observation must contain only finite values")
    normalized = normalize_dual(dual_lambda, dual_upper_bound)
    if checked.ndim == 1:
        return np.concatenate((checked, np.asarray([normalized], dtype=np.float64)))
    dual_column = np.full((checked.shape[0], 1), normalized, dtype=np.float64)
    return np.concatenate((checked, dual_column), axis=1)


def build_environment_state(
    positions: ArrayLike,
    coverage: ArrayLike,
    grid_width: int,
    grid_height: int,
) -> FloatArray:
    """构造不含算法变量的联合环境状态 ``[agent xy, coverage]``。

    基础配置中该向量维数是 ``2 * 4 + 8 = 16``。Agent 按 index 排列，
    每个位置都严格使用 PDF 的 ``(x, y)`` 特征顺序。
    """

    width = _positive_dimension(grid_width, name="grid_width", minimum=2)
    height = _positive_dimension(grid_height, name="grid_height", minimum=2)
    agents = _positions_array(
        positions,
        grid_width=width,
        grid_height=height,
    )
    occupancies = np.asarray(coverage)
    if occupancies.ndim != 1:
        raise ValueError("coverage must be one-dimensional")
    checked_coverage = _coverage_array(occupancies, len(occupancies))
    normalized_positions = _normalized_xy(
        agents,
        grid_width=width,
        grid_height=height,
    )
    return np.concatenate((normalized_positions.ravel(), checked_coverage))


def build_centralized_critic_state(
    positions: ArrayLike,
    coverage: ArrayLike,
    grid_width: int,
    grid_height: int,
    dual_lambda: float,
    dual_upper_bound: float,
) -> FloatArray:
    """构造 PDF 式 (9.10) 的 critic input；基础配置维数为 17。"""

    environment_state = build_environment_state(
        positions,
        coverage,
        grid_width,
        grid_height,
    )
    return append_normalized_dual(
        environment_state,
        dual_lambda,
        dual_upper_bound,
    )


__all__ = [
    "TARGET_FEATURE_SIZE",
    "append_normalized_dual",
    "build_actor_observation",
    "build_actor_observations",
    "build_centralized_critic_state",
    "build_environment_state",
    "normalize_dual",
    "observation_layout",
]
