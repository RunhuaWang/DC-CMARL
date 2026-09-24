"""HRMR 布局、解析 operating mode 与机器人轨迹绘图。"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

# CI/受限沙箱中的用户 matplotlib 目录可能不可写。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hrmr-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from hrmr.config import HRMRConfig
from hrmr.constants import HAZARDOUS_TARGET_IDS
from hrmr.geometry import position_in_grid
from hrmr.types import Position

_SAFE_COLOR = "#7DB7E8"
_HAZARD_COLOR = "#E9785D"
_GRID_COLOR = "#B9BEC7"
_AGENT_COLORS = ("#2C3E50", "#7B2CBF", "#00897B", "#D95F02")

RenderResult = tuple[Figure, Axes] | Path


def _new_or_existing_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """创建新 axes, 或清理调用者提供的 axes。"""

    if ax is None:
        figure, axes = plt.subplots(figsize=(8.4, 8.0), constrained_layout=True)
        return figure, axes
    ax.clear()
    return ax.figure, ax


def _validate_positions(
    positions: Iterable[Position],
    config: HRMRConfig,
    *,
    expected_count: int | None = None,
) -> tuple[Position, ...]:
    """验证绘图输入是地图内的内部坐标。"""

    try:
        checked = tuple(tuple(position) for position in positions)
    except TypeError as exc:
        raise TypeError("positions must be an iterable of two-dimensional coordinates") from exc
    if expected_count is not None and len(checked) != expected_count:
        raise ValueError(f"expected {expected_count} robot positions, received {len(checked)}")
    for index, position in enumerate(checked):
        if not position_in_grid(position, config.grid_height, config.grid_width):
            raise ValueError(f"positions[{index}] lies outside the grid: {position}")
    return checked


def _draw_static_map(ax: Axes, config: HRMRConfig) -> None:
    """绘制网格、目标区域与静态 target metadata。"""

    regions = config.monitoring_regions
    for target_id in config.target_ids:
        region = regions[target_id]
        rows = [position[0] for position in region]
        columns = [position[1] for position in region]
        is_hazardous = target_id in HAZARDOUS_TARGET_IDS
        hazard = float(config.hazard_intensities[target_id])
        face_color = _HAZARD_COLOR if is_hazardous else _SAFE_COLOR
        alpha = 0.28 + 0.40 * hazard if is_hazardous else 0.34
        patch = Rectangle(
            (min(columns) - 0.5, min(rows) - 0.5),
            max(columns) - min(columns) + 1,
            max(rows) - min(rows) + 1,
            facecolor=face_color,
            edgecolor="#34495E",
            linewidth=1.3,
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)

        row, column = config.target_centers[target_id]
        target_type = "hazard" if is_hazardous else "safe"
        ax.text(
            column,
            row - 0.42,
            (
                f"{target_id} ({target_type})\n"
                f"value={config.target_values[target_id]:g}, hazard={hazard:g}"
            ),
            ha="center",
            va="center",
            fontsize=7.6,
            color="#17202A",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            zorder=3,
        )

    ax.set_xlim(-0.5, config.grid_width - 0.5)
    ax.set_ylim(-0.5, config.grid_height - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(config.grid_width))
    ax.set_yticks(np.arange(config.grid_height))
    ax.set_xticks(np.arange(-0.5, config.grid_width, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, config.grid_height, 1.0), minor=True)
    ax.grid(which="minor", color=_GRID_COLOR, linewidth=0.55, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("Internal column (0-based, east →)")
    ax.set_ylabel("Internal row (0-based, north ↑)")


def _draw_robots(
    ax: Axes,
    positions: Sequence[Position],
    *,
    marker: str,
    label_prefix: str = "R",
) -> None:
    """按 agent index 绘制机器人位置。"""

    for index, (row, column) in enumerate(positions):
        color = _AGENT_COLORS[index % len(_AGENT_COLORS)]
        ax.scatter(
            [column],
            [row],
            marker=marker,
            s=150 if marker == "*" else 92,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            zorder=6,
        )
        ax.annotate(
            f"{label_prefix}{index + 1}",
            (column, row),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8.5,
            weight="bold",
            color=color,
            zorder=7,
        )


def _add_map_legend(ax: Axes, robot_label: str, *, robot_marker: str = "o") -> None:
    """添加区域类型和机器人图例。"""

    handles = [
        Patch(facecolor=_SAFE_COLOR, edgecolor="#34495E", alpha=0.45, label="Safe region"),
        Patch(
            facecolor=_HAZARD_COLOR,
            edgecolor="#34495E",
            alpha=0.55,
            label="Hazardous region",
        ),
        Line2D(
            [],
            [],
            marker=robot_marker,
            linestyle="none",
            markerfacecolor=_AGENT_COLORS[0],
            markeredgecolor="white",
            markersize=8,
            label=robot_label,
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)


def _save_or_return(
    figure: Figure,
    axes: Axes,
    output_path: str | Path | None,
    *,
    dpi: int,
    close_after_save: bool,
) -> RenderResult:
    """在请求输出路径时保存 PNG, 否则返回 figure/axes 供调用者继续编辑。"""

    if output_path is None:
        return figure, axes
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    if close_after_save:
        plt.close(figure)
    return destination


def render_layout(
    config: HRMRConfig,
    output_path: str | Path | None = None,
    positions: Iterable[Position] | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    dpi: int = 180,
) -> RenderResult:
    """绘制 HRMR 网格、8 个监控区域与 4 个初始机器人。

    所有坐标均是内部 ``(row, column)``、0-based 坐标。如果未传入
    ``positions``, 使用 PDF 指定的四个初始位置。传入 ``output_path``
    时直接保存并返回路径; 否则返回 ``(figure, axes)``。
    """

    uses_initial_positions = positions is None
    positions = _validate_positions(
        config.initial_positions if positions is None else positions,
        config,
        expected_count=config.num_agents,
    )
    figure, axes = _new_or_existing_axes(ax)
    _draw_static_map(axes, config)
    _draw_robots(axes, positions, marker="o")
    _add_map_legend(axes, "Initial robot" if uses_initial_positions else "Robot")
    axes.set_title(title or "HRMR Canonical Balanced Layout", fontsize=13, weight="bold")
    return _save_or_return(
        figure,
        axes,
        output_path,
        dpi=dpi,
        close_after_save=ax is None,
    )


def render_operating_mode(
    config: HRMRConfig,
    assignment: Sequence[str],
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    dpi: int = 180,
) -> RenderResult:
    """将一个解析 operating mode 的机器人放在所分配目标中心。"""

    assignments = tuple(assignment)
    if len(assignments) != config.num_agents:
        raise ValueError(
            f"an operating mode needs {config.num_agents} assignments; got {len(assignments)}"
        )
    if len(set(assignments)) != len(assignments):
        raise ValueError("operating-mode target assignments must be distinct")
    unknown = [target_id for target_id in assignments if target_id not in config.target_ids]
    if unknown:
        raise ValueError(f"unknown target assignments: {unknown}")

    positions = tuple(config.target_centers[target_id] for target_id in assignments)
    reward = sum(float(config.target_values[target_id]) for target_id in assignments)
    reward /= config.reward_normalizer
    cost = sum(float(config.hazard_intensities[target_id]) for target_id in assignments)
    cost /= config.num_agents

    figure, axes = _new_or_existing_axes(ax)
    _draw_static_map(axes, config)
    _draw_robots(axes, positions, marker="*")
    _add_map_legend(axes, "Assigned robot", robot_marker="*")
    axes.set_title(
        f"{title or 'HRMR Analytical Operating Mode'}\n"
        f"Targets: {', '.join(assignments)}   |   R={reward:.2f}, C={cost:.2f}",
        fontsize=12,
        weight="bold",
    )
    return _save_or_return(
        figure,
        axes,
        output_path,
        dpi=dpi,
        close_after_save=ax is None,
    )


def render_trajectory(
    config: HRMRConfig,
    trajectories: Sequence[Sequence[Position]] | np.ndarray,
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    dpi: int = 180,
) -> RenderResult:
    """绘制形状为 ``(time, agent, 2)`` 的内部坐标轨迹。"""

    positions = np.asarray(trajectories)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("trajectory must have shape (time, agent, 2)")
    if positions.shape[0] == 0:
        raise ValueError("trajectory must contain at least one time step")
    if positions.shape[1] != config.num_agents:
        raise ValueError(
            f"trajectory must contain {config.num_agents} agents; got {positions.shape[1]}"
        )
    if not np.issubdtype(positions.dtype, np.integer):
        raise TypeError("trajectory coordinates must be integers")
    checked = positions.astype(np.int64, copy=False)
    for time_index in range(checked.shape[0]):
        _validate_positions(checked[time_index], config, expected_count=config.num_agents)

    figure, axes = _new_or_existing_axes(ax)
    _draw_static_map(axes, config)
    for agent_index in range(config.num_agents):
        rows = checked[:, agent_index, 0]
        columns = checked[:, agent_index, 1]
        color = _AGENT_COLORS[agent_index % len(_AGENT_COLORS)]
        axes.plot(
            columns,
            rows,
            color=color,
            linewidth=2.0,
            alpha=0.88,
            label=f"R{agent_index + 1}",
            zorder=5,
        )
        axes.scatter(
            [columns[0]],
            [rows[0]],
            marker="o",
            s=48,
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=6,
        )
        axes.scatter(
            [columns[-1]],
            [rows[-1]],
            marker="X",
            s=72,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
        )
    axes.legend(loc="upper right", ncols=2, fontsize=8, framealpha=0.92)
    axes.set_title(title or "HRMR Robot Trajectories", fontsize=13, weight="bold")
    return _save_or_return(
        figure,
        axes,
        output_path,
        dpi=dpi,
        close_after_save=ax is None,
    )


__all__ = ["RenderResult", "render_layout", "render_operating_mode", "render_trajectory"]
