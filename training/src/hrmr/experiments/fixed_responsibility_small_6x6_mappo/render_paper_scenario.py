"""生成论文使用的 6×6 固定责任监控场景图。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# 在导入 pyplot 前指定可写缓存目录。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hrmr-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from hrmr.experiments.assigned_targets_ppo.environment import DEFAULT_ASSIGNMENTS
from hrmr.experiments.fixed_responsibility_small_6x6_mappo.config import (
    CONFIG_PATHS,
    MAP_ROOT,
    load_config,
    load_environment,
)

OUTPUT_DIRECTORY = MAP_ROOT / "images" / "paper"
PNG_OUTPUT_PATH = OUTPUT_DIRECTORY / "fixed_responsibility_6x6_scenario.png"
PDF_OUTPUT_PATH = OUTPUT_DIRECTORY / "fixed_responsibility_6x6_scenario.pdf"

# Okabe-Ito 色板：色盲友好、打印辨识度高。
ROBOT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
GRID_COLOR = "#CBD2D9"
TEXT_COLOR = "#17202A"
ANNOTATION_SIZE = 18.0


def _to_plot_xy(position: tuple[int, int]) -> tuple[float, float]:
    """内部 (row,column), 0-based 转换为图中的 (x,y), 1-based。"""

    row, column = position
    return float(column + 1), float(row + 1)


def _draw_robot(axis, x: float, y: float, color: str, label: str) -> None:
    """在一个网格内绘制紧凑机器人图标。"""

    wheel_color = "#30343B"
    for offset in (-0.22, 0.22):
        axis.add_patch(
            Circle(
                (x + offset, y - 0.18),
                radius=0.075,
                facecolor=wheel_color,
                edgecolor="white",
                linewidth=0.7,
                zorder=7,
            )
        )
    axis.add_patch(
        FancyBboxPatch(
            (x - 0.27, y - 0.16),
            0.54,
            0.38,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor="#24313A",
            linewidth=1.1,
            zorder=8,
        )
    )
    axis.plot(
        (x, x),
        (y + 0.22, y + 0.34),
        color="#24313A",
        linewidth=1.1,
        solid_capstyle="round",
        zorder=8,
    )
    axis.add_patch(
        Circle(
            (x, y + 0.37),
            radius=0.035,
            facecolor=color,
            edgecolor="#24313A",
            linewidth=0.7,
            zorder=8,
        )
    )
    axis.text(
        x,
        y - 0.02,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=ANNOTATION_SIZE,
        fontweight="normal",
        zorder=10,
    )


def render_paper_scenario(
    png_path: str | Path = PNG_OUTPUT_PATH,
    pdf_path: str | Path = PDF_OUTPUT_PATH,
) -> tuple[Path, Path]:
    """按照真实配置绘制场景，并保存高分辨率 PNG 与矢量 PDF。"""

    experiment = load_config(CONFIG_PATHS[0.0])
    environment = load_environment(experiment)
    owners = {
        target_id: agent_index
        for agent_index, pair in enumerate(DEFAULT_ASSIGNMENTS)
        for target_id in pair
    }
    png_destination = Path(png_path).expanduser().resolve()
    pdf_destination = Path(pdf_path).expanduser().resolve()
    png_destination.parent.mkdir(parents=True, exist_ok=True)
    pdf_destination.parent.mkdir(parents=True, exist_ok=True)

    style = {
        "font.family": "DejaVu Sans",
        "font.size": ANNOTATION_SIZE,
        "axes.titlesize": ANNOTATION_SIZE,
        "axes.labelsize": ANNOTATION_SIZE,
        "xtick.labelsize": ANNOTATION_SIZE,
        "ytick.labelsize": ANNOTATION_SIZE,
        "legend.fontsize": ANNOTATION_SIZE,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7.4, 7.4), constrained_layout=True)
        axis.set_facecolor("#FAFBFC")

        for target_id in environment.target_ids:
            x, y = _to_plot_xy(environment.target_centers[target_id])
            owner = owners[target_id]
            cost = float(environment.hazard_intensities[target_id])
            reward = float(environment.target_values[target_id])
            axis.add_patch(
                Rectangle(
                    (x - 0.47, y - 0.47),
                    0.94,
                    0.94,
                    facecolor="none",
                    edgecolor=ROBOT_COLORS[owner],
                    linewidth=3.0,
                    joinstyle="round",
                    zorder=3,
                )
            )
            axis.text(
                x,
                y,
                f"{target_id}\n$r_t = {reward:.1f}$\n$g_t = {cost:.1f}$",
                ha="center",
                va="center",
                color=TEXT_COLOR,
                fontsize=ANNOTATION_SIZE,
                fontweight="normal",
                linespacing=1.25,
                zorder=5,
            )

        for agent, position in enumerate(environment.initial_positions):
            x, y = _to_plot_xy(position)
            _draw_robot(axis, x, y, ROBOT_COLORS[agent], f"A{agent + 1}")

        axis.set_xlim(0.5, environment.grid_width + 0.5)
        axis.set_ylim(0.5, environment.grid_height + 0.5)
        axis.set_aspect("equal", adjustable="box")
        for boundary in range(1, environment.grid_width):
            axis.axvline(boundary + 0.5, color=GRID_COLOR, linewidth=0.8, zorder=1)
        for boundary in range(1, environment.grid_height):
            axis.axhline(boundary + 0.5, color=GRID_COLOR, linewidth=0.8, zorder=1)
        axis.set_xticks(())
        axis.set_yticks(())
        axis.set_xlabel("")
        axis.set_ylabel("")

        figure.savefig(png_destination, dpi=600, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_destination, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return png_destination, pdf_destination


def main() -> int:
    png_path, pdf_path = render_paper_scenario()
    print(f"[DONE] PNG: {png_path}")
    print(f"[DONE] PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PDF_OUTPUT_PATH", "PNG_OUTPUT_PATH", "render_paper_scenario"]
