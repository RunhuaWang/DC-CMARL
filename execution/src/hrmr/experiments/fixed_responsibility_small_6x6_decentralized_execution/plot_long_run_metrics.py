"""IDE 直接运行：从执行日志绘制累计 reward 与约束违反曲线。"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path

# 在导入 pyplot 前指定可写缓存，保证 IDE/沙箱中也能独立运行。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hrmr-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hrmr.experiments.fixed_responsibility_small_6x6_decentralized_execution.config import (
    load_config,
)

REQUIRED_COLUMNS = (
    "environment_steps",
    "constraint_threshold",
    "seed",
    "window_mean_reward",
    "window_global_cost",
)

# 使用高区分度的离散色板，避免相邻阈值在连续色图中颜色过于接近。
THRESHOLD_COLORS = (
    "#1f77b4",  # 蓝
    "#ff7f0e",  # 橙
    "#2ca02c",  # 绿
    "#d62728",  # 红
    "#9467bd",  # 紫
    "#8c564b",  # 棕
    "#e377c2",  # 粉
    "#17becf",  # 青
    "#bcbd22",  # 橄榄
    "#7f7f7f",  # 灰
)
PLOT_FONT_SIZE = 20.0


def _read_window_rows(path: Path) -> list[dict[str, float | int]]:
    """读取每个 H-step 窗口的真实 reward/cost，不使用 critic 输出。"""

    if not path.is_file():
        raise FileNotFoundError(f"execution log does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"execution log is missing columns: {sorted(missing)}")
        rows: list[dict[str, float | int]] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                row: dict[str, float | int] = {
                    "environment_steps": int(raw["environment_steps"]),
                    "constraint_threshold": float(raw["constraint_threshold"]),
                    "seed": int(raw["seed"]),
                    "window_mean_reward": float(raw["window_mean_reward"]),
                    "window_global_cost": float(raw["window_global_cost"]),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric field at CSV line {line_number}") from error
            numeric = tuple(float(row[name]) for name in REQUIRED_COLUMNS)
            if not all(isfinite(value) for value in numeric):
                raise ValueError(f"non-finite value at CSV line {line_number}")
            if row["environment_steps"] <= 0 or row["seed"] < 0:
                raise ValueError(f"invalid step or seed at CSV line {line_number}")
            if row["constraint_threshold"] < 0.0 or row["window_global_cost"] < 0.0:
                raise ValueError(f"negative threshold or cost at CSV line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("execution log is empty")
    return rows


def compute_long_run_metrics(
    window_rows: Sequence[dict[str, float | int]],
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    """先逐 seed 求累计时间平均，再跨 seed 求均值和标准差。"""

    grouped: dict[tuple[float, int], list[dict[str, float | int]]] = defaultdict(list)
    for row in window_rows:
        grouped[(float(row["constraint_threshold"]), int(row["seed"]))].append(row)

    seed_level: list[dict[str, float | int]] = []
    expected_seed_sets: dict[float, set[int]] = defaultdict(set)
    for (threshold, seed), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["environment_steps"]))
        steps = np.asarray([int(row["environment_steps"]) for row in ordered], dtype=np.int64)
        if len(np.unique(steps)) != len(steps):
            raise ValueError(f"duplicate T values for c={threshold:g}, seed={seed}")
        durations = np.diff(np.concatenate((np.asarray((0,), dtype=np.int64), steps)))
        if np.any(durations <= 0) or np.any(durations != durations[0]):
            raise ValueError(
                f"incomplete or non-uniform execution windows for c={threshold:g}, seed={seed}"
            )
        rewards = np.asarray([float(row["window_mean_reward"]) for row in ordered])
        costs = np.asarray([float(row["window_global_cost"]) for row in ordered])
        cumulative_reward = np.cumsum(rewards * durations) / steps
        cumulative_cost = np.cumsum(costs * durations) / steps
        violations = np.maximum(cumulative_cost - threshold, 0.0)
        for index, step in enumerate(steps):
            seed_level.append(
                {
                    "constraint_threshold": threshold,
                    "seed": seed,
                    "environment_steps": int(step),
                    "long_run_average_reward": float(cumulative_reward[index]),
                    "long_run_average_global_cost": float(cumulative_cost[index]),
                    "constraint_violation": float(violations[index]),
                }
            )
        expected_seed_sets[threshold].add(seed)

    by_threshold_step: dict[tuple[float, int], list[dict[str, float | int]]] = defaultdict(list)
    for row in seed_level:
        key = (float(row["constraint_threshold"]), int(row["environment_steps"]))
        by_threshold_step[key].append(row)

    summary: list[dict[str, float | int]] = []
    for (threshold, step), rows in sorted(by_threshold_step.items()):
        observed_seeds = {int(row["seed"]) for row in rows}
        if observed_seeds != expected_seed_sets[threshold]:
            raise ValueError(f"seed set is inconsistent at c={threshold:g}, T={step}")
        reward = np.asarray([float(row["long_run_average_reward"]) for row in rows])
        cost = np.asarray([float(row["long_run_average_global_cost"]) for row in rows])
        violation = np.asarray([float(row["constraint_violation"]) for row in rows])
        summary.append(
            {
                "constraint_threshold": threshold,
                "environment_steps": step,
                "num_seeds": len(rows),
                "long_run_average_reward_mean": float(np.mean(reward)),
                "long_run_average_reward_std": float(np.std(reward)),
                "long_run_average_global_cost_mean": float(np.mean(cost)),
                "long_run_average_global_cost_std": float(np.std(cost)),
                "constraint_violation_mean": float(np.mean(violation)),
                "constraint_violation_std": float(np.std(violation)),
            }
        )
    return seed_level, summary


def _write_csv(path: Path, rows: Sequence[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metric table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_panel(
    axis,
    grouped: dict[float, list[dict[str, float | int]]],
    *,
    mean_column: str,
    std_column: str,
    ylabel: str,
    reference_values: Mapping[float, float] | None = None,
) -> None:
    """在一个无网格的正方形 panel 中绘制跨 seeds 均值和标准差。"""

    is_violation_panel = mean_column == "constraint_violation_mean"
    zero_violation_thresholds = {
        threshold
        for threshold, rows in grouped.items()
        if is_violation_panel
        and all(
            abs(float(row[mean_column])) <= 1e-12 and abs(float(row[std_column])) <= 1e-12
            for row in rows
        )
    }
    zero_violation_panel = is_violation_panel and len(zero_violation_thresholds) == len(grouped)
    colors = [THRESHOLD_COLORS[index % len(THRESHOLD_COLORS)] for index in range(len(grouped))]
    for color, (threshold, rows) in zip(colors, sorted(grouped.items()), strict=True):
        ordered = sorted(rows, key=lambda item: int(item["environment_steps"]))
        steps = np.asarray([int(row["environment_steps"]) for row in ordered])
        means = np.asarray([float(row[mean_column]) for row in ordered])
        standard_deviations = np.asarray([float(row[std_column]) for row in ordered])
        axis.plot(
            steps,
            means,
            color=color,
            linewidth=1.8,
            zorder=3,
            label=f"c={threshold:g}",
        )
        lower = means - standard_deviations
        if is_violation_panel:
            lower = np.maximum(lower, 0.0)
        axis.fill_between(
            steps,
            lower,
            means + standard_deviations,
            color=color,
            alpha=0.13,
            linewidth=0.0,
        )
        if reference_values is not None:
            reference_start = steps[0] + 0.78 * (steps[-1] - steps[0])
            axis.plot(
                (reference_start, steps[-1]),
                (reference_values[threshold], reference_values[threshold]),
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.85,
                zorder=2,
            )
    axis.set_xlabel("Execution time T", fontsize=PLOT_FONT_SIZE)
    axis.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
    axis.set_box_aspect(1)
    axis.grid(False)
    if zero_violation_panel:
        # 将零线与底部坐标轴分开，避免严格可行的结果看起来像没有绘制。
        axis.set_ylim(0.0, 1e-3)
        axis.spines["bottom"].set_position(("outward", 5))
        axis.text(
            0.5,
            0.5,
            "Constraint violation = 0 over all T",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=PLOT_FONT_SIZE,
        )
    elif is_violation_panel:
        upper = axis.get_ylim()[1]
        lower = -0.025 * upper if zero_violation_thresholds else 0.0
        axis.set_ylim(bottom=lower)


def _plot_reward_and_violation(
    summary: Sequence[dict[str, float | int]],
    output_path: Path,
    *,
    optimal_rewards: Mapping[float, float] | None = None,
) -> None:
    """左 reward、右 violation；两个 panel 均为正方形且不画网格。"""

    grouped: dict[float, list[dict[str, float | int]]] = defaultdict(list)
    for row in summary:
        grouped[float(row["constraint_threshold"])].append(row)

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 5.9))
    _plot_panel(
        axes[0],
        grouped,
        mean_column="long_run_average_reward_mean",
        std_column="long_run_average_reward_std",
        ylabel="Long-term average reward",
        reference_values=optimal_rewards,
    )
    _plot_panel(
        axes[1],
        grouped,
        mean_column="constraint_violation_mean",
        std_column="constraint_violation_std",
        ylabel="Constraint violation",
    )
    if len(grouped) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=4,
            fontsize=PLOT_FONT_SIZE,
            frameon=False,
            handlelength=2.8,
            columnspacing=2.0,
        )
        figure.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.31, wspace=0.17)
    else:
        figure.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.12, wspace=0.17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def generate_plots(
    input_path: Path,
    data_directory: Path,
    image_directory: Path,
) -> Path:
    """读取执行窗口日志并输出一张双 panel 图和两级统计表。"""

    window_rows = _read_window_rows(input_path)
    seed_level, summary = compute_long_run_metrics(window_rows)
    _write_csv(data_directory / "long_run_metrics_seed_level.csv", seed_level)
    _write_csv(data_directory / "long_run_metrics_summary.csv", summary)

    output_path = image_directory / "reward_constraint_violation_vs_t.png"
    _plot_reward_and_violation(summary, output_path)
    for obsolete_name in ("long_run_average_reward.png", "constraint_violation.png"):
        obsolete_path = image_directory / obsolete_name
        if obsolete_path.is_file():
            obsolete_path.unlink()
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    settings = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.data_directory / "dual_update_log.csv",
        help="无中心执行生成的 dual_update_log.csv",
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=settings.data_directory,
        help="处理后统计 CSV 的输出目录",
    )
    parser.add_argument(
        "--image-directory",
        type=Path,
        default=settings.image_directory,
        help="两张 PNG 的输出目录",
    )
    arguments = parser.parse_args(argv)
    try:
        output_path = generate_plots(
            arguments.input.expanduser().resolve(),
            arguments.data_directory.expanduser().resolve(),
            arguments.image_directory.expanduser().resolve(),
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"[DONE] reward/violation plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compute_long_run_metrics", "generate_plots", "main"]
