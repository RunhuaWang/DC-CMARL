"""IDE 直接运行：比较 H=64 下八个约束阈值的 reward 与约束违反。"""

from __future__ import annotations

import csv
from pathlib import Path

from hrmr.experiments.fixed_responsibility_small_6x6_decentralized_execution import (
    config,
    plot_long_run_metrics,
)

CONFIG_PATHS = (
    config.C0_H64_CONFIG_PATH,
    config.C0P1_H64_CONFIG_PATH,
    config.C0P2_H64_CONFIG_PATH,
    config.C0P4_H64_CONFIG_PATH,
    config.C0P6_H64_CONFIG_PATH,
    config.C0P9_H64_CONFIG_PATH,
    config.C1_H64_CONFIG_PATH,
    config.C1P2_H64_CONFIG_PATH,
)
EXPECTED_THRESHOLDS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.9, 1.0, 1.2)
OPTIMAL_REWARDS = {
    0.0: 4.0,
    0.1: 4.4,
    0.2: 4.8,
    0.4: 5.4,
    0.6: 6.0,
    0.9: 6.6,
    1.0: 6.8,
    1.2: 7.2,
}
DATA_DIRECTORY = (
    config.PROJECT_ROOT / "map_6x6/data/decentralized_execution/h64_c0_to_c1p2_comparison"
)
IMAGE_DIRECTORY = (
    config.PROJECT_ROOT / "map_6x6/images/decentralized_execution/h64_c0_to_c1p2_comparison"
)


def _write_summary(path: Path, rows: list[dict[str, float | int]]) -> None:
    """保存绘图实际使用的跨 seed 汇总数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """读取八组既有日志并生成一张双 panel 对比图。"""

    window_rows: list[dict[str, float | int]] = []
    observed_thresholds: list[float] = []
    for config_path in CONFIG_PATHS:
        settings = config.load_config(config_path)
        input_path = settings.data_directory / "dual_update_log.csv"
        if not input_path.is_file():
            raise SystemExit(f"缺少执行日志，请先运行对应 H=64 实验：\n{input_path}")
        rows = plot_long_run_metrics._read_window_rows(input_path)
        thresholds = sorted({float(row["constraint_threshold"]) for row in rows})
        if thresholds != list(settings.constraint_thresholds):
            raise SystemExit(f"日志中的阈值与配置不一致：{input_path}")
        observed_thresholds.extend(thresholds)
        window_rows.extend(rows)

    if tuple(observed_thresholds) != EXPECTED_THRESHOLDS:
        raise SystemExit(f"需要阈值 {EXPECTED_THRESHOLDS}，实际读取到 {tuple(observed_thresholds)}")

    _, summary = plot_long_run_metrics.compute_long_run_metrics(window_rows)
    summary_path = DATA_DIRECTORY / "long_run_metrics_summary.csv"
    output_path = IMAGE_DIRECTORY / "reward_constraint_violation_comparison.png"
    _write_summary(summary_path, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_long_run_metrics._plot_reward_and_violation(
        summary,
        output_path,
        optimal_rewards=OPTIMAL_REWARDS,
    )

    print(f"[DONE] comparison data: {summary_path}")
    print(f"[DONE] comparison plot: {output_path}")
    print(f"[DONE] comparison PDF: {output_path.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
