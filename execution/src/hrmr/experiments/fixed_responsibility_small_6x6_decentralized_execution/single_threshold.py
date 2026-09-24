"""单阈值无中心执行入口的共享调度逻辑。"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from .cli import main as run_execution
from .config import load_config
from .plot_long_run_metrics import generate_plots


def run_single_threshold(
    config_path: str | Path,
    argv: Sequence[str] | None = None,
) -> int:
    """执行一个阈值；成功后立即生成 reward/violation 双子图。"""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    exit_code = run_execution(arguments, default_config=Path(config_path))
    if exit_code != 0 or "--check-only" in arguments:
        return exit_code
    settings = load_config(config_path)
    output_path = generate_plots(
        settings.data_directory / "dual_update_log.csv",
        settings.data_directory,
        settings.image_directory,
    )
    print(f"[DONE] reward/constraint violation vs T: {output_path}")
    return 0


__all__ = ["run_single_threshold"]
