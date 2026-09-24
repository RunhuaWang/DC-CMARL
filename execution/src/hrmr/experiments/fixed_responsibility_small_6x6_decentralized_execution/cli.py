"""6×6 固定策略无中心执行的命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import CONFIG_PATH, load_config


def main(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path = CONFIG_PATH,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda", "mps"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        settings = load_config(arguments.config)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    if arguments.check_only:
        print(f"Configuration OK: {settings.profile}")
        print(f"Checkpoint: {settings.checkpoint_path}")
        print(f"Thresholds: {settings.constraint_thresholds}")
        print(
            f"Dual: lambda0={settings.initial_dual:g}, eta={settings.dual_step_size:g}, "
            f"range=[{settings.dual_min:g},{settings.dual_max:g}]"
        )
        print(
            f"Execution: H={settings.cost_estimation_horizon}, "
            f"T={settings.total_environment_steps}, "
            f"updates={settings.num_dual_updates}, seeds={settings.evaluation_seeds}"
        )
        print(f"Communication graph={settings.graph}; rounds={settings.communication_rounds}")
        print("Frozen actors; stochastic actions; no critics; no parameter updates; no resets")
        print(f"Data output: {settings.data_directory}")
        print(f"Image output: {settings.image_directory}")
        return 0

    from .execution import execute

    try:
        execute(
            arguments.config,
            device_override=arguments.device,
            overwrite_existing=not arguments.no_overwrite,
        )
    except FileExistsError as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("[STOPPED] Decentralized execution interrupted.", flush=True)
        return 130
    return 0


__all__ = ["main"]
