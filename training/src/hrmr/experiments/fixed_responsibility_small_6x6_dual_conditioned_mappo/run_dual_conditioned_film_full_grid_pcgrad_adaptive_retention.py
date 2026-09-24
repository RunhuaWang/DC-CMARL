"""IDE 直接运行：21点 FiLM Actor + PCGrad + per-lambda adaptive retention。"""

from __future__ import annotations

from collections.abc import Sequence

from hrmr.experiments.fixed_responsibility_small_6x6_dual_conditioned_mappo.cli import (
    main as run_experiment,
)
from hrmr.experiments.fixed_responsibility_small_6x6_dual_conditioned_mappo.config import (
    PCGRAD_ADAPTIVE_RETENTION_FULL_GRID_CONFIG_PATH,
)


def main(argv: Sequence[str] | None = None) -> int:
    return run_experiment(argv, default_config=PCGRAD_ADAPTIVE_RETENTION_FULL_GRID_CONFIG_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
