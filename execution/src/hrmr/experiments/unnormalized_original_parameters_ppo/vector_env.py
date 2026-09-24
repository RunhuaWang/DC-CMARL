"""原始异质参数、未归一化信号环境的同步 vector wrapper。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hrmr.config import HRMRConfig
from hrmr.rl.vector_env import SyncVectorHRMR, VectorCurrent

from .environment import (
    UnnormalizedOriginalParametersEnvironment,
    load_unnormalized_original_environment_config,
    validate_unnormalized_original_environment_config,
)

SeedValue = int | None
SeedInput = int | Sequence[SeedValue] | None


class UnnormalizedOriginalParametersVectorEnv(SyncVectorHRMR):
    """保持正式 vector API，但所有子环境均暴露原参数 raw signals。"""

    def __init__(
        self,
        num_envs: int,
        config: HRMRConfig | None = None,
        seeds: SeedInput = None,
    ) -> None:
        selected = load_unnormalized_original_environment_config() if config is None else config
        validate_unnormalized_original_environment_config(selected)
        super().__init__(num_envs=num_envs, config=selected, seeds=seeds)
        self._envs = tuple(
            UnnormalizedOriginalParametersEnvironment(config=selected)
            for _ in range(self._num_envs)
        )

    @property
    def envs(self) -> tuple[UnnormalizedOriginalParametersEnvironment, ...]:
        return self._envs

    def current(self) -> VectorCurrent:
        """从 raw 环境构造快照，避免正式 wrapper 的 ``h/N`` 路径。"""

        results = [environment.current() for environment in self._envs]
        observations = np.stack([result[0] for result in results]).astype(
            np.float64,
            copy=False,
        )
        states = np.stack([result[1] for result in results]).astype(
            np.float64,
            copy=False,
        )
        return observations, states, [result[2] for result in results]


__all__ = ["UnnormalizedOriginalParametersVectorEnv"]
