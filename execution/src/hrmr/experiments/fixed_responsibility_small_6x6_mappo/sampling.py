"""6×6 环境的同步随机重启 vector wrapper。"""

from __future__ import annotations

import numpy as np

from hrmr.experiments.assigned_targets_random_restart_ppo.environment import (
    sample_joint_positions,
)
from hrmr.rl.vector_env import SyncVectorHRMR

from .environment import SmallSixBySixAssignedEnvironment


class SmallSixBySixRandomRestartVectorEnv(SyncVectorHRMR):
    """保持原采样/重启协议，只将子环境替换为 6×6 责任区环境。"""

    def __init__(self, num_envs, config):
        super().__init__(num_envs=num_envs, config=config, seeds=0)
        self._envs = tuple(
            SmallSixBySixAssignedEnvironment(config=config) for _ in range(self._num_envs)
        )

    @property
    def observation_dim(self):
        return 56

    def current(self):
        results = [environment.current() for environment in self._envs]
        return (
            np.stack([result[0] for result in results]).astype(np.float64, copy=False),
            np.stack([result[1] for result in results]).astype(np.float64, copy=False),
            [result[2] for result in results],
        )

    def reset_random(self, rng, *, seed_start=0):
        seeds = tuple(seed_start + index for index in range(self.num_envs))
        for environment, seed in zip(self.envs, seeds, strict=True):
            environment.reset_positions(sample_joint_positions(environment.config, rng), seed=seed)
        self._last_seeds = seeds
        self._has_reset = True
        return self.current()


__all__ = ["SmallSixBySixRandomRestartVectorEnv"]
