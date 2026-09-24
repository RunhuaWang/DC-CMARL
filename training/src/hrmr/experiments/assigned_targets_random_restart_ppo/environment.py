"""训练采样的显式初态覆盖；不改变责任区环境的任何一步数学定义。"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.config import HRMRConfig
from hrmr.experiments.assigned_targets_ppo.environment import AssignedTargetsEnvironment

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

# 独立于训练位置、动作和 target-order 随机流；同一 seed 的评价起点不变。
EVAL_RANDOM_DOMAIN = 0x4556414C525354


def sample_joint_positions(config: HRMRConfig, rng: np.random.Generator) -> IntArray:
    """从全部网格均匀无放回抽取带身份的联合初态，不筛选目标类型。"""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    indices = rng.choice(config.grid_height * config.grid_width, config.num_agents, replace=False)
    return np.column_stack((indices // config.grid_width, indices % config.grid_width)).astype(
        np.int64
    )


def _evaluation_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("evaluation seed must be a non-negative integer")
    return int(seed)


def evaluation_random_initial_positions(
    config: HRMRConfig, seeds: Sequence[int]
) -> dict[str, list[list[int]]]:
    """预生成与训练无关的固定评价初态，可直接写入 JSON 供后续公平比较。"""

    seed_values = tuple(_evaluation_seed(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("evaluation seeds must be nonempty and unique")
    return {
        str(seed): sample_joint_positions(
            config, np.random.default_rng(np.random.SeedSequence([EVAL_RANDOM_DOMAIN, seed]))
        ).tolist()
        for seed in seed_values
    }


class RandomStartAssignedEnvironment(AssignedTargetsEnvironment):
    """只添加显式初态接口，默认 reset 仍保留原固定初态行为。"""

    def reset_positions(
        self, positions: ArrayLike, *, seed: int | None = None
    ) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
        """重建独立环境随机流并返回初态快照；不产生 reward 或 transition。"""

        checked = np.asarray(positions)
        if checked.shape != (self.config.num_agents, 2):
            raise ValueError("positions must have shape [num_agents, 2]")
        if checked.dtype.kind not in {"i", "u"}:
            raise TypeError("positions must contain integer row/column coordinates")
        if (
            np.any(checked < 0)
            or np.any(checked[:, 0] >= self.config.grid_height)
            or np.any(checked[:, 1] >= self.config.grid_width)
        ):
            raise ValueError("positions must be within the grid")
        if len(np.unique(checked, axis=0)) != self.config.num_agents:
            raise ValueError("positions must be distinct")
        rng = np.random.default_rng(seed)
        self._positions = checked.astype(np.int64, copy=True)
        self._rng = rng
        return self.current()


class RandomInitialEvaluationEnvironment(RandomStartAssignedEnvironment):
    """仅在评价开始时随机初始化；step 继承原实现，永不周期重启。"""

    def reset(self, seed: int | None = None) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
        """每个显式 seed 对应一个固定、与 target order 无关的评价起点。"""

        checked_seed = _evaluation_seed(seed)
        positions = evaluation_random_initial_positions(self.config, (checked_seed,))[
            str(checked_seed)
        ]
        return self.reset_positions(positions, seed=checked_seed)


__all__ = [
    "EVAL_RANDOM_DOMAIN",
    "RandomInitialEvaluationEnvironment",
    "RandomStartAssignedEnvironment",
    "evaluation_random_initial_positions",
    "sample_joint_positions",
]
