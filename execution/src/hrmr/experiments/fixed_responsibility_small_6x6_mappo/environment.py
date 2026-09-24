"""6×6 单格几何上的 owner-only raw reward 环境。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hrmr.config import HRMRConfig
from hrmr.environment import HRMREnvironment
from hrmr.experiments.assigned_targets_ppo.environment import (
    DEFAULT_ASSIGNMENTS,
    AssignedTargetsEnvironment,
)
from hrmr.experiments.assigned_targets_random_restart_ppo.environment import (
    evaluation_random_initial_positions,
)

from .config import CONFIG_PATHS, load_config, load_environment, validate_small_environment_config


class SmallSixBySixAssignedEnvironment(AssignedTargetsEnvironment):
    """仅替换几何；step/reward/cost/observation 均继承既有责任区环境。"""

    signal_contract = "assigned_targets_unnormalized_original_parameters_small_6x6_v1"

    def __init__(
        self,
        config: HRMRConfig | None = None,
        assignments: Sequence[Sequence[str]] = DEFAULT_ASSIGNMENTS,
    ) -> None:
        selected = load_environment(load_config(CONFIG_PATHS[0.0])) if config is None else config
        validate_small_environment_config(selected)
        # 父类构造器刻意锁定正式 13×13 地图；本隔离类直接复用基础环境构造，
        # 随后建立完全相同的 owner metadata，避免放宽正式环境的校验边界。
        HRMREnvironment.__init__(self, config=selected)
        if isinstance(assignments, (str, bytes)):
            raise TypeError("assignments must contain one target pair per agent")
        checked = tuple(tuple(pair) for pair in assignments)
        if len(checked) != selected.num_agents or any(len(pair) != 2 for pair in checked):
            raise ValueError("assignments must contain exactly two targets per agent")
        flat = tuple(target_id for pair in checked for target_id in pair)
        if any(not isinstance(target_id, str) for target_id in flat):
            raise TypeError("assignment target ids must be strings")
        if len(set(flat)) != len(flat) or set(flat) != set(selected.target_ids):
            raise ValueError("assignments must assign every target to exactly one owner")
        for pair in checked:
            if sum(selected.hazard_intensities[target_id] > 0.0 for target_id in pair) != 1:
                raise ValueError("each agent must own one safe and one hazardous target")
        self._assignments = checked
        owners = {target_id: agent for agent, pair in enumerate(checked) for target_id in pair}
        self._target_owner_indices = np.asarray(
            [owners[target_id] for target_id in selected.target_ids], dtype=np.int64
        )
        self._target_values = np.asarray(
            [selected.target_values[target_id] for target_id in selected.target_ids],
            dtype=np.float64,
        )

    def reset_positions(self, positions, *, seed=None):
        """训练重启使用任意合法、互异的四机器人联合位置。"""

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
        self._positions = checked.astype(np.int64, copy=True)
        self._rng = np.random.default_rng(seed)
        return self.current()


class SmallSixBySixRandomEvaluationEnvironment(SmallSixBySixAssignedEnvironment):
    """评价只在 t=0 随机一次，此后连续运行且不周期重启。"""

    def reset(self, seed=None):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("evaluation seed must be a non-negative integer")
        positions = evaluation_random_initial_positions(self.config, (seed,))[str(seed)]
        return self.reset_positions(positions, seed=seed)


__all__ = [
    "SmallSixBySixAssignedEnvironment",
    "SmallSixBySixRandomEvaluationEnvironment",
]
