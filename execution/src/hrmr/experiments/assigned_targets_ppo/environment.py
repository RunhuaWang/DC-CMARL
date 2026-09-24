"""固定责任目标实验：只有目标 owner 的覆盖产生原始团队奖励。

这是用户授权的问题定义变体，复用当前 balanced 13x13 布局（含 H3/H4
中心交换）与原始 target value/hazard。每个机器人固定负责一安全、一危险
目标；任何机器人进入任何危险区，仍承担真实的原始 hazard cost。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.config import HRMRConfig
from hrmr.experiments.unnormalized_original_parameters_ppo.environment import (
    UnnormalizedOriginalParametersEnvironment,
)
from hrmr.geometry import compute_target_coverage
from hrmr.rl.target_permutation import validate_target_permutation
from hrmr.transitions import TransitionResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_ASSIGNMENTS = (("S1", "H1"), ("S4", "H3"), ("S2", "H2"), ("S3", "H4"))
ASSIGNED_TARGET_FEATURE_SIZE = 6
ASSIGNED_OBSERVATION_DIM = 56
ASSIGNED_STATE_DIM = 16


class AssignedTargetsEnvironment(UnnormalizedOriginalParametersEnvironment):
    """固定 owner 的 raw-reward 环境，状态转移与真实 hazard 保持原样。"""

    signal_contract = "assigned_targets_unnormalized_original_parameters_v1"
    observation_dim = ASSIGNED_OBSERVATION_DIM
    state_dim = ASSIGNED_STATE_DIM

    def __init__(
        self,
        config: HRMRConfig | None = None,
        assignments: Sequence[Sequence[str]] = DEFAULT_ASSIGNMENTS,
    ) -> None:
        super().__init__(config=config)
        if isinstance(assignments, (str, bytes)):
            raise TypeError("assignments must contain one target pair per agent")
        checked = tuple(tuple(pair) for pair in assignments)
        if len(checked) != self.config.num_agents or any(len(pair) != 2 for pair in checked):
            raise ValueError("assignments must contain exactly two targets per agent")
        flat = tuple(target_id for pair in checked for target_id in pair)
        if any(not isinstance(target_id, str) for target_id in flat):
            raise TypeError("assignment target ids must be strings")
        if len(set(flat)) != len(flat) or set(flat) != set(self.config.target_ids):
            raise ValueError("assignments must assign every target to exactly one owner")
        for pair in checked:
            if sum(self.config.hazard_intensities[target_id] > 0.0 for target_id in pair) != 1:
                raise ValueError("each agent must own one safe and one hazardous target")
        self._assignments = checked
        owners = {target_id: agent for agent, pair in enumerate(checked) for target_id in pair}
        self._target_owner_indices = np.asarray(
            [owners[target_id] for target_id in self.config.target_ids], dtype=np.int64
        )
        self._target_values = np.asarray(
            [self.config.target_values[target_id] for target_id in self.config.target_ids],
            dtype=np.float64,
        )

    @property
    def assignments(self) -> tuple[tuple[str, ...], ...]:
        """按机器人顺序给出不可变的责任目标集合。"""

        return self._assignments

    @property
    def target_owner_indices(self) -> IntArray:
        """按 ``config.target_ids`` 顺序返回 owner index 的副本。"""

        return self._target_owner_indices.copy()

    def _coverage(self) -> NDArray[np.int8]:
        """只有 owner 位于监控区时，该目标才算有效覆盖。"""

        return np.asarray(
            [
                tuple(self._positions[owner]) in self._regions[target_id]
                for target_id, owner in zip(
                    self.config.target_ids, self._target_owner_indices, strict=True
                )
            ],
            dtype=np.int8,
        )

    def _build_observations(self, coverage: ArrayLike) -> FloatArray:
        """为原有五字段 target block 追加属于当前机器人的标记。"""

        original = super()._build_observations(coverage)
        blocks = np.empty((self.config.num_agents, 8, 6), dtype=np.float64)
        blocks[..., :5] = original[:, 2:42].reshape(self.config.num_agents, 8, 5)
        blocks[..., 5] = (
            np.arange(self.config.num_agents)[:, None] == self._target_owner_indices[None, :]
        )
        return np.concatenate(
            (original[:, :2], blocks.reshape(self.config.num_agents, 48), original[:, 42:]),
            axis=-1,
        )

    def _build_info(
        self,
        coverage: NDArray[np.int8],
        local_costs: FloatArray,
        transition: TransitionResult,
    ) -> dict[str, Any]:
        """保留真实位置/成本，分别暴露有效覆盖与物理覆盖。"""

        info = super()._build_info(coverage, local_costs, transition)
        physical = compute_target_coverage(self._positions, self._regions, self.config.target_ids)
        info["physical_target_coverage"] = physical.copy()
        info["physical_covered_target_ids"] = list(info["covered_target_ids"])
        info["covered_target_ids"] = [
            target_id
            for target_id, covered in zip(self.config.target_ids, coverage, strict=True)
            if covered
        ]
        info["target_owner_indices"] = self.target_owner_indices
        info["assignments"] = [list(pair) for pair in self._assignments]
        return info

    def step(
        self,
        joint_actions: ArrayLike,
    ) -> tuple[FloatArray, FloatArray, float, FloatArray, bool, bool, dict[str, Any]]:
        """复用原状态转移与 hazard 成本，再返回 owner 覆盖的 value 和。"""

        observations, state, _, local_costs, terminated, truncated, info = super().step(
            joint_actions
        )
        reward = float(np.dot(info["target_coverage"], self._target_values))
        return observations, state, reward, local_costs, terminated, truncated, info


def permute_assigned_target_blocks(
    observations: ArrayLike,
    permutations: ArrayLike,
) -> FloatArray:
    """排列完整六字段 target blocks，返回副本且不移动 self/teammates。

    支持 ``[4,56]`` 配 ``[8]``，以及 ``[E,4,56]`` 配 ``[E,8]``。
    批量 observation 也可使用一个共享的 ``[8]`` 排列。
    """

    raw = np.asarray(observations)
    if raw.ndim not in {2, 3} or raw.shape[-2:] != (4, ASSIGNED_OBSERVATION_DIM):
        raise ValueError("observations must have shape [4,56] or [num_envs,4,56]")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("observations must contain real numeric values")
    result = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("observations must be finite")
    supplied = np.asarray(permutations)
    blocks = result[..., 2:50].reshape(*result.shape[:-1], 8, 6)
    if supplied.ndim == 1:
        checked = validate_target_permutation(supplied, num_targets=8)
        permuted = blocks[..., checked, :]
    elif result.ndim == 3 and supplied.shape == (result.shape[0], 8):
        checked = np.asarray(
            [validate_target_permutation(item, num_targets=8) for item in supplied],
            dtype=np.int64,
        ).reshape(result.shape[0], 8)
        indices = np.broadcast_to(checked[:, None, :, None], blocks.shape)
        permuted = np.take_along_axis(blocks, indices, axis=2)
    else:
        raise ValueError("permutations must have shape [8], or [num_envs,8] for batched input")
    result[..., 2:50] = permuted.reshape(*result.shape[:-1], 48)
    return result


__all__ = [
    "ASSIGNED_OBSERVATION_DIM",
    "ASSIGNED_STATE_DIM",
    "ASSIGNED_TARGET_FEATURE_SIZE",
    "DEFAULT_ASSIGNMENTS",
    "AssignedTargetsEnvironment",
    "permute_assigned_target_blocks",
]
