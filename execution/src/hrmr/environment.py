"""HRMR Phase 1 continuing-task 环境接口。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.config import HRMRConfig, load_config, validate_config
from hrmr.constants import Action
from hrmr.geometry import (
    agent_region_ids,
    compute_hazard_intensities,
    compute_target_coverage,
    covered_target_ids,
)
from hrmr.observations import (
    append_normalized_dual,
    build_actor_observations,
    build_centralized_critic_state,
    build_environment_state,
)
from hrmr.rewards_costs import (
    aggregate_global_cost,
    compute_local_costs,
    compute_team_reward,
)
from hrmr.transitions import TransitionResult, transition_positions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class HRMREnvironment:
    """不依赖 MARL 框架的 HRMR 基础静态环境。

    环境持有自己的 :class:`numpy.random.Generator`。它不接收 constraint
    threshold 或 dual variable；dual 只可通过独立的输入构造 helper 追加，
    因而不会改变状态转移、reward 或 cost。
    """

    def __init__(self, config: HRMRConfig | None = None) -> None:
        self.config = load_config() if config is None else config
        validate_config(self.config)
        self._regions = self.config.monitoring_regions
        self._rng = np.random.default_rng()
        self._positions = np.asarray(self.config.initial_positions, dtype=np.int64)

    @property
    def positions(self) -> IntArray:
        """返回当前联合位置的副本，避免外部静默改写环境状态。"""

        return self._positions.copy()

    @property
    def monitoring_regions(self) -> dict[str, frozenset[tuple[int, int]]]:
        """返回监控区域的浅副本；每个区域本身是不可变集合。"""

        return dict(self._regions)

    def reset(self, seed: int | None = None) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
        """恢复 PDF 式 (3.6) 初始位置并重建环境随机流。"""

        self._rng = np.random.default_rng(seed)
        self._positions = np.asarray(self.config.initial_positions, dtype=np.int64).copy()
        coverage = self._coverage()
        local_costs = compute_local_costs(
            self._positions,
            self._regions,
            self.config.hazard_intensities,
            self.config.num_agents,
            self.config.target_ids,
        )
        stay_actions = np.full(self.config.num_agents, int(Action.STAY), dtype=np.int64)
        transition = TransitionResult(
            requested_actions=stay_actions,
            executed_actions=stay_actions,
            bounded_candidate_positions=self._positions,
            final_positions=self._positions,
            boundary_block_flags=np.zeros(self.config.num_agents, dtype=np.bool_),
            collision_flags=np.zeros(self.config.num_agents, dtype=np.bool_),
        )
        return (
            self._build_observations(coverage),
            self._build_state(coverage),
            self._build_info(coverage, local_costs, transition),
        )

    def step(
        self,
        joint_actions: ArrayLike,
    ) -> tuple[
        FloatArray,
        FloatArray,
        float,
        FloatArray,
        bool,
        bool,
        dict[str, Any],
    ]:
        """严格按 PDF §10.1 顺序执行一个 continuing-task 环境步。"""

        transition = transition_positions(
            self._positions,
            joint_actions,
            grid_width=self.config.grid_width,
            grid_height=self.config.grid_height,
            slip_probability=self.config.action_slip_probability,
            rng=self._rng,
        )
        self._positions = transition.final_positions.copy()

        coverage = self._coverage()
        team_reward = compute_team_reward(
            self._positions,
            self._regions,
            self.config.target_values,
            self.config.reward_normalizer,
            self.config.target_ids,
        )
        local_costs = compute_local_costs(
            self._positions,
            self._regions,
            self.config.hazard_intensities,
            self.config.num_agents,
            self.config.target_ids,
        )
        observations = self._build_observations(coverage)
        state = self._build_state(coverage)
        info = self._build_info(coverage, local_costs, transition)

        # HRMR 基础任务没有自然终止或内部 horizon。
        terminated = False
        truncated = False
        return (
            observations,
            state,
            float(team_reward),
            local_costs.copy(),
            terminated,
            truncated,
            info,
        )

    def actor_inputs(self, dual_lambda: float) -> FloatArray:
        """基于当前 48/52 维 observation 构造 dual-conditioned actor input。"""

        coverage = self._coverage()
        observations = self._build_observations(coverage)
        return append_normalized_dual(
            observations,
            dual_lambda,
            self.config.dual_upper_bound,
        )

    def centralized_critic_state(self, dual_lambda: float) -> FloatArray:
        """基于当前环境状态构造 PDF §9.5 的 17 维 critic input。"""

        return build_centralized_critic_state(
            self._positions,
            self._coverage(),
            self.config.grid_width,
            self.config.grid_height,
            dual_lambda,
            self.config.dual_upper_bound,
        )

    def _coverage(self) -> NDArray[np.int8]:
        return compute_target_coverage(
            self._positions,
            self._regions,
            self.config.target_ids,
        )

    def _build_observations(self, coverage: ArrayLike) -> FloatArray:
        return build_actor_observations(
            self._positions,
            self.config.target_centers,
            self.config.target_values,
            self.config.hazard_intensities,
            coverage,
            self.config.grid_width,
            self.config.grid_height,
            self.config.use_agent_id,
            self.config.target_ids,
        )

    def _build_state(self, coverage: ArrayLike) -> FloatArray:
        # 环境 state 不携带算法 dual；独立 helper 可进一步构造 17 维 critic input。
        return build_environment_state(
            self._positions,
            coverage,
            self.config.grid_width,
            self.config.grid_height,
        )

    def _build_info(
        self,
        coverage: NDArray[np.int8],
        local_costs: FloatArray,
        transition: TransitionResult,
    ) -> dict[str, Any]:
        hazards = compute_hazard_intensities(
            self._positions,
            self._regions,
            self.config.hazard_intensities,
            self.config.target_ids,
        )
        region_ids = agent_region_ids(
            self._positions,
            self._regions,
            self.config.target_ids,
        )
        covered_ids = covered_target_ids(
            self._positions,
            self._regions,
            self.config.target_ids,
        )
        positions = self._positions.copy()
        target_coverage = np.asarray(coverage, dtype=np.int8).copy()
        return {
            "global_cost": float(aggregate_global_cost(local_costs)),
            "local_costs": local_costs.copy(),
            "target_coverage": target_coverage,
            "target_occupancy": target_coverage.copy(),
            "covered_target_ids": list(covered_ids),
            "agent_positions": positions,
            "robot_positions": positions.copy(),
            "agent_region_ids": list(region_ids),
            "hazard_intensity_per_robot": hazards.copy(),
            "requested_actions": transition.requested_actions.copy(),
            "executed_actions": transition.executed_actions.copy(),
            "collision_flags": transition.collision_flags.copy(),
            "boundary_block_flags": transition.boundary_block_flags.copy(),
        }


__all__ = ["HRMREnvironment"]
