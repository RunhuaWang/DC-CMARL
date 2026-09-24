"""固定责任区 MAPPO 使用的原始异质参数与 raw reward/cost 环境。

本模块逐项复用当前正式 balanced 13x13 地图及 target value/hazard，且复用
正式状态转移。唯一信号差异是：team reward 对 distinct coverage 的 target
value 直接求和；每个 agent 的 local cost 等于所在位置的 hazard intensity，
不再除以团队规模。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.config import DEFAULT_CONFIG_PATH, HRMRConfig, load_config, validate_config
from hrmr.constants import Action
from hrmr.environment import HRMREnvironment
from hrmr.experiments.unnormalized_equal_hazard_ppo.signals import (
    aggregate_unnormalized_global_cost,
    compute_unnormalized_local_costs,
    compute_unnormalized_team_reward,
)
from hrmr.geometry import (
    agent_region_ids,
    compute_hazard_intensities,
    covered_target_ids,
)
from hrmr.transitions import TransitionResult, transition_positions

FloatArray = NDArray[np.float64]

UNNORMALIZED_ORIGINAL_ENVIRONMENT_CONFIG_PATH = DEFAULT_CONFIG_PATH


@lru_cache(maxsize=1)
def _formal_environment_config() -> HRMRConfig:
    """读取当前正式 balanced map，作为全部配置字段的唯一锚点。"""

    return load_config(DEFAULT_CONFIG_PATH)


def _environment_signature(config: HRMRConfig) -> tuple[Any, ...]:
    """返回新环境必须逐项继承的正式字段。"""

    return (
        config.grid_width,
        config.grid_height,
        config.num_agents,
        config.monitoring_radius,
        config.action_slip_probability,
        config.reward_normalizer,
        config.dual_upper_bound,
        config.use_agent_id,
        tuple(config.initial_positions),
        tuple(config.target_centers.items()),
        tuple(config.target_values.items()),
        tuple(config.hazard_intensities.items()),
    )


def validate_unnormalized_original_environment_config(config: HRMRConfig) -> None:
    """验证配置与正式 balanced environment 完全一致。"""

    validate_config(config)
    if _environment_signature(config) != _environment_signature(_formal_environment_config()):
        raise ValueError(
            "unnormalized original-parameter environment must exactly match "
            "the formal balanced configuration"
        )


def load_unnormalized_original_environment_config(
    path: str | Path | None = None,
) -> HRMRConfig:
    """加载并严格验证 balanced 13×13 environment 配置。"""

    source = UNNORMALIZED_ORIGINAL_ENVIRONMENT_CONFIG_PATH if path is None else Path(path)
    config = load_config(source)
    validate_unnormalized_original_environment_config(config)
    return config


class UnnormalizedOriginalParametersEnvironment(HRMREnvironment):
    """正式环境几何与参数不变、仅暴露 raw additive signals。"""

    signal_contract = "unnormalized_original_parameters_v1"

    def __init__(self, config: HRMRConfig | None = None) -> None:
        selected = load_unnormalized_original_environment_config() if config is None else config
        validate_unnormalized_original_environment_config(selected)
        super().__init__(config=selected)

    def _raw_local_costs(self) -> FloatArray:
        return compute_unnormalized_local_costs(
            self._positions,
            self._regions,
            self.config.hazard_intensities,
            self.config.target_ids,
        )

    def _snapshot_transition(self) -> TransitionResult:
        stay_actions = np.full(self.config.num_agents, int(Action.STAY), dtype=np.int64)
        return TransitionResult(
            requested_actions=stay_actions,
            executed_actions=stay_actions,
            bounded_candidate_positions=self._positions,
            final_positions=self._positions,
            boundary_block_flags=np.zeros(self.config.num_agents, dtype=np.bool_),
            collision_flags=np.zeros(self.config.num_agents, dtype=np.bool_),
        )

    def reset(self, seed: int | None = None) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
        """恢复共同正式初态；reset 本身不产生 reward transition。"""

        self._rng = np.random.default_rng(seed)
        self._positions = np.asarray(self.config.initial_positions, dtype=np.int64).copy()
        return self.current()

    def current(self) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
        """只读构造当前 observation/state/raw-cost info，不推进 RNG。"""

        coverage = self._coverage()
        local_costs = self._raw_local_costs()
        return (
            self._build_observations(coverage),
            self._build_state(coverage),
            self._build_info(coverage, local_costs, self._snapshot_transition()),
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
        """执行正式 transition，再按原参数计算未归一化 reward/cost。"""

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
        reward = compute_unnormalized_team_reward(
            self._positions,
            self._regions,
            self.config.target_values,
            self.config.target_ids,
        )
        local_costs = self._raw_local_costs()
        return (
            self._build_observations(coverage),
            self._build_state(coverage),
            reward,
            local_costs.copy(),
            False,
            False,
            self._build_info(coverage, local_costs, transition),
        )

    def _build_info(
        self,
        coverage: NDArray[np.int8],
        local_costs: FloatArray,
        transition: TransitionResult,
    ) -> dict[str, Any]:
        """构造与正式环境同键、但数值遵循 raw cost 语义的 info。"""

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
            "global_cost": aggregate_unnormalized_global_cost(local_costs),
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
            "signal_contract": self.signal_contract,
        }


__all__ = [
    "UNNORMALIZED_ORIGINAL_ENVIRONMENT_CONFIG_PATH",
    "UnnormalizedOriginalParametersEnvironment",
    "load_unnormalized_original_environment_config",
    "validate_unnormalized_original_environment_config",
]
