"""独立 fixed-lambda average-reward MAPPO-style 训练器。"""

from __future__ import annotations

import csv
import json
import math
import platform
import shutil
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from hrmr.config import HRMRConfig, load_config
from hrmr.constants import Action
from hrmr.environment import HRMREnvironment
from hrmr.rendering import render_trajectory
from hrmr.rl.analytic_modes import (
    DIAGNOSTIC_FIXED_LAMBDAS,
    TRAINABLE_FIXED_LAMBDAS,
    analytic_mode,
    assess_operating_mode,
)
from hrmr.rl.average_rate import AverageRateEstimator
from hrmr.rl.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from hrmr.rl.device import resolve_device, seed_everything
from hrmr.rl.differential_td import (
    combine_reward_cost_advantages,
    differential_n_step_target,
)
from hrmr.rl.evaluation import (
    FixedLambdaEvaluation,
    evaluate_fixed_lambda,
)
from hrmr.rl.formal_evaluation import evaluate_formal_actor
from hrmr.rl.logging import (
    FORMAL_TRAINING_DIAGNOSTIC_COLUMNS,
    OCCUPANCY_COLUMNS,
    TrainingCSVLogger,
    write_evaluation_csv,
)
from hrmr.rl.networks import DifferentialCritic, IndependentActors
from hrmr.rl.ppo_losses import compute_ppo_policy_loss, differential_value_loss
from hrmr.rl.rollout_buffer import RolloutBatch, RolloutBuffer
from hrmr.rl.safe_actor_view import (
    SAFE_ACTOR_NUM_TARGETS,
    SAFE_ACTOR_OBSERVATION_DIM,
    SafeTargetActorView,
    project_safe_target_actor_observations,
)
from hrmr.rl.target_permutation import (
    permute_target_blocks_by_environment,
    sample_target_permutations,
)
from hrmr.rl.training_config import (
    FixedLambdaConfig,
    load_training_config,
    resolve_environment_config,
)
from hrmr.rl.training_milestones import (
    AllHazardousTrainingTracker,
    OperatingModeTrainingTracker,
)
from hrmr.rl.vector_env import SyncVectorHRMR

_TARGET_PERMUTATION_RNG_STREAM = 0x54415247
_FORMAL_N_STEP_HORIZONS = (8, 16)
SAFE_ACTOR_VIEW_DIAGNOSTIC_PROFILE = "diagnostic_lambda4_safe_actor_view_seed0"
SAFE_ACTOR_VIEW_SMOKE_PROFILE = "smoke_lambda4_safe_actor_view_seed0"
STATE_COVERAGE_UPTAKE_DIAGNOSTIC_PROFILE = "diagnostic_lambda4_state_coverage_uptake_seed0"
STATE_COVERAGE_UPTAKE_SMOKE_PROFILE = "smoke_lambda4_state_coverage_uptake_seed0"
STATE_COVERAGE_UPTAKE_ENVIRONMENT_STEPS = 100_000
STATE_COVERAGE_UPTAKE_SMOKE_ENVIRONMENT_STEPS = 8_192
_SAFE_ACTOR_VIEW_PROFILES = frozenset(
    (SAFE_ACTOR_VIEW_DIAGNOSTIC_PROFILE, SAFE_ACTOR_VIEW_SMOKE_PROFILE)
)
_STATE_COVERAGE_UPTAKE_PROFILES = frozenset(
    (
        STATE_COVERAGE_UPTAKE_DIAGNOSTIC_PROFILE,
        STATE_COVERAGE_UPTAKE_SMOKE_PROFILE,
    )
)
_DIAGNOSTIC_LAMBDA_PROFILES = _SAFE_ACTOR_VIEW_PROFILES | _STATE_COVERAGE_UPTAKE_PROFILES
_HAZARDOUS_TARGET_INDICES = tuple(
    index for index, column in enumerate(OCCUPANCY_COLUMNS) if column.startswith("occupancy_H")
)


class IndependentActorOptimizerGroup:
    """四个 actor 各自持有的 Adam optimizer 集合。"""

    def __init__(self, actor: IndependentActors, learning_rate: float) -> None:
        self.optimizers = tuple(
            torch.optim.Adam(policy.parameters(), lr=learning_rate) for policy in actor.actors
        )

    def optimizer(self, agent_index: int) -> torch.optim.Adam:
        """返回只拥有 ``actor_i`` 参数的 optimizer。"""

        return self.optimizers[agent_index]

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict.get("format_version", -1)) != 1:
            raise ValueError("unsupported independent actor optimizer format")
        states = state_dict.get("optimizers")
        if not isinstance(states, list) or len(states) != len(self.optimizers):
            raise ValueError("independent actor optimizer state count does not match actors")
        for optimizer, state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(state)


@dataclass(frozen=True)
class UpdateStatistics:
    """一次 frozen-rollout PPO update 的聚合诊断。"""

    actor_loss: float
    reward_critic_loss: float
    cost_critic_loss: float
    value_anchor_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    actor_grad_norm: float
    reward_critic_grad_norm: float
    cost_critic_grad_norm: float
    reward_differential_value_mean: float
    reward_differential_value_std: float
    reward_differential_value_abs_max: float
    actor_loss_by_agent: tuple[float, ...]
    actor_entropy_by_agent: tuple[float, ...]


@dataclass(frozen=True)
class TrainingResult:
    """一个独立 fixed-lambda run 的最终路径与正式评价摘要。"""

    fixed_lambda: float
    training_seed: int
    profile: str
    device: str
    environment_steps: int
    agent_action_steps: int
    updates: int
    n_step: int
    duration_seconds: float
    stopped_early: bool
    run_directory: Path
    best_checkpoint: Path
    last_checkpoint: Path
    training_log: Path
    evaluation_json: Path
    evaluation_csv: Path
    steady_reward: float
    steady_cost: float
    steady_scalarized_objective: float
    scalarized_optimality_gap: float
    target_occupancy: tuple[float, ...]
    mode_success: bool

    def to_dict(self) -> dict[str, Any]:
        """转成适合 CLI JSON 输出和 sweep 汇总的字典。"""

        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


class FixedLambdaTrainingDiagnosticHook(Protocol):
    """不改变正式 PPO 路径的 rollout-boundary 诊断旁路接口。"""

    def before_rollout(
        self,
        *,
        vector_env: SyncVectorHRMR,
        actor: IndependentActors,
        environment_steps: int,
        update_index: int,
        device: torch.device,
    ) -> ArrayLike:
        """可在边界注入状态，并返回需切断 tracker 连续性的 ``[E]`` mask。"""

        ...

    def after_frozen_quantities(
        self,
        *,
        batch: RolloutBatch,
        actor: IndependentActors,
        reward_critic: DifferentialCritic,
        cost_critic: DifferentialCritic,
        reward_targets: torch.Tensor,
        cost_targets: torch.Tensor,
        combined_advantages: torch.Tensor,
        reward_rate: float,
        cost_rate: float,
        fixed_lambda: float,
        rollout_start_environment_steps: int,
        rollout_end_environment_steps: int,
        update_index: int,
        target_block_permutations: np.ndarray | None,
        device: torch.device,
    ) -> None:
        """在 quantities 冻结后、任何模型更新前记录旁路诊断。"""

        ...

    def after_update(
        self,
        *,
        batch: RolloutBatch,
        actor: IndependentActors,
        statistics: UpdateStatistics,
        fixed_lambda: float,
        environment_steps: int,
        update_index: int,
        device: torch.device,
    ) -> None:
        """在一次 PPO/critic update 后记录同一批数据的旁路诊断。"""

        ...

    def state_dict(self) -> Mapping[str, Any]:
        """返回可写入 checkpoint extra 的独立诊断状态。"""

        ...


def run_name(profile: str, fixed_lambda: float, training_seed: int) -> str:
    """生成稳定且跨平台安全的 run 目录名。"""

    lambda_token = f"{fixed_lambda:.2f}".replace(".", "p")
    return f"{profile}_lambda_{lambda_token}_seed_{training_seed}"


def _validate_run_inputs(
    fixed_lambda: float,
    training_seed: int,
    *,
    profile: str | None = None,
) -> tuple[float, int]:
    lambda_value = float(fixed_lambda)
    diagnostic_profile = profile in _DIAGNOSTIC_LAMBDA_PROFILES
    allowed_lambdas = DIAGNOSTIC_FIXED_LAMBDAS if diagnostic_profile else TRAINABLE_FIXED_LAMBDAS
    if lambda_value not in allowed_lambdas:
        qualifier = "large-dual diagnostic" if diagnostic_profile else "formal"
        raise ValueError(f"{qualifier} fixed_lambda must be one of {allowed_lambdas}")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise TypeError("training_seed must be an integer")
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    if diagnostic_profile and training_seed != 0:
        raise ValueError("large-dual diagnostics require training_seed=0")
    return lambda_value, training_seed


def _validate_n_step(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("n_step must be a positive integer")
    return value


def _validate_actor_configuration(
    config: FixedLambdaConfig,
    environment_config: HRMRConfig,
) -> None:
    """验证正式独立 actor 架构；正式 profile 额外锁定运行预算。"""

    if not config.experiment.independent_actors:
        raise ValueError("formal fixed-lambda training requires independent actors")
    if environment_config.use_agent_id:
        raise ValueError("independent actors must not receive one-hot agent ID")
    if not config.experiment.randomize_actor_target_order:
        raise ValueError("independent actors require random target-order training")
    if config.training.n_step not in _FORMAL_N_STEP_HORIZONS:
        raise ValueError(
            f"formal fixed-lambda training requires n_step in {_FORMAL_N_STEP_HORIZONS}"
        )
    if config.training.entropy_coefficient_start != 0.02:
        raise ValueError("formal fixed-lambda training requires entropy coefficient 0.02")
    if config.training.entropy_coefficient_end != 0.02:
        raise ValueError("formal fixed-lambda entropy coefficient must remain fixed")
    if config.training.early_stopping:
        raise ValueError("formal fixed-lambda training requires early_stopping=false")
    if config.environment.action_slip_probability != 0.0:
        raise ValueError("formal fixed-lambda training requires action_slip_probability=0")
    if environment_config.num_agents != 4 or environment_config.num_targets != 8:
        raise ValueError("formal fixed-lambda training requires N=4 and M=8")
    if config.experiment.profile in _SAFE_ACTOR_VIEW_PROFILES:
        if config.experiment.actor_target_view != "safe_only":
            raise ValueError("safe-actor-view diagnostic requires actor_target_view='safe_only'")
    elif config.experiment.actor_target_view != "all":
        raise ValueError("formal/state-coverage runs require actor_target_view='all'")
    if (
        config.experiment.profile == "formal_independent_fixed_lambda"
        and config.training.total_environment_steps != 500_000
    ):
        raise ValueError("formal fixed-lambda training requires 500000 environment steps")
    if config.experiment.profile == SAFE_ACTOR_VIEW_DIAGNOSTIC_PROFILE:
        if config.training.n_step != 16:
            raise ValueError("safe-actor-view diagnostic training requires n_step=16")
        if config.training.total_environment_steps != 500_000:
            raise ValueError(
                "safe-actor-view diagnostic training requires 500000 environment steps"
            )
    if config.experiment.profile == SAFE_ACTOR_VIEW_SMOKE_PROFILE:
        if config.training.n_step != 16:
            raise ValueError("safe-actor-view smoke requires n_step=16")
        if config.training.total_environment_steps != 9_984:
            raise ValueError("safe-actor-view smoke requires 9984 environment steps")
    if config.experiment.profile == STATE_COVERAGE_UPTAKE_DIAGNOSTIC_PROFILE:
        if config.training.num_parallel_envs != 32 or config.training.rollout_length != 128:
            raise ValueError("state-coverage uptake diagnostic requires 32 envs and rollout 128")
        if config.training.n_step != 16:
            raise ValueError("state-coverage uptake diagnostic requires n_step=16")
        if config.training.total_environment_steps != STATE_COVERAGE_UPTAKE_ENVIRONMENT_STEPS:
            raise ValueError("state-coverage uptake diagnostic requires 100000 environment steps")
    if config.experiment.profile == STATE_COVERAGE_UPTAKE_SMOKE_PROFILE:
        if config.training.num_parallel_envs != 32 or config.training.rollout_length != 128:
            raise ValueError("state-coverage uptake smoke requires 32 envs and rollout 128")
        if config.training.n_step != 16:
            raise ValueError("state-coverage uptake smoke requires n_step=16")
        if config.training.total_environment_steps != STATE_COVERAGE_UPTAKE_SMOKE_ENVIRONMENT_STEPS:
            raise ValueError("state-coverage uptake smoke requires 8192 environment steps")


def _training_environment_config(config: FixedLambdaConfig) -> HRMRConfig:
    base = load_config(resolve_environment_config(config))
    return replace(
        base,
        action_slip_probability=config.environment.action_slip_probability,
    )


def _network_kwargs(config: FixedLambdaConfig) -> dict[str, str]:
    """只向支持 activation 配置的当前网络版本传参。"""

    return {"activation": config.experiment.activation}


def _make_networks(
    config: FixedLambdaConfig,
    device: torch.device,
    training_seed: int = 0,
) -> tuple[IndependentActors, DifferentialCritic, DifferentialCritic]:
    kwargs = _network_kwargs(config)
    environment_config = _training_environment_config(config)
    if not config.experiment.independent_actors or environment_config.use_agent_id:
        raise ValueError("formal networks require independent actors without Agent ID")
    seed_base = config.experiment.independent_actor_init_seed_base
    actor_seeds = tuple(
        seed_base + training_seed * environment_config.num_agents + agent_index
        for agent_index in range(environment_config.num_agents)
    )
    actor = IndependentActors(
        num_agents=environment_config.num_agents,
        input_dim=(
            SAFE_ACTOR_OBSERVATION_DIM if config.experiment.actor_target_view == "safe_only" else 48
        ),
        hidden_dims=(128, 128),
        num_actions=5,
        initialization_seeds=actor_seeds,
        **kwargs,
    )
    reward_critic = DifferentialCritic(input_dim=16, hidden_dims=(256, 256), **kwargs)
    cost_critic = DifferentialCritic(input_dim=16, hidden_dims=(256, 256), **kwargs)
    return actor.to(device), reward_critic.to(device), cost_critic.to(device)


def _make_actor_optimizer(
    actor: IndependentActors,
    learning_rate: float,
) -> IndependentActorOptimizerGroup:
    """为四个独立 actors 各建一个 optimizer。"""

    return IndependentActorOptimizerGroup(actor, learning_rate)


def _entropy_coefficient(config: FixedLambdaConfig, environment_steps: int) -> float:
    settings = config.training
    decay_steps = max(
        1.0,
        settings.total_environment_steps * settings.entropy_decay_fraction,
    )
    fraction = min(1.0, environment_steps / decay_steps)
    return settings.entropy_coefficient_start + fraction * (
        settings.entropy_coefficient_end - settings.entropy_coefficient_start
    )


def _initial_occupancies(infos: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([np.asarray(info["target_occupancy"], dtype=np.int8) for info in infos])


def _diagnostic_discontinuity_mask(value: ArrayLike, num_envs: int) -> np.ndarray:
    """严格验证 hook 声明的人工边界跳转 slots。"""

    raw = np.asarray(value)
    if raw.shape != (num_envs,):
        raise ValueError(f"diagnostic discontinuity mask must have shape ({num_envs},)")
    if raw.dtype.kind != "b":
        raise TypeError("diagnostic discontinuity mask must contain booleans")
    return raw.astype(np.bool_, copy=True)


def _diagnostic_hook_state(
    diagnostic_hook: FixedLambdaTrainingDiagnosticHook | None,
) -> Mapping[str, Any] | None:
    if diagnostic_hook is None:
        return None
    state = diagnostic_hook.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("diagnostic_hook.state_dict() must return a mapping")
    return deepcopy(dict(state))


def _agent_target_occupancies(
    infos: list[dict[str, Any]],
    target_ids: tuple[str, ...],
    num_agents: int,
) -> np.ndarray:
    """把环境 region IDs 转成 ``[E,N,M]`` 二值归属，保留重复覆盖信息。"""

    target_index = {target_id: index for index, target_id in enumerate(target_ids)}
    result = np.zeros((len(infos), num_agents, len(target_ids)), dtype=np.int8)
    for environment_index, info in enumerate(infos):
        region_ids = tuple(info["agent_region_ids"])
        if len(region_ids) != num_agents:
            raise ValueError("agent_region_ids must contain one entry per agent")
        for agent_index, region_id in enumerate(region_ids):
            if region_id is not None:
                result[environment_index, agent_index, target_index[str(region_id)]] = 1
    return result


def _collect_rollout(
    actor: IndependentActors,
    vector_env: SyncVectorHRMR,
    observations: np.ndarray,
    states: np.ndarray,
    occupancies: np.ndarray,
    rollout_steps: int,
    device: torch.device,
    target_block_permutations: np.ndarray | None = None,
    actor_target_view: str = "all",
) -> tuple[RolloutBatch, np.ndarray, np.ndarray, np.ndarray]:
    """从 continuing environments 收集一个边界不终止的 rollout。"""

    buffer = RolloutBuffer(
        rollout_steps,
        vector_env.num_envs,
        vector_env.num_agents,
        actor.input_dim,
        vector_env.state_dim,
        vector_env.num_targets,
    )
    actor.eval()
    with torch.inference_mode():
        for _ in range(rollout_steps):
            actor_observations = (
                project_safe_target_actor_observations(observations)
                if actor_target_view == "safe_only"
                else observations
            )
            if target_block_permutations is not None:
                actor_observations = permute_target_blocks_by_environment(
                    actor_observations,
                    target_block_permutations,
                    num_agents=vector_env.num_agents,
                    num_targets=(
                        SAFE_ACTOR_NUM_TARGETS
                        if actor_target_view == "safe_only"
                        else vector_env.num_targets
                    ),
                    use_agent_id=vector_env.envs[0].config.use_agent_id,
                )
            observation_tensor = torch.as_tensor(
                actor_observations,
                dtype=torch.float32,
                device=device,
            )
            actions, log_probs, _ = actor.sample(observation_tensor, deterministic=False)
            action_array = actions.to("cpu").numpy().astype(np.int64, copy=False)
            log_prob_array = log_probs.to("cpu").numpy().astype(np.float64, copy=False)
            (
                next_observations,
                next_states,
                rewards,
                local_costs,
                global_costs,
                terminated,
                truncated,
                infos,
            ) = vector_env.step(action_array)
            if np.any(terminated) or np.any(truncated):
                raise AssertionError("HRMR continuing rollout unexpectedly terminated")
            next_occupancies = _initial_occupancies(infos)
            agent_target_occupancies = _agent_target_occupancies(
                infos,
                tuple(vector_env.envs[0].config.target_ids),
                vector_env.num_agents,
            )
            next_actor_observations = (
                project_safe_target_actor_observations(next_observations)
                if actor_target_view == "safe_only"
                else next_observations
            )
            if target_block_permutations is not None:
                next_actor_observations = permute_target_blocks_by_environment(
                    next_actor_observations,
                    target_block_permutations,
                    num_agents=vector_env.num_agents,
                    num_targets=(
                        SAFE_ACTOR_NUM_TARGETS
                        if actor_target_view == "safe_only"
                        else vector_env.num_targets
                    ),
                    use_agent_id=vector_env.envs[0].config.use_agent_id,
                )
            buffer.add(
                actor_observations,
                states,
                action_array,
                log_prob_array,
                rewards,
                global_costs,
                local_costs,
                occupancies,
                agent_target_occupancies,
                next_actor_observations,
                next_states,
                next_occupancies,
            )
            observations = next_observations
            states = next_states
            occupancies = next_occupancies
    actor.train()
    return buffer.freeze(), observations, states, occupancies


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    # Frozen rollout arrays 是只读的；显式复制避免 PyTorch 的可写性警告。
    return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)


def _frozen_rollout_quantities(
    batch: RolloutBatch,
    reward_critic: DifferentialCritic,
    cost_critic: DifferentialCritic,
    reward_rate: float,
    cost_rate: float,
    fixed_lambda: float,
    normalize_combined_advantage: bool,
    device: torch.device,
    n_step: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """用更新前 critics 一次计算并冻结 targets 与 combined advantage。"""

    steps = _validate_n_step(n_step)
    time_shape = (batch.num_steps, batch.num_envs)
    states = _tensor(batch.states.reshape(-1, batch.states.shape[-1]), device)
    next_states = _tensor(
        batch.next_states.reshape(-1, batch.next_states.shape[-1]),
        device,
    )
    rewards = _tensor(batch.rewards, device)
    costs = _tensor(batch.global_costs, device)
    reward_critic.eval()
    cost_critic.eval()
    with torch.inference_mode():
        reward_values = reward_critic(states).squeeze(-1).reshape(time_shape)
        cost_values = cost_critic(states).squeeze(-1).reshape(time_shape)
        next_reward_values = reward_critic(next_states).squeeze(-1).reshape(time_shape)
        next_cost_values = cost_critic(next_states).squeeze(-1).reshape(time_shape)
        reward_targets = differential_n_step_target(
            rewards,
            reward_rate,
            next_reward_values,
            steps,
        )
        cost_targets = differential_n_step_target(
            costs,
            cost_rate,
            next_cost_values,
            steps,
        )
        reward_advantages = reward_targets - reward_values
        cost_advantages = cost_targets - cost_values
        combined_advantages = combine_reward_cost_advantages(
            reward_advantages,
            cost_advantages,
            fixed_lambda,
            normalize=normalize_combined_advantage,
        )
    reward_critic.train()
    cost_critic.train()
    return (
        reward_targets.reshape(-1).detach().clone(),
        cost_targets.reshape(-1).detach().clone(),
        combined_advantages.reshape(-1).detach().clone(),
    )


def _minibatches(num_items: int, minibatch_size: int, device: torch.device) -> list[torch.Tensor]:
    permutation = torch.randperm(num_items, device=device)
    return list(permutation.split(minibatch_size))


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot aggregate empty update statistics")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _update_models(
    batch: RolloutBatch,
    actor: IndependentActors,
    reward_critic: DifferentialCritic,
    cost_critic: DifferentialCritic,
    actor_optimizer: IndependentActorOptimizerGroup,
    reward_optimizer: torch.optim.Optimizer,
    cost_optimizer: torch.optim.Optimizer,
    reward_targets: torch.Tensor,
    cost_targets: torch.Tensor,
    combined_advantages: torch.Tensor,
    entropy_coefficient: float,
    config: FixedLambdaConfig,
    device: torch.device,
) -> UpdateStatistics:
    """在同一组 frozen quantities 上完成多个 PPO epochs。"""

    settings = config.training
    num_time_env = batch.num_steps * batch.num_envs
    joint_actor_observations = _tensor(
        batch.observations.reshape(
            num_time_env,
            batch.num_agents,
            batch.observations.shape[-1],
        ),
        device,
    )
    joint_actor_actions = torch.tensor(
        batch.actions.reshape(num_time_env, batch.num_agents),
        dtype=torch.long,
        device=device,
    )
    joint_old_log_probs = _tensor(
        batch.log_probs.reshape(num_time_env, batch.num_agents),
        device,
    )
    critic_states = _tensor(batch.states.reshape(num_time_env, -1), device)

    actor_losses: list[float] = []
    actor_losses_by_agent: list[list[float]] = [[] for _ in range(batch.num_agents)]
    reward_losses: list[float] = []
    cost_losses: list[float] = []
    anchor_losses: list[float] = []
    entropies: list[float] = []
    entropies_by_agent: list[list[float]] = [[] for _ in range(batch.num_agents)]
    approximate_kls: list[float] = []
    clip_fractions: list[float] = []
    actor_grad_norms: list[float] = []
    reward_grad_norms: list[float] = []
    cost_grad_norms: list[float] = []

    actor.train()
    reward_critic.train()
    cost_critic.train()
    for _ in range(settings.ppo_epochs):
        for agent_index in range(batch.num_agents):
            policy_actor = actor.actor(agent_index)
            optimizer = actor_optimizer.optimizer(agent_index)
            for indices in _minibatches(
                num_time_env,
                settings.minibatch_size,
                device,
            ):
                new_log_probs, entropy = policy_actor.evaluate_actions(
                    joint_actor_observations[indices, agent_index],
                    joint_actor_actions[indices, agent_index],
                )
                policy = compute_ppo_policy_loss(
                    new_log_probs,
                    joint_old_log_probs[indices, agent_index],
                    combined_advantages[indices],
                    clip_epsilon=settings.ppo_clip,
                    entropy=entropy,
                    entropy_coefficient=entropy_coefficient,
                )
                optimizer.zero_grad(set_to_none=True)
                policy.loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    policy_actor.parameters(),
                    settings.max_grad_norm,
                )
                optimizer.step()
                actor_loss_value = float(policy.loss.detach().cpu())
                entropy_value = float(policy.entropy.detach().cpu())
                actor_losses.append(actor_loss_value)
                actor_losses_by_agent[agent_index].append(actor_loss_value)
                entropies.append(entropy_value)
                entropies_by_agent[agent_index].append(entropy_value)
                approximate_kls.append(float(policy.approximate_kl.cpu()))
                clip_fractions.append(float(policy.clip_fraction.cpu()))
                actor_grad_norms.append(float(actor_grad_norm.detach().cpu()))

        # Critic rollout 至多 4096 个 joint states；使用整个 frozen rollout，
        # 使 anchor 严格对应 (mean_t h(s_t))^2，而不是 minibatch mean 的有偏替代。
        reward_predictions = reward_critic(critic_states).squeeze(-1)
        cost_predictions = cost_critic(critic_states).squeeze(-1)
        reward_loss = differential_value_loss(reward_predictions, reward_targets)
        cost_loss = differential_value_loss(cost_predictions, cost_targets)
        anchor_loss = reward_predictions.mean().square() + cost_predictions.mean().square()
        total_critic_loss = (
            reward_loss + cost_loss + settings.value_anchor_coefficient * anchor_loss
        )
        reward_optimizer.zero_grad(set_to_none=True)
        cost_optimizer.zero_grad(set_to_none=True)
        total_critic_loss.backward()
        reward_grad_norm = nn.utils.clip_grad_norm_(
            reward_critic.parameters(), settings.max_grad_norm
        )
        cost_grad_norm = nn.utils.clip_grad_norm_(cost_critic.parameters(), settings.max_grad_norm)
        reward_optimizer.step()
        cost_optimizer.step()
        reward_losses.append(float(reward_loss.detach().cpu()))
        cost_losses.append(float(cost_loss.detach().cpu()))
        anchor_losses.append(float(anchor_loss.detach().cpu()))
        reward_grad_norms.append(float(reward_grad_norm.detach().cpu()))
        cost_grad_norms.append(float(cost_grad_norm.detach().cpu()))

    reward_critic.eval()
    with torch.inference_mode():
        reward_differential_values = reward_critic(critic_states).squeeze(-1)
    reward_critic.train()

    return UpdateStatistics(
        actor_loss=_mean(actor_losses),
        reward_critic_loss=_mean(reward_losses),
        cost_critic_loss=_mean(cost_losses),
        value_anchor_loss=_mean(anchor_losses),
        entropy=_mean(entropies),
        approx_kl=_mean(approximate_kls),
        clip_fraction=_mean(clip_fractions),
        actor_grad_norm=_mean(actor_grad_norms),
        reward_critic_grad_norm=_mean(reward_grad_norms),
        cost_critic_grad_norm=_mean(cost_grad_norms),
        reward_differential_value_mean=float(reward_differential_values.mean().detach().cpu()),
        reward_differential_value_std=float(
            reward_differential_values.std(unbiased=False).detach().cpu()
        ),
        reward_differential_value_abs_max=float(
            reward_differential_values.abs().max().detach().cpu()
        ),
        actor_loss_by_agent=tuple(_mean(values) for values in actor_losses_by_agent),
        actor_entropy_by_agent=tuple(_mean(values) for values in entropies_by_agent),
    )


def _training_row(
    *,
    fixed_lambda: float,
    training_seed: int,
    device: torch.device,
    environment_steps: int,
    num_agents: int,
    batch: RolloutBatch,
    reward_rate: float,
    cost_rate: float,
    statistics: UpdateStatistics,
    learning_rate: float,
    entropy_coefficient: float,
    all_hazardous_training_diagnostics: Mapping[str, int | float],
    operating_mode_training_diagnostics: Mapping[str, int | float],
) -> dict[str, Any]:
    mean_reward = float(np.mean(batch.rewards))
    mean_cost = float(np.mean(batch.global_costs))
    mean_occupancy = np.mean(batch.next_occupancies, axis=(0, 1))
    per_target_agent_counts = np.sum(batch.agent_target_occupancies, axis=2)
    unique_membership = batch.agent_target_occupancies * (
        per_target_agent_counts[:, :, None, :] == 1
    )
    duplicate_membership = batch.agent_target_occupancies * (
        per_target_agent_counts[:, :, None, :] > 1
    )
    unique_rates = np.mean(np.sum(unique_membership, axis=-1), axis=(0, 1))
    duplicate_rates = np.mean(np.sum(duplicate_membership, axis=-1), axis=(0, 1))
    non_target_rates = np.mean(
        np.sum(batch.agent_target_occupancies, axis=-1) == 0,
        axis=(0, 1),
    )
    row = {
        "fixed_lambda": fixed_lambda,
        "training_seed": training_seed,
        "device": str(device),
        "environment_steps": environment_steps,
        "agent_action_steps": environment_steps * num_agents,
        "rollout_mean_reward": mean_reward,
        "rollout_mean_cost": mean_cost,
        "avg_reward_estimate": reward_rate,
        "avg_cost_estimate": cost_rate,
        "rollout_scalarized_objective": mean_reward - fixed_lambda * mean_cost,
        "actor_loss": statistics.actor_loss,
        "reward_critic_loss": statistics.reward_critic_loss,
        "cost_critic_loss": statistics.cost_critic_loss,
        "value_anchor_loss": statistics.value_anchor_loss,
        "entropy": statistics.entropy,
        "approx_kl": statistics.approx_kl,
        "clip_fraction": statistics.clip_fraction,
        "actor_grad_norm": statistics.actor_grad_norm,
        "reward_critic_grad_norm": statistics.reward_critic_grad_norm,
        "cost_critic_grad_norm": statistics.cost_critic_grad_norm,
        "learning_rate": learning_rate,
        "entropy_coefficient": entropy_coefficient,
        **{
            column: float(value)
            for column, value in zip(OCCUPANCY_COLUMNS, mean_occupancy, strict=True)
        },
        "rollout_mean_num_distinct_targets": float(
            np.mean(np.sum(batch.next_occupancies, axis=-1))
        ),
        "rollout_all_hazardous_mode_rate": float(
            np.mean(
                np.all(
                    batch.next_occupancies[..., _HAZARDOUS_TARGET_INDICES] == 1,
                    axis=-1,
                )
            )
        ),
        "rollout_stay_action_rate": float(np.mean(batch.actions == int(Action.STAY))),
        "reward_differential_value_mean": statistics.reward_differential_value_mean,
        "reward_differential_value_std": statistics.reward_differential_value_std,
        "reward_differential_value_abs_max": statistics.reward_differential_value_abs_max,
        **dict(all_hazardous_training_diagnostics),
        **dict(operating_mode_training_diagnostics),
        **{
            f"actor_loss_agent_{agent_index + 1}": value
            for agent_index, value in enumerate(statistics.actor_loss_by_agent)
        },
        **{
            f"actor_entropy_agent_{agent_index + 1}": value
            for agent_index, value in enumerate(statistics.actor_entropy_by_agent)
        },
        **{
            f"unique_contribution_agent_{agent_index + 1}": float(value)
            for agent_index, value in enumerate(unique_rates)
        },
        **{
            f"duplicate_target_agent_{agent_index + 1}": float(value)
            for agent_index, value in enumerate(duplicate_rates)
        },
        **{
            f"non_target_agent_{agent_index + 1}": float(value)
            for agent_index, value in enumerate(non_target_rates)
        },
    }
    return row


def _checkpoint_extra(
    *,
    vector_env: SyncVectorHRMR,
    observations: np.ndarray,
    states: np.ndarray,
    occupancies: np.ndarray,
    reward_estimator: AverageRateEstimator,
    cost_estimator: AverageRateEstimator,
    training_seed: int,
    update_index: int,
    best_objective: float,
    best_environment_steps: int,
    next_evaluation_step: int,
    early_stopping_count: int,
    previous_evaluation: Mapping[str, Any] | None,
    target_permutation_rng_state: Mapping[str, Any] | None,
    all_hazardous_training_tracker_state: Mapping[str, Any],
    operating_mode_training_tracker_state: Mapping[str, Any],
    diagnostic_hook_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "vector_env": vector_env.state_dict(),
        "observations": np.asarray(observations).copy(),
        "states": np.asarray(states).copy(),
        "occupancies": np.asarray(occupancies).copy(),
        "reward_rate_estimator": reward_estimator.state_dict(),
        "cost_rate_estimator": cost_estimator.state_dict(),
        "training_seed": training_seed,
        "update_index": update_index,
        "best_objective": best_objective,
        "best_environment_steps": best_environment_steps,
        "next_evaluation_step": next_evaluation_step,
        "early_stopping_count": early_stopping_count,
        "previous_evaluation": (None if previous_evaluation is None else dict(previous_evaluation)),
        "target_permutation_rng_state": (
            None
            if target_permutation_rng_state is None
            else deepcopy(dict(target_permutation_rng_state))
        ),
        "all_hazardous_training_tracker": deepcopy(dict(all_hazardous_training_tracker_state)),
        "operating_mode_training_tracker": deepcopy(dict(operating_mode_training_tracker_state)),
    }
    if diagnostic_hook_state is not None:
        payload["diagnostic_hook"] = deepcopy(dict(diagnostic_hook_state))
    return payload


def _save_training_checkpoint(
    path: Path,
    *,
    actor: IndependentActors,
    reward_critic: DifferentialCritic,
    cost_critic: DifferentialCritic,
    actor_optimizer: IndependentActorOptimizerGroup,
    reward_optimizer: torch.optim.Optimizer,
    cost_optimizer: torch.optim.Optimizer,
    reward_rate: float,
    cost_rate: float,
    environment_steps: int,
    config: FixedLambdaConfig,
    fixed_lambda: float,
    extra: Mapping[str, Any],
) -> None:
    save_checkpoint(
        path,
        actor=actor,
        reward_critic=reward_critic,
        cost_critic=cost_critic,
        actor_optimizer=actor_optimizer,
        reward_critic_optimizer=reward_optimizer,
        cost_critic_optimizer=cost_optimizer,
        rho_reward=reward_rate,
        rho_cost=cost_rate,
        environment_steps=environment_steps,
        config=config,
        fixed_lambda=fixed_lambda,
        extra=extra,
    )


def _evaluate(
    actor: IndependentActors,
    environment_config: HRMRConfig,
    fixed_lambda: float,
    config: FixedLambdaConfig,
    device: torch.device,
) -> FixedLambdaEvaluation:
    evaluation_actor = (
        SafeTargetActorView(actor) if config.experiment.actor_target_view == "safe_only" else actor
    )
    return evaluate_fixed_lambda(
        evaluation_actor,
        lambda: HRMREnvironment(config=environment_config),
        fixed_lambda,
        config.evaluation.seeds,
        evaluation_steps=config.evaluation.evaluation_steps,
        burn_in_steps=config.evaluation.burn_in_steps,
        device=device,
    )


def _confidence_interval(sample_std: float, count: int) -> float:
    """用样本标准差计算正态近似 95% CI；n=1 时不可估并明确记为 0。"""

    return 0.0 if count <= 1 else 1.96 * float(sample_std) / math.sqrt(count)


def _reward_critic_evaluation_diagnostics(
    result: FixedLambdaEvaluation,
    reward_critic: DifferentialCritic,
    reward_rate: float,
    device: torch.device,
    n_step: int,
) -> dict[str, Any]:
    """按训练时的 reward horizon 评价 best differential critic。"""

    rate = float(reward_rate)
    if not math.isfinite(rate):
        raise ValueError("reward_rate must be finite")
    steps = _validate_n_step(n_step)
    was_training = reward_critic.training
    reward_critic.eval()
    per_seed = []
    all_values = []
    try:
        with torch.inference_mode():
            for trajectory in result.trajectories:
                start = trajectory.burn_in_steps
                current_states = _tensor(trajectory.state_history[start:-1], device)
                next_states = _tensor(trajectory.state_history[start + 1 :], device)
                rewards = _tensor(trajectory.reward_history[start:], device)
                values = reward_critic(current_states).squeeze(-1)
                next_values = reward_critic(next_states).squeeze(-1)
                targets = differential_n_step_target(
                    rewards,
                    rate,
                    next_values,
                    steps,
                )
                td_errors = targets - values
                value_array = values.detach().to("cpu").numpy().astype(np.float64)
                all_values.append(value_array)
                per_seed.append(
                    {
                        "evaluation_seed": trajectory.seed,
                        "reward_critic_loss": float(td_errors.square().mean().detach().cpu()),
                        "reward_differential_value_mean": float(np.mean(value_array)),
                        "reward_differential_value_std": float(np.std(value_array, ddof=0)),
                        "reward_differential_value_abs_max": float(np.max(np.abs(value_array))),
                    }
                )
    finally:
        reward_critic.train(was_training)

    pooled_values = np.concatenate(all_values)
    seed_losses = np.asarray(
        [item["reward_critic_loss"] for item in per_seed],
        dtype=np.float64,
    )
    return {
        "rho_reward": rate,
        "n_step": steps,
        "loss_definition": (
            "mean((sum_{l=0}^{k-1}(r[t+l] - rho_reward) "
            "+ h(s[t+k]) - h(s[t]))^2), "
            "k=min(n_step, remaining_steps)"
        ),
        "rollout_boundary_terminal": False,
        "value_std_ddof": 0,
        "aggregate": {
            "reward_critic_loss": float(np.mean(seed_losses)),
            "reward_critic_loss_seed_std": (
                0.0 if len(seed_losses) <= 1 else float(np.std(seed_losses, ddof=1))
            ),
            "reward_differential_value_mean": float(np.mean(pooled_values)),
            "reward_differential_value_std": float(np.std(pooled_values, ddof=0)),
            "reward_differential_value_abs_max": float(np.max(np.abs(pooled_values))),
        },
        "trajectories": per_seed,
    }


def evaluation_payload(
    result: FixedLambdaEvaluation,
    *,
    training_seed: int,
    reward_critic: DifferentialCritic | None = None,
    reward_rate: float | None = None,
    device: torch.device | None = None,
    n_step: int = 8,
) -> dict[str, Any]:
    """构造不包含长轨迹数组的正式评价 JSON。"""

    aggregate = result.aggregate
    mode = analytic_mode(result.fixed_lambda)
    gap = mode.scalarized_objective - aggregate.steady_mean_scalarized_objective
    occupancies = aggregate.steady_mean_target_occupancy
    learned_hazardous = tuple(
        target_id
        for target_id, value in zip(result.target_ids, occupancies, strict=True)
        if target_id.startswith("H") and value >= 0.5
    )
    learned_safe_count = int(
        sum(
            target_id.startswith("S") and value >= 0.5
            for target_id, value in zip(result.target_ids, occupancies, strict=True)
        )
    )
    operating_mode_assessment = assess_operating_mode(
        result.fixed_lambda,
        result.target_ids,
        occupancies,
        steady_reward=aggregate.steady_mean_reward,
        steady_cost=aggregate.steady_mean_cost,
        steady_mean_num_distinct_targets=aggregate.steady_mean_num_distinct_targets,
        steady_operating_mode_rate=aggregate.steady_mean_operating_mode_rate,
    )
    analytic_mode_success = operating_mode_assessment.success
    hazardous_occupancies = {
        target_id: float(value)
        for target_id, value in zip(result.target_ids, occupancies, strict=True)
        if target_id.startswith("H")
    }
    lambda0_criteria = {
        "steady_reward_at_least_0p90": aggregate.steady_mean_reward >= 0.90,
        "steady_cost_at_least_0p40": aggregate.steady_mean_cost >= 0.40,
        "mean_num_distinct_targets_at_least_3p8": (
            aggregate.steady_mean_num_distinct_targets >= 3.8
        ),
        "all_hazardous_mode_rate_at_least_0p75": (
            aggregate.steady_mean_all_hazardous_mode_rate >= 0.75
        ),
        **{
            f"occupancy_{target_id}_at_least_0p80": value >= 0.80
            for target_id, value in hazardous_occupancies.items()
        },
    }
    lambda0_success = result.fixed_lambda == 0.0 and all(lambda0_criteria.values())
    mode_success = analytic_mode_success
    steady_actor_entropy_by_agent = np.mean(
        np.stack(
            [
                trajectory.entropy_history[trajectory.burn_in_steps :]
                for trajectory in result.trajectories
            ]
        ),
        axis=(0, 1),
    )
    trajectories = []
    for trajectory in result.trajectories:
        trajectories.append(
            {
                "evaluation_seed": trajectory.seed,
                "whole_reward": trajectory.whole_reward,
                "whole_cost": trajectory.whole_cost,
                "whole_scalarized_objective": trajectory.whole_scalarized_objective,
                "steady_reward": trajectory.steady_reward,
                "steady_cost": trajectory.steady_cost,
                "steady_scalarized_objective": trajectory.steady_scalarized_objective,
                "steady_mean_num_distinct_targets": (trajectory.steady_mean_num_distinct_targets),
                "steady_all_hazardous_mode_rate": (trajectory.steady_all_hazardous_mode_rate),
                "steady_operating_mode_rate": trajectory.steady_operating_mode_rate,
                "steady_mean_hazardous_targets_covered": (
                    trajectory.steady_mean_hazardous_targets_covered
                ),
                "steady_mean_safe_targets_covered": (trajectory.steady_mean_safe_targets_covered),
                "steady_stay_action_rate": trajectory.steady_stay_action_rate,
                "steady_actor_entropy": trajectory.steady_actor_entropy,
                "steady_actor_entropy_by_agent": np.mean(
                    trajectory.entropy_history[trajectory.burn_in_steps :],
                    axis=0,
                ).tolist(),
                "steady_target_occupancy": trajectory.steady_target_occupancy.tolist(),
            }
        )
    payload = {
        "fixed_lambda": result.fixed_lambda,
        "training_seed": training_seed,
        "n_step": n_step,
        "sampling_mode": result.sampling_mode,
        "formal_stochastic_evaluation": result.is_formal_evaluation,
        "evaluation_seeds": list(result.evaluation_seeds),
        "burn_in_steps": result.burn_in_steps,
        "evaluation_steps": result.evaluation_steps,
        "target_ids": list(result.target_ids),
        "aggregate": {
            "num_trajectories": aggregate.num_trajectories,
            "whole_mean_reward": aggregate.whole_mean_reward,
            "whole_reward_ci95_half_width": _confidence_interval(
                aggregate.whole_std_reward, aggregate.num_trajectories
            ),
            "whole_mean_cost": aggregate.whole_mean_cost,
            "whole_cost_ci95_half_width": _confidence_interval(
                aggregate.whole_std_cost, aggregate.num_trajectories
            ),
            "steady_mean_reward": aggregate.steady_mean_reward,
            "steady_reward_ci95_half_width": _confidence_interval(
                aggregate.steady_std_reward, aggregate.num_trajectories
            ),
            "steady_mean_cost": aggregate.steady_mean_cost,
            "steady_cost_ci95_half_width": _confidence_interval(
                aggregate.steady_std_cost, aggregate.num_trajectories
            ),
            "whole_mean_scalarized_objective": aggregate.whole_mean_scalarized_objective,
            "steady_mean_scalarized_objective": aggregate.steady_mean_scalarized_objective,
            "steady_mean_num_distinct_targets": (aggregate.steady_mean_num_distinct_targets),
            "steady_num_distinct_targets_ci95_half_width": _confidence_interval(
                aggregate.steady_std_num_distinct_targets,
                aggregate.num_trajectories,
            ),
            "steady_all_hazardous_mode_rate": (aggregate.steady_mean_all_hazardous_mode_rate),
            "steady_all_hazardous_mode_rate_ci95_half_width": _confidence_interval(
                aggregate.steady_std_all_hazardous_mode_rate,
                aggregate.num_trajectories,
            ),
            "steady_operating_mode_rate": aggregate.steady_mean_operating_mode_rate,
            "steady_operating_mode_rate_ci95_half_width": _confidence_interval(
                aggregate.steady_std_operating_mode_rate,
                aggregate.num_trajectories,
            ),
            "steady_mean_hazardous_targets_covered": (
                aggregate.steady_mean_hazardous_targets_covered
            ),
            "steady_mean_safe_targets_covered": aggregate.steady_mean_safe_targets_covered,
            "steady_stay_action_rate": aggregate.steady_mean_stay_action_rate,
            "steady_stay_action_rate_ci95_half_width": _confidence_interval(
                aggregate.steady_std_stay_action_rate,
                aggregate.num_trajectories,
            ),
            "steady_actor_entropy": aggregate.steady_mean_actor_entropy,
            "steady_actor_entropy_by_agent": steady_actor_entropy_by_agent.tolist(),
            "steady_actor_entropy_ci95_half_width": _confidence_interval(
                aggregate.steady_std_actor_entropy,
                aggregate.num_trajectories,
            ),
            "whole_mean_target_occupancy": aggregate.whole_mean_target_occupancy.tolist(),
            "steady_mean_target_occupancy": occupancies.tolist(),
            "steady_mean_local_costs": aggregate.steady_mean_local_costs.tolist(),
        },
        "analytic_reference": asdict(mode),
        "scalarized_optimality_gap": gap,
        "learned_hazardous_targets_at_0p5": list(learned_hazardous),
        "learned_safe_target_count_at_0p5": learned_safe_count,
        "analytic_mode_success": analytic_mode_success,
        "operating_mode_assessment": asdict(operating_mode_assessment),
        "lambda0_success_criteria": lambda0_criteria if result.fixed_lambda == 0.0 else None,
        "lambda0_success": lambda0_success if result.fixed_lambda == 0.0 else None,
        "mode_success": mode_success,
        "trajectories": trajectories,
    }
    if (reward_critic is None) != (reward_rate is None):
        raise ValueError("reward_critic and reward_rate must be provided together")
    if reward_critic is not None:
        diagnostics_device = device or next(reward_critic.parameters()).device
        payload["evaluation_reward_critic_diagnostics"] = _reward_critic_evaluation_diagnostics(
            result,
            reward_critic,
            float(reward_rate),
            torch.device(diagnostics_device),
            n_step,
        )
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _training_diagnostics_at_step(
    path: Path,
    environment_steps: int,
    *,
    extra_metrics: tuple[str, ...] = (),
) -> dict[str, float | int]:
    """读取与 best checkpoint 同一步的正式训练诊断。"""

    selected: Mapping[str, str] | None = None
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["environment_steps"]) == environment_steps:
                selected = row
    if selected is None:
        raise ValueError("training log contains no row for the best checkpoint step")
    metrics = (
        "entropy",
        "entropy_coefficient",
        "reward_critic_loss",
        "cost_critic_loss",
        *FORMAL_TRAINING_DIAGNOSTIC_COLUMNS,
        *extra_metrics,
    )
    return {
        "environment_steps": environment_steps,
        **{metric: float(selected[metric]) for metric in metrics},
    }


def _write_run_config(
    destination: Path,
    source: Path,
    *,
    fixed_lambda: float,
    training_seed: int,
    device: torch.device,
) -> None:
    text = source.read_text(encoding="utf-8").rstrip()
    text += (
        "\n\n[run]\n"
        f"fixed_lambda = {fixed_lambda}\n"
        f"training_seed = {training_seed}\n"
        f'device = "{device}"\n'
    )
    destination.write_text(text, encoding="utf-8")


def _software_payload(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "matplotlib": version("matplotlib"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }


def _evaluation_snapshot(evaluation: FixedLambdaEvaluation) -> dict[str, Any]:
    aggregate = evaluation.aggregate
    return {
        "objective": aggregate.steady_mean_scalarized_objective,
        "reward": aggregate.steady_mean_reward,
        "cost": aggregate.steady_mean_cost,
        "occupancy": aggregate.steady_mean_target_occupancy.tolist(),
    }


def _early_stopping_update(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    fixed_lambda: float,
    config: FixedLambdaConfig,
    current_count: int,
) -> int:
    if previous is None:
        return 0
    settings = config.training
    mode = analytic_mode(fixed_lambda)
    objective_stable = (
        abs(float(current["objective"]) - float(previous["objective"]))
        <= settings.objective_tolerance
    )
    occupancy_stable = bool(
        np.max(
            np.abs(
                np.asarray(current["occupancy"], dtype=np.float64)
                - np.asarray(previous["occupancy"], dtype=np.float64)
            )
        )
        <= settings.occupancy_tolerance
    )
    analytic_close = (
        abs(float(current["reward"]) - mode.reward) <= settings.analytic_tolerance
        and abs(float(current["cost"]) - mode.cost) <= settings.analytic_tolerance
    )
    return current_count + 1 if objective_stable and occupancy_stable and analytic_close else 0


def _restore_continuing_rollout_state(
    extra: Mapping[str, Any],
    *,
    vector_env: SyncVectorHRMR,
    reward_estimator: AverageRateEstimator,
    cost_estimator: AverageRateEstimator,
    target_permutation_rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """恢复 continuing environment、rho estimators 与独立 permutation RNG。"""

    vector_env.load_state_dict(extra["vector_env"])
    observations = np.asarray(extra["observations"], dtype=np.float64)
    states = np.asarray(extra["states"], dtype=np.float64)
    occupancies = np.asarray(extra["occupancies"], dtype=np.int8)
    current_observations, current_states, current_infos = vector_env.current()
    current_occupancies = _initial_occupancies(current_infos)
    if not (
        np.array_equal(observations, current_observations)
        and np.array_equal(states, current_states)
        and np.array_equal(occupancies, current_occupancies)
    ):
        raise ValueError(
            "checkpoint observations/state/occupancy do not match restored vector environment"
        )
    reward_estimator.load_state_dict(extra["reward_rate_estimator"])
    cost_estimator.load_state_dict(extra["cost_rate_estimator"])
    if target_permutation_rng is not None:
        permutation_rng_state = extra.get("target_permutation_rng_state")
        if not isinstance(permutation_rng_state, Mapping):
            raise ValueError("random target-order checkpoint is missing permutation RNG state")
        target_permutation_rng.bit_generator.state = deepcopy(dict(permutation_rng_state))
    return observations.copy(), states.copy(), occupancies.copy()


def train_fixed_lambda(
    config_path: str | Path,
    fixed_lambda: float,
    training_seed: int,
    *,
    output_root: str | Path = "map_13x13/data/fixed_lambda_core",
    device_override: str | None = None,
    resume_from: str | Path | None = None,
    run_directory: str | Path | None = None,
    diagnostic_hook: FixedLambdaTrainingDiagnosticHook | None = None,
) -> TrainingResult:
    """训练或恢复一个 fixed-lambda run。"""

    config = load_training_config(config_path)
    lambda_value, seed = _validate_run_inputs(
        fixed_lambda,
        training_seed,
        profile=config.experiment.profile,
    )
    state_coverage_profile = config.experiment.profile in _STATE_COVERAGE_UPTAKE_PROFILES
    if state_coverage_profile != (diagnostic_hook is not None):
        raise ValueError(
            "state-coverage uptake profiles require exactly one diagnostic_hook, "
            "and other profiles forbid it"
        )
    if config.experiment.profile in _DIAGNOSTIC_LAMBDA_PROFILES and resume_from is not None:
        raise ValueError("large-dual diagnostics forbid warm start/resume")
    n_step = _validate_n_step(config.training.n_step)
    torch.set_num_threads(config.experiment.torch_num_threads)
    requested_device = device_override or config.experiment.device
    device = resolve_device(requested_device)
    seed_everything(seed)
    environment_config = _training_environment_config(config)
    _validate_actor_configuration(config, environment_config)
    target_permutation_rng = (
        np.random.default_rng(np.random.SeedSequence((seed, _TARGET_PERMUTATION_RNG_STREAM)))
        if config.experiment.randomize_actor_target_order
        else None
    )

    root = Path(output_root).expanduser().resolve()
    directory = (
        root / "runs" / run_name(config.experiment.profile, lambda_value, seed)
        if run_directory is None
        else Path(run_directory).expanduser().resolve()
    )
    directory.mkdir(parents=True, exist_ok=True)
    config_destination = directory / "config.toml"
    _write_run_config(
        config_destination,
        Path(config.source_path),
        fixed_lambda=lambda_value,
        training_seed=seed,
        device=device,
    )
    _write_json(directory / "software.json", _software_payload(device))

    actor, reward_critic, cost_critic = _make_networks(
        config,
        device,
        training_seed=seed,
    )
    actor_optimizer = _make_actor_optimizer(actor, config.training.actor_learning_rate)
    reward_optimizer = torch.optim.Adam(
        reward_critic.parameters(), lr=config.training.critic_learning_rate
    )
    cost_optimizer = torch.optim.Adam(
        cost_critic.parameters(),
        lr=config.training.critic_learning_rate,
    )
    vector_env = SyncVectorHRMR(
        config.training.num_parallel_envs,
        environment_config,
        seeds=seed * 100_000,
    )
    observations, states, infos = vector_env.reset()
    occupancies = _initial_occupancies(infos)
    reward_estimator = AverageRateEstimator(config.training.avg_rate_ema)
    cost_estimator = AverageRateEstimator(config.training.avg_rate_ema)
    all_hazardous_tracker = AllHazardousTrainingTracker(vector_env.num_envs)
    operating_mode_tracker = OperatingModeTrainingTracker(
        vector_env.num_envs,
        lambda_value,
        environment_config.target_ids,
    )

    environment_steps = 0
    update_index = 0
    best_objective = -math.inf
    best_environment_steps = 0
    next_evaluation_step = config.training.evaluation_interval
    early_stopping_count = 0
    previous_evaluation: Mapping[str, Any] | None = None
    append_log = False
    training_log = directory / "training_log.csv"
    best_checkpoint = directory / "best.pt"
    last_checkpoint = directory / "last.pt"
    if resume_from is not None:
        metadata, extra = load_checkpoint(
            resume_from,
            actor=actor,
            reward_critic=reward_critic,
            cost_critic=cost_critic,
            actor_optimizer=actor_optimizer,
            reward_critic_optimizer=reward_optimizer,
            cost_critic_optimizer=cost_optimizer,
            map_location=device,
            expected_fixed_lambda=lambda_value,
            expected_config=config,
        )
        observations, states, occupancies = _restore_continuing_rollout_state(
            extra,
            vector_env=vector_env,
            reward_estimator=reward_estimator,
            cost_estimator=cost_estimator,
            target_permutation_rng=target_permutation_rng,
        )
        if int(extra["training_seed"]) != seed:
            raise ValueError("checkpoint training_seed does not match requested training_seed")
        environment_steps = metadata.environment_steps
        update_index = int(extra["update_index"])
        best_objective = float(extra["best_objective"])
        best_environment_steps = int(extra["best_environment_steps"])
        next_evaluation_step = int(extra["next_evaluation_step"])
        early_stopping_count = int(extra["early_stopping_count"])
        raw_previous = extra["previous_evaluation"]
        previous_evaluation = None if raw_previous is None else dict(raw_previous)
        raw_tracker_state = extra.get("all_hazardous_training_tracker")
        if not isinstance(raw_tracker_state, Mapping):
            raise ValueError("checkpoint is missing all-hazardous training tracker state")
        all_hazardous_tracker.load_state_dict(raw_tracker_state)
        if all_hazardous_tracker.processed_environment_steps != environment_steps:
            raise ValueError(
                "all-hazardous tracker steps do not match checkpoint environment_steps"
            )
        raw_mode_tracker_state = extra.get("operating_mode_training_tracker")
        if not isinstance(raw_mode_tracker_state, Mapping):
            raise ValueError("checkpoint is missing operating-mode training tracker state")
        operating_mode_tracker.load_state_dict(raw_mode_tracker_state)
        if operating_mode_tracker.processed_environment_steps != environment_steps:
            raise ValueError(
                "operating-mode tracker steps do not match checkpoint environment_steps"
            )
        resume_path = Path(resume_from).resolve()
        source_best = (
            resume_path
            if resume_path.name == best_checkpoint.name
            else (resume_path.parent / best_checkpoint.name)
        )
        if best_environment_steps > 0 and not best_checkpoint.is_file():
            if not source_best.is_file():
                raise FileNotFoundError(
                    f"resuming after first eligible best requires sibling {best_checkpoint.name}"
                )
            if source_best != best_checkpoint.resolve():
                shutil.copy2(source_best, best_checkpoint)
        append_log = True

    start_time = time.perf_counter()
    stopped_early = False

    with TrainingCSVLogger(
        training_log,
        append=append_log,
        extra_fields=FORMAL_TRAINING_DIAGNOSTIC_COLUMNS,
    ) as logger:
        while environment_steps < config.training.total_environment_steps:
            rollout_start_environment_steps = environment_steps
            if diagnostic_hook is not None:
                discontinuity_mask = _diagnostic_discontinuity_mask(
                    diagnostic_hook.before_rollout(
                        vector_env=vector_env,
                        actor=actor,
                        environment_steps=environment_steps,
                        update_index=update_index,
                        device=device,
                    ),
                    vector_env.num_envs,
                )
                # Hook 只负责边界位置注入；trainer 从真实环境对象重新构造全部
                # current tensors，避免把 hook 的缓存或人工跳转写入 rollout。
                observations, states, infos = vector_env.current()
                occupancies = _initial_occupancies(infos)
                all_hazardous_tracker.break_continuity(discontinuity_mask)
                operating_mode_tracker.break_continuity(discontinuity_mask)
            remaining_steps = config.training.total_environment_steps - environment_steps
            if remaining_steps % vector_env.num_envs != 0:
                raise AssertionError("remaining environment steps must align with vector envs")
            rollout_steps = min(
                config.training.rollout_length,
                remaining_steps // vector_env.num_envs,
            )
            target_block_permutations = (
                None
                if target_permutation_rng is None
                else sample_target_permutations(
                    target_permutation_rng,
                    num_permutations=vector_env.num_envs,
                    num_targets=(
                        SAFE_ACTOR_NUM_TARGETS
                        if config.experiment.actor_target_view == "safe_only"
                        else vector_env.num_targets
                    ),
                )
            )
            batch, observations, states, occupancies = _collect_rollout(
                actor,
                vector_env,
                observations,
                states,
                occupancies,
                rollout_steps,
                device,
                target_block_permutations,
                actor_target_view=config.experiment.actor_target_view,
            )
            all_hazardous_training_diagnostics = all_hazardous_tracker.update(
                batch.next_occupancies,
                _HAZARDOUS_TARGET_INDICES,
                rollout_start_environment_steps=rollout_start_environment_steps,
            )
            operating_mode_training_diagnostics = operating_mode_tracker.update(
                batch.next_occupancies,
                rollout_start_environment_steps=rollout_start_environment_steps,
            )
            environment_steps += batch.num_steps * batch.num_envs
            update_index += 1
            reward_rate = reward_estimator.update(_tensor(batch.rewards, device))
            cost_rate = cost_estimator.update(_tensor(batch.global_costs, device))
            reward_targets, cost_targets, combined_advantages = _frozen_rollout_quantities(
                batch,
                reward_critic,
                cost_critic,
                reward_rate,
                cost_rate,
                lambda_value,
                config.training.normalize_combined_advantage,
                device,
                n_step,
            )
            if diagnostic_hook is not None:
                diagnostic_hook.after_frozen_quantities(
                    batch=batch,
                    actor=actor,
                    reward_critic=reward_critic,
                    cost_critic=cost_critic,
                    reward_targets=reward_targets.detach().clone(),
                    cost_targets=cost_targets.detach().clone(),
                    combined_advantages=combined_advantages.detach().clone(),
                    reward_rate=reward_rate,
                    cost_rate=cost_rate,
                    fixed_lambda=lambda_value,
                    rollout_start_environment_steps=rollout_start_environment_steps,
                    rollout_end_environment_steps=environment_steps,
                    update_index=update_index,
                    target_block_permutations=(
                        None
                        if target_block_permutations is None
                        else target_block_permutations.copy()
                    ),
                    device=device,
                )
            entropy_coefficient = _entropy_coefficient(config, environment_steps)
            statistics = _update_models(
                batch,
                actor,
                reward_critic,
                cost_critic,
                actor_optimizer,
                reward_optimizer,
                cost_optimizer,
                reward_targets,
                cost_targets,
                combined_advantages,
                entropy_coefficient,
                config,
                device,
            )
            if diagnostic_hook is not None:
                diagnostic_hook.after_update(
                    batch=batch,
                    actor=actor,
                    statistics=statistics,
                    fixed_lambda=lambda_value,
                    environment_steps=environment_steps,
                    update_index=update_index,
                    device=device,
                )
            logger.log(
                _training_row(
                    fixed_lambda=lambda_value,
                    training_seed=seed,
                    device=device,
                    environment_steps=environment_steps,
                    num_agents=vector_env.num_agents,
                    batch=batch,
                    reward_rate=reward_rate,
                    cost_rate=cost_rate,
                    statistics=statistics,
                    learning_rate=config.training.actor_learning_rate,
                    entropy_coefficient=entropy_coefficient,
                    all_hazardous_training_diagnostics=(all_hazardous_training_diagnostics),
                    operating_mode_training_diagnostics=(operating_mode_training_diagnostics),
                )
            )

            is_final_update = environment_steps >= config.training.total_environment_steps
            should_evaluate = environment_steps >= next_evaluation_step or is_final_update
            if should_evaluate:
                evaluation = _evaluate(
                    actor,
                    environment_config,
                    lambda_value,
                    config,
                    device,
                )
                evaluation_dir = directory / "evaluations"
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                evaluation_path = evaluation_dir / f"step_{environment_steps}.csv"
                write_evaluation_csv(evaluation_path, evaluation, training_seed=seed)
                current_objective = evaluation.aggregate.steady_mean_scalarized_objective
                current_evaluation = _evaluation_snapshot(evaluation)
                if config.training.early_stopping:
                    early_stopping_count = _early_stopping_update(
                        previous_evaluation,
                        current_evaluation,
                        lambda_value,
                        config,
                        early_stopping_count,
                    )
                previous_evaluation = current_evaluation
                is_new_best = current_objective > best_objective
                if is_new_best:
                    best_objective = current_objective
                    best_environment_steps = environment_steps
                next_evaluation_step = (
                    environment_steps // config.training.evaluation_interval + 1
                ) * config.training.evaluation_interval
                extra = _checkpoint_extra(
                    vector_env=vector_env,
                    observations=observations,
                    states=states,
                    occupancies=occupancies,
                    reward_estimator=reward_estimator,
                    cost_estimator=cost_estimator,
                    training_seed=seed,
                    update_index=update_index,
                    best_objective=best_objective,
                    best_environment_steps=best_environment_steps,
                    next_evaluation_step=next_evaluation_step,
                    early_stopping_count=early_stopping_count,
                    previous_evaluation=previous_evaluation,
                    target_permutation_rng_state=(
                        None
                        if target_permutation_rng is None
                        else target_permutation_rng.bit_generator.state
                    ),
                    all_hazardous_training_tracker_state=(all_hazardous_tracker.state_dict()),
                    operating_mode_training_tracker_state=(operating_mode_tracker.state_dict()),
                    diagnostic_hook_state=_diagnostic_hook_state(diagnostic_hook),
                )
                if is_new_best:
                    _save_training_checkpoint(
                        best_checkpoint,
                        actor=actor,
                        reward_critic=reward_critic,
                        cost_critic=cost_critic,
                        actor_optimizer=actor_optimizer,
                        reward_optimizer=reward_optimizer,
                        cost_optimizer=cost_optimizer,
                        reward_rate=reward_rate,
                        cost_rate=cost_rate,
                        environment_steps=environment_steps,
                        config=config,
                        fixed_lambda=lambda_value,
                        extra=extra,
                    )
                if (
                    config.training.early_stopping
                    and early_stopping_count >= config.training.early_stopping_patience
                ):
                    stopped_early = True

            checkpoint_extra = _checkpoint_extra(
                vector_env=vector_env,
                observations=observations,
                states=states,
                occupancies=occupancies,
                reward_estimator=reward_estimator,
                cost_estimator=cost_estimator,
                training_seed=seed,
                update_index=update_index,
                best_objective=best_objective,
                best_environment_steps=best_environment_steps,
                next_evaluation_step=next_evaluation_step,
                early_stopping_count=early_stopping_count,
                previous_evaluation=previous_evaluation,
                target_permutation_rng_state=(
                    None
                    if target_permutation_rng is None
                    else target_permutation_rng.bit_generator.state
                ),
                all_hazardous_training_tracker_state=(all_hazardous_tracker.state_dict()),
                operating_mode_training_tracker_state=(operating_mode_tracker.state_dict()),
                diagnostic_hook_state=_diagnostic_hook_state(diagnostic_hook),
            )
            if should_evaluate or is_final_update or stopped_early:
                _save_training_checkpoint(
                    last_checkpoint,
                    actor=actor,
                    reward_critic=reward_critic,
                    cost_critic=cost_critic,
                    actor_optimizer=actor_optimizer,
                    reward_optimizer=reward_optimizer,
                    cost_optimizer=cost_optimizer,
                    reward_rate=reward_estimator.value,
                    cost_rate=cost_estimator.value,
                    environment_steps=environment_steps,
                    config=config,
                    fixed_lambda=lambda_value,
                    extra=checkpoint_extra,
                )
            if stopped_early:
                break

    if not best_checkpoint.is_file():
        raise RuntimeError("training ended without a best checkpoint")
    best_metadata, _ = load_checkpoint(
        best_checkpoint,
        actor=actor,
        reward_critic=reward_critic,
        cost_critic=cost_critic,
        actor_optimizer=actor_optimizer,
        reward_critic_optimizer=reward_optimizer,
        cost_critic_optimizer=cost_optimizer,
        map_location=device,
        restore_rng=False,
        expected_fixed_lambda=lambda_value,
        expected_config=config,
    )
    formal_evaluation = evaluate_formal_actor(
        actor,
        environment_config,
        lambda_value,
        config.evaluation.seeds,
        evaluation_steps=config.evaluation.evaluation_steps,
        burn_in_steps=config.evaluation.burn_in_steps,
        device=device,
        actor_target_view=config.experiment.actor_target_view,
    )
    final_evaluation = formal_evaluation.canonical
    evaluation_csv = directory / "evaluation_canonical.csv"
    write_evaluation_csv(evaluation_csv, final_evaluation, training_seed=seed)
    payload = evaluation_payload(
        final_evaluation,
        training_seed=seed,
        reward_critic=reward_critic,
        reward_rate=best_metadata.rho_reward,
        device=device,
        n_step=n_step,
    )
    random_payload = evaluation_payload(
        formal_evaluation.random_order,
        training_seed=seed,
        n_step=n_step,
    )
    payload.update(
        actor_target_order_mode="canonical",
        agent_occupancy=formal_evaluation.canonical_payload["agent_occupancy"],
    )
    random_payload.update(
        actor_target_order_mode="random",
        agent_occupancy=formal_evaluation.random_order_payload["agent_occupancy"],
        source_target_order_in_actor_slots_by_seed=formal_evaluation.random_order_payload[
            "source_target_order_in_actor_slots_by_seed"
        ],
    )
    training_diagnostics = _training_diagnostics_at_step(
        training_log,
        best_environment_steps,
    )
    actor_advantage_contract = {
        "combined_advantage_before_optional_normalization": (
            f"A_R^({n_step}) - lambda * A_C^({n_step})"
        ),
        "n_step": n_step,
        "reward_target_definition": (
            "sum_{l=0}^{k-1}(r[t+l] - rho_reward) + h_R(s[t+k]), "
            "k=min(n_step, remaining_rollout_steps)"
        ),
        "cost_target_definition": (
            "sum_{l=0}^{k-1}(c[t+l] - rho_cost) + h_C(s[t+k]), "
            "k=min(n_step, remaining_rollout_steps)"
        ),
        "rollout_boundary_terminal": False,
        "normalize_combined_advantage": config.training.normalize_combined_advantage,
        "normalized_dual_used": False,
        "lambda_is_network_input": False,
        "cost_critic_affects_actor_loss": lambda_value > 0.0,
        "cost_credit_scope": "team_global",
        "team_reward_advantage_shared": True,
        "actor_i_uses_only_cost_advantage_i": False,
        "advantage_normalization_scope": "shared team advantage",
    }
    actor_observation_contract = {
        "base_observation_dimension": 48,
        "agent_id_dimension": 0,
        "actor_input_dimension": actor.input_dim,
        "one_hot_agent_id_appended": False,
        "parameter_sharing": False,
        "num_actor_policies": environment_config.num_agents,
        "num_actor_optimizers": environment_config.num_agents,
        "independent_actor_initialization_seeds": list(actor.initialization_seeds),
        "actor_parameter_count_by_policy": list(actor.parameter_count_by_actor),
        "actor_parameter_count_total": sum(parameter.numel() for parameter in actor.parameters()),
        "critic_input_dimension": reward_critic.input_dim,
        "critic_receives_agent_id": False,
        "random_target_order_training": config.experiment.randomize_actor_target_order,
        "target_permutation_scope": "one independent permutation per environment per rollout",
        "rollout_buffer_stores_exact_actor_observations": True,
        "ppo_regenerates_target_permutation": False,
        "critic_target_order_permuted": False,
        "actor_target_view": config.experiment.actor_target_view,
        "actor_visible_target_ids": (
            ["S1", "S2", "S3", "S4"]
            if config.experiment.actor_target_view == "safe_only"
            else list(environment_config.target_ids)
        ),
        "actor_omitted_target_ids": (
            ["H1", "H2", "H3", "H4"] if config.experiment.actor_target_view == "safe_only" else []
        ),
        "environment_num_targets": environment_config.num_targets,
        "actor_num_target_blocks": (
            SAFE_ACTOR_NUM_TARGETS
            if config.experiment.actor_target_view == "safe_only"
            else environment_config.num_targets
        ),
    }
    for evaluation_output in (payload, random_payload):
        evaluation_output["best_checkpoint_environment_steps"] = best_environment_steps
        evaluation_output["training_environment_steps"] = environment_steps
        evaluation_output["stopped_early"] = stopped_early
        evaluation_output["best_checkpoint_training_diagnostics"] = training_diagnostics
        evaluation_output["actor_advantage_contract"] = actor_advantage_contract
        evaluation_output["actor_observation_contract"] = actor_observation_contract
        evaluation_output["training_all_hazardous_tracking"] = {
            **all_hazardous_tracker.diagnostics(),
            "never_reached_first_step_sentinel": -1,
            "first_step_definition": (
                "cumulative environment-step boundary after the first synchronous "
                "vector transition where any training environment covers H1-H4"
            ),
            "since_first_rate_definition": (
                "all-H environment samples divided by all environment samples from "
                "the complete first-hit vector transition through training end"
            ),
            "retention_definition": (
                "same-environment probability that the transition immediately after "
                "an all-H transition is also all-H"
            ),
            "longest_streak_definition": (
                "maximum consecutive all-H vector transitions within one environment"
            ),
        }
        evaluation_output["training_optimal_mode_tracking"] = {
            **operating_mode_tracker.diagnostics(),
            "fixed_lambda": lambda_value,
            "target_ids": list(environment_config.target_ids),
            "never_reached_first_step_sentinel": -1,
            "first_step_definition": (
                "cumulative environment-step boundary after the first synchronous vector "
                "transition where any training environment exactly matches the analytic "
                "operating-mode allocation"
            ),
            "safe_target_identity_equivalent": True,
            "retention_definition": (
                "same-environment probability that the transition immediately after an "
                "optimal-mode transition is also in any equivalent optimal mode"
            ),
        }
    evaluation_json = directory / "evaluation_canonical.json"
    random_evaluation_json = directory / "evaluation_random_order.json"
    _write_json(evaluation_json, payload)
    _write_json(random_evaluation_json, random_payload)

    stochastic_trajectory = final_evaluation.trajectories[0]
    render_trajectory(
        environment_config,
        stochastic_trajectory.position_history,
        directory / "trajectory.png",
        title=(
            "Stochastic whole evaluation trajectory (burn-in + steady) | "
            f"lambda={lambda_value:.2f}, train seed={seed}, "
            f"eval seed={stochastic_trajectory.seed}; trajectory uncertainty N/A"
        ),
    )
    duration = time.perf_counter() - start_time
    payload["duration_seconds"] = duration
    random_payload["duration_seconds"] = duration
    _write_json(evaluation_json, payload)
    _write_json(random_evaluation_json, random_payload)

    aggregate = final_evaluation.aggregate
    mode = analytic_mode(lambda_value)
    return TrainingResult(
        fixed_lambda=lambda_value,
        training_seed=seed,
        profile=config.experiment.profile,
        device=str(device),
        environment_steps=environment_steps,
        agent_action_steps=environment_steps * environment_config.num_agents,
        updates=update_index,
        n_step=n_step,
        duration_seconds=duration,
        stopped_early=stopped_early,
        run_directory=directory,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        training_log=training_log,
        evaluation_json=evaluation_json,
        evaluation_csv=evaluation_csv,
        steady_reward=aggregate.steady_mean_reward,
        steady_cost=aggregate.steady_mean_cost,
        steady_scalarized_objective=aggregate.steady_mean_scalarized_objective,
        scalarized_optimality_gap=(
            mode.scalarized_objective - aggregate.steady_mean_scalarized_objective
        ),
        target_occupancy=tuple(float(value) for value in aggregate.steady_mean_target_occupancy),
        mode_success=bool(payload["mode_success"]),
    )


__all__ = [
    "SAFE_ACTOR_VIEW_DIAGNOSTIC_PROFILE",
    "SAFE_ACTOR_VIEW_SMOKE_PROFILE",
    "STATE_COVERAGE_UPTAKE_DIAGNOSTIC_PROFILE",
    "STATE_COVERAGE_UPTAKE_ENVIRONMENT_STEPS",
    "STATE_COVERAGE_UPTAKE_SMOKE_ENVIRONMENT_STEPS",
    "STATE_COVERAGE_UPTAKE_SMOKE_PROFILE",
    "FixedLambdaTrainingDiagnosticHook",
    "TrainingResult",
    "UpdateStatistics",
    "evaluation_payload",
    "run_name",
    "train_fixed_lambda",
]
