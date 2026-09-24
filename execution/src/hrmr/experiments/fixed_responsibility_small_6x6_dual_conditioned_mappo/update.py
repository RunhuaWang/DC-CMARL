"""Mixed-dual PPO 更新；actor minibatch 对 21 个 λ 严格等额。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from hrmr.rl.fixed_lambda_trainer import UpdateStatistics
from hrmr.rl.ppo_losses import compute_ppo_policy_loss, differential_value_loss

from .minibatches import stratified_dual_minibatches
from .pcgrad import (
    assign_flat_gradient,
    per_task_ppo_gradients,
    per_task_reference_kl,
    project_conflicting_gradients,
)

MEAN_GRADIENT_AGGREGATION = "mean"
PCGRAD_GRADIENT_AGGREGATION = "pcgrad"
SUPPORTED_GRADIENT_AGGREGATIONS = (
    MEAN_GRADIENT_AGGREGATION,
    PCGRAD_GRADIENT_AGGREGATION,
)


@dataclass(frozen=True)
class DualConditionedUpdateStatistics(UpdateStatistics):
    """在通用 PPO 统计之外记录 gradient surgery 的实际介入程度。"""

    actor_gradient_aggregation: str
    pcgrad_conflict_fraction: float
    pcgrad_mean_pairwise_cosine: float
    pcgrad_projection_relative_change: float
    pcgrad_conflict_fraction_by_agent: tuple[float, ...]
    pcgrad_mean_pairwise_cosine_by_agent: tuple[float, ...]
    pcgrad_projection_relative_change_by_agent: tuple[float, ...]
    policy_retention_active: bool
    policy_retention_kl: float
    policy_retention_loss: float
    policy_retention_kl_by_dual: tuple[float, ...]
    policy_retention_coefficient_by_dual: tuple[float, ...]
    policy_retention_next_coefficient_by_dual: tuple[float, ...]
    policy_retention_evaluations_per_dual: int


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot aggregate empty update statistics")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def update_models(
    batch,
    dual_lambdas: np.ndarray,
    actor,
    reward_critic,
    cost_critic,
    actor_optimizer,
    reward_optimizer,
    cost_optimizer,
    reward_targets: torch.Tensor,
    cost_targets: torch.Tensor,
    combined_advantages: torch.Tensor,
    config,
    device: torch.device,
    *,
    actor_gradient_aggregation: str = MEAN_GRADIENT_AGGREGATION,
    pcgrad_generator: torch.Generator | None = None,
    policy_retention=None,
    policy_retention_coefficient: float = 0.0,
    policy_retention_apply_every_minibatch: bool = False,
) -> DualConditionedUpdateStatistics:
    """在冻结 targets 上完成 PPO；每个 actor 仍只更新自己的参数。"""

    if actor_gradient_aggregation not in SUPPORTED_GRADIENT_AGGREGATIONS:
        raise ValueError(
            f"actor_gradient_aggregation must be one of {SUPPORTED_GRADIENT_AGGREGATIONS}"
        )
    settings = config.training
    num_time_env = batch.num_steps * batch.num_envs
    observations = _tensor(
        batch.observations.reshape(num_time_env, batch.num_agents, -1),
        device,
    )
    actions = torch.tensor(
        batch.actions.reshape(num_time_env, batch.num_agents),
        dtype=torch.long,
        device=device,
    )
    old_log_probs = _tensor(batch.log_probs.reshape(num_time_env, batch.num_agents), device)
    states = _tensor(batch.states.reshape(num_time_env, -1), device)

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
    pcgrad_conflict_fractions: list[float] = []
    pcgrad_cosines: list[float] = []
    pcgrad_relative_changes: list[float] = []
    pcgrad_conflicts_by_agent: list[list[float]] = [[] for _ in range(batch.num_agents)]
    pcgrad_cosines_by_agent: list[list[float]] = [[] for _ in range(batch.num_agents)]
    pcgrad_changes_by_agent: list[list[float]] = [[] for _ in range(batch.num_agents)]
    flat_duals = torch.tensor(
        np.tile(np.asarray(dual_lambdas, dtype=np.float32), batch.num_steps),
        dtype=torch.float32,
        device=device,
    )
    unique_duals = torch.unique(flat_duals, sorted=True)
    retention_active = policy_retention is not None and policy_retention.active
    if policy_retention is not None:
        actual_duals = tuple(float(value) for value in unique_duals.detach().cpu().tolist())
        if actual_duals != policy_retention.dual_values:
            raise ValueError("policy retention dual order does not match the rollout tasks")
        if not 0.0 < policy_retention_coefficient < float("inf"):
            raise ValueError("active policy retention requires a finite positive coefficient")
    elif policy_retention_coefficient != 0.0:
        raise ValueError("policy_retention_coefficient requires a retention bank")
    retention_kls_by_dual: list[list[float]] = [[] for _ in range(len(unique_duals))]
    retention_coefficients = None
    if retention_active:
        retention_coefficients = policy_retention.task_coefficients(policy_retention_coefficient)
        if retention_coefficients.shape != (len(unique_duals),):
            raise ValueError("policy retention returned one coefficient per configured dual")

    actor.train()
    reward_critic.train()
    cost_critic.train()
    for _ in range(settings.ppo_epochs):
        for agent_index in range(batch.num_agents):
            policy_actor = actor.actor(agent_index)
            optimizer = actor_optimizer.optimizer(agent_index)
            retention_observations = reference_log_probs = None
            if retention_active:
                retention_observations, reference_log_probs = policy_retention.task_tensors(
                    agent_index
                )
            minibatches = stratified_dual_minibatches(
                dual_lambdas,
                num_steps=batch.num_steps,
                minibatch_size=settings.minibatch_size,
                device=device,
            )
            for minibatch_index, indices in enumerate(minibatches):
                new_log_probs, entropy = policy_actor.evaluate_actions(
                    observations[indices, agent_index],
                    actions[indices, agent_index],
                )
                policy = compute_ppo_policy_loss(
                    new_log_probs,
                    old_log_probs[indices, agent_index],
                    combined_advantages[indices],
                    clip_epsilon=settings.ppo_clip,
                    entropy=entropy,
                    entropy_coefficient=settings.entropy_coefficient_start,
                )
                optimizer.zero_grad(set_to_none=True)
                # 固定系数基线保持每个 epoch 使用一次；自适应 trust-region
                # 实验则在每个 minibatch 都约束各 dual 的历史最佳 reference。
                apply_retention = retention_active and (
                    policy_retention_apply_every_minibatch or minibatch_index == 0
                )
                retention_kls = None
                if apply_retention:
                    assert retention_observations is not None
                    assert reference_log_probs is not None
                    retention_kls = per_task_reference_kl(
                        policy_actor,
                        retention_observations,
                        reference_log_probs,
                    )
                    assert retention_coefficients is not None
                    for dual_index, (value, coefficient) in enumerate(
                        zip(
                            retention_kls.detach().cpu().tolist(),
                            retention_coefficients.detach().cpu().tolist(),
                            strict=True,
                        )
                    ):
                        if coefficient > 0.0:
                            retention_kls_by_dual[dual_index].append(float(value))
                if actor_gradient_aggregation == PCGRAD_GRADIENT_AGGREGATION:
                    grouped_indices = tuple(
                        indices[flat_duals[indices] == dual] for dual in unique_duals
                    )
                    samples_per_task = {int(group.numel()) for group in grouped_indices}
                    if len(samples_per_task) != 1 or 0 in samples_per_task:
                        raise RuntimeError(
                            "PCGrad minibatch must contain equal non-empty dual groups"
                        )
                    task_indices = torch.stack(grouped_indices)
                    task_gradients, parameters = per_task_ppo_gradients(
                        policy_actor,
                        observations[task_indices, agent_index],
                        actions[task_indices, agent_index],
                        old_log_probs[task_indices, agent_index],
                        combined_advantages[task_indices],
                        clip_epsilon=settings.ppo_clip,
                        entropy_coefficient=settings.entropy_coefficient_start,
                        retention_observations=(
                            retention_observations if apply_retention else None
                        ),
                        reference_log_probs=(reference_log_probs if apply_retention else None),
                        retention_coefficient=(retention_coefficients if apply_retention else 0.0),
                    )
                    projection = project_conflicting_gradients(
                        task_gradients,
                        generator=pcgrad_generator,
                    )
                    assign_flat_gradient(parameters, projection.gradient)
                    pcgrad_conflict_fractions.append(projection.conflict_fraction)
                    pcgrad_cosines.append(projection.mean_pairwise_cosine)
                    pcgrad_relative_changes.append(projection.projection_relative_change)
                    pcgrad_conflicts_by_agent[agent_index].append(projection.conflict_fraction)
                    pcgrad_cosines_by_agent[agent_index].append(projection.mean_pairwise_cosine)
                    pcgrad_changes_by_agent[agent_index].append(
                        projection.projection_relative_change
                    )
                else:
                    total_policy_loss = policy.loss
                    if retention_kls is not None:
                        assert retention_coefficients is not None
                        total_policy_loss = (
                            total_policy_loss + (retention_coefficients * retention_kls).mean()
                        )
                    total_policy_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    policy_actor.parameters(), settings.max_grad_norm
                )
                optimizer.step()
                reported_actor_loss = policy.loss
                if retention_kls is not None:
                    assert retention_coefficients is not None
                    reported_actor_loss = (
                        reported_actor_loss + (retention_coefficients * retention_kls).mean()
                    )
                actor_loss = float(reported_actor_loss.detach().cpu())
                entropy_value = float(policy.entropy.detach().cpu())
                actor_losses.append(actor_loss)
                actor_losses_by_agent[agent_index].append(actor_loss)
                entropies.append(entropy_value)
                entropies_by_agent[agent_index].append(entropy_value)
                approximate_kls.append(float(policy.approximate_kl.cpu()))
                clip_fractions.append(float(policy.clip_fraction.cpu()))
                actor_grad_norms.append(float(grad_norm.detach().cpu()))

        reward_predictions = reward_critic(states).squeeze(-1)
        cost_predictions = cost_critic(states).squeeze(-1)
        reward_loss = differential_value_loss(reward_predictions, reward_targets)
        cost_loss = differential_value_loss(cost_predictions, cost_targets)
        anchor_loss = reward_predictions.mean().square() + cost_predictions.mean().square()
        critic_loss = reward_loss + cost_loss + settings.value_anchor_coefficient * anchor_loss
        reward_optimizer.zero_grad(set_to_none=True)
        cost_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
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
        reward_values = reward_critic(states).squeeze(-1)
    reward_critic.train()
    pcgrad_active = actor_gradient_aggregation == PCGRAD_GRADIENT_AGGREGATION
    retention_kl_by_dual = tuple(
        _mean(values) if values else float("nan") for values in retention_kls_by_dual
    )
    finite_retention_kls = [
        value for values in retention_kls_by_dual for value in values if np.isfinite(value)
    ]
    mean_retention_kl = _mean(finite_retention_kls) if finite_retention_kls else float("nan")
    if retention_active:
        assert retention_coefficients is not None
        mean_kl_tensor = torch.tensor(
            [0.0 if not values else _mean(values) for values in retention_kls_by_dual],
            dtype=torch.float32,
            device=retention_coefficients.device,
        )
        mean_retention_loss = float((retention_coefficients * mean_kl_tensor).mean().detach().cpu())
        used_retention_coefficients = tuple(
            float(value) for value in retention_coefficients.detach().cpu().tolist()
        )
        if getattr(policy_retention, "adaptive_kl", False):
            next_retention_coefficients = policy_retention.adapt_coefficients(
                max(0.0, value) if np.isfinite(value) else 0.0 for value in retention_kl_by_dual
            )
        else:
            next_retention_coefficients = used_retention_coefficients
    else:
        mean_retention_loss = float("nan")
        used_retention_coefficients = tuple(float("nan") for _ in unique_duals)
        next_retention_coefficients = used_retention_coefficients
    return DualConditionedUpdateStatistics(
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
        reward_differential_value_mean=float(reward_values.mean().detach().cpu()),
        reward_differential_value_std=float(reward_values.std(unbiased=False).detach().cpu()),
        reward_differential_value_abs_max=float(reward_values.abs().max().detach().cpu()),
        actor_loss_by_agent=tuple(_mean(values) for values in actor_losses_by_agent),
        actor_entropy_by_agent=tuple(_mean(values) for values in entropies_by_agent),
        actor_gradient_aggregation=actor_gradient_aggregation,
        pcgrad_conflict_fraction=(
            _mean(pcgrad_conflict_fractions) if pcgrad_active else float("nan")
        ),
        pcgrad_mean_pairwise_cosine=(_mean(pcgrad_cosines) if pcgrad_active else float("nan")),
        pcgrad_projection_relative_change=(
            _mean(pcgrad_relative_changes) if pcgrad_active else float("nan")
        ),
        pcgrad_conflict_fraction_by_agent=(
            tuple(_mean(values) for values in pcgrad_conflicts_by_agent)
            if pcgrad_active
            else tuple(float("nan") for _ in range(batch.num_agents))
        ),
        pcgrad_mean_pairwise_cosine_by_agent=(
            tuple(_mean(values) for values in pcgrad_cosines_by_agent)
            if pcgrad_active
            else tuple(float("nan") for _ in range(batch.num_agents))
        ),
        pcgrad_projection_relative_change_by_agent=(
            tuple(_mean(values) for values in pcgrad_changes_by_agent)
            if pcgrad_active
            else tuple(float("nan") for _ in range(batch.num_agents))
        ),
        policy_retention_active=retention_active,
        policy_retention_kl=mean_retention_kl,
        policy_retention_loss=mean_retention_loss,
        policy_retention_kl_by_dual=retention_kl_by_dual,
        policy_retention_coefficient_by_dual=used_retention_coefficients,
        policy_retention_next_coefficient_by_dual=next_retention_coefficients,
        policy_retention_evaluations_per_dual=(
            len(retention_kls_by_dual[0]) if retention_kls_by_dual else 0
        ),
    )


__all__ = [
    "MEAN_GRADIENT_AGGREGATION",
    "PCGRAD_GRADIENT_AGGREGATION",
    "SUPPORTED_GRADIENT_AGGREGATIONS",
    "DualConditionedUpdateStatistics",
    "update_models",
]
