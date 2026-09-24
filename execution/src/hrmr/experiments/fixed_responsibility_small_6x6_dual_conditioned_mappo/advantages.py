"""Dual-conditioned differential targets；每个训练 dual 拥有独立平均率。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hrmr.experiments.assigned_targets_random_restart_ppo.sampling import (
    RestartRollout,
    frozen_restart_quantities,
)
from hrmr.rl.differential_td import normalize_advantages

from .schedule import ConditionalAverageRateBank


@dataclass(frozen=True)
class ConditionedFrozenQuantities:
    """一个 rollout 在更新前冻结的 critic targets 与 actor advantage。"""

    reward_targets: torch.Tensor
    cost_targets: torch.Tensor
    combined_advantages: torch.Tensor
    reward_rates: tuple[float, ...]
    cost_rates: tuple[float, ...]
    dual_lambdas: tuple[float, ...]

    @property
    def reward_rate(self) -> float:
        """兼容单 dual 调用的标量访问。"""

        if len(self.reward_rates) != 1:
            raise AttributeError("mixed-dual quantities have multiple reward rates")
        return self.reward_rates[0]

    @property
    def cost_rate(self) -> float:
        """兼容单 dual 调用的标量访问。"""

        if len(self.cost_rates) != 1:
            raise AttributeError("mixed-dual quantities have multiple cost rates")
        return self.cost_rates[0]

    @property
    def dual_lambda(self) -> float:
        """兼容单 dual 调用的标量访问。"""

        if len(self.dual_lambdas) != 1:
            raise AttributeError("mixed-dual quantities contain multiple dual values")
        return self.dual_lambdas[0]


def conditioned_restart_quantities(
    rollout: RestartRollout,
    reward_critic,
    cost_critic,
    reward_rate_bank: ConditionalAverageRateBank,
    cost_rate_bank: ConditionalAverageRateBank,
    dual_lambda: float,
    normalize_combined_advantage: bool,
    device: torch.device,
    n_step: int = 32,
) -> ConditionedFrozenQuantities:
    """更新当前 dual 的 rho，并计算 ``A_R - lambda * A_C``。"""

    reward_rate = reward_rate_bank.update(dual_lambda, rollout.batch.rewards.reshape(-1))
    cost_rate = cost_rate_bank.update(
        dual_lambda,
        rollout.batch.global_costs.reshape(-1),
    )
    reward_targets, cost_targets, combined = frozen_restart_quantities(
        rollout,
        reward_critic,
        cost_critic,
        reward_rate,
        cost_rate,
        dual_lambda,
        normalize_combined_advantage,
        device,
        n_step,
    )
    return ConditionedFrozenQuantities(
        reward_targets=reward_targets,
        cost_targets=cost_targets,
        combined_advantages=combined,
        reward_rates=(reward_rate,),
        cost_rates=(cost_rate,),
        dual_lambdas=(float(dual_lambda),),
    )


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)


def _differential_n_step_target_by_environment(
    signals: torch.Tensor,
    average_rates: torch.Tensor,
    next_values: torch.Tensor,
    n_step: int,
) -> torch.Tensor:
    """以逐环境 rho 计算 continuing n-step differential target。"""

    if signals.shape != next_values.shape or signals.ndim != 2:
        raise ValueError("signals and next_values must have identical [T,E] shapes")
    if average_rates.shape != (signals.shape[1],):
        raise ValueError("average_rates must contain one value per environment")
    with torch.no_grad():
        centered = signals.detach() - average_rates.detach().reshape(1, -1)
        prefix = torch.cat((torch.zeros_like(centered[:1]), torch.cumsum(centered, dim=0)))
        starts = torch.arange(signals.shape[0], device=signals.device)
        ends = torch.clamp(starts + n_step, max=signals.shape[0])
        centered_returns = prefix.index_select(0, ends) - prefix[:-1]
        bootstrap = next_values.detach().index_select(0, ends - 1)
        return (centered_returns + bootstrap).detach()


def _segment_quantities(
    segment,
    reward_critic,
    cost_critic,
    reward_rates: torch.Tensor,
    cost_rates: torch.Tensor,
    dual_lambdas: torch.Tensor,
    device: torch.device,
    n_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time_shape = (segment.num_steps, segment.num_envs)
    states = _tensor(segment.states.reshape(-1, segment.states.shape[-1]), device)
    next_states = _tensor(segment.next_states.reshape(-1, segment.next_states.shape[-1]), device)
    rewards = _tensor(segment.rewards, device)
    costs = _tensor(segment.global_costs, device)
    with torch.inference_mode():
        reward_values = reward_critic(states).squeeze(-1).reshape(time_shape)
        cost_values = cost_critic(states).squeeze(-1).reshape(time_shape)
        next_reward_values = reward_critic(next_states).squeeze(-1).reshape(time_shape)
        next_cost_values = cost_critic(next_states).squeeze(-1).reshape(time_shape)
        reward_targets = _differential_n_step_target_by_environment(
            rewards, reward_rates, next_reward_values, n_step
        )
        cost_targets = _differential_n_step_target_by_environment(
            costs, cost_rates, next_cost_values, n_step
        )
        combined = (reward_targets - reward_values) - dual_lambdas.reshape(1, -1) * (
            cost_targets - cost_values
        )
    return reward_targets, cost_targets, combined.detach()


def _normalize_by_dual(
    advantages: torch.Tensor,
    dual_lambdas: torch.Tensor,
) -> torch.Tensor:
    """每个 λ 独立标准化，避免不同 scalarization 尺度彼此污染。"""

    normalized = torch.empty_like(advantages)
    for dual in torch.unique(dual_lambdas):
        mask = dual_lambdas == dual
        values = advantages[:, mask]
        normalized[:, mask] = normalize_advantages(values.reshape(-1)).reshape(values.shape)
    return normalized


def mixed_conditioned_restart_quantities(
    rollout: RestartRollout,
    reward_critic,
    cost_critic,
    reward_rate_bank: ConditionalAverageRateBank,
    cost_rate_bank: ConditionalAverageRateBank,
    dual_lambdas: np.ndarray,
    normalize_combined_advantage: bool,
    device: torch.device,
    n_step: int = 32,
) -> ConditionedFrozenQuantities:
    """计算42环境 mixed-dual rollout 的逐 λ rho、targets 与优势。"""

    assignments = np.asarray(dual_lambdas, dtype=np.float64)
    if assignments.shape != (rollout.batch.num_envs,):
        raise ValueError("dual_lambdas must contain one value per rollout environment")
    if isinstance(n_step, bool) or not isinstance(n_step, int) or n_step <= 0:
        raise ValueError("n_step must be a positive integer")
    reward_rates_array = reward_rate_bank.update_by_environment(assignments, rollout.batch.rewards)
    cost_rates_array = cost_rate_bank.update_by_environment(assignments, rollout.batch.global_costs)
    reward_rates = _tensor(reward_rates_array, device)
    cost_rates = _tensor(cost_rates_array, device)
    duals = _tensor(assignments, device)
    reward_critic.eval()
    cost_critic.eval()
    quantities = [
        _segment_quantities(
            segment,
            reward_critic,
            cost_critic,
            reward_rates,
            cost_rates,
            duals,
            device,
            n_step,
        )
        for segment in rollout.segments
    ]
    reward_critic.train()
    cost_critic.train()
    reward_targets, cost_targets, combined = (
        torch.cat([part[index] for part in quantities], dim=0) for index in range(3)
    )
    if normalize_combined_advantage:
        combined = _normalize_by_dual(combined, duals)
    unique_duals = tuple(float(value) for value in reward_rate_bank.dual_values)
    return ConditionedFrozenQuantities(
        reward_targets=reward_targets.reshape(-1).detach().clone(),
        cost_targets=cost_targets.reshape(-1).detach().clone(),
        combined_advantages=combined.reshape(-1).detach().clone(),
        reward_rates=tuple(reward_rate_bank.value(value) for value in unique_duals),
        cost_rates=tuple(cost_rate_bank.value(value) for value in unique_duals),
        dual_lambdas=unique_duals,
    )


__all__ = [
    "ConditionedFrozenQuantities",
    "conditioned_restart_quantities",
    "mixed_conditioned_restart_quantities",
]
