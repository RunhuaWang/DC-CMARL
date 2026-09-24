"""无折扣、无 terminal mask 的 differential TD 计算。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import torch


def _matching_tensors(
    signals: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
) -> None:
    for name, tensor in (
        ("signals", signals),
        ("values", values),
        ("next_values", next_values),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must use a floating-point dtype")
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise ValueError(f"{name} must be finite")
    if signals.shape != values.shape or values.shape != next_values.shape:
        raise ValueError("signals, values, and next_values must have identical shapes")
    if signals.device != values.device or values.device != next_values.device:
        raise ValueError("signals, values, and next_values must be on the same device")


def _rate_tensor(average_rate: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(average_rate, torch.Tensor):
        if average_rate.numel() != 1:
            raise ValueError("average_rate tensor must contain exactly one value")
        if not bool(torch.isfinite(average_rate.detach()).all().item()):
            raise ValueError("average_rate must be finite")
        return average_rate.detach().to(device=reference.device, dtype=reference.dtype).reshape(())
    if isinstance(average_rate, bool) or not isinstance(average_rate, Real):
        raise TypeError("average_rate must be a real scalar or scalar tensor")
    rate = torch.as_tensor(average_rate, device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(rate).item()):
        raise ValueError("average_rate must be finite")
    return rate


def differential_td_target(
    signals: torch.Tensor,
    average_rate: float | torch.Tensor,
    next_values: torch.Tensor,
) -> torch.Tensor:
    """返回 detached 一步 target ``signal - average_rate + h(next)``。

    这里故意没有 discount factor、terminal mask 或 rollout-boundary mask；HRMR
    是 continuing task，rollout 边界仍应 bootstrap。
    """

    if not isinstance(signals, torch.Tensor) or not isinstance(next_values, torch.Tensor):
        raise TypeError("signals and next_values must be torch.Tensor objects")
    if signals.shape != next_values.shape:
        raise ValueError("signals and next_values must have identical shapes")
    if not torch.is_floating_point(signals) or not torch.is_floating_point(next_values):
        raise TypeError("signals and next_values must use floating-point dtypes")
    if signals.device != next_values.device:
        raise ValueError("signals and next_values must be on the same device")
    rate = _rate_tensor(average_rate, signals)
    with torch.no_grad():
        target = signals.detach() - rate + next_values.detach()
        if not bool(torch.isfinite(target).all().item()):
            raise ValueError("differential TD target must be finite")
    return target.detach()


def differential_n_step_target(
    signals: torch.Tensor,
    average_rate: float | torch.Tensor,
    next_values: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """沿首维返回正 bootstrap 的无折扣 n-step differential target。

    输入首维是 rollout 时间维，其他维（例如 parallel environments）彼此
    独立。对时刻 ``t`` 使用 ``k=min(num_steps, T-t)``：

    ``sum_{l=0}^{k-1}(signal[t+l]-average_rate) + h(s[t+k])``。

    ``next_values[t]`` 对应 ``h(s[t+1])``。末端不足 ``num_steps`` 时缩短
    累积长度并在可用的最后状态 bootstrap；rollout 边界从不视作 terminal。
    """

    if not isinstance(signals, torch.Tensor) or not isinstance(next_values, torch.Tensor):
        raise TypeError("signals and next_values must be torch.Tensor objects")
    if signals.shape != next_values.shape:
        raise ValueError("signals and next_values must have identical shapes")
    if signals.ndim == 0 or signals.shape[0] == 0:
        raise ValueError("signals and next_values must have a non-empty time dimension")
    if not torch.is_floating_point(signals) or not torch.is_floating_point(next_values):
        raise TypeError("signals and next_values must use floating-point dtypes")
    if signals.device != next_values.device:
        raise ValueError("signals and next_values must be on the same device")
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
        raise TypeError("num_steps must be an integer")
    horizon = int(num_steps)
    if horizon <= 0:
        raise ValueError("num_steps must be positive")

    rate = _rate_tensor(average_rate, signals)
    with torch.no_grad():
        centered = signals.detach() - rate
        zero = torch.zeros_like(centered[:1])
        prefix = torch.cat((zero, torch.cumsum(centered, dim=0)), dim=0)
        starts = torch.arange(signals.shape[0], device=signals.device)
        ends = torch.clamp(starts + horizon, max=signals.shape[0])
        centered_returns = prefix.index_select(0, ends) - prefix[:-1]
        bootstrap = next_values.detach().index_select(0, ends - 1)
        target = centered_returns + bootstrap
        if not bool(torch.isfinite(target).all().item()):
            raise ValueError("n-step differential TD target must be finite")
    return target.detach()


def differential_td_error(
    signals: torch.Tensor,
    average_rate: float | torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
) -> torch.Tensor:
    """返回 ``target.detach() - h(current)``，保留 current critic 的梯度。"""

    _matching_tensors(signals, values, next_values)
    target = differential_td_target(signals, average_rate, next_values)
    return target - values


def differential_td_advantage(
    signals: torch.Tensor,
    average_rate: float | torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
) -> torch.Tensor:
    """返回供 actor 使用的完全 detached 一步 differential TD advantage。"""

    return differential_td_error(signals, average_rate, values, next_values).detach()


def normalize_advantages(
    advantages: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """用 population standard deviation 对一个 minibatch 做标准化。"""

    if not isinstance(advantages, torch.Tensor):
        raise TypeError("advantages must be a torch.Tensor")
    if not torch.is_floating_point(advantages):
        raise TypeError("advantages must use a floating-point dtype")
    if advantages.numel() == 0:
        raise ValueError("advantages must not be empty")
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real number")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    detached = advantages.detach()
    mean = detached.mean()
    standard_deviation = detached.std(unbiased=False)
    return (detached - mean) / (standard_deviation + float(epsilon))


def combine_reward_cost_advantages(
    reward_advantages: torch.Tensor,
    cost_advantages: torch.Tensor,
    dual_lambda: float | torch.Tensor,
    *,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """先计算 ``A_R - lambda*A_C``，再按需标准化。

    ``dual_lambda`` 是原始对偶变量，不是网络输入使用的归一化 dual。输入
    advantages 会先 detach，防止 actor loss 反向传播进两个 critics。
    """

    if not isinstance(reward_advantages, torch.Tensor) or not isinstance(
        cost_advantages, torch.Tensor
    ):
        raise TypeError("reward_advantages and cost_advantages must be tensors")
    if reward_advantages.shape != cost_advantages.shape:
        raise ValueError("reward and cost advantages must have identical shapes")
    if reward_advantages.device != cost_advantages.device:
        raise ValueError("reward and cost advantages must be on the same device")
    if not torch.is_floating_point(reward_advantages) or not torch.is_floating_point(
        cost_advantages
    ):
        raise TypeError("reward and cost advantages must use floating-point dtypes")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")

    dual = _rate_tensor(dual_lambda, reward_advantages)
    if bool((dual < 0.0).item()):
        raise ValueError("dual_lambda must be non-negative")
    combined = reward_advantages.detach() - dual * cost_advantages.detach()
    if not bool(torch.isfinite(combined).all().item()):
        raise ValueError("combined advantages must be finite")
    return normalize_advantages(combined, epsilon=epsilon) if normalize else combined


@dataclass(frozen=True)
class RewardCostTD:
    """同一步 reward/cost critic 的 targets 与 TD errors。"""

    reward_target: torch.Tensor
    cost_target: torch.Tensor
    reward_error: torch.Tensor
    cost_error: torch.Tensor


def reward_cost_differential_td(
    rewards: torch.Tensor,
    costs: torch.Tensor,
    average_reward: float | torch.Tensor,
    average_cost: float | torch.Tensor,
    reward_values: torch.Tensor,
    cost_values: torch.Tensor,
    next_reward_values: torch.Tensor,
    next_cost_values: torch.Tensor,
) -> RewardCostTD:
    """一次性计算分离 reward/cost critics 的一步 differential TD 量。"""

    reward_target = differential_td_target(rewards, average_reward, next_reward_values)
    cost_target = differential_td_target(costs, average_cost, next_cost_values)
    reward_error = differential_td_error(
        rewards,
        average_reward,
        reward_values,
        next_reward_values,
    )
    cost_error = differential_td_error(
        costs,
        average_cost,
        cost_values,
        next_cost_values,
    )
    return RewardCostTD(
        reward_target=reward_target,
        cost_target=cost_target,
        reward_error=reward_error,
        cost_error=cost_error,
    )


compute_differential_td_target = differential_td_target
compute_differential_td_error = differential_td_error
combine_advantages = combine_reward_cost_advantages


__all__ = [
    "RewardCostTD",
    "combine_advantages",
    "combine_reward_cost_advantages",
    "compute_differential_td_error",
    "compute_differential_td_target",
    "differential_n_step_target",
    "differential_td_advantage",
    "differential_td_error",
    "differential_td_target",
    "normalize_advantages",
    "reward_cost_differential_td",
]
