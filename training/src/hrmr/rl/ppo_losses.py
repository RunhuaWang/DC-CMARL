"""HRMR Phase 2 的 PPO clipped policy objective 与诊断量。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import torch
from torch.nn import functional


def _matching_vectors(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> None:
    for name, tensor in (
        ("new_log_probs", new_log_probs),
        ("old_log_probs", old_log_probs),
        ("advantages", advantages),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise ValueError(f"{name} must be finite")
    if new_log_probs.shape != old_log_probs.shape or old_log_probs.shape != advantages.shape:
        raise ValueError("new_log_probs, old_log_probs, and advantages must have identical shapes")
    if new_log_probs.device != old_log_probs.device or old_log_probs.device != advantages.device:
        raise ValueError("PPO inputs must be on the same device")


def _clip_epsilon(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("clip_epsilon must be a real number")
    checked = float(value)
    if not isfinite(checked) or not 0.0 < checked < 1.0:
        raise ValueError("clip_epsilon must lie in (0, 1)")
    return checked


@dataclass(frozen=True)
class PPOPolicyLoss:
    """PPO actor loss 及不参与梯度的诊断统计。"""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio: torch.Tensor


def compute_ppo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    entropy: torch.Tensor | None = None,
    entropy_coefficient: float = 0.0,
) -> PPOPolicyLoss:
    """计算 PPO clipped surrogate loss。

    ``old_log_probs`` 与 ``advantages`` 被视为 rollout 常量并 detach；entropy
    项保留当前 actor 梯度。
    """

    _matching_vectors(new_log_probs, old_log_probs, advantages)
    clipping = _clip_epsilon(clip_epsilon)
    if isinstance(entropy_coefficient, bool) or not isinstance(entropy_coefficient, Real):
        raise TypeError("entropy_coefficient must be a real number")
    entropy_weight = float(entropy_coefficient)
    if not isfinite(entropy_weight) or entropy_weight < 0.0:
        raise ValueError("entropy_coefficient must be finite and non-negative")

    old = old_log_probs.detach()
    actor_advantages = advantages.detach()
    log_ratio = new_log_probs - old
    ratio = torch.exp(log_ratio)
    unclipped = ratio * actor_advantages
    clipped = torch.clamp(ratio, 1.0 - clipping, 1.0 + clipping) * actor_advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    if entropy is None:
        mean_entropy = torch.zeros((), device=new_log_probs.device, dtype=new_log_probs.dtype)
    else:
        if not isinstance(entropy, torch.Tensor):
            raise TypeError("entropy must be a torch.Tensor or None")
        if entropy.shape != new_log_probs.shape:
            raise ValueError("entropy must have the same shape as log probabilities")
        if entropy.device != new_log_probs.device:
            raise ValueError("entropy must be on the same device as log probabilities")
        if not torch.is_floating_point(entropy) or not bool(
            torch.isfinite(entropy.detach()).all().item()
        ):
            raise ValueError("entropy must be finite and floating point")
        mean_entropy = entropy.mean()

    loss = policy_loss - entropy_weight * mean_entropy
    with torch.no_grad():
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > clipping).to(ratio.dtype).mean()
    return PPOPolicyLoss(
        loss=loss,
        policy_loss=policy_loss,
        entropy=mean_entropy,
        approximate_kl=approximate_kl.detach(),
        clip_fraction=clip_fraction.detach(),
        ratio=ratio.detach(),
    )


def ppo_clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    entropy: torch.Tensor | None = None,
    entropy_coefficient: float = 0.0,
) -> torch.Tensor:
    """只返回可直接反向传播的 PPO actor scalar loss。"""

    return compute_ppo_policy_loss(
        new_log_probs,
        old_log_probs,
        advantages,
        clip_epsilon=clip_epsilon,
        entropy=entropy,
        entropy_coefficient=entropy_coefficient,
    ).loss


def differential_value_loss(
    predicted_values: torch.Tensor,
    detached_targets: torch.Tensor,
) -> torch.Tensor:
    """计算 differential critic 的均方误差；target 在边界再次 detach。"""

    if not isinstance(predicted_values, torch.Tensor) or not isinstance(
        detached_targets, torch.Tensor
    ):
        raise TypeError("predicted_values and detached_targets must be tensors")
    if predicted_values.shape != detached_targets.shape:
        raise ValueError("predicted_values and detached_targets must have identical shapes")
    if predicted_values.numel() == 0:
        raise ValueError("critic values must not be empty")
    if not torch.is_floating_point(predicted_values) or not torch.is_floating_point(
        detached_targets
    ):
        raise TypeError("predicted_values and detached_targets must be floating point")
    if predicted_values.device != detached_targets.device:
        raise ValueError("predicted_values and detached_targets must be on the same device")
    if not bool(torch.isfinite(predicted_values.detach()).all().item()) or not bool(
        torch.isfinite(detached_targets.detach()).all().item()
    ):
        raise ValueError("predicted_values and detached_targets must be finite")
    return functional.mse_loss(predicted_values, detached_targets.detach())


ppo_clipped_loss = ppo_clipped_policy_loss
value_loss = differential_value_loss


__all__ = [
    "PPOPolicyLoss",
    "compute_ppo_policy_loss",
    "differential_value_loss",
    "ppo_clipped_loss",
    "ppo_clipped_policy_loss",
    "value_loss",
]
