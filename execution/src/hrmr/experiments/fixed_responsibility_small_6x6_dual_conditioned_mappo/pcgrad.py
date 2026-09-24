"""用于 dual-conditioned Actor 的 per-lambda PCGrad。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional


@dataclass(frozen=True)
class PCGradProjection:
    """PCGrad 聚合梯度及投影前冲突诊断。"""

    gradient: torch.Tensor
    conflict_fraction: float
    mean_pairwise_cosine: float
    projection_relative_change: float


def project_conflicting_gradients(
    task_gradients: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> PCGradProjection:
    """按标准 PCGrad 顺序投影冲突梯度，并对任务等权平均。

    ``task_gradients`` 的每一行对应一个 dual task。投影系数在 task-space
    中计算，最后只执行一次矩阵乘法，避免逐参数反复投影带来的高开销。
    """

    if not isinstance(task_gradients, torch.Tensor):
        raise TypeError("task_gradients must be a torch.Tensor")
    if task_gradients.ndim != 2 or task_gradients.shape[0] < 2:
        raise ValueError("task_gradients must have shape [num_tasks >= 2, num_parameters]")
    if task_gradients.shape[1] == 0:
        raise ValueError("task_gradients must contain at least one parameter")
    if not torch.is_floating_point(task_gradients):
        raise TypeError("task_gradients must use a floating-point dtype")
    if not bool(torch.isfinite(task_gradients).all().item()):
        raise ValueError("task_gradients must contain only finite values")

    num_tasks = task_gradients.shape[0]
    gram = task_gradients @ task_gradients.transpose(0, 1)
    squared_norms = torch.diagonal(gram)
    norm_products = torch.sqrt(torch.clamp(squared_norms[:, None] * squared_norms[None, :], min=0))
    valid_pairs = norm_products > 0
    off_diagonal = ~torch.eye(num_tasks, dtype=torch.bool, device=task_gradients.device)
    measured_pairs = valid_pairs & off_diagonal
    cosines = torch.zeros_like(gram)
    cosines[valid_pairs] = gram[valid_pairs] / norm_products[valid_pairs]
    pair_count = int(measured_pairs.sum().item())
    if pair_count:
        conflict_fraction = float(
            ((gram < 0) & measured_pairs).sum().detach().cpu().item() / pair_count
        )
        mean_pairwise_cosine = float(cosines[measured_pairs].mean().detach().cpu().item())
    else:
        conflict_fraction = 0.0
        mean_pairwise_cosine = 0.0

    # projected_i 始终可写成 coefficient_i @ raw_gradients。利用 Gram matrix
    # 即可严格复现顺序投影，而无需在 parameter-space 中执行 O(T^2) 次 axpy。
    coefficients = torch.eye(num_tasks, dtype=task_gradients.dtype, device=task_gradients.device)
    for task_index in range(num_tasks):
        order = torch.randperm(num_tasks, generator=generator).tolist()
        for other_index in order:
            if task_index == other_index or float(squared_norms[other_index]) == 0.0:
                continue
            current_dot = torch.dot(coefficients[task_index], gram[:, other_index])
            if float(current_dot) < 0.0:
                coefficients[task_index, other_index] -= current_dot / squared_norms[other_index]

    projected = coefficients @ task_gradients
    raw_mean = task_gradients.mean(dim=0)
    projected_mean = projected.mean(dim=0)
    denominator = torch.linalg.vector_norm(raw_mean)
    if float(denominator) > 0.0:
        relative_change = float(
            (torch.linalg.vector_norm(projected_mean - raw_mean) / denominator)
            .detach()
            .cpu()
            .item()
        )
    else:
        relative_change = float(torch.linalg.vector_norm(projected_mean).detach().cpu().item())
    return PCGradProjection(
        gradient=projected_mean,
        conflict_fraction=conflict_fraction,
        mean_pairwise_cosine=mean_pairwise_cosine,
        projection_relative_change=relative_change,
    )


def _functional_ppo_loss(
    parameters: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    actor: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    entropy_coefficient: float,
) -> torch.Tensor:
    """无数据依赖校验的 PPO loss 核心，供 ``vmap(grad)`` 使用。"""

    logits = functional_call(
        actor,
        (parameters, buffers),
        (observations,),
        {"validate_values": False},
    )
    all_log_probs = functional.log_softmax(logits, dim=-1)
    probabilities = all_log_probs.exp()
    new_log_probs = all_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(probabilities * all_log_probs).sum(dim=-1).mean()
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.minimum(unclipped, clipped).mean() - entropy_coefficient * entropy


def _functional_retained_ppo_loss(
    parameters: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    actor: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    retention_observations: torch.Tensor,
    reference_log_probs: torch.Tensor,
    clip_epsilon: float,
    entropy_coefficient: float,
    retention_coefficient: float,
) -> torch.Tensor:
    """PPO task loss 加同一 dual 历史最佳策略的 forward-KL。"""

    ppo_loss = _functional_ppo_loss(
        parameters,
        buffers,
        actor,
        observations,
        actions,
        old_log_probs,
        advantages,
        clip_epsilon,
        entropy_coefficient,
    )
    current_logits = functional_call(
        actor,
        (parameters, buffers),
        (retention_observations,),
        {"validate_values": False},
    )
    current_log_probs = functional.log_softmax(current_logits, dim=-1)
    reference_probabilities = reference_log_probs.exp()
    retention_kl = (
        (reference_probabilities * (reference_log_probs - current_log_probs)).sum(dim=-1).mean()
    )
    return ppo_loss + retention_coefficient * retention_kl


def per_task_ppo_gradients(
    actor: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    entropy_coefficient: float,
    retention_observations: torch.Tensor | None = None,
    reference_log_probs: torch.Tensor | None = None,
    retention_coefficient: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, tuple[nn.Parameter, ...]]:
    """并行计算每个 dual task 对同一 Actor 的完整 PPO 梯度。

    输入前两维为 ``[num_tasks, samples_per_task]``。返回矩阵的每一行
    是相同参数顺序下的一个 task gradient。
    """

    expected_prefix = observations.shape[:2]
    if observations.ndim != 3 or observations.shape[0] < 2:
        raise ValueError("observations must have shape [num_tasks >= 2, samples, features]")
    for name, tensor in (
        ("actions", actions),
        ("old_log_probs", old_log_probs),
        ("advantages", advantages),
    ):
        if tensor.shape != expected_prefix:
            raise ValueError(f"{name} must match the first two observation dimensions")

    retention_enabled = retention_observations is not None or reference_log_probs is not None
    if retention_enabled:
        if retention_observations is None or reference_log_probs is None:
            raise ValueError(
                "retention_observations and reference_log_probs must be provided together"
            )
        if retention_observations.ndim != 3:
            raise ValueError(
                "retention_observations must have shape [num_tasks, anchors, features]"
            )
        if retention_observations.shape[0] != observations.shape[0]:
            raise ValueError("retention observations must match the number of dual tasks")
        if reference_log_probs.shape[:2] != retention_observations.shape[:2]:
            raise ValueError("reference_log_probs must match retention task and anchor dimensions")
        if reference_log_probs.ndim != 3:
            raise ValueError("reference_log_probs must have shape [num_tasks, anchors, actions]")
        if isinstance(retention_coefficient, torch.Tensor):
            retention_coefficients = retention_coefficient.to(
                device=observations.device,
                dtype=observations.dtype,
            )
            if retention_coefficients.shape != (observations.shape[0],):
                raise ValueError("retention coefficients must contain one value per task")
            if not bool(torch.all(torch.isfinite(retention_coefficients))):
                raise ValueError("retention coefficients must be finite")
            if bool(torch.any(retention_coefficients < 0.0)) or not bool(
                torch.any(retention_coefficients > 0.0)
            ):
                raise ValueError(
                    "retention coefficients must be non-negative with a positive entry"
                )
        else:
            coefficient = float(retention_coefficient)
            if not 0.0 < coefficient < float("inf"):
                raise ValueError("retention_coefficient must be finite and positive")
            retention_coefficients = torch.full(
                (observations.shape[0],),
                coefficient,
                dtype=observations.dtype,
                device=observations.device,
            )
    else:
        if isinstance(retention_coefficient, torch.Tensor):
            disabled_coefficient = bool(torch.any(retention_coefficient != 0.0))
        else:
            disabled_coefficient = float(retention_coefficient) != 0.0
        if disabled_coefficient:
            raise ValueError("retention_coefficient requires retention tensors")
        retention_coefficients = None

    named_parameters = tuple(actor.named_parameters())
    parameters = {name: parameter for name, parameter in named_parameters}
    buffers = dict(actor.named_buffers())
    if retention_enabled:
        gradient_function = grad(_functional_retained_ppo_loss, argnums=0)
        task_gradient_tree = vmap(
            gradient_function,
            in_dims=(None, None, None, 0, 0, 0, 0, 0, 0, None, None, 0),
            randomness="error",
        )(
            parameters,
            buffers,
            actor,
            observations,
            actions,
            old_log_probs.detach(),
            advantages.detach(),
            retention_observations,
            reference_log_probs.detach(),
            float(clip_epsilon),
            float(entropy_coefficient),
            retention_coefficients,
        )
    else:
        gradient_function = grad(_functional_ppo_loss, argnums=0)
        task_gradient_tree = vmap(
            gradient_function,
            in_dims=(None, None, None, 0, 0, 0, 0, None, None),
            randomness="error",
        )(
            parameters,
            buffers,
            actor,
            observations,
            actions,
            old_log_probs.detach(),
            advantages.detach(),
            float(clip_epsilon),
            float(entropy_coefficient),
        )
    flat_gradients = torch.cat(
        [
            task_gradient_tree[name].reshape(observations.shape[0], -1)
            for name, _ in named_parameters
        ],
        dim=1,
    )
    return flat_gradients.detach(), tuple(parameter for _, parameter in named_parameters)


def per_task_reference_kl(
    actor: nn.Module,
    retention_observations: torch.Tensor,
    reference_log_probs: torch.Tensor,
) -> torch.Tensor:
    """计算每个 dual reference 到当前 actor 的 forward-KL。"""

    if retention_observations.ndim != 3:
        raise ValueError("retention_observations must have shape [tasks, anchors, features]")
    if reference_log_probs.ndim != 3:
        raise ValueError("reference_log_probs must have shape [tasks, anchors, actions]")
    if reference_log_probs.shape[:2] != retention_observations.shape[:2]:
        raise ValueError("reference_log_probs must match retention task and anchor dimensions")
    logits = actor(retention_observations.reshape(-1, retention_observations.shape[-1]))
    current = functional.log_softmax(logits, dim=-1).reshape_as(reference_log_probs)
    probabilities = reference_log_probs.exp()
    return (probabilities * (reference_log_probs - current)).sum(dim=-1).mean(dim=-1)


def assign_flat_gradient(parameters: tuple[nn.Parameter, ...], flat_gradient: torch.Tensor) -> None:
    """把扁平梯度按参数顺序写回 ``.grad``，供原 Adam optimizer 使用。"""

    expected = sum(parameter.numel() for parameter in parameters)
    if flat_gradient.ndim != 1 or flat_gradient.numel() != expected:
        raise ValueError("flat_gradient size does not match actor parameters")
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = flat_gradient[offset : offset + count].view_as(parameter).detach().clone()
        offset += count


__all__ = [
    "PCGradProjection",
    "assign_flat_gradient",
    "per_task_ppo_gradients",
    "per_task_reference_kl",
    "project_conflicting_gradients",
]
