"""为 mixed-dual rollout 构造严格均衡的 PPO minibatches。"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import torch


def stratified_dual_minibatches(
    dual_lambdas: np.ndarray,
    *,
    num_steps: int,
    minibatch_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, ...]:
    """按 dual 等额切分 time-major ``[T,E]`` 样本，不遗漏也不重复。"""

    assignments = np.asarray(dual_lambdas)
    if assignments.ndim != 1 or assignments.size == 0:
        raise ValueError("dual_lambdas must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(assignments)):
        raise ValueError("dual_lambdas must contain only finite values")
    for name, value in (("num_steps", num_steps), ("minibatch_size", minibatch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    num_items = int(num_steps) * int(assignments.size)
    if num_items % int(minibatch_size) != 0:
        raise ValueError("rollout sample count must be divisible by minibatch_size")
    num_minibatches = num_items // int(minibatch_size)
    unique, counts = np.unique(assignments, return_counts=True)
    if not np.all(counts == counts[0]):
        raise ValueError("parallel environments must be balanced across dual values")
    samples_per_dual = int(num_steps) * int(counts[0])
    if samples_per_dual % num_minibatches != 0:
        raise ValueError("each dual sample count must be divisible by num_minibatches")

    target = torch.device(device)
    flat_duals = np.tile(assignments, int(num_steps))
    buckets: list[list[torch.Tensor]] = [[] for _ in range(num_minibatches)]
    for dual in unique:
        indices = torch.as_tensor(
            np.flatnonzero(flat_duals == dual),
            dtype=torch.long,
            device=target,
        )
        order = torch.randperm(indices.numel(), generator=generator, device=target)
        chunks = indices[order].chunk(num_minibatches)
        for batch_index, chunk in enumerate(chunks):
            buckets[batch_index].append(chunk)

    result = []
    for parts in buckets:
        combined = torch.cat(parts)
        order = torch.randperm(combined.numel(), generator=generator, device=target)
        result.append(combined[order])
    return tuple(result)


__all__ = ["stratified_dual_minibatches"]
