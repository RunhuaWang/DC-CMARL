"""6×6 owner-aware MAPPO 的 dual 条件编码；不修改环境 observation/state。"""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.experiments.assigned_targets_ppo.environment import (
    ASSIGNED_OBSERVATION_DIM,
    ASSIGNED_STATE_DIM,
    permute_assigned_target_blocks,
)

FloatArray = NDArray[np.float64]
DUAL_ACTOR_INPUT_DIM = ASSIGNED_OBSERVATION_DIM + 1
DUAL_CRITIC_INPUT_DIM = ASSIGNED_STATE_DIM + 1


def normalize_conditioning_lambda(dual_lambda: Real, conditioning_max: Real) -> float:
    """验证算法 dual 区间并返回 ``lambda / conditioning_max``。"""

    if isinstance(dual_lambda, (bool, np.bool_)) or not isinstance(dual_lambda, Real):
        raise TypeError("dual_lambda must be a real number")
    if isinstance(conditioning_max, (bool, np.bool_)) or not isinstance(conditioning_max, Real):
        raise TypeError("conditioning_max must be a real number")
    value = float(dual_lambda)
    upper = float(conditioning_max)
    if not np.isfinite(value) or not np.isfinite(upper):
        raise ValueError("dual values must be finite")
    if upper <= 0.0:
        raise ValueError("conditioning_max must be positive")
    if not 0.0 <= value <= upper:
        raise ValueError("dual_lambda must lie in [0, conditioning_max]")
    return value / upper


def normalize_conditioning_lambdas(
    dual_lambdas: ArrayLike,
    conditioning_max: Real,
) -> FloatArray:
    """验证一维 dual batch，并返回逐环境归一化条件。"""

    if isinstance(conditioning_max, (bool, np.bool_)) or not isinstance(conditioning_max, Real):
        raise TypeError("conditioning_max must be a real number")
    upper = float(conditioning_max)
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("conditioning_max must be positive and finite")
    raw = np.asarray(dual_lambdas)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("dual_lambdas must be a non-empty one-dimensional array")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("dual_lambdas must contain real numeric values")
    checked = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(checked)):
        raise ValueError("dual_lambdas must contain only finite values")
    if np.any((checked < 0.0) | (checked > upper)):
        raise ValueError("dual_lambdas must lie in [0, conditioning_max]")
    return checked / upper


def _append_condition(
    values: ArrayLike,
    dual_lambda: Real | ArrayLike,
    conditioning_max: Real,
    *,
    base_dim: int,
    name: str,
) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim == 0 or raw.shape[-1] != base_dim:
        raise ValueError(f"{name} must have final dimension {base_dim}")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain real numeric values")
    checked = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain only finite values")
    if isinstance(dual_lambda, Real) and not isinstance(dual_lambda, (bool, np.bool_)):
        normalized = normalize_conditioning_lambda(dual_lambda, conditioning_max)
        condition = np.full((*checked.shape[:-1], 1), normalized, dtype=np.float64)
    else:
        if checked.ndim < 2:
            raise ValueError(f"batched dual_lambdas require batched {name}")
        normalized_batch = normalize_conditioning_lambdas(dual_lambda, conditioning_max)
        if normalized_batch.shape != (checked.shape[0],):
            raise ValueError(
                f"dual_lambdas must have shape ({checked.shape[0]},) for batched {name}"
            )
        reshape = (checked.shape[0],) + (1,) * (checked.ndim - 1)
        condition = np.broadcast_to(
            normalized_batch.reshape(reshape),
            (*checked.shape[:-1], 1),
        ).copy()
    return np.concatenate((checked, condition), axis=-1)


def condition_actor_observations(
    observations: ArrayLike,
    dual_lambda: Real | ArrayLike,
    conditioning_max: Real,
) -> FloatArray:
    """把 owner-aware 56D observation 转为 57D dual-conditioned 输入。"""

    return _append_condition(
        observations,
        dual_lambda,
        conditioning_max,
        base_dim=ASSIGNED_OBSERVATION_DIM,
        name="observations",
    )


def condition_critic_states(
    states: ArrayLike,
    dual_lambda: Real | ArrayLike,
    conditioning_max: Real,
) -> FloatArray:
    """把 16D centralized environment state 转为 17D critic 输入。"""

    return _append_condition(
        states,
        dual_lambda,
        conditioning_max,
        base_dim=ASSIGNED_STATE_DIM,
        name="states",
    )


def permute_conditioned_target_blocks(
    observations: ArrayLike,
    permutations: ArrayLike,
) -> FloatArray:
    """只排列 57D 输入中的完整六字段 target blocks，保持末尾 dual 不动。"""

    raw = np.asarray(observations)
    if raw.ndim not in {2, 3} or raw.shape[-1] != DUAL_ACTOR_INPUT_DIM:
        raise ValueError("observations must have shape [4,57] or [num_envs,4,57]")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("observations must contain real numeric values")
    checked = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(checked)):
        raise ValueError("observations must contain only finite values")
    if np.any((checked[..., -1] < 0.0) | (checked[..., -1] > 1.0)):
        raise ValueError("normalized dual condition must lie in [0, 1]")
    permuted = permute_assigned_target_blocks(checked[..., :-1], permutations)
    return np.concatenate((permuted, checked[..., -1:]), axis=-1)


__all__ = [
    "DUAL_ACTOR_INPUT_DIM",
    "DUAL_CRITIC_INPUT_DIM",
    "condition_actor_observations",
    "condition_critic_states",
    "normalize_conditioning_lambda",
    "normalize_conditioning_lambdas",
    "permute_conditioned_target_blocks",
]
