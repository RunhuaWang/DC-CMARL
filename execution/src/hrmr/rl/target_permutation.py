"""Actor observation 中完整 target feature blocks 的排列工具。

函数只返回输入副本，不改变环境 observation、centralized state、reward 或 cost。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import factorial
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.observations import TARGET_FEATURE_SIZE, observation_layout

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def validate_target_permutation(
    permutation: Sequence[int] | ArrayLike,
    *,
    num_targets: int,
) -> IntArray:
    """验证并复制一个 zero-based target-block 排列。"""

    if isinstance(num_targets, bool) or not isinstance(num_targets, Integral):
        raise TypeError("num_targets must be an integer")
    targets = int(num_targets)
    if targets <= 0:
        raise ValueError("num_targets must be positive")
    array = np.asarray(permutation)
    if array.shape != (targets,):
        raise ValueError(f"permutation must have shape ({targets},)")
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError("permutation must contain integers")
    checked = array.astype(np.int64, copy=True)
    if not np.array_equal(np.sort(checked), np.arange(targets, dtype=np.int64)):
        raise ValueError("permutation must contain each target index exactly once")
    return checked


def _checked_observations(
    observations: ArrayLike,
    *,
    num_agents: int,
    num_targets: int,
    use_agent_id: bool,
) -> tuple[FloatArray, slice]:
    array = np.asarray(observations)
    if array.ndim < 1 or array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("observations must be a numeric array with a feature dimension")
    result = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("observations must be finite")
    layout = observation_layout(
        use_agent_id=use_agent_id,
        num_agents=num_agents,
        num_targets=num_targets,
    )
    expected_dim = max(feature_slice.stop for feature_slice in layout.values())
    if result.shape[-1] != expected_dim:
        raise ValueError(
            f"observation final dimension must be {expected_dim}; got {result.shape[-1]}"
        )
    return result, layout["target_features"]


def permute_target_blocks(
    observations: ArrayLike,
    permutation: Sequence[int] | ArrayLike,
    *,
    num_agents: int,
    num_targets: int,
    use_agent_id: bool,
) -> FloatArray:
    """对任意 leading shape 使用同一个完整 target-block 排列。"""

    checked_permutation = validate_target_permutation(
        permutation,
        num_targets=num_targets,
    )
    result, target_slice = _checked_observations(
        observations,
        num_agents=num_agents,
        num_targets=num_targets,
        use_agent_id=use_agent_id,
    )
    target_blocks = result[..., target_slice].reshape(
        *result.shape[:-1],
        num_targets,
        TARGET_FEATURE_SIZE,
    )
    result[..., target_slice] = target_blocks[..., checked_permutation, :].reshape(
        *result.shape[:-1],
        num_targets * TARGET_FEATURE_SIZE,
    )
    return result


def permute_target_blocks_by_environment(
    observations: ArrayLike,
    permutations: ArrayLike,
    *,
    num_agents: int,
    num_targets: int,
    use_agent_id: bool,
) -> FloatArray:
    """为 ``[E,N,D]`` 中每个 environment 使用独立 target 排列。"""

    result, target_slice = _checked_observations(
        observations,
        num_agents=num_agents,
        num_targets=num_targets,
        use_agent_id=use_agent_id,
    )
    if result.ndim != 3 or result.shape[1] != num_agents:
        raise ValueError("observations must have shape [num_envs,num_agents,observation_dim]")
    raw_permutations = np.asarray(permutations)
    expected_shape = (result.shape[0], num_targets)
    if raw_permutations.shape != expected_shape:
        raise ValueError(f"permutations must have shape {expected_shape}")
    checked = np.stack(
        [validate_target_permutation(item, num_targets=num_targets) for item in raw_permutations]
    )
    blocks = result[..., target_slice].reshape(
        result.shape[0],
        num_agents,
        num_targets,
        TARGET_FEATURE_SIZE,
    )
    indices = np.broadcast_to(
        checked[:, None, :, None],
        blocks.shape,
    )
    result[..., target_slice] = np.take_along_axis(blocks, indices, axis=2).reshape(
        result.shape[0],
        num_agents,
        num_targets * TARGET_FEATURE_SIZE,
    )
    return result


def sample_target_permutations(
    generator: np.random.Generator,
    *,
    num_permutations: int,
    num_targets: int,
) -> IntArray:
    """独立采样一组排列；允许不同 environment 偶然取得同一排列。"""

    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be numpy.random.Generator")
    for name, value in (
        ("num_permutations", num_permutations),
        ("num_targets", num_targets),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    return np.stack(
        [generator.permutation(int(num_targets)) for _ in range(int(num_permutations))]
    ).astype(np.int64, copy=False)


def unique_random_target_permutations(
    generator: np.random.Generator,
    *,
    count: int,
    num_targets: int,
    excluded: Sequence[Sequence[int]] = (),
) -> tuple[tuple[int, ...], ...]:
    """生成不与给定集合重复的固定数量随机排列。"""

    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError("count must be an integer")
    requested = int(count)
    if requested <= 0:
        raise ValueError("count must be positive")
    excluded_set = {
        tuple(validate_target_permutation(item, num_targets=num_targets).tolist())
        for item in excluded
    }
    if requested > factorial(num_targets) - len(excluded_set):
        raise ValueError("requested unique permutations exceed the available space")
    selected: list[tuple[int, ...]] = []
    seen = set(excluded_set)
    while len(selected) < requested:
        candidate = tuple(generator.permutation(num_targets).tolist())
        if candidate in seen:
            continue
        seen.add(candidate)
        selected.append(candidate)
    return tuple(selected)


__all__ = [
    "permute_target_blocks",
    "permute_target_blocks_by_environment",
    "sample_target_permutations",
    "unique_random_target_permutations",
    "validate_target_permutation",
]
