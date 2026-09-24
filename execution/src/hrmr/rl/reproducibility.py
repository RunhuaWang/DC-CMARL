"""模型初始化与 checkpoint 审计使用的稳定 SHA-256 工具。"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def _length_prefixed(hasher: Any, value: bytes) -> None:
    hasher.update(struct.pack("<Q", len(value)))
    hasher.update(value)


def module_state_sha256(module: torch.nn.Module) -> str:
    """稳定哈希 module state，包括 key、dtype、shape 与 CPU tensor bytes。"""

    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    state = module.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("module.state_dict() must return a mapping")
    hasher = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry {key!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        _length_prefixed(hasher, key.encode("utf-8"))
        _length_prefixed(hasher, str(tensor.dtype).encode("ascii"))
        shape = np.asarray(tensor.shape, dtype="<i8")
        _length_prefixed(hasher, shape.tobytes(order="C"))
        _length_prefixed(hasher, tensor.numpy().tobytes(order="C"))
    return hasher.hexdigest()


def torch_rng_state_sha256() -> str:
    """返回当前 CPU Torch RNG state 的 SHA-256。"""

    state = torch.get_rng_state().detach().cpu().contiguous()
    return hashlib.sha256(state.numpy().tobytes(order="C")).hexdigest()


__all__ = ["module_state_sha256", "torch_rng_state_sha256"]
