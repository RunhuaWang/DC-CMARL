"""PyTorch 设备选择与可复现随机种子工具。"""

from __future__ import annotations

import random
from numbers import Integral

import numpy as np
import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """解析计算设备；``None``/``"auto"`` 按 CUDA、MPS、CPU 顺序选择。

    显式请求不可用的加速器时抛出异常，避免训练静默落回 CPU 后产生难以
    解释的性能差异。
    """

    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid torch device: {device!r}") from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return resolved


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """设置 Python、NumPy 与 PyTorch 随机流。

    ``deterministic=True`` 会请求 PyTorch deterministic algorithms。调用方
    应了解个别算子或设备可能因此抛出不支持异常。
    """

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    checked_seed = int(seed)
    if checked_seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(checked_seed)
    np.random.seed(checked_seed)
    torch.manual_seed(checked_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(checked_seed)
    torch.use_deterministic_algorithms(bool(deterministic))


# 兼容更短、直观的调用名。
select_device = resolve_device
set_random_seed = seed_everything


__all__ = [
    "resolve_device",
    "seed_everything",
    "select_device",
    "set_random_seed",
]
