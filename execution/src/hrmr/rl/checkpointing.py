"""Phase 2 fixed-λ 训练 checkpoint 的原子保存与完整恢复。"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class CheckpointMetadata:
    """从 checkpoint 恢复的非网络训练状态。"""

    format_version: int
    environment_steps: int
    fixed_lambda: float
    rho_reward: float
    rho_cost: float
    config: dict[str, Any]
    path: Path

    @property
    def rho_r(self) -> float:
        return self.rho_reward

    @property
    def rho_c(self) -> float:
        return self.rho_cost


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - train extra 提供 torch
        raise RuntimeError(
            "checkpointing requires PyTorch; install the project's train extra"
        ) from exc
    return torch


def _finite_real(value: Real, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _environment_steps(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("environment_steps must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("environment_steps must be non-negative")
    return result


def _config_payload(value: Any) -> Any:
    """递归转换配置，避开 MappingProxyType 等不可 pickle 的内部容器。"""

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _config_payload(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _config_payload(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return _config_payload(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_config_payload(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _config_payload(to_dict())
    raise TypeError(f"config contains unsupported value of type {type(value).__qualname__}")


def normalize_checkpoint_config(config: Any) -> dict[str, Any]:
    """返回可序列化、可稳定比较的配置字典。"""

    normalized = _config_payload(config)
    if not isinstance(normalized, dict):
        raise TypeError("config must normalize to a mapping or dataclass")
    return normalized


def capture_rng_states() -> dict[str, Any]:
    """捕获 Python、NumPy legacy global RNG 与 Torch CPU/CUDA/MPS 状态。"""

    torch = _require_torch()
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "torch_mps": None,
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    mps = getattr(torch, "mps", None)
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_is_built = getattr(mps_backend, "is_built", None)
    get_mps_rng_state = getattr(mps, "get_rng_state", None)
    if callable(get_mps_rng_state) and (not callable(mps_is_built) or bool(mps_is_built())):
        states["torch_mps"] = get_mps_rng_state()
    return states


def restore_rng_states(states: Mapping[str, Any]) -> None:
    """恢复 :func:`capture_rng_states` 保存的随机流。"""

    torch = _require_torch()
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "torch_mps"}
    missing = required - set(states)
    if missing:
        raise ValueError(f"checkpoint RNG state is missing keys: {sorted(missing)}")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    cuda_states = states["torch_cuda"]
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
    mps_state = states["torch_mps"]
    set_mps_rng_state = getattr(getattr(torch, "mps", None), "set_rng_state", None)
    if mps_state is not None and callable(set_mps_rng_state):
        set_mps_rng_state(mps_state)


def _state_dict(owner: Any, *, name: str) -> Mapping[str, Any]:
    method = getattr(owner, "state_dict", None)
    if not callable(method):
        raise TypeError(f"{name} must provide state_dict()")
    state = method()
    if not isinstance(state, Mapping):
        raise TypeError(f"{name}.state_dict() must return a mapping")
    return state


def save_checkpoint(
    path: str | Path,
    *,
    actor: Any,
    reward_critic: Any,
    cost_critic: Any,
    actor_optimizer: Any,
    reward_critic_optimizer: Any,
    cost_critic_optimizer: Any,
    rho_reward: float,
    rho_cost: float,
    environment_steps: int,
    config: Any,
    fixed_lambda: float,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """原子保存完整训练状态，成功时返回最终 checkpoint 路径。"""

    torch = _require_torch()
    checkpoint_path = Path(path)
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise IsADirectoryError(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    extras = {} if extra is None else dict(extra)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "models": {
            "actor": _state_dict(actor, name="actor"),
            "reward_critic": _state_dict(reward_critic, name="reward_critic"),
            "cost_critic": _state_dict(cost_critic, name="cost_critic"),
        },
        "optimizers": {
            "actor": _state_dict(actor_optimizer, name="actor_optimizer"),
            "reward_critic": _state_dict(
                reward_critic_optimizer,
                name="reward_critic_optimizer",
            ),
            "cost_critic": _state_dict(
                cost_critic_optimizer,
                name="cost_critic_optimizer",
            ),
        },
        "module_training_modes": {
            "actor": bool(getattr(actor, "training", True)),
            "reward_critic": bool(getattr(reward_critic, "training", True)),
            "cost_critic": bool(getattr(cost_critic, "training", True)),
        },
        "rho_reward": _finite_real(
            rho_reward,
            name="rho_reward",
            nonnegative=True,
        ),
        "rho_cost": _finite_real(
            rho_cost,
            name="rho_cost",
            nonnegative=True,
        ),
        "environment_steps": _environment_steps(environment_steps),
        "fixed_lambda": _finite_real(
            fixed_lambda,
            name="fixed_lambda",
            nonnegative=True,
        ),
        "config": normalize_checkpoint_config(config),
        "rng_states": capture_rng_states(),
        "extra": extras,
    }

    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return checkpoint_path


def _load_torch_payload(path: Path, map_location: Any) -> Any:
    torch = _require_torch()
    # PyTorch 2.6 默认 weights_only=True；本地可信 checkpoint 还含 Python/NumPy RNG。
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - 兼容较旧 PyTorch
        return torch.load(path, map_location=map_location)


def _load_state(owner: Any, state: Mapping[str, Any], *, name: str, strict: bool) -> None:
    method = getattr(owner, "load_state_dict", None)
    if not callable(method):
        raise TypeError(f"{name} must provide load_state_dict()")
    if name.endswith("optimizer"):
        method(state)
    else:
        method(state, strict=strict)


def _restore_training_mode(module: Any, training: bool) -> None:
    method = getattr(module, "train", None)
    if callable(method):
        method(training)


def load_checkpoint(
    path: str | Path,
    *,
    actor: Any,
    reward_critic: Any,
    cost_critic: Any,
    actor_optimizer: Any,
    reward_critic_optimizer: Any,
    cost_critic_optimizer: Any,
    map_location: Any = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
    expected_fixed_lambda: float | None = None,
    expected_config: Any | None = None,
) -> tuple[CheckpointMetadata, dict[str, Any]]:
    """恢复网络、optimizers 和 RNG，并返回 metadata 与 trainer ``extra``。"""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = _load_torch_payload(checkpoint_path, map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    required = {
        "format_version",
        "models",
        "optimizers",
        "module_training_modes",
        "rho_reward",
        "rho_cost",
        "environment_steps",
        "fixed_lambda",
        "config",
        "rng_states",
        "extra",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    version = int(payload["format_version"])
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )

    fixed_lambda = _finite_real(
        payload["fixed_lambda"],
        name="checkpoint fixed_lambda",
        nonnegative=True,
    )
    raw_config = payload["config"]
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint config must be a dictionary")
    config = normalize_checkpoint_config(raw_config)
    if expected_fixed_lambda is not None:
        expected_lambda = _finite_real(
            expected_fixed_lambda,
            name="expected_fixed_lambda",
            nonnegative=True,
        )
        if fixed_lambda != expected_lambda:
            raise ValueError(
                f"fixed lambda mismatch: checkpoint={fixed_lambda}, expected={expected_lambda}"
            )
    if expected_config is not None:
        normalized_expected = normalize_checkpoint_config(expected_config)
        if config != normalized_expected:
            raise ValueError("checkpoint config does not match expected_config")

    models = payload["models"]
    optimizers = payload["optimizers"]
    modes = payload["module_training_modes"]
    if not all(isinstance(item, Mapping) for item in (models, optimizers, modes)):
        raise ValueError("checkpoint model/optimizer metadata must be mappings")
    for key in ("actor", "reward_critic", "cost_critic"):
        if key not in models or key not in modes:
            raise ValueError(f"checkpoint is missing model state {key!r}")
    for key in ("actor", "reward_critic", "cost_critic"):
        if key not in optimizers:
            raise ValueError(f"checkpoint is missing optimizer state {key!r}")

    _load_state(actor, models["actor"], name="actor", strict=strict)
    _load_state(
        reward_critic,
        models["reward_critic"],
        name="reward_critic",
        strict=strict,
    )
    _load_state(cost_critic, models["cost_critic"], name="cost_critic", strict=strict)
    _load_state(
        actor_optimizer,
        optimizers["actor"],
        name="actor_optimizer",
        strict=strict,
    )
    _load_state(
        reward_critic_optimizer,
        optimizers["reward_critic"],
        name="reward_critic_optimizer",
        strict=strict,
    )
    _load_state(
        cost_critic_optimizer,
        optimizers["cost_critic"],
        name="cost_critic_optimizer",
        strict=strict,
    )
    _restore_training_mode(actor, bool(modes["actor"]))
    _restore_training_mode(reward_critic, bool(modes["reward_critic"]))
    _restore_training_mode(cost_critic, bool(modes["cost_critic"]))
    if restore_rng:
        rng_states = payload["rng_states"]
        if not isinstance(rng_states, Mapping):
            raise ValueError("checkpoint rng_states must be a mapping")
        restore_rng_states(rng_states)

    metadata = CheckpointMetadata(
        format_version=version,
        environment_steps=_environment_steps(payload["environment_steps"]),
        fixed_lambda=fixed_lambda,
        rho_reward=_finite_real(
            payload["rho_reward"],
            name="checkpoint rho_reward",
            nonnegative=True,
        ),
        rho_cost=_finite_real(
            payload["rho_cost"],
            name="checkpoint rho_cost",
            nonnegative=True,
        ),
        config=dict(config),
        path=checkpoint_path,
    )
    extra = payload["extra"]
    if not isinstance(extra, Mapping):
        raise ValueError("checkpoint extra must be a mapping")
    return metadata, dict(extra)


save_training_checkpoint = save_checkpoint
load_training_checkpoint = load_checkpoint


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointMetadata",
    "capture_rng_states",
    "load_checkpoint",
    "load_training_checkpoint",
    "normalize_checkpoint_config",
    "restore_rng_states",
    "save_checkpoint",
    "save_training_checkpoint",
]
