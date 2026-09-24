"""长期平均 reward/cost 的首 rollout 初始化与 EMA 跟踪。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from numbers import Real
from typing import Any

import torch


def _finite_scalar(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    checked = float(value)
    if not isfinite(checked):
        raise ValueError(f"{name} must be finite")
    return checked


def rollout_mean(values: torch.Tensor | Iterable[float] | float) -> float:
    """把一个非空 rollout 的逐步信号聚合为有限标量均值。"""

    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            raise ValueError("rollout values must not be empty")
        if not torch.is_floating_point(values):
            values = values.to(dtype=torch.get_default_dtype())
        detached = values.detach()
        if not bool(torch.isfinite(detached).all().item()):
            raise ValueError("rollout values must be finite")
        return float(detached.mean().item())
    if isinstance(values, Real) and not isinstance(values, bool):
        return _finite_scalar(values, name="rollout value")
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise TypeError("rollout values must be a tensor, scalar, or iterable") from exc
    if not materialized:
        raise ValueError("rollout values must not be empty")
    checked = tuple(
        _finite_scalar(value, name=f"rollout values[{index}]")
        for index, value in enumerate(materialized)
    )
    return sum(checked) / len(checked)


def exponential_moving_average(previous: float, observed: float, alpha: float) -> float:
    """计算 ``(1-alpha)*previous + alpha*observed``。"""

    old = _finite_scalar(previous, name="previous")
    new = _finite_scalar(observed, name="observed")
    weight = _finite_scalar(alpha, name="alpha")
    if not 0.0 < weight <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    return (1.0 - weight) * old + weight * new


class AverageRateEstimator:
    """用 rollout mean 跟踪一个长期平均率。

    第一次 :meth:`update` 直接把完整 rollout 的均值作为估计；从第二次开始
    才使用 EMA。这避免人为的零初值在训练早期引入偏差。
    """

    def __init__(self, ema_alpha: float = 0.05, *, alpha: float | None = None) -> None:
        if alpha is not None:
            if ema_alpha != 0.05:
                raise ValueError("specify only one of ema_alpha and alpha")
            ema_alpha = alpha
        checked_alpha = _finite_scalar(ema_alpha, name="ema_alpha")
        if not 0.0 < checked_alpha <= 1.0:
            raise ValueError("ema_alpha must lie in (0, 1]")
        self.ema_alpha = checked_alpha
        self._value: float | None = None
        self._num_updates = 0

    @property
    def initialized(self) -> bool:
        return self._value is not None

    @property
    def num_updates(self) -> int:
        return self._num_updates

    @property
    def value(self) -> float:
        """返回当前估计；首次 update 前拒绝提供虚假的零估计。"""

        if self._value is None:
            raise RuntimeError("average rate has not been initialized by a rollout")
        return self._value

    def update(self, values: torch.Tensor | Iterable[float] | float) -> float:
        """用一个 rollout 的均值直接初始化或执行一次 EMA 更新。"""

        observed = rollout_mean(values)
        if self._value is None:
            self._value = observed
        else:
            self._value = exponential_moving_average(
                self._value,
                observed,
                self.ema_alpha,
            )
        self._num_updates += 1
        return self._value

    def reset(self) -> None:
        """清除估计，使下一 rollout 再次采用直接初始化。"""

        self._value = None
        self._num_updates = 0

    def state_dict(self) -> dict[str, float | int | None]:
        """返回可安全写入 checkpoint 的最小状态。"""

        return {
            "ema_alpha": self.ema_alpha,
            "value": self._value,
            "num_updates": self._num_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """恢复由 :meth:`state_dict` 生成的状态并严格校验。"""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if set(state) != {"ema_alpha", "value", "num_updates"}:
            raise ValueError("average-rate state has unexpected fields")
        alpha = _finite_scalar(state["ema_alpha"], name="state ema_alpha")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("state ema_alpha must lie in (0, 1]")
        updates = state["num_updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("state num_updates must be a non-negative integer")
        raw_value = state["value"]
        value = None if raw_value is None else _finite_scalar(raw_value, name="state value")
        if (updates == 0) != (value is None):
            raise ValueError("state value and num_updates are inconsistent")
        self.ema_alpha = alpha
        self._value = value
        self._num_updates = updates


AverageRateTracker = AverageRateEstimator


__all__ = [
    "AverageRateEstimator",
    "AverageRateTracker",
    "exponential_moving_average",
    "rollout_mean",
]
