"""未归一化原参数实验的最优 operating-mode 训练里程碑。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .analytic_modes import analytic_mode, optimal_mode_mask


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _positive_integer(value: Any, *, name: str) -> int:
    checked = _nonnegative_integer(value, name=name)
    if checked == 0:
        raise ValueError(f"{name} must be positive")
    return checked


def _target_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("target_ids must be a sequence of target identifiers")
    result = tuple(values)
    if not result or any(not isinstance(item, str) for item in result):
        raise TypeError("target_ids must contain string identifiers")
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError("target_ids must be non-empty and unique")
    return result


class OriginalParametersOperatingModeTrainingTracker:
    """跨 rollout 跟踪五个 raw-λ 解析最优模式的命中与保持。

    该类不借用正式归一化实验的 λ 身份。checkpoint 中保存的是当前 raw
    scalarization 实际使用的 λ，因此恢复时不会出现隐式 λ 映射。
    """

    _FORMAT_VERSION = 1
    _STATE_KEYS = frozenset(
        {
            "format_version",
            "num_envs",
            "fixed_lambda",
            "target_ids",
            "processed_environment_steps",
            "first_instantaneous_optimal_mode_step",
            "instantaneous_optimal_mode_hits_since_first",
            "instantaneous_optimal_mode_samples_since_first",
            "instantaneous_optimal_mode_retained_next_transition_count",
            "instantaneous_optimal_mode_retention_opportunity_count",
            "longest_consecutive_instantaneous_optimal_mode_vector_transitions",
            "previous_instantaneous_optimal_mode",
            "current_streaks",
        }
    )
    _OPTIONAL_STATE_KEYS = frozenset({"retention_discontinuity_count"})

    def __init__(
        self,
        num_envs: int,
        fixed_lambda: float,
        target_ids: Sequence[str],
    ) -> None:
        self.num_envs = _positive_integer(num_envs, name="num_envs")
        self.fixed_lambda = analytic_mode(fixed_lambda).fixed_lambda
        self.target_ids = _target_ids(target_ids)
        # 让 analytic mask 在构造期立即验证 canonical target order。
        optimal_mode_mask(
            np.zeros((1, len(self.target_ids)), dtype=np.int8),
            self.target_ids,
            fixed_lambda=self.fixed_lambda,
        )
        self.processed_environment_steps = 0
        self.first_instantaneous_optimal_mode_step: int | None = None
        self.instantaneous_optimal_mode_hits_since_first = 0
        self.instantaneous_optimal_mode_samples_since_first = 0
        self.instantaneous_optimal_mode_retained_next_transition_count = 0
        self.instantaneous_optimal_mode_retention_opportunity_count = 0
        self.longest_consecutive_instantaneous_optimal_mode_vector_transitions = 0
        self._retention_discontinuity_count = 0
        self._previous_instantaneous_optimal_mode = np.zeros(
            self.num_envs,
            dtype=np.bool_,
        )
        self._current_streaks = np.zeros(self.num_envs, dtype=np.int64)

    def break_continuity(self, environment_mask: ArrayLike) -> None:
        """切断人工状态注入前后的 retention/streak 连续性。"""

        raw = np.asarray(environment_mask)
        if raw.shape != (self.num_envs,):
            raise ValueError(f"environment_mask must have shape ({self.num_envs},)")
        if raw.dtype.kind != "b":
            raise TypeError("environment_mask must contain booleans")
        mask = raw.astype(np.bool_, copy=False)
        self._retention_discontinuity_count += int(
            np.count_nonzero(self._previous_instantaneous_optimal_mode & mask)
        )
        self._previous_instantaneous_optimal_mode[mask] = False
        self._current_streaks[mask] = 0

    def diagnostics(self) -> dict[str, int | float]:
        """返回与正式训练日志兼容的累计标量。"""

        samples = self.instantaneous_optimal_mode_samples_since_first
        opportunities = self.instantaneous_optimal_mode_retention_opportunity_count
        since_first_rate = (
            self.instantaneous_optimal_mode_hits_since_first / samples if samples else 0.0
        )
        retention_rate = (
            self.instantaneous_optimal_mode_retained_next_transition_count / opportunities
            if opportunities
            else 0.0
        )
        return {
            "first_instantaneous_optimal_mode_step": (
                -1
                if self.first_instantaneous_optimal_mode_step is None
                else self.first_instantaneous_optimal_mode_step
            ),
            "instantaneous_optimal_mode_rate_since_first_hit": float(since_first_rate),
            "instantaneous_optimal_mode_samples_since_first_hit": samples,
            "instantaneous_optimal_mode_retained_next_transition_count": (
                self.instantaneous_optimal_mode_retained_next_transition_count
            ),
            "instantaneous_optimal_mode_retention_opportunity_count": opportunities,
            "instantaneous_optimal_mode_next_transition_retention_rate": float(retention_rate),
            "longest_consecutive_instantaneous_optimal_mode_vector_transitions": (
                self.longest_consecutive_instantaneous_optimal_mode_vector_transitions
            ),
        }

    def update(
        self,
        next_occupancies: ArrayLike,
        *,
        rollout_start_environment_steps: int,
    ) -> dict[str, int | float]:
        """消费一个连续 ``[T,E,M]`` rollout，并更新累计里程碑。"""

        start = _nonnegative_integer(
            rollout_start_environment_steps,
            name="rollout_start_environment_steps",
        )
        if start != self.processed_environment_steps:
            raise ValueError(
                "rollout_start_environment_steps must equal processed_environment_steps"
            )
        raw = np.asarray(next_occupancies)
        expected_suffix = (self.num_envs, len(self.target_ids))
        if raw.ndim != 3 or raw.shape[0] == 0 or raw.shape[1:] != expected_suffix:
            raise ValueError("next_occupancies must have shape [T,E,M] with configured E/M")
        mask = optimal_mode_mask(
            raw,
            self.target_ids,
            fixed_lambda=self.fixed_lambda,
        )

        if self.first_instantaneous_optimal_mode_step is None:
            vector_hits = np.flatnonzero(np.any(mask, axis=1))
            if len(vector_hits):
                first_time_index = int(vector_hits[0])
                self.first_instantaneous_optimal_mode_step = (
                    start + (first_time_index + 1) * self.num_envs
                )
                selected = mask[first_time_index:]
                self.instantaneous_optimal_mode_hits_since_first += int(np.count_nonzero(selected))
                self.instantaneous_optimal_mode_samples_since_first += int(selected.size)
        else:
            self.instantaneous_optimal_mode_hits_since_first += int(np.count_nonzero(mask))
            self.instantaneous_optimal_mode_samples_since_first += int(mask.size)

        for current in mask:
            previous = self._previous_instantaneous_optimal_mode
            self.instantaneous_optimal_mode_retention_opportunity_count += int(
                np.count_nonzero(previous)
            )
            self.instantaneous_optimal_mode_retained_next_transition_count += int(
                np.count_nonzero(previous & current)
            )
            self._current_streaks = np.where(current, self._current_streaks + 1, 0)
            self.longest_consecutive_instantaneous_optimal_mode_vector_transitions = max(
                self.longest_consecutive_instantaneous_optimal_mode_vector_transitions,
                int(np.max(self._current_streaks)),
            )
            self._previous_instantaneous_optimal_mode = current.copy()

        self.processed_environment_steps = start + mask.shape[0] * self.num_envs
        return self.diagnostics()

    def state_dict(self) -> dict[str, Any]:
        """返回包含实际 raw λ 与连续状态的 checkpoint payload。"""

        state = {
            "format_version": self._FORMAT_VERSION,
            "num_envs": self.num_envs,
            "fixed_lambda": self.fixed_lambda,
            "target_ids": list(self.target_ids),
            "processed_environment_steps": self.processed_environment_steps,
            "first_instantaneous_optimal_mode_step": (self.first_instantaneous_optimal_mode_step),
            "instantaneous_optimal_mode_hits_since_first": (
                self.instantaneous_optimal_mode_hits_since_first
            ),
            "instantaneous_optimal_mode_samples_since_first": (
                self.instantaneous_optimal_mode_samples_since_first
            ),
            "instantaneous_optimal_mode_retained_next_transition_count": (
                self.instantaneous_optimal_mode_retained_next_transition_count
            ),
            "instantaneous_optimal_mode_retention_opportunity_count": (
                self.instantaneous_optimal_mode_retention_opportunity_count
            ),
            "longest_consecutive_instantaneous_optimal_mode_vector_transitions": (
                self.longest_consecutive_instantaneous_optimal_mode_vector_transitions
            ),
            "previous_instantaneous_optimal_mode": (
                self._previous_instantaneous_optimal_mode.tolist()
            ),
            "current_streaks": self._current_streaks.tolist(),
        }
        if self._retention_discontinuity_count:
            state["retention_discontinuity_count"] = self._retention_discontinuity_count
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """严格恢复同一 raw λ、target order 与 vector width 的状态。"""

        if not isinstance(state, Mapping):
            raise TypeError("operating-mode tracker state must be a mapping")
        actual_keys = frozenset(state)
        required_keys = actual_keys - self._OPTIONAL_STATE_KEYS
        if required_keys != self._STATE_KEYS or not actual_keys.issubset(
            self._STATE_KEYS | self._OPTIONAL_STATE_KEYS
        ):
            raise ValueError("operating-mode tracker state fields mismatch")
        if _nonnegative_integer(state["format_version"], name="format_version") != 1:
            raise ValueError("unsupported operating-mode tracker state format")
        if _positive_integer(state["num_envs"], name="num_envs") != self.num_envs:
            raise ValueError("operating-mode tracker num_envs mismatch")
        if analytic_mode(state["fixed_lambda"]).fixed_lambda != self.fixed_lambda:
            raise ValueError("operating-mode tracker fixed_lambda mismatch")
        if _target_ids(state["target_ids"]) != self.target_ids:
            raise ValueError("operating-mode tracker target_ids mismatch")

        processed = _nonnegative_integer(
            state["processed_environment_steps"],
            name="processed_environment_steps",
        )
        if processed % self.num_envs:
            raise ValueError("processed_environment_steps must align to vector transitions")
        raw_first = state["first_instantaneous_optimal_mode_step"]
        first = (
            None
            if raw_first is None
            else _positive_integer(raw_first, name="first_instantaneous_optimal_mode_step")
        )
        if first is not None and (first > processed or first % self.num_envs):
            raise ValueError("first instantaneous optimal-mode step is inconsistent")
        hits = _nonnegative_integer(
            state["instantaneous_optimal_mode_hits_since_first"],
            name="instantaneous_optimal_mode_hits_since_first",
        )
        samples = _nonnegative_integer(
            state["instantaneous_optimal_mode_samples_since_first"],
            name="instantaneous_optimal_mode_samples_since_first",
        )
        retained = _nonnegative_integer(
            state["instantaneous_optimal_mode_retained_next_transition_count"],
            name="instantaneous_optimal_mode_retained_next_transition_count",
        )
        opportunities = _nonnegative_integer(
            state["instantaneous_optimal_mode_retention_opportunity_count"],
            name="instantaneous_optimal_mode_retention_opportunity_count",
        )
        discontinuities = _nonnegative_integer(
            state.get("retention_discontinuity_count", 0),
            name="retention_discontinuity_count",
        )
        longest = _nonnegative_integer(
            state["longest_consecutive_instantaneous_optimal_mode_vector_transitions"],
            name="longest_consecutive_instantaneous_optimal_mode_vector_transitions",
        )
        previous = np.asarray(state["previous_instantaneous_optimal_mode"])
        streaks = np.asarray(state["current_streaks"])
        if previous.shape != (self.num_envs,) or previous.dtype.kind != "b":
            raise ValueError("previous_instantaneous_optimal_mode has invalid shape/type")
        if streaks.shape != (self.num_envs,) or streaks.dtype.kind not in {"i", "u"}:
            raise ValueError("current_streaks has invalid shape/type")
        checked_streaks = streaks.astype(np.int64, copy=True)
        if np.any(checked_streaks < 0):
            raise ValueError("current_streaks must be non-negative")
        if not np.array_equal(checked_streaks > 0, previous):
            raise ValueError("current_streaks must agree with previous mode mask")
        if hits > samples or retained > opportunities:
            raise ValueError("operating-mode hit/retention counts are inconsistent")
        if longest < int(np.max(checked_streaks)):
            raise ValueError("operating-mode longest streak is inconsistent")
        if opportunities + discontinuities != hits - int(np.count_nonzero(previous)):
            raise ValueError("operating-mode retention opportunities are inconsistent")
        if first is None:
            if any((hits, samples, retained, opportunities, discontinuities, longest)) or np.any(
                previous
            ):
                raise ValueError("pre-hit operating-mode state must contain zero statistics")
        else:
            expected_samples = processed - first + self.num_envs
            if samples != expected_samples or hits == 0 or longest == 0:
                raise ValueError("post-hit operating-mode state is inconsistent")

        self.processed_environment_steps = processed
        self.first_instantaneous_optimal_mode_step = first
        self.instantaneous_optimal_mode_hits_since_first = hits
        self.instantaneous_optimal_mode_samples_since_first = samples
        self.instantaneous_optimal_mode_retained_next_transition_count = retained
        self.instantaneous_optimal_mode_retention_opportunity_count = opportunities
        self._retention_discontinuity_count = discontinuities
        self.longest_consecutive_instantaneous_optimal_mode_vector_transitions = longest
        self._previous_instantaneous_optimal_mode = previous.astype(np.bool_, copy=True)
        self._current_streaks = checked_streaks


__all__ = ["OriginalParametersOperatingModeTrainingTracker"]
