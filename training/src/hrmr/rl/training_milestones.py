"""不影响训练更新的 rollout 行为里程碑统计。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.rl.analytic_modes import analytic_mode, instantaneous_optimal_mode_mask

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


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


def _environment_discontinuity_mask(value: ArrayLike, *, num_envs: int) -> BoolArray:
    """验证人工状态跳转影响的 environment slots。"""

    raw = np.asarray(value)
    if raw.shape != (num_envs,):
        raise ValueError(f"environment_mask must have shape ({num_envs},)")
    if raw.dtype.kind != "b":
        raise TypeError("environment_mask must contain booleans")
    return raw.astype(np.bool_, copy=True)


def _hazardous_indices(values: Sequence[int], *, num_targets: int) -> tuple[int, ...]:
    indices = tuple(_nonnegative_integer(value, name="hazardous index") for value in values)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("hazardous_indices must be non-empty and unique")
    if any(index >= num_targets for index in indices):
        raise ValueError("hazardous index lies outside the target dimension")
    return indices


def _all_hazardous_mask(
    next_occupancies: ArrayLike,
    hazardous_indices: Sequence[int],
    *,
    num_envs: int,
) -> BoolArray:
    raw = np.asarray(next_occupancies)
    if raw.ndim != 3 or raw.shape[1] != num_envs or raw.shape[0] == 0 or raw.shape[2] == 0:
        raise ValueError("next_occupancies must have shape [T,E,M] with positive dimensions")
    if raw.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError("next_occupancies must contain binary numeric values")
    if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("next_occupancies entries must be finite and binary")
    indices = _hazardous_indices(hazardous_indices, num_targets=raw.shape[2])
    return np.all(raw[..., list(indices)] == 1, axis=-1)


class AllHazardousTrainingTracker:
    """跨 rollout 跟踪首次 all-H、保持率与同环境连续性。

    一个 vector transition 同步产生 ``E`` 个 environment samples。首次命中步
    记为该同步 transition 完成后的累计 environment-step 边界，因而不会用
    environment index 人为区分同时发生的事件。首次命中的整个 vector step
    都进入 since-first 分母。
    """

    _FORMAT_VERSION = 1

    def __init__(self, num_envs: int) -> None:
        self.num_envs = _positive_integer(num_envs, name="num_envs")
        self.processed_environment_steps = 0
        self.first_all_hazardous_environment_step: int | None = None
        self.all_hazardous_hits_since_first = 0
        self.all_hazardous_samples_since_first = 0
        self.retained_next_transition_count = 0
        self.retention_opportunity_count = 0
        self.longest_consecutive_all_hazardous_vector_transitions = 0
        self._retention_discontinuity_count = 0
        self._previous_all_hazardous = np.zeros(self.num_envs, dtype=np.bool_)
        self._current_streaks = np.zeros(self.num_envs, dtype=np.int64)

    def break_continuity(self, environment_mask: ArrayLike) -> None:
        """在人工边界跳转处关闭跨 rollout retention/streak 链。

        该操作不消费 environment step，也不改变历史 hit/rate；它只保证下一次
        :meth:`update` 不会把人工注入前后的两个状态误当成相邻 transition。
        """

        mask = _environment_discontinuity_mask(
            environment_mask,
            num_envs=self.num_envs,
        )
        self._retention_discontinuity_count += int(
            np.count_nonzero(self._previous_all_hazardous & mask)
        )
        self._previous_all_hazardous[mask] = False
        self._current_streaks[mask] = 0

    def diagnostics(self) -> dict[str, int | float]:
        """返回可直接写入 training CSV 的有限标量。"""

        since_first_rate = (
            self.all_hazardous_hits_since_first / self.all_hazardous_samples_since_first
            if self.all_hazardous_samples_since_first
            else 0.0
        )
        retention_rate = (
            self.retained_next_transition_count / self.retention_opportunity_count
            if self.retention_opportunity_count
            else 0.0
        )
        return {
            "first_all_hazardous_environment_step": (
                -1
                if self.first_all_hazardous_environment_step is None
                else self.first_all_hazardous_environment_step
            ),
            "all_hazardous_mode_rate_since_first_hit": float(since_first_rate),
            "all_hazardous_samples_since_first_hit": self.all_hazardous_samples_since_first,
            "all_hazardous_retained_next_transition_count": (self.retained_next_transition_count),
            "all_hazardous_retention_opportunity_count": self.retention_opportunity_count,
            "all_hazardous_next_transition_retention_rate": float(retention_rate),
            "longest_consecutive_all_hazardous_vector_transitions": (
                self.longest_consecutive_all_hazardous_vector_transitions
            ),
        }

    def update(
        self,
        next_occupancies: ArrayLike,
        hazardous_indices: Sequence[int],
        *,
        rollout_start_environment_steps: int,
    ) -> dict[str, int | float]:
        """消费一个连续 rollout，并返回更新后的累计诊断。"""

        start = _nonnegative_integer(
            rollout_start_environment_steps,
            name="rollout_start_environment_steps",
        )
        if start != self.processed_environment_steps:
            raise ValueError(
                "rollout_start_environment_steps must equal processed_environment_steps"
            )
        mask = _all_hazardous_mask(
            next_occupancies,
            hazardous_indices,
            num_envs=self.num_envs,
        )

        if self.first_all_hazardous_environment_step is None:
            vector_hits = np.flatnonzero(np.any(mask, axis=1))
            if len(vector_hits):
                first_time_index = int(vector_hits[0])
                self.first_all_hazardous_environment_step = (
                    start + (first_time_index + 1) * self.num_envs
                )
                selected = mask[first_time_index:]
                self.all_hazardous_hits_since_first += int(np.count_nonzero(selected))
                self.all_hazardous_samples_since_first += int(selected.size)
        else:
            self.all_hazardous_hits_since_first += int(np.count_nonzero(mask))
            self.all_hazardous_samples_since_first += int(mask.size)

        for current in mask:
            self.retention_opportunity_count += int(np.count_nonzero(self._previous_all_hazardous))
            self.retained_next_transition_count += int(
                np.count_nonzero(self._previous_all_hazardous & current)
            )
            self._current_streaks = np.where(current, self._current_streaks + 1, 0)
            self.longest_consecutive_all_hazardous_vector_transitions = max(
                self.longest_consecutive_all_hazardous_vector_transitions,
                int(np.max(self._current_streaks)),
            )
            self._previous_all_hazardous = current.copy()

        self.processed_environment_steps = start + mask.shape[0] * self.num_envs
        return self.diagnostics()

    def state_dict(self) -> dict[str, Any]:
        """返回 checkpoint 可序列化状态。"""

        state = {
            "format_version": self._FORMAT_VERSION,
            "num_envs": self.num_envs,
            "processed_environment_steps": self.processed_environment_steps,
            "first_all_hazardous_environment_step": (self.first_all_hazardous_environment_step),
            "all_hazardous_hits_since_first": self.all_hazardous_hits_since_first,
            "all_hazardous_samples_since_first": self.all_hazardous_samples_since_first,
            "retained_next_transition_count": self.retained_next_transition_count,
            "retention_opportunity_count": self.retention_opportunity_count,
            "longest_consecutive_all_hazardous_vector_transitions": (
                self.longest_consecutive_all_hazardous_vector_transitions
            ),
            "previous_all_hazardous": self._previous_all_hazardous.tolist(),
            "current_streaks": self._current_streaks.tolist(),
        }
        # 保持未使用人工跳转时的既有 checkpoint payload 完全不变；该可选字段
        # 只用于严格恢复 diagnostic retention invariant。
        if self._retention_discontinuity_count:
            state["retention_discontinuity_count"] = self._retention_discontinuity_count
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """严格恢复累计统计，以保持跨 rollout/checkpoint 连续性。"""

        if not isinstance(state, Mapping):
            raise TypeError("all-hazardous tracker state must be a mapping")
        if _nonnegative_integer(state.get("format_version"), name="format_version") != 1:
            raise ValueError("unsupported all-hazardous tracker state format")
        if _positive_integer(state.get("num_envs"), name="num_envs") != self.num_envs:
            raise ValueError("all-hazardous tracker num_envs mismatch")

        processed = _nonnegative_integer(
            state.get("processed_environment_steps"),
            name="processed_environment_steps",
        )
        if processed % self.num_envs:
            raise ValueError("processed_environment_steps must align to vector transitions")
        raw_first = state.get("first_all_hazardous_environment_step")
        first = (
            None
            if raw_first is None
            else _positive_integer(raw_first, name="first_all_hazardous_environment_step")
        )
        if first is not None and (first > processed or first % self.num_envs):
            raise ValueError("first all-hazardous step is inconsistent with processed steps")
        hits = _nonnegative_integer(
            state.get("all_hazardous_hits_since_first"),
            name="all_hazardous_hits_since_first",
        )
        samples = _nonnegative_integer(
            state.get("all_hazardous_samples_since_first"),
            name="all_hazardous_samples_since_first",
        )
        retained = _nonnegative_integer(
            state.get("retained_next_transition_count"),
            name="retained_next_transition_count",
        )
        opportunities = _nonnegative_integer(
            state.get("retention_opportunity_count"),
            name="retention_opportunity_count",
        )
        discontinuities = _nonnegative_integer(
            state.get("retention_discontinuity_count", 0),
            name="retention_discontinuity_count",
        )
        longest = _nonnegative_integer(
            state.get("longest_consecutive_all_hazardous_vector_transitions"),
            name="longest_consecutive_all_hazardous_vector_transitions",
        )
        previous = np.asarray(state.get("previous_all_hazardous"))
        streaks = np.asarray(state.get("current_streaks"))
        if previous.shape != (self.num_envs,) or previous.dtype.kind != "b":
            raise ValueError("previous_all_hazardous must be a boolean vector of length num_envs")
        if streaks.shape != (self.num_envs,) or streaks.dtype.kind not in {"i", "u"}:
            raise ValueError("current_streaks must be an integer vector of length num_envs")
        checked_streaks = streaks.astype(np.int64, copy=True)
        if np.any(checked_streaks < 0):
            raise ValueError("current_streaks must be non-negative")
        if not np.array_equal(checked_streaks > 0, previous):
            raise ValueError("current_streaks must agree with previous_all_hazardous")
        if retained > opportunities or longest < int(np.max(checked_streaks)):
            raise ValueError("all-hazardous retention/streak state is inconsistent")
        if opportunities + discontinuities != hits - int(np.count_nonzero(previous)):
            raise ValueError("all-hazardous retention/discontinuity counts are inconsistent")
        if first is None:
            if any((hits, samples, retained, opportunities, discontinuities, longest)) or np.any(
                previous
            ):
                raise ValueError("pre-hit all-hazardous tracker state must contain zero statistics")
        else:
            expected_samples = processed - first + self.num_envs
            if samples != expected_samples or hits <= 0 or hits > samples or longest <= 0:
                raise ValueError("post-hit all-hazardous tracker state is inconsistent")

        self.processed_environment_steps = processed
        self.first_all_hazardous_environment_step = first
        self.all_hazardous_hits_since_first = hits
        self.all_hazardous_samples_since_first = samples
        self.retained_next_transition_count = retained
        self.retention_opportunity_count = opportunities
        self._retention_discontinuity_count = discontinuities
        self.longest_consecutive_all_hazardous_vector_transitions = longest
        self._previous_all_hazardous = previous.astype(np.bool_, copy=True)
        self._current_streaks = checked_streaks


def _supported_fixed_lambda(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("fixed_lambda must be a real number")
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError("fixed_lambda must be finite")
    return analytic_mode(checked).fixed_lambda


def _target_id_tuple(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("target_ids must be a sequence of target identifiers")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError("target_ids must be a sequence of target identifiers") from exc
    if not result or any(not isinstance(item, str) for item in result):
        raise TypeError("target_ids must contain non-empty string identifiers")
    if any(not item for item in result):
        raise ValueError("target_ids must contain non-empty string identifiers")
    return result


class OperatingModeTrainingTracker:
    """跨 rollout 跟踪给定 fixed-λ 解析最优 mode 的瞬时命中与保持。

    Mode 判定完全委托给 :func:`instantaneous_optimal_mode_mask`。因此 safe target
    只按数量区分；相邻状态即使使用不同的等价 safe subset，只要各自仍满足解析
    allocation，就会计为同一环境的一次成功保持。

    首次命中步是第一个命中同步 transition 完成后的累计 environment-step 边界。
    与旧 all-H tracker 一致，同一 vector transition 内的 environments 不进行人为排序，
    且首次命中的完整 vector transition 都进入 since-first 分母。
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
        self.fixed_lambda = _supported_fixed_lambda(fixed_lambda)
        self.target_ids = _target_id_tuple(target_ids)
        # 统一复用正式瞬时 mode 判定，同时在构造时立即校验 target order。
        instantaneous_optimal_mode_mask(
            np.zeros((1, len(self.target_ids)), dtype=np.int8),
            self.target_ids,
            self.fixed_lambda,
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
        """在人工状态注入处切断跨边界 mode retention/streak。"""

        mask = _environment_discontinuity_mask(
            environment_mask,
            num_envs=self.num_envs,
        )
        self._retention_discontinuity_count += int(
            np.count_nonzero(self._previous_instantaneous_optimal_mode & mask)
        )
        self._previous_instantaneous_optimal_mode[mask] = False
        self._current_streaks[mask] = 0

    def diagnostics(self) -> dict[str, int | float]:
        """返回可直接写入训练日志的有限标量。"""

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
        """消费连续 rollout 的 post-transition occupancies 并更新累计诊断。"""

        start = _nonnegative_integer(
            rollout_start_environment_steps,
            name="rollout_start_environment_steps",
        )
        if start != self.processed_environment_steps:
            raise ValueError(
                "rollout_start_environment_steps must equal processed_environment_steps"
            )
        raw = np.asarray(next_occupancies)
        expected_shape_suffix = (self.num_envs, len(self.target_ids))
        if raw.ndim != 3 or raw.shape[0] == 0 or raw.shape[1:] != expected_shape_suffix:
            raise ValueError(
                "next_occupancies must have shape [T,E,M] with positive T and configured E/M"
            )
        mask = instantaneous_optimal_mode_mask(
            raw,
            self.target_ids,
            self.fixed_lambda,
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
        """返回包含 mode 身份与跨 rollout 连续状态的严格 checkpoint payload。"""

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
        """严格恢复同一 fixed-λ、target order 与 vector width 的 tracker。"""

        if not isinstance(state, Mapping):
            raise TypeError("operating-mode tracker state must be a mapping")
        actual_keys = frozenset(state)
        required_keys = actual_keys - self._OPTIONAL_STATE_KEYS
        if required_keys != self._STATE_KEYS or not actual_keys.issubset(
            self._STATE_KEYS | self._OPTIONAL_STATE_KEYS
        ):
            missing = sorted(self._STATE_KEYS - required_keys)
            unexpected = sorted(actual_keys - self._STATE_KEYS - self._OPTIONAL_STATE_KEYS)
            raise ValueError(
                "operating-mode tracker state fields mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if _nonnegative_integer(state["format_version"], name="format_version") != 1:
            raise ValueError("unsupported operating-mode tracker state format")
        if _positive_integer(state["num_envs"], name="num_envs") != self.num_envs:
            raise ValueError("operating-mode tracker num_envs mismatch")
        if _supported_fixed_lambda(state["fixed_lambda"]) != self.fixed_lambda:
            raise ValueError("operating-mode tracker fixed_lambda mismatch")
        stored_target_ids = _target_id_tuple(state["target_ids"])
        if stored_target_ids != self.target_ids:
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
            raise ValueError(
                "previous_instantaneous_optimal_mode must be a boolean vector of length num_envs"
            )
        if streaks.shape != (self.num_envs,) or streaks.dtype.kind not in {"i", "u"}:
            raise ValueError("current_streaks must be an integer vector of length num_envs")
        checked_streaks = streaks.astype(np.int64, copy=True)
        if np.any(checked_streaks < 0):
            raise ValueError("current_streaks must be non-negative")
        if not np.array_equal(checked_streaks > 0, previous):
            raise ValueError("current_streaks must agree with previous optimal-mode mask")
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
                raise ValueError(
                    "pre-hit operating-mode tracker state must contain zero statistics"
                )
        else:
            expected_samples = processed - first + self.num_envs
            if samples != expected_samples or hits == 0 or longest == 0:
                raise ValueError("post-hit operating-mode tracker state is inconsistent")

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


__all__ = ["AllHazardousTrainingTracker", "OperatingModeTrainingTracker"]
