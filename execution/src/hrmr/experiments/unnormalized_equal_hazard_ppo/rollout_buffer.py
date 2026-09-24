"""raw reward/cost 专用 continuing rollout buffer。

字段与正式 :mod:`hrmr.rl.rollout_buffer` 一致，但只要求 reward/cost 有限且
非负，不施加单位上界。正式 ``RolloutBuffer`` 的 ``[0,1]`` 防线保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.constants import NUM_ACTIONS

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
CoverageArray = NDArray[np.int8]


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    checked = int(value)
    if checked <= 0:
        raise ValueError(f"{name} must be positive")
    return checked


def _float_array(value: ArrayLike, shape: tuple[int, ...], *, name: str) -> FloatArray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain real numeric values")
    checked = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain only finite values")
    return checked


def _actions_array(value: ArrayLike, shape: tuple[int, ...]) -> IntArray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"actions must have shape {shape}; got {array.shape}")
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError("actions must contain integers")
    checked = array.astype(np.int64, copy=True)
    if np.any((checked < 0) | (checked >= NUM_ACTIONS)):
        raise ValueError(f"actions must lie in [0, {NUM_ACTIONS - 1}]")
    return checked


def _coverage_array(value: ArrayLike, shape: tuple[int, ...], *, name: str) -> CoverageArray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"{name} must contain binary numeric values")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} entries must be finite and exactly 0 or 1")
    return array.astype(np.int8, copy=True)


def _validate_raw_signals(
    rewards: FloatArray,
    global_costs: FloatArray,
    local_costs: FloatArray,
    *,
    agent_axis: int,
) -> None:
    if np.any(rewards < 0.0):
        raise ValueError("rewards must be non-negative")
    if np.any(global_costs < 0.0):
        raise ValueError("global_costs must be non-negative")
    if np.any(local_costs < 0.0):
        raise ValueError("local_costs must be non-negative")
    if not np.allclose(
        global_costs,
        np.sum(local_costs, axis=agent_axis),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("global_costs must equal the sum of local_costs")


@dataclass(frozen=True)
class UnnormalizedRolloutBatch:
    """冻结的 ``[T,E,...]`` raw-signal rollout 快照。"""

    observations: FloatArray
    next_observations: FloatArray
    states: FloatArray
    next_states: FloatArray
    actions: IntArray
    log_probs: FloatArray
    rewards: FloatArray
    global_costs: FloatArray
    local_costs: FloatArray
    occupancies: CoverageArray
    agent_target_occupancies: CoverageArray
    next_occupancies: CoverageArray

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations)
        if observations.ndim != 4:
            raise ValueError("observations must have shape [T,E,N,D]")
        num_steps, num_envs, num_agents, observation_dim = observations.shape
        if min(num_steps, num_envs, num_agents, observation_dim) <= 0:
            raise ValueError("rollout batch dimensions must be positive")
        states = np.asarray(self.states)
        if states.ndim != 3 or states.shape[:2] != (num_steps, num_envs):
            raise ValueError("states must have shape [T,E,S]")
        state_dim = states.shape[2]
        if state_dim <= 0:
            raise ValueError("state feature dimension must be positive")
        occupancies = np.asarray(self.occupancies)
        if occupancies.ndim != 3 or occupancies.shape[:2] != (num_steps, num_envs):
            raise ValueError("occupancies must have shape [T,E,M]")
        num_targets = occupancies.shape[2]
        if num_targets <= 0:
            raise ValueError("occupancy target dimension must be positive")

        shapes = {
            "observations": (num_steps, num_envs, num_agents, observation_dim),
            "next_observations": (num_steps, num_envs, num_agents, observation_dim),
            "states": (num_steps, num_envs, state_dim),
            "next_states": (num_steps, num_envs, state_dim),
            "actions": (num_steps, num_envs, num_agents),
            "log_probs": (num_steps, num_envs, num_agents),
            "rewards": (num_steps, num_envs),
            "global_costs": (num_steps, num_envs),
            "local_costs": (num_steps, num_envs, num_agents),
            "occupancies": (num_steps, num_envs, num_targets),
            "agent_target_occupancies": (num_steps, num_envs, num_agents, num_targets),
            "next_occupancies": (num_steps, num_envs, num_targets),
        }
        for item in fields(self):
            raw = getattr(self, item.name)
            shape = shapes[item.name]
            if item.name == "actions":
                checked: np.ndarray = _actions_array(raw, shape)
            elif item.name in {"occupancies", "agent_target_occupancies", "next_occupancies"}:
                checked = _coverage_array(raw, shape, name=item.name)
            else:
                checked = _float_array(raw, shape, name=item.name)
            checked.setflags(write=False)
            object.__setattr__(self, item.name, checked)

        _validate_raw_signals(
            self.rewards,
            self.global_costs,
            self.local_costs,
            agent_axis=2,
        )
        if np.any(np.sum(self.agent_target_occupancies, axis=3) > 1):
            raise ValueError("each agent may occupy at most one target region")
        if not np.array_equal(
            np.any(self.agent_target_occupancies, axis=2).astype(np.int8),
            self.next_occupancies,
        ):
            raise ValueError("agent_target_occupancies union must equal next_occupancies")

    @property
    def num_steps(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.observations.shape[1])

    @property
    def num_agents(self) -> int:
        return int(self.observations.shape[2])

    @property
    def current_observations(self) -> FloatArray:
        return self.observations

    @property
    def current_states(self) -> FloatArray:
        return self.states

    @property
    def current_occupancies(self) -> CoverageArray:
        return self.occupancies

    @property
    def coverage(self) -> CoverageArray:
        return self.occupancies

    @property
    def next_coverage(self) -> CoverageArray:
        return self.next_occupancies

    @property
    def logprobs(self) -> FloatArray:
        return self.log_probs


class UnnormalizedRolloutBuffer:
    """预分配 raw-signal rollout 存储；rollout 边界不含 terminal 语义。"""

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        num_agents: int,
        observation_dim: int = 48,
        state_dim: int = 16,
        num_targets: int = 8,
    ) -> None:
        self.capacity = _positive_integer(capacity, name="capacity")
        self.num_envs = _positive_integer(num_envs, name="num_envs")
        self.num_agents = _positive_integer(num_agents, name="num_agents")
        self.observation_dim = _positive_integer(observation_dim, name="observation_dim")
        self.state_dim = _positive_integer(state_dim, name="state_dim")
        self.num_targets = _positive_integer(num_targets, name="num_targets")
        self._size = 0

        agent_shape = (self.capacity, self.num_envs, self.num_agents)
        self._observations = np.empty((*agent_shape, self.observation_dim), dtype=np.float64)
        self._next_observations = np.empty_like(self._observations)
        self._states = np.empty(
            (self.capacity, self.num_envs, self.state_dim),
            dtype=np.float64,
        )
        self._next_states = np.empty_like(self._states)
        self._actions = np.empty(agent_shape, dtype=np.int64)
        self._log_probs = np.empty(agent_shape, dtype=np.float64)
        self._rewards = np.empty((self.capacity, self.num_envs), dtype=np.float64)
        self._global_costs = np.empty_like(self._rewards)
        self._local_costs = np.empty(agent_shape, dtype=np.float64)
        self._occupancies = np.empty(
            (self.capacity, self.num_envs, self.num_targets),
            dtype=np.int8,
        )
        self._agent_target_occupancies = np.empty(
            (*agent_shape, self.num_targets),
            dtype=np.int8,
        )
        self._next_occupancies = np.empty_like(self._occupancies)

    def __len__(self) -> int:
        return self._size

    @property
    def rollout_length(self) -> int:
        return self.capacity

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def clear(self) -> None:
        self._size = 0

    def add(
        self,
        observations: ArrayLike,
        states: ArrayLike,
        actions: ArrayLike,
        log_probs: ArrayLike,
        rewards: ArrayLike,
        global_costs: ArrayLike,
        local_costs: ArrayLike,
        occupancies: ArrayLike,
        agent_target_occupancies: ArrayLike,
        next_observations: ArrayLike,
        next_states: ArrayLike,
        next_occupancies: ArrayLike,
    ) -> None:
        """原子追加一个包含全部 ``E`` 个环境的时间步。"""

        if self.is_full:
            raise RuntimeError("rollout buffer is full; freeze and clear it explicitly")
        agent_shape = (self.num_envs, self.num_agents)
        scalar_shape = (self.num_envs,)
        occupancy_shape = (self.num_envs, self.num_targets)
        checked = {
            "observations": _float_array(
                observations,
                (*agent_shape, self.observation_dim),
                name="observations",
            ),
            "next_observations": _float_array(
                next_observations,
                (*agent_shape, self.observation_dim),
                name="next_observations",
            ),
            "states": _float_array(states, (self.num_envs, self.state_dim), name="states"),
            "next_states": _float_array(
                next_states,
                (self.num_envs, self.state_dim),
                name="next_states",
            ),
            "actions": _actions_array(actions, agent_shape),
            "log_probs": _float_array(log_probs, agent_shape, name="log_probs"),
            "rewards": _float_array(rewards, scalar_shape, name="rewards"),
            "global_costs": _float_array(global_costs, scalar_shape, name="global_costs"),
            "local_costs": _float_array(local_costs, agent_shape, name="local_costs"),
            "occupancies": _coverage_array(
                occupancies,
                occupancy_shape,
                name="occupancies",
            ),
            "agent_target_occupancies": _coverage_array(
                agent_target_occupancies,
                (*agent_shape, self.num_targets),
                name="agent_target_occupancies",
            ),
            "next_occupancies": _coverage_array(
                next_occupancies,
                occupancy_shape,
                name="next_occupancies",
            ),
        }
        _validate_raw_signals(
            checked["rewards"],
            checked["global_costs"],
            checked["local_costs"],
            agent_axis=1,
        )
        if np.any(np.sum(checked["agent_target_occupancies"], axis=2) > 1):
            raise ValueError("each agent may occupy at most one target region")
        if not np.array_equal(
            np.any(checked["agent_target_occupancies"], axis=1).astype(np.int8),
            checked["next_occupancies"],
        ):
            raise ValueError("agent_target_occupancies union must equal next_occupancies")

        index = self._size
        for name, value in checked.items():
            getattr(self, f"_{name}")[index] = value
        self._size += 1

    def freeze(self) -> UnnormalizedRolloutBatch:
        if self._size == 0:
            raise RuntimeError("cannot freeze an empty rollout buffer")
        valid = slice(0, self._size)
        return UnnormalizedRolloutBatch(
            observations=self._observations[valid],
            next_observations=self._next_observations[valid],
            states=self._states[valid],
            next_states=self._next_states[valid],
            actions=self._actions[valid],
            log_probs=self._log_probs[valid],
            rewards=self._rewards[valid],
            global_costs=self._global_costs[valid],
            local_costs=self._local_costs[valid],
            occupancies=self._occupancies[valid],
            agent_target_occupancies=self._agent_target_occupancies[valid],
            next_occupancies=self._next_occupancies[valid],
        )


__all__ = ["UnnormalizedRolloutBatch", "UnnormalizedRolloutBuffer"]
