"""Continuing-task rollout 的定长 NumPy 数据缓冲区。

本模块只保存环境交互张量并生成不可变快照，不计算 return、TD error、
advantage、GAE 或任何策略更新量。Rollout 边界只是数据切片边界，不包含
terminal/reset 语义。
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
    """验证正整数 buffer 维度。"""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _float_array(value: ArrayLike, shape: tuple[int, ...], *, name: str) -> FloatArray:
    """验证有限浮点张量并返回独立 ``float64`` 副本。"""

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
    """验证策略动作张量并返回独立 ``int64`` 副本。"""

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
    """验证二值 occupancy/coverage 张量。"""

    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"{name} must contain binary numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} entries must be exactly 0 or 1")
    return array.astype(np.int8, copy=True)


@dataclass(frozen=True)
class RolloutBatch:
    """冻结后的 rollout 张量快照。

    所有数组都拥有独立内存并被设为只读。前两个维度统一为
    ``[time, environment]``，因此 trainer 可显式将它们展平，而不会混淆
    agent 或 feature 维度。``agent_target_occupancies[T,E,N,M]`` 保存每次
    transition 后的逐 agent target 归属，其 agent 维并集必须等于同一步的
    ``next_occupancies[T,E,M]``。
    """

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
            "agent_target_occupancies": (
                num_steps,
                num_envs,
                num_agents,
                num_targets,
            ),
            "next_occupancies": (num_steps, num_envs, num_targets),
        }
        for item in fields(self):
            raw = getattr(self, item.name)
            expected_shape = shapes[item.name]
            if item.name == "actions":
                checked: np.ndarray = _actions_array(raw, expected_shape)
            elif item.name in {
                "occupancies",
                "agent_target_occupancies",
                "next_occupancies",
            }:
                checked = _coverage_array(raw, expected_shape, name=item.name)
            else:
                checked = _float_array(raw, expected_shape, name=item.name)
            checked.setflags(write=False)
            object.__setattr__(self, item.name, checked)

        if np.any((self.rewards < 0.0) | (self.rewards > 1.0)):
            raise ValueError("rewards must lie in [0, 1]")
        if np.any((self.global_costs < 0.0) | (self.global_costs > 1.0)):
            raise ValueError("global_costs must lie in [0, 1]")
        if np.any(self.local_costs < 0.0):
            raise ValueError("local_costs must be non-negative")
        if not np.allclose(
            self.global_costs,
            np.sum(self.local_costs, axis=2),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("global_costs must equal the sum of local_costs")
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
        """``observations`` 的显式 current-state 别名。"""

        return self.observations

    @property
    def current_states(self) -> FloatArray:
        """``states`` 的显式 current-state 别名。"""

        return self.states

    @property
    def current_occupancies(self) -> CoverageArray:
        """``occupancies`` 的显式 current-state 别名。"""

        return self.occupancies

    @property
    def coverage(self) -> CoverageArray:
        """兼容环境命名的 current occupancy 别名。"""

        return self.occupancies

    @property
    def next_coverage(self) -> CoverageArray:
        """兼容环境命名的 next occupancy 别名。"""

        return self.next_occupancies

    @property
    def logprobs(self) -> FloatArray:
        """``log_probs`` 的无下划线兼容别名。"""

        return self.log_probs


class RolloutBuffer:
    """预分配的 ``[T,E,...]`` continuing rollout 存储。"""

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
        self.observation_dim = _positive_integer(
            observation_dim,
            name="observation_dim",
        )
        self.state_dim = _positive_integer(state_dim, name="state_dim")
        self.num_targets = _positive_integer(num_targets, name="num_targets")
        self._size = 0

        observation_shape = (
            self.capacity,
            self.num_envs,
            self.num_agents,
            self.observation_dim,
        )
        state_shape = (self.capacity, self.num_envs, self.state_dim)
        agent_shape = (self.capacity, self.num_envs, self.num_agents)
        scalar_shape = (self.capacity, self.num_envs)
        occupancy_shape = (self.capacity, self.num_envs, self.num_targets)

        self._observations = np.empty(observation_shape, dtype=np.float64)
        self._next_observations = np.empty(observation_shape, dtype=np.float64)
        self._states = np.empty(state_shape, dtype=np.float64)
        self._next_states = np.empty(state_shape, dtype=np.float64)
        self._actions = np.empty(agent_shape, dtype=np.int64)
        self._log_probs = np.empty(agent_shape, dtype=np.float64)
        self._rewards = np.empty(scalar_shape, dtype=np.float64)
        self._global_costs = np.empty(scalar_shape, dtype=np.float64)
        self._local_costs = np.empty(agent_shape, dtype=np.float64)
        self._occupancies = np.empty(occupancy_shape, dtype=np.int8)
        self._agent_target_occupancies = np.empty(
            (*agent_shape, self.num_targets),
            dtype=np.int8,
        )
        self._next_occupancies = np.empty(occupancy_shape, dtype=np.int8)

    def __len__(self) -> int:
        return self._size

    @property
    def rollout_length(self) -> int:
        """容量 ``T`` 的语义别名。"""

        return self.capacity

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def clear(self) -> None:
        """只移动写指针；不会改变任何环境状态或触发 reset。"""

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
        """原子地追加一个包含全部 ``E`` 个环境的时间步。"""

        if self.is_full:
            raise RuntimeError("rollout buffer is full; freeze and clear it explicitly")

        observation_shape = (self.num_envs, self.num_agents, self.observation_dim)
        state_shape = (self.num_envs, self.state_dim)
        agent_shape = (self.num_envs, self.num_agents)
        scalar_shape = (self.num_envs,)
        occupancy_shape = (self.num_envs, self.num_targets)
        agent_target_occupancy_shape = (
            self.num_envs,
            self.num_agents,
            self.num_targets,
        )

        checked_observations = _float_array(
            observations,
            observation_shape,
            name="observations",
        )
        checked_next_observations = _float_array(
            next_observations,
            observation_shape,
            name="next_observations",
        )
        checked_states = _float_array(states, state_shape, name="states")
        checked_next_states = _float_array(
            next_states,
            state_shape,
            name="next_states",
        )
        checked_actions = _actions_array(actions, agent_shape)
        checked_log_probs = _float_array(log_probs, agent_shape, name="log_probs")
        checked_rewards = _float_array(rewards, scalar_shape, name="rewards")
        checked_global_costs = _float_array(
            global_costs,
            scalar_shape,
            name="global_costs",
        )
        checked_local_costs = _float_array(
            local_costs,
            agent_shape,
            name="local_costs",
        )
        checked_occupancies = _coverage_array(
            occupancies,
            occupancy_shape,
            name="occupancies",
        )
        checked_agent_target_occupancies = _coverage_array(
            agent_target_occupancies,
            agent_target_occupancy_shape,
            name="agent_target_occupancies",
        )
        checked_next_occupancies = _coverage_array(
            next_occupancies,
            occupancy_shape,
            name="next_occupancies",
        )

        if np.any((checked_rewards < 0.0) | (checked_rewards > 1.0)):
            raise ValueError("rewards must lie in [0, 1]")
        if np.any((checked_global_costs < 0.0) | (checked_global_costs > 1.0)):
            raise ValueError("global_costs must lie in [0, 1]")
        if np.any(checked_local_costs < 0.0):
            raise ValueError("local_costs must be non-negative")
        if not np.allclose(
            checked_global_costs,
            np.sum(checked_local_costs, axis=1),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("global_costs must equal the sum of local_costs")
        if np.any(np.sum(checked_agent_target_occupancies, axis=2) > 1):
            raise ValueError("each agent may occupy at most one target region")
        if not np.array_equal(
            np.any(checked_agent_target_occupancies, axis=1).astype(np.int8),
            checked_next_occupancies,
        ):
            raise ValueError("agent_target_occupancies union must equal next_occupancies")

        index = self._size
        self._observations[index] = checked_observations
        self._next_observations[index] = checked_next_observations
        self._states[index] = checked_states
        self._next_states[index] = checked_next_states
        self._actions[index] = checked_actions
        self._log_probs[index] = checked_log_probs
        self._rewards[index] = checked_rewards
        self._global_costs[index] = checked_global_costs
        self._local_costs[index] = checked_local_costs
        self._occupancies[index] = checked_occupancies
        self._agent_target_occupancies[index] = checked_agent_target_occupancies
        self._next_occupancies[index] = checked_next_occupancies
        self._size += 1

    def freeze(self) -> RolloutBatch:
        """返回当前有效时间步的不可变深拷贝，不清空 buffer。"""

        if self._size == 0:
            raise RuntimeError("cannot freeze an empty rollout buffer")
        valid = slice(0, self._size)
        return RolloutBatch(
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


ContinuingRolloutBuffer = RolloutBuffer

__all__ = ["ContinuingRolloutBuffer", "RolloutBatch", "RolloutBuffer"]
