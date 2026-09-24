"""HRMR formal independent actors 与长期平均 differential critics。"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from numbers import Integral, Real

import torch
from torch import nn
from torch.distributions import Categorical


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    checked = int(value)
    if checked <= 0:
        raise ValueError(f"{name} must be positive")
    return checked


def _positive_gain(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    checked = float(value)
    if not torch.isfinite(torch.tensor(checked)) or checked <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return checked


def orthogonal_initialize(linear: nn.Linear, gain: float) -> None:
    """用指定 gain 正交初始化线性层，并把 bias 清零。"""

    if not isinstance(linear, nn.Linear):
        raise TypeError("orthogonal_initialize expects torch.nn.Linear")
    checked_gain = _positive_gain(gain, name="gain")
    nn.init.orthogonal_(linear.weight, gain=checked_gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _hidden_dimensions(values: Sequence[int], *, expected_length: int = 2) -> tuple[int, ...]:
    dimensions = tuple(
        _positive_integer(value, name=f"hidden_dims[{index}]") for index, value in enumerate(values)
    )
    if len(dimensions) != expected_length:
        raise ValueError(f"hidden_dims must contain exactly {expected_length} entries")
    return dimensions


def _activation_module(name: str) -> tuple[str, nn.Module]:
    if not isinstance(name, str):
        raise TypeError("activation must be a string")
    normalized = name.lower()
    if normalized == "tanh":
        return normalized, nn.Tanh()
    if normalized == "relu":
        return normalized, nn.ReLU()
    raise ValueError("activation must be 'tanh' or 'relu'")


class CategoricalActor(nn.Module):
    """单个机器人的 Categorical actor，默认结构 ``48-128-128-5``。"""

    def __init__(
        self,
        input_dim: int = 48,
        hidden_dims: Sequence[int] = (128, 128),
        num_actions: int = 5,
        *,
        activation: str = "tanh",
        hidden_gain: float = sqrt(2.0),
        output_gain: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, name="input_dim")
        self.hidden_dims = _hidden_dimensions(hidden_dims)
        self.num_actions = _positive_integer(num_actions, name="num_actions")
        self.hidden_gain = _positive_gain(hidden_gain, name="hidden_gain")
        self.output_gain = _positive_gain(output_gain, name="output_gain")
        self.activation_name, self.activation = _activation_module(activation)

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.output_layer = nn.Linear(self.hidden_dims[1], self.num_actions)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """应用 MAPPO 常用的正交初始化与小策略输出 gain。"""

        orthogonal_initialize(self.fc1, self.hidden_gain)
        orthogonal_initialize(self.fc2, self.hidden_gain)
        orthogonal_initialize(self.output_layer, self.output_gain)

    def _validate_observations(self, observations: torch.Tensor) -> None:
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor")
        if observations.ndim == 0 or observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"observations must have final dimension {self.input_dim}; "
                f"got shape {tuple(observations.shape)}"
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """返回未归一化的五动作 logits。"""

        self._validate_observations(observations)
        hidden = self.activation(self.fc1(observations))
        hidden = self.activation(self.fc2(hidden))
        return self.output_layer(hidden)

    def distribution(self, observations: torch.Tensor) -> Categorical:
        """构造策略的 Categorical 分布。"""

        return Categorical(logits=self(observations))

    def sample(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作并返回 ``(actions, log_probs, entropy)``。

        正式评价应保持默认随机采样；``deterministic=True`` 只用于额外的轨迹
        可视化或诊断。
        """

        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a boolean")
        distribution = self.distribution(observations)
        actions = (
            torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        )
        return actions, distribution.log_prob(actions), distribution.entropy()

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算给定动作的 log probability 与策略 entropy。"""

        distribution = self.distribution(observations)
        return distribution.log_prob(actions), distribution.entropy()


class IndependentActors(nn.Module):
    """四个互不共享参数的 categorical actors。"""

    def __init__(
        self,
        num_agents: int = 4,
        input_dim: int = 48,
        hidden_dims: Sequence[int] = (128, 128),
        num_actions: int = 5,
        *,
        activation: str = "tanh",
        initialization_seeds: Sequence[int] = (100, 101, 102, 103),
    ) -> None:
        super().__init__()
        self.num_agents = _positive_integer(num_agents, name="num_agents")
        self.input_dim = _positive_integer(input_dim, name="input_dim")
        self.hidden_dims = _hidden_dimensions(hidden_dims)
        self.num_actions = _positive_integer(num_actions, name="num_actions")
        seeds = tuple(initialization_seeds)
        if len(seeds) != self.num_agents:
            raise ValueError("initialization_seeds must contain one seed per agent")
        if any(
            isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0
            for seed in seeds
        ):
            raise ValueError("initialization seeds must be non-negative integers")
        normalized_seeds = tuple(int(seed) for seed in seeds)
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("initialization seeds must be unique")
        self.initialization_seeds = normalized_seeds

        actors = []
        for seed in normalized_seeds:
            # 每个 actor 使用独立显式 seed，同时不消耗 trainer 的全局 RNG stream。
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                actors.append(
                    CategoricalActor(
                        input_dim=self.input_dim,
                        hidden_dims=self.hidden_dims,
                        num_actions=self.num_actions,
                        activation=activation,
                    )
                )
        self.actors = nn.ModuleList(actors)

    def _validate_joint_observations(self, observations: torch.Tensor) -> None:
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor")
        if (
            observations.ndim < 2
            or observations.shape[-2] != self.num_agents
            or observations.shape[-1] != self.input_dim
        ):
            raise ValueError(
                "observations must have final dimensions "
                f"({self.num_agents}, {self.input_dim}); got {tuple(observations.shape)}"
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """分别调用 actor_i(o_i)，再恢复 joint-agent 维。"""

        self._validate_joint_observations(observations)
        return torch.stack(
            [
                actor(observations[..., agent_index, :])
                for agent_index, actor in enumerate(self.actors)
            ],
            dim=-2,
        )

    def distribution(self, observations: torch.Tensor) -> Categorical:
        return Categorical(logits=self(observations))

    def sample(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a boolean")
        distribution = self.distribution(observations)
        actions = (
            torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        )
        return actions, distribution.log_prob(actions), distribution.entropy()

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        return distribution.log_prob(actions), distribution.entropy()

    def actor(self, agent_index: int) -> CategoricalActor:
        """返回指定 agent 的独立 actor。"""

        if isinstance(agent_index, bool) or not isinstance(agent_index, Integral):
            raise TypeError("agent_index must be an integer")
        checked = int(agent_index)
        if not 0 <= checked < self.num_agents:
            raise IndexError("agent_index is out of range")
        return self.actors[checked]

    @property
    def parameter_count_by_actor(self) -> tuple[int, ...]:
        return tuple(
            sum(parameter.numel() for parameter in actor.parameters()) for actor in self.actors
        )


class DifferentialCritic(nn.Module):
    """长期平均 differential value critic，默认结构 ``16-256-256-1``。"""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        activation: str = "tanh",
        hidden_gain: float = sqrt(2.0),
        output_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, name="input_dim")
        self.hidden_dims = _hidden_dimensions(hidden_dims)
        self.hidden_gain = _positive_gain(hidden_gain, name="hidden_gain")
        self.output_gain = _positive_gain(output_gain, name="output_gain")
        self.activation_name, self.activation = _activation_module(activation)

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.output_layer = nn.Linear(self.hidden_dims[1], 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """正交初始化 critic；输出层保持单位 gain。"""

        orthogonal_initialize(self.fc1, self.hidden_gain)
        orthogonal_initialize(self.fc2, self.hidden_gain)
        orthogonal_initialize(self.output_layer, self.output_gain)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """返回保留末尾单值维的 differential values。"""

        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch.Tensor")
        if states.ndim == 0 or states.shape[-1] != self.input_dim:
            raise ValueError(
                f"states must have final dimension {self.input_dim}; "
                f"got shape {tuple(states.shape)}"
            )
        hidden = self.activation(self.fc1(states))
        hidden = self.activation(self.fc2(hidden))
        return self.output_layer(hidden)


class DifferentialCritics(nn.Module):
    """互不共享参数的 reward 与 cost differential critic 对。"""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.reward_critic = DifferentialCritic(
            input_dim,
            hidden_dims,
            activation=activation,
        )
        self.cost_critic = DifferentialCritic(
            input_dim,
            hidden_dims,
            activation=activation,
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reward_critic(states), self.cost_critic(states)


DifferentialValueNetwork = DifferentialCritic
RewardCostCritics = DifferentialCritics


__all__ = [
    "CategoricalActor",
    "DifferentialCritic",
    "DifferentialCritics",
    "DifferentialValueNetwork",
    "IndependentActors",
    "RewardCostCritics",
    "orthogonal_initialize",
]
