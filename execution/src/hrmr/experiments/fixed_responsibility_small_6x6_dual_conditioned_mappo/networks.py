"""Dual-conditioned 独立 actors 与 centralized differential critics。"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import torch
from torch import nn

from hrmr.rl.networks import (
    CategoricalActor,
    DifferentialCritic,
    IndependentActors,
    orthogonal_initialize,
)

from .conditioning import DUAL_ACTOR_INPUT_DIM, DUAL_CRITIC_INPUT_DIM

FILM_ACTOR_CONDITIONING = "continuous_film"
CONCATENATED_CRITIC_CONDITIONING = "concatenated_mlp"


class FiLMCategoricalActor(CategoricalActor):
    """用连续 dual 对两层 observation 特征执行 FiLM 调制。"""

    def __init__(
        self,
        input_dim: int = DUAL_ACTOR_INPUT_DIM,
        hidden_dims: Sequence[int] = (128, 128),
        num_actions: int = 5,
        *,
        activation: str = "tanh",
        dual_embedding_dim: int = 32,
        hidden_gain: float = sqrt(2.0),
        output_gain: float = 0.01,
        modulation_gain: float = 0.01,
    ) -> None:
        if input_dim != DUAL_ACTOR_INPUT_DIM:
            raise ValueError(f"FiLM actor input_dim must equal {DUAL_ACTOR_INPUT_DIM}")
        if isinstance(dual_embedding_dim, bool) or not isinstance(dual_embedding_dim, int):
            raise TypeError("dual_embedding_dim must be an integer")
        if dual_embedding_dim <= 0:
            raise ValueError("dual_embedding_dim must be positive")
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_actions=num_actions,
            activation=activation,
            hidden_gain=hidden_gain,
            output_gain=output_gain,
        )
        self.base_observation_dim = input_dim - 1
        self.dual_embedding_dim = dual_embedding_dim

        # 替换直接读取 57D 的首层；dual 不与 observation 简单拼接进入该层。
        self.fc1 = nn.Linear(self.base_observation_dim, self.hidden_dims[0])
        self.dual_fc1 = nn.Linear(1, dual_embedding_dim)
        self.dual_fc2 = nn.Linear(dual_embedding_dim, dual_embedding_dim)
        self.film1 = nn.Linear(dual_embedding_dim, 2 * self.hidden_dims[0])
        self.film2 = nn.Linear(dual_embedding_dim, 2 * self.hidden_dims[1])
        orthogonal_initialize(self.fc1, hidden_gain)
        orthogonal_initialize(self.dual_fc1, hidden_gain)
        orthogonal_initialize(self.dual_fc2, hidden_gain)
        orthogonal_initialize(self.film1, modulation_gain)
        orthogonal_initialize(self.film2, modulation_gain)

    def forward(
        self,
        observations: torch.Tensor,
        *,
        validate_values: bool = True,
    ) -> torch.Tensor:
        """以同一连续函数处理任意归一化 dual，不使用离散策略 head。"""

        self._validate_observations(observations)
        if validate_values and not torch.all(torch.isfinite(observations)):
            raise ValueError("observations must contain only finite values")
        base_observation = observations[..., : self.base_observation_dim]
        normalized_dual = observations[..., self.base_observation_dim :]
        if validate_values and bool(torch.any((normalized_dual < 0.0) | (normalized_dual > 1.0))):
            raise ValueError("normalized dual condition must lie in [0, 1]")

        # 把两个端点对称映射为 -1/+1，避免结构上把 λ=0 固定为无调制基线。
        centered_dual = 2.0 * normalized_dual - 1.0
        dual_embedding = self.activation(self.dual_fc1(centered_dual))
        dual_embedding = self.activation(self.dual_fc2(dual_embedding))
        gamma1, beta1 = self.film1(dual_embedding).chunk(2, dim=-1)
        hidden = self.activation((1.0 + gamma1) * self.fc1(base_observation) + beta1)
        gamma2, beta2 = self.film2(dual_embedding).chunk(2, dim=-1)
        hidden = self.activation((1.0 + gamma2) * self.fc2(hidden) + beta2)
        return self.output_layer(hidden)


class FiLMIndependentActors(IndependentActors):
    """四个参数独立、各自用同一连续 FiLM 结构的 actors。"""

    def __init__(
        self,
        num_agents: int = 4,
        input_dim: int = DUAL_ACTOR_INPUT_DIM,
        hidden_dims: Sequence[int] = (128, 128),
        num_actions: int = 5,
        *,
        activation: str = "tanh",
        initialization_seeds: Sequence[int] = (100, 101, 102, 103),
    ) -> None:
        super().__init__(
            num_agents=num_agents,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_actions=num_actions,
            activation=activation,
            initialization_seeds=initialization_seeds,
        )
        actors = []
        for seed in self.initialization_seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                actors.append(
                    FiLMCategoricalActor(
                        input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        num_actions=num_actions,
                        activation=activation,
                    )
                )
        self.actors = nn.ModuleList(actors)


def make_dual_conditioned_networks(
    initialization_seeds: Sequence[int],
    *,
    activation: str = "tanh",
    actor_conditioning: str = FILM_ACTOR_CONDITIONING,
    critic_conditioning: str = CONCATENATED_CRITIC_CONDITIONING,
    device: str | torch.device = "cpu",
) -> tuple[IndependentActors, DifferentialCritic, DifferentialCritic]:
    """建立四个条件化 actors 和两个独立的 17D centralized critics。"""

    seeds = tuple(initialization_seeds)
    if len(seeds) != 4:
        raise ValueError("initialization_seeds must contain four entries")
    if actor_conditioning != FILM_ACTOR_CONDITIONING:
        raise ValueError(f"actor_conditioning must equal {FILM_ACTOR_CONDITIONING!r}")
    if critic_conditioning != CONCATENATED_CRITIC_CONDITIONING:
        raise ValueError(f"critic_conditioning must equal {CONCATENATED_CRITIC_CONDITIONING!r}")
    actor = FiLMIndependentActors(
        num_agents=4,
        input_dim=DUAL_ACTOR_INPUT_DIM,
        hidden_dims=(128, 128),
        num_actions=5,
        initialization_seeds=seeds,
        activation=activation,
    )
    reward_critic = DifferentialCritic(
        input_dim=DUAL_CRITIC_INPUT_DIM,
        hidden_dims=(256, 256),
        activation=activation,
    )
    cost_critic = DifferentialCritic(
        input_dim=DUAL_CRITIC_INPUT_DIM,
        hidden_dims=(256, 256),
        activation=activation,
    )
    target = torch.device(device)
    return actor.to(target), reward_critic.to(target), cost_critic.to(target)


__all__ = [
    "CONCATENATED_CRITIC_CONDITIONING",
    "FILM_ACTOR_CONDITIONING",
    "FiLMCategoricalActor",
    "FiLMIndependentActors",
    "make_dual_conditioned_networks",
]
