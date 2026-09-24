"""完整 HRMR 环境上的 safe-target actor observation 薄适配器。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from hrmr.observations import TARGET_FEATURE_SIZE
from hrmr.rl.networks import IndependentActors
from hrmr.rl.target_permutation import validate_target_permutation

SAFE_ACTOR_NUM_AGENTS = 4
SAFE_ACTOR_NUM_TARGETS = 4
SAFE_ACTOR_OBSERVATION_DIM = 28
FULL_ACTOR_OBSERVATION_DIM = 48

_SELF_AND_SAFE_STOP = 2 + SAFE_ACTOR_NUM_TARGETS * TARGET_FEATURE_SIZE
_TEAMMATE_START = 2 + 8 * TARGET_FEATURE_SIZE


def project_safe_target_actor_observations(
    observations: ArrayLike,
) -> NDArray[np.float64]:
    """从正式 48D observation 中保留 self、S1--S4 blocks 与 teammates。"""

    raw = np.asarray(observations)
    if raw.ndim < 2 or raw.shape[-2:] != (
        SAFE_ACTOR_NUM_AGENTS,
        FULL_ACTOR_OBSERVATION_DIM,
    ):
        raise ValueError("full observations must end with shape (4,48)")
    if raw.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(raw)):
        raise ValueError("full observations must contain finite numeric values")
    projected = np.concatenate(
        (raw[..., :_SELF_AND_SAFE_STOP], raw[..., _TEAMMATE_START:]),
        axis=-1,
    ).astype(np.float64, copy=False)
    if projected.shape[-1] != SAFE_ACTOR_OBSERVATION_DIM:
        raise AssertionError("safe-only actor observation dimension drifted")
    return projected.copy()


def project_safe_target_actor_observations_tensor(
    observations: torch.Tensor,
) -> torch.Tensor:
    """Torch 版本投影，用于 stochastic evaluation wrapper。"""

    if not isinstance(observations, torch.Tensor):
        raise TypeError("observations must be a torch.Tensor")
    if observations.ndim != 3 or tuple(observations.shape[-2:]) != (
        SAFE_ACTOR_NUM_AGENTS,
        FULL_ACTOR_OBSERVATION_DIM,
    ):
        raise ValueError("full evaluation observations must have shape [B,4,48]")
    return torch.cat(
        (observations[..., :_SELF_AND_SAFE_STOP], observations[..., _TEAMMATE_START:]),
        dim=-1,
    )


class SafeTargetActorView:
    """让 28D actor 接收 evaluator 提供的完整 48D environment observation。"""

    def __init__(
        self,
        actor: IndependentActors,
        permutations: ArrayLike | None = None,
    ) -> None:
        if not isinstance(actor, IndependentActors):
            raise TypeError("safe target view requires IndependentActors")
        if actor.num_agents != SAFE_ACTOR_NUM_AGENTS:
            raise ValueError("safe target view requires four actors")
        if actor.input_dim != SAFE_ACTOR_OBSERVATION_DIM:
            raise ValueError("safe target view actors must have input_dim=28")
        self.actor = actor
        self.num_agents = actor.num_agents
        self.num_actions = actor.num_actions
        self.input_dim = FULL_ACTOR_OBSERVATION_DIM
        self.permutations: NDArray[np.int64] | None = None
        if permutations is not None:
            raw = np.asarray(permutations)
            if raw.ndim != 2 or raw.shape[1] != SAFE_ACTOR_NUM_TARGETS:
                raise ValueError("safe target permutations must have shape [B,4]")
            self.permutations = np.stack(
                [
                    validate_target_permutation(row, num_targets=SAFE_ACTOR_NUM_TARGETS)
                    for row in raw
                ]
            ).astype(np.int64, copy=False)

    @property
    def training(self) -> bool:
        return bool(self.actor.training)

    def parameters(self) -> Any:
        return self.actor.parameters()

    def eval(self) -> SafeTargetActorView:
        self.actor.eval()
        return self

    def train(self, mode: bool = True) -> SafeTargetActorView:
        self.actor.train(mode)
        return self

    def _actor_observations(self, observations: torch.Tensor) -> torch.Tensor:
        projected = project_safe_target_actor_observations_tensor(observations)
        if self.permutations is None:
            return projected
        if projected.shape[0] != len(self.permutations):
            raise ValueError("evaluation batch and safe target permutation counts differ")
        blocks = projected[..., 2:_SELF_AND_SAFE_STOP].reshape(
            projected.shape[0],
            SAFE_ACTOR_NUM_AGENTS,
            SAFE_ACTOR_NUM_TARGETS,
            TARGET_FEATURE_SIZE,
        )
        permutation_tensor = torch.as_tensor(
            self.permutations,
            dtype=torch.long,
            device=observations.device,
        )
        indices = permutation_tensor[:, None, :, None].expand_as(blocks)
        result = projected.clone()
        result[..., 2:_SELF_AND_SAFE_STOP] = torch.gather(
            blocks,
            dim=2,
            index=indices,
        ).reshape(projected.shape[0], SAFE_ACTOR_NUM_AGENTS, -1)
        return result

    def distribution(self, observations: torch.Tensor) -> Any:
        return self.actor.distribution(self._actor_observations(observations))

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


__all__ = [
    "FULL_ACTOR_OBSERVATION_DIM",
    "SAFE_ACTOR_NUM_AGENTS",
    "SAFE_ACTOR_NUM_TARGETS",
    "SAFE_ACTOR_OBSERVATION_DIM",
    "SafeTargetActorView",
    "project_safe_target_actor_observations",
    "project_safe_target_actor_observations_tensor",
]
