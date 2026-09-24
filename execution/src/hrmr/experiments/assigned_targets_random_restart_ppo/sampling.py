"""保留128步更新批量，用连续片段隔离训练重启前后的TD与保持率。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import torch

from hrmr.experiments.assigned_targets_ppo.trainer import (
    AssignedVectorEnv,
    collect_rollout,
)
from hrmr.experiments.unnormalized_equal_hazard_ppo.rollout_buffer import (
    UnnormalizedRolloutBatch,
)
from hrmr.rl.differential_td import normalize_advantages
from hrmr.rl.fixed_lambda_trainer import _frozen_rollout_quantities, _initial_occupancies

from .environment import RandomStartAssignedEnvironment, sample_joint_positions


class RandomRestartVectorEnv(AssignedVectorEnv):
    """只允许采样器显式重启；step沿用原continuing环境。"""

    def __init__(self, num_envs, config):
        super().__init__(num_envs, config, environment_class=RandomStartAssignedEnvironment)

    def reset_random(self, rng, *, seed_start=0):
        seeds = tuple(seed_start + index for index in range(self.num_envs))
        for environment, seed in zip(self.envs, seeds, strict=True):
            environment.reset_positions(sample_joint_positions(environment.config, rng), seed=seed)
        self._last_seeds = seeds
        self._has_reset = True
        return self.current()


@dataclass
class RestartSchedule:
    """计时单位是每个子环境实际走的步数，不是joint总步数或rollout数。"""

    interval: int = 1000
    age: int = 0
    restart_count: int = 0

    def __post_init__(self):
        if isinstance(self.interval, bool) or not isinstance(self.interval, int):
            raise TypeError("restart interval must be an integer")
        if self.interval <= 0 or not 0 <= self.age <= self.interval or self.restart_count < 0:
            raise ValueError("invalid restart schedule")


@dataclass(frozen=True)
class RestartRollout:
    """batch只保存真实transitions；segments明确标出不连续边界。"""

    batch: UnnormalizedRolloutBatch
    segments: tuple[UnnormalizedRolloutBatch, ...]
    observations: np.ndarray
    states: np.ndarray
    occupancies: np.ndarray
    physical_membership: np.ndarray


def concatenate_batches(segments):
    if not segments:
        raise ValueError("at least one continuous segment is required")
    return UnnormalizedRolloutBatch(
        **{
            field.name: np.concatenate([getattr(part, field.name) for part in segments], axis=0)
            for field in fields(UnnormalizedRolloutBatch)
        }
    )


def collect_restart_rollout(
    actor,
    vector,
    observations,
    states,
    occupancies,
    steps,
    device,
    permutations,
    *,
    schedule: RestartSchedule,
    reset_rng: np.random.Generator,
    tracker,
    rollout_start_environment_steps: int,
    on_reset: Callable | None = None,
):
    """重启处拆段但不提前PPO更新；整批仍使用同一组target permutations。"""
    if steps <= 0:
        raise ValueError("rollout steps must be positive")
    segments = []
    physical = []
    collected = 0
    while collected < steps:
        joint_steps = rollout_start_environment_steps + collected * vector.num_envs
        if schedule.age == schedule.interval:
            schedule.restart_count += 1
            observations, states, infos = vector.reset_random(
                reset_rng, seed_start=schedule.restart_count * vector.num_envs
            )
            occupancies = _initial_occupancies(infos)
            tracker.break_continuity(np.ones(vector.num_envs, dtype=np.bool_))
            schedule.age = 0
            if on_reset is not None:
                on_reset(joint_steps, schedule.restart_count, vector.positions, occupancies)
        length = min(steps - collected, schedule.interval - schedule.age)
        part, observations, states, occupancies, membership = collect_rollout(
            actor, vector, observations, states, occupancies, length, device, permutations
        )
        tracker.update(part.next_occupancies, rollout_start_environment_steps=joint_steps)
        segments.append(part)
        physical.append(membership)
        collected += length
        schedule.age += length
    membership = np.concatenate(physical, axis=0)
    membership.setflags(write=False)
    return RestartRollout(
        concatenate_batches(segments),
        tuple(segments),
        observations,
        states,
        occupancies,
        membership,
    )


def frozen_restart_quantities(
    rollout: RestartRollout,
    reward_critic,
    cost_critic,
    reward_rate,
    cost_rate,
    fixed_lambda,
    normalize_combined_advantage,
    device,
    n_step=32,
):
    """各段末尾用真实末状态bootstrap，拼接后只做一次整batch优势标准化。

    平均率由调用方按整batch更新一次，所有片段使用相同rho。原始reward/cost
    target与PPO公式直接复用；绝不把重启后的初态作为上一片段的bootstrap。
    """
    quantities = [
        _frozen_rollout_quantities(
            segment,
            reward_critic,
            cost_critic,
            reward_rate,
            cost_rate,
            fixed_lambda,
            False,
            device,
            n_step,
        )
        for segment in rollout.segments
    ]
    rewards, costs, combined = (
        torch.cat([values[index] for values in quantities], dim=0) for index in range(3)
    )
    if normalize_combined_advantage:
        combined = normalize_advantages(combined)
    return rewards.detach(), costs.detach(), combined.detach()
