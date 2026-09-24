"""Dual-conditioned owner-aware rollout；实际网络输入原样写入 buffer。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from hrmr.experiments.assigned_targets_random_restart_ppo.sampling import (
    RestartRollout,
    RestartSchedule,
    concatenate_batches,
)
from hrmr.experiments.unnormalized_equal_hazard_ppo.rollout_buffer import (
    UnnormalizedRolloutBuffer,
)
from hrmr.rl.fixed_lambda_trainer import _agent_target_occupancies, _initial_occupancies

from .conditioning import (
    DUAL_ACTOR_INPUT_DIM,
    DUAL_CRITIC_INPUT_DIM,
    condition_actor_observations,
    condition_critic_states,
    permute_conditioned_target_blocks,
)


def collect_dual_conditioned_rollout(
    actor,
    vector,
    observations,
    states,
    occupancies,
    steps,
    device,
    permutations,
    *,
    dual_lambda,
    conditioning_max,
):
    """采样固定 dual 的连续片段，并保存真实使用的 57D/17D 网络输入。"""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    buffer = UnnormalizedRolloutBuffer(
        steps,
        vector.num_envs,
        vector.num_agents,
        DUAL_ACTOR_INPUT_DIM,
        DUAL_CRITIC_INPUT_DIM,
        vector.num_targets,
    )
    owner_mask = (
        np.arange(vector.num_agents)[:, None] == vector.envs[0].target_owner_indices[None, :]
    )
    physical_memberships = []
    actor.eval()
    with torch.inference_mode():
        for _ in range(steps):
            conditioned_observations = condition_actor_observations(
                observations, dual_lambda, conditioning_max
            )
            policy_observations = permute_conditioned_target_blocks(
                conditioned_observations, permutations
            )
            critic_states = condition_critic_states(states, dual_lambda, conditioning_max)
            actions, log_probs, _ = actor.sample(
                torch.as_tensor(policy_observations, dtype=torch.float32, device=device)
            )
            action_array = actions.cpu().numpy().astype(np.int64, copy=False)
            next_observations, next_states, rewards, local, costs, terminated, truncated, infos = (
                vector.step(action_array)
            )
            if np.any(terminated) or np.any(truncated):
                raise AssertionError("continuing rollout unexpectedly terminated")
            next_occupancies = _initial_occupancies(infos)
            membership = _agent_target_occupancies(
                infos,
                tuple(vector.envs[0].config.target_ids),
                vector.num_agents,
            )
            physical_memberships.append(membership.copy())
            next_policy_observations = permute_conditioned_target_blocks(
                condition_actor_observations(next_observations, dual_lambda, conditioning_max),
                permutations,
            )
            next_critic_states = condition_critic_states(next_states, dual_lambda, conditioning_max)
            buffer.add(
                policy_observations,
                critic_states,
                action_array,
                log_probs.cpu().numpy(),
                rewards,
                costs,
                local,
                occupancies,
                membership * owner_mask,
                next_policy_observations,
                next_critic_states,
                next_occupancies,
            )
            observations, states, occupancies = (
                next_observations,
                next_states,
                next_occupancies,
            )
    actor.train()
    physical = np.asarray(physical_memberships, dtype=np.int8)
    physical.setflags(write=False)
    return buffer.freeze(), observations, states, occupancies, physical


def collect_dual_conditioned_restart_rollout(
    actor,
    vector,
    observations,
    states,
    occupancies,
    steps,
    device,
    permutations,
    *,
    dual_lambda,
    conditioning_max,
    schedule: RestartSchedule,
    reset_rng: np.random.Generator,
    rollout_start_environment_steps: int,
    tracker=None,
    on_reset: Callable | None = None,
) -> RestartRollout:
    """在人工重启处拆段；同一 PPO rollout 的 dual 始终保持不变。"""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    segments = []
    physical = []
    collected = 0
    while collected < steps:
        joint_steps = rollout_start_environment_steps + collected * vector.num_envs
        if schedule.age == schedule.interval:
            schedule.restart_count += 1
            observations, states, infos = vector.reset_random(
                reset_rng,
                seed_start=schedule.restart_count * vector.num_envs,
            )
            occupancies = _initial_occupancies(infos)
            if tracker is not None:
                tracker.break_continuity(np.ones(vector.num_envs, dtype=np.bool_))
            schedule.age = 0
            if on_reset is not None:
                on_reset(joint_steps, schedule.restart_count, vector.positions, occupancies)
        length = min(steps - collected, schedule.interval - schedule.age)
        part, observations, states, occupancies, membership = collect_dual_conditioned_rollout(
            actor,
            vector,
            observations,
            states,
            occupancies,
            length,
            device,
            permutations,
            dual_lambda=dual_lambda,
            conditioning_max=conditioning_max,
        )
        if tracker is not None:
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


__all__ = [
    "collect_dual_conditioned_restart_rollout",
    "collect_dual_conditioned_rollout",
]
