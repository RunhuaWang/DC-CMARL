"""固定责任区 MAPPO 共用的向量采样与结果序列化组件。"""

from __future__ import annotations

import csv
import json

import numpy as np
import torch

from hrmr.experiments.unnormalized_equal_hazard_ppo.rollout_buffer import (
    UnnormalizedRolloutBuffer,
)
from hrmr.experiments.unnormalized_original_parameters_ppo.vector_env import (
    UnnormalizedOriginalParametersVectorEnv,
)
from hrmr.rl.fixed_lambda_trainer import _agent_target_occupancies, _initial_occupancies

from .environment import AssignedTargetsEnvironment, permute_assigned_target_blocks


class AssignedVectorEnv(UnnormalizedOriginalParametersVectorEnv):
    """复用同步 continuing 推进，只替换为 owner-aware 环境。"""

    def __init__(self, num_envs, config, *, environment_class=AssignedTargetsEnvironment):
        super().__init__(num_envs, config, seeds=0)
        self._envs = tuple(environment_class(config) for _ in range(num_envs))

    @property
    def observation_dim(self):
        return 56


def collect_rollout(actor, vector, observations, states, occupancies, steps, device, permutations):
    """采样实际排列后的 56D observation 原样入 buffer。"""

    buffer = UnnormalizedRolloutBuffer(steps, vector.num_envs, 4, 56, 16, 8)
    owner_mask = np.arange(4)[:, None] == vector.envs[0].target_owner_indices[None, :]
    physical_memberships = []
    actor.eval()
    with torch.inference_mode():
        for _ in range(steps):
            policy_obs = permute_assigned_target_blocks(observations, permutations)
            actions, log_probs, _ = actor.sample(
                torch.as_tensor(policy_obs, dtype=torch.float32, device=device)
            )
            actions = actions.cpu().numpy()
            next_obs, next_states, rewards, local, costs, terminated, truncated, infos = (
                vector.step(actions)
            )
            if np.any(terminated) or np.any(truncated):
                raise AssertionError("continuing rollout unexpectedly terminated")
            next_occupancies = _initial_occupancies(infos)
            membership = _agent_target_occupancies(infos, vector.envs[0].config.target_ids, 4)
            physical_memberships.append(membership.copy())
            buffer.add(
                policy_obs,
                states,
                actions,
                log_probs.cpu().numpy(),
                rewards,
                costs,
                local,
                occupancies,
                membership * owner_mask,
                permute_assigned_target_blocks(next_obs, permutations),
                next_states,
                next_occupancies,
            )
            observations, states, occupancies = next_obs, next_states, next_occupancies
    actor.train()
    physical = np.asarray(physical_memberships, dtype=np.int8)
    physical.setflags(write=False)
    return buffer.freeze(), observations, states, occupancies, physical


def physical_occupancy_diagnostics(membership):
    """按真实位置统计 unique、duplicate 与 off，不受 owner 过滤影响。"""

    counts = np.sum(membership, axis=2)
    rates = {
        "unique_contribution": np.mean(
            np.sum(membership * (counts[:, :, None, :] == 1), axis=-1), axis=(0, 1)
        ),
        "duplicate_target": np.mean(
            np.sum(membership * (counts[:, :, None, :] > 1), axis=-1), axis=(0, 1)
        ),
        "non_target": np.mean(np.sum(membership, axis=-1) == 0, axis=(0, 1)),
    }
    return {
        f"{name}_agent_{index + 1}": float(value)
        for name, values in rates.items()
        for index, value in enumerate(values)
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluation_row(evaluation, payload, step, order):
    """将 owner-valid 指标与物理 duplicate/off 汇总为单行。"""

    aggregate = evaluation.aggregate
    physical = payload["agent_occupancy"]
    return {
        "environment_steps": step,
        "target_order": order,
        "reward": aggregate.steady_mean_reward,
        "cost": aggregate.steady_mean_cost,
        "objective": aggregate.steady_mean_scalarized_objective,
        "all_safe_rate": payload["aggregate"]["steady_owner_valid_all_safe_mode_rate"],
        "valid_distinct_targets": aggregate.steady_mean_num_distinct_targets,
        "duplicate_rate": float(np.mean(physical["duplicate_target_rate"])),
        "off_target_rate": float(np.mean(physical["non_target_rate"])),
        "entropy": aggregate.steady_mean_actor_entropy,
        **{
            f"occupancy_{target_id}": float(value)
            for target_id, value in zip(
                evaluation.target_ids, aggregate.steady_mean_target_occupancy, strict=True
            )
        },
    }


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "AssignedVectorEnv",
    "collect_rollout",
    "evaluation_row",
    "physical_occupancy_diagnostics",
    "write_json",
    "write_rows",
]
