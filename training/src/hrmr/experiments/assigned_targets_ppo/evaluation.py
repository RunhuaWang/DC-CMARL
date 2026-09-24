"""固定责任区域的随机策略评价，分别报告奖励资格覆盖与物理占用。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from hrmr.config import HRMRConfig
from hrmr.constants import SAFE_TARGET_IDS
from hrmr.experiments.unnormalized_original_parameters_ppo.analytic_modes import analytic_mode
from hrmr.experiments.unnormalized_original_parameters_ppo.evaluation import (
    original_parameters_operating_mode_mask,
)
from hrmr.rl.evaluation import FixedLambdaEvaluation, evaluate_fixed_lambda
from hrmr.rl.formal_evaluation import (
    aggregate_evaluation_occupancy,
    fixed_random_target_permutations,
    formal_evaluation_payload,
)
from hrmr.rl.networks import IndependentActors
from hrmr.rl.target_permutation import validate_target_permutation

from .environment import AssignedTargetsEnvironment, permute_assigned_target_blocks


class AssignedPermutationEvaluationActor:
    """每条 seed 轨迹固定排列全部六维 target blocks，包括 owner 标记。"""

    def __init__(self, actor: IndependentActors, permutations: ArrayLike) -> None:
        if not isinstance(actor, IndependentActors) or actor.input_dim != 56:
            raise ValueError("assigned evaluation requires independent 56D actors")
        raw = np.asarray(permutations)
        if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != 8:
            raise ValueError("permutations must have shape [num_seeds,8]")
        self.permutations = np.stack(
            [validate_target_permutation(item, num_targets=8) for item in raw]
        )
        self.actor = actor
        self.num_agents = actor.num_agents
        self.num_actions = actor.num_actions
        self.input_dim = actor.input_dim

    @property
    def training(self) -> bool:
        return bool(self.actor.training)

    def parameters(self) -> Any:
        return self.actor.parameters()

    def eval(self) -> AssignedPermutationEvaluationActor:
        self.actor.eval()
        return self

    def train(self, mode: bool = True) -> AssignedPermutationEvaluationActor:
        self.actor.train(mode)
        return self

    def distribution(self, observations: torch.Tensor) -> Any:
        expected = (len(self.permutations), self.num_agents, self.input_dim)
        if tuple(observations.shape) != expected:
            raise ValueError(f"assigned evaluation observations must have shape {expected}")
        # 评价无梯度；复用环境的完整 block 变换，保持训练/评价布局一致。
        values = observations.detach().cpu().numpy()
        permuted = permute_assigned_target_blocks(values, self.permutations)
        tensor = torch.as_tensor(permuted, dtype=observations.dtype, device=observations.device)
        return self.actor.distribution(tensor)

    def sample(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        actions = (
            torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        )
        return actions, distribution.log_prob(actions), distribution.entropy()


def _coverage_diagnostics(
    evaluation: FixedLambdaEvaluation,
    config: HRMRConfig,
) -> dict[str, Any]:
    """用稳态 post-step 状态取有效覆盖，以真实位置重建物理覆盖。"""

    target_ids = evaluation.target_ids
    safe_indices = [target_ids.index(target_id) for target_id in SAFE_TARGET_IDS]
    lookup = {
        position: index
        for index, target_id in enumerate(target_ids)
        for position in config.monitoring_regions[target_id]
    }
    valid_histories = []
    physical_histories = []
    for trajectory in evaluation.trajectories:
        steady_slice = slice(trajectory.burn_in_steps + 1, None)
        valid_histories.append(trajectory.state_history[steady_slice, -len(target_ids) :])
        positions = trajectory.position_history[steady_slice]
        memberships = np.asarray(
            [
                [lookup.get(tuple(position), -1) for position in joint_positions]
                for joint_positions in positions
            ],
            dtype=np.int64,
        )
        physical_histories.append(
            np.any(memberships[..., None] == np.arange(len(target_ids)), axis=1)
        )
    valid = np.stack(valid_histories)
    physical = np.stack(physical_histories)
    return {
        "steady_all_safe_mode_rate": float(np.mean(np.all(valid[..., safe_indices] == 1, axis=-1))),
        "steady_owner_valid_all_safe_mode_rate": float(
            np.mean(np.all(valid[..., safe_indices] == 1, axis=-1))
        ),
        "steady_owner_valid_mean_num_distinct_targets": float(np.mean(np.sum(valid, axis=-1))),
        "steady_physical_all_safe_mode_rate": float(
            np.mean(np.all(physical[..., safe_indices], axis=-1))
        ),
        "steady_physical_mean_num_distinct_targets": float(np.mean(np.sum(physical, axis=-1))),
        "steady_physical_target_occupancy": np.mean(physical, axis=(0, 1)).tolist(),
    }


def evaluate_assigned_actor(
    actor: IndependentActors,
    environment_config: HRMRConfig,
    assignments: Sequence[Sequence[str]],
    fixed_lambda: float,
    seeds: Sequence[int],
    *,
    evaluation_steps: int = 2_000,
    burn_in_steps: int = 100,
    device: str | torch.device | None = "cpu",
    target_order_mode: str = "canonical",
    environment_class: type[AssignedTargetsEnvironment] = AssignedTargetsEnvironment,
) -> tuple[FixedLambdaEvaluation, dict[str, Any]]:
    """评价单个排列模式；调用方提供独立于训练的共同评价 seeds。

    正式指标始终使用 stochastic sampling。通用 evaluator 隔离 Torch RNG、
    恢复 actor 的 training 状态；环境只处理原始信号，不接收 λ。
    """

    if target_order_mode not in {"canonical", "random"}:
        raise ValueError("target_order_mode must be 'canonical' or 'random'")
    if not isinstance(actor, IndependentActors):
        raise TypeError("assigned evaluation requires IndependentActors")
    if actor.input_dim != 56 or actor.num_agents != environment_config.num_agents:
        raise ValueError("assigned evaluation requires four independent 56D actors")
    lambda_value = analytic_mode(fixed_lambda).fixed_lambda
    # 同时验证 seeds 并使用独立 NumPy RNG 生成可复现的排列。
    seed_values = tuple(seeds)
    permutations = fixed_random_target_permutations(seed_values, num_targets=8)
    assignment_values = tuple(tuple(pair) for pair in assignments)
    if not issubclass(environment_class, AssignedTargetsEnvironment):
        raise TypeError("evaluation requires an assigned-target environment class")
    environment_class(environment_config, assignments=assignment_values)

    def environment_factory() -> AssignedTargetsEnvironment:
        return environment_class(environment_config, assignments=assignment_values)

    evaluated_actor = (
        AssignedPermutationEvaluationActor(actor, permutations)
        if target_order_mode == "random"
        else actor
    )
    parameter_snapshot = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    evaluation = evaluate_fixed_lambda(
        evaluated_actor,
        environment_factory,
        lambda_value,
        seed_values,
        evaluation_steps=evaluation_steps,
        burn_in_steps=burn_in_steps,
        device=device,
        operating_mode_mask_fn=original_parameters_operating_mode_mask,
    )
    if any(
        not torch.equal(value.detach().cpu(), parameter_snapshot[name])
        for name, value in actor.state_dict().items()
    ):
        raise AssertionError("evaluation modified actor parameters")

    occupancy = aggregate_evaluation_occupancy(evaluation, environment_config)
    source_orders = (
        {
            seed: tuple(environment_config.target_ids[index] for index in permutation)
            for seed, permutation in zip(seed_values, permutations, strict=True)
        }
        if target_order_mode == "random"
        else None
    )
    payload = formal_evaluation_payload(
        target_order_mode,
        evaluation,
        occupancy,
        source_target_order_by_seed=source_orders,
    )
    payload.update(
        experiment_kind="assigned_targets_ppo",
        target_ids=list(evaluation.target_ids),
        assignments=[list(pair) for pair in assignment_values],
        coverage_semantics={
            "target_occupancy": "owner-valid reward coverage",
            "agent_occupancy": "physical occupancy, independent of reward ownership",
            "cost": "raw additive hazard exposure in every hazardous region",
        },
    )
    aggregate = evaluation.aggregate
    payload["aggregate"].update(
        steady_optimal_mode_rate=aggregate.steady_mean_operating_mode_rate,
        steady_mean_safe_targets_covered=aggregate.steady_mean_safe_targets_covered,
        steady_mean_hazardous_targets_covered=aggregate.steady_mean_hazardous_targets_covered,
        steady_std_reward=aggregate.steady_std_reward,
        steady_std_cost=aggregate.steady_std_cost,
        **_coverage_diagnostics(evaluation, environment_config),
    )
    matrix = occupancy.mean.agent_target_occupancy_matrix
    assigned_rates = np.asarray(
        [
            sum(matrix[agent, evaluation.target_ids.index(target_id)] for target_id in pair)
            for agent, pair in enumerate(assignment_values)
        ]
    )
    payload["agent_occupancy"].update(
        assigned_target_rate=assigned_rates.tolist(),
        unassigned_target_rate=np.maximum(0.0, np.sum(matrix, axis=1) - assigned_rates).tolist(),
    )
    return evaluation, payload


__all__ = ["AssignedPermutationEvaluationActor", "evaluate_assigned_actor"]
