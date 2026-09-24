"""原始异质参数、未归一化信号实验的 stochastic 评价。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from hrmr.config import HRMRConfig
from hrmr.rl.evaluation import FixedLambdaEvaluation, evaluate_fixed_lambda
from hrmr.rl.formal_evaluation import (
    CANONICAL_TARGET_ORDER_MODE,
    RANDOM_TARGET_ORDER_MODE,
    FixedPermutationEvaluationActor,
    FormalEvaluationResult,
    agent_target_occupancy_rows,
    aggregate_evaluation_occupancy,
    evaluation_comparison_row,
    fixed_random_target_permutations,
    formal_evaluation_payload,
)
from hrmr.rl.networks import IndependentActors

from .analytic_modes import analytic_mode, assess_mode, optimal_mode_mask
from .environment import (
    UnnormalizedOriginalParametersEnvironment,
    validate_unnormalized_original_environment_config,
)


def original_parameters_operating_mode_mask(
    target_occupancies: ArrayLike,
    target_ids: Sequence[str],
    *,
    fixed_lambda: float,
) -> np.ndarray:
    """将通用 evaluator 的 callback 接到本实验五级解析 mode。"""

    return optimal_mode_mask(
        target_occupancies,
        tuple(target_ids),
        fixed_lambda=fixed_lambda,
    )


def _paired_row(
    order: str,
    evaluation: FixedLambdaEvaluation,
    occupancy: Any,
) -> dict[str, Any]:
    row = evaluation_comparison_row(order, evaluation, occupancy)
    aggregate = evaluation.aggregate
    row.update(
        optimal_mode_rate=aggregate.steady_mean_operating_mode_rate,
        mean_safe_targets=aggregate.steady_mean_safe_targets_covered,
        mean_hazardous_targets=aggregate.steady_mean_hazardous_targets_covered,
    )
    return row


def evaluate_unnormalized_original_parameters_actor(
    actor: IndependentActors,
    environment_config: HRMRConfig,
    fixed_lambda: float,
    evaluation_seeds: Sequence[int],
    *,
    evaluation_steps: int = 2_000,
    burn_in_steps: int = 100,
    device: str | torch.device | None = None,
    random_permutations: ArrayLike | None = None,
) -> FormalEvaluationResult:
    """执行 canonical/random-order 配对 stochastic 评价。

    Reward/cost 与 scalarized objective 均使用未归一化实际单位；只有 operating
    mode 判定通过 callback 指向本实验的五级解析表。
    """

    if not isinstance(actor, IndependentActors):
        raise TypeError("evaluation requires IndependentActors")
    validate_unnormalized_original_environment_config(environment_config)
    if actor.input_dim != 48 or actor.num_agents != environment_config.num_agents:
        raise ValueError("actor must contain four independent 48D policies")
    # 在启动任何 rollout 前验证 λ，避免无效请求产生部分评价结果。
    lambda_value = analytic_mode(fixed_lambda).fixed_lambda
    seeds = tuple(int(seed) for seed in evaluation_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be unique non-negative integers")
    permutations = (
        fixed_random_target_permutations(seeds, num_targets=environment_config.num_targets)
        if random_permutations is None
        else np.asarray(random_permutations, dtype=np.int64)
    )
    if permutations.shape != (len(seeds), environment_config.num_targets):
        raise ValueError("random_permutations has the wrong shape")

    def environment_factory() -> UnnormalizedOriginalParametersEnvironment:
        return UnnormalizedOriginalParametersEnvironment(environment_config)

    parameter_snapshot = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    canonical = evaluate_fixed_lambda(
        actor,
        environment_factory,
        lambda_value,
        seeds,
        evaluation_steps=evaluation_steps,
        burn_in_steps=burn_in_steps,
        device=device,
        operating_mode_mask_fn=original_parameters_operating_mode_mask,
    )
    random_actor = FixedPermutationEvaluationActor(
        actor,
        permutations,
        num_targets=environment_config.num_targets,
        use_agent_id=False,
    )
    random_order = evaluate_fixed_lambda(
        random_actor,
        environment_factory,
        lambda_value,
        seeds,
        evaluation_steps=evaluation_steps,
        burn_in_steps=burn_in_steps,
        device=device,
        operating_mode_mask_fn=original_parameters_operating_mode_mask,
    )
    if not canonical.is_formal_evaluation or not random_order.is_formal_evaluation:
        raise AssertionError("formal ablation evaluation must be stochastic")
    if any(
        not torch.equal(value.detach().cpu(), parameter_snapshot[name])
        for name, value in actor.state_dict().items()
    ):
        raise AssertionError("evaluation modified actor parameters")

    canonical_occupancy = aggregate_evaluation_occupancy(canonical, environment_config)
    random_occupancy = aggregate_evaluation_occupancy(random_order, environment_config)
    target_orders = {
        seed: tuple(environment_config.target_ids[index] for index in permutation)
        for seed, permutation in zip(seeds, permutations, strict=True)
    }
    canonical_payload = formal_evaluation_payload(
        CANONICAL_TARGET_ORDER_MODE,
        canonical,
        canonical_occupancy,
    )
    random_payload = formal_evaluation_payload(
        RANDOM_TARGET_ORDER_MODE,
        random_order,
        random_occupancy,
        source_target_order_by_seed=target_orders,
    )
    return FormalEvaluationResult(
        canonical=canonical,
        random_order=random_order,
        canonical_occupancy=canonical_occupancy,
        random_order_occupancy=random_occupancy,
        random_permutations=permutations,
        canonical_payload=canonical_payload,
        random_order_payload=random_payload,
        comparison_rows=(
            _paired_row(CANONICAL_TARGET_ORDER_MODE, canonical, canonical_occupancy),
            _paired_row(RANDOM_TARGET_ORDER_MODE, random_order, random_occupancy),
        ),
        agent_target_occupancy_rows=(
            agent_target_occupancy_rows(
                CANONICAL_TARGET_ORDER_MODE,
                canonical_occupancy,
            )
            + agent_target_occupancy_rows(
                RANDOM_TARGET_ORDER_MODE,
                random_occupancy,
            )
        ),
    )


def evaluation_payload(
    evaluation: FixedLambdaEvaluation,
    occupancy_payload: Mapping[str, Any],
    *,
    training_seed: int,
    n_step: int,
    target_order_mode: str,
) -> dict[str, Any]:
    """构造不引用正式归一化解析数值的评价 JSON。"""

    aggregate = evaluation.aggregate
    mode = analytic_mode(evaluation.fixed_lambda)
    assessment = assess_mode(
        evaluation.fixed_lambda,
        aggregate.steady_mean_target_occupancy,
        mean_num_distinct_targets=aggregate.steady_mean_num_distinct_targets,
        optimal_mode_rate=aggregate.steady_mean_operating_mode_rate,
        steady_reward=aggregate.steady_mean_reward,
        steady_cost=aggregate.steady_mean_cost,
    )
    trajectories = [
        {
            "evaluation_seed": trajectory.seed,
            "steady_reward": trajectory.steady_reward,
            "steady_cost": trajectory.steady_cost,
            "steady_scalarized_objective": trajectory.steady_scalarized_objective,
            "steady_mean_num_distinct_targets": trajectory.steady_mean_num_distinct_targets,
            "steady_optimal_mode_rate": trajectory.steady_operating_mode_rate,
            "steady_all_hazardous_mode_rate": trajectory.steady_all_hazardous_mode_rate,
            "steady_target_occupancy": trajectory.steady_target_occupancy.tolist(),
        }
        for trajectory in evaluation.trajectories
    ]
    return {
        "experiment_kind": "unnormalized_original_parameters_ppo_ablation",
        "fixed_lambda": evaluation.fixed_lambda,
        "training_seed": int(training_seed),
        "n_step": int(n_step),
        "actor_target_order_mode": target_order_mode,
        "sampling_mode": evaluation.sampling_mode,
        "formal_stochastic_evaluation": evaluation.is_formal_evaluation,
        "evaluation_seeds": list(evaluation.evaluation_seeds),
        "burn_in_steps": evaluation.burn_in_steps,
        "evaluation_steps": evaluation.evaluation_steps,
        "target_ids": list(evaluation.target_ids),
        "signal_contract": {
            "team_reward": "sum_m target_value[m] * coverage[m]",
            "agent_local_cost": "hazard_intensity_at_agent_position",
            "global_cost": "sum_i agent_local_cost[i]",
            "target_parameters": "current formal heterogeneous values/hazards",
            "reward_range": [0.0, 8.0],
            "global_cost_range": [0.0, 3.2],
            "reward_normalized": False,
            "cost_normalized_by_num_agents": False,
        },
        "aggregate": {
            "whole_mean_reward": aggregate.whole_mean_reward,
            "whole_mean_cost": aggregate.whole_mean_cost,
            "whole_mean_scalarized_objective": aggregate.whole_mean_scalarized_objective,
            "steady_mean_reward": aggregate.steady_mean_reward,
            "steady_mean_cost": aggregate.steady_mean_cost,
            "steady_mean_scalarized_objective": aggregate.steady_mean_scalarized_objective,
            "steady_mean_num_distinct_targets": aggregate.steady_mean_num_distinct_targets,
            "steady_optimal_mode_rate": aggregate.steady_mean_operating_mode_rate,
            "steady_all_hazardous_mode_rate": aggregate.steady_mean_all_hazardous_mode_rate,
            "steady_mean_safe_targets_covered": aggregate.steady_mean_safe_targets_covered,
            "steady_mean_hazardous_targets_covered": (
                aggregate.steady_mean_hazardous_targets_covered
            ),
            "steady_stay_action_rate": aggregate.steady_mean_stay_action_rate,
            "steady_actor_entropy": aggregate.steady_mean_actor_entropy,
            "steady_mean_target_occupancy": aggregate.steady_mean_target_occupancy.tolist(),
            "steady_mean_local_costs": aggregate.steady_mean_local_costs.tolist(),
        },
        "analytic_reference": mode.to_dict(),
        "mode_assessment": assessment,
        "scalarized_optimality_gap": assessment["scalarized_optimality_gap"],
        "agent_occupancy": dict(occupancy_payload),
        "trajectories": trajectories,
    }


__all__ = [
    "evaluate_unnormalized_original_parameters_actor",
    "evaluation_payload",
    "original_parameters_operating_mode_mask",
]
