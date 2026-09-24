"""固定/随机初态的连续 stochastic 评价及自身安全区到达、留守指标。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from hrmr.config import HRMRConfig
from hrmr.experiments.assigned_targets_ppo.environment import AssignedTargetsEnvironment
from hrmr.experiments.assigned_targets_ppo.evaluation import evaluate_assigned_actor
from hrmr.rl.evaluation import FixedLambdaEvaluation
from hrmr.rl.networks import IndependentActors

from .environment import (
    EVAL_RANDOM_DOMAIN,
    RandomInitialEvaluationEnvironment,
    evaluation_random_initial_positions,
)


def _arrival_metrics(
    evaluation: FixedLambdaEvaluation,
    config: HRMRConfig,
    assignments: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """重用已有 position_history，不运行额外轨迹或隐式重启。"""

    safe_targets = [
        next(target_id for target_id in pair if config.hazard_intensities[target_id] == 0.0)
        for pair in assignments
    ]
    per_seed = []
    for trajectory in evaluation.trajectories:
        records = []
        for index, target_id in enumerate(safe_targets):
            covered = np.asarray(
                [
                    tuple(position) in config.monitoring_regions[target_id]
                    for position in trajectory.position_history[:, index, :]
                ],
                dtype=np.bool_,
            )
            hits = np.flatnonzero(covered)
            first = int(hits[0]) if len(hits) else None
            records.append(
                {
                    "agent": f"A{index + 1}",
                    "assigned_safe_target": target_id,
                    "initially_in_assigned_safe_target": bool(covered[0]),
                    "arrival_success": first is not None,
                    "first_arrival_step": first,
                    "right_censored": first is None,
                    "observed_steps": len(covered) - 1,
                    "post_arrival_occupancy_retention": (
                        float(np.mean(covered[first:])) if first is not None else None
                    ),
                    "post_arrival_observed_states": len(covered) - first
                    if first is not None
                    else 0,
                    "steady_assigned_safe_occupancy": float(
                        np.mean(covered[trajectory.burn_in_steps + 1 :])
                    ),
                }
            )
        per_seed.append({"seed": trajectory.seed, "agents": records})
    aggregate = []
    for index, target_id in enumerate(safe_targets):
        records = [item["agents"][index] for item in per_seed]
        successful = [record for record in records if record["arrival_success"]]
        first_times = [record["first_arrival_step"] for record in successful]
        aggregate.append(
            {
                "agent": f"A{index + 1}",
                "assigned_safe_target": target_id,
                "arrival_success_rate": len(successful) / len(records),
                "num_successful_trajectories": len(successful),
                "num_censored_trajectories": len(records) - len(successful),
                "mean_first_arrival_step_successful_only": (
                    float(np.mean(first_times)) if first_times else None
                ),
                "median_first_arrival_step_successful_only": (
                    float(np.median(first_times)) if first_times else None
                ),
                "mean_post_arrival_occupancy_retention_successful_only": (
                    float(
                        np.mean([item["post_arrival_occupancy_retention"] for item in successful])
                    )
                    if successful
                    else None
                ),
                "steady_assigned_safe_occupancy": float(
                    np.mean([record["steady_assigned_safe_occupancy"] for record in records])
                ),
            }
        )
    return {
        "definitions": {
            "time_origin": "initial state is t=0; times include burn-in and all evaluation steps",
            "first_arrival": "first physical entry into the agent's assigned safe region",
            "censoring": "no arrival by trajectory end is null, not treated as zero travel time",
            "post_arrival_occupancy_retention": (
                "fraction of states in own safe region from first arrival through final state; "
                "including the arrival state; null if never reached; not uninterrupted survival"
            ),
            "aggregate_retention": "equal-weight mean over successful trajectories only",
            "steady_occupancy": "post-step states after burn-in, matching formal reward evaluation",
        },
        "per_seed": per_seed,
        "aggregate_by_agent": aggregate,
    }


def evaluate_restart_actor(
    actor: IndependentActors,
    environment_config: HRMRConfig,
    assignments: Sequence[Sequence[str]],
    fixed_lambda: float,
    seeds: Sequence[int],
    *,
    initial_state_mode: str = "fixed",
    evaluation_steps: int = 2_000,
    burn_in_steps: int = 100,
    device: str | torch.device | None = "cpu",
    target_order_mode: str = "canonical",
) -> tuple[FixedLambdaEvaluation, dict[str, Any]]:
    """正式评价不重启、不更新参数；仅切换初态采样方式。"""

    if initial_state_mode not in {"fixed", "random"}:
        raise ValueError("initial_state_mode must be 'fixed' or 'random'")
    seed_values = tuple(seeds)
    assignment_values = tuple(tuple(pair) for pair in assignments)
    evaluation, payload = evaluate_assigned_actor(
        actor,
        environment_config,
        assignment_values,
        fixed_lambda,
        seed_values,
        evaluation_steps=evaluation_steps,
        burn_in_steps=burn_in_steps,
        device=device,
        target_order_mode=target_order_mode,
        environment_class=(
            RandomInitialEvaluationEnvironment
            if initial_state_mode == "random"
            else AssignedTargetsEnvironment
        ),
    )
    actual_initial_positions = {
        str(trajectory.seed): trajectory.position_history[0].tolist()
        for trajectory in evaluation.trajectories
    }
    if initial_state_mode == "random":
        expected = evaluation_random_initial_positions(environment_config, seed_values)
        if actual_initial_positions != expected:
            raise AssertionError("random evaluation initial positions differ from frozen seed set")
    payload.update(
        experiment_kind="assigned_targets_random_restart_ppo",
        initial_state_mode=initial_state_mode,
        random_init_description=(
            "uniform without replacement over all grid cells; identities are ordered; "
            "safe, hazardous and off-target cells are all eligible"
            if initial_state_mode == "random"
            else "original fixed joint initial positions"
        ),
        random_initialization_rng_domain=(
            EVAL_RANDOM_DOMAIN if initial_state_mode == "random" else None
        ),
        initial_positions_coordinate_convention="zero-based internal [row, column]",
        initial_positions_by_seed=actual_initial_positions,
        no_periodic_resets=True,
        arrival_metrics=_arrival_metrics(evaluation, environment_config, assignment_values),
    )
    return evaluation, payload


__all__ = ["evaluate_restart_actor"]
