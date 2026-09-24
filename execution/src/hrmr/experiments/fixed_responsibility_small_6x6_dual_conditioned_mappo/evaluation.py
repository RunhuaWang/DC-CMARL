"""57D dual-conditioned actor 在 6×6 owner-aware 环境中的 stochastic 评价。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from hrmr.experiments.assigned_targets_ppo.environment import DEFAULT_ASSIGNMENTS
from hrmr.experiments.assigned_targets_ppo.evaluation import _coverage_diagnostics
from hrmr.experiments.assigned_targets_random_restart_ppo.environment import (
    EVAL_RANDOM_DOMAIN,
    evaluation_random_initial_positions,
)
from hrmr.experiments.assigned_targets_random_restart_ppo.evaluation import _arrival_metrics
from hrmr.experiments.fixed_responsibility_small_6x6_mappo.environment import (
    SmallSixBySixAssignedEnvironment,
    SmallSixBySixRandomEvaluationEnvironment,
)
from hrmr.rl.evaluation import evaluate_fixed_lambda
from hrmr.rl.formal_evaluation import (
    aggregate_evaluation_occupancy,
    fixed_random_target_permutations,
    formal_evaluation_payload,
)
from hrmr.rl.networks import IndependentActors

from .conditioning import condition_actor_observations, permute_conditioned_target_blocks
from .theory import all_mode_rates, analytic_reference, optimal_mode_mask


class DualConditionedEvaluationActor:
    """把 evaluator 的 56D observation 变成指定 λ 的 57D policy 输入。"""

    def __init__(
        self,
        actor: IndependentActors,
        dual_lambda: float,
        conditioning_max: float,
        permutations: np.ndarray | None = None,
    ) -> None:
        if not isinstance(actor, IndependentActors) or actor.input_dim != 57:
            raise ValueError("dual-conditioned evaluation requires independent 57D actors")
        self.actor = actor
        self.dual_lambda = float(dual_lambda)
        self.conditioning_max = float(conditioning_max)
        self.permutations = None if permutations is None else np.asarray(permutations).copy()
        self.num_agents = actor.num_agents
        self.num_actions = actor.num_actions
        # evaluator 输入仍来自环境，所以这里公开基础 observation 维数。
        self.input_dim = 56

    @property
    def training(self) -> bool:
        return bool(self.actor.training)

    def parameters(self) -> Any:
        return self.actor.parameters()

    def eval(self) -> DualConditionedEvaluationActor:
        self.actor.eval()
        return self

    def train(self, mode: bool = True) -> DualConditionedEvaluationActor:
        self.actor.train(mode)
        return self

    def distribution(self, observations: torch.Tensor) -> Any:
        if not isinstance(observations, torch.Tensor) or observations.shape[-1] != 56:
            raise ValueError("evaluation observations must have final dimension 56")
        conditioned = condition_actor_observations(
            observations.detach().cpu().numpy(),
            self.dual_lambda,
            self.conditioning_max,
        )
        if self.permutations is not None:
            conditioned = permute_conditioned_target_blocks(conditioned, self.permutations)
        policy_input = torch.as_tensor(
            conditioned,
            dtype=observations.dtype,
            device=observations.device,
        )
        return self.actor.distribution(policy_input)

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


def evaluate_dual_conditioned_actor(
    actor: IndependentActors,
    environment_config,
    dual_lambda: float,
    conditioning_max: float,
    seeds: Sequence[int],
    *,
    initial_state_mode: str = "random",
    target_order_mode: str = "canonical",
    evaluation_steps: int = 2_000,
    burn_in_steps: int = 100,
    device: str | torch.device = "cpu",
):
    """按一个给定 λ 评价共享条件策略，不更新模型或平均率。"""

    if initial_state_mode not in {"fixed", "random"}:
        raise ValueError("initial_state_mode must be fixed or random")
    if target_order_mode not in {"canonical", "random"}:
        raise ValueError("target_order_mode must be canonical or random")
    seed_values = tuple(seeds)
    permutations = (
        fixed_random_target_permutations(seed_values, num_targets=8)
        if target_order_mode == "random"
        else None
    )
    evaluated_actor = DualConditionedEvaluationActor(
        actor,
        dual_lambda,
        conditioning_max,
        permutations,
    )
    environment_class = (
        SmallSixBySixRandomEvaluationEnvironment
        if initial_state_mode == "random"
        else SmallSixBySixAssignedEnvironment
    )

    def environment_factory():
        return environment_class(environment_config, assignments=DEFAULT_ASSIGNMENTS)

    parameter_snapshot = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    evaluation = evaluate_fixed_lambda(
        evaluated_actor,
        environment_factory,
        dual_lambda,
        seed_values,
        evaluation_steps=evaluation_steps,
        burn_in_steps=burn_in_steps,
        device=device,
        operating_mode_mask_fn=optimal_mode_mask,
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
        if permutations is not None
        else None
    )
    payload = formal_evaluation_payload(
        target_order_mode,
        evaluation,
        occupancy,
        source_target_order_by_seed=source_orders,
    )
    coverage = _coverage_diagnostics(evaluation, environment_config)
    steady_occupancies = np.concatenate(
        [
            trajectory.state_history[trajectory.burn_in_steps + 1 :, -8:]
            for trajectory in evaluation.trajectories
        ],
        axis=0,
    )
    modes = all_mode_rates(steady_occupancies, tuple(environment_config.target_ids))
    payload["aggregate"].update(**coverage, steady_operating_mode_rates=modes)
    matrix = occupancy.mean.agent_target_occupancy_matrix
    assigned_rates = np.asarray(
        [
            sum(matrix[agent, evaluation.target_ids.index(target)] for target in pair)
            for agent, pair in enumerate(DEFAULT_ASSIGNMENTS)
        ]
    )
    payload["agent_occupancy"].update(
        assigned_target_rate=assigned_rates.tolist(),
        unassigned_target_rate=np.maximum(0.0, np.sum(matrix, axis=1) - assigned_rates).tolist(),
    )
    initial_positions = {
        str(trajectory.seed): trajectory.position_history[0].tolist()
        for trajectory in evaluation.trajectories
    }
    if initial_state_mode == "random":
        expected = evaluation_random_initial_positions(environment_config, seed_values)
        if initial_positions != expected:
            raise AssertionError("random evaluation starts differ from frozen seed set")
    payload.update(
        experiment_kind="fixed_responsibility_small_6x6_dual_conditioned_mappo",
        dual_lambda=float(dual_lambda),
        normalized_dual=float(dual_lambda) / float(conditioning_max),
        conditioning_max=float(conditioning_max),
        analytic_reference=analytic_reference(dual_lambda).to_dict(),
        assignments=[list(pair) for pair in DEFAULT_ASSIGNMENTS],
        initial_state_mode=initial_state_mode,
        random_initialization_rng_domain=(
            EVAL_RANDOM_DOMAIN if initial_state_mode == "random" else None
        ),
        initial_positions_coordinate_convention="zero-based internal [row, column]",
        initial_positions_by_seed=initial_positions,
        no_periodic_resets=True,
        assigned_safe_arrival_metrics=_arrival_metrics(
            evaluation, environment_config, DEFAULT_ASSIGNMENTS
        ),
    )
    return evaluation, payload


__all__ = ["DualConditionedEvaluationActor", "evaluate_dual_conditioned_actor"]
