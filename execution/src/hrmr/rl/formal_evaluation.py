"""Independent fixed-lambda checkpoint 的 canonical/random-order 正式评价。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from hrmr.config import HRMRConfig, load_config, validate_config
from hrmr.constants import HAZARDOUS_TARGET_IDS, SAFE_TARGET_IDS
from hrmr.environment import HRMREnvironment
from hrmr.observations import TARGET_FEATURE_SIZE, observation_layout
from hrmr.rl.agent_occupancy import (
    AgentTargetOccupancyAggregate,
    AgentTargetOccupancyStatistics,
    aggregate_agent_target_occupancy,
    summarize_agent_target_occupancy,
)
from hrmr.rl.checkpointing import CHECKPOINT_FORMAT_VERSION
from hrmr.rl.device import resolve_device
from hrmr.rl.evaluation import FixedLambdaEvaluation, evaluate_fixed_lambda
from hrmr.rl.networks import IndependentActors
from hrmr.rl.safe_actor_view import (
    SAFE_ACTOR_NUM_TARGETS,
    SAFE_ACTOR_OBSERVATION_DIM,
    SafeTargetActorView,
)
from hrmr.rl.target_permutation import validate_target_permutation

IntArray = NDArray[np.int64]

RANDOM_EVALUATION_PERMUTATION_STREAM = 0x4556414C
CANONICAL_TARGET_ORDER_MODE = "canonical"
RANDOM_TARGET_ORDER_MODE = "random"


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _positive_integer(value: Any, *, name: str) -> int:
    checked = _nonnegative_integer(value, name=name)
    if checked == 0:
        raise ValueError(f"{name} must be positive")
    return checked


def _fixed_lambda(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("fixed_lambda must be a real number")
    checked = float(value)
    if not isfinite(checked) or checked < 0.0:
        raise ValueError("fixed_lambda must be finite and non-negative")
    return checked


def _evaluation_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(_nonnegative_integer(value, name="evaluation seed") for value in values)
    if not seeds:
        raise ValueError("evaluation_seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation_seeds must be unique")
    return seeds


def fixed_random_target_permutations(
    evaluation_seeds: Sequence[int],
    *,
    num_targets: int,
    stream: int = RANDOM_EVALUATION_PERMUTATION_STREAM,
) -> IntArray:
    """为每个 evaluation seed 确定性地产生一个完整 target-block 排列。"""

    seeds = _evaluation_seeds(evaluation_seeds)
    targets = _positive_integer(num_targets, name="num_targets")
    stream_id = _nonnegative_integer(stream, name="stream")
    return np.stack(
        [
            np.random.default_rng(np.random.SeedSequence((seed, stream_id))).permutation(targets)
            for seed in seeds
        ]
    ).astype(np.int64, copy=False)


def _checked_permutations(
    values: ArrayLike,
    *,
    num_seeds: int,
    num_targets: int,
) -> IntArray:
    raw = np.asarray(values)
    expected_shape = (num_seeds, num_targets)
    if raw.shape != expected_shape:
        raise ValueError(f"random_permutations must have shape {expected_shape}")
    checked = np.stack(
        [validate_target_permutation(row, num_targets=num_targets) for row in raw]
    ).astype(np.int64, copy=False)
    checked.setflags(write=False)
    return checked


class FixedPermutationEvaluationActor:
    """在 batched evaluator 中为每条 seed trajectory 固定一个 target 排列。"""

    def __init__(
        self,
        actor: IndependentActors,
        permutations: ArrayLike,
        *,
        num_targets: int,
        use_agent_id: bool = False,
    ) -> None:
        if not isinstance(actor, IndependentActors):
            raise TypeError("formal random-order evaluation requires IndependentActors")
        if not isinstance(use_agent_id, bool):
            raise TypeError("use_agent_id must be a boolean")
        targets = _positive_integer(num_targets, name="num_targets")
        raw = np.asarray(permutations)
        if raw.ndim != 2:
            raise ValueError("permutations must have shape [num_seeds,num_targets]")
        self.permutations = _checked_permutations(
            raw,
            num_seeds=raw.shape[0],
            num_targets=targets,
        )
        layout = observation_layout(
            use_agent_id=use_agent_id,
            num_agents=actor.num_agents,
            num_targets=targets,
        )
        expected_input_dim = max(feature_slice.stop for feature_slice in layout.values())
        if actor.input_dim != expected_input_dim:
            raise ValueError(
                "actor input dimension does not match the requested observation layout"
            )
        self.actor = actor
        self.num_agents = actor.num_agents
        self.num_actions = actor.num_actions
        self.input_dim = actor.input_dim
        self.num_targets = targets
        self.target_slice = layout["target_features"]

    @property
    def training(self) -> bool:
        return bool(self.actor.training)

    def parameters(self) -> Any:
        return self.actor.parameters()

    def eval(self) -> FixedPermutationEvaluationActor:
        self.actor.eval()
        return self

    def train(self, mode: bool = True) -> FixedPermutationEvaluationActor:
        self.actor.train(mode)
        return self

    def _permuted_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor")
        expected_shape = (
            len(self.permutations),
            self.num_agents,
            self.input_dim,
        )
        if tuple(observations.shape) != expected_shape:
            raise ValueError(
                "fixed-permutation evaluation observations must have shape "
                f"{expected_shape}; got {tuple(observations.shape)}"
            )
        result = observations.clone()
        blocks = result[..., self.target_slice].reshape(
            len(self.permutations),
            self.num_agents,
            self.num_targets,
            TARGET_FEATURE_SIZE,
        )
        permutation_tensor = torch.tensor(
            self.permutations,
            dtype=torch.long,
            device=observations.device,
        )
        indices = permutation_tensor[:, None, :, None].expand_as(blocks)
        result[..., self.target_slice] = torch.gather(blocks, dim=2, index=indices).reshape(
            len(self.permutations),
            self.num_agents,
            self.num_targets * TARGET_FEATURE_SIZE,
        )
        return result

    def distribution(self, observations: torch.Tensor) -> Any:
        return self.actor.distribution(self._permuted_observations(observations))

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


def aggregate_evaluation_occupancy(
    evaluation: FixedLambdaEvaluation,
    environment_config: HRMRConfig,
) -> AgentTargetOccupancyAggregate:
    """从正式 trajectories 聚合 4x8 matrix 与 unique/duplicate/non-target rates。"""

    if not isinstance(evaluation, FixedLambdaEvaluation):
        raise TypeError("evaluation must be a FixedLambdaEvaluation")
    if not evaluation.is_formal_evaluation:
        raise ValueError("occupancy payload requires stochastic formal evaluation")
    validate_config(environment_config)
    if tuple(evaluation.target_ids) != tuple(environment_config.target_ids):
        raise ValueError("evaluation and environment target orders do not match")
    per_seed = tuple(
        summarize_agent_target_occupancy(
            trajectory.position_history,
            environment_config.monitoring_regions,
            environment_config.target_ids,
            HAZARDOUS_TARGET_IDS,
            burn_in_steps=trajectory.burn_in_steps,
            evaluation_steps=trajectory.evaluation_steps,
        )
        for trajectory in evaluation.trajectories
    )
    return aggregate_agent_target_occupancy(per_seed)


def _occupancy_statistics_payload(
    values: AgentTargetOccupancyStatistics,
) -> dict[str, Any]:
    return {
        "agent_target_occupancy_matrix": values.agent_target_occupancy_matrix.tolist(),
        "unique_target_contribution_rate": (
            values.unique_target_marginal_contribution_rate.tolist()
        ),
        "duplicate_target_rate": values.duplicate_target_occupancy_rate.tolist(),
        "non_target_rate": values.non_target_occupancy_rate.tolist(),
        "hazard_occupancy_rate": values.hazard_occupancy_rate.tolist(),
    }


def agent_target_occupancy_payload(
    occupancy: AgentTargetOccupancyAggregate,
) -> dict[str, Any]:
    """把 occupancy aggregate 转为 JSON-safe payload。"""

    return {
        "num_evaluation_seeds": occupancy.num_trajectories,
        "target_ids": list(occupancy.target_ids),
        **_occupancy_statistics_payload(occupancy.mean),
        "seed_sample_std": _occupancy_statistics_payload(occupancy.sample_std),
        "ci95_half_width": _occupancy_statistics_payload(occupancy.ci95_half_width),
    }


def _actor_entropy_by_agent(evaluation: FixedLambdaEvaluation) -> tuple[float, ...]:
    steady_entropies = np.concatenate(
        [
            trajectory.entropy_history[trajectory.burn_in_steps :][None, ...]
            for trajectory in evaluation.trajectories
        ],
        axis=0,
    )
    return tuple(float(value) for value in np.mean(steady_entropies, axis=(0, 1)))


def formal_evaluation_payload(
    target_order_mode: str,
    evaluation: FixedLambdaEvaluation,
    occupancy: AgentTargetOccupancyAggregate,
    *,
    source_target_order_by_seed: Mapping[int, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """构造一套 canonical 或 random-order 正式评价的 JSON-safe payload。"""

    if target_order_mode not in {CANONICAL_TARGET_ORDER_MODE, RANDOM_TARGET_ORDER_MODE}:
        raise ValueError("target_order_mode must be 'canonical' or 'random'")
    if not evaluation.is_formal_evaluation:
        raise ValueError("formal payload requires stochastic evaluation")
    aggregate = evaluation.aggregate
    entropy_by_agent = _actor_entropy_by_agent(evaluation)
    payload: dict[str, Any] = {
        "fixed_lambda": evaluation.fixed_lambda,
        "actor_target_order_mode": target_order_mode,
        "sampling_mode": evaluation.sampling_mode,
        "formal_stochastic_evaluation": evaluation.is_formal_evaluation,
        "evaluation_seeds": list(evaluation.evaluation_seeds),
        "burn_in_steps": evaluation.burn_in_steps,
        "evaluation_steps": evaluation.evaluation_steps,
        "aggregate": {
            "whole_mean_reward": aggregate.whole_mean_reward,
            "whole_mean_cost": aggregate.whole_mean_cost,
            "whole_mean_scalarized_objective": aggregate.whole_mean_scalarized_objective,
            "steady_mean_reward": aggregate.steady_mean_reward,
            "steady_mean_cost": aggregate.steady_mean_cost,
            "steady_mean_scalarized_objective": aggregate.steady_mean_scalarized_objective,
            "steady_mean_num_distinct_targets": aggregate.steady_mean_num_distinct_targets,
            "steady_all_hazardous_mode_rate": aggregate.steady_mean_all_hazardous_mode_rate,
            "steady_stay_action_rate": aggregate.steady_mean_stay_action_rate,
            "steady_actor_entropy": aggregate.steady_mean_actor_entropy,
            "steady_actor_entropy_by_agent": list(entropy_by_agent),
            "whole_mean_target_occupancy": aggregate.whole_mean_target_occupancy.tolist(),
            "steady_mean_target_occupancy": aggregate.steady_mean_target_occupancy.tolist(),
            "steady_mean_local_costs": aggregate.steady_mean_local_costs.tolist(),
        },
        "agent_occupancy": agent_target_occupancy_payload(occupancy),
    }
    if source_target_order_by_seed is not None:
        payload["source_target_order_in_actor_slots_by_seed"] = {
            str(seed): list(order) for seed, order in source_target_order_by_seed.items()
        }
    return payload


def evaluation_comparison_row(
    target_order_mode: str,
    evaluation: FixedLambdaEvaluation,
    occupancy: AgentTargetOccupancyAggregate,
) -> dict[str, Any]:
    """生成 canonical/random comparison CSV 的一行。"""

    if target_order_mode not in {CANONICAL_TARGET_ORDER_MODE, RANDOM_TARGET_ORDER_MODE}:
        raise ValueError("target_order_mode must be 'canonical' or 'random'")
    aggregate = evaluation.aggregate
    entropy_by_agent = _actor_entropy_by_agent(evaluation)
    return {
        "fixed_lambda": evaluation.fixed_lambda,
        "target_order_mode": target_order_mode,
        "steady_reward": aggregate.steady_mean_reward,
        "steady_cost": aggregate.steady_mean_cost,
        "steady_scalarized_objective": aggregate.steady_mean_scalarized_objective,
        "mean_num_distinct_targets": aggregate.steady_mean_num_distinct_targets,
        "all_hazardous_mode_rate": aggregate.steady_mean_all_hazardous_mode_rate,
        "stay_action_rate": aggregate.steady_mean_stay_action_rate,
        "actor_entropy": aggregate.steady_mean_actor_entropy,
        "unique_contribution_rate": float(
            np.mean(occupancy.mean.unique_target_marginal_contribution_rate)
        ),
        "duplicate_rate": float(np.mean(occupancy.mean.duplicate_target_occupancy_rate)),
        "non_target_rate": float(np.mean(occupancy.mean.non_target_occupancy_rate)),
        **{
            f"actor_{agent_index + 1}_entropy": value
            for agent_index, value in enumerate(entropy_by_agent)
        },
        **{
            f"occupancy_{target_id}": float(value)
            for target_id, value in zip(
                evaluation.target_ids,
                aggregate.steady_mean_target_occupancy,
                strict=True,
            )
        },
    }


def agent_target_occupancy_rows(
    target_order_mode: str,
    occupancy: AgentTargetOccupancyAggregate,
) -> tuple[dict[str, Any], ...]:
    """生成正式 agent-target occupancy CSV rows。"""

    if target_order_mode not in {CANONICAL_TARGET_ORDER_MODE, RANDOM_TARGET_ORDER_MODE}:
        raise ValueError("target_order_mode must be 'canonical' or 'random'")
    matrix = occupancy.mean.agent_target_occupancy_matrix
    rows = []
    for agent_index in range(occupancy.mean.num_agents):
        rows.append(
            {
                "target_order_mode": target_order_mode,
                "agent_id": agent_index + 1,
                **{
                    f"occupancy_{target_id}": float(matrix[agent_index, target_index])
                    for target_index, target_id in enumerate(occupancy.target_ids)
                },
                "unique_target_contribution_rate": float(
                    occupancy.mean.unique_target_marginal_contribution_rate[agent_index]
                ),
                "duplicate_target_rate": float(
                    occupancy.mean.duplicate_target_occupancy_rate[agent_index]
                ),
                "non_target_rate": float(occupancy.mean.non_target_occupancy_rate[agent_index]),
                "hazard_occupancy_rate": float(occupancy.mean.hazard_occupancy_rate[agent_index]),
            }
        )
    return tuple(rows)


@dataclass(frozen=True)
class FormalEvaluationResult:
    """同一 actor 的 canonical/random-order 配对 stochastic evaluation。"""

    canonical: FixedLambdaEvaluation
    random_order: FixedLambdaEvaluation
    canonical_occupancy: AgentTargetOccupancyAggregate
    random_order_occupancy: AgentTargetOccupancyAggregate
    random_permutations: IntArray
    canonical_payload: dict[str, Any]
    random_order_payload: dict[str, Any]
    comparison_rows: tuple[dict[str, Any], ...]
    agent_target_occupancy_rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        frozen = np.asarray(self.random_permutations, dtype=np.int64).copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "random_permutations", frozen)

    @property
    def random_minus_canonical(self) -> dict[str, float]:
        canonical, random_order = self.comparison_rows
        return {
            key: float(random_order[key]) - float(canonical[key])
            for key in canonical
            if key not in {"target_order_mode"}
        }


def evaluate_formal_actor(
    actor: IndependentActors,
    environment_config: HRMRConfig,
    fixed_lambda: float,
    evaluation_seeds: Sequence[int],
    *,
    evaluation_steps: int = 2_000,
    burn_in_steps: int = 100,
    device: str | torch.device | None = None,
    random_permutations: ArrayLike | None = None,
    actor_target_view: str = "all",
) -> FormalEvaluationResult:
    """对 IndependentActors 执行配对 canonical/random-order stochastic evaluation。"""

    if not isinstance(actor, IndependentActors):
        raise TypeError("formal evaluation requires IndependentActors")
    validate_config(environment_config)
    if environment_config.use_agent_id:
        raise ValueError("formal evaluation forbids Agent ID")
    if environment_config.action_slip_probability != 0.0:
        raise ValueError("formal evaluation requires action_slip_probability=0")
    if actor.num_agents != environment_config.num_agents:
        raise ValueError("actor and environment agent counts do not match")
    if actor_target_view not in {"all", "safe_only"}:
        raise ValueError("actor_target_view must be 'all' or 'safe_only'")
    safe_only_view = actor_target_view == "safe_only"
    if safe_only_view:
        expected_input_dim = SAFE_ACTOR_OBSERVATION_DIM
        actor_target_ids = SAFE_TARGET_IDS
        actor_num_targets = SAFE_ACTOR_NUM_TARGETS
    else:
        expected_layout = observation_layout(
            use_agent_id=False,
            num_agents=environment_config.num_agents,
            num_targets=environment_config.num_targets,
        )
        expected_input_dim = max(feature_slice.stop for feature_slice in expected_layout.values())
        actor_target_ids = tuple(environment_config.target_ids)
        actor_num_targets = environment_config.num_targets
    if actor.input_dim != expected_input_dim:
        raise ValueError("formal actor input dimension does not match environment observation")
    lambda_value = _fixed_lambda(fixed_lambda)
    seeds = _evaluation_seeds(evaluation_seeds)
    permutations = (
        fixed_random_target_permutations(seeds, num_targets=actor_num_targets)
        if random_permutations is None
        else _checked_permutations(
            random_permutations,
            num_seeds=len(seeds),
            num_targets=actor_num_targets,
        )
    )
    parameter_snapshot = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }

    def environment_factory() -> HRMREnvironment:
        return HRMREnvironment(config=environment_config)

    canonical_actor = SafeTargetActorView(actor) if safe_only_view else actor
    canonical = evaluate_fixed_lambda(
        canonical_actor,
        environment_factory,
        lambda_value,
        seeds,
        evaluation_steps=evaluation_steps,
        burn_in_steps=burn_in_steps,
        device=device,
    )
    random_actor = (
        SafeTargetActorView(actor, permutations)
        if safe_only_view
        else FixedPermutationEvaluationActor(
            actor,
            permutations,
            num_targets=environment_config.num_targets,
            use_agent_id=False,
        )
    )
    random_order = evaluate_fixed_lambda(
        random_actor,
        environment_factory,
        lambda_value,
        seeds,
        evaluation_steps=evaluation_steps,
        burn_in_steps=burn_in_steps,
        device=device,
    )
    if not canonical.is_formal_evaluation or not random_order.is_formal_evaluation:
        raise AssertionError("formal evaluation must use stochastic action sampling")
    if any(
        not torch.equal(value.detach().cpu(), parameter_snapshot[name])
        for name, value in actor.state_dict().items()
    ):
        raise AssertionError("formal evaluation modified actor parameters")

    canonical_occupancy = aggregate_evaluation_occupancy(canonical, environment_config)
    random_occupancy = aggregate_evaluation_occupancy(random_order, environment_config)
    target_orders = {
        seed: tuple(actor_target_ids[index] for index in permutation)
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
    comparison_rows = (
        evaluation_comparison_row(
            CANONICAL_TARGET_ORDER_MODE,
            canonical,
            canonical_occupancy,
        ),
        evaluation_comparison_row(
            RANDOM_TARGET_ORDER_MODE,
            random_order,
            random_occupancy,
        ),
    )
    occupancy_rows = agent_target_occupancy_rows(
        CANONICAL_TARGET_ORDER_MODE,
        canonical_occupancy,
    ) + agent_target_occupancy_rows(
        RANDOM_TARGET_ORDER_MODE,
        random_occupancy,
    )
    return FormalEvaluationResult(
        canonical=canonical,
        random_order=random_order,
        canonical_occupancy=canonical_occupancy,
        random_order_occupancy=random_occupancy,
        random_permutations=permutations,
        canonical_payload=canonical_payload,
        random_order_payload=random_payload,
        comparison_rows=comparison_rows,
        agent_target_occupancy_rows=occupancy_rows,
    )


def _load_torch_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - 兼容旧 PyTorch
        payload = torch.load(path, map_location=device)
    return _mapping(payload, name="checkpoint root")


def _resolve_environment_path(
    checkpoint_path: Path,
    checkpoint_config: Mapping[str, Any],
) -> Path:
    experiment = _mapping(checkpoint_config.get("experiment"), name="checkpoint experiment")
    configured_value = experiment.get("environment_config")
    if not isinstance(configured_value, str) or not configured_value:
        raise ValueError("checkpoint environment_config must be a non-empty string")
    configured = Path(configured_value).expanduser()
    candidates = [configured]
    source_value = checkpoint_config.get("source_path")
    if isinstance(source_value, str) and source_value:
        source = Path(source_value).expanduser()
        candidates.extend((source.parent / configured, source.parent.parent / configured))
    candidates.extend(parent / configured for parent in checkpoint_path.parents)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(configured_value)


def _checkpoint_environment_config(
    checkpoint_path: Path,
    checkpoint_config: Mapping[str, Any],
    override: HRMRConfig | str | Path | None,
) -> HRMRConfig:
    if isinstance(override, HRMRConfig):
        environment_config = override
    else:
        environment_path = (
            _resolve_environment_path(checkpoint_path, checkpoint_config)
            if override is None
            else Path(override).expanduser().resolve()
        )
        environment_config = load_config(environment_path)
    environment = _mapping(checkpoint_config.get("environment"), name="checkpoint environment")
    slip = float(
        environment.get("action_slip_probability", environment_config.action_slip_probability)
    )
    return replace(environment_config, action_slip_probability=slip)


@dataclass(frozen=True)
class FormalCheckpointEvaluation:
    """Checkpoint metadata 与配对正式评价结果。"""

    checkpoint_path: Path
    checkpoint_environment_steps: int
    training_seed: int
    fixed_lambda: float
    result: FormalEvaluationResult

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return self.result.canonical_payload

    @property
    def random_order_payload(self) -> dict[str, Any]:
        return self.result.random_order_payload

    @property
    def comparison_rows(self) -> tuple[dict[str, Any], ...]:
        return self.result.comparison_rows


def evaluate_formal_checkpoint(
    checkpoint: str | Path,
    *,
    environment_config: HRMRConfig | str | Path | None = None,
    evaluation_seeds: Sequence[int] | None = None,
    evaluation_steps: int | None = None,
    burn_in_steps: int | None = None,
    device: str | torch.device | None = "cpu",
    random_permutations: ArrayLike | None = None,
) -> FormalCheckpointEvaluation:
    """只恢复 actor，并评价任意非负 fixed-lambda formal IndependentActors checkpoint。"""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    resolved_device = resolve_device(device)
    raw = _load_torch_checkpoint(checkpoint_path, resolved_device)
    if int(raw.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    checkpoint_config = _mapping(raw.get("config"), name="checkpoint config")
    experiment = _mapping(checkpoint_config.get("experiment"), name="checkpoint experiment")
    if experiment.get("independent_actors") is not True:
        raise ValueError("formal checkpoint must contain independent actors")
    if experiment.get("randomize_actor_target_order") is not True:
        raise ValueError("formal checkpoint must use random target-order training")
    training = _mapping(checkpoint_config.get("training"), name="checkpoint training")
    if training.get("total_environment_steps") != 500_000:
        raise ValueError("formal checkpoint must configure 500000 environment steps")
    if training.get("n_step") not in (8, 16):
        raise ValueError("formal checkpoint must use n_step=8 or n_step=16")
    if (
        float(training.get("entropy_coefficient_start", -1.0)) != 0.02
        or float(training.get("entropy_coefficient_end", -1.0)) != 0.02
    ):
        raise ValueError("formal checkpoint must use fixed entropy coefficient 0.02")
    if training.get("early_stopping") is not False:
        raise ValueError("formal checkpoint must disable early stopping")
    activation = str(experiment.get("activation", "tanh")).lower()
    seed_base = _nonnegative_integer(
        experiment.get("independent_actor_init_seed_base", 100),
        name="independent_actor_init_seed_base",
    )
    extra = _mapping(raw.get("extra"), name="checkpoint extra")
    training_seed = _nonnegative_integer(extra.get("training_seed"), name="training_seed")
    loaded_environment = _checkpoint_environment_config(
        checkpoint_path,
        checkpoint_config,
        environment_config,
    )
    if loaded_environment.use_agent_id:
        raise ValueError("formal checkpoint environment must not use Agent ID")
    actor_seeds = tuple(
        seed_base + training_seed * loaded_environment.num_agents + agent_index
        for agent_index in range(loaded_environment.num_agents)
    )
    actor_target_view = str(experiment.get("actor_target_view", "all")).lower()
    if actor_target_view not in {"all", "safe_only"}:
        raise ValueError("checkpoint actor_target_view must be 'all' or 'safe_only'")
    actor_input_dim = 48 if actor_target_view == "all" else SAFE_ACTOR_OBSERVATION_DIM
    actor = IndependentActors(
        num_agents=loaded_environment.num_agents,
        input_dim=actor_input_dim,
        hidden_dims=(128, 128),
        num_actions=5,
        activation=activation,
        initialization_seeds=actor_seeds,
    ).to(resolved_device)
    models = _mapping(raw.get("models"), name="checkpoint models")
    actor_state = _mapping(models.get("actor"), name="checkpoint actor state")
    actor.load_state_dict(actor_state, strict=True)
    modes = raw.get("module_training_modes")
    if isinstance(modes, Mapping):
        actor.train(bool(modes.get("actor", True)))

    evaluation = _mapping(checkpoint_config.get("evaluation"), name="checkpoint evaluation")
    seeds = (
        _evaluation_seeds(evaluation_seeds)
        if evaluation_seeds is not None
        else _evaluation_seeds(evaluation.get("seeds", ()))
    )
    steps = (
        _positive_integer(evaluation_steps, name="evaluation_steps")
        if evaluation_steps is not None
        else _positive_integer(evaluation.get("evaluation_steps"), name="evaluation_steps")
    )
    burn_in = (
        _nonnegative_integer(burn_in_steps, name="burn_in_steps")
        if burn_in_steps is not None
        else _nonnegative_integer(evaluation.get("burn_in_steps"), name="burn_in_steps")
    )
    lambda_value = _fixed_lambda(raw.get("fixed_lambda"))
    result = evaluate_formal_actor(
        actor,
        loaded_environment,
        lambda_value,
        seeds,
        evaluation_steps=steps,
        burn_in_steps=burn_in,
        device=resolved_device,
        random_permutations=random_permutations,
        actor_target_view=actor_target_view,
    )
    return FormalCheckpointEvaluation(
        checkpoint_path=checkpoint_path,
        checkpoint_environment_steps=_nonnegative_integer(
            raw.get("environment_steps"),
            name="checkpoint environment_steps",
        ),
        training_seed=training_seed,
        fixed_lambda=lambda_value,
        result=result,
    )


__all__ = [
    "CANONICAL_TARGET_ORDER_MODE",
    "RANDOM_EVALUATION_PERMUTATION_STREAM",
    "RANDOM_TARGET_ORDER_MODE",
    "FixedPermutationEvaluationActor",
    "FormalCheckpointEvaluation",
    "FormalEvaluationResult",
    "agent_target_occupancy_payload",
    "agent_target_occupancy_rows",
    "aggregate_evaluation_occupancy",
    "evaluate_formal_actor",
    "evaluate_formal_checkpoint",
    "evaluation_comparison_row",
    "fixed_random_target_permutations",
    "formal_evaluation_payload",
]
