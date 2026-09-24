"""Independent fixed-λ policy 的无更新 stochastic evaluation。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Integral, Real
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrmr.constants import Action
from hrmr.metrics import RolloutSummary, summarize_rollout
from hrmr.rl.analytic_modes import instantaneous_optimal_mode_mask

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_POLICY_RNG_STREAM = 0x48524D52
OperatingModeMaskFunction = Callable[[ArrayLike, Sequence[str], float], ArrayLike]


class _BatchedEvaluationUnavailable(RuntimeError):
    """Factory 无法提供独立环境时退回通用串行实现。"""


class ActorProtocol(Protocol):
    """Evaluator 所需的最小 actor 接口。"""

    training: bool

    def eval(self) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...

    def parameters(self) -> Any: ...

    def sample(
        self,
        observations: Any,
        deterministic: bool = False,
    ) -> tuple[Any, Any, Any]: ...


@dataclass(frozen=True)
class EvaluationTrajectory:
    """单一 evaluation seed 的 whole/steady 原始指标。"""

    seed: int
    fixed_lambda: float
    deterministic: bool
    visualization_only: bool
    burn_in_steps: int
    evaluation_steps: int
    whole_summary: RolloutSummary
    steady_summary: RolloutSummary
    whole_mean_num_distinct_targets: float
    steady_mean_num_distinct_targets: float
    whole_all_hazardous_mode_rate: float
    steady_all_hazardous_mode_rate: float
    whole_operating_mode_rate: float
    steady_operating_mode_rate: float
    whole_mean_hazardous_targets_covered: float
    steady_mean_hazardous_targets_covered: float
    whole_mean_safe_targets_covered: float
    steady_mean_safe_targets_covered: float
    whole_stay_action_rate: float
    steady_stay_action_rate: float
    whole_actor_entropy: float
    steady_actor_entropy: float
    reward_history: FloatArray
    action_history: IntArray
    entropy_history: FloatArray
    position_history: IntArray
    state_history: FloatArray

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                copied = value.copy()
                copied.setflags(write=False)
                object.__setattr__(self, item.name, copied)

    @property
    def whole_reward(self) -> float:
        return self.whole_summary.mean_reward

    @property
    def whole_cost(self) -> float:
        return self.whole_summary.mean_global_cost

    @property
    def steady_reward(self) -> float:
        return self.steady_summary.mean_reward

    @property
    def steady_cost(self) -> float:
        return self.steady_summary.mean_global_cost

    @property
    def whole_scalarized_objective(self) -> float:
        return self.whole_reward - self.fixed_lambda * self.whole_cost

    @property
    def steady_scalarized_objective(self) -> float:
        return self.steady_reward - self.fixed_lambda * self.steady_cost

    @property
    def steady_target_occupancy(self) -> FloatArray:
        return self.steady_summary.mean_target_occupancy


@dataclass(frozen=True)
class EvaluationAggregate:
    """对共同 evaluation seed set 的等权均值与 seed-level 样本标准差。"""

    num_trajectories: int
    whole_mean_reward: float
    whole_std_reward: float
    whole_mean_cost: float
    whole_std_cost: float
    steady_mean_reward: float
    steady_std_reward: float
    steady_mean_cost: float
    steady_std_cost: float
    whole_mean_scalarized_objective: float
    steady_mean_scalarized_objective: float
    whole_mean_num_distinct_targets: float
    whole_std_num_distinct_targets: float
    steady_mean_num_distinct_targets: float
    steady_std_num_distinct_targets: float
    whole_mean_all_hazardous_mode_rate: float
    whole_std_all_hazardous_mode_rate: float
    steady_mean_all_hazardous_mode_rate: float
    steady_std_all_hazardous_mode_rate: float
    whole_mean_operating_mode_rate: float
    whole_std_operating_mode_rate: float
    steady_mean_operating_mode_rate: float
    steady_std_operating_mode_rate: float
    whole_mean_hazardous_targets_covered: float
    steady_mean_hazardous_targets_covered: float
    whole_mean_safe_targets_covered: float
    steady_mean_safe_targets_covered: float
    whole_mean_stay_action_rate: float
    whole_std_stay_action_rate: float
    steady_mean_stay_action_rate: float
    steady_std_stay_action_rate: float
    whole_mean_actor_entropy: float
    whole_std_actor_entropy: float
    steady_mean_actor_entropy: float
    steady_std_actor_entropy: float
    whole_mean_target_occupancy: FloatArray
    steady_mean_target_occupancy: FloatArray
    steady_mean_local_costs: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "whole_mean_target_occupancy",
            "steady_mean_target_occupancy",
            "steady_mean_local_costs",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FixedLambdaEvaluation:
    """一个 fixed-λ actor 在共同 seeds 上的正式或可视化评价。"""

    fixed_lambda: float
    target_ids: tuple[str, ...]
    evaluation_seeds: tuple[int, ...]
    burn_in_steps: int
    evaluation_steps: int
    deterministic: bool
    visualization_only: bool
    trajectories: tuple[EvaluationTrajectory, ...]
    aggregate: EvaluationAggregate

    @property
    def sampling_mode(self) -> str:
        return "deterministic_visualization" if self.deterministic else "stochastic"

    @property
    def is_formal_evaluation(self) -> bool:
        return not self.deterministic and not self.visualization_only


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - train extra 提供 torch
        raise RuntimeError(
            "policy evaluation requires PyTorch; install the project's train extra"
        ) from exc
    return torch


def _finite_lambda(value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("fixed_lambda must be a real number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError("fixed_lambda must be finite and non-negative")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _evaluation_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError("evaluation_seeds must not be empty")
    if any(
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0
        for seed in seeds
    ):
        raise ValueError("evaluation seeds must be non-negative integers")
    normalized = tuple(int(seed) for seed in seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("evaluation_seeds must not contain duplicates")
    return normalized


def _actor_device(actor: ActorProtocol, requested_device: Any | None) -> Any:
    torch = _require_torch()
    if requested_device is not None:
        return torch.device(requested_device)
    try:
        return next(iter(actor.parameters())).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _environment(factory_or_env: Callable[[], Any] | Any) -> Any:
    return factory_or_env() if callable(factory_or_env) else factory_or_env


def _position_array(env: Any, info: Mapping[str, Any]) -> IntArray:
    if hasattr(env, "positions"):
        positions = np.asarray(env.positions)
    elif "robot_positions" in info:
        positions = np.asarray(info["robot_positions"])
    elif "agent_positions" in info:
        positions = np.asarray(info["agent_positions"])
    else:
        raise KeyError("evaluation environment exposes no robot positions")
    if positions.ndim != 2 or positions.shape[1:] != (2,):
        raise ValueError("robot positions must have shape (num_agents, 2)")
    return positions.astype(np.int64, copy=True)


def _target_occupancy(info: Mapping[str, Any], num_targets: int) -> FloatArray:
    for key in ("target_occupancy", "target_coverage"):
        if key in info:
            occupancy = np.asarray(info[key], dtype=np.float64)
            if occupancy.shape != (num_targets,):
                raise ValueError(f"{key} must have shape ({num_targets},)")
            return occupancy.copy()
    raise KeyError("evaluation info contains no target occupancy")


@contextmanager
def _fork_evaluation_rng(torch: Any, device: Any, seed: int) -> Iterator[None]:
    """隔离 evaluator 使用的 Torch RNG，包括 ``fork_rng`` 未覆盖的 MPS。"""

    mps_state = None
    set_mps_rng_state = None
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        get_mps_rng_state = getattr(mps, "get_rng_state", None)
        candidate_setter = getattr(mps, "set_rng_state", None)
        if callable(get_mps_rng_state) and callable(candidate_setter):
            mps_state = get_mps_rng_state()
            set_mps_rng_state = candidate_setter

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        if set_mps_rng_state is not None:
            set_mps_rng_state(mps_state)


def _sample_actions(
    actor: ActorProtocol,
    observations: Any,
    *,
    deterministic: bool,
    device: Any,
) -> tuple[IntArray, FloatArray]:
    torch = _require_torch()
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    sample = actor.sample(tensor, deterministic=deterministic)
    if not isinstance(sample, tuple) or len(sample) != 3:
        raise TypeError("actor.sample() must return (actions, log_probs, entropy)")
    actions, _, entropy = sample
    if not hasattr(actions, "detach") or not hasattr(entropy, "detach"):
        raise TypeError("actor.sample() actions and entropy must be torch tensors")
    action_array = actions.detach().to("cpu").numpy()
    if action_array.ndim != 1:
        raise ValueError("single-environment actor actions must have shape (num_agents,)")
    if action_array.dtype.kind not in {"i", "u"}:
        raise TypeError("actor.sample() must return integer actions")
    entropy_array = np.asarray(entropy.detach().to("cpu").numpy(), dtype=np.float64)
    if entropy_array.shape != action_array.shape:
        raise ValueError("single-environment actor entropy must have shape (num_agents,)")
    if not np.all(np.isfinite(entropy_array)):
        raise ValueError("actor entropy must be finite")
    return action_array.astype(np.int64, copy=True), entropy_array.copy()


def _supports_batched_evaluation(actor: ActorProtocol, env_factory: Any) -> bool:
    """支持 joint-batch 的 policy 才启用 fast path，其他协议保持串行。"""

    return (
        callable(env_factory)
        and callable(getattr(actor, "distribution", None))
        and getattr(actor, "num_actions", None) == 5
    )


def _batched_action_probabilities(
    actor: ActorProtocol,
    observations: Sequence[Any],
    *,
    device: Any,
) -> FloatArray:
    """单次 actor forward 返回 ``(num_seeds, num_agents, 5)`` 概率。"""

    torch = _require_torch()
    try:
        observation_array = np.stack(observations)
    except ValueError as exc:
        raise ValueError("all evaluation observations must have identical shapes") from exc
    if observation_array.ndim != 3:
        raise ValueError(
            "batched evaluation observations must have shape "
            "(num_seeds, num_agents, observation_dim)"
        )
    tensor = torch.as_tensor(observation_array, dtype=torch.float32, device=device)
    distribution = actor.distribution(tensor)
    probabilities = getattr(distribution, "probs", None)
    if probabilities is None or not hasattr(probabilities, "detach"):
        raise TypeError("batched actor.distribution() must expose tensor probabilities")
    probability_array = np.asarray(
        probabilities.detach().to("cpu").numpy(),
        dtype=np.float64,
    )
    expected_shape = (*observation_array.shape[:2], 5)
    if probability_array.shape != expected_shape:
        raise ValueError(
            "batched actor probabilities must have shape "
            f"{expected_shape}; got {probability_array.shape}"
        )
    if not np.all(np.isfinite(probability_array)) or np.any(probability_array < 0.0):
        raise ValueError("batched actor probabilities must be finite and non-negative")
    totals = np.sum(probability_array, axis=-1, keepdims=True, dtype=np.float64)
    if np.any(totals <= 0.0):
        raise ValueError("batched actor probabilities must have positive row sums")
    return probability_array / totals


def _policy_generators(seeds: Sequence[int]) -> tuple[np.random.Generator, ...]:
    """用独立域避免 policy sampling 与同 seed 的环境 RNG 形成相关流。"""

    return tuple(
        np.random.default_rng(np.random.SeedSequence((seed, _POLICY_RNG_STREAM))) for seed in seeds
    )


def _sample_batched_actions(
    probabilities: FloatArray,
    generators: Sequence[np.random.Generator],
    *,
    deterministic: bool,
) -> IntArray:
    """按 seed 独立采样，seed 列表的排列不会改变任一 seed 的动作流。"""

    if deterministic:
        return np.argmax(probabilities, axis=-1).astype(np.int64, copy=False)
    if len(generators) != len(probabilities):
        raise ValueError("one policy RNG is required per evaluation seed")
    sampled = np.empty(probabilities.shape[:2], dtype=np.int64)
    for seed_index, generator in enumerate(generators):
        uniforms = generator.random((probabilities.shape[1], 1))
        cumulative = np.cumsum(probabilities[seed_index], axis=-1, dtype=np.float64)
        actions = np.sum(uniforms >= cumulative, axis=-1, dtype=np.int64)
        sampled[seed_index] = np.minimum(actions, probabilities.shape[-1] - 1)
    return sampled


def _build_trajectory(
    *,
    seed: int,
    fixed_lambda: float,
    deterministic: bool,
    visualization_only: bool,
    burn_in_steps: int,
    evaluation_steps: int,
    target_ids: Sequence[str],
    rewards: Sequence[float],
    global_costs: Sequence[float],
    local_costs: Sequence[Any],
    occupancies: Sequence[Any],
    actions: Sequence[Any],
    entropies: Sequence[Any],
    positions: Sequence[Any],
    states: Sequence[Any],
    operating_mode_mask_fn: OperatingModeMaskFunction | None = None,
) -> EvaluationTrajectory:
    reward_array = np.asarray(rewards, dtype=np.float64)
    global_cost_array = np.asarray(global_costs, dtype=np.float64)
    local_cost_array = np.asarray(local_costs, dtype=np.float64)
    occupancy_array = np.asarray(occupancies, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.int64)
    entropy_array = np.asarray(entropies, dtype=np.float64)
    state_array = np.asarray(states, dtype=np.float64)
    num_transitions = burn_in_steps + evaluation_steps
    if reward_array.shape != (num_transitions,):
        raise ValueError("evaluation reward history has an unexpected length")
    if occupancy_array.shape != (num_transitions, len(target_ids)):
        raise ValueError("evaluation occupancy history has an unexpected shape")
    if not np.all((occupancy_array == 0.0) | (occupancy_array == 1.0)):
        raise ValueError("evaluation occupancy history must be binary")
    if action_array.ndim != 2 or action_array.shape[0] != num_transitions:
        raise ValueError("evaluation action history has an unexpected shape")
    if entropy_array.shape != action_array.shape:
        raise ValueError("evaluation entropy history must match action history")
    if state_array.ndim != 2 or state_array.shape[0] != num_transitions + 1:
        raise ValueError("evaluation state history must include reset plus every next state")
    try:
        hazardous_indices = tuple(
            target_ids.index(target_id) for target_id in ("H1", "H2", "H3", "H4")
        )
        safe_indices = tuple(target_ids.index(target_id) for target_id in ("S1", "S2", "S3", "S4"))
    except ValueError as exc:
        raise ValueError("evaluation target order must include S1-S4 and H1-H4") from exc
    mode_mask_function = (
        instantaneous_optimal_mode_mask
        if operating_mode_mask_fn is None
        else operating_mode_mask_fn
    )
    if not callable(mode_mask_function):
        raise TypeError("operating_mode_mask_fn must be callable or None")
    raw_operating_mode_mask = np.asarray(
        mode_mask_function(
            occupancy_array,
            target_ids,
            fixed_lambda=fixed_lambda,
        )
    )
    if raw_operating_mode_mask.shape != (num_transitions,):
        raise ValueError(
            "operating_mode_mask_fn must return one mask value per evaluation transition"
        )
    if raw_operating_mode_mask.dtype.kind != "b":
        raise TypeError("operating_mode_mask_fn must return a boolean mask")
    operating_mode_mask = raw_operating_mode_mask.astype(np.bool_, copy=False)

    def behavior_metrics(
        start_step: int,
    ) -> tuple[float, float, float, float, float, float, float]:
        selected_occupancies = occupancy_array[start_step:]
        selected_actions = action_array[start_step:]
        selected_entropies = entropy_array[start_step:]
        return (
            float(np.mean(np.sum(selected_occupancies, axis=-1))),
            float(np.mean(np.all(selected_occupancies[:, hazardous_indices] == 1.0, axis=-1))),
            float(np.mean(operating_mode_mask[start_step:])),
            float(np.mean(np.sum(selected_occupancies[:, hazardous_indices], axis=-1))),
            float(np.mean(np.sum(selected_occupancies[:, safe_indices], axis=-1))),
            float(np.mean(selected_actions == int(Action.STAY))),
            float(np.mean(selected_entropies)),
        )

    (
        whole_distinct,
        whole_all_hazardous,
        whole_operating_mode,
        whole_hazardous_count,
        whole_safe_count,
        whole_stay,
        whole_entropy,
    ) = behavior_metrics(0)
    (
        steady_distinct,
        steady_all_hazardous,
        steady_operating_mode,
        steady_hazardous_count,
        steady_safe_count,
        steady_stay,
        steady_entropy,
    ) = behavior_metrics(burn_in_steps)
    whole = summarize_rollout(
        reward_array,
        global_cost_array,
        occupancy_array,
        local_cost_array,
    )
    steady = summarize_rollout(
        reward_array,
        global_cost_array,
        occupancy_array,
        local_cost_array,
        start_step=burn_in_steps,
    )
    return EvaluationTrajectory(
        seed=seed,
        fixed_lambda=fixed_lambda,
        deterministic=deterministic,
        visualization_only=visualization_only,
        burn_in_steps=burn_in_steps,
        evaluation_steps=evaluation_steps,
        whole_summary=whole,
        steady_summary=steady,
        whole_mean_num_distinct_targets=whole_distinct,
        steady_mean_num_distinct_targets=steady_distinct,
        whole_all_hazardous_mode_rate=whole_all_hazardous,
        steady_all_hazardous_mode_rate=steady_all_hazardous,
        whole_operating_mode_rate=whole_operating_mode,
        steady_operating_mode_rate=steady_operating_mode,
        whole_mean_hazardous_targets_covered=whole_hazardous_count,
        steady_mean_hazardous_targets_covered=steady_hazardous_count,
        whole_mean_safe_targets_covered=whole_safe_count,
        steady_mean_safe_targets_covered=steady_safe_count,
        whole_stay_action_rate=whole_stay,
        steady_stay_action_rate=steady_stay,
        whole_actor_entropy=whole_entropy,
        steady_actor_entropy=steady_entropy,
        reward_history=reward_array,
        action_history=action_array,
        entropy_history=entropy_array,
        position_history=np.asarray(positions, dtype=np.int64),
        state_history=state_array,
    )


def _rollout_one_seed(
    actor: ActorProtocol,
    env_factory: Callable[[], Any] | Any,
    *,
    fixed_lambda: float,
    seed: int,
    evaluation_steps: int,
    burn_in_steps: int,
    deterministic: bool,
    visualization_only: bool,
    device: Any,
    operating_mode_mask_fn: OperatingModeMaskFunction | None = None,
) -> tuple[EvaluationTrajectory, tuple[str, ...]]:
    torch = _require_torch()
    env = _environment(env_factory)
    reset_result = env.reset(seed=seed)
    if not isinstance(reset_result, tuple) or len(reset_result) != 3:
        raise TypeError("evaluation env.reset() must return three values")
    observations, state, reset_info = reset_result
    if not isinstance(reset_info, Mapping):
        raise TypeError("evaluation reset info must be a mapping")
    if not hasattr(env, "config") or not hasattr(env.config, "target_ids"):
        raise TypeError("evaluation environment must expose config.target_ids")
    target_ids = tuple(env.config.target_ids)

    rewards = []
    global_costs = []
    local_costs = []
    occupancies = []
    actions = []
    entropies = []
    positions = [_position_array(env, reset_info)]
    states = [np.asarray(state, dtype=np.float64).copy()]

    with _fork_evaluation_rng(torch, device, seed), torch.inference_mode():
        for _ in range(burn_in_steps + evaluation_steps):
            joint_actions, actor_entropy = _sample_actions(
                actor,
                observations,
                deterministic=deterministic,
                device=device,
            )
            step_result = env.step(joint_actions)
            if not isinstance(step_result, tuple) or len(step_result) != 7:
                raise TypeError("single evaluation env.step() must return seven values")
            (
                observations,
                state,
                team_reward,
                step_local_costs,
                terminated,
                truncated,
                info,
            ) = step_result
            if bool(terminated) or bool(truncated):
                raise AssertionError(
                    "fixed-lambda evaluation must not terminate or truncate the continuing task"
                )
            if not isinstance(info, Mapping):
                raise TypeError("evaluation step info must be a mapping")
            local_cost_array = np.asarray(step_local_costs, dtype=np.float64)
            if local_cost_array.ndim != 1:
                raise ValueError("local costs must have shape (num_agents,)")
            rewards.append(float(team_reward))
            local_costs.append(local_cost_array.copy())
            global_costs.append(
                float(info.get("global_cost", np.sum(local_cost_array, dtype=np.float64)))
            )
            occupancies.append(_target_occupancy(info, len(target_ids)))
            actions.append(joint_actions)
            entropies.append(actor_entropy)
            positions.append(_position_array(env, info))
            states.append(np.asarray(state, dtype=np.float64).copy())

    trajectory = _build_trajectory(
        seed=seed,
        fixed_lambda=fixed_lambda,
        deterministic=deterministic,
        visualization_only=visualization_only,
        burn_in_steps=burn_in_steps,
        evaluation_steps=evaluation_steps,
        target_ids=target_ids,
        rewards=rewards,
        global_costs=global_costs,
        local_costs=local_costs,
        occupancies=occupancies,
        actions=actions,
        entropies=entropies,
        positions=positions,
        states=states,
        operating_mode_mask_fn=operating_mode_mask_fn,
    )
    return trajectory, target_ids


def _rollout_seeds_batched(
    actor: ActorProtocol,
    env_factory: Callable[[], Any],
    *,
    fixed_lambda: float,
    seeds: Sequence[int],
    evaluation_steps: int,
    burn_in_steps: int,
    deterministic: bool,
    visualization_only: bool,
    device: Any,
    operating_mode_mask_fn: OperatingModeMaskFunction | None = None,
) -> tuple[tuple[EvaluationTrajectory, ...], tuple[str, ...]]:
    """同步推进 seed 环境，actor 每个时刻仅执行一次批量 forward。"""

    torch = _require_torch()
    environments = tuple(_environment(env_factory) for _ in seeds)
    if len({id(environment) for environment in environments}) != len(environments):
        raise _BatchedEvaluationUnavailable("environment factory returned shared instances")

    observations: list[Any] = []
    state_histories: list[list[FloatArray]] = []
    canonical_target_ids: tuple[str, ...] | None = None
    position_histories: list[list[IntArray]] = []
    for environment, seed in zip(environments, seeds, strict=True):
        reset_result = environment.reset(seed=seed)
        if not isinstance(reset_result, tuple) or len(reset_result) != 3:
            raise TypeError("evaluation env.reset() must return three values")
        seed_observations, seed_state, reset_info = reset_result
        if not isinstance(reset_info, Mapping):
            raise TypeError("evaluation reset info must be a mapping")
        if not hasattr(environment, "config") or not hasattr(environment.config, "target_ids"):
            raise TypeError("evaluation environment must expose config.target_ids")
        target_ids = tuple(environment.config.target_ids)
        if canonical_target_ids is None:
            canonical_target_ids = target_ids
        elif target_ids != canonical_target_ids:
            raise ValueError("all evaluation environments must use the same target order")
        observations.append(seed_observations)
        state_histories.append([np.asarray(seed_state, dtype=np.float64).copy()])
        position_histories.append([_position_array(environment, reset_info)])

    num_seeds = len(seeds)
    reward_histories: list[list[float]] = [[] for _ in range(num_seeds)]
    global_cost_histories: list[list[float]] = [[] for _ in range(num_seeds)]
    local_cost_histories: list[list[FloatArray]] = [[] for _ in range(num_seeds)]
    occupancy_histories: list[list[FloatArray]] = [[] for _ in range(num_seeds)]
    action_histories: list[list[IntArray]] = [[] for _ in range(num_seeds)]
    entropy_histories: list[list[FloatArray]] = [[] for _ in range(num_seeds)]
    policy_generators = _policy_generators(seeds)

    with _fork_evaluation_rng(torch, device, seeds[0]), torch.inference_mode():
        for _ in range(burn_in_steps + evaluation_steps):
            probabilities = _batched_action_probabilities(
                actor,
                observations,
                device=device,
            )
            batched_actions = _sample_batched_actions(
                probabilities,
                policy_generators,
                deterministic=deterministic,
            )
            log_probabilities = np.zeros_like(probabilities)
            np.log(
                probabilities,
                out=log_probabilities,
                where=probabilities > 0.0,
            )
            batched_entropies = -np.sum(
                probabilities * log_probabilities,
                axis=-1,
            )
            for seed_index, (environment, joint_actions) in enumerate(
                zip(environments, batched_actions, strict=True)
            ):
                step_result = environment.step(joint_actions)
                if not isinstance(step_result, tuple) or len(step_result) != 7:
                    raise TypeError("single evaluation env.step() must return seven values")
                (
                    next_observations,
                    next_state,
                    team_reward,
                    step_local_costs,
                    terminated,
                    truncated,
                    info,
                ) = step_result
                if bool(terminated) or bool(truncated):
                    raise AssertionError(
                        "fixed-lambda evaluation must not terminate or truncate the continuing task"
                    )
                if not isinstance(info, Mapping):
                    raise TypeError("evaluation step info must be a mapping")
                local_cost_array = np.asarray(step_local_costs, dtype=np.float64)
                if local_cost_array.ndim != 1:
                    raise ValueError("local costs must have shape (num_agents,)")
                observations[seed_index] = next_observations
                reward_histories[seed_index].append(float(team_reward))
                local_cost_histories[seed_index].append(local_cost_array.copy())
                global_cost_histories[seed_index].append(
                    float(info.get("global_cost", np.sum(local_cost_array, dtype=np.float64)))
                )
                assert canonical_target_ids is not None
                occupancy_histories[seed_index].append(
                    _target_occupancy(info, len(canonical_target_ids))
                )
                action_histories[seed_index].append(np.asarray(joint_actions, dtype=np.int64))
                entropy_histories[seed_index].append(batched_entropies[seed_index].copy())
                position_histories[seed_index].append(_position_array(environment, info))
                state_histories[seed_index].append(np.asarray(next_state, dtype=np.float64).copy())

    assert canonical_target_ids is not None
    trajectories = tuple(
        _build_trajectory(
            seed=seed,
            fixed_lambda=fixed_lambda,
            deterministic=deterministic,
            visualization_only=visualization_only,
            burn_in_steps=burn_in_steps,
            evaluation_steps=evaluation_steps,
            target_ids=canonical_target_ids,
            rewards=reward_histories[index],
            global_costs=global_cost_histories[index],
            local_costs=local_cost_histories[index],
            occupancies=occupancy_histories[index],
            actions=action_histories[index],
            entropies=entropy_histories[index],
            positions=position_histories[index],
            states=state_histories[index],
            operating_mode_mask_fn=operating_mode_mask_fn,
        )
        for index, seed in enumerate(seeds)
    )
    return trajectories, canonical_target_ids


def _aggregate_trajectories(
    trajectories: Sequence[EvaluationTrajectory],
) -> EvaluationAggregate:
    whole_rewards = np.asarray([item.whole_reward for item in trajectories])
    whole_costs = np.asarray([item.whole_cost for item in trajectories])
    steady_rewards = np.asarray([item.steady_reward for item in trajectories])
    steady_costs = np.asarray([item.steady_cost for item in trajectories])
    whole_objectives = np.asarray([item.whole_scalarized_objective for item in trajectories])
    steady_objectives = np.asarray([item.steady_scalarized_objective for item in trajectories])
    metric_arrays = {
        name: np.asarray([getattr(item, name) for item in trajectories], dtype=np.float64)
        for name in (
            "whole_mean_num_distinct_targets",
            "steady_mean_num_distinct_targets",
            "whole_all_hazardous_mode_rate",
            "steady_all_hazardous_mode_rate",
            "whole_operating_mode_rate",
            "steady_operating_mode_rate",
            "whole_mean_hazardous_targets_covered",
            "steady_mean_hazardous_targets_covered",
            "whole_mean_safe_targets_covered",
            "steady_mean_safe_targets_covered",
            "whole_stay_action_rate",
            "steady_stay_action_rate",
            "whole_actor_entropy",
            "steady_actor_entropy",
        )
    }
    whole_occupancies = np.stack(
        [item.whole_summary.mean_target_occupancy for item in trajectories]
    )
    steady_occupancies = np.stack(
        [item.steady_summary.mean_target_occupancy for item in trajectories]
    )
    steady_local_costs = np.stack([item.steady_summary.mean_local_costs for item in trajectories])
    return EvaluationAggregate(
        num_trajectories=len(trajectories),
        whole_mean_reward=float(np.mean(whole_rewards)),
        whole_std_reward=_sample_standard_deviation(whole_rewards),
        whole_mean_cost=float(np.mean(whole_costs)),
        whole_std_cost=_sample_standard_deviation(whole_costs),
        steady_mean_reward=float(np.mean(steady_rewards)),
        steady_std_reward=_sample_standard_deviation(steady_rewards),
        steady_mean_cost=float(np.mean(steady_costs)),
        steady_std_cost=_sample_standard_deviation(steady_costs),
        whole_mean_scalarized_objective=float(np.mean(whole_objectives)),
        steady_mean_scalarized_objective=float(np.mean(steady_objectives)),
        whole_mean_num_distinct_targets=float(
            np.mean(metric_arrays["whole_mean_num_distinct_targets"])
        ),
        whole_std_num_distinct_targets=_sample_standard_deviation(
            metric_arrays["whole_mean_num_distinct_targets"]
        ),
        steady_mean_num_distinct_targets=float(
            np.mean(metric_arrays["steady_mean_num_distinct_targets"])
        ),
        steady_std_num_distinct_targets=_sample_standard_deviation(
            metric_arrays["steady_mean_num_distinct_targets"]
        ),
        whole_mean_all_hazardous_mode_rate=float(
            np.mean(metric_arrays["whole_all_hazardous_mode_rate"])
        ),
        whole_std_all_hazardous_mode_rate=_sample_standard_deviation(
            metric_arrays["whole_all_hazardous_mode_rate"]
        ),
        steady_mean_all_hazardous_mode_rate=float(
            np.mean(metric_arrays["steady_all_hazardous_mode_rate"])
        ),
        steady_std_all_hazardous_mode_rate=_sample_standard_deviation(
            metric_arrays["steady_all_hazardous_mode_rate"]
        ),
        whole_mean_operating_mode_rate=float(np.mean(metric_arrays["whole_operating_mode_rate"])),
        whole_std_operating_mode_rate=_sample_standard_deviation(
            metric_arrays["whole_operating_mode_rate"]
        ),
        steady_mean_operating_mode_rate=float(np.mean(metric_arrays["steady_operating_mode_rate"])),
        steady_std_operating_mode_rate=_sample_standard_deviation(
            metric_arrays["steady_operating_mode_rate"]
        ),
        whole_mean_hazardous_targets_covered=float(
            np.mean(metric_arrays["whole_mean_hazardous_targets_covered"])
        ),
        steady_mean_hazardous_targets_covered=float(
            np.mean(metric_arrays["steady_mean_hazardous_targets_covered"])
        ),
        whole_mean_safe_targets_covered=float(
            np.mean(metric_arrays["whole_mean_safe_targets_covered"])
        ),
        steady_mean_safe_targets_covered=float(
            np.mean(metric_arrays["steady_mean_safe_targets_covered"])
        ),
        whole_mean_stay_action_rate=float(np.mean(metric_arrays["whole_stay_action_rate"])),
        whole_std_stay_action_rate=_sample_standard_deviation(
            metric_arrays["whole_stay_action_rate"]
        ),
        steady_mean_stay_action_rate=float(np.mean(metric_arrays["steady_stay_action_rate"])),
        steady_std_stay_action_rate=_sample_standard_deviation(
            metric_arrays["steady_stay_action_rate"]
        ),
        whole_mean_actor_entropy=float(np.mean(metric_arrays["whole_actor_entropy"])),
        whole_std_actor_entropy=_sample_standard_deviation(metric_arrays["whole_actor_entropy"]),
        steady_mean_actor_entropy=float(np.mean(metric_arrays["steady_actor_entropy"])),
        steady_std_actor_entropy=_sample_standard_deviation(metric_arrays["steady_actor_entropy"]),
        whole_mean_target_occupancy=np.mean(whole_occupancies, axis=0),
        steady_mean_target_occupancy=np.mean(steady_occupancies, axis=0),
        steady_mean_local_costs=np.mean(steady_local_costs, axis=0),
    )


def _sample_standard_deviation(values: FloatArray) -> float:
    """返回 evaluation-seed 样本标准差；单一样本时按不可估约定记为 0。"""

    return 0.0 if len(values) <= 1 else float(np.std(values, ddof=1))


def evaluate_fixed_lambda(
    actor: ActorProtocol,
    env_factory: Callable[[], Any] | Any,
    fixed_lambda: float,
    evaluation_seeds: Sequence[int],
    *,
    evaluation_steps: int = 2_000,
    burn_in_steps: int = 100,
    device: Any | None = None,
    deterministic: bool = False,
    visualization_only: bool = False,
    operating_mode_mask_fn: OperatingModeMaskFunction | None = None,
) -> FixedLambdaEvaluation:
    """冻结 actor 并评价原始 reward/cost；正式调用必须 stochastic。

    ``deterministic=True`` 只有同时显式设置 ``visualization_only=True`` 才被
    接受，防止 argmax trajectory 被误当作 Phase 2 正式性能。
    """

    if deterministic and not visualization_only:
        raise ValueError(
            "deterministic evaluation is visualization-only; set visualization_only=True explicitly"
        )
    if operating_mode_mask_fn is not None and not callable(operating_mode_mask_fn):
        raise TypeError("operating_mode_mask_fn must be callable or None")
    lambda_value = _finite_lambda(fixed_lambda)
    seeds = _evaluation_seeds(evaluation_seeds)
    num_steps = _positive_integer(evaluation_steps, name="evaluation_steps")
    if isinstance(burn_in_steps, (bool, np.bool_)) or not isinstance(burn_in_steps, Integral):
        raise TypeError("burn_in_steps must be an integer")
    burn_in = int(burn_in_steps)
    if burn_in < 0:
        raise ValueError("burn_in_steps must be non-negative")

    actor_device = _actor_device(actor, device)
    was_training = bool(getattr(actor, "training", False))
    actor.eval()
    trajectories = []
    canonical_target_ids: tuple[str, ...] | None = None
    try:
        if _supports_batched_evaluation(actor, env_factory):
            try:
                batched_trajectories, canonical_target_ids = _rollout_seeds_batched(
                    actor,
                    env_factory,
                    fixed_lambda=lambda_value,
                    seeds=seeds,
                    evaluation_steps=num_steps,
                    burn_in_steps=burn_in,
                    deterministic=deterministic,
                    visualization_only=visualization_only,
                    device=actor_device,
                    operating_mode_mask_fn=operating_mode_mask_fn,
                )
            except _BatchedEvaluationUnavailable:
                pass
            else:
                trajectories.extend(batched_trajectories)
        if not trajectories:
            for seed in seeds:
                trajectory, target_ids = _rollout_one_seed(
                    actor,
                    env_factory,
                    fixed_lambda=lambda_value,
                    seed=seed,
                    evaluation_steps=num_steps,
                    burn_in_steps=burn_in,
                    deterministic=deterministic,
                    visualization_only=visualization_only,
                    device=actor_device,
                    operating_mode_mask_fn=operating_mode_mask_fn,
                )
                if canonical_target_ids is None:
                    canonical_target_ids = target_ids
                elif target_ids != canonical_target_ids:
                    raise ValueError("all evaluation environments must use the same target order")
                trajectories.append(trajectory)
    finally:
        actor.train(was_training)

    assert canonical_target_ids is not None
    frozen_trajectories = tuple(trajectories)
    return FixedLambdaEvaluation(
        fixed_lambda=lambda_value,
        target_ids=canonical_target_ids,
        evaluation_seeds=seeds,
        burn_in_steps=burn_in,
        evaluation_steps=num_steps,
        deterministic=bool(deterministic),
        visualization_only=bool(visualization_only),
        trajectories=frozen_trajectories,
        aggregate=_aggregate_trajectories(frozen_trajectories),
    )


def evaluate_deterministic_trajectory(
    actor: ActorProtocol,
    env_factory: Callable[[], Any] | Any,
    fixed_lambda: float,
    *,
    seed: int,
    evaluation_steps: int = 2_000,
    burn_in_steps: int = 100,
    device: Any | None = None,
    operating_mode_mask_fn: OperatingModeMaskFunction | None = None,
) -> EvaluationTrajectory:
    """生成仅供可视化的 argmax trajectory，绝不作为正式 aggregate 指标。"""

    result = evaluate_fixed_lambda(
        actor,
        env_factory,
        fixed_lambda,
        [seed],
        evaluation_steps=evaluation_steps,
        burn_in_steps=burn_in_steps,
        device=device,
        deterministic=True,
        visualization_only=True,
        operating_mode_mask_fn=operating_mode_mask_fn,
    )
    return result.trajectories[0]


__all__ = [
    "ActorProtocol",
    "EvaluationAggregate",
    "EvaluationTrajectory",
    "FixedLambdaEvaluation",
    "OperatingModeMaskFunction",
    "evaluate_deterministic_trajectory",
    "evaluate_fixed_lambda",
]
