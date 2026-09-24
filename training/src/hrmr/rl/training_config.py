"""Phase 2 fixed-lambda 训练配置。"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSettings:
    """实验级设置。"""

    profile: str
    environment_config: str
    activation: str
    device: str
    torch_num_threads: int
    randomize_actor_target_order: bool = False
    independent_actors: bool = False
    independent_actor_init_seed_base: int = 100
    actor_target_view: str = "all"


@dataclass(frozen=True)
class TrainingSettings:
    """PPO 与 differential actor-critic 超参数。"""

    num_parallel_envs: int
    rollout_length: int
    total_environment_steps: int
    ppo_epochs: int
    minibatch_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    ppo_clip: float
    max_grad_norm: float
    entropy_coefficient_start: float
    entropy_coefficient_end: float
    entropy_decay_fraction: float
    avg_rate_ema: float
    value_anchor_coefficient: float
    normalize_combined_advantage: bool
    n_step: int
    evaluation_interval: int
    early_stopping: bool
    early_stopping_patience: int
    objective_tolerance: float
    occupancy_tolerance: float
    analytic_tolerance: float


@dataclass(frozen=True)
class EnvironmentSettings:
    """训练时允许显式覆盖的环境设置。"""

    action_slip_probability: float


@dataclass(frozen=True)
class EvaluationSettings:
    """共同 stochastic evaluation 协议。"""

    seeds: tuple[int, ...]
    burn_in_steps: int
    evaluation_steps: int


@dataclass(frozen=True)
class FixedLambdaConfig:
    """一个 fixed-lambda run 的完整、可序列化配置。"""

    experiment: ExperimentSettings
    training: TrainingSettings
    environment: EnvironmentSettings
    evaluation: EvaluationSettings
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        """返回适合 checkpoint/JSON 的普通容器。"""

        return asdict(self)


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _probability(value: Any, name: str, *, positive: bool = False) -> float:
    result = float(value)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not lower_ok or result > 1.0:
        boundary = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {boundary}")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def load_training_config(path: str | Path) -> FixedLambdaConfig:
    """读取并严格验证 Phase 2 TOML。"""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    experiment_raw = _table(raw, "experiment")
    training_raw = _table(raw, "training")
    environment_raw = _table(raw, "environment")
    evaluation_raw = _table(raw, "evaluation")

    experiment = ExperimentSettings(
        profile=str(experiment_raw["profile"]),
        environment_config=str(experiment_raw["environment_config"]),
        activation=str(experiment_raw["activation"]).lower(),
        device=str(experiment_raw["device"]).lower(),
        torch_num_threads=_positive_int(experiment_raw["torch_num_threads"], "torch_num_threads"),
        randomize_actor_target_order=_boolean(
            experiment_raw.get("randomize_actor_target_order", False),
            "randomize_actor_target_order",
        ),
        independent_actors=_boolean(
            experiment_raw.get("independent_actors", False),
            "independent_actors",
        ),
        independent_actor_init_seed_base=_nonnegative_int(
            experiment_raw.get("independent_actor_init_seed_base", 100),
            "independent_actor_init_seed_base",
        ),
        actor_target_view=str(experiment_raw.get("actor_target_view", "all")).lower(),
    )
    if experiment.activation not in {"tanh", "relu"}:
        raise ValueError("activation must be 'tanh' or 'relu'")
    if experiment.device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    if experiment.actor_target_view not in {"all", "safe_only"}:
        raise ValueError("actor_target_view must be 'all' or 'safe_only'")

    training = TrainingSettings(
        num_parallel_envs=_positive_int(training_raw["num_parallel_envs"], "num_parallel_envs"),
        rollout_length=_positive_int(training_raw["rollout_length"], "rollout_length"),
        total_environment_steps=_positive_int(
            training_raw["total_environment_steps"], "total_environment_steps"
        ),
        ppo_epochs=_positive_int(training_raw["ppo_epochs"], "ppo_epochs"),
        minibatch_size=_positive_int(training_raw["minibatch_size"], "minibatch_size"),
        actor_learning_rate=_positive_float(
            training_raw["actor_learning_rate"], "actor_learning_rate"
        ),
        critic_learning_rate=_positive_float(
            training_raw["critic_learning_rate"], "critic_learning_rate"
        ),
        ppo_clip=_probability(training_raw["ppo_clip"], "ppo_clip", positive=True),
        max_grad_norm=_positive_float(training_raw["max_grad_norm"], "max_grad_norm"),
        entropy_coefficient_start=_nonnegative_float(
            training_raw["entropy_coefficient_start"], "entropy_coefficient_start"
        ),
        entropy_coefficient_end=_nonnegative_float(
            training_raw["entropy_coefficient_end"], "entropy_coefficient_end"
        ),
        entropy_decay_fraction=_probability(
            training_raw["entropy_decay_fraction"],
            "entropy_decay_fraction",
            positive=True,
        ),
        avg_rate_ema=_probability(training_raw["avg_rate_ema"], "avg_rate_ema", positive=True),
        value_anchor_coefficient=_nonnegative_float(
            training_raw["value_anchor_coefficient"], "value_anchor_coefficient"
        ),
        normalize_combined_advantage=_boolean(
            training_raw["normalize_combined_advantage"], "normalize_combined_advantage"
        ),
        n_step=_positive_int(training_raw["n_step"], "n_step"),
        evaluation_interval=_positive_int(
            training_raw["evaluation_interval"], "evaluation_interval"
        ),
        early_stopping=_boolean(training_raw["early_stopping"], "early_stopping"),
        early_stopping_patience=_positive_int(
            training_raw["early_stopping_patience"], "early_stopping_patience"
        ),
        objective_tolerance=_nonnegative_float(
            training_raw["objective_tolerance"], "objective_tolerance"
        ),
        occupancy_tolerance=_nonnegative_float(
            training_raw["occupancy_tolerance"], "occupancy_tolerance"
        ),
        analytic_tolerance=_nonnegative_float(
            training_raw["analytic_tolerance"], "analytic_tolerance"
        ),
    )
    if training.entropy_coefficient_end > training.entropy_coefficient_start:
        raise ValueError("entropy coefficient schedule must be non-increasing")
    if training.total_environment_steps % training.num_parallel_envs != 0:
        raise ValueError(
            "total_environment_steps must be divisible by num_parallel_envs; "
            "a synchronous vector step cannot execute a partial environment batch"
        )

    environment = EnvironmentSettings(
        action_slip_probability=_probability(
            environment_raw["action_slip_probability"], "action_slip_probability"
        )
    )
    seeds_raw = evaluation_raw["seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("evaluation.seeds must be a non-empty list")
    seeds = tuple(_nonnegative_int(seed, "evaluation seed") for seed in seeds_raw)
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    evaluation = EvaluationSettings(
        seeds=seeds,
        burn_in_steps=_nonnegative_int(evaluation_raw["burn_in_steps"], "burn_in_steps"),
        evaluation_steps=_positive_int(evaluation_raw["evaluation_steps"], "evaluation_steps"),
    )
    return FixedLambdaConfig(experiment, training, environment, evaluation, str(source))


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def resolve_environment_config(config: FixedLambdaConfig) -> Path:
    """解析环境配置路径，同时支持仓库根目录与已安装调用。"""

    configured = Path(config.experiment.environment_config).expanduser()
    candidates = (
        configured,
        Path(config.source_path).parent / configured,
        Path(config.source_path).parent.parent / configured,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(config.experiment.environment_config)


__all__ = [
    "EnvironmentSettings",
    "EvaluationSettings",
    "ExperimentSettings",
    "FixedLambdaConfig",
    "TrainingSettings",
    "load_training_config",
    "resolve_environment_config",
]
