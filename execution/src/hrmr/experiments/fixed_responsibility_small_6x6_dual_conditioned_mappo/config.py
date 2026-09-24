"""6×6 固定责任区 dual-conditioned MAPPO 的正式训练契约。"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from hrmr.config import HRMRConfig
from hrmr.config import load_config as load_environment_config
from hrmr.experiments.assigned_targets_ppo.environment import DEFAULT_ASSIGNMENTS
from hrmr.experiments.fixed_responsibility_small_6x6_mappo.config import (
    CONFIG_PATHS as FIXED_CONFIG_PATHS,
)
from hrmr.experiments.fixed_responsibility_small_6x6_mappo.config import (
    load_config as load_fixed_config,
)
from hrmr.experiments.fixed_responsibility_small_6x6_mappo.config import (
    validate_small_environment_config,
)
from hrmr.rl.training_config import FixedLambdaConfig, load_training_config

from .networks import CONCATENATED_CRITIC_CONDITIONING, FILM_ACTOR_CONDITIONING
from .policy_retention import ADAPTIVE_BEST_CALIBRATION_ANCHOR_KL
from .update import PCGRAD_GRADIENT_AGGREGATION

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[3]
CONFIG_PATH = (
    PACKAGE_ROOT
    / "configs"
    / "dual_conditioned_film_full_grid_pcgrad_adaptive_retention_seed0.toml"
)
# 保留清晰的语义别名，供 IDE 可直接运行入口使用。
PCGRAD_ADAPTIVE_RETENTION_FULL_GRID_CONFIG_PATH = CONFIG_PATH

DUAL_VALUES = tuple(index / 2.0 for index in range(21))
NUM_PARALLEL_ENVS = 42
ENVIRONMENTS_PER_DUAL = 2
ROLLOUT_LENGTH = 128
MINIBATCH_SIZE = 2688
NUM_UPDATES = 4_200
TOTAL_ENVIRONMENT_STEPS = NUM_PARALLEL_ENVS * ROLLOUT_LENGTH * NUM_UPDATES
EVALUATION_INTERVAL = 2_257_920
EXPERIMENT_PROFILE = (
    "fixed_responsibility_small_6x6_dual_conditioned_"
    "film_actor_pcgrad_adaptive_retention_full_grid_mappo_seed0"
)
OUTPUT_SLUG = "film_actor_pcgrad_adaptive_retention_full_grid_21_lambda_seed0"
OUTPUT_ROOT = PROJECT_ROOT / "map_6x6" / "data" / "dual_conditioned"
IMAGE_ROOT = PROJECT_ROOT / "map_6x6" / "images" / "dual_conditioned"


@dataclass(frozen=True)
class DualConditionedConfig:
    """FiLM、PCGrad 与自适应 per-λ KL retention 的冻结配置。"""

    base: FixedLambdaConfig
    training_seed: int
    dual_values: tuple[float, ...]
    conditioning_max: float
    actor_conditioning: str
    critic_conditioning: str
    actor_gradient_aggregation: str
    policy_retention: str
    policy_retention_coefficient: float
    policy_retention_anchor_capacity: int
    policy_retention_improvement_tolerance: float
    policy_retention_apply_every_minibatch: bool
    policy_retention_target_kl: float
    policy_retention_adaptation_rate: float
    policy_retention_min_coefficient: float
    policy_retention_max_coefficient: float
    environments_per_dual: int
    dual_assignment: str
    restart_interval_per_environment: int
    reset_rng_domain: int
    progress_every_updates: int


def _finite_positive(table: dict, name: str) -> float:
    value = table.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 < float(value) < float("inf")
    ):
        raise ValueError(f"policy_retention.{name} must be finite and positive")
    return float(value)


def load_config(path: str | Path = CONFIG_PATH) -> DualConditionedConfig:
    """加载并严格验证当前唯一保留的 dual-conditioned 配置。"""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    assignment = raw.get("assignment")
    dual = raw.get("dual_conditioning")
    sampling = raw.get("sampling")
    retention = raw.get("policy_retention")
    if not all(isinstance(table, dict) for table in (assignment, dual, sampling, retention)):
        raise ValueError(
            "missing assignment, dual_conditioning, sampling, or policy_retention table"
        )

    base = load_training_config(source)
    values = tuple(float(value) for value in dual.get("values", ()))
    if values != DUAL_VALUES:
        raise ValueError("dual_conditioning.values must be the 21-point grid 0, 0.5, ..., 10")
    expected_dual = {
        "conditioning_max": 10.0,
        "assignment": "fixed_environment_slots",
        "environments_per_dual": ENVIRONMENTS_PER_DUAL,
        "advantage_normalization": "per_lambda",
        "minibatch_sampling": "stratified_by_lambda",
        "actor_conditioning": FILM_ACTOR_CONDITIONING,
        "critic_conditioning": CONCATENATED_CRITIC_CONDITIONING,
        "actor_gradient_aggregation": PCGRAD_GRADIENT_AGGREGATION,
        "progress_every_updates": 100,
    }
    for name, expected in expected_dual.items():
        if dual.get(name) != expected:
            raise ValueError(f"dual_conditioning.{name} must equal {expected!r}")

    if base.experiment.profile != EXPERIMENT_PROFILE:
        raise ValueError("experiment profile does not match the retained formal method")
    expected_training = (
        NUM_PARALLEL_ENVS,
        ROLLOUT_LENGTH,
        TOTAL_ENVIRONMENT_STEPS,
        MINIBATCH_SIZE,
        EVALUATION_INTERVAL,
    )
    actual_training = (
        base.training.num_parallel_envs,
        base.training.rollout_length,
        base.training.total_environment_steps,
        base.training.minibatch_size,
        base.training.evaluation_interval,
    )
    if actual_training != expected_training:
        raise ValueError("dual-conditioned training dimensions changed")

    reference = load_fixed_config(FIXED_CONFIG_PATHS[0.0]).base
    current_training = asdict(base.training)
    reference_training = asdict(reference.training)
    allowed_training_changes = {
        "num_parallel_envs",
        "rollout_length",
        "total_environment_steps",
        "minibatch_size",
        "evaluation_interval",
    }
    for field in allowed_training_changes:
        current_training.pop(field)
        reference_training.pop(field)
    if current_training != reference_training:
        raise ValueError("dual-conditioned run changed a frozen PPO hyperparameter")
    current_experiment = asdict(base.experiment)
    reference_experiment = asdict(reference.experiment)
    current_experiment.pop("profile")
    reference_experiment.pop("profile")
    if current_experiment != reference_experiment:
        raise ValueError("dual-conditioned run changed a frozen network setting")
    if base.environment != reference.environment or base.evaluation != reference.evaluation:
        raise ValueError("dual-conditioned run changed environment or formal evaluation settings")

    if assignment.get("training_seed") != 0:
        raise ValueError("the retained formal run requires training_seed=0")
    if assignment.get("agent_targets") != [list(pair) for pair in DEFAULT_ASSIGNMENTS]:
        raise ValueError("fixed responsibility assignment changed")
    expected_assignment = {
        "actor_input_dimension": 57,
        "critic_input_dimension": 17,
        "reward_eligibility": "owner_only",
        "cost_scope": "all_physical_hazard_exposure",
    }
    for name, expected in expected_assignment.items():
        if assignment.get(name) != expected:
            raise ValueError(f"assignment.{name} must equal {expected!r}")

    expected_sampling = {
        "initial_state_distribution": "uniform_without_replacement_all_grid_cells",
        "restart_interval_per_environment": 1000,
        "reset_rng_domain": 1381192786,
    }
    for name, expected in expected_sampling.items():
        if sampling.get(name) != expected:
            raise ValueError(f"sampling.{name} must equal {expected!r}")

    expected_retention_keys = {
        "mode",
        "coefficient",
        "anchor_capacity",
        "improvement_tolerance",
        "apply_every_minibatch",
        "target_kl",
        "adaptation_rate",
        "min_coefficient",
        "max_coefficient",
    }
    if set(retention) != expected_retention_keys:
        raise ValueError("policy_retention must define exactly the formal adaptive settings")
    if retention.get("mode") != ADAPTIVE_BEST_CALIBRATION_ANCHOR_KL:
        raise ValueError("only adaptive_best_calibration_anchor_kl retention is retained")
    coefficient = _finite_positive(retention, "coefficient")
    target_kl = _finite_positive(retention, "target_kl")
    adaptation_rate = _finite_positive(retention, "adaptation_rate")
    min_coefficient = _finite_positive(retention, "min_coefficient")
    max_coefficient = _finite_positive(retention, "max_coefficient")
    anchor_capacity = retention.get("anchor_capacity")
    if isinstance(anchor_capacity, bool) or not isinstance(anchor_capacity, int):
        raise ValueError("policy_retention.anchor_capacity must be a positive integer")
    if anchor_capacity <= 0:
        raise ValueError("policy_retention.anchor_capacity must be a positive integer")
    improvement_tolerance = retention.get("improvement_tolerance")
    if (
        isinstance(improvement_tolerance, bool)
        or not isinstance(improvement_tolerance, (int, float))
        or not 0.0 <= float(improvement_tolerance) < float("inf")
    ):
        raise ValueError("policy_retention.improvement_tolerance must be finite and non-negative")
    if retention.get("apply_every_minibatch") is not True:
        raise ValueError("adaptive policy retention must apply on every PPO minibatch")
    if not min_coefficient <= coefficient <= max_coefficient:
        raise ValueError("initial retention coefficient must lie within its bounds")

    return DualConditionedConfig(
        base=base,
        training_seed=0,
        dual_values=values,
        conditioning_max=10.0,
        actor_conditioning=FILM_ACTOR_CONDITIONING,
        critic_conditioning=CONCATENATED_CRITIC_CONDITIONING,
        actor_gradient_aggregation=PCGRAD_GRADIENT_AGGREGATION,
        policy_retention=ADAPTIVE_BEST_CALIBRATION_ANCHOR_KL,
        policy_retention_coefficient=coefficient,
        policy_retention_anchor_capacity=anchor_capacity,
        policy_retention_improvement_tolerance=float(improvement_tolerance),
        policy_retention_apply_every_minibatch=True,
        policy_retention_target_kl=target_kl,
        policy_retention_adaptation_rate=adaptation_rate,
        policy_retention_min_coefficient=min_coefficient,
        policy_retention_max_coefficient=max_coefficient,
        environments_per_dual=ENVIRONMENTS_PER_DUAL,
        dual_assignment="fixed_environment_slots",
        restart_interval_per_environment=1000,
        reset_rng_domain=1381192786,
        progress_every_updates=100,
    )


def default_output(spec: DualConditionedConfig) -> Path:
    """返回当前正式方法的数据输出目录。"""

    del spec
    return OUTPUT_ROOT / OUTPUT_SLUG


def default_image_output(spec: DualConditionedConfig) -> Path:
    """返回当前正式方法的图片输出目录。"""

    del spec
    return IMAGE_ROOT / OUTPUT_SLUG


def load_environment(spec: DualConditionedConfig) -> HRMRConfig:
    """加载隔离的 6×6 快照并复用冻结几何校验。"""

    source = Path(spec.base.source_path).parent / spec.base.experiment.environment_config
    environment = load_environment_config(source)
    validate_small_environment_config(environment)
    return environment


__all__ = [
    "CONFIG_PATH",
    "DUAL_VALUES",
    "ENVIRONMENTS_PER_DUAL",
    "EVALUATION_INTERVAL",
    "IMAGE_ROOT",
    "MINIBATCH_SIZE",
    "NUM_PARALLEL_ENVS",
    "NUM_UPDATES",
    "OUTPUT_ROOT",
    "PACKAGE_ROOT",
    "PCGRAD_ADAPTIVE_RETENTION_FULL_GRID_CONFIG_PATH",
    "PROJECT_ROOT",
    "ROLLOUT_LENGTH",
    "TOTAL_ENVIRONMENT_STEPS",
    "DualConditionedConfig",
    "default_image_output",
    "default_output",
    "load_config",
    "load_environment",
]
