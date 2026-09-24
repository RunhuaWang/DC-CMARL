"""固定责任区 MAPPO 配置：固定实验契约，不在加载阶段采样或创建模型。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrmr.config import HRMRConfig
from hrmr.experiments.assigned_targets_ppo.environment import DEFAULT_ASSIGNMENTS
from hrmr.experiments.unnormalized_original_parameters_ppo.environment import (
    load_unnormalized_original_environment_config,
)
from hrmr.rl.training_config import FixedLambdaConfig, load_training_config

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[3]
CONFIG_PATHS = {
    0.0: PACKAGE_ROOT / "configs" / "lambda0_seed0.toml",
    9.0: PACKAGE_ROOT / "configs" / "lambda9_seed0.toml",
}
TRAINING_CONFIG_PATHS = {
    (0.0, 0): CONFIG_PATHS[0.0],
    (0.0, 1): PACKAGE_ROOT / "configs" / "lambda0_seed1.toml",
    (0.0, 2): PACKAGE_ROOT / "configs" / "lambda0_seed2.toml",
    (9.0, 0): CONFIG_PATHS[9.0],
}
MAP_ROOT = PROJECT_ROOT / "map_13x13"
OUTPUT_ROOT = MAP_ROOT / "data"
IMAGE_ROOT = MAP_ROOT / "images"
ASSIGNMENTS = DEFAULT_ASSIGNMENTS

_TRAINING_CONTRACT = {
    "num_parallel_envs": 32,
    "rollout_length": 128,
    "total_environment_steps": 500000,
    "ppo_epochs": 5,
    "minibatch_size": 2048,
    "actor_learning_rate": 0.0003,
    "critic_learning_rate": 0.0003,
    "ppo_clip": 0.2,
    "max_grad_norm": 0.5,
    "entropy_coefficient_start": 0.02,
    "entropy_coefficient_end": 0.02,
    "entropy_decay_fraction": 0.70,
    "avg_rate_ema": 0.05,
    "value_anchor_coefficient": 0.001,
    "normalize_combined_advantage": True,
    "n_step": 32,
    "evaluation_interval": 25000,
    "early_stopping": False,
    "early_stopping_patience": 3,
    "objective_tolerance": 0.005,
    "occupancy_tolerance": 0.05,
    "analytic_tolerance": 0.08,
}
_SAMPLING_CONTRACT = {
    "initial_state_distribution": "uniform_without_replacement_all_grid_cells",
    "restart_interval_per_environment": 1000,
    "reset_rng_domain": 1381192786,
    "best_selection": "fixed_initial_canonical_objective",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """将固定 λ、责任分配采样设置与既有 differential PPO 配置组合。"""

    base: FixedLambdaConfig
    fixed_lambda: float
    training_seed: int
    initial_state_mode: str
    restart_interval_per_environment: int
    reset_rng_domain: int
    progress_every_updates: int


def _expected_raw(fixed_lambda: float, training_seed: int = 0) -> dict[str, Any]:
    """固定复用已授权超参数；额外训练 seed 仅改变 seed 与 profile。"""

    return {
        "experiment": {
            "profile": (
                f"fixed_responsibility_mappo_lambda{int(fixed_lambda)}_seed{training_seed}"
            ),
            "environment_config": "environment.toml",
            "activation": "tanh",
            "device": "auto",
            "torch_num_threads": 1,
            "randomize_actor_target_order": True,
            "independent_actors": True,
            "independent_actor_init_seed_base": 100,
        },
        "training": dict(_TRAINING_CONTRACT),
        "environment": {"action_slip_probability": 0.0},
        "evaluation": {
            "seeds": list(range(1000, 1020)),
            "burn_in_steps": 100,
            "evaluation_steps": 2000,
        },
        "assignment": {
            "fixed_lambda": fixed_lambda,
            "training_seed": training_seed,
            "agent_targets": [list(pair) for pair in ASSIGNMENTS],
            "actor_input_dimension": 56,
            "critic_input_dimension": 16,
            "reward_eligibility": "owner_only",
            "cost_scope": "all_physical_hazard_exposure",
            "progress_every_updates": 5,
        },
        "sampling": dict(_SAMPLING_CONTRACT),
    }


def load_config(path: str | Path = CONFIG_PATHS[0.0]) -> ExperimentConfig:
    """严格加载 λ=0 的 seed=0/1/2 或 λ=9 的 seed=0，不改变其他实验契约。"""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    assignment = raw.get("assignment")
    if not isinstance(assignment, dict):
        raise ValueError("missing [assignment] table")
    fixed_lambda = assignment.get("fixed_lambda")
    if (
        isinstance(fixed_lambda, bool)
        or not isinstance(fixed_lambda, (int, float))
        or fixed_lambda not in CONFIG_PATHS
    ):
        raise ValueError("fixed_responsibility_mappo only supports fixed_lambda=0 or 9")
    training_seed = assignment.get("training_seed")
    if type(training_seed) is not int:
        raise ValueError("assignment.training_seed must be an integer")
    if (float(fixed_lambda), training_seed) not in TRAINING_CONFIG_PATHS:
        raise ValueError(
            "fixed_responsibility_mappo only supports lambda=0 with seed=0/1/2 "
            "or lambda=9 with seed=0"
        )
    expected = _expected_raw(float(fixed_lambda), training_seed)
    if raw != expected:
        mismatches = sorted(
            key for key in set(raw) | set(expected) if raw.get(key) != expected.get(key)
        )
        raise ValueError(f"fixed responsibility configuration contract mismatch: {mismatches}")
    for table, name in (
        ("assignment", "training_seed"),
        ("assignment", "progress_every_updates"),
        ("sampling", "restart_interval_per_environment"),
        ("sampling", "reset_rng_domain"),
    ):
        if type(raw[table][name]) is not int:
            raise ValueError(f"{table}.{name} must be an integer")
    base = load_training_config(source)
    return ExperimentConfig(
        base=base,
        fixed_lambda=float(fixed_lambda),
        training_seed=training_seed,
        initial_state_mode="random",
        restart_interval_per_environment=raw["sampling"]["restart_interval_per_environment"],
        reset_rng_domain=raw["sampling"]["reset_rng_domain"],
        progress_every_updates=assignment["progress_every_updates"],
    )


def load_environment(spec: ExperimentConfig) -> HRMRConfig:
    """只读取配置同目录的原始 balanced 3×3 快照，不构造或 reset 环境。"""

    source = Path(spec.base.source_path).parent / spec.base.experiment.environment_config
    environment = load_unnormalized_original_environment_config(source)
    if any(len(region) != 9 for region in environment.monitoring_regions.values()):
        raise ValueError("fixed responsibility MAPPO requires the original balanced 3x3 regions")
    return environment


def default_output(spec: ExperimentConfig) -> Path:
    """返回每个固定 λ 的独立默认输出目录，不创建目录。"""

    return OUTPUT_ROOT / f"lambda{int(spec.fixed_lambda)}_seed{spec.training_seed}"


def default_image_output(spec: ExperimentConfig) -> Path:
    """返回与数值数据分离的图片目录。"""

    return IMAGE_ROOT / f"lambda{int(spec.fixed_lambda)}_seed{spec.training_seed}"


def actor_initialization_seeds(spec: ExperimentConfig) -> tuple[int, ...]:
    """每个训练 seed 使用独立的四个 actor 初始化 seed；seed=0 保持原值。"""

    seed_base = spec.base.experiment.independent_actor_init_seed_base + 4 * spec.training_seed
    return tuple(seed_base + agent for agent in range(4))


__all__ = [
    "ASSIGNMENTS",
    "CONFIG_PATHS",
    "IMAGE_ROOT",
    "MAP_ROOT",
    "OUTPUT_ROOT",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "TRAINING_CONFIG_PATHS",
    "ExperimentConfig",
    "actor_initialization_seeds",
    "default_image_output",
    "default_output",
    "load_config",
    "load_environment",
]
