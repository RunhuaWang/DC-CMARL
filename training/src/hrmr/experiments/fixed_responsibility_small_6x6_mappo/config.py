"""6×6 单格固定责任实验的严格配置契约。"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

from hrmr.config import HRMRConfig, validate_config
from hrmr.config import load_config as load_environment_config
from hrmr.experiments.assigned_targets_ppo.environment import DEFAULT_ASSIGNMENTS
from hrmr.experiments.fixed_responsibility_mappo.config import (
    CONFIG_PATHS as BASE_CONFIG_PATHS,
)
from hrmr.experiments.fixed_responsibility_mappo.config import (
    ExperimentConfig,
    actor_initialization_seeds,
)
from hrmr.experiments.unnormalized_original_parameters_ppo.environment import (
    load_unnormalized_original_environment_config,
)
from hrmr.rl.training_config import load_training_config

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[3]
FIXED_LAMBDAS = (0.0, 2.5, 3.5, 4.5, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0)


def lambda_slug(fixed_lambda: float) -> str:
    """生成不会因小数点形成歧义的固定 λ 文件名片段。"""

    value = float(fixed_lambda)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


CONFIG_PATHS = {
    fixed_lambda: PACKAGE_ROOT / "configs" / f"lambda{lambda_slug(fixed_lambda)}_seed0.toml"
    for fixed_lambda in FIXED_LAMBDAS
}
TRAINING_CONFIG_PATHS = {
    (fixed_lambda, 0): CONFIG_PATHS[fixed_lambda] for fixed_lambda in FIXED_LAMBDAS
}
TRAINING_CONFIG_PATHS.update(
    {
        (fixed_lambda, training_seed): PACKAGE_ROOT
        / "configs"
        / f"lambda{lambda_slug(fixed_lambda)}_seed{training_seed}.toml"
        for fixed_lambda in (0.0, 9.0)
        for training_seed in (1, 2)
    }
)
MAP_ROOT = PROJECT_ROOT / "map_6x6"
OUTPUT_ROOT = MAP_ROOT / "data" / "fixed_lambda"
IMAGE_ROOT = MAP_ROOT / "images" / "fixed_lambda"
LAYOUT_OUTPUT_PATH = IMAGE_ROOT / "small_6x6_layout.png"
ASSIGNMENTS = DEFAULT_ASSIGNMENTS

_EXPECTED_INITIAL_POSITIONS = ((2, 2), (3, 2), (2, 3), (3, 3))
_EXPECTED_TARGET_CENTERS = {
    "S1": (3, 0),
    "S2": (0, 2),
    "S3": (2, 5),
    "S4": (5, 3),
    "H1": (0, 0),
    "H2": (0, 5),
    "H3": (5, 0),
    "H4": (5, 5),
}


def _read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _profile_name(fixed_lambda: float, training_seed: int) -> str:
    return (
        "fixed_responsibility_small_6x6_mappo_"
        f"lambda{lambda_slug(fixed_lambda)}_seed{training_seed}"
    )


def _expected_training_raw(fixed_lambda: float, training_seed: int) -> dict:
    """以原固定责任端点为锚，只替换实验名和环境文件。"""

    expected = copy.deepcopy(_read_toml(BASE_CONFIG_PATHS[0.0]))
    expected["experiment"]["profile"] = _profile_name(fixed_lambda, training_seed)
    expected["experiment"]["environment_config"] = "environment.toml"
    expected["assignment"]["fixed_lambda"] = fixed_lambda
    expected["assignment"]["training_seed"] = training_seed
    return expected


def load_config(path: str | Path = CONFIG_PATHS[0.0]) -> ExperimentConfig:
    """接受十个固定 λ；除端点复现实验外仅开放 seed=0。"""

    source = Path(path).expanduser().resolve()
    raw = _read_toml(source)
    assignment = raw.get("assignment")
    if not isinstance(assignment, dict):
        raise ValueError("missing [assignment] table")
    fixed_lambda = assignment.get("fixed_lambda")
    if isinstance(fixed_lambda, bool) or fixed_lambda not in CONFIG_PATHS:
        raise ValueError(f"small 6x6 experiment only supports fixed_lambda in {FIXED_LAMBDAS}")
    fixed_lambda = float(fixed_lambda)
    training_seed = assignment.get("training_seed")
    if type(training_seed) is not int or (fixed_lambda, training_seed) not in TRAINING_CONFIG_PATHS:
        raise ValueError(
            "small 6x6 experiment supports seeds 0/1/2 for lambda=0/9, "
            "and only seed=0 for intermediate lambdas"
        )
    expected = _expected_training_raw(fixed_lambda, training_seed)
    if raw != expected:
        mismatches = sorted(
            key for key in set(raw) | set(expected) if raw.get(key) != expected.get(key)
        )
        raise ValueError(f"small 6x6 training contract mismatch: {mismatches}")
    base = load_training_config(source)
    return ExperimentConfig(
        base=base,
        fixed_lambda=fixed_lambda,
        training_seed=training_seed,
        initial_state_mode="random",
        restart_interval_per_environment=1000,
        reset_rng_domain=1381192786,
        progress_every_updates=5,
    )


def validate_small_environment_config(config: HRMRConfig) -> None:
    """只允许约定的 6×6 几何，并锁定正式实验的全部非几何参数。"""

    validate_config(config)
    reference = load_unnormalized_original_environment_config()
    if (
        config.grid_width,
        config.grid_height,
        config.monitoring_radius,
        tuple(config.initial_positions),
        dict(config.target_centers),
    ) != (
        6,
        6,
        0,
        _EXPECTED_INITIAL_POSITIONS,
        _EXPECTED_TARGET_CENTERS,
    ):
        raise ValueError("small environment must use the frozen 6x6 single-cell geometry")
    unchanged = (
        config.num_agents,
        config.action_slip_probability,
        config.reward_normalizer,
        config.dual_upper_bound,
        config.use_agent_id,
        tuple(config.target_values.items()),
        tuple(config.hazard_intensities.items()),
    )
    reference_values = (
        reference.num_agents,
        reference.action_slip_probability,
        reference.reward_normalizer,
        reference.dual_upper_bound,
        reference.use_agent_id,
        tuple(reference.target_values.items()),
        tuple(reference.hazard_intensities.items()),
    )
    if unchanged != reference_values:
        raise ValueError("small environment changed a non-geometric formal parameter")
    if any(len(region) != 1 for region in config.monitoring_regions.values()):
        raise ValueError("every monitoring region must contain exactly one cell")


def load_environment(spec: ExperimentConfig) -> HRMRConfig:
    """读取训练配置旁的小地图快照，不触碰正式 13×13 配置。"""

    source = Path(spec.base.source_path).parent / spec.base.experiment.environment_config
    environment = load_environment_config(source)
    validate_small_environment_config(environment)
    return environment


def default_output(spec: ExperimentConfig) -> Path:
    """返回各固定 λ/seed 彼此隔离的结果目录。"""

    return OUTPUT_ROOT / f"lambda{lambda_slug(spec.fixed_lambda)}_seed{spec.training_seed}"


def default_image_output(spec: ExperimentConfig) -> Path:
    """返回与数值数据分离的图片目录。"""

    return IMAGE_ROOT / f"lambda{lambda_slug(spec.fixed_lambda)}_seed{spec.training_seed}"


__all__ = [
    "ASSIGNMENTS",
    "CONFIG_PATHS",
    "FIXED_LAMBDAS",
    "IMAGE_ROOT",
    "LAYOUT_OUTPUT_PATH",
    "MAP_ROOT",
    "OUTPUT_ROOT",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "TRAINING_CONFIG_PATHS",
    "ExperimentConfig",
    "actor_initialization_seeds",
    "default_image_output",
    "default_output",
    "lambda_slug",
    "load_config",
    "load_environment",
    "validate_small_environment_config",
]
