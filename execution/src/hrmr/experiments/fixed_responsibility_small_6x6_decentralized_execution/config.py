"""6×6 无中心执行阶段的冻结实验配置。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .communication import Graph, build_undirected_graph, graph_diameter

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[3]
C0_H64_CONFIG_PATH = PACKAGE_ROOT / "configs" / "decentralized_execution_c0_h64.toml"
C0P1_H64_CONFIG_PATH = PACKAGE_ROOT / "configs" / "decentralized_execution_c0p1_h64.toml"
C0P2_H64_CONFIG_PATH = PACKAGE_ROOT / "configs" / "decentralized_execution_c0p2_h64.toml"
C0P4_H64_CONFIG_PATH = PACKAGE_ROOT / "configs" / "decentralized_execution_c0p4_h64.toml"
C0P6_H64_CONFIG_PATH = PACKAGE_ROOT / "configs" / "decentralized_execution_c0p6_h64.toml"
C0P9_H64_CONFIG_PATH = PACKAGE_ROOT / "configs" / "decentralized_execution_c0p9_h64.toml"
C1_H64_CONFIG_PATH = PACKAGE_ROOT / "configs" / "decentralized_execution_c1_h64.toml"
C1P2_H64_CONFIG_PATH = PACKAGE_ROOT / "configs" / "decentralized_execution_c1p2_h64.toml"
CONFIG_PATH = C0_H64_CONFIG_PATH
EXPECTED_PROFILE = "fixed_responsibility_small_6x6_decentralized_dual_execution"
C0_H64_PROFILE = f"{EXPECTED_PROFILE}_c0_h64_1024_updates"
C0P1_H64_PROFILE = f"{EXPECTED_PROFILE}_c0p1_h64_1024_updates"
C0P2_H64_PROFILE = f"{EXPECTED_PROFILE}_c0p2_h64_1024_updates"
C0P4_H64_PROFILE = f"{EXPECTED_PROFILE}_c0p4_h64_1024_updates"
C0P6_H64_PROFILE = f"{EXPECTED_PROFILE}_c0p6_h64_1024_updates"
C0P9_H64_PROFILE = f"{EXPECTED_PROFILE}_c0p9_h64_1024_updates"
C1_H64_PROFILE = f"{EXPECTED_PROFILE}_c1_h64_1024_updates"
C1P2_H64_PROFILE = f"{EXPECTED_PROFILE}_c1p2_h64_1024_updates"
EXPECTED_THRESHOLDS = {
    C0_H64_PROFILE: (0.0,),
    C0P1_H64_PROFILE: (0.1,),
    C0P2_H64_PROFILE: (0.2,),
    C0P4_H64_PROFILE: (0.4,),
    C0P6_H64_PROFILE: (0.6,),
    C0P9_H64_PROFILE: (0.9,),
    C1_H64_PROFILE: (1.0,),
    C1P2_H64_PROFILE: (1.2,),
}
EXPECTED_TOTAL_STEPS = {profile: 65_536 for profile in EXPECTED_THRESHOLDS}
EXPECTED_HORIZONS = {profile: 64 for profile in EXPECTED_THRESHOLDS}


@dataclass(frozen=True)
class DecentralizedExecutionConfig:
    """固定策略下的通信、对偶动态与评价输出契约。"""

    source_path: Path
    profile: str
    checkpoint_path: Path
    training_config_path: Path
    target_order: str
    device: str
    constraint_thresholds: tuple[float, ...]
    evaluation_seeds: tuple[int, ...]
    dual_min: float
    dual_max: float
    initial_dual: float
    dual_step_size: float
    cost_estimation_horizon: int
    total_environment_steps: int
    progress_every_dual_updates: int
    graph: Graph
    communication_rounds: int
    data_directory: Path
    image_directory: Path
    representative_seed: int

    @property
    def num_dual_updates(self) -> int:
        return self.total_environment_steps // self.cost_estimation_horizon


def _positive_int(table: dict, name: str) -> int:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(table: dict, name: str) -> float:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    checked = float(value)
    if not float("-inf") < checked < float("inf"):
        raise ValueError(f"{name} must be a finite real number")
    return checked


def load_config(path: str | Path = CONFIG_PATH) -> DecentralizedExecutionConfig:
    """加载并验证用户确认的 λ0=0 无中心执行设置。"""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = raw.get("experiment")
    execution = raw.get("execution")
    communication = raw.get("communication")
    output = raw.get("output")
    if not all(isinstance(table, dict) for table in (experiment, execution, communication, output)):
        raise ValueError("missing experiment, execution, communication, or output table")

    profile = experiment.get("profile")
    if profile not in EXPECTED_THRESHOLDS:
        raise ValueError(f"experiment.profile must be one of {tuple(EXPECTED_THRESHOLDS)!r}")
    target_order = experiment.get("target_order")
    if target_order != "canonical":
        raise ValueError("the formal decentralized run uses canonical target order")
    device = experiment.get("device")
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("experiment.device is invalid")
    checkpoint_path = (PROJECT_ROOT / str(experiment.get("checkpoint", ""))).resolve()
    training_config_path = (PROJECT_ROOT / str(experiment.get("training_config", ""))).resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    if not training_config_path.is_file():
        raise ValueError(f"training config does not exist: {training_config_path}")

    thresholds_raw = execution.get("constraint_thresholds")
    if not isinstance(thresholds_raw, list) or not thresholds_raw:
        raise ValueError("execution.constraint_thresholds must be a non-empty list")
    thresholds = tuple(float(value) for value in thresholds_raw)
    if thresholds != EXPECTED_THRESHOLDS[profile]:
        raise ValueError("constraint thresholds do not match the selected execution profile")
    seeds_raw = execution.get("evaluation_seeds")
    if not isinstance(seeds_raw, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds_raw
    ):
        raise ValueError("execution.evaluation_seeds must contain non-negative integers")
    seeds = tuple(seeds_raw)
    if seeds != tuple(range(1000, 1020)):
        raise ValueError("formal evaluation seeds must equal 1000,...,1019")

    dual_min = _finite_number(execution, "dual_min")
    dual_max = _finite_number(execution, "dual_max")
    initial_dual = _finite_number(execution, "initial_dual")
    dual_step_size = _finite_number(execution, "dual_step_size")
    horizon = _positive_int(execution, "cost_estimation_horizon")
    total_steps = _positive_int(execution, "total_environment_steps")
    progress_every = _positive_int(execution, "progress_every_dual_updates")
    if (dual_min, dual_max, initial_dual, dual_step_size) != (0.0, 10.0, 0.0, 0.25):
        raise ValueError("dual range, λ0, or dual step size changed from the confirmed settings")
    if (
        horizon != EXPECTED_HORIZONS[profile]
        or total_steps != EXPECTED_TOTAL_STEPS[profile]
        or total_steps % horizon
    ):
        raise ValueError("execution H or T does not match the selected execution profile")
    if progress_every != 32:
        raise ValueError("progress interval must equal 32 dual updates")
    if execution.get("stochastic_action_sampling") is not True:
        raise ValueError("formal execution must use stochastic action sampling")
    if execution.get("periodic_environment_resets") is not False:
        raise ValueError("continuing execution cannot use periodic resets")

    edges_raw = communication.get("edges")
    if not isinstance(edges_raw, list):
        raise ValueError("communication.edges must be a list")
    # TOML 使用论文中的 1-based agent 编号。
    zero_based_edges = tuple(tuple(int(value) - 1 for value in edge) for edge in edges_raw)
    graph = build_undirected_graph(4, zero_based_edges)
    expected_graph = ((1, 3), (0, 2), (1, 3), (0, 2))
    if graph != expected_graph:
        raise ValueError("communication graph must be the confirmed four-agent ring")
    rounds = _positive_int(communication, "rounds")
    if rounds != graph_diameter(graph) or rounds != 2:
        raise ValueError("communication rounds must equal the ring graph diameter 2")
    if communication.get("synchronous") is not True:
        raise ValueError("the first execution experiment assumes synchronous communication")
    if communication.get("reliable") is not True:
        raise ValueError("the first execution experiment assumes reliable communication")
    if communication.get("aggregation") != "tagged_set_union_then_sum":
        raise ValueError("communication aggregation contract changed")

    data_directory = (PROJECT_ROOT / str(output.get("data_directory", ""))).resolve()
    image_directory = (PROJECT_ROOT / str(output.get("image_directory", ""))).resolve()
    representative_seed = output.get("representative_seed")
    if representative_seed != 1000 or representative_seed not in seeds:
        raise ValueError("representative seed must equal 1000")

    return DecentralizedExecutionConfig(
        source_path=source,
        profile=profile,
        checkpoint_path=checkpoint_path,
        training_config_path=training_config_path,
        target_order=target_order,
        device=device,
        constraint_thresholds=thresholds,
        evaluation_seeds=seeds,
        dual_min=dual_min,
        dual_max=dual_max,
        initial_dual=initial_dual,
        dual_step_size=dual_step_size,
        cost_estimation_horizon=horizon,
        total_environment_steps=total_steps,
        progress_every_dual_updates=progress_every,
        graph=graph,
        communication_rounds=rounds,
        data_directory=data_directory,
        image_directory=image_directory,
        representative_seed=representative_seed,
    )


__all__ = [
    "C0P1_H64_CONFIG_PATH",
    "C0P2_H64_CONFIG_PATH",
    "C0P4_H64_CONFIG_PATH",
    "C0P6_H64_CONFIG_PATH",
    "C0P9_H64_CONFIG_PATH",
    "C0_H64_CONFIG_PATH",
    "C1P2_H64_CONFIG_PATH",
    "C1_H64_CONFIG_PATH",
    "CONFIG_PATH",
    "DecentralizedExecutionConfig",
    "load_config",
]
