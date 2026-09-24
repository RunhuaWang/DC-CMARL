"""HRMR 正式环境的 TOML 配置加载与严格验证。"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:  # Python >=3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 仅用于本地 Python 3.9/3.10 验证
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Python <3.11 requires 'tomli' to load HRMR TOML configuration") from exc

from hrmr.constants import (
    HAZARDOUS_TARGET_IDS,
    PDF_COORDINATE_SYSTEM,
    SAFE_TARGET_IDS,
    TARGET_IDS,
)
from hrmr.geometry import (
    compute_monitoring_regions,
    pdf_to_internal,
    position_in_grid,
)
from hrmr.types import MonitoringRegions, Position, TargetPositions, TargetScalars


def _resolve_default_config_path() -> Path:
    """同时支持源码/editable 与普通 wheel 安装的默认配置定位。"""

    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[2] / "configs" / "hrmr_formal_environment.toml",
        module_path.parents[1] / "configs" / "hrmr_formal_environment.toml",
        Path(sys.prefix) / "configs" / "hrmr_formal_environment.toml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


DEFAULT_CONFIG_PATH = _resolve_default_config_path()

_TOP_LEVEL_FIELDS = frozenset(
    {
        "coords",
        "grid_width",
        "grid_height",
        "num_agents",
        "monitoring_radius",
        "action_slip_probability",
        "initial_positions",
        "target_centers",
        "target_values",
        "hazard_intensities",
        "reward_normalizer",
        "dual_upper_bound",
        "use_agent_id",
    }
)


def _canonical_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """复制并按 PDF occupancy 向量顺序固定 mapping。"""

    canonical = {target_id: values[target_id] for target_id in TARGET_IDS if target_id in values}
    canonical.update({key: value for key, value in values.items() if key not in canonical})
    return MappingProxyType(canonical)


@dataclass(frozen=True)
class HRMRConfig:
    """HRMR 环境配置（坐标已是内部表示）。"""

    grid_width: int
    grid_height: int
    num_agents: int
    monitoring_radius: int
    action_slip_probability: float
    initial_positions: tuple[Position, ...]
    target_centers: TargetPositions
    target_values: TargetScalars
    hazard_intensities: TargetScalars
    reward_normalizer: float
    dual_upper_bound: float
    use_agent_id: bool

    def __post_init__(self) -> None:
        # frozen dataclass 仍需防止内嵌 dict/list 被外部篡改。
        try:
            positions = tuple(tuple(position) for position in self.initial_positions)
        except TypeError as exc:
            raise TypeError("initial_positions must be a sequence of positions") from exc
        object.__setattr__(self, "initial_positions", positions)

        if not isinstance(self.target_centers, Mapping):
            raise TypeError("target_centers must be a mapping")
        centers = {
            target_id: tuple(position) for target_id, position in self.target_centers.items()
        }
        object.__setattr__(self, "target_centers", _canonical_mapping(centers))

        if not isinstance(self.target_values, Mapping):
            raise TypeError("target_values must be a mapping")
        object.__setattr__(self, "target_values", _canonical_mapping(self.target_values))

        if not isinstance(self.hazard_intensities, Mapping):
            raise TypeError("hazard_intensities must be a mapping")
        object.__setattr__(
            self,
            "hazard_intensities",
            _canonical_mapping(self.hazard_intensities),
        )

    @property
    def target_ids(self) -> tuple[str, ...]:
        """PDF 式 (12.9) 定义的规范目标顺序。"""

        return TARGET_IDS

    @property
    def num_targets(self) -> int:
        return len(TARGET_IDS)

    @property
    def monitoring_regions(self) -> MonitoringRegions:
        """由内部中心坐标派生当前监控区域。"""

        return compute_monitoring_regions(
            self.target_centers,
            self.monitoring_radius,
            self.grid_height,
            self.grid_width,
        )


def _validate_exact_keys(name: str, values: Mapping[str, Any]) -> None:
    actual = set(values)
    expected = set(TARGET_IDS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{name} target ids mismatch; missing={missing}, extra={extra}")


def _require_integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < minimum:
        comparison = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {comparison}")
    return integer


def _require_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_internal_position(name: str, position: Sequence[int], config: HRMRConfig) -> Position:
    if isinstance(position, (str, bytes)):
        raise TypeError(f"{name} must be a two-element integer coordinate")
    try:
        coordinate = tuple(position)
    except TypeError as exc:
        raise TypeError(f"{name} must be a two-element integer coordinate") from exc
    if len(coordinate) != 2:
        raise ValueError(f"{name} must contain exactly two components")
    if any(
        isinstance(component, bool) or not isinstance(component, Integral)
        for component in coordinate
    ):
        raise TypeError(f"{name} components must be integers")
    internal = int(coordinate[0]), int(coordinate[1])
    if not position_in_grid(internal, config.grid_height, config.grid_width):
        raise ValueError(f"{name} lies outside the grid: {internal}")
    return internal


def validate_config(config: HRMRConfig) -> None:
    """验证 PDF 中 HRMR 基础环境的全部结构性约束。"""

    if not isinstance(config, HRMRConfig):
        raise TypeError("config must be an HRMRConfig")

    _require_integer("grid_width", config.grid_width, minimum=1)
    _require_integer("grid_height", config.grid_height, minimum=1)
    num_agents = _require_integer("num_agents", config.num_agents, minimum=1)
    _require_integer("monitoring_radius", config.monitoring_radius, minimum=0)

    slip = _require_finite_number("action_slip_probability", config.action_slip_probability)
    if not 0.0 <= slip <= 1.0:
        raise ValueError("action_slip_probability must lie in [0, 1]")

    reward_normalizer = _require_finite_number("reward_normalizer", config.reward_normalizer)
    if reward_normalizer <= 0.0:
        raise ValueError("reward_normalizer must be positive")

    dual_upper_bound = _require_finite_number("dual_upper_bound", config.dual_upper_bound)
    if dual_upper_bound <= 0.0:
        raise ValueError("dual_upper_bound must be positive")
    if not isinstance(config.use_agent_id, bool):
        raise TypeError("use_agent_id must be a boolean")

    if len(config.initial_positions) != num_agents:
        raise ValueError(
            "initial_positions length must equal num_agents "
            f"({len(config.initial_positions)} != {num_agents})"
        )
    initial_positions = tuple(
        _validate_internal_position(f"initial_positions[{index}]", position, config)
        for index, position in enumerate(config.initial_positions)
    )
    if len(set(initial_positions)) != len(initial_positions):
        raise ValueError("initial robot positions must be pairwise distinct")

    for name, values in (
        ("target_centers", config.target_centers),
        ("target_values", config.target_values),
        ("hazard_intensities", config.hazard_intensities),
    ):
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping")
        _validate_exact_keys(name, values)

    for target_id in TARGET_IDS:
        _validate_internal_position(
            f"target_centers[{target_id}]", config.target_centers[target_id], config
        )

    # 该调用同时验证每个区域完全在地图内且两两不相交。
    regions = compute_monitoring_regions(
        config.target_centers,
        config.monitoring_radius,
        config.grid_height,
        config.grid_width,
    )
    expected_region_size = (2 * config.monitoring_radius + 1) ** 2
    if any(len(region) != expected_region_size for region in regions.values()):
        raise ValueError("every monitoring region must have its full Chebyshev area")

    target_values = []
    for target_id in TARGET_IDS:
        value = _require_finite_number(
            f"target_values[{target_id}]", config.target_values[target_id]
        )
        if value <= 0.0:
            raise ValueError(f"target_values[{target_id}] must be positive")
        target_values.append(value)

        intensity = _require_finite_number(
            f"hazard_intensities[{target_id}]",
            config.hazard_intensities[target_id],
        )
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"hazard_intensities[{target_id}] must lie in [0, 1]")

    for target_id in SAFE_TARGET_IDS:
        if float(config.hazard_intensities[target_id]) != 0.0:
            raise ValueError(f"safe target {target_id} must have zero hazard intensity")
    for target_id in HAZARDOUS_TARGET_IDS:
        if float(config.hazard_intensities[target_id]) <= 0.0:
            raise ValueError(f"hazardous target {target_id} must have positive intensity")

    # 四个机器人最多覆盖四个不同区域; W_max 必须保证 r_t <= 1。
    max_covered = min(num_agents, len(target_values))
    maximum_value = sum(sorted(target_values, reverse=True)[:max_covered])
    if maximum_value > reward_normalizer + 1e-12:
        raise ValueError(
            "reward_normalizer is smaller than the maximum simultaneously "
            f"coverable value ({reward_normalizer} < {maximum_value})"
        )


def _validate_raw_schema(raw: Mapping[str, Any]) -> None:
    missing = sorted(_TOP_LEVEL_FIELDS - set(raw))
    extra = sorted(set(raw) - _TOP_LEVEL_FIELDS)
    if missing or extra:
        raise ValueError(f"configuration fields mismatch; missing={missing}, extra={extra}")
    if raw["coords"] != PDF_COORDINATE_SYSTEM:
        raise ValueError(f"coords must be {PDF_COORDINATE_SYSTEM!r}; got {raw['coords']!r}")
    for table_name in ("target_centers", "target_values", "hazard_intensities"):
        table = raw[table_name]
        if not isinstance(table, Mapping):
            raise TypeError(f"{table_name} must be a TOML table")
        _validate_exact_keys(table_name, table)


def _parse_pdf_positions(name: str, positions: Any) -> tuple[Position, ...]:
    if isinstance(positions, (str, bytes)):
        raise TypeError(f"{name} must be an array of coordinates")
    try:
        return tuple(pdf_to_internal(position) for position in positions)
    except TypeError as exc:
        raise TypeError(f"{name} must be an array of coordinates") from exc


def load_config(path: str | Path | None = None) -> HRMRConfig:
    """从 TOML 加载 HRMR 配置, 并在边界处集中转换 PDF 坐标。"""

    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    if not isinstance(raw, Mapping):  # pragma: no cover - TOML parser 通常保证
        raise TypeError("configuration root must be a TOML table")
    _validate_raw_schema(raw)

    raw_centers = raw["target_centers"]
    config = HRMRConfig(
        grid_width=raw["grid_width"],
        grid_height=raw["grid_height"],
        num_agents=raw["num_agents"],
        monitoring_radius=raw["monitoring_radius"],
        action_slip_probability=raw["action_slip_probability"],
        initial_positions=_parse_pdf_positions("initial_positions", raw["initial_positions"]),
        target_centers={
            target_id: pdf_to_internal(raw_centers[target_id]) for target_id in TARGET_IDS
        },
        target_values={target_id: raw["target_values"][target_id] for target_id in TARGET_IDS},
        hazard_intensities={
            target_id: raw["hazard_intensities"][target_id] for target_id in TARGET_IDS
        },
        reward_normalizer=raw["reward_normalizer"],
        dual_upper_bound=raw["dual_upper_bound"],
        use_agent_id=raw["use_agent_id"],
    )
    validate_config(config)
    return config


__all__ = ["DEFAULT_CONFIG_PATH", "HRMRConfig", "load_config", "validate_config"]
