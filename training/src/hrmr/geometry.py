"""HRMR 地图几何、坐标转换与目标区域查询。

PDF 中坐标为 ``(x, y)``、1-based; 包内部坐标为
``(row, column)``、0-based, 且 row 向北递增。两种坐标的转换只在
:func:`pdf_to_internal` 和 :func:`internal_to_pdf` 中进行。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real

import numpy as np

from hrmr.constants import TARGET_IDS
from hrmr.types import MonitoringRegions, Position, TargetId, TargetPositions, TargetScalars


def _as_position(position: Sequence[int], *, name: str = "position") -> Position:
    """验证二维整数坐标, 并转成内置 ``int`` 二元组。"""

    if isinstance(position, (str, bytes)):
        raise TypeError(f"{name} must be a two-element integer coordinate")
    try:
        values = tuple(position)
    except TypeError as exc:
        raise TypeError(f"{name} must be a two-element integer coordinate") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two components")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise TypeError(f"{name} components must be integers")
    return int(values[0]), int(values[1])


def pdf_to_internal(position: Position) -> Position:
    """PDF ``(x, y)`` 1-based 坐标转为内部 ``(row, column)``。"""

    x, y = _as_position(position, name="PDF position")
    if x < 1 or y < 1:
        raise ValueError("PDF coordinates are 1-based and must be at least 1")
    return y - 1, x - 1


def internal_to_pdf(position: Position) -> Position:
    """内部 ``(row, column)`` 0-based 坐标转为 PDF ``(x, y)``。"""

    row, column = _as_position(position, name="internal position")
    if row < 0 or column < 0:
        raise ValueError("internal coordinates are 0-based and must be non-negative")
    return column + 1, row + 1


def position_in_grid(
    position: Position,
    grid_height: int,
    grid_width: int,
) -> bool:
    """返回内部坐标是否位于给定网格中。"""

    row, column = _as_position(position)
    _validate_grid_dimensions(grid_height, grid_width)
    return 0 <= row < grid_height and 0 <= column < grid_width


def chebyshev_distance(first: Position, second: Position) -> int:
    """计算两个内部坐标的 Chebyshev 距离。"""

    first_row, first_column = _as_position(first, name="first position")
    second_row, second_column = _as_position(second, name="second position")
    return max(abs(first_row - second_row), abs(first_column - second_column))


def _validate_grid_dimensions(grid_height: int, grid_width: int) -> None:
    for name, value in (("grid_height", grid_height), ("grid_width", grid_width)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def compute_monitoring_regions(
    centers: TargetPositions,
    radius: int,
    grid_height: int,
    grid_width: int,
) -> dict[TargetId, frozenset[Position]]:
    """构造所有 Chebyshev 监控区域并验证边界和两两不相交。

    ``centers`` 必须已使用内部坐标。一个中心的完整
    ``(2 * radius + 1) ** 2`` 区域若越界, 本函数会拒绝配置而不是裁剪区域。
    """

    _validate_grid_dimensions(grid_height, grid_width)
    if isinstance(radius, bool) or not isinstance(radius, Integral):
        raise TypeError("radius must be an integer")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if not isinstance(centers, Mapping):
        raise TypeError("centers must be a mapping from target id to position")

    regions: dict[TargetId, frozenset[Position]] = {}
    occupied_cells: dict[Position, TargetId] = {}
    expected_size = (2 * int(radius) + 1) ** 2

    for target_id, raw_center in centers.items():
        if not isinstance(target_id, str) or not target_id:
            raise TypeError("target ids must be non-empty strings")
        center = _as_position(raw_center, name=f"center of {target_id}")
        if not position_in_grid(center, int(grid_height), int(grid_width)):
            raise ValueError(f"center of {target_id} lies outside the grid: {center}")

        row, column = center
        cells = frozenset(
            (region_row, region_column)
            for region_row in range(row - int(radius), row + int(radius) + 1)
            for region_column in range(column - int(radius), column + int(radius) + 1)
        )
        if len(cells) != expected_size or any(
            not position_in_grid(cell, int(grid_height), int(grid_width)) for cell in cells
        ):
            raise ValueError(f"monitoring region {target_id} is not fully contained in the grid")

        for cell in cells:
            other_target = occupied_cells.get(cell)
            if other_target is not None:
                raise ValueError(
                    f"monitoring regions {other_target} and {target_id} overlap at {cell}"
                )
            occupied_cells[cell] = target_id
        regions[target_id] = cells

    return regions


def _ordered_target_ids(
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId],
) -> tuple[TargetId, ...]:
    ids = tuple(target_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("target_ids must not contain duplicates")
    missing = [target_id for target_id in ids if target_id not in regions]
    if missing:
        raise KeyError(f"regions are missing target ids: {missing}")
    return ids


def _validated_positions(positions: Iterable[Position]) -> tuple[Position, ...]:
    if isinstance(positions, (str, bytes)):
        raise TypeError("positions must be an iterable of two-dimensional coordinates")
    try:
        return tuple(
            _as_position(position, name=f"positions[{index}]")
            for index, position in enumerate(positions)
        )
    except TypeError as exc:
        if "positions[" in str(exc):
            raise
        raise TypeError("positions must be an iterable of two-dimensional coordinates") from exc


def compute_target_coverage(
    positions: Iterable[Position],
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> np.ndarray:
    """计算 PDF 式 (5.1) 的区域 coverage indicator。

    返回数组顺序由 ``target_ids`` 决定; 默认严格为
    ``S1,S2,S3,S4,H1,H2,H3,H4``。多个机器人覆盖同一区域仍只记 1。
    """

    if not isinstance(regions, Mapping):
        raise TypeError("regions must be a mapping")
    ordered_ids = _ordered_target_ids(regions, target_ids)
    position_set = set(_validated_positions(positions))
    return np.asarray(
        [bool(position_set.intersection(regions[target_id])) for target_id in ordered_ids],
        dtype=np.int8,
    )


# PDF 和用户验收清单采用 occupancy 这一名称; 两者定义相同。
compute_target_occupancy = compute_target_coverage


def region_id_at(
    position: Position,
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> TargetId | None:
    """返回位置所在区域 ID; 区域外返回 ``None``。"""

    checked_position = _as_position(position)
    ordered_ids = _ordered_target_ids(regions, target_ids)
    matches = [target_id for target_id in ordered_ids if checked_position in regions[target_id]]
    if len(matches) > 1:
        raise ValueError(f"position {checked_position} belongs to overlapping regions {matches}")
    return matches[0] if matches else None


def agent_region_ids(
    positions: Iterable[Position],
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> tuple[TargetId | None, ...]:
    """按机器人顺序返回每个位置所在的区域 ID。"""

    return tuple(
        region_id_at(position, regions, target_ids) for position in _validated_positions(positions)
    )


def covered_target_ids(
    positions: Iterable[Position],
    regions: MonitoringRegions,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> tuple[TargetId, ...]:
    """按规范顺序返回已覆盖的目标 ID。"""

    ordered_ids = tuple(target_ids)
    coverage = compute_target_coverage(positions, regions, ordered_ids)
    return tuple(
        target_id
        for target_id, is_covered in zip(ordered_ids, coverage)  # noqa: B905
        if bool(is_covered)
    )


def hazard_intensity_at(
    position: Position,
    regions: MonitoringRegions,
    hazard_intensities: TargetScalars,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> float:
    """实现 PDF 式 (6.2) 的空间危险强度函数 ``h(x)``。"""

    target_id = region_id_at(position, regions, target_ids)
    if target_id is None:
        return 0.0
    raw_intensity = hazard_intensities.get(target_id, 0.0)
    if isinstance(raw_intensity, bool) or not isinstance(raw_intensity, Real):
        raise TypeError(f"hazard intensity for {target_id} must be numeric")
    intensity = float(raw_intensity)
    if not np.isfinite(intensity) or not 0.0 <= intensity <= 1.0:
        raise ValueError(f"hazard intensity for {target_id} must lie in [0, 1]")
    return intensity


def compute_hazard_intensities(
    positions: Iterable[Position],
    regions: MonitoringRegions,
    hazard_intensities: TargetScalars,
    target_ids: Sequence[TargetId] = TARGET_IDS,
) -> np.ndarray:
    """按机器人顺序返回 ``h(x_i)``, dtype 固定为 ``float64``。"""

    return np.asarray(
        [
            hazard_intensity_at(position, regions, hazard_intensities, target_ids)
            for position in _validated_positions(positions)
        ],
        dtype=np.float64,
    )


__all__ = [
    "agent_region_ids",
    "chebyshev_distance",
    "compute_hazard_intensities",
    "compute_monitoring_regions",
    "compute_target_coverage",
    "compute_target_occupancy",
    "covered_target_ids",
    "hazard_intensity_at",
    "internal_to_pdf",
    "pdf_to_internal",
    "position_in_grid",
    "region_id_at",
]
