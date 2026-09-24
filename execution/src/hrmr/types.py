"""HRMR 公共类型别名。"""

from collections.abc import Mapping, Sequence

# 内置容器泛型与项目支持的 Python >=3.11 一致, 也可在本地 3.9 解析。
Position = tuple[int, int]
TargetId = str
MonitoringRegion = frozenset[Position]
MonitoringRegions = Mapping[TargetId, MonitoringRegion]
TargetPositions = Mapping[TargetId, Position]
TargetScalars = Mapping[TargetId, float]
JointPositions = Sequence[Position]

__all__ = [
    "JointPositions",
    "MonitoringRegion",
    "MonitoringRegions",
    "Position",
    "TargetId",
    "TargetPositions",
    "TargetScalars",
]
