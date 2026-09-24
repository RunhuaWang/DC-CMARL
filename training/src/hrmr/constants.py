"""HRMR 环境中跨模块共享的符号常量。"""

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType

from hrmr.types import Position


class Action(IntEnum):
    """PDF 第 4.1 节定义的五个离散动作。"""

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


# 内部坐标是 (row, column), row 向北增大。
ACTION_DELTAS: Mapping[Action, Position] = MappingProxyType(
    {
        Action.STAY: (0, 0),
        Action.UP: (1, 0),
        Action.DOWN: (-1, 0),
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1),
    }
)

NUM_ACTIONS = len(Action)

# 此顺序是 PDF 式 (12.9) 中 occupancy vector 的规范顺序。
SAFE_TARGET_IDS = ("S1", "S2", "S3", "S4")
HAZARDOUS_TARGET_IDS = ("H1", "H2", "H3", "H4")
TARGET_IDS = SAFE_TARGET_IDS + HAZARDOUS_TARGET_IDS

PDF_COORDINATE_SYSTEM = "pdf_xy_1_based"

__all__ = [
    "ACTION_DELTAS",
    "HAZARDOUS_TARGET_IDS",
    "NUM_ACTIONS",
    "PDF_COORDINATE_SYSTEM",
    "SAFE_TARGET_IDS",
    "TARGET_IDS",
    "Action",
]
