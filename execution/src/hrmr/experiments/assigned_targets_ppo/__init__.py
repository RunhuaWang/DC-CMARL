"""经典固定责任区 MAPPO 的 owner-aware 环境组件。"""

from .environment import (
    ASSIGNED_OBSERVATION_DIM,
    ASSIGNED_STATE_DIM,
    ASSIGNED_TARGET_FEATURE_SIZE,
    DEFAULT_ASSIGNMENTS,
    AssignedTargetsEnvironment,
    permute_assigned_target_blocks,
)

__all__ = [
    "ASSIGNED_OBSERVATION_DIM",
    "ASSIGNED_STATE_DIM",
    "ASSIGNED_TARGET_FEATURE_SIZE",
    "DEFAULT_ASSIGNMENTS",
    "AssignedTargetsEnvironment",
    "permute_assigned_target_blocks",
]
