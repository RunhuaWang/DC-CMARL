"""固定责任区 MAPPO 复用的 raw signal 与 rollout 数据结构。"""

from .rollout_buffer import UnnormalizedRolloutBatch, UnnormalizedRolloutBuffer
from .signals import (
    aggregate_unnormalized_global_cost,
    compute_unnormalized_local_costs,
    compute_unnormalized_team_reward,
)

__all__ = [
    "UnnormalizedRolloutBatch",
    "UnnormalizedRolloutBuffer",
    "aggregate_unnormalized_global_cost",
    "compute_unnormalized_local_costs",
    "compute_unnormalized_team_reward",
]
