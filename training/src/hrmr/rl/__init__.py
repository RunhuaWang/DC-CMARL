"""HRMR Phase 2 average-reward fixed-lambda 训练组件。"""

from hrmr.rl.analytic_modes import (
    ANALYTIC_MODES,
    DIAGNOSTIC_FIXED_LAMBDAS,
    FIXED_LAMBDAS,
    LARGE_DUAL_DIAGNOSTIC_LAMBDA,
    REWARD_ONLY_LAMBDA,
    TRAINABLE_FIXED_LAMBDAS,
)
from hrmr.rl.average_rate import AverageRateEstimator
from hrmr.rl.networks import CategoricalActor, DifferentialCritic, IndependentActors
from hrmr.rl.vector_env import SyncVectorHRMR

__all__ = [
    "ANALYTIC_MODES",
    "DIAGNOSTIC_FIXED_LAMBDAS",
    "FIXED_LAMBDAS",
    "LARGE_DUAL_DIAGNOSTIC_LAMBDA",
    "REWARD_ONLY_LAMBDA",
    "TRAINABLE_FIXED_LAMBDAS",
    "AverageRateEstimator",
    "CategoricalActor",
    "DifferentialCritic",
    "IndependentActors",
    "SyncVectorHRMR",
]
