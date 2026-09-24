"""6×6 固定责任区 dual-conditioned MAPPO 的独立实现命名空间。"""

from .advantages import (
    ConditionedFrozenQuantities,
    conditioned_restart_quantities,
    mixed_conditioned_restart_quantities,
)
from .conditioning import (
    DUAL_ACTOR_INPUT_DIM,
    DUAL_CRITIC_INPUT_DIM,
    condition_actor_observations,
    condition_critic_states,
    normalize_conditioning_lambda,
    normalize_conditioning_lambdas,
    permute_conditioned_target_blocks,
)
from .minibatches import stratified_dual_minibatches
from .networks import (
    CONCATENATED_CRITIC_CONDITIONING,
    FILM_ACTOR_CONDITIONING,
    FiLMCategoricalActor,
    FiLMIndependentActors,
    make_dual_conditioned_networks,
)
from .pcgrad import PCGradProjection, project_conflicting_gradients
from .policy_retention import (
    ADAPTIVE_BEST_CALIBRATION_ANCHOR_KL,
    PerLambdaPolicyRetention,
)
from .sampling import (
    collect_dual_conditioned_restart_rollout,
    collect_dual_conditioned_rollout,
)
from .schedule import ConditionalAverageRateBank, PersistentDualBatchSchedule

__all__ = [
    "ADAPTIVE_BEST_CALIBRATION_ANCHOR_KL",
    "CONCATENATED_CRITIC_CONDITIONING",
    "DUAL_ACTOR_INPUT_DIM",
    "DUAL_CRITIC_INPUT_DIM",
    "FILM_ACTOR_CONDITIONING",
    "ConditionalAverageRateBank",
    "ConditionedFrozenQuantities",
    "FiLMCategoricalActor",
    "FiLMIndependentActors",
    "PCGradProjection",
    "PerLambdaPolicyRetention",
    "PersistentDualBatchSchedule",
    "collect_dual_conditioned_restart_rollout",
    "collect_dual_conditioned_rollout",
    "condition_actor_observations",
    "condition_critic_states",
    "conditioned_restart_quantities",
    "make_dual_conditioned_networks",
    "mixed_conditioned_restart_quantities",
    "normalize_conditioning_lambda",
    "normalize_conditioning_lambdas",
    "permute_conditioned_target_blocks",
    "project_conflicting_gradients",
    "stratified_dual_minibatches",
]
