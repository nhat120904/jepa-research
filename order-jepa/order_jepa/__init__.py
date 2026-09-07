"""ORDER-JEPA diagnostics and training objectives."""

from .core import (
    angular_distance,
    latent_cost,
    order_vector_metrics,
    pusht_physical_cost,
    selection_regret,
)
from .loss import order_effect_loss, paired_error_decomposition

__all__ = [
    "angular_distance",
    "latent_cost",
    "order_effect_loss",
    "order_vector_metrics",
    "paired_error_decomposition",
    "pusht_physical_cost",
    "selection_regret",
]
