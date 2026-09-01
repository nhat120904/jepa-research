"""Planner-operator ablations for latent world models.

The deployed CEM refits a Gaussian to the top-K candidates and executes the
resulting **mean** action sequence.  On OGBench-Cube we measured that this
executed mean is worse than the elites it was built from (0.159 m vs 0.128 m),
worse than the single best candidate (0.122 m), and worse than a typical random
candidate on 20 of 39 states -- so the aggregation step itself loses quality.

``BestCandidateCEMSolver`` isolates exactly that step: sampling, scoring, elite
selection and the refit trajectory are inherited unchanged from ``CEMSolver``,
and the ONLY difference is which action sequence is handed back for execution.
Nothing about the search is altered, so any outcome difference is attributable
to the aggregation operator alone.
"""

from .best_candidate_cem import BestCandidateCEMSolver

__all__ = ["BestCandidateCEMSolver"]
