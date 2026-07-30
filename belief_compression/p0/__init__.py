"""Gate P0 — the diversity smoke test.

A cheap, pre-registered gate that must pass BEFORE the ~10-12 week / ~1000
A100-hour Gate C1 program is committed to.  It answers one question:

    Does the K-hypothesis / M-mode regime that Gate C0 measured on ORACLE
    particle filters exist at all for a LEARNED belief model over pixels?

Two opposite outcomes are both fatal and both cheap to detect:

  VACUOUS      the K learned hypotheses are near-identical, so M = 1.  The
               compression is trivially lossless and means nothing; there is
               nothing to plan over and no VOI to amortize.  (Gate C1 STOP S5.)
  NO-COMPRESS  the decision structure is rich relative to the filter, so M = K
               and the method buys nothing.                (Gate C1 STOP S1.)

plus one accounting outcome:

  FILTER-EATS-IT   the belief model's own forward cost is >= 80% of total
               compute, so saving planning compute is irrelevant end-to-end.
                                                            (Gate C1 STOP S3.)

Design decisions, all justified in `docs/gateP0_design.md`:

  * The measurement is the EXISTING `belief_compression.compression.compress`.
    `hypotheses.HypothesisTask` adapts a set of decoded hypotheses to the
    `core.Task` interface so `compress()` runs verbatim -- same signature
    extraction, same `rep_rule`s, same `ComputeCounter`.  Nothing is
    reimplemented in parallel.
  * The belief model is the cheapest thing that yields multiple hypotheses
    (`belief.FactoredBernoulliBelief`), and the SAME class serves as the
    offline synthetic stand-in and as the production head, so the unit tests
    exercise the production path (repo convention).
  * P0 measures M, M/K, hypothesis diversity and the belief/search cost split.
    It does NOT measure planning regret or closed-loop return -- that is P4.

Offline (no jax, no GPU, no trained model) everything in `hypotheses`,
`belief`, `decision` and `measure` runs on numpy alone; `envs` imports jax
lazily and is the only module that needs the substrate installed.
"""

__all__ = [
    "decision",
    "hypotheses",
    "belief",
    "measure",
    "envs",
]
