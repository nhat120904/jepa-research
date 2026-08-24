# CROD H0 locked protocol

Locked before any CROD physical outcome is observed: 2026-08-18 UTC.

## Hypothesis

Cross-Representation Ordinal Disagreement (CROD) enriches physical mistakes of
the deployed LeWM planning proxy better than action-space diversity under the
same physical-query budget.

The native LeWM+CEM planner returns the exact action chunk `a0` and a final
300-candidate population. The auxiliary is an independently formed visual
representation: a frozen DINOv2-Small encoder with an action-only DINO-WM
predictor trained on the same OGBench-Cube demonstrations. It never proposes a
new action in H0; it only rescores the native planner-induced population.

For lower-is-better joint ranks, candidate `i` receives

`max(r_L(i)-r_L(a0), 0) * max(r_D(a0)-r_D(i), 0)`.

## Fresh matched-budget experiment

- 128 snapshots from episodes disjoint from Phase 0d, Phase 1a, and Phase
  1a-v2.
- Native CEM: 300 samples, 30 updates, top-30, horizon 5, action block 5.
- Every arm is charged one exact CEM-returned anchor plus eight alternatives.
  Overlapping alternatives are physically executed once and their deterministic
  same-state outcome is reused, but each arm retains the same logical 1+8 query
  budget.
- A corrective hit requires a native-rejected alternative that improves final
  physical goal distance by at least 2 cm relative to the anchor.
- Primary arm: `crod_directional`, the eight largest directional scores among
  native-rejected candidates. Zero-score candidates remain zero-score rather
  than being mislabeled as directional; their fraction is reported.
- Primary control: the previously strongest `action_diverse` selector over the
  full final population.
- Additional controls: support-matched rejected diversity, DINO-best among
  native-rejected candidates, prior native uncertainty, and support-matched
  random.
- Homogeneous LeWM disagreement is omitted because no matched independent LeWM
  seed checkpoint is available; native instability is reported instead and is
  not described as an ensemble.

The primary endpoint is per-state corrective hit rate. The preregistered gate
is a paired snapshot bootstrap: the 95% CI lower bound for CROD minus action
diversity must exceed zero, and the point contrast in best corrective physical
improvement must be positive. A pass authorizes only a simple matched-data BC
pilot, not a flow policy.

## Separate mechanism audit

The 32 Phase-0d final populations are already fully physics-labelled. They are
rescored with the auxiliary without new outcome queries. With a 2 cm tie band,
the audit reports `P(L wrong)`, `P(L wrong | L != D)`, joint error, and the
enrichment of corrective candidates under directional support. Snapshot is the
bootstrap cluster. These diagnostic populations are not reused for the fresh
sample-efficiency endpoint.
