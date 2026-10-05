# CTA world-model manuscript — revision 2026-10-03

Title: **Predict Once, Query Many: World Models that Forecast Conditional Trajectory Abstractions**.

`main.tex` and `supplement.tex` use the existing official CVPR 2026 author kit with `review` enabled, anonymous authors, line numbers, and the confidential review header. The displayed conference year is 2027 and the paper ID remains unassigned. This is a research draft, not a submitted manuscript; the official kit and rules for the eventual target year must be checked before submission.

The main paper centers on the learned future target and action-conditioned world model. It contains four figures: an archived PushT inference/training example, the detailed two-stage architecture, learned-method ranking with root-bootstrap intervals, and candidate-quality/component-cost scaling. Main performance comparisons include only deployable methods. Actual-future and privileged-selector diagnostics are documented in the supplement.

## Build

On a local workstation with TeX Live/MacTeX:

```sh
bash build.sh
```

On H100, submit document compilation to a compute node:

```sh
sbatch build.sh
```

Record the job ID and inspect both `squeue` and `sacct`. This revision was compiled and visually checked on the local Mac; no cluster job was submitted. Compilation is reproducible from the bundled source, style, bibliography and figure assets.

## Evidence and limits

`fig/evidence_data.json` records plotted estimates and their archived sources. `EVIDENCE_PROVENANCE.json` records 14 small archived JSON hashes. `REVISION_MANIFEST.json` records final source/PDF hashes and layout validation. `H100_SYNC_VERIFICATION.json` records the verified server copy after synchronization.

The main results use reused development roots and one training seed. Historical control uses the old batched sampler. Candidate expansion is a separate 100-root logged-bank experiment, not a new held-out test. No new model experiment, sealed multi-seed control, or total-planner Pareto result is claimed. The sixteen queries are views of one target pose. Actual-future diagnostics identify discrepancies rather than supplying deployable performance or an upper bound on closed-loop success.

## Revision history

`draft_v4_before_20261002_review/` preserves the original active manuscript. `draft_v5_before_20261002_user_feedback/` preserves the rejected action-ranking rewrite. Both snapshots contain `SHA256.json`. The earlier releases and snapshots remain intact. The correction report is at `../docs/CTA_PAPER_REVISION_20261003_VI.md`.
