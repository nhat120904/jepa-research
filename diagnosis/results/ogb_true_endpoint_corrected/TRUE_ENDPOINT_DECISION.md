# Corrected OGBench true-endpoint decision

Date: 2026-08-11. Locked verdict: **STRONG SUPPORT on this diagnostic**.

## Bottom line

The corrected audit recovers the substantive Stage-0 finding without using the
invalid historical rollouts. On the same 32 snapshots and same 300 final CEM
candidates per snapshot:

| Selector | Mean selected distance | Mean regret | Success |
|---|---:|---:|---:|
| learned predicted-endpoint latent L2 | 12.11 cm | 7.80 cm [4.86, 10.96] | 16/32 |
| true endpoint, dataset goal image | 7.63 cm | 3.32 cm [1.54, 5.70] | 21/32 |
| true endpoint, same-renderer goal | 7.60 cm | **3.29 cm [1.49, 5.67]** | **21/32** |
| physical best candidate | 4.31 cm | 0 | 25/32 available |

The primary same-renderer true-endpoint selector therefore leaves 3.29 cm of
physical selection regret after learned dynamics error is removed. Its paired
success gap to the physical best candidate is 12.5 percentage points (4/32),
95% CI [3.125, 25.0] points. Both precommitted representation criteria and the
candidate-coverage criterion have lower 95% bounds above zero.

Replacing predicted endpoints by true rendered endpoints removes 4.51 cm of
regret [1.84, 7.52] and adds 15.625 success points; the latter CI is [0, 31.25]
and therefore does not exclude zero. Learned-cost versus physical Spearman is
0.006 [−0.025, 0.037], while same-renderer true-endpoint cost versus physical
Spearman is 0.395 [0.236, 0.541]. Thus dynamics prediction error matters, but
the frozen encoder plus terminal squared-L2 cost still misorders physical
outcomes even with exact candidate dynamics.

## Why the corrected result is credible

Every snapshot was evaluated in two independently compiled `World`/`MjModel`
instances. Before every render or action sequence, the evaluator reset all
`mjData`, restored dataset state and goal, called `mj_forward`, aligned
finite-difference state, and disabled solver warm-start.

- 32/32 shards passed every locked gate.
- Worst repeat error across 9,600 candidates: zero for physical distance,
  success, executed steps, endpoint pixels, encoder costs, and selected index.
- Every physical array exactly matches the separately corrected PFCG evaluator.
- Camera/light/material/texture/geometry signatures match across independent
  worlds.
- Mean same-state renderer-domain ratio is `1.6568e-4`; worst snapshot is
  `6.5906e-4`, far below the precommitted 0.25 gate.
- Dataset-goal and same-renderer-goal arms have identical 21/32 success. Their
  mean regret differs by only 0.0315 cm, with paired CI [0, 0.0945] cm.

The dataset and renderer can have large local raw-pixel maxima, but the frozen
encoder's same-state discrepancy is tiny relative to the task separation. The
same-renderer goal arm additionally removes that cross-domain offset from the
primary representation diagnostic.

## Relationship to historical Stage 0

The historical candidate-level artifact remains invalid and must not be cited:
it called `CubeEnv.reset()` before candidates, allowing controller/contact
solver state from goal construction to leak into rollouts. The old 3.13 cm
estimate and its stored endpoints are not retroactively repaired.

The new independent audit nevertheless reproduces the cohort-level conclusion:
21/32 true-endpoint successes, 25/32 physical-oracle availability, and residual
regret 3.29 cm instead of the historical 3.13 cm. The defensible claim is now
based only on the corrected outputs in this directory.

## Claim boundary

This establishes a residual failure of the released `quentinll/lewm-cube`
encoder **together with terminal squared-L2 scoring** on one OGBench-Cube task
and checkpoint. It does not isolate the encoder from the cost form, show that
representation learning generally fails, show that scaling cannot fix the
problem, or make simulator-assisted selection a fair deployable method.

The result is strong mechanistic replication outside MetaWorld, not a top-tier
method contribution by itself. The PFCG mitigation remains a locked no-go; this
audit does not reopen it.

## Execution and artifacts

- hard-case smokes: jobs `38298` (snapshot 3), `38297` (6), `38296` (9), all
  completed and passed;
- locked array: `38299_[0-31]`, all 32 tasks completed with exit code 0;
- aggregate: `38300`, 10,000 snapshot-bootstrap draws, completed.

Output SHA256:

```text
0f2d00cd92b0fe27ccda92c42604f8b39ddfd34d8621347a77ab6f1bf17c7a50  summary.json
a9e6c733e0ddb448cd38629342fa15b7584b866f17d9e94fee509c55acf128ad  snapshot_metrics.csv
c02943e819d81503f07a4bdadf4b3f188d27de57849e1b6fb61802af5e69a939  candidate_costs.csv.gz
```

The locked protocol and complete job ledger are in
`../../docs/plans/2026-08-11-ogb-corrected-true-endpoint-design.md`.
