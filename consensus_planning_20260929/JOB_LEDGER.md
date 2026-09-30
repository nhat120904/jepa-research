# Job ledger: consensus_planning_20260929

Probe: M=10 independent CEM runs per plan; execute a consensus plan (action
medoid, mean, lowest-cost, predicted-endpoint medoid) of 3-iteration runs at the
compute of one 30-iteration run, versus single-seed CEM-30/CEM-3 (10 seeds each,
giving seed variance). Development roots 0-99 of the cem_stopping manifest
(Cube, Reacher); test roots untouched. Harness imported from
../cem_stopping_20260929 (exact restore, verified equivalence with upstream CEM).
Outputs /mnt/data/nhatnc129/jepa/consensus_planning/.

## 2026-09-29
Quota before: GPU 208/243 h, CPU 1781/1945 h; queue empty.

| Job | Work | Resources | State |
|---|---|---|---|
| 55963_[0-1] | Smoke, roots 0-1 per task | mig, 4 CPU, 32 GB, 20 min | COMPLETED (Cube ~40 s/root, Reacher ~18 s) |
| 55965 | Full dev run, roots 0-99, 4 shards/task (idx 0-3 Cube, 4-7 Reacher) | mig, 4 CPU, 32 GB, 35 min, %2 | COMPLETED (15 min Cube shards, 7 min Reacher) |
| 55966 | Analysis, afterok | main, 2 CPU, 8 GB, 15 min | COMPLETED |

## Result (analysis_55966.json, 100/100 dev roots per task) -- pre-registered kill criterion MET

Kill rule fixed before reading: on Reacher, medoid3 must beat both random-run
(single3 expected) and lowest-cost selection with a paired CI excluding zero.

| | Cube | Reacher |
|---|---|---|
| single30, expected over 10 seeds | .680 | .667 |
| single3, expected (= random run) | .657 | .613 |
| medoid3 (10x3, same compute as 1x30) | .63 (vs s3 -2.7 [-6.6,+0.5]) | .59 (vs s3 -2.3 [-9.5,+4.7]) |
| bestcost3 (DecentCEM-like selection) | .65 | .61 |
| latmedoid3 | .64 | .59 |
| avg3 | .57 | .52 |
| medoid30 (10x compute) | .68 (+0.0 vs s30) | .73 (+6.3 [-1.6,+14.1] vs s30) |

Plan-1 open loop: action medoid of the 10 three-iteration plans is no closer to
the goal than the seed-average plan (Cube +.003 m, Reacher -.003 rad, both CI
include 0). Cross-seed consensus does not identify physically better plans at
equal compute. World-model cost vs outcome over seeds: Spearman .04 Cube, .12 Reacher.

Evaluation-noise by-product (useful regardless): with the released planner
(CEM-30), success on the SAME 50 roots ranges .62-.72 (Cube) and .60-.76
(Reacher) across 10 CEM seeds; a 50-episode evaluation has SD 5.5 / 6.1 pp over
seeds and root draws. Best-of-10-seeds per root: Cube .83, Reacher .99.
Decision: consensus-planning direction closed; no follow-up arms (wide CEM,
DecentCEM split, basin consensus) because the same-compute core effect is absent.
