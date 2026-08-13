# PFCG same-candidate pilot decision

Date: 2026-08-11. Locked verdict: **NO-GO**.

## What was tested

Probe-Frozen Controllability Geometry (PFCG) estimates a finite-horizon latent
response Gramian from 32 paired `+u/-u` action probes rolled through the frozen
released `quentinll/lewm-cube` model. It ranks a candidate endpoint by an
inverse-eigenvalue-weighted terminal residual in the retained response
subspace. No model weights are trained and no simulator value enters a selector.

The pilot reranked exactly the persisted 300 final CEM candidates at each of the
32 locked Stage-0 snapshots. All deployable selector indices were fixed before
MuJoCo evaluation. The final evaluator returned physical distance and success
only, reset full `mjData` before every candidate, disabled solver warm-start,
and repeated every population twice. Simulator rendering and true-endpoint
encoding were excluded after smoke tests showed they were not reproducible.

## Locked result

| Selector | Mean selected distance | Mean regret | Success |
|---|---:|---:|---:|
| latent L2 | 12.11 cm | 7.80 cm | 16/32 |
| PFCG | 12.57 cm | 8.26 cm | 14/32 |
| same-subspace projection | 11.24 cm | 6.92 cm | 17/32 |
| diagonal response whitening | 12.32 cm | 8.00 cm | 16/32 |
| matched-rank random subspace | 10.40 cm | 6.09 cm | 18/32 |

Paired PFCG gains are defined as comparator regret minus PFCG regret and PFCG
success minus comparator success:

| Contrast | Regret gain, mean [95% CI] | Success gain, mean [95% CI] |
|---|---:|---:|
| PFCG vs latent L2 | -0.46 cm [-2.11, 0.84] | -6.25 pp [-15.63, 0] |
| PFCG vs projection | -1.33 cm [-3.15, -0.005] | -9.38 pp [-21.88, 0] |
| PFCG vs random subspace | -2.16 cm [-4.56, -0.28] | -12.50 pp [-25.0, -3.13] |

Every precommitted go criterion is false. PFCG has higher mean regret and lower
success than latent L2, while both the unweighted projection and random-subspace
controls have lower mean regret. The locked verdict is therefore **NO-GO**.

## Interpretation

All probe geometries had rank 31, the maximum possible after centering 32
responses, and retained essentially 100% of measured response energy. The
failure is therefore not an accidental empty or rank-collapsed geometry.

The evidence rejects the proposed inverse-controllability weighting on this
pilot. Weak response directions are likely being amplified without evidence
that they are task-relevant. The unweighted projection's better mean and the
matched-random control's still better result show that low-rank reranking can
change choices, but do not identify the model's controllability geometry as the
cause. The random control is a negative control on only 32 snapshots, not a new
method result.

Per the locked rule, do not sweep probe count, rank, ridge, or eigenvalue floor
on these snapshots. Do not run n=128 or closed-loop PFCG search. PFCG should not
be developed as the top-tier method candidate.

## Evaluator finding and Stage-0 boundary

The pilot uncovered a separate measurement defect in the historical Stage-0
artifact. `CubeEnv.initialize_episode` executes random actions and restores only
qpos/qvel, leaving controller/contact-solver state outside the persisted
snapshot. Direct replay of stored candidates showed per-snapshot worst
candidate discrepancies as large as 25.8 cm, 12 success labels, and 15 executed
steps. Independent endpoint rendering also differed by as much as 38 pixel
levels.

The corrected physical evaluator is exactly repeatable across both repeated
populations for all 32 snapshots. Aggregate learned-L2 success (16/32), oracle
availability (25/32), and selected distance (12.11 cm versus the historical
12.09 cm) remain essentially unchanged. Thus the broad same-population
selection-failure/coverage result survives. At the time of this pilot, the
historical true-endpoint decomposition had not been reproduced with a controlled
renderer and was blocked from causal use. A subsequent fresh two-world audit
completed that validation: 32/32 shards pass exact renderer/physics gates and
the same-renderer true-endpoint selector leaves 3.29 cm regret [1.49, 5.67] with
21/32 successes. See
`../ogb_true_endpoint_corrected/TRUE_ENDPOINT_DECISION.md`. This later result
does not change the locked PFCG no-go.

## Integrity

- Locked array `38185_[0-31]`: 32/32 tasks completed.
- Aggregate `38223`: completed with 10,000 snapshot-bootstrap draws.
- Repeated-replay worst error: zero for physical distance, success, and steps.
- Learned-cost reproduction worst absolute error: `2.59e-5`.
- Geometry rank: 31/192 for all snapshots.

Output SHA256:

```text
e5090fcc3ee5ca1c45d926e6c0f9b8591682be4b61532787b65def71aca64572  summary.json
7f6ab0a2a5efed2948bceffd98038caec61ab81489b514b0208863455106db99  snapshot_metrics.csv
caa89d249479e2bb1f94131704dd1332bdbcb9e2b63a9217c0c6485417f6b80f  candidate_costs.csv.gz
```
