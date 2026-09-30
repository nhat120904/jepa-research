# Job ledger

## 2026-09-28 implementation

Source: `pipeline.py` and `slurm_pipeline.sh`. The independent Cube
experiment uses the previously corrected `diagnosis/scripts/76_ogb_true_endpoint_corrected.py`
restore, CEM candidate recorder from `72_ogb_stage0_candidate_audit.py`,
and the released `quentinll/lewm-cube` checkpoint. The external AD-WM project
URL timed out from the login node and no callable code/checkpoint was found;
its reported numbers will be treated as an unmatched external reference.
AD-WM is a preprint, not an accepted-paper performance target.

No Cube action-response result is claimed before closed-loop jobs complete.

Python compilation and shell syntax checks passed. The released Cube
checkpoint and dataset paths exist. `squeue` found unrelated jobs 55409/55411
before submission and no duplicate action-response Cube work. Job 55412 was
submitted as a technical end-to-end smoke (`COLLECT=2`, 2 scorer updates,
2 dynamics updates per arm, one common evaluation root), with H100 3g.40gb
MIG, 8 CPUs, 96 GB, and 3 h limit. Unique output:
`/mnt/data/nhatnc129/jepa/action_response_cube/run_55412_20260928T023155Z/`.
Both `squeue` and `sacct` showed RUNNING at first verification. Its results
are pending.

Job 55412 completed `0:0` in 37 seconds (`squeue` empty; `sacct`
COMPLETED). Exact branch replay passed and all four evaluation arms ran on
one root. The one-root outcomes are a technical check only. The scorer had
only two training roots and is not interpretable. Before the full run, source
was amended to record held-out true-endpoint scorer versus native-L2
selection regret and selected success.

Python and shell syntax checks passed again. `squeue` showed only unrelated
job 55411 running and no duplicate Cube action-response job. Full job 55413
was submitted: 160 distinct training episodes, 8 selected CEM candidates
each, 300 scorer updates, 200 predictor updates/arm, 50 common test roots,
four evaluation arms. It has H100 3g.40gb MIG, 8 CPUs, 96 GB, 3 h limit.
Unique output: `/mnt/data/nhatnc129/jepa/action_response_cube/run_55413_20260928T023327Z/`.
Both `squeue` and `sacct` showed RUNNING at first verification. No result
is claimed yet.

Job 55413 completed `0:0` in 11:37 (`squeue` empty, `sacct` COMPLETED).
All 160 exact-state CEM branches and all 50 roots per arm completed. The
held-out scorer on *true* endpoints had 6.79 mm selection regret vs 6.77 mm
for native latent-L2; both selected success on 75% of the 32 dev banks.
Closed-loop success: original dynamics + scorer 29/50, prediction-only +
native latent-L2 **36/50**, prediction-only + scorer 26/50, ranking loss +
scorer 28/50. Mean final cube distances were 9.43, 6.42, 9.92, 9.99 cm
respectively. CPU paired-analysis job 55422 completed `0:0` and found
ranking+scorer vs prediction-only+scorer +4 pp, bootstrap 95% interval
[-4, 12], while ranking+scorer vs prediction-only+native is -16 pp
[-28, -5.95]. Among the 50 original roots, 20–22 successes per arm occurred
within three steps, so much of this original protocol is near solved.
The scorer is the observed bottleneck; ranking supervision does not rescue
the full system. Exact result: `run_55413_20260928T023327Z/paired_analysis.json`.

An independent non-contact hard-start evaluator was implemented in
`eval_hard_start.py`: it excludes all 160 training episodes, enforces initial
cube-goal distance ≥5 cm and hand-cube separation ≥2 cm, applies a 2 cm XY
cube perturbation, and reuses complete MuJoCo restoration. It evaluates the
same learned checkpoints with five matched arms, including original/native.
This is inspired by AD-WM's hard-start idea but is not asserted to be its
exact published protocol. Syntax checks passed. Technical smoke job 55423
was submitted with two roots, 1 H100 MIG 3g.40gb, 8 CPU, 96 GB, 1:30 h.
Before submission, `squeue` showed unrelated 55415/55417 running and
55416/55418 pending, no duplicate Cube action-response work. Both `squeue`
and `sacct` showed 55423 PENDING due to QOSMaxGRESPerUser at first check.

Because the learned scorer reduced closed-loop success and gave no true-endpoint
selection advantage, a focused native-cost continuation was implemented in
`native_followup.py`. It reuses the exact saved 160×8 branch cache, leaves
LeWM's native latent-L2 CEM objective in place, and trains the predictor with
endpoint prediction plus a within-bank normalized ranking loss based on
realized physical distance. This is still task-supervised simulator-assisted
training. The matched prediction-only native arm is read from 55413's same
50 roots; the new arm executes them with the identical reset/planning code.
Python/shell syntax checks passed. Technical smoke 55424 was submitted
(2 updates, one root, H100 MIG 3g.40gb, 8 CPU, 96 GB, 1:30 h). `squeue` and
`sacct` both showed PENDING under the user GPU QOS while unrelated 55415/55417
were running. This is a separate method from the scorer-based 55423 smoke,
not a duplicate submission. No native-rank result is claimed yet.

Focused CPU bank diagnostic 55426 completed `0:0`. Of 160 distinct
training banks, 127 had at least one successful candidate among eight, while
the released CEM rank-0 candidate already succeeded in 113. Only 36.9% of
rank-0 candidates were physically best, but median rank-0 regret was just
2.59 mm and median best-to-worst spread only 1.11 cm. Thus the final CEM
neighborhood has narrow physical separation despite a measurable ranking
gap. These are training-bank diagnostics, not held-out control results.
Output: `run_55413_20260928T023327Z/bank_diagnostics.json`.

In response to the narrow-bank finding, `pipeline.py` now supports a
`mixed` candidate population: four actions from the initial CEM solve
(ranks 0/10/30/90) and four from the final solve (ranks 0/3/10/29), all
branched from the same exact state. The existing `final` behavior remains
the default and the completed 55413 source snapshot is unchanged. Python
and shell syntax checks passed. A full mixed-population run has not been
submitted while unrelated OGB jobs occupy the GPU quota and the hard/native
technical smokes are pending.

The following bounded continuations are submitted with Slurm `afterok`
dependencies so no GPU is occupied while waiting:

| Job | Dependency | Purpose | Explicit resources / limit |
| --- | --- | --- | --- |
| 55428 | afterok:55423 | full 50-root non-contact hard-start on saved 55413 checkpoints | H100 MIG 3g.40gb, 8 CPU, 96 GB, 1:30 h |
| 55429 | afterok:55424 | 200-update native-cost ranking continuation + 50 original roots | same |
| 55430 | afterok:55428 | paired hard-start report | 2 CPU, 4 GB, 10 min |
| 55431 | afterok:55429 | paired native-cost report | 2 CPU, 4 GB, 10 min |

Before submissions, Python and shell syntax checks passed and `squeue`
showed no duplicate Cube action-response work beyond each intended smoke.
`squeue` and `sacct` showed the full/report jobs PENDING on dependency at
submission. The smoke jobs remain pending because unrelated 55415/55417
occupy the user GPU quota. These jobs have not produced a result yet.

## 2026-09-28 method focus correction

The learned scorer/ranking continuation would move away from the user's
direct action-response method. Jobs 55423, 55424, 55428, 55429, 55430, and
55431 were therefore cancelled **before starting**. Both `squeue` (empty for
these IDs) and `sacct` (CANCELLED, elapsed 00:00:00 for all six) confirmed
cancellation. The preceding table is historical submission information.

New source: `response_followup.py` and `slurm_response_followup.sh`.
The saved exact-state branch cache from 55413 is reused read-only. Loss is
endpoint prediction MSE plus weight 1.0 times the mean squared error in
predicted versus observed pairwise endpoint changes within each eight-action
bank (computed as twice centered residual variance). No physical-distance
labels or goal scorer enter training. The released LeWM GoalMSE planning cost,
300-sample/30-iteration CEM, and 50-root manifest remain unchanged.
Prediction-only+native-cost from 55413 (36/50) is the matched reference;
this is a single-checkpoint development comparison, not a published-baseline
claim. Syntax checks passed. At submission, unrelated OGB jobs 55415/55417
were RUNNING and 55416/55418 PENDING; `squeue` and `sacct` agreed. No duplicate
Cube action-response job was active.

Submitted bounded end-to-end continuation:

| Job | Purpose | Resources / time | Initial state verified by both `squeue` and `sacct` |
| --- | --- | --- | --- |
| 55433 | Two-update, one-root technical smoke | MIG H100 3g.40gb, 8 CPU, 96 GB, 1:30 h | PENDING, GPU QOS |
| 55434 | 200-update response training and 50 paired original roots, `afterok:55433` | same | PENDING, dependency |
| 55435 | CPU paired bootstrap and root report, `afterok:55434` | main, 2 CPU, 4 GB, 10 min | PENDING, dependency |

Each GPU job writes a distinct
`/mnt/data/nhatnc129/jepa/action_response_cube/response_<jobid>_<timestamp>/`
directory and records its path as `job_<jobid>.path`; the Slurm script copies
source and SHA-256 sums before training. The CPU report will write
`paired_analysis.json` into the full-run directory. These jobs have not
started, so there is **no action-response Cube control result yet**.

Status check at 2026-09-28 06:06 UTC: `squeue` and `sacct` both show 55433
PENDING (QOSMaxGRESPerUser) and 55434/55435 PENDING (dependencies), all with
00:00:00 elapsed. No run path or log exists yet. The user's other OGB eval
jobs 55416/55418 are running on the MIG worker; their parent train jobs
55415/55417 completed successfully. No Cube result had been produced at this check.

## 2026-09-28 06:57 UTC: direct response result and focused iteration

Both `squeue` (empty) and `sacct` (COMPLETED 0:0) confirm the response chain:
55433 22 s, 55434 1:44, 55435 1 s. Run directory:
`/mnt/data/nhatnc129/jepa/action_response_cube/response_55434_20260928T064139Z/`.
On 50 paired original-protocol roots with native LeWM cost, direct
same-state response training succeeded **35/50** versus prediction-only
**36/50**. Paired gain -2 pp, bootstrap 95% interval [-8,+4]; one root was
won only by response and two only by prediction. Mean final cube distance
was 7.05 cm versus 6.42 cm; response-minus-control distance improvement
-0.64 cm, interval [-2.49,+1.01] cm. Dev endpoint prediction loss fell
0.887→0.693, but action-response loss moved only 0.2631→0.2559 over 200
updates. Thus this run does not show improved control. Full evidence:
`paired_analysis.json`, `train.json`, and per-root `evaluation.json` in the
run directory. This single seeded fine-tune is not an uncertainty assessment
over independent model training runs.

The 55413 training banks used eight **final** CEM candidates; prior bank
diagnostics found median best-to-worst physical spread only 1.11 cm. A
targeted method iteration, `response_mixed.py`, collects four initial plus
four final CEM candidates from the same exact state. It records endpoint
response signal and physical spread as diagnostics, then trains both
prediction-only and response arms on this same mixed cache with identical
initialization, batches, dropout RNG, optimizer, CEM and 50 held-out roots.
Physical distance labels are never used for training. Python and shell
syntax checks passed. Before submission, `squeue` showed only unrelated OGB
eval 55418 running and no duplicate Cube work; `sacct` confirmed the previous
Cube chain complete.

| Job | Purpose | Resources / time | Initial state by both `squeue` and `sacct` |
| --- | --- | --- | --- |
| 55493 | Mixed-branch technical smoke, 2 banks/2 updates/1 root per arm | H100 MIG 3g.40gb, 8 CPU, 96 GB, 1:30 h | RUNNING |
| 55494 | 160 mixed banks, 200 updates per arm, 50 roots per arm; `afterok:55493` | same | PENDING dependency |
| 55495 | Paired CPU report; `afterok:55494` | main, 2 CPU, 4 GB, 10 min | PENDING dependency |

Each run uses a unique `mixed_<jobid>_<timestamp>/` directory with source
snapshot and SHA-256 sums. No result from the mixed experiment is claimed.

## 2026-09-28 07:40 UTC: mixed-bank result

Both `squeue` (empty) and `sacct` (COMPLETED 0:0) confirm 55493 (33 s),
55494 (8:49), and 55495 (2 s). Full run:
`/mnt/data/nhatnc129/jepa/action_response_cube/mixed_55494_20260928T065913Z/`.
The mixed bank increased mean true-endpoint latent pair MSE from 0.271 to
0.669 and median physical-distance spread from 1.11 to 2.71 cm. This tests
the narrow-final-bank hypothesis on the same 160 training states.

On the same 50 held-out evaluation roots, mixed prediction-only achieved
**37/50** and mixed action-response **34/50**. Paired gain -6 pp, bootstrap
95% interval [-14,0]; response won no root exclusively, prediction-only won
three. Mean final distance was 7.79 cm versus 6.71 cm. Despite this,
response had slightly lower dev prediction loss (0.572 vs 0.576) and lower
dev action-response loss (0.614 vs 0.646). Thus increasing candidate-effect
spread improved the supervised signal but did not improve deployed control.
Evidence: `bank_signal.json`, each arm's `train.json`, and
`paired_analysis.json` in the run directory. This is a single training seed
and a development comparison, not proof that CTA or every action-response
formulation is ineffective. The direct all-pair loss tested here has not
established a control benefit; it should not be the main novelty claim.
