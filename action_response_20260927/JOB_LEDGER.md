# Job ledger

2026-09-27: New independent PushT action-response pipeline submitted. Source is
`pipeline.py`, launcher `slurm_pipeline.sh`; each run snapshots Python source,
hashes source/launcher, records upstream commit and official checkpoint hash.
Output directories are unique `run_<jobid>_<UTC>` under
`/mnt/data/nhatnc129/jepa/action_response/`; `job_<jobid>.path` points to each.

| Job | Purpose | Resources / limit | Verified state |
| --- | --- | --- | --- |
| 55338 | Technical end-to-end smoke: 16 branch pairs, 2 updates per arm, one common PushT eval root | mig 3g.40gb, 8 CPU, 96 GB, 3 h | Canceled while pending after CAI-JEPA overlap review; no compute run |
| 55339 | Initial full submission with only four eval roots | same | CANCELLED before start; four roots were insufficient for the paper's 50-task comparison |
| 55340 | Full development run afterok:55338: 96 branch pairs, 160 updates per arm, 50 common eval roots, prediction/response | same | Canceled while pending after CAI-JEPA overlap review; no compute run |

At submission, unrelated CTA and OGBench jobs 55330_1 and 55328 were running,
using the available user MIG quota. No duplicate action-response job was active.
Python compilation and shell syntax passed. No model, training, or control
result is claimed yet. Scientific comparison is paired on 50 roots; a single
fine-tuning seed and 96 pairs make this a development result, not a confirmed
paper-level gain. The published LeWM PushT reference is 96.0 ± 2.83% across
three training seeds on 50 tasks.

Before either job started, evaluation was narrowed to prediction-only versus
response to honor the user's instruction to use the published LeWM baseline
without rerunning its checkpoint. The job snapshots the exact `pipeline.py`
loaded at execution and writes its hash in `SOURCE_SHA256SUMS`.

After inspecting the prior CAI-JEPA same-state, multi-horizon diagnostic and
counterfactual predictor results, both 55338 and 55340 were canceled before
starting. See `OVERLAP_REVIEW.md`. The source is preserved for reuse; no
training or evaluation artifacts exist for these jobs.

## Correction and resubmission — 2026-09-27

The user pointed out that the old MetaWorld null cannot rule out LeWM PushT,
internal prior attempts do not themselves remove novelty, and CTA is a
separate project. These corrections are incorporated in `OVERLAP_REVIEW.md`.
Canceling both pending jobs had been premature. The run now additionally
records native final-state distance because published PushT success is near
the ceiling. Original-LeWM evaluation remains omitted at the user's request;
prediction-only is the matched-data control.

Before resubmission, `squeue` showed only unrelated job 55328 running;
`sacct` confirmed 55338 and 55340 were canceled with zero elapsed time. Python
compilation and bash syntax checks passed. No duplicate action-response job
was running.

| Job | Purpose | Resources / limit | Submission state |
| --- | --- | --- | --- |
| 55341 | Technical end-to-end smoke: 16 branch pairs, two updates per arm, one closed-loop root | mig 3g.40gb, 8 CPU, 96 GB, 3 h | Submitted |
| 55342 | Full run afterok:55341: 96 branch pairs, 160 updates per arm, 50 common roots | same | Submitted, dependency |

Both use the same unique-run naming and source snapshots described above.
Results and job completion remain unverified at submission time.

## First closed-loop result and follow-up — 2026-09-28

`squeue` showed neither 55341 nor 55342 active; `sacct` verified both
`COMPLETED` with exit code `0:0` (39 seconds and 2:23 respectively). Full
artifacts: `/mnt/data/nhatnc129/jepa/action_response/run_55342_20260927T175256Z/`.
All 50 common evaluation roots completed for both arms. Prediction-only
fine-tuning succeeded on 30/50 (60%); prediction plus action-response on
34/50 (68%). Paired result: 28 both succeed, 14 neither, six response-only,
two prediction-only. CPU job 55384 completed `0:0` and wrote
`paired_analysis.json`: +8 percentage points, paired bootstrap 95% interval
[-2, 18] points, exact two-sided McNemar p=0.289. This is a single
fine-tuning seed and is not a robust improvement, let alone a win over the
published 96%. The official checkpoint was not rerun.

The first run's `native_final_state_distance` is an inappropriate quality
metric: upstream `eval_state` applies a symmetry-aware angle tolerance to
success, but its Euclidean distance also includes agent velocity and raw
angle. Thus its mean (149.3 prediction, 136.2 response) should not be read
as goal error. The new pipeline records position and symmetry-aware angle
error explicitly. Training dev losses worsened late in both arms (at step
160: prediction dev MSE 0.2335 vs earlier minimum 0.1988; response dev
objective likewise fluctuated), consistent with a checkpoint-selection and
small-data problem rather than evidence that action-response has no value.

Source updated in `pipeline.py` and `slurm_pipeline.sh` to evaluate at step
zero and each 20 updates, retain the best checkpoint by each arm's held-out
training objective, lower LR to 2e-6, and report the appropriate state errors.
Syntax checks passed. Before submission, `squeue` found no duplicate
action-response job; only unrelated 55379 was running. Follow-up 55386 was
submitted with 384 branch pairs, 200 updates per arm, 50 common roots,
H100 3g.40gb MIG, 8 CPUs, 96 GB, 3 h. Its unique run directory is
`/mnt/data/nhatnc129/jepa/action_response/run_55386_20260928T013356Z/`;
`squeue` and `sacct` both showed RUNNING at the first check. This is an
iteration aimed at preserving the pretrained planner while testing whether
more branch coverage helps; no outcome is claimed yet.

The 55386 training report selected step 0 for **both** arms: at step 0,
prediction dev MSE was 0.0875 and response dev objective 0.2030; at step 20,
these had worsened to 0.1297 and 0.2613. Its 50-root continuation would
therefore only evaluate two copies of the unchanged checkpoint. Inspection
of the checkpoint architecture exposed a concrete training error:
`pred_proj` contains `BatchNorm1d`, whose running statistics the small,
correlated rollout batches were overwriting in `train()` mode. Job 55386 was
`scancel`ed after 2:29; `squeue` became empty and `sacct` confirmed CANCELLED.
Its branch cache was complete and is preserved read-only.

`pipeline.py` now keeps `pred_proj` in eval mode while gradients still update
its parameters, and accepts a previously collected branch cache. Syntax
checks passed. After confirming no duplicate action-response job in `squeue`,
job 55387 was submitted using the 55386 cache, the same 384 pairs, 200
updates/arm, 50 roots, and 2e-6 LR, with the same explicit H100 MIG/CPU/memory
resources and 3 h limit. Unique output:
`/mnt/data/nhatnc129/jepa/action_response/run_55387_20260928T013708Z/`.
`squeue` and `sacct` both showed RUNNING at first verification; results
remain pending.

Job 55387 later completed `0:0` in 2:06, verified by both `squeue` and
`sacct`. Step-40 checkpoint was selected in both arms (prediction dev MSE
0.0831 vs step-zero 0.0875; response dev objective 0.1966 vs 0.2030).
Closed-loop success on the same 50 roots was 46/50 (92%) prediction-only and
47/50 (94%) response. Mean position error was 20.41 vs 16.25 pixels;
mean symmetry-aware angle error 0.131 vs 0.096 radians. CPU paired-analysis
jobs 55390 and 55391 completed `0:0`. Final `paired_analysis.json` reports
45 successes in both arms, two in neither, two response-only and one
prediction-only: +2 percentage points, bootstrap 95% interval [-4, 10],
exact McNemar p=1.0. Mean position-error reduction was 4.16 pixels with
paired bootstrap interval [-1.03, 11.98]. Thus this run is consistent with
benefit but does not establish a reliable advantage, and remains below the
paper's 96% point estimate. The 96% uses different training seeds; the
published reference is not a paired outcome for this fine-tune.

The next focused iteration changes only the response target to the fifth
action block, since CEM's goal cost is on the planned endpoint. Prediction
loss still spans blocks 1, 3 and 5 and the matched prediction-only arm is
otherwise identical. This tests whether allocating response supervision to
the decision endpoint helps control, using the same saved branch cache and
50 roots; it is not a sweep over many configurations.

Python and shell syntax checks passed; `squeue` showed no duplicate
action-response job before submission. Job 55392 was submitted with
`RESPONSE_TARGET=endpoint`, the 55386 branch cache, 384 pairs, 200 updates,
50 roots and 2e-6 LR (H100 MIG 3g.40gb, 8 CPU, 96 GB, 3 h). Its distinct
output directory is `/mnt/data/nhatnc129/jepa/action_response/run_55392_20260928T014050Z/`.
Both `squeue` and `sacct` showed RUNNING at first verification.

Job 55392 completed `0:0` in 2:10, verified by `squeue` and `sacct`.
The endpoint-response arm selected step 20; the matched prediction-only arm
again selected step 40. On 50 shared roots, prediction-only succeeded on
46/50 (92%) and endpoint-response on 48/50 (96%), with mean position error
20.29 vs 17.28 pixels and symmetry-aware angle error 0.130 vs 0.112 radians.
CPU paired-analysis job 55394 completed `0:0`: two response-only successes,
zero prediction-only successes, 46 both successes, two neither. Paired gain
was +4 points, bootstrap 95% interval [0, 10], exact McNemar p=0.5;
position-error reduction 3.00 pixels with interval [-1.01, 9.68]. This
equals the published 96% point estimate, not a win, and is only one
fine-tuning seed. The paper's 96% ±2.83 is across three training seeds;
the original checkpoint was not rerun.

To check independent repeatability before a stronger claim, job 55395 was
submitted with a new branch/data/training seed 28, otherwise identical
384 pairs, 200 updates, endpoint loss, 50 roots, 2e-6 LR, H100 MIG 3g.40gb,
8 CPU, 96 GB, 3 h. Before submission `squeue` showed no duplicate
action-response job. Unique directory:
`/mnt/data/nhatnc129/jepa/action_response/run_55395_20260928T014356Z/`.
At first check `squeue` showed RUNNING; `sacct` showed its batch step
RUNNING and parent PENDING (a startup accounting lag). No result is claimed.

Job 55395 completed `0:0` in 2:55, verified by `squeue` and `sacct`.
Independent seed 28 yielded the same rates: prediction-only 46/50 (92%),
endpoint-response 48/50 (96%). It selected steps 180 and 200 respectively.
CPU paired-analysis job 55397 completed `0:0`: three response-only successes,
one prediction-only, 45 both, one neither. Paired gain +4 points,
bootstrap interval [-4, 12], exact McNemar p=0.625. Mean position error
was 16.59 prediction and 16.72 response; there is no continuous-error
improvement on this seed. This repeat supports a success-count trend but
does not by itself establish a reliable advantage or beat the paper.

For a three-seed comparison to the paper's reporting scale, job 55398 was
submitted with independent seed 29, otherwise identical settings and
resources (384 pairs, 200 updates, endpoint loss, 50 roots, 2e-6 LR,
H100 MIG 3g.40gb, 8 CPU, 96 GB, 3 h). Syntax checks passed and `squeue`
showed no duplicate action-response job before submission. Unique output:
`/mnt/data/nhatnc129/jepa/action_response/run_55398_20260928T014725Z/`.
`squeue` and `sacct` both showed RUNNING at initial verification.

Job 55398 completed `0:0` in 2:28 (`squeue` empty, `sacct` COMPLETED).
Independent seed 29 selected step 100 for prediction-only and step 80 for
endpoint-response, but control went **backwards**: 47/50 (94%) prediction-only
vs 45/50 (90%) response. Response also had larger mean position error
(24.07 vs 17.34 pixels). CPU paired-analysis job 55400 completed `0:0`:
one response-only success, three prediction-only, 44 both, two neither;
paired gain -4 points with bootstrap interval [-12, 4]. This is evidence of
seed-dependent closed-loop degradation despite lower held-out training loss.

CPU three-seed aggregation job 55401 completed `0:0`. Across seeds 27, 28,
29 on 50 common roots, prediction-only scored [46, 46, 47]/50, mean 92.67%;
endpoint-response scored [48, 48, 45]/50, mean 94.0%. Mean paired gain was
+1.33 percentage points, two-level paired bootstrap 95% interval [-6, 8]
points (resampling training seeds and shared roots). The endpoint method
does **not** beat the paper's 96% PushT point estimate, nor establish a
reliable gain over the matched fine-tune. The published score is unpaired
with these runs. The complete machine-readable aggregation is
`/mnt/data/nhatnc129/jepa/action_response/three_seed_endpoint_summary.json`.

Observed bottleneck: dev embedding MSE/response loss alone did not predict
the seed-29 control result. A next method iteration should incorporate
planner-relevant ranking or on-policy candidate coverage, with a held-out
selection signal and independent seeds, rather than tune on the 50 test
roots. No further job is active for this direction.
