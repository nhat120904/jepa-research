# Jobs and implementation status

2026-09-21: new programme directory created; active research note moved with a redirect.
Queue/accounting checked before implementation: no active jobs returned for this user.

No method result yet. The first submission is a bounded CPU preflight/data job.
GPU encoding/training is not automatically scheduled behind an unqualified preflight.

## 53521 — RGB navigation preflight + small dataset

- Submitted 2026-09-21 via `sbatch scripts/slurm_preflight.sh`.
- Resources: partition main, 4 CPU, 16 GB RAM, 30-minute limit, **no GPU requested**.
- Scope: unit/model-interface tests on CPU; original-Wall RGB/replay test; 24 prefixes ×
  64 action candidates × 32 native steps; 64/16/16 train/val/test episodes of 128 steps.
- Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/preflight_53521.out`.
- Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/preflight_53521/`.
- Runtime dependencies are installed into the run's private `deps/`, not shared venvs.
- Pre-submission: `py_compile` and `bash -n` passed. Runtime test outcomes pending.
- Post-submission: both `squeue` and `sacct` reported RUNNING on worker-0; accounting
  showed 4 CPU / 16 GB and no GPU allocation. This is a submission-time snapshot.
- No WM training, GPU encoding, or learned-control claim. No dependent GPU job submitted.

Result: COMPLETED in 53 seconds, exit 0; 9/9 tests passed; 96 RGB/action episodes
collected (64/16/16, 7.8 MB total run). Reach-B oracle-bank and RGB-selected success were
both 12/24; mean random candidate success 6.25%. Ordered A-before-B oracle-bank success
was 0/24 (0/1536 candidate branches), so the unguided bank provides no ordered-selection
headroom. This is a proposal failure, not a result about an untrained WM. Protocol v2
switches the rerun to an RGB-goal-conditioned shared proposal and preserves this dataset.

## 53529 — ordered-headroom v2

- Submitted after both queue/accounting checks showed 53521 completed and no active job.
- CPU-only, 4 CPU / 16 GB / 30 minutes; config `headroom_v2.json`, screen-only mode.
- 48 prefixes, 128 shared RGB-goal-conditioned candidates, horizon 48; no new dataset,
  feature encoding or WM training. Goals remain sampled before candidates/outcomes.
- Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/preflight_53529.out`.
- Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/preflight_53529/`.
- Submission-time state from both `squeue` and `sacct`: RUNNING on worker-0, no GPU.

Result: COMPLETED in 3m25s, exit 0; 10/10 tests passed. Ordered headroom became nonzero
and non-saturated: 19/48 prefixes contained at least one A-before-B success, mean random
candidate success was 7.44%, oracle-bank success 39.58%, and the fixed RGB query selected
a successful candidate in the same 19/48 prefixes (zero physical selection regret on
this development screen). Reach-B had 38 informative prefixes and 79.17% oracle/selected
success. This qualifies proposal/query plumbing for learned-WM implementation; it is not
a learned result because scores were computed from realized future RGB.

Canonical interpretation: [HEADROOM_RESULT_53521_53529.md](HEADROOM_RESULT_53521_53529.md).

## 53550 — frozen DINOv3 spatial encoding

- Submitted after `squeue` showed no active jobs and `sacct` showed both preflights
  completed. One H100, 8 CPU, 64 GB, explicit 30-minute limit.
- Input: immutable dataset from 53521. Reads train+val only (64+16 episodes); test sealed.
- Encoder: local DINOv3 ViT-L/16 checkpoint, frozen; 224 input, 4x4 pooled patch grid,
  16x1024 fp16 per frame. No network download and no model training.
- Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/encode_53550.out`.
- Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/encode_53550/`.
- Submission-time state verified by both tools: PENDING (Priority); requested resources
  include one GPU. The job exits after encoding and has no dependent training submission.

## 53563 — CPU unit tests for complete pilot interfaces

- Submitted while 53550 remained PENDING and was not holding a GPU.
- CPU-only, 4 CPU / 12 GB / five-minute limit. Tests all old invariants plus new spatial
  query/generic codecs and direct/frame/endpoint predictor shapes and gradients.
- Submission-time state: PENDING (Priority) by both `squeue` and `sacct`; no GPU requested.
- Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/unit_53563.out`.

Result: COMPLETED in six seconds, exit 0; 11/11 tests passed, including all five model
arms, frozen-reader gradient flow, strict query semantics and exact environment replay.

## 53568 — review: competent default versus oracle bank

- CPU-only diagnostic, 2 CPU / 8 GB / five-minute limit; completed in six seconds, exit 0.
- Replayed immutable 53529 source and goals. All 48 old selected outcomes reproduced;
  regenerated banks match the two archived arrays exactly.
- Candidate 1 (fixed A-then-B) succeeds 11/48; oracle bank 19/48. Remaining selection
  headroom is 8/48 = 16.67 pp in this full-chunk development screen.
- Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/review_default_53568/`.
- [Review and recommendations](REVIEW_AND_POLICY_GUIDANCE_20260921.md) qualify previous
  headroom statements and identify protocol/code issues before training.
- At review time, encode 53550 remains PENDING (Priority); no training jobs submitted.

## Update 2026-09-22: 53550 completed

Verified using both `squeue` and `sacct`: COMPLETED, exit 0, 41 seconds. Manifest
contains 80 train/validation episodes, each 129×16×1024 fp16; no test read.
No model training occurred. Existing features remain reusable for the old dataset;
the prospective branching correction requires encoding new RGB, not rerunning these.

## 53629 — professor demo + prospective branch dataset

CPU-only, 4 CPU / 16 GB / 15-minute limit. COMPLETED in 42 seconds, exit 0, verified
using both queue/accounting. Immutable code snapshot and hashes under run directory.
Three-case actual simulator replay demo (standalone HTML, GIF, preview PNG), with
explicit oracle-not-learned labels and aggregate default 11/48 versus oracle 19/48.
Replay assertions passed. Fresh dataset: 672 branches from 24 train/12 validation
prefixes, eight fixed candidates; shared goals chosen before outcomes. No test read.
Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/prepare_53629/`.
New training entry point/protocol: `train_branches.py`, `TRAIN_PROTOCOL_V2.md`.

## 53630 — new branch encoding + corrected end-to-end profile

Submitted 2026-09-22 after queue empty/accounting completed checks, via
`scripts/slurm_branch_profile.sh`. One GPU, 8 CPU, 64 GB, explicit 30-minute limit.
Runs unit tests first; exits on any failure. Encodes ONLY new prospective branches
from 53629 (not duplicate encoding of the 53550 dataset), then 20 updates per stage
to exercise codecs, all forecasting/control arms, within-prefix ranking and outputs.
No dependent full-training job. Profile metrics MUST NOT be used as method results.
Source/config/docs/tests snapshotted with hashes before work.

Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/branch_profile_53630.out`.
Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/branch_profile_53630/`.
At handoff: both `squeue` and `sacct` report RUNNING on worker-0 (15 seconds elapsed).
Completion and runtime test outcomes not yet verified. No foreground polling loop.

Completion update: both queue/accounting checked; 53630 COMPLETED, exit 0, 1m48s.
14/14 tests passed; all 672 branches encoded; all arms finished the 20-update profile.
See `PROFILE_RESULT_53630.md` for the data-only small-headroom warning and next bounded
full pilot. No scientific accuracy verdict is drawn from profile checkpoints.

## 53642 — full one-seed prospective branch pilot

Submitted through `scripts/slurm_branch_train.sh`, after checking both queue/accounting
and finding no active duplicate. One GPU, 8 CPU, 64 GB, explicit 30-minute limit.
Reuse immutable features from 53630; no re-encoding and no sealed test read. Fresh seed
20260922; 1,200 updates per codec, 600 generic-reader updates and 1,800 updates per
forecasting/control arm, batch 16. Seven arms plus observed-codec and constant diagnostics.
Same training protocol as profile; added post-training train-fit MSE and readable report.
Source/docs/tests snapshot and feature-manifest hash saved in run directory.

Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/branch_train_53642.out`.
Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/branch_train_53642/output/`.
Expected outputs: `result.json`, `predictions.jsonl`, `checkpoint.pt`, `REPORT.md`.
No follow-on job or planning experiment submitted. This is development, not confirmation.

Completion update: 53642 COMPLETED, exit 0, 2m19s; 14/14 tests passed. Verified via
both queue and accounting. On A-before-B, query-summary selection regrets at H32/48/64
are .10018/.15300/.17948 versus default .01853/.02369/.00615. Even the observed-future
query codec ranks poorly. No control benefit demonstrated; single-seed development only.
Next bounded scope: `CODEC_RESCUE_PROTOCOL.md`, observed-future recoverability and
matched temporal-compression × within-prefix-loss ablation, not another MPC trial.

## 53659 — observed-future codec rescue

Submitted after empty-queue and completed-53642 accounting checks. One GPU, 8 CPU,
64 GB, explicit 20-minute limit. New `codec_rescue.py` and four extra tests; all tests
run first and launcher exits on failure. Reuses 53630 frozen cache, no encoding.
Train-only frame metric (1,000 updates), two tiny-fit tests (600 each), then four
matched codec/loss arms (1,200 each) ONLY if at least one tiny fit passes. All targets
RGB-derived, no privileged states/rewards, no test set and no policy/control rollout.

Protocol: `CODEC_RESCUE_PROTOCOL.md`; source/docs/tests snapshotted with hashes.
Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/codec_rescue_53659.out`.
Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/codec_rescue_53659/output/`.
Expected `result.json`, `REPORT.md`, `diagnostic_checkpoints.pt`. No dependent jobs.

53659 completion: queue/accounting both checked, COMPLETED, exit 0, 1m15s. 18/18 tests
passed; both tiny fits passed, all four ablations ran. Learned frame affinity + exact
query aggregation recovered ordered targets well (val MSE .000417/.002839/.000878),
but pair loss did not consistently rescue the summary. This is observed-future signal
recoverability, NOT forecasting or planning. Follow-up protocol: `METRIC_BRIDGE_PROTOCOL.md`.

## 53674 — frozen visual metric -> codec -> action-conditioned prediction

Submitted via `scripts/slurm_metric_bridge.sh` after duplicate queue/accounting checks.
One GPU, 8 CPU, 64 GB, explicit 30-minute cap. Runs tests first, then verifies frozen
53659 reference reproduction before new training. Same 53630 features; no re-encoding.
Observed codecs 4/16 tokens (1,600 updates each), fixed pooling references, full metric
frame predictor, separately trained no-action frame predictor, direct query predictor
(1,800 updates each). Summary students ONLY for codecs passing the predeclared
development retention gate; no MPC, policy training or automatic next job.

Source/docs/tests and input hashes saved under run directory. Log:
`/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/metric_bridge_53674.out`.
Output: `/mnt/data/nhatnc129/jepa/predictive_abstraction/metric_bridge_53674/output/`.
Expected artifacts: `result.json`, `REPORT.md`, `checkpoints.pt`.
Submission-time status verified with both `squeue` and `sacct`: PENDING (Resources).
Not running/holding a GPU at this snapshot; completion not yet claimed.

53674 completion: both queue/accounting checked, COMPLETED, exit 0, 1m39s. 22 tests
passed and frozen metric reference reproduced. Learned summary4 failed all retention
gates; summary16 only passed H32, so NO summary student was trained. Fixed 16-bin
compression preserves queries much better, but full metric-frame forecast remains
weak versus the no-action control on validation A-before-B. No control benefit.
Follow-up: `FIXED_SUMMARY_FORECAST_PROTOCOL.md`, tiny forecast fit then (only if it
passes) compact fixed-bin vs full-frame prediction with matched coarse-loss control.

## 53698 — fixed-summary forecast feasibility

Submitted via `scripts/slurm_fixed_summary_forecast.sh` after queue/accounting checks.
One GPU, 8 CPU, 64 GB, explicit 20-minute cap. Runs all tests first and reproduces
the 53674 fixed-bin reference. Two tiny TRAIN forecast fits (1,000 updates each).
Only if at least one passes: four fresh 1,800-update arms (dense-frame, coarse-loss
frame, compact16, compact16-no-action). No data expansion, encoding, privileged labels,
test reads, policy training, MPC or automatic follow-on jobs.

Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/fixed_summary_53698.out`.
Output: `/mnt/data/nhatnc129/jepa/predictive_abstraction/fixed_summary_53698/output/`.
Expected: `result.json`, `REPORT.md`, `checkpoints.pt`. Source/docs/tests and input
hashes snapshotted in run directory. Completion not yet claimed.

53698 completion: queue/accounting both checked, COMPLETED exit 0, 1m41s. 25 tests
passed. Compact prediction contains action signal (H48 bin MSE .533 vs no-action
.881; shuffled actions 1.009), but ordered query/ranking did not improve. No-action
ties favor default under argmax-first; do not equate this with prediction skill.
Next bounded scope: READOUT_DIAGNOSTIC_PROTOCOL.md, frozen predictors and tiny readers
only. Conditional architecture plan: ARCHITECTURE_DECISION_AFTER_FORECAST_FAILURE.md.

## 53700 — frozen-predictor readout diagnostic

Submitted after empty queue and completed prior-job accounting checks. One GPU,
8 CPU, 64 GB, explicit 15-minute cap. Existing 53698 metric/predictors remain frozen;
four small readers fit on 18 old train prefixes, with 6 readout-held-out old training
prefixes and 12 reused development validation prefixes. Tests run first on compute.
Includes frozen-reference reproduction, event/order diagnostics, no-action control,
argmax-first and uniform-tie regret, paired prefix bootstrap. No test reads, new
rollouts, encoding, policy training, WM retraining or automatic follow-on jobs.

Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/readout_53700.out`.
Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/readout_53700/output/`.
Expected `result.json`, `REPORT.md`, `readers.pt`; source/docs/tests and input hashes
snapshotted in the run directory. Local Python/shell syntax checks passed before
submission; compute test outcomes and diagnostic results not yet claimed.
Submission-time `squeue` and `sacct` both report RUNNING on worker-0 (elapsed 16s).

53700 completion: both queue/accounting checked, COMPLETED exit 0, 31 seconds; 29
tests passed. Reference reproduced. Readout adaptation does not rescue validation
ordered prediction/ranking consistently across horizons. No-action first-argmax
advantage is tie-breaking, not prediction skill. See READOUT_RESULT_53700.md.
Proceed to LOCAL_TRANSITION_PROTOCOL.md, the previously specified short-dynamics
branch, without new data/encoding/policy/MPC or another summary/readout sweep.

## 53741 — local spatial transition feasibility

Submitted through `scripts/slurm_local_transition.sh` after empty queue and completed
53700 accounting checks. One GPU, 8 CPU, 64 GB, explicit 20-minute limit. Cached
features only; no new data/encoding. Tests first, then 400-step tiny fit; only if tiny
fit beats persistence by 50% train step1 / step4 / step4-no-action for 1,200 updates
each. Image-feature training targets, frozen RGB-affinity metric for evaluation only.
Teacher-forced and open-loop eval, persistence and action-shuffle controls, explicit
bank vs evaluated horizon. No new summary, VLA, physical labels or planning rollout.

Protocol: LOCAL_TRANSITION_PROTOCOL.md. Local Python/shell syntax and diff checks
passed; compute tests/results not yet claimed. Source/docs/tests and input hashes
snapshotted per run. Log:
`/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/local_transition_53741.out`.
Outputs: `/mnt/data/nhatnc129/jepa/predictive_abstraction/local_transition_53741/output/`.
Expected `result.json`, `REPORT.md`, and model checkpoints if full training runs.
Submission snapshot: both `squeue` and `sacct` report PENDING (Priority); not yet
running or holding a GPU. No automatic follow-on jobs submitted.

53741 completion: squeue/accounting checked, COMPLETED exit 0, 54 seconds; 33 tests
passed. Tiny fit and observed reference reproduced; short-dynamics gate passed
(14.5% TF, 51.8% H8 vs persistence, 46.0% H8 vs no-action feature-error reductions).
Long query selection still fails: H48 step4 misses 31/41 events, regret .19435 vs
default .02369. H8 bank has no positive events, and TF persistence also reads events
well; do not call the gate a planning/event-success result. See LOCAL_TRANSITION_RESULT_53741.md.
Next bounded scope: ROLLOUT_EXTENSION_PROTOCOL.md, matched continuation budget controls.

## 53748 — controlled rollout extension

Submitted via scripts/slurm_rollout_extension.sh after empty queue and accounting
checks. One GPU, 8 CPU, 64 GB, explicit 20-minute cap. Tests first; reproduce frozen
53741 reference before new training. Same step4 checkpoint initializes all three
continuations: 4x600, 16x600, 4x2400 (unroll length x optimizer updates). Image-feature
loss only; metric frozen for evaluation. Includes matched observation-refresh controls,
train/val query and feature metrics, prefix bootstrap, default and tie handling.
No new data/encoding, physical labels, VLA, composer or control/MPC rollouts.

Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/rollout_extension_53748.out`.
Output: `/mnt/data/nhatnc129/jepa/predictive_abstraction/rollout_extension_53748/output/`.
Expected result.json, REPORT.md and checkpoints.pt; source/docs/tests/input hashes
snapshotted. Local syntax/diff checks passed; compute test and training outcomes not
yet claimed. No automatic follow-on jobs.
Submission-time squeue and sacct both report RUNNING, worker-0, elapsed 15 seconds.

53748 completion: both queue/accounting checked, COMPLETED exit 0, 1m38s; 37 tests
passed and reference reproduced. Query-progress gate fails H48 and H64. Feature
errors decrease but ordered queries/ranking do not improve enough; close longer-
rollout rescue, no additional unroll sweep. See ROLLOUT_EXTENSION_RESULT_53748.md.
Last scoped implementation test: QUERY_NATIVE_STOP_PROTOCOL.md with explicit stop
rules, matched query-supervised arms, no latent-coordinate loss and no physical labels.

## 53750 — final query-native Wall implementation pilot

Submitted after empty queue/accounting checks, via scripts/slurm_query_native.sh.
One GPU, 8 CPU, 64 GB, explicit 20-minute limit. Tests first, then four 2,000-update
arms: summary16, ordered frame tokens, cached direct query predictor, summary no-action.
Same RGB-derived dense/query/pair loss, no latent-coordinate alignment loss. Same
cached data and frozen anchor features; no physical labels, policy or MPC. Cross-query
anchor donors locked by same-layout prefix order, validation-only labels never trained.
Operational stopping rule locked in QUERY_NATIVE_STOP_PROTOCOL.md. No automatic
follow-on job; negative result closes the current implementation repair cycle.

Log: `/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/query_native_53750.out`.
Outputs: `/mnt/data/nhatnc129/jepa/predictive_abstraction/query_native_53750/output/`.
Expected result.json, REPORT.md, checkpoints.pt; immutable code/docs/test snapshot
and input hashes. Local syntax/diff checks passed; compute tests/results not yet claimed.
Submission snapshot: squeue and sacct both RUNNING on worker-0, elapsed 15 seconds.

53750 completion: squeue/accounting checked, COMPLETED exit 0, 2m05s; 41 tests
passed. Predeclared verdict `STOP_CURRENT_WALL_METHOD_IMPLEMENTATION`: summary,
frame-token and direct action-aware arms all fail native H48/H64 feasibility gates.
Summary H48/H64 ordered MSE .04876/.05870 and regret .11694/.17948, versus actual
default regret .02369/.00615. No-action's low first-argmax regret is a tie artifact;
uniform-tie regret .15629/.15999. Summary and frame are statistically unresolved on
ordered MSE. See QUERY_NATIVE_RESULT_53750.md. Current Wall repair cycle is closed;
no automatic follow-on job submitted.
