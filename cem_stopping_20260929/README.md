# Adaptive CEM stopping on frozen LeWM (pilot, v1)

Question: with a frozen, released world model, does refining CEM for more
iterations ever make the executed plan worse, and can a deployable signal
(computed only from the planner's own costs) decide when to stop better than
a fixed iteration count tuned on development roots?

Status: implementation + smoke. No result is claimed yet. Job IDs and outcomes
are in [JOB_LEDGER.md](JOB_LEDGER.md).

## Controller under test

Released LeWM planner (`stable-worldmodel` pinned at the commit recorded in
every run directory): CEM, 300 samples, 30 elites, initial std 1, horizon 5
blocks of 5 raw actions, receding horizon 5 (the whole plan is executed),
history length 1, goal offset 25, budget 50 steps, so an episode has at most
two plans. The executed action is CEM's elite mean, as upstream. The released
config uses 30 iterations on every task; the paper reports 10 outside PushT.

Deviation from the released evaluator, applied identically to every arm: the
CEM random generator is reseeded before each plan with
`seed = base + 1000 * root + plan_index`. Upstream seeds it once and shares
the stream across environments and plans, so changing the iteration count
changes every later draw. With per-plan seeding, stopping after k iterations
returns exactly the elite mean after iteration k of a full 30-iteration run
(prefix property; checked in the smoke job).

Start frames are rendered from an exact simulator restore (Cube: the
`restore_complete` procedure verified in
`diagnosis/results/ogb_true_endpoint_corrected/`; Reacher: reset with a fixed
seed, `set_state`, `set_target_qpos`). Goal images are the dataset frames, as
upstream. The smoke job logs the dataset-versus-render start-image difference.

## Arenas

- OGBench Cube (`quentinll/lewm-cube`, `ogbench/cube_single_expert.h5`).
  Success: termination at goal during the episode or final cube-goal distance
  <= 0.04 m.
- DMC Reacher (`quentinll/lewm-reacher`, the `reacher.h5` of the LeWM release).
  Success: the environment's `qpos_match` termination (every joint within
  0.05 rad).

PushT is near ceiling (92-96%); TwoRoom needs a checkpoint download. Both are
deferred.

## Stopping tree

For each root, plan 1 runs one 30-iteration CEM and records the elite mean at
the checkpoints `C = {1, 2, 3, 5, 7, 10, 15, 20, 30}`. Each checkpoint mean is
executed from an exact restore. If the episode has not succeeded, plan 2 runs
one 30-iteration CEM from the resulting frame (seed independent of the first
branch) and each of its checkpoint means is executed after replaying the first
branch from the root. This gives 81 leaves per root.

- Diagonal `(k, k)`: closed-loop outcome of fixed-K CEM.
- Any stopping rule that stops at a checkpoint of `C` defines one path through
  the tree, so it is evaluated offline and paired on the same roots.
- Best leaf per root: oracle over this finite set of per-plan choices, with the
  simulator and CEM noise fixed. It is not a bound for rules that change the
  CEM update itself, other checkpoints, or other noise.

Rules are causal: at checkpoint k they read only the costs of iterations
1..k, and return the elite mean after iteration k. Compute is counted as
iterations actually run (k1 + k2, plan 2 only if executed).

## Rules (locked before the dev pilot)

All use the per-iteration candidate costs of the current plan; `s` is the
sorted cost vector of the current iteration, `K = 30` elites.

- `fixed(K)`: stop at K.
- `converge(eps)`: stop at checkpoint k (not the first) when the relative
  decrease of the elite-mean cost since the previous checkpoint is below eps.
  The elite-mean cost is the mean of the 30 smallest candidate costs of that
  iteration (the quantity upstream reports), not the cost of the returned
  mean action.
- `gap(tau)` (R2-CEM as proposed): stop when `(s[K] - s[K-1]) / IQR_1 < tau`,
  where `IQR_1` is the interquartile range of the first population's costs.
  If `IQR_1 <= 1e-12 * max(1, |median_1|)` stop at the first checkpoint.
- `band(tau)` (secondary): stop when at least K candidates lie within
  `tau * IQR_1` of the elite boundary `s[K-1]`; same degenerate handling.
- First checkpoint satisfying the condition stops (no consecutive-hit rule).
  If no checkpoint fires, the rule returns the mean after 30 iterations.

Thresholds are tuned on development roots by success (ties: fewer
iterations). They are dev-tuned stopping thresholds, not ranking-reliability
calibrations. A calibrated variant (tau from pairwise inversion rates on
labelled dev candidates) is reported separately.

## Splits and inference

300 roots per task sampled uniformly over valid dataset starts
(seed 20260929), randomly permuted; roots 0-99 are development, 100-299 test.
Test roots are not run until the rules and thresholds are frozen. Uncertainty:
paired bootstrap over roots (10,000 draws). Pre-set non-inferiority margin for
an efficiency claim: 5 success points versus the dev-tuned fixed K, with the
test-set 95% lower bound above -5.

Reading of outcomes (from the review, 2026-09-29):
- equal to tuned fixed-K without clear compute saving: weak contribution;
- non-inferior within the margin and clearly cheaper than fixed-K and
  convergence stopping: useful;
- better success and cheaper: stronger evidence.

## Diagnostics (not gates)

On the first 20 development roots, plan 1 also stores 32 fixed candidates at
iterations {1, 3, 5, 10, 20, 30}; each is simulated for 25 steps to give
pairwise ranking accuracy with a physical indifference zone (Cube 1 cm,
Reacher 0.0125 rad; a quarter of the success tolerance) and the margin versus
reliability curve.

## Checks in the smoke job

1. Recorder + reseeding leaves upstream CEM unchanged: an uninstrumented
   `CEMSolver` with the same seed returns the same actions; a 10-iteration
   solver equals the recorded checkpoint-10 mean (prefix property).
2. Exact replay: restoring a root twice gives identical pixels, and replaying
   a leaf twice gives identical final simulator state.
3. Tree equals live closed loop: continuous episodes for several rules match
   their tree leaves exactly.
4. Unit tests for the rules and the offline evaluator.

## Layout

- `cemstop/`: rules, recorder, task adapters, tree and live runners.
- `scripts/`: entry points (`run_tree.py`, `run_checks.py`, `analyze.py`).
- `slurm/`: batch scripts. All model, physics and analysis work runs under
  Slurm; scripts refuse to start without `SLURM_JOB_ID`.
- Outputs: `/mnt/data/nhatnc129/jepa/cem_stopping/`.
