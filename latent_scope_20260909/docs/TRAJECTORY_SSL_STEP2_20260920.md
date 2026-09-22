# Self-supervised trajectory targets: step 2

Date: 2026-09-20. Requested by the user after comparing trajectory JEPA with a memory-enabled VLA.

## Scope and verdict discipline

This step builds and checks **infrastructure**, not a learned predictor, value model,
representation-adequacy result, selection-headroom estimate, or controller. A numerical
Chen identity pass is an implementation check, not evidence of a method advantage.

All new code lives in `latent_scope_20260909/trajectory_ssl/`. Historical `comp_pilot/`
code, checkpoints and reports remain unchanged to preserve reproducibility. The new
temporal convention is deliberately versioned and must NOT be silently combined with
old trained memory modules. Reusing an old backbone/teacher that learned privileged
contact labels would invalidate a fully self-supervised training claim.

No GR00T fine-tuning, new arena installation, physics, rendering or GPU allocation is
authorized/started by this step. Native Scrub success is not replaced by reference
trajectory tracking in any reported result. The deployed objective remains to be chosen
before a new planning experiment.

## Corrected temporal convention

- Observation `o[r]` occurs before action `a[r]`; history at time `t` contains
  observations through `t` and actions strictly before `t`.
- A sampled history frame `o[r]` is paired with the incoming action `a[r-1]`.
- At reset, the absent incoming action is zeroed and has an explicit validity mask.
- A proposed chunk is `a[t:t+H]`, future observations are `o[t+1:t+H+1]`,
  and the signature path includes **both** boundaries `o[t:t+H+1]`.
- For recurrent updates, future frame index `i` is paired with action index `i`,
  not `i+1` and not an end-of-chunk clamped next action.
- The stride grid has a carried offset. Splitting at non-multiples of the stride
  must not change which frame/action pairs enter memory.
- Current sampling takes one incoming action at each sampled frame, matching the
  old sparse-memory input budget. It does not claim to summarize all intervening
  actions; a future complete-history baseline may encode those action blocks too.

`temporal.py` is the corrected window builder and memory adapter for subsequent
experiments. The tests include candidate-independent prefixes and an order-sensitive
GRU whose final state is invariant to arbitrary partitioning of the same observed data.
This addresses the identified convention bug; its impact on old method scores has NOT
been measured, and no old negative verdict is reversed.

## Fixed target construction

Use the existing 20 Hz DINOv2-S 4x4 patch-grid caches, never a grounded checkpoint.
Select exactly 32 train and 16 validation episodes by seeded sampling of manifest IDs,
without accessing contact/success labels. Old validation remains development data.
Drop test entries before loading any episode payload.

The bounded diagnostic uses four fixed random-projected visual channels (flattening
preserves camera/patch indices before projection) and four fixed random-projected
proprio channels. This is an arbitrary small plumbing configuration, **not evidence
that eight channels suffice**, and is not a learned spatial model. Affine statistics
are fitted only on the selected train episodes, then frozen for all windows/splits.

For projected observation features `f(t)`, construct the piecewise-linear lifted path

    X(t) = (t, integral f(s) ds)

with 0.05-second intervals and trapezoidal integration of sampled features. This is a
declared observation-derived target, not a contact count or physical coverage estimate.
No action enters this target; actions are reserved as predictor inputs. The integrated
lift retains level/occupation information that pure feature increments would discard.
It does not guarantee useful task semantics, full path recovery, or rate invariance
under undersampling. No per-chunk duration normalization is applied.

Compute tensor signatures through degree 3, in fp64 for auditing. With nine lifted
channels the stored target has 9 + 81 + 729 = 819 nonconstant coefficients; level zero
is implicitly one. Degree 2 is the first 90 coefficients. The fixed truncated Chen
product, not a Transformer, composes them. Logsignatures are not implemented.

Targets use original-cadence windows of 32/64/128 transitions at stride 16. Prefix
signatures and their algebraic inverses accelerate interval extraction; independent
direct computations verify representative windows. Every saved window is also checked
against a non-midpoint split. Comparisons use `atol=rtol=1e-8` in raw coordinates.

Raw coordinates are saved. Separate train-only target loss statistics are supplied
for later predictor training; any whitening MUST be inverted before Chen composition.
The same projection/normalization/version must be used for both constituent segments.
Predicted expected signatures need not be realizable single-path signatures; sampling,
conditional dependence and nonlinear scoring require a separate model design.

## Verification and outputs

CPU unit tests cover:

- analytic straight-line and noncommuting x-then-y paths;
- every split, degrees 1/2/3, three-part associativity, empty-interval identity;
- inverse/reversal and prefix-based interval extraction;
- shared boundaries, translation and collinear subdivision;
- differentiability via finite-difference gradient checking;
- candidate-independent causal history and reset masks;
- recurrent state equality across aligned and non-aligned partitions.

Configuration: `configs/trajectory_ssl_step2.json`.
Submission: `scripts/slurm_trajectory_ssl_step2.sh`, CPU only, four CPUs, 16 GB,
hard 20-minute cap and internal 15-minute target-building cap. Tests gate target
construction. Each job snapshots source/config and checksums into its own directory.
Do not execute either tests or feature processing on the login node.

Job **53273** was submitted after both `squeue` and `sacct` showed no competing user
jobs. Output directory:

    /mnt/data/nhatnc129/jepa/latent_scope_trajectory_ssl/step2_53273/

Artifacts: `tests.json`, `result.json`, `projection.pt`, `target_loss_stats.pt`,
`manifest.json`, per-episode `targets/*.pt`, and `code/SHA256SUMS`.

Positive technical verdict: `STEP2_PLUMBING_PASS_NOT_METHOD_EVIDENCE`.
If it passes, the next experiment is a matched target/predictor comparison, not a
claim that signature JEPA beats GR00T or a request to launch full training automatically.

## Result: job 53273

Verified after execution using both `squeue` (no entry) and `sacct` (COMPLETED,
00:00:15, exit 0:0; allocation four CPUs, 16 GB, no GPU).

- **14/14 unit tests passed** in 0.864 seconds.
- Target construction/audit took 8.748 seconds inside the job.
- **48 episodes / 3,225 windows** processed, 819 target coefficients each.
- Maximum absolute discrepancy across split and direct/prefix-inversion audits:
  **1.2150280781497713e-11**, within the declared fp64 tolerances.
- No task labels, grounded model checkpoints or test episodes loaded.
- Verdict: **STEP2_PLUMBING_PASS_NOT_METHOD_EVIDENCE**.

This establishes only the corrected interface and exact sampled-path algebra. The
projection is still fixed/random, no learned summary or action-conditioned predictor
was trained, and no memory-VLA or JEPA control comparison has been run.
