# Job ledger

All job IDs, exact commands, dependencies, outputs, and terminal states are
recorded here immediately after submission and reconciled with both `squeue`
and `sacct`.  The protocol must not be edited to follow observed results.

## Stage 1 (no-training diagnostic)

| UTC | Job ID | Stage | Exact command/config | Dependency | Output | State |
|---|---:|---|---|---|---|---|
| 2026-08-24 | `45646` | smoke | `sbatch action_curvature_h0/scripts/slurm_00_smoke.sh` | none | none | **CANCELLED while PENDING, never ran, no scientific result.** Cancelled after an external review found six confirmed code defects and one protocol defect (below). It would also have died at its first command: `pytest` is not installed in `$STAGE0_ROOT/.venv`, and the script runs under `set -euo pipefail`. |

### Defects confirmed before any measurement (2026-08-24)

Verified directly against the code, not accepted on assertion:

1. **Scaling-exponent fit is invalid.** `draw_directions` is called inside the
   `sigma` loop (`measure_curvature.py:340,347,358`), so `direction=k` at one
   sigma and `direction=k` at the next are different random vectors, while
   `write_outputs` pairs records by `(horizon, direction)` across sigmas to fit
   `alpha`. The slope is therefore not a scaling exponent along a fixed
   direction. Readout 6 and every scale-response reading are void; per-record
   `E_K` is unaffected.
2. **Clip validity is computed on the full `--horizon` chunk before truncation**
   (`measure_curvature.py:459-467`), so a valid short-horizon triplet is
   discarded for a bound violation in tail actions that are never executed.
3. **Smoke cannot exercise its own checklist**: two sigmas against
   `min_points=3`, so `alpha` is NaN by construction.
4. **`H=10`**: the protocol asks for it but the frozen planner configuration is
   `PlanConfig(horizon=5, action_block=5)` and the cached populations are
   `(2, 96, 5, 25)`. H=10 is not reachable without changing the released
   planning configuration; the protocol, not the array, is what is wrong.
5. **`R^2` is computed and stored but never gates anything** in `aggregate.py`,
   contrary to the protocol's fit-quality rule.
6. **Evaluations are float32** (`measure_curvature.py:206`); casting to float64
   afterwards does not recover precision lost in the forward pass, which
   weakens the cancellation guard the protocol claims.
7. **Aggregate thresholds are undeclared.** `aggregate.py:57` states the 0.25
   cut-points are "declared in PROTOCOL.md"; they are not in PROTOCOL.md.
   Also in the verdict path: `max()` over CEM sources is a selection bias, the
   comparison uses point estimates with no paired CI, horizons/sigmas/sources
   are pooled into one median, `alpha` is counted once per sigma record rather
   than once per fit, and `all(... if mode in smooth_alpha)` is vacuously true
   when no usable same-mode alpha exists.

| 2026-08-24 | `45719` | offline numerics | `sbatch action_curvature_h0/scripts/slurm_test.sh`; CPU-only, runs `tests/test_core.py` standalone plus `aggregate.py --self-test` | none | log `/mnt/data/nhatnc129/jepa_runs/logs/acm_test_45719.out` | **COMPLETED**, exit `0:0`, `00:00:05` on `worker-1`: `29/29 passed`, aggregate self-test passed |

### Fixes landed and verified (2026-08-24, job `45719`)

Defects 1, 2, 3 and 6 from the list above are fixed; `pytest` is no longer a
dependency of any gate.

- **1 (voided scaling exponent).** `draw_unit_directions` draws one base
  direction per `(snapshot, source)` and `scale_direction` scales that same
  vector across every sigma and horizon.  Both moved from the script into
  `core.py`, because the most bug-prone step was the one part that could not be
  tested offline (the script imports torch).  Two tests lock the property:
  cosine between deltas at every sigma is 1, and the norm scales linearly.
- **2 (clip validity).** Computed on the truncated chunk, so a valid
  short-horizon triplet is no longer discarded for tail-action bound violations.
- **3 (smoke could not exercise its own checklist).** Smoke now sweeps four
  sigmas, and runs `tests/test_core.py` standalone instead of `pytest`, which
  removes the missing-dependency failure at the first command.
- **6 (contact mode).** Replaced the geom-pair union.  Categories now resolve
  from MuJoCo body ids (cube body from `object_joint_0`, world body as static
  scene, anything else as robot) and the run is refused if the cube body owns
  no geoms.  Only cube-robot contact stratifies; cube-table is a covariate,
  because the cube rests on the table for essentially every step and the old
  rule would have left `same_mode_non_contact` empty.  Traces are per step, so
  contact onset is part of the mode signature.

New quantities added at the same time: `E_J` (first-order mismatch, covering
`E_K`'s affine blind spot) and the exact scalar-cost decomposition
`D2 C = 2<r0, v+ - v-> + ||v+||^2 + ||v-||^2` for both the model and the
realized cost, with local concavity flagged by `ratio < -1`.

| 2026-08-24 | `45720` | offline numerics | `sbatch scripts/slurm_test.sh` after the group 5+7 fixes | none | log `acm_test_45720.out` | **COMPLETED**, exit `0:0`, `00:00:02`: `30/30 passed`, aggregate self-test passed |
| 2026-08-24 | `45721` | smoke | `sbatch --dependency=afterok:45720 scripts/slurm_00_smoke.sh` | afterok:45720 | no measurement artifact; log `acm_smoke_45721.out` | **FAILED**, exit `1:0`, `00:00:31` on `worker-0`: `RuntimeError: object_joint_0 resolved to 6 bodies`. No scientific result was produced or consumed. |

### What job 45721 established

The failure is the defensive guard in `build_contact_classifier` firing exactly
as intended. `model.joint("object_joint_0").bodyid` returned **six** entries, so
taking element zero would have selected the wrong body and turned the whole
contact stratification into noise without any visible error.

Fixed by resolving on the canonical index path instead of the named view:
`mujoco.mj_name2id(..., mjOBJ_JOINT, "object_joint_0")` then
`model.jnt_bodyid[joint_id]`, which has unambiguous shape semantics. The
classifier now also returns what it resolved -- cube body id and name, cube geom
ids and names, world-geom count -- and that lands in `summary.json` under
`contact_resolution`, so the next smoke can be checked rather than trusted.

Verified at runtime by this job, before the failure point:

- the frozen `quentinll/lewm-cube` checkpoint loads on an H100 node;
- `corrected.make_world` builds `CubeEnv` and EGL initialises (the
  `/dev/dri/*` permission warnings are non-fatal fallbacks);
- **`raw_env._pinch_site_id` exists** -- the effector accessor guard sits before
  the classifier and passed, closing the one item that had only static
  verification through script 84's call path.

| 2026-08-25 | `45974` | smoke (mig) | moved to `--partition=mig` because `worker-0/1` had all 8 GPUs allocated and `worker-2/3` were `DOWN+DRAIN`; `main` estimated an 18-hour wait | none | none | **FAILED** at `write_outputs`: `touched_bodies` is a list and the npz path forced every field to float64. The measurement had completed and was lost because the derived npz was written before the authoritative JSON. Fixed: JSON first, dtype by inspection, skipped keys reported. |
| 2026-08-25 | `46036` | smoke (mig) | `sbatch --partition=mig --mem=48G scripts/slurm_00_smoke.sh` | none | `outputs/smoke/` | **COMPLETED**, exit `0:0`, `00:01:37` on `worker-mig-3g40gb-0` |

### What job 46036 established

**Feasibility fix works for the dataset source.** 32/32 records valid, `H=1` and
`H=5` both measurable (16 each), masked fraction 4.8%, shrink 1.0. Before the
fix: 8/16 valid, `H=1` only.

**Contact classification verified conclusively, and the earlier ambiguity is
resolved.** `touched_bodies` is `(16, 22)` on every dataset record, and body 22
is `ur5e/robotiq/left_pad` -- a gripper pad. Body 0 is `world` and appears only
for the CEM sources. So the cube is **grasped** at this snapshot; the table was
never being misclassified. The instrumentation added in amendment M answered the
question it was added for.

**Both CEM sources still yield 0/32 valid, for a different reason.**
`direction_feasibility` reports the directions as feasible (masked 1.6%, shrink
1.0), yet every record is clipped. That can only happen if the elite centre
action is itself outside the action box: no perturbation rescues a centre that
is already out of bounds, and `make_feasible` masks the direction there without
being able to fix the centre. CEM samples a Gaussian in normalized space and the
environment clips at execution, so out-of-box elites are expected. Readout 9 is
still unobtainable and needs a decision, not a patch.

**The decisive finding is about the observation path, not the model.** For the
dataset source at `H=5`, fitting the raw second difference against `||d||`:

| quantity | alpha | interpretation |
|---|---:|---|
| object position, from simulator state | **1.86** | smooth, essentially the `delta^2` regime |
| terminal latent, `E o render o sim` | **0.37** | not differentiable at any tested scale |

Across a 160x range of `||d||` the latent second difference moves by 3.5x; a
smooth map would move by 25,600x. Meanwhile the physics is smooth over the whole
sweep. The object displacement spans **4.5e-5 m to 7.1e-3 m**, i.e. roughly
1/100 of a pixel to 1.6 pixels at 224x224. Normalized curvature at the smallest
scale is 2.6 in latent and 1.1e-3 in object position -- a factor of ~2400.

The realized-reference construction, which is the novel core of this program,
therefore has a resolution floor set by the rasterizer plus the frozen encoder,
and this snapshot's entire sigma sweep sits below it. `repeat_floor` is exactly
0, so this is not stochasticity; it is quantization. Amendment L anticipated
this case and specified reporting it rather than working around it.

Not yet established: whether this holds at snapshots where the cube is free
rather than grasped, where the same action perturbation should move it much
further.

### Fourth pre-execution fix chain (2026-08-25, before the probe was trusted)

Two more issues surfaced while diagnosing why `cem_fixed` was still 0/16 valid
on 4 of the 8 probe snapshots (`46123`) despite `direction_feasibility`
reporting `feasible=1.0`:

1. **CEM centre clipping added.** `clip_to_bounds` (core.py) clips a raw action
   chunk into the box and reports how much moved; applied to every source's
   centre right after `resolve_centre`, since a centre outside the box cannot
   be rescued by any perturbation and the simulator executes the clipped action
   regardless.
2. **Diagnostic instrumentation added** (`margin_diagnostics`, `clip_diagnostic`)
   after static algebraic review of the margin/cap/scale_direction/clip_validity
   chain found no error, to get real numbers instead of continuing to reason
   about it. First run (`46138`) crashed at write time with
   `NameError: clip_diagnostic` -- `write_outputs` never received it as a
   parameter, an implementation slip caught immediately by the same instinct
   that added the instrumentation. Fixed by threading it through the signature.
3. **The actual bug**, found by the corrected diagnostic job (`46139`,
   snapshot 2, `cem_fixed`): `max_under_low = 2.220446049250313e-16`, exactly
   `np.finfo(float64).eps` -- one ULP, not a real bound violation. A centre
   clamped exactly onto a bound by `clip_to_bounds` is not bit-identical after
   the normalize/denormalize round trip through the scaler, and `clip_validity`
   used a strict `>=`/`<=` comparison, rejecting every triplet built on that
   centre at every sigma (a property of the centre, not of any given delta).
   Fixed with `atol=1e-9` in `clip_validity` -- far above float64 roundoff, far
   below the smallest sigma this protocol uses (1.25e-3). Verified with numerics
   job `46141`: `38/38 passed`.

All three fixes were caught by instrumentation or by running the numerics gate
immediately after each edit, not by static reasoning alone -- the algebraic
review of the clip chain was correct and did not find this bug; only real
numbers did.

Still open before any measurement: defects 4 (`H=10` unreachable under the
frozen `PlanConfig(horizon=5)`), 5 (`R^2` gates nothing), 7 (undeclared
thresholds, unpaired comparison, pooled median, duplicated alpha, vacuous
`all()`), the minimum true-sensitivity gate, the float32 evaluation caveat, and
the Stage-2 reference-matching arm.  **No array job is submitted, and the GPU
and simulator paths of `measure_curvature.py` have still never executed.**

Not yet submitted, gated on the smoke being inspected and passed:

- `slurm_01_measure.sh`, array `0-63%2`, once per `ACM_SOURCE` in
  {`dataset`, `cem_fixed`, `cem_local`} -> `outputs/stage1/<source>/snapshot_XXX/`.
- `slurm_02_aggregate.sh`, which refuses to run unless all 64x3 shards exist.

## Smoke inspection checklist (per PROTOCOL.md, before the array)

1. `floor.repeat_floor` is exactly 0 in every shard; any nonzero value means a
   nondeterministic source must be found and locked first.
2. `k_true_state_effector` is low; a high value indicts the measurement
   pipeline rather than the physics and blocks the run.
3. `discard_by_mode` rates are comparable across strata; a differential rate
   biases the same-mode vs cross-mode comparison and requires a smaller sigma.
4. `scaling_fits` contain finite `alpha` with sensible `r_squared`, and
   `excluded_below_floor` is not consuming the whole sweep.
