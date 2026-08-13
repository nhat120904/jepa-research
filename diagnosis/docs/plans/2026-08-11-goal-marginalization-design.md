# Goal-marginalization pilot: arm-nuisance in the L2 terminal cost (pre-registered 2026-08-11)

Status: **design locked, not yet run**. This is a bounded two-gate feasibility
sprint testing the Idea-A hypothesis from the 2026-08-11 novel-methods
brainstorm: that the upstream terminal cost `‖z_H − z_goal‖²` fails on contact
tasks because most of its squared-error budget is spent matching the arm/
gripper configuration of the goal frame, not the object configuration, and
that this is a *quotient/weighting* defect, not an information defect (the
object is already linearly decodable off the frozen latent at 2.01 cm / 91.5%
<5cm, per `results/encoder_info_upperbound.md`).

Both gates reuse only validated primitives already in the repo
(`make_env`, `rollout_expert`, `snapshot`/`restore`, `render`, `encode_frame`,
`encode_batch`, `cem_plan_latent`, `build_oracle_cost` from scripts/18/29/30)
and touch no MuJoCo internals directly — no manual `qpos` indexing, since the
exact free-joint offset for the object body is not established in this repo
and guessing it risks silently corrupting renders. All new state variation is
produced by **stepping the real simulator**, never by hand-writing physics
state.

All GPU/simulator work runs on a compute node via `sbatch`; the login node is
used only for editing and job monitoring (`squeue`/`sacct`), per `CLAUDE.md`.

## Gate 1 — Task-level SNR law (cheap, first)

### Question

Does a single scalar computed from the expert trajectory — the fraction of
total goal-latent displacement that occurs strictly *after* the last
pre-contact frame — predict which tasks the plain `l2` oracle arm solves?

### Definition

For each episode, using the SAME expert rollout already used to build the
goal frame (`rollout_expert`), classify every transition with
`stratification.metaworld_regimes.classify_metaworld_regime` on the raw
39-dim states (already the repo's contact proxy — object displacement over
5 mm). Let `t*` be the **last** transition index classified `free_space` or
`pre_grasp` (i.e. the last moment before sustained contact/gripper actuation
begins). Encode three frames with the real frozen encoder: the episode start
(`t=0`), the pre-contact frame (`t=t*`), and the expert's final/goal frame
(`t=T`). Define

```
SNR = ‖z_T − z_t*‖ / ‖z_T − z_0‖
```

`z_T − z_0` is the total goal-latent displacement the L2 cost has to resolve;
`z_T − z_t*` is the portion of that displacement that happens during and
after contact. A task where the arm alone can walk the latent most of the way
to `z_goal` before contact starts (reach, drawer-close: the gripper closing
motion IS the task) has low `1 − SNR`... **wording fixed below to avoid
sign confusion**:

- `SNR` close to **1** → almost all goal-latent movement is downstream of
  contact → contact-relevant movement dominates the cost → predicted **success**.
- `SNR` close to **0** → almost all goal-latent movement already happened by
  `t*`, pre-contact (arm transport) → the cost is front-loaded on arm motion
  the planner can match trivially → predicted **failure**.

This is a diagnostic correlation, not a new metric contribution — CRA/AUG/ECS/
CTD and boundary-blindness already exist for that. It exists only to gate
whether Idea A is worth building.

### Protocol

- Tasks: the 9 with a usable positive control — `mw-reach`, `mw-push`,
  `mw-pick-place` (original 3) plus the 6 task-breadth-ladder tasks whose
  `l2`/`oracle` numbers are already on disk with oracle `success_end ≥ 8/16`
  (excludes `mw-door-open` 2/16, `mw-assembly` 0/16, `mw-box-close` 0/16,
  `mw-shelf-place` 2/16, `mw-lever-pull` 6/16 as weak/failed positive
  controls, matching the discipline already applied to door-open/assembly in
  `CURRENT_STATUS.md`): `mw-button-press`, `mw-drawer-close`,
  `mw-window-close`, `mw-faucet-open`, `mw-plate-slide`, `mw-soccer`.
- Model: `dino_wm_metaworld` only. The shared-encoder finding
  (`CURRENT_STATUS.md` §"Shared-encoder finding") already established that
  `dino_wm`/`jepa_wm` give bit-identical L2 geometry under exact dynamics
  since the predictor is never called; running both would not add an
  independent data point.
- Seeds: fresh block `80000..80015` (16 episodes/task), disjoint from every
  seed range used elsewhere in the project.
- Regime thresholds: the stratification module's defaults, unchanged
  (`GRIPPER_DELTA_THRESHOLD=0.10`, `OBJECT_MOVE_THRESHOLD=0.005`,
  `PRE_GRASP_DISTANCE=0.10`). No sweep.
- Episodes where every transition classifies as `free_space` (no `t*` before
  the very last step — the episode never leaves free space, e.g. a task
  that "succeeds" without visible contact) are recorded with `t*=T-1` and a
  `no_contact_detected=True` flag; they are not dropped, but Gate 1's
  analysis reports the count and is not allowed to hide it.

### Readout and decision rule

Per-task SNR is the across-episode mean. The label is each task's already-
measured `l2` oracle `success_end` rate (task-breadth ladder CSVs on disk,
seeds 70000/71000) plus, for push/pick-place/reach, the original confirmatory
`l2` numbers. Because success is saturated at {0/16, 16/16} for all but one
task (drawer-close), the primary statistic is **Spearman correlation between
per-task mean SNR and per-task mean `l2`-arm final object-goal distance**
(continuous, not saturated), with success rate reported as a secondary,
sign-consistent check. CI is a task-level bootstrap (9 tasks is a small n;
report the point estimate and CI honestly, do not oversell significance).

- **Pass → proceed to interpret Gate 2 as informative of a general mechanism:**
  Spearman rho sign matches prediction (higher SNR → lower final distance)
  with |rho| ≥ 0.6, and this is not driven by a single outlier task (rank
  correlation with the extreme task removed keeps the same sign).
- **Fail:** rho is near zero, wrong-signed, or driven by one task. Gate 2 can
  still run (it is a causal test, not dependent on Gate 1 passing), but a
  Gate-1 failure means a positive Gate-2 result would be a push-specific
  finding, not evidence of the general task-SNR law, and the paper framing
  must say so.

Gate 1 does not license any method claim by itself — it is a correlational
sanity check for whether the mechanism story is worth the Gate 2 compute.

## Gate 2 — Causal arm-marginalization pilot (mw-push, primary)

### Question

If the goal latent is replaced by an estimate that averages out arm/gripper
pose variation while holding the object at its goal position — constructed
by actually perturbing the arm in simulation, not by hand-editing state —
does the plain `l2` oracle-latent CEM (perfect dynamics, real frozen encoder,
scripts/30 `--cost l2`) stop failing on `mw-push`?

### Three arms, same episode, same encoder, same CEM budget

For each episode: build the env, roll the scripted expert to completion
exactly as `scripts/29`/`30` already do, and snapshot the resulting state
(`snap_goal`). This is the SAME state that produces the standard goal frame.

1. **`baseline`** — `z_goal` = encode(expert's final frame). Identical to the
   existing `l2` oracle arm; this row must reproduce the known push 0/16.
2. **`arm_marginalized`** — restore to `snap_goal` `K=8` times; from each
   restore, step the simulator forward `n_pert=10` raw steps (2 model-steps)
   of i.i.d. small random actions (`a ~ clip(N(0, 0.15), -0.3, 0.3)` in raw
   action units, well below the `[-1,1]` range CEM candidates use, so the
   perturbation moves the arm/gripper locally without re-opening a completed
   grasp or driving the object). Render+encode each resulting frame; average
   the `K` latents. **Instrumented, not assumed:** record the object
   displacement `‖obj_k − obj_goal‖` for every `k`; if the across-episode
   median exceeds a precommitted `1 cm` (a fifth of the 5 cm success radius)
   the episode's `arm_marginalization_clean` flag is set False and it is
   still run and reported, but excluded from the primary contrast (this
   mirrors the honesty discipline in the PFCG pilot's evaluator amendments —
   report what happened, do not silently assume the intervention was clean).
3. **`noise_matched_control`** — same `K=8` averaging, but the `K` latents
   are `z_goal_baseline + eps_k` with `eps_k ~ N(0, sigma²)`, `sigma` set
   per-episode to the empirical RMS of the arm-marginalized arm's `K`
   residuals around their own mean (so both arms inject an *identical total
   noise energy* into the average — the only difference is whether the noise
   is structured by physically re-simulating arm motion or is unstructured
   isotropic noise added directly in latent space). This is the control that
   separates "averaging/shrinkage helps generically" from "removing the
   arm-specific nuisance direction helps": if `noise_matched_control` matches
   `arm_marginalized`, the mechanism is generic shrinkage, not arm-nuisance
   removal, and the paper's causal story is wrong.

All three arms then run the **unmodified** `cem_plan_latent`/`run_episode`
closed loop (horizon 6, 100 samples, 6 iterations, elite frac 0.1, var0 1.0,
3 model-steps executed per replan, 100 max environment steps, strict
end-of-episode success) — the only thing that varies across arms is which
`z_goal` tensor the `l2` cost function closes over. Perfect (oracle) latent
dynamics throughout, matching `scripts/30`, so no learned-dynamics error can
explain any difference between arms.

### Fixed scope

- Task: `mw-push` only for the primary gate (the task where the existing
  ladder shows plain `l2` at a clean 0/16 with an already-solved 16/16
  positive control). `mw-pick-place` is a secondary confirmation only if the
  primary gate passes.
- Model: `dino_wm_metaworld`. Same shared-encoder justification as Gate 1.
- Seeds: fresh block `90000..90015` (16 episodes), disjoint from Gate 1 and
  every prior pilot's seed range.
- Perturbation RNG: `stable_seed(episode_seed, arm_variant_index)` (blake2b
  keyed, same pattern `scripts/64_planner_induced_cost_pilot.py` and the PFCG
  design use) so every arm's noise draws are reproducible from the episode
  seed alone.
- No hyperparameter sweep on `K`, `n_pert`, or the perturbation scale on the
  locked 16 episodes. If Gate 2 is promising, a confirmation run may sweep
  these on a fresh seed block, not on the seeds used for the decision.

### Locked comparators and decision rule

Primary endpoints: strict `success_end` and final object-goal distance,
paired within episode across the three arms (same seed, same expert
trajectory, same CEM budget — only `z_goal` differs). Bootstrap is an
episode-seed cluster bootstrap (`metrics/bootstrap.py:bootstrap_ci`,
`n_resamples=10000`), matching the repo's standard.

- **Go:** `arm_marginalized` success_end is higher than `baseline` with the
  paired-difference lower 95% CI above zero, **and** `arm_marginalized`
  outperforms `noise_matched_control` in mean success (rules out the generic-
  averaging confound). A go licenses building the full quotient-projection
  method (Idea A) as a real method chapter: a learned `Π` trained to remove
  the same self-attributable variation, evaluated on the full task ladder and
  a fresh confirmation seed block.
- **Conditional / mechanism-only:** `arm_marginalized` beats `baseline` but
  does not clearly beat `noise_matched_control` (CIs overlap). This says
  averaging under search helps for a reason not yet isolated as
  arm-specific; do not claim the arm-nuisance mechanism. Worth a `K`/scale
  ablation before deciding on a method direction.
- **No-go:** `arm_marginalized` does not beat `baseline` (CI includes zero
  or is negative), or `arm_marginalization_clean` is False for more than
  half the episodes (the intervention itself was not clean, so the test
  is uninformative and must be redesigned, not reinterpreted). A no-go here
  is informative: it would mean the arm-nuisance story is wrong even though
  the object is linearly decodable, and the field's whole "temporal-distance
  weighting" lane (RC-aux, TD-JEPA, TRM, temporal straightening) should be
  expected to fail the same way at contact, since none of them test this
  specific causal intervention.

### Novelty boundary

Goal averaging/marginalization over nuisance variation is not itself novel
(data augmentation, domain randomization, and multi-goal averaging all exist).
The candidate contribution, gated on a Go verdict, is specific: identifying
that the terminal-cost failure at contact is explained by an **arm-pose
nuisance subspace that dominates the L2 budget**, demonstrated by a **causal,
simulator-grounded, structure-vs-noise-matched intervention** (not a
correlational readout-based diagnosis, which the repo has already tried and
which CEM was shown to exploit — `results/exploitation_gap_ladder.md`), and,
if it survives, a learned Π that removes this subspace without any GT-object
regression target (unlike `gobj`/`stateprobe`/`phi`, all of which train a
scalar/vector readout that CEM's search then targets and exploits). This
pilot only tests necessity of the mechanism; it does not implement Π.

## Execution ledger

To be filled in as jobs are submitted. All commands run from `diagnosis/`.

| Job | Wrapper | Dependency | Output | State |
|---|---|---|---|---|
| 38703 | `sbatch scripts/slurm_task_snr_smoke.sh` | none | `results/task_snr_smoke.csv` | **SUBMITTED** 2026-08-11, queued (QOSMaxGRESPerUser) |
| (pending) | `scripts/slurm_task_snr_pilot.sh` | 38703 exits 0 | `results/task_snr_pilot.csv` | not yet submitted |
| (pending) | `scripts/slurm_task_snr_analyze.sh` | pilot completes | `results/task_snr_law.md` | not yet submitted |
| 38704 | `sbatch scripts/slurm_goal_marginalization_smoke.sh` | none | `results/goal_marg_smoke.csv` | **SUBMITTED** 2026-08-11, queued (QOSMaxGRESPerUser) |
| (pending) | `scripts/slurm_goal_marginalization_pilot.sh` | 38704 exits 0 | `results/goal_marginalization_mw-push_seed90000_n16.csv` | not yet submitted |
| (pending) | `scripts/slurm_goal_marginalization_analyze.sh` | pilot completes | `results/goal_marginalization_report.md` | not yet submitted |

Only the two smoke jobs (38703, 38704) were submitted. The full pilots and both
analysis jobs are intentionally held until each smoke job is inspected —
GPU allocation on this cluster is quota-limited (`QOSMaxGRESPerUser`), so
launching everything at once would only queue-block other users' jobs
without buying anything, since a smoke failure would need the full run
re-submitted anyway.
