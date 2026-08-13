# Probe-Frozen Controllability Geometry: locked Stage-M0 pilot

Date: 2026-08-11. Status: **complete; locked verdict NO-GO**.

## Question and scope

Historical Stage 0 reported that the released LeWM latent terminal cost had
residual physical selection regret even at rendered simulator endpoints. This
pilot tests a necessary condition for a replacement method; the evaluator
caveat discovered during execution is recorded below.

> Can an action-response geometry computed only from the frozen world model
> rerank the exact Stage-0 final CEM candidates better than latent L2, without
> receiving simulator state, reward, success, or physical distance?

This is a reranking pilot, not a full PFCG-guided CEM evaluation. A positive
result licenses closed-loop implementation; a negative result stops the method
before a larger experiment.

## Method locked before results

At each persisted Stage-0 snapshot, draw 32 Gaussian action sequences
`u_k ~ N(0, I)` in the official normalized action space and include their 32
negations. The probe seed is `20260811 + 50000 + snapshot`. Roll all probes
through the frozen released LeWM predictor. For terminal embeddings, form

```
r_k = 0.5 * (z_H(+u_k) - z_H(-u_k)).
```

After centering the responses, compute their SVD. Retain every response mode
whose empirical Gramian eigenvalue is at least `1e-6` of the largest mode. The
PFCG cost for candidate endpoint `z` and goal `g` is

```
c(z, g) = sum_j <u_j, z-g>^2 / (lambda_j + ridge),
ridge = 0.1 * median(retained lambda_j).
```

The probe geometry is computed before candidate evaluation and held fixed. No
hyperparameter sweep is allowed on the locked 32 snapshots.

## Identical candidate population

Use only the persisted final CEM populations from Stage 0:

`$STABLEWM_HOME/artifacts/audit_locked_array/snapshot_NNN_final.npz`.

The method does not rerun or alter CEM. It first computes all deployable
selector costs from the checkpoint, observation, goal, and stored action
sequences. Only after those choices are fixed does MuJoCo replay the actions for
offline physical evaluation. Recomputed learned L2 must match the Stage-0
artifact or the shard fails.

### Pre-result evaluator amendment (2026-08-11)

Smoke jobs 38133--38135 exposed that the Stage-0 final-population outcome
artifact is not reliably bitwise reproducible: two unchanged processes on the
same H100 worker produced worst physical differences of 5.06 mm and 0.35
micrometres respectively. Stage 0 verified qpos/qvel/pixel restoration before
dynamics, but its restore seam did not persist full dynamic `mjData` state.

Before inspecting any PFCG outcome, the pilot evaluator was amended as follows.
Each final-protocol shard compiles one `World`, disables solver warm-start, and
replays all 300 stored action sequences twice, resetting before every candidate.
The shard fails unless success and executed steps agree exactly and physical
distances agree within `1e-5` m. All selectors are evaluated on the first
repeated replay. Stored Stage-0 physical outcomes are reported as reproduction
diagnostics but are no longer gates. Candidate actions and the model-predicted
learned cost remain artifact-locked.

Job 38136 then showed that physical dynamics passed this gate while separately
rendered endpoint images differed by up to 38 intensity levels. Before any
selector outcome was produced, all privileged true-endpoint encoding arms were
therefore removed. The final pilot uses no simulator rendering at all: MuJoCo
returns only physical distance, success, and executed steps after deployable
selections are frozen. The method formula, probe seed, ridge, rank threshold,
and deployable comparators did not change. The decision rule below was reduced
to deployable arms because the removed true-endpoint criterion was not
measurable reproducibly.

The first locked array exposed the remaining restore problem directly:
snapshots 3, 6, and 9 failed repeat replay by 5.0 mm, 35.6 mm, and 0.8 mm, so
the array was cancelled before aggregation. Targeted jobs 38177/38178 still
failed after visual variations were disabled. Inspection of upstream then
found that `CubeEnv.initialize_episode` executes two random actions to build a
goal observation and restores only qpos/qvel afterward, leaking controller and
contact-solver state into candidate rollouts.

The final evaluator calls `mujoco.mj_resetData` directly before every candidate,
then restores dataset qpos/qvel and target, calls `mj_forward`, and disables
`mjDSBL_WARMSTART` for the same-state audit. Both repeated populations use one
compiled `MjModel`; a second full compile/run is required only if the pilot
reaches a go verdict. All v1 outputs remain excluded from the fresh v2 root.
This changes only hidden measurement: no selector result was aggregated or
inspected and no method/decision parameter changed.

## Locked comparators

- predicted-endpoint latent L2;
- PFCG on predicted endpoints (primary deployable selector);
- unwhitened projection onto the same response subspace;
- coordinate-wise response whitening;
- a seed-fixed random subspace with matched rank and eigen spectrum;
- physical best-in-population oracle, evaluation only.

The unit of uncertainty is the snapshot. All comparisons are paired within the
same 300 candidates and use 10,000 snapshot-bootstrap draws in the final
aggregate.

## Precommitted decision rule

Let regret gain be latent-L2 selected physical distance minus PFCG-selected
physical distance. Let success gain be PFCG success minus latent-L2 success.

- **Strong go:** predicted-endpoint regret-gain and success-gain lower 95% CIs
  are both above zero, and predicted PFCG beats the projected and matched-random
  controls in mean regret.
- **Conditional expand:** predicted regret-gain lower CI is above zero,
  success-gain mean is positive but its CI includes zero, and PFCG is not worse
  in mean regret than the projected control. Run a fresh locked n=128
  confirmation before closed-loop search.
- **No-go:** predicted regret-gain lower CI is not above zero, mean success
  does not increase, or a simple projected control has lower mean regret. Do
  not rescue the method with a post-hoc rank/ridge sweep on these snapshots.

## Fairness and claim boundary

The deployable PFCG selector uses only the frozen model, current/goal images,
action limits, and internally predicted action-probe responses. Simulator
rollouts are a hidden evaluator after selection and return physical scalars
only; simulator images are neither encoded nor exposed to the method.

Even a strong go would establish only same-population reranking. A method paper
would still require PFCG-guided closed-loop search, direct TRM and proposal
baselines, multiple model families/tasks, non-CEM replication, and a linear-case
minimum-control-energy/invariance result.

## Final decision

The locked aggregate is **NO-GO**. PFCG selected 14/32 successes versus 16/32
for latent L2 and increased mean physical selection regret by 0.46 cm. Both the
same-subspace projection and matched-random control had lower mean regret; the
random control also beat PFCG with paired 95% CIs excluding zero for both regret
and success. No rank/ridge/probe sweep or larger PFCG run is licensed. Full
numbers and the Stage-0 claim boundary are in
`../../results/ogb_pfcg/PFCG_PILOT_DECISION.md`.

## Execution ledger

| Job | Command | Dependency | Output | State |
|---|---|---|---|---|
| 38133 | `sbatch diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | none | `diagnosis/results/ogb_pfcg/smoke/`; `/mnt/data/nhatnc129/jepa_runs/logs/ogb_pfcg_smoke_38133.out` | **FAILED** after 4:00: learned-cost gate passed; true-endpoint L2 max-abs mismatch 8.1624 before physical gate was reached |
| 38134 | `sbatch diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | none | no result (gate fired before diagnostic write); `/mnt/data/nhatnc129/jepa_runs/logs/ogb_pfcg_smoke_38134.out` | **FAILED** after 3:56: physical-distance max-abs mismatch 0.005063 m |
| 38135 | `sbatch diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | none | `diagnosis/results/ogb_pfcg/smoke/reproduction_debug.npz`; `/mnt/data/nhatnc129/jepa_runs/logs/ogb_pfcg_smoke_38135.out` | **FAILED** after 4:00 at intentionally exact physical gate; diagnostic max-abs 0.35 micrometres, no success disagreement |
| 38136 | `sbatch diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | none | no accepted result; `/mnt/data/nhatnc129/jepa_runs/logs/ogb_pfcg_smoke_38136.out` | **FAILED** after 7:33: physical repeat gate passed, but removed true-endpoint renderer arm differed by 38 pixel levels across independent worlds |
| 38148 | `sbatch diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | none | `diagnosis/results/ogb_pfcg/smoke/`; `/mnt/data/nhatnc129/jepa_runs/logs/ogb_pfcg_smoke_38148.out` | **COMPLETED** in 2:28: independent replay max 2.21 micrometres, exact success/steps, learned-cost error 2.81e-6 |
| 38156_[0-31] | `sbatch diagnosis/scripts/slurm_ogb_pfcg_array.sh` | completed smoke 38148 | invalid first-array shards under `diagnosis/results/ogb_pfcg/locked_shards/`; logs `ogb_pfcg_array_38156_%a.out` | **CANCELLED** 2026-08-11 after tasks 3/6/9 failed the replay gate; completed shards excluded from all analysis |
| 38159 | `sbatch --dependency=afterok:38156 diagnosis/scripts/slurm_ogb_pfcg_aggregate.sh` | invalid array 38156 | none | **CANCELLED** before execution; no partial aggregate |
| 38175 | `sbatch diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | none | `diagnosis/results/ogb_pfcg/smoke/`; `/mnt/data/nhatnc129/jepa_runs/logs/ogb_pfcg_smoke_38175.out` | **COMPLETED** in 2:29: independent replay max 0.394 micrometres with deterministic empty-variation reset |
| 38177 | `sbatch --export=ALL,PFCG_SNAPSHOT_INDEX=3,PFCG_OUT_TAG=smoke_v2_3 diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | smoke 38175 | `diagnosis/results/ogb_pfcg/smoke_v2_3/`; log `ogb_pfcg_smoke_38177.out` | **FAILED** replay gate: max physical difference 31.7 micrometres |
| 38178 | `sbatch --export=ALL,PFCG_SNAPSHOT_INDEX=6,PFCG_OUT_TAG=smoke_v2_6 diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | smoke 38175 | `diagnosis/results/ogb_pfcg/smoke_v2_6/`; log `ogb_pfcg_smoke_38178.out` | **FAILED** replay gate: max physical difference 42.5 mm |
| 38182 | `sbatch --export=ALL,PFCG_SNAPSHOT_INDEX=3,PFCG_OUT_TAG=smoke_v4_3 diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | failed targeted smoke 38177 | `diagnosis/results/ogb_pfcg/smoke_v4_3/`; log `ogb_pfcg_smoke_38182.out` | **COMPLETED** in 41s: repeat physical/success/steps exactly identical |
| 38183 | `sbatch --export=ALL,PFCG_SNAPSHOT_INDEX=6,PFCG_OUT_TAG=smoke_v4_6 diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | failed targeted smoke 38178 | `diagnosis/results/ogb_pfcg/smoke_v4_6/`; log `ogb_pfcg_smoke_38183.out` | **COMPLETED** in 36s: repeat physical/success/steps exactly identical |
| 38185_[0-31] | `sbatch diagnosis/scripts/slurm_ogb_pfcg_array.sh` | targeted hard-case smokes 38182/38183 | `diagnosis/results/ogb_pfcg/locked_v2_shards/$TASK_ID/`; logs `ogb_pfcg_array_38185_%a.out` | submitted fresh locked v2 array 2026-08-11; code hashes match targeted smoke |
| 38188 | `sbatch --dependency=afterok:38185 diagnosis/scripts/slurm_ogb_pfcg_aggregate.sh` | all v2 tasks 38185 | `diagnosis/results/ogb_pfcg/locked_v2/`; log `ogb_pfcg_aggregate_38188.out` | submitted with `afterok` dependency |
| 38190_[0-31], 38191 | duplicate array and dependent aggregate submissions | duplicate of 38185/38188 | none | **CANCELLED** before any array task ran to prevent concurrent writes to the same v2 output root |
| 38179 | `sbatch --export=ALL,PFCG_SNAPSHOT_INDEX=6,PFCG_OUT_TAG=smoke_v2_6 diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | redundant with 38178 | same target as 38178 | **CANCELLED** while pending to prevent concurrent writes |
| 38180 | `sbatch --export=ALL,PFCG_SNAPSHOT_INDEX=3,PFCG_OUT_TAG=smoke_v4_3 diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | full restore fix | `diagnosis/results/ogb_pfcg/smoke_v4_3/`; log `ogb_pfcg_smoke_38180.out` | **COMPLETED** in 43s; exact-zero repeat error for physical/success/steps |
| 38181 | `sbatch --export=ALL,PFCG_SNAPSHOT_INDEX=6,PFCG_OUT_TAG=smoke_v4_6 diagnosis/scripts/slurm_ogb_pfcg_smoke.sh` | full restore fix | `diagnosis/results/ogb_pfcg/smoke_v4_6/`; log `ogb_pfcg_smoke_38181.out` | **COMPLETED** in 37s; exact-zero repeat error for physical/success/steps |
| 38185_[0-31] | `sbatch diagnosis/scripts/slurm_ogb_pfcg_array.sh` | hard-snapshot smokes 38180/38181 | `diagnosis/results/ogb_pfcg/locked_v2_shards/$TASK_ID/`; logs `ogb_pfcg_array_38185_%a.out` | **COMPLETED**: 32/32 shards passed full-state repeat gate |
| 38188 | `sbatch --dependency=afterok:38185 diagnosis/scripts/slurm_ogb_pfcg_aggregate.sh` | all v2 array tasks | no combined output; log `ogb_pfcg_aggregate_38188.out` | **FAILED** in 1s: undefined Spearman on constant-physical snapshots was not filtered |
| 38223 | `sbatch diagnosis/scripts/slurm_ogb_pfcg_aggregate.sh` | completed v2 array | `diagnosis/results/ogb_pfcg/locked_v2/`; log `ogb_pfcg_aggregate_38223.out` | **COMPLETED** in 1s; locked verdict `no_go` |
| 38225 | `sbatch diagnosis/scripts/slurm_ogb_pfcg_aggregate.sh` | completed v2 array | same locked-v2 output; log `ogb_pfcg_aggregate_38225.out` | **COMPLETED** in 1s; added explanatory projected/random-vs-L2 contrasts, verdict unchanged |
