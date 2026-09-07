# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`scene_progress_wm/` is one experiment program inside the larger `jepa-research`
repo (git root is `..`). It asks whether a **latched, history-conditioned progress
cost** beats **latent-L2 to a goal image** when a *learned* action-conditioned world
model plans on OGBench-Scene. It is the successor to `event_smdp_h0/`, which
established the latching finding using the simulator as the dynamics model, a
hand-written automaton and privileged skill controllers. Here none of that is
available at deployment: the model consumes pixel history plus a candidate 5-dim
action chunk, the planner is CEM over action chunks, and success is OGBench's own
`SceneEnv._compute_successes` against a goal drawn from a dataset row.

Read `docs/SCENE_PROGRESS_WM_PROTOCOL.md` first — it is locked-before-execution and
carries the arena definition, the arm table, every stage verdict, and the amendments
made (with reasons) when a gate did not clear its threshold.
`docs/JOB_LEDGER.md` is the authoritative record of every Slurm job: id, exact
command, dependency, output path, final state, headline numbers.

Sibling programs each carry their own `CLAUDE.md` (`../event_smdp_h0/CLAUDE.md`).
The repo-root `CLAUDE.md`/`AGENTS.md` describe the older CAI-JEPA diagnostic
orientation and are deleted in the working tree but still in git:
`git show HEAD:CLAUDE.md`.

## Cluster compute policy (mandatory)

This checkout lives on a Slurm **login node**. Never run MuJoCo, rendering, model
loading, training, encoding or bulk result scans here — submit with `sbatch`.
Login-node work is `rg`/`sed`/`git`, small metadata reads, `python -m py_compile`,
and `squeue`/`sacct`. This is enforced in code: every script that touches the
simulator, a model or the cache raises unless `SLURM_JOB_ID` is set, including the
CPU-only ones (`check_progress_objective.py`, `analyze_scene_ladder.py`).

Verify claimed job state with **both** `squeue` and `sacct`. After a host restart,
unexplained file timestamps in this directory are far more likely to be this
session's own lost turns than another writer — check the transcripts before
cancelling or overwriting anything (see the withdrawn "concurrent session" entry at
the top of the ledger).

## Running things

Every script runs from the **repo root** (`..`); modules import as
`scene_progress_wm.*` and each script does `sys.path.insert(0, parents[2])`.
`scripts/_common.sh` is sourced by every wrapper and defines the whole environment:

- Interpreter `/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python` (`$PY`)
- `STABLEWM_HOME=/mnt/data/nhatnc129/jepa/lewm_stage0`, `MUJOCO_GL=egl`,
  `PYOPENGL_PLATFORM=egl`, `OMP_NUM_THREADS=1`
- `$DATA_ROOT` = released OGBench npz; `$CACHE_ROOT=/mnt/data/nhatnc129/jepa/scene_progress_wm`
  holds `cache/{train,val}/` and `checkpoints/<run>/...`
- Logs `/mnt/data/nhatnc129/jepa_runs/logs/<name>_%j.out`
- `stable_worldmodel` resolves to the editable checkout at
  `../diagnosis/external/stable-worldmodel`

Wrappers take parameters through `--export`, not argv:

```bash
sbatch --export=ALL,ARM=prog_ssl,SEED=0,STEPS=12000,RUN_ID=progress_20260904 \
  scene_progress_wm/scripts/slurm_train_progress.sh
```

Pipeline order, one wrapper per stage:

1. `slurm_stage0.sh` — plumbing gate (state restore, offline readout, goal wiring)
2. `slurm_cache_prep.sh` → `slurm_cache_render.sh` (array, `NUM_SHARDS`) →
   `slurm_cache_finalize.sh`, per `SPLIT=train|val`
3. `slurm_harness_gate.sh` — replay ceiling; `slurm_train_lewm.sh` — LeWM from scratch
4. `slurm_encode_latents.sh` — freeze the world model, encode the grid once
5. `slurm_train_progress.sh` per `ARM`; `check_progress_objective.py` for shape/invariance
6. `slurm_eval_ladder.sh` per `ARM`×`PLAN_SEED`×shard → `slurm_analyze_ladder.sh`

There is no pytest suite and no lint config. Correctness checks are in-band: the
Stage-0 gate, the harness replay gate, `check_progress_objective.py` (runs with no
checkpoint, so it can gate the pipeline before anything is trained), a short
`RUN_ID=*_smoke` run of each new arm, and the runtime assertions below.

## Architecture

Pure modules at the top level, executables under `scripts/`. Top-level modules
import MuJoCo, `stable_worldmodel` and `torch` lazily *inside* functions where they
can, so shapes and invariants are checkable without a simulator.

- `scene_data.py` — streaming npz access (`stream_rows` decompresses forward only),
  `OfflineSceneState` (forward kinematics on its own `MjData`, never steps physics),
  and `goal_match`, a component-for-component reimplementation of
  `SceneEnv._compute_successes` against a dataset goal row. Stage 0 verified the two
  agree exactly; keep them in lockstep.
- `scene_render.py` — **the only place that builds an environment.** Returns a
  `render_info` block (quality `fast` = shadows off, `offsamples=0`, 7× faster) that
  the cache records and every downstream job asserts against, so train-time and
  plan-time appearance cannot drift.
- `scene_lewm.py` — `LeWMConfig`, `build_lewm`, `SceneWindowDataset`, the published
  LeWM loss (`pred_loss + 0.09 * sigreg`) unchanged. Two declared deviations: 64px/
  patch 8, and **actions are not normalised** (OGBench actions already live in
  `Box(-1,1)`; adding a normaliser is the classic silent train/plan mismatch here).
- `progress_head.py` — the method. GRU over `[latent ‖ action-block embedding]`
  carrying a progress vector accumulated with `softplus`, so latching is
  architectural, not a penalty. `ARMS` is the arm table; each arm is the *same*
  architecture with different flags (`use_obs`, `use_action`, `use_history`,
  `monotone`), matched in parameter count. `ProgressTracker` holds the
  executed-history state.
- `progress_objective.py` — `ProgressCost`, `ScaledGoalMSE`, `build_mixture`. These
  implement the upstream `Objective` protocol, so an arm differs from the baseline
  **only** in the scalar CEM descends.
- `scene_eval.py` — the closed loop. Deliberately bespoke rather than
  `World.evaluate`, because the arena resets from one dataset row and takes its goal
  from another; `CEMSolver`, `ShootingCostEvaluator` and `PlanConfig` are reused
  unchanged.

Checkpoints and artifacts carry a `protocol` string
(`scene_progress_wm_lewm_v1`, `..._progress_head_v1`, `..._cache_v2`, …) and loaders
reject mismatches. Bump it when a payload format changes rather than reinterpreting
old files.

## Invariants that silently ruin results if broken

- **The progress tracker must be stepped only by the environment loop.** CEM calls
  the cost hundreds of times per replan; if the recurrent state advanced on those
  calls, the head would be integrating imagined rollouts into its memory of what
  actually happened. `ProgressCost` reads a detached copy, and `eval_scene_plan.py`
  asserts per episode that the head advanced exactly `env_steps / action_block + 1`
  times, raising otherwise.
- **The render grid is absolute** (`arange(0, total, stride)`), not per-episode, and
  `1001 % 5 != 0`, so each episode meets the grid at a different phase. Select
  windows by absolute row. Job 49520 died on exactly this assumption.
- **Every frame is self-rendered.** Stage 0 (`RERENDER_REQUIRED`) found the released
  pixels are not reproducible by this renderer (~20% pixels exact), so absolute
  numbers here are not comparable to any published `visual-scene-play` result. Only
  within-arena contrasts are claimed. Never mix stored and rendered pixels.
- **Trivial starts are screened out** in `build_episode_specs`; the screen reads only
  the dataset, never the arm, so all arms still see identical episodes. Unscreened,
  17/50 episodes at offset 25 were already solved at the start row.
- **`--mixture-weight 1.0` drops the progress term entirely** rather than multiplying
  by zero, so that arm must reproduce the baseline row for row. `--goal-scale` is
  measured once on the baseline arm and frozen; never refit per arm or per offset.
- Arms are paired: the plan seed is derived per episode
  (`plan_seed + 1_000_003*episode + 10_007*offset`), so every arm at a given offset
  and seed sees identical (start, goal) pairs and identical plan noise.
- The `slurm_*` wrappers `sha256sum` the sources they execute into the log, and
  result JSONs record checkpoint hashes and `cache_meta`, so a run's exact code and
  data version is recoverable from the artifact alone. Preserve that when adding a
  stage.

## Reading and reporting results

- **Report every planning number next to its offset's replay ceiling** from
  `harness_gate.json` (100/98/98/92% at offsets 25/50/100/200, n=50, ±11 points).
  The ceiling falls with horizon because the dataset stores `qpos`/`qvel` in float32
  while MuJoCo integrates in float64 — that is calibration, not a broken arena.
- **Verdict tokens are not rewritten after the fact.** `harness_gate.json` still
  reads `HARNESS_BROKEN` because the locked 0.95 bar is unchanged in code; the
  protocol's amendment explains why the bar was mis-specified and what replaced it.
  Amend in the protocol document, in the open — do not relax a threshold to pass.
- `outputs/**` is gitignored except `**/aggregate/`, `**/diagnostic/` and
  `training_summary.json`. Caches and checkpoints live under `$CACHE_ROOT` and are
  reproducible from the ledger; do not commit them.
- `prog_frame` and `prog_nomono` are **not** the same ablation: severing the
  recurrence severs the accumulator too, so `prog_frame` gives up history and
  latching together and only `prog_nomono` isolates latching. Job 49531 failed on
  assuming otherwise.
- Priors from `event_smdp_h0` that constrain interpretation: oracle-state evaluation
  can invert feedback rankings, so never select a cost under privileged state; and
  action-only dead reckoning is a strong baseline that collapses under execution
  failure — `prog_action_only` exists to re-measure that here.
- Standing limits, declared before the run: 64px is not the resolution the released
  LeWM-cube checkpoint was trained at; the play data is not expert data (hence the
  `τ=0.2` expectile); one benchmark and one backbone family, so representation
  independence is out of scope.

## Open thread

`slurm_encode_latents.sh` and `slurm_eval_ladder.sh` load `lewm_best.pt`, while the
protocol's "Checkpoint selection" section says run 49521 must use `lewm_last.pt`
(fixed budget, no model selection) because it ran under the old SIGReg-dominated
criterion. `train_scene_lewm.py` now selects on validation *prediction* loss, so
`lewm_best.pt` is sound for runs after that fix — confirm which run a wrapper is
pointed at before trusting the checkpoint it loads.
