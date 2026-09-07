# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`event_smdp_h0/` is one experiment program inside the larger `jepa-research` repo
(git root is `..`). It runs **preregistered H0/H1/H2 gates** on whether backing up
*intermediate event state* through a planner beats backing up only terminal
success, on OGBench Cube-single (closed, negative) and OGBench-Scene tasks 4 and 5
(active). MuJoCo is the oracle world model throughout, so these are causal-room
experiments, not learned-world-model claims.

Read `README.md` first — it is the running narrative of every gate and its verdict,
kept in chronological order. `docs/*_PROTOCOL.md` are the locked-before-execution
protocols; `docs/JOB_LEDGER.md` is the authoritative record of every Slurm job.
`docs/SCENE_RESEARCH_POSITIONING.md` holds the literature positioning and the
claim limits (`hint^2`, EV-WM, conformal prediction neighbours).

The repo-root `CLAUDE.md`/`AGENTS.md` (older CAI-JEPA diagnostic orientation) are
deleted in the working tree but still in git: `git show HEAD:CLAUDE.md`.

## Cluster compute policy (mandatory)

This checkout lives on a Slurm **login node**. Never run MuJoCo rollouts, model
loading, training, or bulk result scans directly here — submit them with `sbatch`.
Login-node work is limited to `rg`/`sed`/`git`, small metadata reads, syntax
checks, and `squeue`/`sacct`. Even CPU-only analysis goes through Slurm
(`slurm_*_analyze.sh`), which is why every analysis has its own thin wrapper.

Verify claimed job state with **both** `squeue` and `sacct` before acting on it.
Other agents/sessions submit into the same queue (see the `49363`/`49364` rows in
the ledger); check for duplicate work before submitting a long array.

## Running things

Everything runs from the **repo root** (`..`), because modules import as
`event_smdp_h0.*`. The interpreter is the shared runtime venv, not a local one:

```bash
sbatch --export=ALL,START_INDEX=0,END_INDEX=0,RUN_ID=smoke event_smdp_h0/scripts/slurm_scene_skill_failure_chunk.sh
```

- Interpreter: `/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python`
- `STABLEWM_HOME=/mnt/data/nhatnc129/jepa/lewm_stage0`, `MUJOCO_GL=egl`,
  `PYOPENGL_PLATFORM=egl`, `OMP_NUM_THREADS=1`
- Logs: `/mnt/data/nhatnc129/jepa_runs/logs/<name>_%j.out`
- Partition `mig`, one GPU for anything that renders or trains

There is no pytest suite and no lint config. The correctness checks that exist are
in-band: `core.self_check()` / `scene_core.self_check()` (pure, importable without
MuJoCo), the per-run smoke shard, and the anchor-reproduction assertions below.

## The experiment lifecycle

Every gate follows the same five steps, and skipping one invalidates the result:

1. **Lock the protocol** — write `docs/SCENE_<NAME>_PROTOCOL.md` with arms, reset
   seeds, endpoint, and the pass/fail rule *before* any cell is evaluated. Commit it.
2. **Train** (if needed) — `slurm_*_train.sh` writes checkpoints under
   `outputs/<experiment>/checkpoints/<arm>/seed<N>/`.
3. **Smoke one shard** — `RUN_ID=smoke` with `START_INDEX=END_INDEX`. This is where
   design confounds get caught (the fixed-budget bug in job `49453`). A confound
   found here can be fixed; one found after real data cannot.
4. **Evaluate** — `slurm_*_chunk.sh` over index ranges, one `result.json` per reset
   under `outputs/<experiment>/eval/<RUN_ID>/<index>/`.
5. **Analyze** — `slurm_*_analyze.sh` → `analyze_*.py` emits a single `verdict`
   string (`MECHANISM_CONFIRMED`, `DEAD_RECKONING_REFUTED`, `PARTIAL`,
   `INCONCLUSIVE`, `NONDETERMINISTIC_EVAL`, …) plus CIs into
   `outputs/<experiment>/aggregate/<RUN_ID>/`. `diagnose_*.py` scripts produce
   mechanism evidence and are labelled exploratory unless preregistered.

Then append a ledger row (job id, exact command, dependency, output path, state,
and the headline numbers) to `docs/JOB_LEDGER.md`, and extend the `README.md`
narrative.

### Anchor reproduction

Each new experiment deliberately re-runs the previous experiment's arms on the
**same reset-seed band** (`88400–88463` = task 4, `88500–88563` = task 5), and the
analysis asserts those shared cells reproduce the earlier run *exactly* — success
and deployed skill sequence, e.g. "896/896 rows". A mismatch means the eval became
nondeterministic and the verdict is void. Preserve this when adding arms: give new
arms new names, never change an existing arm's computation.

Slurm scripts `sha256sum` the source files they execute into the log, so a run's
exact code version is recoverable from the log alone.

## Code layout

Pure modules at the top level, executables under `scripts/`. The separation is
load-bearing: **top-level modules must not import MuJoCo, `stable_worldmodel`, or
encoders at module scope**, so their invariants can be checked before an expensive
simulator job starts. Scripts import the simulator lazily *inside* functions.

- `core.py` — arm names (`ARM_TERMINAL`/`ARM_EVENT`), `EventSummary` ordinal reward,
  proposal ordering. The closed Cube-single gate.
- `scene_core.py` — the Scene event automaton: `ScenePredicates` (raw physical
  predicates) → `MilestoneState` (task-conditioned, *history-bearing*, latched
  milestone counters) via `advance_milestones`, plus `feedback_reward` and the
  matched `uct_plan_search`. Both arms share this search; **only the scalar
  feedback differs**. Keep it that way — it is what makes the comparison auditable.
- `scene_feedback.py` — the planner-facing scalar family. `event_progress` and
  `automaton_potential` are not two families: both equal
  `0.90 * (w*cube/5 + (1-w)*window/3)` at `w=0.500` and `w=0.625`. New feedbacks
  belong in `SWEEP_FEEDBACKS`, not as a new special case in `scalar`.
- `scene_event_perception.py` — single-frame observer (`EventStateObserver`).
- `scene_event_history.py` — GRU observer serving the whole 2×2 factorial and its
  input ablations at **identical parameter count**; `history_length=1` degrades it
  to a frame observer so the contrast isolates history, not capacity.
  `ABLATIONS = (none, action_only, obs_history)` decompose "history" into its
  observation and action parts.
- `scene_history_dataset.py`, `scene_learning.py`, `scene_abstract_smdp.py` —
  dataset construction, H1 contextual heads, and the event-state-closed H1b SMDP.

Checkpoints carry a `protocol` string (`scene_event_perception_v1`,
`scene_event_history_v1`, …) and loaders reject mismatches. Bump it when the
payload format changes rather than silently reinterpreting old files.

`outputs/**` is gitignored except `aggregate/`, `diagnostic/`, and
`training_summary.json` — raw per-shard eval and checkpoints are large and
reproducible from the ledger. Do not commit them.

## Findings that constrain how you interpret new results

These were established here and are easy to re-break:

- **Error direction dominates exact accuracy.** Over-reading event progress is
  unrecoverable (1.92% episode success); under-reading is often recoverable. A
  *more* accurate observer can plan *worse* (`frame_full` 76.5% exact but 247
  over-reads → 69.01% success). Never select an observer on exact-q alone.
- **Under-read recoverability is a property of the feedback, not a law.** Under
  `automaton_potential` under-reads cause livelock instead.
- **Oracle-state evaluation inverts feedback rankings** (Spearman 0.357, n=8): the
  best feedback with simulator `q` is nearly the worst with a learned one. Never
  select a feedback function under oracle state.
- **Action tokens cause over-reading.** Vision-given-actions is worth +9.90 points;
  actions-given-vision adds nothing and degrades under skill failure. Prefer
  `obs_history_full` as the reference observer arm.
- **Dead reckoning is a real, strong baseline** (untrained `openloop_transition`
  reaches 80.47%). Any new "the model reads the scene" claim must beat it.

## Writing style for results

Verdicts are locked and are **not** relaxed or reclassified after the fact — the
3-seed replication failed on one episode and stayed a FAIL; `branch_w040` missed on
CI width and stayed a miss. Report negatives and near-misses in the README with the
same prominence as positives, mark exploratory contrasts explicitly, and state
standing limits (hand-specified automaton, skill library and labels; MuJoCo oracle
dynamics; oracle arm reads simulator `q`). Commit messages here are long-form: a
one-line claim, then the verdict, the numbers with CIs, and any confound found and
when it was fixed.
