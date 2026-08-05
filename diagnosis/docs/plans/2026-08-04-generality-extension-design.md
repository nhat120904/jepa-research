# Generality extension: task breadth + representation independence

Date: 2026-08-04. Status: **Track A gate-1 complete, gate-2+ladder running.
Track B paused by user decision (see below) after Stage 0 found no
already-implemented candidate.**

## Why

Post-submission review of the TMLR draft argued the paper is an
existence-and-mechanism case study (2 checkpoints x 2 MetaWorld tasks), not a
prevalence/generality claim, and asked for staged extension along two axes:
representation-family breadth and contact-regime breadth. This is tracked
outside the paper for now -- **no paper edits in this pass**, per user
instruction. Findings feed `docs/CURRENT_STATUS.md` / `docs/CLAIMS_EVIDENCE.md`
and the paper only after the user decides how to present them.

## A finding that changed the plan

`dino_wm_metaworld` and `jepa_wm_metaworld` both use a **frozen
`dinov2_vits14`** visual encoder (`external/jepa-wms/configs/evals/
simu_env_planning/mw/{dino-wm,jepa-wm}/*.yaml`, `visual_encoder.enc_version`
identical, `pretrain_enc_path:` empty in both). The encoder is never trained
(`external/jepa-wms/app/vjepa_wm/utils.py:685-692` freezes it). The two
checkpoints differ only in predictor architecture/conditioning and input
normalization constants.

Under exact-dynamics evaluation the learned predictor is **never called**
(`scripts/30_latent_oracle.py:12`; only `adapter.encode` runs, at `:221` and
`18_closed_loop_eval.py:262`). So for the latent-$L_2$ arm, the two "models"
differ only by an affine input rescale. Confirmed empirically: `dino_wm` and
`jepa_wm` L2 mean object-goal distance on mw-pick-place both equal
26.6314 cm to 4 decimal places (`results/confirmatory_{dino,jepa}_wm_metaworld
_l2_mw-pick-place_seed20000_n64.csv`).

Consequence: on MetaWorld the paper currently has **n=1 representation, not
n=2**. This reframes the two tracks:

- **Track A (task breadth)** answers the contact-regime objection but, with
  the current two checkpoints, does **not** answer the representation-count
  objection.
- **Track B (representation independence)** is the only track that answers
  the representation-count objection, and must select a model whose visual
  encoder is independently trained (not a reused frozen DINOv2), because the
  predictor plays no role under exact dynamics.

## Track A -- task breadth

### Task selection

Candidates already in the 12-task diagnostic config
(`configs/diagnostic_metaworld.yaml:15-31`) and MuJoCo-generic per the codebase
scan: `mw-door-open`, `mw-drawer-close`, `mw-button-press`, `mw-window-close`,
`mw-assembly`, `mw-peg-insert-side`. `mw-reach` remains the free-space control
via existing evidence (`results/metaworld_reach_strict.csv`,
`results/confirmatory_*_oracle_*`) -- not rerun here.

Excluded with reason:
- `mw-hammer`: `OBJECT_SLICE = slice(4,7)` (`stratification/metaworld_regimes.py
  :38-40`) reads the hammer head, not the nail (`obs[11:14]`), so the
  simulator-state reference cost `||o_T-o_g|| + 0.5||h_T-o_T||` would optimize
  the wrong target. Needs a code change (task-aware object slot); out of scope
  for this pass.
- `mw-drawer-open`, `mw-bin-picking`: in the HF dataset but not in the 12-task
  config, so not in the precomputed latent cache -- usable for the L2 arm
  (needs no cache) but not for probe-based arms without first extending
  `configs/diagnostic_metaworld.yaml` and rerunning `03_extract_latents.py`.
  Deferred.
- `mw-sweep`: does not exist; only `mw-sweep-into`, which has an env but no
  offline data. Deferred.

### Gate 1 -- expert competence (submitted, pre-registered here)

`rollout_expert` (`scripts/18_closed_loop_eval.py:238`) hardcodes
`max_steps=100` at both of its call sites (`scripts/29_oracle_ceiling.py:192`,
`scripts/30_latent_oracle.py:511`), while MetaWorld's own `max_path_length` is
500. `mw-peg-insert-side`'s expert already showed this failure mode: only 4/16
goal rollouts fired success within the 100-step cap
(`results/metaworld_precision_ladder.csv`). A task whose goal frame is not
actually a success state would make any oracle/L2 result on it uninterpretable.

**Pre-registered criterion**, run via the new standalone
`scripts/70_task_breadth_expert_check.py` (reruns the identical `make_env` +
`rollout_expert` path used by scripts 29/30, only raising `max_steps`, no
model/GPU-encode/CEM involved):

- ELIGIBLE: >=75% of episodes succeed within `max_steps=100` (the production
  cap) AND within `max_steps=500` (MetaWorld's own budget).
- INELIGIBLE-BUDGET: succeeds at 500 but not at 100 -- a budget problem, not a
  policy problem; would need a code change (raise the cap) to use, not
  attempted in this pass.
- INELIGIBLE-POLICY: fails to reach 75% even at 500 -- the scripted policy
  itself is unreliable on this task in this harness; excluded.

16 episodes, seed0=70000 (fresh base; existing bases in use: 10000 default,
20000 confirmatory/off-policy, 30000 confirmatory terminal-cost, 41000
preselection, 42000 shared-branch, 61000-63000 factorized/selection-sprint).

Submitted: `sbatch scripts/slurm_task_breadth_expert_check.sh` (job 35500, single
job, no array, GPU node required only because MuJoCo rendering needs the EGL
context per repo convention -- CLAUDE.md forbids simulator work on the login
node). **Completed in under 2 minutes wall time** (rollout_expert has no
model/GPU-encode cost, just MuJoCo stepping).

Output: `results/task_breadth_expert_check.{csv,md}`.

**Result:**

| task | within capA (100) | within capB (500) | verdict |
|---|---|---|---|
| mw-door-open | 94% (15/16) | 100% (16/16) | ELIGIBLE |
| mw-drawer-close | 100% (16/16) | 100% (16/16) | ELIGIBLE |
| mw-button-press | 100% (16/16) | 100% (16/16) | ELIGIBLE |
| mw-window-close | 100% (16/16) | 100% (16/16) | ELIGIBLE |
| mw-assembly | 100% (16/16) | 100% (16/16) | ELIGIBLE |
| mw-peg-insert-side | 69% (11/16) | 81% (13/16) | INELIGIBLE-BUDGET |

5/6 tasks eligible. `mw-peg-insert-side` fails the pre-registered rule (needs
>=75% at *both* caps; capA=69%<75% even though capB=81%>=75%), confirming the
earlier qualitative signal from `results/metaworld_precision_ladder.csv`.
Per the pre-registered rule this task is excluded from the ladder in this
pass, not silently included with a caveat.

### Gate 2 + terminal-cost ladder (run together, one array)

For each of the 5 ELIGIBLE tasks: one reference cell
(`scripts/29_oracle_ceiling.py`, no model -- doubles as Gate 2, the positive
control) + one L2 cell (`scripts/30_latent_oracle.py --model
dino_wm_metaworld --cost l2`). **Only `dino_wm_metaworld` runs the L2 arm** --
since the two checkpoints share an encoder, running both would duplicate the
same experiment (see finding above). If a task shows a surprising result,
`jepa_wm_metaworld` can be added as a same-encoder replication check, not as
an independent data point.

10 cells (5 tasks x {oracle, l2}), 16 episodes each, seed0=70000 (same base as
gate 1 -- oracle and l2 arms are paired per-seed, matching the
`slurm_confirmatory_locked.sh` convention). Array script:
`scripts/slurm_task_breadth_ladder.sh`. Estimated cost: 5 x (1.4 GPU-h L2 +
~1 GPU-h reference) ~= 12 GPU-h.

A task where the reference also fails cannot distinguish "cost is
uninformative" from "task infeasible at this budget" -- reported but excluded
from the primary comparison, per the pre-registered rule in the parent plan.

### Reporting rule (pre-registered)

Per the parent plan: any task or cell where latent $L_2$ succeeds is a
**boundary condition to report**, not a cell to discard. A boundary is
stronger evidence than a uniform null.

## Track B -- representation independence

### Stage 0 criteria

A candidate world model qualifies only if all five hold:
1. Visual encoder is independently trained (not a reused frozen DINOv2).
2. Checkpoint is public and loadable.
3. Runs in an environment with snapshot/restore (required for exact dynamics).
4. Terminal cost is comparable to a goal image, or reducible to one.
5. Action space is compatible or mappable.

### Stage 0 scan result -- executed, no already-implemented candidate found

Checked every upstream `dino-wm`/`jepa-wm` environment family (metaworld,
wall, pusht, point-maze, droid, robocasa) in
`external/jepa-wms/configs/evals/simu_env_planning/*/*/*.yaml`: **all** use
the same frozen `dinov2_vits14` encoder, `pretrain_enc_path:` empty. No
already-loadable `dino-wm`/`jepa-wm` checkpoint passes criterion 1.

The one genuinely independent encoder in the whole upstream registry is
`vjepa2_ac_droid`/`vjepa2_ac_oss` (`enc_type: vjepa`, V-JEPA-2 ViT-Giant,
`embed_dim: 1408`, confirmed as a distinct family by
`results/droid_scaling_curve.md`). But it only runs on **DROID** --
passively-logged real-robot video, no simulator, so criterion 3
(snapshot/restore) fails hard: there is no state to restore to and no
alternate action to re-execute.

RoboCasa is the only environment in the repo that is both (a) a real
simulator (robosuite/MuJoCo) and (b) already configured to load
`vjepa2_ac_droid` (`configs/diagnostic_robocasa.yaml:15,17`, shared 7-dim
droid action format). It is therefore the only path to pairing the
independent encoder with exact-dynamics evaluation. But verifying
`snapshot()`/`restore()` (`scripts/29_oracle_ceiling.py:62-90`) shows it reads
`mocap_pos`/`mocap_quat`/`curr_path_length` -- MetaWorld/Sawyer-specific
conventions, not generic MuJoCo. `make_env` and the scripted-expert lookup
(`metaworld.policies`) are equally MetaWorld-specific. **No oracle harness
exists for RoboCasa** -- building one is engineering work comparable in scope
to what scripts 18+29+30 already represent for MetaWorld, on top of thin data
(~10 trajectories, prior `insufficient_data` verdict in the observational
diagnostic, `docs/HANDOFF_DROID.md:215`) and no scripted-expert-policy
analogue.

External candidates (TD-MPC2, DreamerV3 on DMControl) would supply a
genuinely independent encoder and their own simulator with native
snapshot/restore, but need an adapter and oracle harness built from scratch,
and their tasks are not contact-rich manipulation, weakening the comparison to
MetaWorld push/pick-place.

**Decision (user, 2026-08-04): pause Track B.** No candidate clears Stage 0
without a harness-building project of its own. Track A proceeds alone for
this pass. If Track B resumes, the first artifact needed is a scoped
cost/effort estimate for a RoboCasa oracle harness (or a from-scratch
DMControl one) before committing GPU-h -- not assumed feasible for free.

## Verification checklist

- Gate 1: `results/task_breadth_expert_check.md` produced, verdict column
  populated for all 6 tasks, no exceptions in the Slurm log, sentinel line
  `TASK_BREADTH_EXPERT_CHECK_DONE` present.
- Gate 2 / terminal-cost ladder cells: CSV row count == `--episodes`, Slurm log
  ends with the script's own per-cell completion line, `git rev-parse HEAD`
  logged.
- Cross-check: rerunning a known cell (`dino_wm_metaworld` x `mw-push` x L2, 16
  seeds) on this harness must reproduce existing numbers before trusting new
  task cells.
