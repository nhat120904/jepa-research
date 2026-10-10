# Method: event world model + search from offline pixel play (one formulation for scene, cube, puzzle)

This folder is the single source of the current method. Everything else in the repository (`scripts/`, `docs/*/source*`)
is history: earlier front ends, the STATE track, the old puzzle- and cube-specific pipelines. Code here was copied from the
frozen sources listed below and changed only where stated. Current results, diagnosis and the agreed direction:
[STATUS.md](STATUS.md).

Input: offline OGBench visual play data (64 x 64 frames + actions). No proprio, no state labels, no task-specific rules.
The same code and settings run on every family; thresholds are fitted from data by the same rules.

## Components

| # | Component | Learned / rule | File |
|---|---|---|---|
| 1 | AgentNet: next frame from (frame, action), action dropout | learned | `frontend.py`, `frontend_train.py --stage agent` |
| 2 | Agent teacher: a token is the agent's when its change is explained by the action (2-means split of the explained share) | rule | `frontend_train.py --stage label` |
| 3 | Segmenter: per-token agent mask from one frame | learned | `--stage seg`, `--stage segpred` |
| 4 | SceneCodes: per-patch discrete code, 16 x 16 tokens, FSQ 5^6 | learned | `--stage scene` |
| 5 | SeeThrough: per-frame code of the scene behind the agent + probability, self-supervised targets | learned | `--stage stargets`, `--stage seethru` |
| 6 | Memory rule: a token takes a code after k = 3 agent-free frames (agent mask dilated by one token) | rule | `frontend.memory_rule` |
| 7a | Token entities: every token whose agent-free memory changes in TRAIN; per frame its SeeThrough code digits + readable flag (used by 7 to read places under the agent) | rule | `memory_entities.py` |
| 7 | Object entities from pixels: PLACES (exactly recurring or majority-changed, persistent, not agent-coloured) and MOVERS (rest instances, colour clusters, K by colour co-occurrence); places read in full view or through the agent with SeeThrough codes; DISCRETE places (bimodal readings, each side dominated by one exact reading) snap to their states at their centre; agent colours refine the mask | rule from data | `objects.py` |
| 8 | Events: rest runs, changes timed per frame, changes grouped by minimum stabbing (no slack; a window holding several moments goes to the one nearest its centre), states read between events with known masks, acted = touched identity nearest the centre of the changes | rule | `events_objects.py` |
| 8b | Relabel by explanation (EM step): the round-1 world model picks, per event, the candidate that best explains the observed after-state; events no single action explains (2-means split of log losses) do not train round 2 | learned + rule | `relabel_events.py` |
| 9 | Event = (acted entity, its target state) | definition | `events.py`, `world_model.py` |
| 10 | World model: transformer over entities + event, predicts every entity (side effects) | learned | `world_model.py --stage wm` |
| 11 | Candidates: per entity its goal state and its most frequent rest states; covered entities are not acted on | rule from data | `world_model.Model.candidates` |
| 12 | Canonicalization: attributes with <= 8 values in TRAIN snapped to them | rule | `world_model.finite_rest_support` |
| 13 | Event support prior (NCE on play events); low support = no effect; ranks candidates | learned | `event_support.py`, `train_support.py` |
| 14 | Cost-to-go by value iteration in the world model, goals from imagined walks, input [s, g, s - g, \|s - g\|] | learned | `world_model.py --stage h` |
| 15 | Batched weighted A*; goal test on the goal entities judged observed | algorithm | `world_model.Model.plan` |
| 16 | Skill: CNN on two frames + Gaussian maps at the entity's current and target position, conditioned on (entity, current state, target state), 8-action chunks | learned (BC) | `skill.py`, `skill_segments.py` |
| 17 | Closed loop: online front end + the table reading (`objects.read_frame`), goal image read with any looked-up SeeThrough code, unseen movers = hidden; plan, execute, end the event when the acted entity is at rest, belief update (entities predicted to change but unseen since take the prediction when every seen entity agrees), replan; timeout recovery | rule | `closed_loop_objects.py` (token track: `closed_loop.py`) |

Object track (current, user-approved direction A, 2026-10-08/09): entities are objects found by data-driven rules
(`objects.py`), with the same backend as the STATE track. The token track (7a as entities, `events.py`) is history: token
entities left side effects unlearnable (world-model side effects within tolerance .09-.49) and lost cube positions.

## Sources (sha256 prefix) and what changed

| File here | Copied from | Changes |
|---|---|---|
| `frontend.py` | `scripts/sm2_model.py` (0e121d4914d5fa05) + FSQ/conv_block/Res/Up of `scripts/sm_model.py` (770e1aa18860dbfc) | none (definitions checked identical); the front-end checkpoints in `E:\jepa-data\event_wm\scene_memory_v2\<family>_grow_*` come from this code |
| `frontend_train.py` | `scripts/sm2_train.py` (feea3560aa527970) | import of `frontend` |
| `memory_entities.py` | new | component 7a |
| `objects.py` | new | component 7 (rules and history in its docstring) |
| `relabel_events.py` | new | component 8b (EM step with small-loss selection; evidence and known limit in its docstring) |
| `events_objects.py` | `docs/generic_state_20261007/base_source/u_events.py` (38fab593d2319c7e), `--per-frame` | identities observed only where the table shows them; tables may cover the first episodes of a split; noise radii from consecutive observations; place readings need no agent-clear test; `--group stab` (default) with states between events, known masks and the touched-centroid acted rule (docstring) |
| `events.py` | `docs/generic_state_20261007/base_source/u_events.py` (38fab593d2319c7e), `--per-frame` mode | constant positions; quantized-appearance threshold rule (see below); vectorized rest runs; rest runs saved instead of per-frame labels; agent distance at token resolution |
| `world_model.py` | `.../u_wm.py` (8e483dcfd98e3acf) | state length D from data; unknown goal entities; prototypes = most frequent rest values for quantized appearance (token track) or the u_wm k-means of per-frame rest labels (object track); candidate budget ranked by the support prior; object events: only events with a known target train, unknown after-states are masked out of the loss and the scores; stages wm / h |
| `event_support.py`, `train_support.py` | `.../event_support.py` (8517a6011a96d4c8), `.../train_event_support.py` (a7d2db35038072a5) | D as a parameter; entities without VAL events get the TRAIN floor threshold; trained before the cost-to-go; object events with an unknown target are left out |
| `skill.py`, `skill_segments.py` | `.../u_skill.py` (5d0c8a5e45b96ec5), `.../skill_segments.py` (2c2a4575af94d804) | appearance has A values; memory-mapped frames; the state-vector skill removed |
| `closed_loop.py` | `.../u_closed_loop.py` (8c31ce882ee485af) | online front end instead of the SAM 2 reader; unknown goal entities; another entity's change ends an event only once the acted entity is at rest |
| `closed_loop_objects.py` | `closed_loop.py` | object reading (`objects.read_frame`); goal places from any looked-up code; hidden movers (covered); unseen entities start at their goal; change / arrival tests with tol_pos and per-entity appearance tolerances |

Changes forced by the token entities, with the evidence (local, 2026-10-08):
- Appearance thresholds: >= 99.9% of consecutive agent-free token readings are identical, and 40-50% of real changes move
  one digit by one step. The `u_events` rule (2-means split of non-zero frame-to-frame differences) put the noise radius
  between 1- and 2-step changes and would have dropped ~70% of real changes. With quantized readings the noise radius is a
  quarter step and any code change counts; continuous readings keep the `u_events` rule.
- Transition timing reads every frame, as `u_events` does: 48% of the frames between two rests are not confidently
  readable (gripper), yet their code is the old or new rest code 77% of the time. Requiring readability stretched change
  windows to 16 / 29 frames (p50 / p90) and chained consecutive puzzle presses (11.3 events per episode for 29.6 presses).

## Settings (same for every family)

memory k = 3, agent dilation 1 token; events m = 5, frame gap 0; world model 30k steps (round 1 and round 2); support 6k steps, quantile .02;
cost-to-go 150k steps, width 2048, walk goals 300k / 30 events, |s - g| input; candidates <= 32 per state by support;
skill 40k steps, chunk 8, history 2, release 10; loop timeout 250, rest 5 frames, recovery 8 steps, 20k expansions.

## Not part of the method (do not import from here)

`scripts/u_*.py` (stale copies of the unified backend), `scripts/sm2_events.py`, `sm2_reader.py`, `sm2_planner.py`,
`sm2_skill.py`, `sm2_loop.py` (cancelled bit backend), `scripts/planner.py`, `build_events.py`, `train_skill3.py`,
`closed_loop.py` (old puzzle pipeline with bit / XOR events), `scripts/cube_*.py` (cube-specific pipeline),
`scripts/sam2_*.py` and colour-identity rules (SAM 2 front end), `s_entities.py`, `g_entities.py` (STATE front ends),
`slowmap*`, `slots.py`, `sfa_code.py`, `sm_model.py` v1 (earlier front-end attempts). `sm2_diag.py` stays the PRIVILEGED
evaluation of the front end (scoring only).

## Run (this PC)

```powershell
.\method\run_family.ps1 -Family puzzle                       # entities .. wm, relabel, wm2, support, h, skill, loop
.\method\run_family.ps1 -Family scene -Stages wm,relabel     # selected stages
```
Object-track outputs: `method\objects`, `method\obj_events` (round 1), `method\obj_model` (round-1 model), `method\obj_events_r2`
and `method\obj_model_r2` (relabelled events, round-2 model + support + cost-to-go: used by the skill and the loop),
`method\obj_skill`, `method\obj_loop_seed<N>`, `method\obj_loop_scripted_seed<N>` (PRIVILEGED press arm, puzzle only).
`local\run_object_queue.ps1` runs the dev families and then the held-out environments (front end via `local\run_sm2.ps1`
with the same settings, then the same chain; `run_family.ps1 -Family <name> -Run <dir> -DataEnv <play env> -EvalEnv <env>`).

Outputs go to `E:\jepa-data\event_wm\scene_memory_v2\<family>_grow_*\method\`. Front-end training (components 1-5) is
`local\run_sm2.ps1` with `scripts/sm2_train.py`, which is identical to `frontend_train.py`.
