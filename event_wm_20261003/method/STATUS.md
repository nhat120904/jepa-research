# Event WM, object track: status and direction (2026-10-09)

Single place for where the method stands and what comes next. Details and every number's provenance: `JOB_LEDGER.md`
(2026-10-09 bullets). Method components: `method/README.md`.

## 1. Pipeline as of 2026-10-09 (one setting for every family)

pixels (no proprio, no labels) -> front end learned per environment (AgentNet, agent teacher, segmenter, SceneCodes,
SeeThrough, memory rule) -> object entities (`objects.py`: places + movers; DISCRETE places snap to exact states at their
centre) -> events (`events_objects.py`: per-frame change windows, greedy minimum stabbing without slack, a window holding
several moments goes to the one nearest its centre, states between events with known masks, acted = touched identity
nearest the centre of the known changes) -> round-1 world model -> relabel by explanation (`relabel_events.py`: an event
trains round 2 if the model explains it or its best candidate is the contact rule's choice) -> round-2 world model,
event support, cost-to-go (value iteration, 150k steps) -> batched weighted A* -> event skill (BC, two frames, history
dropout .5) -> closed loop (`closed_loop_objects.py`: static goal / first frames, belief update for hidden entities,
plan commitment, avoidance of the last 3 states, event end while the acted entity is hidden, current frame only at an
event's first skill query).

## 2. Results (seed 0, 30 episodes per environment = 6 x 5 tasks, no privileged input)

| environment | method | OGBench pixel baselines (best) |
|---|---|---|
| puzzle-4x5 (dev) | 7/30 = 23% with the single-frame skill; 3/30 with the default skill | 17% (GCIVL), HIQL 13% |
| cube-triple (dev) | 4/30 = 13% (task 1 4/6) | 21% (HIQL) |
| scene (dev) | 7/30 = 23% (tasks 1 4/6, 3 3/6; tasks 4-5 find no plan: hidden goal) | 49% (HIQL) |
| puzzle-3x3 (held-out) | 1/30 = 3% | 73% (HIQL) |
| puzzle-4x4 (held-out) | 2/30 = 7% | 60% (GCIVL, HIQL) |
| cube-single (held-out) | not run (front end trained; queue stopped) | 89% (HIQL) |

PRIVILEGED diagnostic arm (scripted presses on the round-2 puzzle model): 7/30, task 1 6/6.

## 3. Diagnosis

1. **Occlusion by the arm** is the main failure, at every stage. Puzzle presses, PRIVILEGED scoring:
   - visible frames per light toggle: 4x5 125, 4x4 90, 3x3 40. The play policy presses at the same rate (~31 per
     episode) on every board, and the arm covers more of a small board.
   - held-out objects: 3x3 7 of 9 buttons found, 4x4 15 of 16. Place discovery needs agent-free rest instances.
   - pressed light = acted label: 4x5 .875, 4x4 .71, 3x3 .47 (presses inside exactly one event core).
   - The event TIMING transfers (4x4: 92% of presses near an event); the event CONTENT does not.
2. **Hand-designed perception and event rules** (snap tests, grouping rule, contact centroid, touch radius, windows),
   added while looking at dev results. Each encodes an assumption that a new environment can break.
3. **Behaviour-cloned skill:**
   - With two frames it copies the ongoing motion and ignores the target (condition swap .095).
   - One frame fixes targeting (puzzle exact presses .865) but loses the grasp phase (cube 0/6).
4. **Hidden goals** (scene cube in the closed drawer, cube stacks): the planner finds no plan.
5. **World-model generalization** on states the loop visits, for effects rarely observed in training (puzzle-4x5
   light 2: 157 of 177 planned presses surprised the model).

## 4. Direction (agreed with the user 2026-10-09)

**Pragmatic step first** (about 1-1.5 weeks, results early):

1. Agent-free perception.
   - SeeThrough pixel inpainting removes the arm before object discovery and per-frame reading.
   - Offline checks (PRIVILEGED scoring): buttons found 9/9 (3x3) and 16/16 (4x4); acted accuracy on held-out
     (now .47 / .71); visible share per light.
   - Risk: the inpainting was trained only on visits where the scene did not change.
2. Low level: offline goal-conditioned RL (IQL / HIQL-style value + policy extraction) conditioned on the event's
   target state, with hindsight relabelling; one setting for cube, puzzle and scene. It replaces the BC skill.
3. Protocol.
   - Develop on cube-triple, puzzle-4x5 and scene; freeze the code.
   - Run held-out once with no change: cube-single, cube-double, cube-quadruple, puzzle-3x3, puzzle-4x4, puzzle-4x6.
   - Compare with OGBench; multiple seeds and ablations before any claim.

**Then v2, if the event layer remains the limit:** events inferred in a learned model instead of rules.
- Factored latent state, with the number of factors over-estimated.
- One binary interaction gate per factor, predicted from the agent's features and action (BISCUIT-style), with gated
  recurrence that keeps hidden factors (object permanence).
- ELBO with a sparsity prior on gates; events = gate openings; the acted entity = the factor the agent's gate opens.
- This replaces the grouping, acted-entity and window rules.
- Go / no-go on puzzle-3x3 + cube-single: acted accuracy above .85 on 3x3 (rules now: .47).

## 5. Related work to position against (2026-10-09 survey)

- **Temporal abstraction by sparse latent change:** THICK (ICLR 2024), GateL0RD (NeurIPS 2021), VTA (NeurIPS 2019),
  EAWM (ICLR 2026).
- **Causal variables with binary agent interactions:** BISCUIT (UAI 2023), CITRIS / iCITRIS.
- **Symbolic models from images:**
  - learned from image transitions: LatPlan (JAIR 2022, Lights Out from images), STRIPS-WM (2026);
  - with given skills: Skills-to-Symbols / James & Konidaris (ICLR 2022);
  - with VLMs: pix2pred, VisualPredicator, ExoPredicator.
- **Object permanence under occlusion:** Loci-Looped, Structured World Belief.
- **Latent actions from video:** LAPO, Genie, LAPA.
- **Hierarchical latent planning:** HWM (2026), Director, Play-LMP / TACO-RL.
- **Offline GCRL:** HIQL, SHARSA (state-based puzzle-4x5/4x6 solvable with up to 1B transitions).
- **Object-centric world models:** Dyn-O, Slot-MPC, Better Slots / Better Worlds (OGBench cube).
- **Skills from offline data:** EXTRACT (CoRL 2024).

Positioning:
- learned from offline pixel play, with no labels, proprio, VLM or given skills;
- explicit event abstraction + symbolic search for combinatorial long-horizon tasks;
- one formulation across families, evaluated on held-out environments.

## 6. Data (E:\jepa-data\ogbench\data)

- Pixel play: cube single / double / triple / quadruple, puzzle 3x3 / 4x4 / 4x5 / 4x6, scene.
- State play + noisy: all cube and puzzle environments, scene play.
- Fetched with `local/fetch_data.py` (groups `visual_cube_all`, `visual_puzzle`, `state_cube_puzzle_all`; user
  approval 2026-10-09).
- Pixel noisy datasets (`visual_noisy_cube_puzzle`, ~43 GB) are not fetched (disk).
