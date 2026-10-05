# Discrete event world models for combinatorial long-horizon manipulation

Direction adopted by the user on 2026-10-03. It grew out of `config_generalization_20261001/` (P1).
This README is the reference for the method, its design decisions and how to run and debug it.
Every job and number lives in `JOB_LEDGER.md`. Run outputs live in `/mnt/data/nhatnc129/jepa/event_wm`.

## 1. Problem and gap

**Target tasks.** OGBench visual puzzle (robotic Lights Out from 64×64 pixels). Combinatorial
complexity is set by grid size: 3x3, 4x4, 4x5, 4x6. Evaluation tasks need 2–24 presses within
500–1000 steps.

**What fails today.**
- Offline goal-conditioned RL collapses on these tasks: HIQL ≤17% (4x5) and ≤15% (4x6) from pixels;
  SHARSA 14% on 4x6 with state input and 1B transitions.
- Horizon reduction (SHARSA, OTA) and graph search over dataset states (SoRB, TTGS) do not fix it.

**Why (measured in P1 and here).**
- A time-step goal-conditioned value is only reliable within about 4–5 presses. GCIVL event-sign
  accuracy is .99 / .92 / .65 / .50 / .46 for true distance d* 1–2 / 3–4 / 5–6 / 7–8 / 9+ (P1 job 56899).
- Image or latent distance to the goal is uninformative beyond about 5 presses: the Hamming distance
  saturates at the random level from d* ≥ 5 (ledger, math check).
- The local rule (one press toggles a plus-shaped set of lights) is learnable and holds in unseen
  configurations: the patch-token WM effect accuracy is .85–.89 regardless of novelty, including a
  held-out region (P1 jobs 56701, 56900).
- But a continuous latent WM reproduces the whole board exactly in only 60–66% of events, so long
  imagined chains drift.

**Closest prior art and what is not novel.**
- **DeepCubeAI** (RLC 2024) and **SPAR** (RLC 2026): a discrete latent WM learned from pixels, a
  cost-to-go learned inside it, and heuristic search. They solve >99% of pixel Rubik's cube, Sokoban,
  IceSlider and DigitJump. They assume a given discrete action set and puzzle environments.
  - So "a heuristic learned in a learned model" is not our novelty.
- Others:
  - DWMR 2603.01748: JEPA-style Boolean WM with a locality prior; no planning.
  - PPGS (NeurIPS 2021): pixel latent plus graph search with discrete actions.
  - LatPlan.
  - THICK (ICLR 2024) and Hi-LeWM 2607.12547: temporal abstraction, without combinatorial search.
  - Bilevel planning with learned predicates: state-based.

**Remaining delta (the claim to test).** Make WM-based heuristic search work for continuous robot
manipulation learned from offline pixel play data. That requires solving three problems that the
puzzle-domain methods do not face:
- (a) discover discrete events without a given action set;
- (b) keep the event-level model exact over 10–24 events (a discrete code with rounding);
- (c) execute each event with a learned low-level controller and replan.

**Hypotheses.**
- **H1:** event effects are local, so an event-level WM learns them and applies them in unseen
  configurations.
- **H2:** discretising the event-level state prevents error from accumulating.
- **H3:** a cost-to-go learned by Bellman backups over imagined events, plus search and replanning
  after each event, turns H1 and H2 into closed-loop success above model-free baselines, with a
  flatter curve over grid size.

## 2. Method (current instantiation)

| Stage | What it does | Script | Key design choice and why |
|---|---|---|---|
| Encoder | ViT-tiny, LeWM recipe, trained per grid size and then frozen; patch tokens 64×192 | `config_generalization_20261001/scripts/train_wm.py` (`slurm/base.sh`) | Patch tokens, not CLS: CLS loses button state (68% bits vs 99.99%, P1 job 56691) |
| Event code | Frame → binary bits that change only at events, without state labels. **Current: linear SFA + band-wise ICA** (`sfa_code.py --bands --ica-dims 64`): PCA-whiten tokens → SFA → ICA below and above the eigen-gap → keep slow (binarised flip-rate 2-means) and binary (bimodality 2-means) components | `sfa_code.py` (`train_code.py` MLP head kept as a failed alternative) | Slowness separates object state from the arm. The flip count, not the variance, is the right slowness for binary state: occlusion makes far-row lights wobble without flipping. Bits allow exact re-identification and rounding |
| Code reader | CNN on the raw frame, trained on segment-majority pseudo-labels of the code (state is constant between events) | `train_reader.py`, `closed_loop.py --reader` | Replaces the linear readout at test time (frame-exact .988 → .998). Goal frames are read once, and a single wrong light breaks the plan |
| Events | Per-bit debounce (3 frames), then merge code changes within 10 frames into one event; type = XOR pattern; vocabulary = patterns covering 99% of train events | `common.detect_events`, `build_events.py` | A press reveals its 3-5 lights over several frames. Puzzle-specific: the XOR pattern is state-independent only because Lights Out effects are (§5) |
| Event WM | MLP f(b, e) → b′ trained on detected events | `planner.py` / `train_planner.py` | Rounded at every step, so plans of any length do not drift |
| Cost-to-go | h(b, g) ← 0 if b = g, else 1 + minₑ h_tgt(f(b, e), g). Goals are generated by up to 30 imagined random events; target net copied every 1000 steps | `train_planner.py` | Min over all events (the WM enumerates them), 1 unit per event, and unlimited imagined states. This is what a value learned from play data lacks |
| Search | Batch weighted A* (λ = 0.6, 256 nodes per iteration), goal test = exact code match | `planner.bwas` | DeepCubeA's BWAS |
| Skill (current: v3) | `train_skill3.py`: CNN on two raw frames, end to end, FiLM event conditioning (type embedding + spatial change map), 8-action chunks, segments = approach + press + lift-off (`--release 3`), last 21 frames before the press (`--max-tau 20`) | `train_skill3.py` | Frozen-token skills (v1/v2) never reach edge buttons and re-press. The lift-off makes the change visible, which ends the skill |
| Skill (v1, superseded) | π(a ∣ patch tokens, e, τ) by behaviour cloning on the segment before each event; τ = steps until the event | `train_skill.py` | Play segments start with ~14 steps of lift-and-wander from the previous press; τ separates that from the approach-and-press. Test-time τ counts down from `--tau0` (15) |
| Closed loop | Encode current and goal frames → plan → execute first event → detect a stable code change (2 frames) → replan; timeout of 80 steps per event | `closed_loop.py` | Replanning after every event absorbs wrong presses and code noise |

Data access: `cache_data.py` streams the first N episodes of each compressed `.npz` into
memory-mappable `.npy` (`cache/<env>/{train,val}_*.npy`). The 4x5/4x6 frame arrays are 37–61 GB
uncompressed. `button_states` is cached for **evaluation only**.

## 3. How to run (puzzle-4x5 example)

```bash
sbatch slurm/inventory.sh                    # data stats, eval-task observations, scripted-executor check
sbatch slurm/cache.sh                        # .npy caches (ENVS=... to select sizes)
ENV=visual-puzzle-4x6-play-v0 sbatch slurm/base.sh       # frozen encoder for another size (4x5 uses P1 56657)
sbatch slurm/code2.sh                        # event code: SFA+ICA and MLP variants, with privileged scoring
CODE=<code.pt> sbatch slurm/pipeline.sh      # events -> planner (+offline checks) -> skill -> closed loop
```

`pipeline.sh` options:
- `STAGES="events planner skill loop"` selects stages; `OUT=<dir>` reuses an earlier run.
- `EPISODES` sets episodes per task.
- `ARMS="skill learned|scripted learned|skill oracle"` selects closed-loop arms.
- `PLANNER_ARGS`, `SKILL_ARGS` and `LOOP_ARGS` pass extra arguments to each script.

Every slurm script calls `record_source`, which writes source hashes into its run directory.

## 4. Diagnostics: where to look when closed-loop success is low

| Stage | Output file | Metric (privileged scoring unless noted) | Healthy value |
|---|---|---|---|
| Code | `code_eval.json` | `per_light_best_bit_acc`, `lights_with_bit_acc_ge_0.99`, purity, event precision/recall, `pattern_to_button_purity` | every light ≥ .99; purity ≈ 1; recall ≈ 1 |
| Events | `events_report.json` | `vocab_size` (= number of buttons), `type_to_button_purity`, `matched_true` | vocabulary = buttons; purity ≈ 1 |
| Event WM | `planner_eval.json` | `wm_val_exact` | ≈ 1 |
| Cost-to-go | `planner_eval.json` | `h_spearman_dstar`, `h_event_sign_by_dstar` (same metric as the P1 GCIVL wall) | sign accuracy stays high beyond d* 7 |
| Search | `planner_eval.json` | `offline_plans` (solved and length/d* by d* bin), `offline_tasks` | solved ≈ 1; length ≈ d* |
| Skill | `skill_log.json` | validation action MSE by τ bin | low at small τ |
| Closed loop | `loop_<low>_<high>/closed_loop.json` | success by task; per-event commanded vs true toggled button; replans | — |

**Attribution arms** (PRIVILEGED, debugging only; never reported as method results):
- `--low scripted`: executes planned events with the scripted press controller on the true button.
  This isolates the learned high level.
- `--high oracle`: plans with the true GF(2) minimal press set mapped to event types. This isolates
  the learned low level.

**Evidence rules.**
- Label every number as offline, privileged or closed-loop.
- Development numbers on 4x5 are not paper evidence. Paper claims need held-out seeds, all grid
  sizes, baselines under the same protocol (or published numbers named as such) and ablations.
- Ablations:
  - GCIVL value instead of the imagined cost-to-go;
  - continuous instead of discrete latent;
  - no replanning;
  - CLS instead of patch tokens;
  - dataset nodes instead of imagined nodes.

## 5. Status, known limitations, next steps

Status as of 2026-10-03. Details are in `JOB_LEDGER.md`.

**Status.**
- **Inventory done.** Play data has ~30 presses per episode, ~33 steps apart. The scripted executor needs ~10 steps per press, so the step limit is not binding.
- **Event code: solved label-free on 4x5 in round 5** (`sfa_code.py --bands`, job 56938).
  - Recipe: SFA on whitened patch tokens; ICA separately below and above the eigen-gap of the slowness spectrum (N = 64); keep components that are slow (binarised flip rate) and binary (bimodality).
  - Result: 20/20 lights at >= .989 agreement, code purity .9995.
  - How the earlier rounds failed:
    - the MLP head finds only the 7 near-row lights, and collapses under stronger slowness;
    - joint ICA over a wide subspace drops those 7 lights;
    - quadratic slowness alone misses the 13 far-row lights, which the arm often occludes.
- **Event detection.** Per-bit debounce, then merge changes within 10 frames (`common.detect_events`). A press reveals its lights over several frames.
- **Code refinement** (`refine_code.py`, job 56951): segment-majority pseudo-labels raise frame-exact reading from .937 to .985, with event precision 1.0 and recall .999.
- **Cost-to-go.** Width 512 saturates at ~5 presses. An exact-label check (56943) shows width 2048 learns the distance perfectly. Planner v2 (width 2048 + b xor g) moves the sign-accuracy wall to ~12 presses.
- **Rendering.** It is CPU software on this cluster (~0.155 s/frame, EGL or OSMesa), so closed loops run parallel workers (`--workers`). CPU-node OSMesa produced black frames once (56924); the closed loop aborts on dark frames.
- **First closed loop (56965, dev).**
  - Method 0/40.
  - Scripted presses + learned high level 15/40: tasks 1-2 solved, tasks 3-5 broken by a one-bit noise event type in the vocabulary.
  - Learned skill + oracle plan 0/40: occlusion-induced code changes reset the skill.
- **Round 2** (56980/56981): count-based vocabulary (exactly 20 event types), WM event-consistency filter in the closed loop, cost-to-go trained 150k steps.
  - **Offline:** the cost-to-go is correctly signed at every distance (.99+, Spearman .954), and all 5 official tasks are planned optimally.
  - **Closed loop: PRIVILEGED scripted presses + learned high level = 29/30 (96.7%)**, each success with exactly d* presses. The learned high level is no longer the bottleneck.
  - **Method with skill v1 (tau / tau-free) = 0/30.** The skill rarely completes a press, so the low level is the bottleneck.
- **Low level.**
  - Skill v1/v2 (frozen tokens) reach at most 7/30 even with the oracle plan: edge buttons are never reached, and presses repeat.
  - **Skill v3** (`train_skill3.py`): a CNN on two raw frames trained end to end, FiLM event conditioning, 8-action chunks, shift augmentation, and **press-and-release segments** (`--release 3`; without them the skill re-presses because the change is only visible after lift-off). It reaches **30/30 with the oracle plan**: exactly d* presses, ~30 steps each.
- **Fully learned method on 4x5.**
  - Dev: 29/30.
  - Official protocol (5 tasks x 20 episodes): 94/100 with the linear code reader (seed-1 episodes; all 6 failures were one misread goal light).
  - **CNN code reader** (`train_reader.py`): trained on the pipeline's own segment-majority pseudo-labels, frame-exact .9977 vs .988. It gives 100/100 on the seed-1 episodes, but it was introduced after inspecting them.
  - **Clean fresh held-out seeds (seed 2): 99/100, 95% CI [97, 100].** The one failure is a goal misread with a low-confidence bit (|logit| 2.0).
  - Published pixel results on 4x5: <= 17%.
- **4x6.** Same recipe, no 4x6-specific tuning except the search budget (200k expansions).
  - Code: exactly 24 components. Events: 24 types. CNN reader .999.
  - Cost-to-go is correctly signed .97 up to d* 12 and .85 beyond.
  - **Official protocol, fresh seeds: 100/100**, every episode with exactly d* presses (task 5, d* = 24, in ~685 steps).
  - Published pixel results: <= 15%; SHARSA (state, 1B transitions) 14%.
- **Not yet done:**
  - 3x3/4x4 (complexity curve);
  - several training seeds per learned component;
  - same-protocol baselines;
  - ablations;
  - a non-puzzle domain (cube), where event identity must not equal the XOR pattern.
  - GPU quota at the time of writing: 24/30 h (monthly rule).

**Known limitations and open problems.**
- Event type = XOR pattern of code bits is valid only for state-independent effects (Lights Out).
  - Cube and scene need an event identity that does not equal its effect, e.g. where the interaction
    happens (the patch-change location), so that the WM has to learn state-dependent effects.
- The number of code bits and the slow-bit selection must stay label-free.
  - Current rules: a 2-means split on log flip rate, and the largest eigen-gap for SFA.
- The skill's τ0 = 15 comes from the play oracle's timeline. Tune it on validation episodes, not on
  evaluation tasks.
- Only the 4x5 encoder exists. 3x3, 4x4 and 4x6 need `slurm/base.sh` (~50 GPU-min each).
- No second domain yet. Candidates are CALVIN (HF mirror) or Franka Kitchen; both unverified.

**Next steps, in order.**
1. ~~A code that captures all lights label-free~~ (done on 4x5; must be repeated per grid size).
2. ~~Full pipeline on 4x5~~ (dev 29/30; official-protocol eval running).
3. Encoders and pipelines for 3x3, 4x4 and 4x6 to get the complexity curve.
4. Baselines and ablations under the same protocol.
5. An event identity that generalises to cube-triple.

## 6. Cube extension (in progress; user priority 2026-10-03)

**Why.** In Lights Out an event's effect does not depend on the state, so event type = XOR pattern of code bits works. That will not hold elsewhere, and it is also the point closest to DeepCubeAI. visual-cube-triple (published pixel SOTA ~21%) needs:
- continuous object state (cube positions);
- events "move cube k to place q" with state-dependent feasibility (only a top cube can be picked; a target can be occupied);
- real combinatorics: the cycle task needs a temporary placement, and the stack/unstack tasks need ordering.

**Planned recipe** (same principles, label-free):

1. Frozen LeWM encoder per domain → SFA on patch tokens → ICA in the slow subspace. Continuous cube coordinates should appear as independent slow sources.
2. **Objects = sources that change together.** A carried cube changes all its coordinates during one carry. Group sources by co-change; each group is one object's state p_k.
3. **Events = one group's change interval**, giving (k, p_before, p_after) from play data.
   - Event WM f(s, k, q) → s′ is learned, so state-dependent effects such as stacking heights are learned rather than hand-coded.
   - Feasibility not seen in play data is the main open risk.
4. **Search** over moves (k, q), with q drawn from the goal state's positions plus free buffer positions from the data. Cost-to-go comes from value iteration in the WM, as in the puzzle.
5. **Skill:** v3 CNN, conditioned on object k and target q (continuous), trained on carry segments with hindsight (k, q) through release + lift-off.
6. Closed loop as before.

**Step 1 (jobs 57041-57043).** Data inventory, cube encoder, and whether SFA/ICA sources track cube coordinates and co-change groups recover cube identity. Privileged qpos is used only to score this.

**Step 1 result (57043): negative for the token route.**
- Linear R² from the slow token subspace: cube x .38–.57, y .60–.79, z ≈ 0.
- ICA sources do not track coordinates.
- A moving object's position is a nonlinear (spatial) function of patch tokens, so the binary-code recipe does not carry over directly.

**Direction D (user choice 2026-10-04: "D first, then A"): privileged object state, everything else learned.**
- Purpose: check whether the high level and the skill transfer to cubes before replacing perception.
- `cube_events.py`: moves from qpos.
- `cube_planner.py`:
  - event WM f(s, k, q_xy) → s′, where the resulting height and stacking are learned;
  - affordance a(s), learned from which cube the play data moves;
  - BFS over (k, q), where q ∈ {goal xy of every cube} ∪ {6 free buffers drawn from data placements}.
  - No cost-to-go: cube tasks need ≤ 4 moves.
- `train_cube_skill.py`: skill-v3 recipe, conditioned on (k, q_xy).
- `cube_closed_loop.py`:
  - a move ends when cube k has moved > 2 cm and then stays still for 5 steps;
  - replan after every move;
  - timeout 250 steps, then the recovery action.

**Direction D result (PRIVILEGED, dev seed 0, 6 episodes × 5 tasks, job 57118): 26/30 = 86.7%.**
- By task: 1 6/6, 2 3/6, 3 5/6, 4 6/6, 5 6/6.
- **Offline (57116):**
  - WM: moved cube median error 5.6 mm, 99.2% within 4 cm (stacked moves 98.8%).
  - Affordance: 0.047 mean probability on covered cubes vs 0.37 on free cubes.
- **High level:** the initial plan is optimal in 29/30 episodes; planning takes ≤ 0.24 s.
  - Exception (task 4 ep 5): a 3-move plan that "pulls the bottom cube out", which exploits the WM/affordance on covered cubes. Replanning recovered it.
- **All 4 failures are skill failures:**
  - task 2: grasp misses, so the cube is pushed 3–4 cm in ~32 steps and drifts on retries;
  - task 3: unstacking topples the stack (7 knocks in 23 moves).
- The skill overfits: val chunk MSE is best at 10k steps (0.363) and 0.438 at 60k. `cube_skill_best.pt` is saved from the next training run onward.
- **Not a paper number:** object state is privileged. Published pixel SOTA on cube-triple is ~21%.

**Direction A (in progress): label-free object state (`cube_discover.py`, `slurm/cube_a.sh`).**
- Objects = what events move.
- Pipeline:
  - background = per-pixel median;
  - foreground colour k-means;
  - object clusters = compact and slow (2-means on each score, as in the puzzle code);
  - merge clusters with coinciding centroids;
  - per-object robust centroid (u, v) in pixels.
- Then:
  - moves from the tracks (same rule as `cube_events.py`);
  - WM / affordance / skill in image coordinates (q = hindsight final pixel position);
  - goal state from the goal image through the same perception.

**Direction A status (2026-10-04): label-free cube-triple ≈ 50%** (13–17/30 across 5 closed-loop variants, dev seed 0; the CIs overlap). D (privileged) is 26/30.

**Label-free pipeline, each stage checked offline (PRIVILEGED diagnostics only):**
1. `cube_discover.py`: 3 objects found = 3 cubes.
2. `cube_events_px.py`:
   - moves with precision .98 / recall .88, thr = one object width;
   - coverage bit = never visible in a static interval, agreeing with qpos at .97–.98.
3. `train_cube_reader.py`: heatmap reader, 0.15 px median; coverage accuracy .976.
4. `cube_planner.py` (pixel + coverage): moved object 0.41 px; covered objects never moved; no target inside an occupied place.
5. `train_cube_skill.py` (optionally `--support`).
6. `cube_closed_loop.py --perception reader`:
   - move end = the play-data rest detector run online;
   - optional `--belief` (WM-consistent coverage).

**Known failure modes (from logs):**
- skill grasp misses and pushes, also seen in D;
- stacking: "on top of B" vs "in front of B" project to the same pixel; addressed by coverage + support conditioning, effect unmeasured;
- covered objects are read poorly by the reader; addressed by the belief, effect unmeasured.

**Open:** attribution of the A-vs-D gap (skill vs perception vs move-end detection), and more episodes per arm.

**Attribution (2026-10-04, PRIVILEGED arms, matched seeds):**
- D skill vs A skill under the same privileged high level: 26/30 vs 17/30.
- Perfect perception + A planner + A skill: 20/30 (support skill 16/30).
- Label-free A: 17/30.
- So the A skill is the bottleneck, and perception and planning cost ≤ ~3/30.
- Next: fix the skill.
  - Training segments: event recall .88 means merged moves.
  - Target ambiguity when stacking.

**Skill fix (2026-10-04).**
- **Diagnosis** (under the privileged high level):
  - The pixel target space and the number of training steps are not the bottleneck: qpos segments with pixel targets give 25/30 ≈ D's 26/30.
  - The label-free segments were: colour-track ends lag the true end by ~14 frames, so segments started ~13 frames late and frames right after a release were only labelled with the previous target.
- **Fix:** `cube_events_refine.py`.
  - The reader's arrival time ends a move (−3 frames vs qpos).
  - Segments are rebuilt; segments containing another object's undetected move are excluded (5%).
- **Results:**
  - Under the privileged high level: 17 → 21/30; placement within 4 cm .43 → .54; knocks .24 → .05.
  - Label-free full loop over 60 matched seeds: 60% → 67% (p = .48); knocks .14 → .04.
- **Remaining:** stacking, especially level-3 placements (label-free recall .78, 371 training segments).
