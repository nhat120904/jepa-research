# V2 plan: execution summary (read this first)

Produced 2026-10-10 by a design-panel workflow (4 independent designs: attribution / low level / system / evaluation;
3 judges: soundness, feasibility, paper value; synthesis; completeness critic; revision; Sonnet subagents). The full
synthesis follows below unchanged. Some of its numbers tagged (m) come from 3-episode privileged forward-kinematics
measurements the agents made while designing; they are design premises, not results, until re-measured.

## Adopted backbone (judges: soundness and feasibility chose design A; paper value chose D's protocol)

- High level: event WM + cost-to-go + weighted A* (kept), with WM v2 changes (relative positions, identity dropout,
  ensemble, soft labels) after the attribution slice.
- Attribution: a learned EFFECTOR TRACK (inverse dynamics with actions integrated over 1-32 steps, clipped at learned
  workspace bounds, a learned camera map, a keypoint and a height) and CONTACT RUNS (the effector at the table floor:
  every press is a floor dip in the play data), with a 4-6 number tip-offset calibration from contact clouds; acted =
  the entity at the calibrated contact point. No hand rule on the arm mask.
- Low level: goal-conditioned executor on the next abstract state ("Delta map" goals, single frame + gripper cue,
  chunked policy), value / AWR only if a measured single-event deficit asks for it; a plain image-goal GCIVL as the
  easy-task baseline and fallback.
- Protocol: dev = puzzle-3x3, 4x4, 4x5, cube-single, cube-triple, scene (3x3 / 4x4 were used to design the perception
  fixes, so they are no longer held-out); held-out = puzzle-4x6, cube-double, cube-quadruple (+ novel boards if the
  installed ogbench can generate them); privileged data only in scoring scripts; frozen code for held-out.

## Order actually executed (cheapest decisive test first)

1. v1 effector (method/effector.py) scored on presses: done 2026-10-10 01:05-01:20 (3x3 acted .90 at the contact
   moment vs .58 for the rule; 4x4 .40: the heatmap is diffuse) -> EffectorNet v2 is needed.
2. EffectorNet v2 + contact runs on 3x3 / 4x4 / 4x5 (GPU ~15 min each); gate D2: press recall / precision of contact
   runs >= .95, timing within 2 frames >= .90, acted >= .85 on all three boards.
3. Oracle ladder on 4x4 then 4x5 (PRIVILEGED press-matched labels -> WM -> scripted loop; then true events): decides
   whether the high level works once attribution is right (decision D1).
4. events v2 with learned attribution -> WM v2 -> scripted loop (R1c) -> learned executor -> full closed loop on dev,
   then the frozen held-out run.

Deviations from the full plan below: the protocol / firewall / freeze files are built incrementally next to the
experiments that need them, not as an upfront 2-day refactor; nothing is committed without the user's approval.

---

# Event WM v2 plan (final synthesis, revised 2026-10-10 after critic review)

Status: plan only; nothing in it has been run. Tags:
- (P) means PRIVILEGED scoring: simulator state is read to score and never enters the method.
- (m) means measured on 3 VAL episodes per env. It must be re-measured on 100 episodes (E0) before anything depends on it.
- (v1) means read from the v1 effector logs of 2026-10-10.

**Splits (changed from the previous draft).**
- DEV: puzzle-3x3, 4x4, 4x5, cube-single, cube-triple, scene.
- HELD-OUT: puzzle-4x6, cube-double, cube-quadruple, plus novel boards (E23).
- cube-single is re-designated DEV in `PROTOCOL.md` before any further use. Its qpos was already read once for the physics premise, and the cube phase work, the T1 harness and the contact-run checks all need it. The paper says so plainly.
- 3x3 and 4x4 are DEV because the 10-09 perception work was designed on them.

Execution rules for this plan:
- The repo `AGENTS.md` and `CLAUDE.md` apply in full wherever a job runs, not only on the cluster.
  - Work is carried through implementation and closed-loop evaluation.
  - There are no stacked go/no-go gates before implementation. This plan has two decision points (D1, D2) that choose between branches. Everything else is a checkpoint reported while the next step is already running.
  - Offline, privileged-oracle and learned closed-loop numbers stay separate. Every result JSON stores its tier.
  - Paper claims use matched controls and held-out evidence.
- Local jobs run through `local/run_stage.ps1`, one GPU job at a time, with the RAM watchdog and CSV-resume.
- Every submitted job is recorded in `JOB_LEDGER.md`. Check the ledger and the output dirs before launching, to avoid duplicates and to preserve peer edits and existing run artifacts.
- If any job moves to the cluster, the repo rules apply: time-limited `sbatch`, the `sreport` top-5 check, and never touching `nhatnc129`'s standing. Ledger line 1332 recorded only about 33 GPU h per month of headroom under the 90%-of-5th-rank rule. The cluster therefore cannot absorb the baselines, and the budget in Sec. 6 is local.
- `method/` is currently untracked. The user is asked to commit it (or approve a commit) before M1, so that "commit hash in every result JSON" and the freeze tag mean something. Nothing is committed without that approval.

---

## 0. Choices at a glance

| Decision | Taken from | Reason |
|---|---|---|
| Attribution signal = a learned unit-gain effector track (inverse dynamics on frame+action windows), fused with weaker experts. Geometry is the anchor of the fusion and its weight is never fitted. | Design A | It is the only signal not computed from the late, partly hidden entity readings that defeated every earlier fix. Its physics premises were measured (m), but the learned keypoint reaching them is not yet shown (Sec. 3.1). |
| Evaluation protocol: dev/held-out re-designation, hash freeze, privileged firewall as a dataflow guarantee (blind cache), constants ledger with data-derived classes, pre-registered decision rule, error-budget ladder, claim ladder | Design D | The central claim is currently unsupported (3x3 3%, 4x4 7%, n=30 episodes, contaminated "held-out"). The protocol makes it falsifiable. |
| Executor = outcome-conditioned "Delta map" goal, single frame, gripper cue, chunked flow policy, hindsight on realised abstract states. IQL value, AWR and chunk critic only if a measured T1 deficit asks for them. Label-free press-point channel as an ablation arm. | Design B, plus an effector channel from A | No dataset acted-entity label reaches execution. It is the best-founded fix for the copy-motion / lost-grasp-phase problem. |
| Easy-task path = a separately trained plain image-goal GCIVL (baseline and fallback) plus a "direct final-goal Delta" mode. Chosen by a label-free competence rule measured on VAL pairs. | Design C (arm), Judge soundness (rule) | The easy-task guarantee becomes real, and good episodes are no longer demoted by a stall count. |
| WM: relative-position inputs normalised by place spacing, identity dropout, 3-model ensemble, confidence-weighted soft labels, soft-completion arm for unknown own after-states | A, C and D agree | Targets the 4x5 light-2 failure (157 of 177 planned presses surprised the model). |
| Tested fallback if the effector fails: code-keyed WM (latent action codes replace the entity id) | Critic review | A G0b-style failure has a concrete branch, run as a rung of the oracle ladder (Rc). |
| Execution order: a vertical slice first (effector -> contact runs -> geometry attribution -> events -> WM -> existing scripted loop on 4x4, then 4x5). Executor work follows the slice, because the privileged scripted press already ties the learned skill on 4x5. | Judge feasibility, critic review | Time to the first decisive end-to-end number is days, and the executor is not what limits 4x5 today. |
| Dropped from v2 | | Hand-built servo as a method component (kept as a privileged/diagnostic arm), goal-view synthesis, the logistic arm gate, the occlusion head, container proposals and hidden-goal machinery (out of claims until after C3), the independence claim for Dawid-Skene, the MIL contact heatmap and EM relabel (kept only as ablation anchors), the stall-count switch to D_i as the default. |

---

## 1. Goal and principle

### 1.1 What must be earned

One frozen formulation, learned from offline play pixels, that is strong on easy tasks and on long combinatorial ones, and that transfers to environments whose code was never tuned on. The targets are puzzle-3x3 and 4x4 (HIQL 73 and 60 on pixels), puzzle-4x5 and 4x6 (best pixel baselines at most 17 and 15), and cube-single (HIQL 89).

Today (v1, seed 0, 30 episodes, SE about 7 points): 3x3 3%, 4x4 7%, 4x5 23%, cube-triple 13%, scene 23%. A 23% vs 17% difference on 4x5 is under one SE. 3x3, 4x4 and cube-single are DEV and cannot count as held-out evidence.

### 1.2 Why a hybrid, and what must hold

Event search removes the horizon problem. A 4x5 task needs about 20 events, and flat goal-conditioned RL collapses there. Flat GCRL is accurate on short goals, which is why HIQL and GCIVL are strong on 3x3 and cube-single. So the high level (event WM, cost-to-go, weighted A*) must emit goals only one event ahead. The low level must be goal-conditioned and reliable on exactly those goals. An easy task must have a path that is at least as good as flat GCRL.

Per-event reliability arithmetic: an open-loop plan of N events succeeds with p^N. At p=.95 and N=20 that is .36; at p=.98 it is .67. Closed-loop retries change this, and the budget model of Sec. 5.5 (E25) quantifies it. It links per-attempt success, verification delay, timeout and the horizon, and it sets the T1 thresholds. Until it is validated against the R3 ceiling, .90 is the working per-event threshold and the loop must recover from the remainder by verification and replanning.

The ledger of where v1 loses, and which v2 component owns each loss:

| Loss | Evidence | Cause | Owner |
|---|---|---|---|
| Wrong acted entity trains the WM, support and cost-to-go on wrong effects | acted .58 (3x3), .69 (4x4), .883 (4x5; bottom row 14/15/17/19 at .14/.41/.26/.30, "the light above" chosen); 3x3 edge presses 5/7/8 at .33/.31/.07 (P) | Every earlier signal is a function of the same late/partly hidden readings, or a free map with a lattice-shift fixed point (EM relabel .46 to .37, MIL contact .10, mask centroid 19%, AgentNet sensitivity .49 on 4x4) | Sec. 3 |
| Over-segmentation, late flips as separate events | 49 / 44 / 34 events per episode vs 31 presses (P); pressed-light reading lag median 11 / 7 frames, p90 27 / 19 (3x3 / 4x4) (m); 15% / 9% never flip within 60 frames | Events are anchored on change windows of readings, not on contacts | Sec. 3.5-3.6 |
| Pressed light read as unchanged by its own event | 26% (3x3), 11% (4x4) | Read under the gripper; lagged SeeThrough | Sec. 3.5 |
| WM cannot generalise to rarely acted entities | 4x5 light 2: 157 of 177 surprises; the pressed light's own after-state is the least readable, so masking biases the data against rare or covered lights | Identity-specific effects plus label bias | Sec. 2.3 (relative-position WM, soft completion) |
| Low level copies motion or loses the grasp phase | two frames: condition-swap sensitivity .095-.13; one frame: .41, cube grasp 0/6; exact presses .865 (4x5, one frame), .35 (4x4 default) | BC on attributed labels; velocity shortcut | Sec. 4 |
| Perception hole | 3x3 button 1 missing (8 of 9); plan of length 0 in 14 of 30 3x3 episodes | Place under the arm base, possibly never visible | Sec. 3.8, E16 |
| Easy tasks | 3x3 3% vs 73% | Abstract pipeline is the only path | Sec. 5.4 |
| Step budget | about 20 events x about 32 steps vs 1000 | Fixed 250-step timeouts, long plans | Sec. 5.5 |
| Cube-triple and scene (13% vs HIQL 21; 23% vs 49) | no owner in the previous draft | Position precision, grasp phase, hidden goals are untested | E26 (ownership analysis), Sec. 6 M2 |
| Evidence | n=30, one seed, contaminated splits | Protocol | Sec. 7 |

### 1.3 Principles

1. The attribution signal must not be a function of the readings it explains. The effector track is learned from (frame, action) windows and never sees entity changes.
2. The executor is trained on realised outcomes (hindsight) and no dataset acted-entity label reaches its training. The planner's planned position may be given to the executor at test time only through the press-point ablation arm (Sec. 4.1), whose training value comes from the effector track and not from a label.
3. The high level emits goals one event ahead. Longer goals are an ablation (horizon k), not the default.
4. Each learned piece has a numeric check, but checks do not block downstream implementation. Only D1 and D2 (Sec. 6) choose between branches.
5. One general formulation: no bits/XOR, no per-task shapes, no proprio and no labels as inputs. Privileged simulator state is only for scoring and for resetting the simulator in the T1 harness and the calibration rollouts. Every decision threshold is derived from data by a stated rule (class (a) below).
6. The easy-task path is a separately trained baseline, so the guarantee does not depend on our perception.

**Non-negotiables (enforced by tests).**
- **Blind cache.** `*_qpos.npy`, `*_button_states.npy`, the privileged diagnostic JSONs and any simulator-state array live in `cache/<env>/_priv/`. Method stages receive a blind `--cache` directory that does not contain `_priv`. `local/run_stage.ps1` passes the blind path and sets `EVENT_WM_BLIND=1`.
- **`tests/test_blind_pipeline.py`.** It runs a small fixture end to end (front end for a few steps, objects, events, WM, support) twice, with and without `_priv` present, and asserts bit-identical outputs. This is the dataflow guarantee.
- **`tests/test_firewall.py`.** It is a second layer. It greps `method/*.py` for `qpos`, `button_states`, `_cur_button_states`, `site_xpos` and `privileged`. Only an explicit allowlist (`diag_privileged.py`, `priv_loop.py`, `calib_rollouts.py`, `skill_eval.py` for the reset path only) may match. The privileged scripted/oracle loop modes and the `sim_start`/`sim_goal` logging move out of `closed_loop*.py` into `priv_loop.py`. Docstring mentions are rewritten.
- **`method/CONSTANTS.md`.** Every constant has a class and provenance.
  - Class (a): decision thresholds. Each has a derivation rule implemented in code (Sec. 1.3.1).
  - Class (b): global training hyperparameters, declared fixed across envs and never tuned per env.
  - `tests/test_constants.py` fails on a class-(a) entry without a rule or without a 0.5x/2x sweep entry.

#### 1.3.1 Constant classes (the rules, not just provenance)

| Class (a) constant | Derivation rule |
|---|---|
| Reliability cutoff `rho*` | Smallest `rho` whose misread rate against the next stable reading is at most the at-rest flip rate between consecutive stable frames in static intervals, plus one SE |
| Null-slot radius `r_null` | The radius at which a 2-component EM (Gaussian around the nearest place plus a uniform background) gives responsibility .5 to the background |
| Mean-shift bandwidth `h` | Start `h0 = .2 x` median nearest-place distance; then 1.0 x the robust (MAD) sigma of the confident geometry residuals, re-estimated per EM iteration |
| Route C match radius `tau` | Median nearest-place distance among discovered places (defined whenever there are at least 2 places; no lattice assumed). With fewer places Route C is skipped and Route M (3.3) is used |
| Reach radius | 2 x the robust sigma of the confident residuals |
| Keyframe window | TRAIN median of (arrival - core departure) of events |
| Interaction merge gap `delta` | 2-means split of log gap, Ashman D above 2, else no merge |
| Relative-position scale | Median nearest-neighbour place spacing |
| Cue gating and midpoint | Ashman D above 2 on the gripper command over TRAIN, midpoint of the 2-means split. If not bimodal (puzzle), the cue is disabled by the same code path |
| Ensemble penalty offset `d0` | 90th percentile of `d_ens` on VAL events |
| Q margin rule | 75th percentile of the Q spread on VAL |
| Failure-memory cost | In units of the median per-event planner cost |

Class (b), fixed across envs: gamma .985, goal-mixture weights, dropout rates (.15, .05, .3, .5), EMA span 8 (two query periods), chunk H=16 and E=8, expectile .9, planner lam .6, per-event timeout multiplier 3, learning rates, batch sizes and step counts. Class (b) values are chosen once on dev T1 and then frozen.

### 1.4 Honest framing

The contribution is a per-env recipe: a learned abstraction from offline play pixels, with the front end retrained per env with identical settings. "Held-out" therefore tests recipe transfer, not model transfer. Model-level transfer is offered only as optional evidence: pooled-board low-level training (E21b) and novel boards (E23). Actions are used as a training signal (effector track, AgentNet, hindsight GCRL) and as online inputs (the causal contact detector, the executor's own gripper-command cue); the paper says this plainly. The tier structure of the claims is in Sec. 7.5.

---

## 2. Architecture

### 2.1 Pipeline

```
pixels + actions (play data, per env; first 1000 episodes = 1M frames; blind cache)
  -> front end (KEPT): AgentNet, Segmenter, SceneCodes, SeeThrough (partial-label) + views
  -> objects (KEPT): places + movers, per-frame readings (objects.py --view)
  -> EffectorNet v2 (NEW): frame -> (kappa u,v ; zeta), tip-like point + height   [learned from frame+action windows]
  -> contact runs (NEW): floor-run now, ContactNet later (M4); label-free per-entity selector (3.2)
  -> attribution (NEW): calibrated tip offset (Routes V, C, M); fused posterior over acted entity (geometry weight fixed); KM lag model
  -> events_v2: A0 change-window (KEPT) | A1 contact-linked, one-to-one assignment + merge (leg 1) | A2 contact-anchored (leg 2)
  -> WM v2: relative-position, identity dropout, 3-model ensemble, soft-label weights, soft completion; support; cost-to-go; weighted A*
        (branch: code-keyed WM where the acted token is a latent action code, Sec. 3.9)
  -> waypoint S'_k = Model.step chain  -> frozen intended Delta maps (no acted entity label)
  -> GC executor (NEW): single-frame flow-chunk policy (L0) -> (+IQL value/AWR, +chunk critic only on measured T1 deficit)
  -> closed loop: belief (belief.py), contact-end belief fill, event_status, replan, competence-based arbitration -> plain image-goal GCIVL (D_i)
```

### 2.2 Component table

| Status | Component | Files | Learned / rule |
|---|---|---|---|
| KEPT | Front end per env incl. partial-label SeeThrough and views | `frontend.py`, `frontend_train.py --partial`, `view.py` | learned |
| KEPT | Objects (places + movers), readings | `objects.py --view` | rule (data-derived tests) on learned codes |
| KEPT | Change-window events, labels/changes npz (arm A0 and input to A1) | `events_objects.py` | rule |
| KEPT | Event support, cost-to-go by value iteration, weighted A*, closed loop v1 (regression path) | `event_support.py`, `train_support.py`, `world_model.py`, `closed_loop_objects.py` | learned + search |
| MODIFIED | `effector.py`: the existing file is v1 (32x32 soft-argmax, inverse model on d in {1,2,4}); v2 adds the unit-gain clipped-integrator loss. v1 stays as `--mode v1` (ablation). | `method/effector.py` | learned |
| MODIFIED | WM: `--rel --id-dropout --weights --ensemble --complete`; `--acted {entity,code}` | `world_model.py`, `event_support.py` | learned |
| MODIFIED | Loop: `--low gc --goal-horizon --goal-mode`, contact-end belief fill, event_status, data-derived timeouts, failure memory, competence arbitration. The privileged scripted/oracle arms move out to `priv_loop.py`. | `closed_loop_objects.py`, `priv_loop.py` | rule around learned parts |
| MODIFIED | Objects: `--seeds` (contact-seeded candidate places) | `objects.py` | rule |
| MODIFIED | `local/run_family.ps1`: new `-Method` parameter (it hard-codes `$run\method` today, while the refreshed perception lives in `method_pl/`) | `local/run_family.ps1` | infrastructure |
| NEW | Contact runs (floor-run; ContactNet MIL in M4); label-free per-entity selector | `contact_runs.py` | learned threshold (2-means); MIL later |
| NEW | Attribution: calibration, fused posterior, KM lag model, one-to-one assignment, export, pre-registered expectations | `attribute.py`, `attr_expect.py`, `reliability.py` (leg 2) | learned (EM, 4-6 shared parameters) |
| NEW | Code-keyed branch: effect-pattern EM / VQ inverse model producing latent action codes | `codes.py` | learned |
| NEW | Events schema v2 with versioned loader and per-consumer contract tests | `events_io.py`, `tests/test_schema.py` | infrastructure |
| NEW | Contact-anchored events (leg 2) | `events_contact.py` | rule on learned quantities |
| NEW | Executor: Delta maps, data builder, policy/value/critic, eval harness | `goal_maps.py`, `skill_data.py`, `skill_gc.py`, `skill_eval.py` | learned; rules for relabelling |
| NEW | Plain image-goal GCIVL (baseline + fallback) | `gcivl_img.py` | learned |
| NEW | Shared belief class + parity test | `belief.py`, `tests/test_belief.py` | rule |
| NEW | Budget model and competence tables | `budget_model.py`, `calib_rollouts.py`, `competence.json` | rule + validation rollouts |
| NEW | Evaluation infrastructure | `eval_harness.py`, `freeze.py`, `diag_privileged.py`, `priv_loop.py`, `aggregate_results.py`, `PROTOCOL.md`, `CONSTANTS.md`, `sensitivity.py`, `tests/test_firewall.py`, `tests/test_blind_pipeline.py`, `tests/test_constants.py`, `local/run_shards.ps1` | protocol |
| NEW (scoring only) | `scripts/priv_effector.py` (FK of pinch site from qpos, press/lift matching, contact-run scoring), `method/attr_eval.py`, `tests/test_attribute.py`, `tests/test_goal_maps.py`, `tests/test_skill_data.py`, `tests/test_skill_gc.py` | | PRIVILEGED scoring |
| DROPPED (ablation anchors only) | EM relabel, MIL contact map, mask-centroid/AgentNet-sensitivity attribution | `relabel_events.py`, `contact.py` | |
| DROPPED / out of claims until after C3 | Hand servo as method, goal-view synthesis, logistic arm gate / reachability DP, anchor-hint channel, occlusion head, container proposals, hidden-goal machinery | | |

### 2.3 Modified world model (exact method)

- **Token.** `[s_k, 1[k = e], phi((pos_k - pos_e)/h_sp) (Fourier/RBF features of the 2-D offset), |pos_k - pos_e|/h_sp, same-type flag]`.
  - `pos` is the place median rest position, or the current position for movers.
  - `h_sp` is the median nearest-neighbour place spacing (class (a)), not a fixed 32 px.
  - The feature bank is dense enough that a cross radius of .2-.35 normalised units is resolvable. Fourier frequencies span 1 to 4 cycles per spacing.
  - The identity embedding is zeroed with probability .5 per sample. The shared relative pathway must then carry the effect, while identity-specific effects (scene locks) remain possible.
  - Ablation: no identity embedding for places.
- **Loss.** Per event, the existing threshold-unit errors on (pos, app), weighted by the attribution confidence.
  - After-states enter only where `after_known` or soft-completed (below).
  - Events below the confidence split train nothing.
  - Unknown after-states are never imputed by a hand rule.
- **Soft completion of the pressed entity's own after-state (round 2, self-training).**
  - For an event whose pressed entity is unread, the target is the WM's (or the effect template's) predicted distribution of that entity's after-state given the observed neighbour flips. There is no hand toggle rule.
  - It is weighted by the completion confidence and enters only above a data-derived confidence split.
  - The MAR assumption is checked: the flip rate among known-after events is reported by visibility bin, and a rate that depends on visibility is flagged.
  - Arms in E15: masked only, masked + soft completion.
- **Label-bias report (E15).** Per-entity known-after share against WM surprise rate, so a bias that makes covered or rare lights rarer is visible.
- **Ensemble.** 3 seeds, 4 min each.
  - Disagreement `d_ens` = max over entities of the prediction std in threshold units.
  - Use in downstream stages:
    - `Model.step` = ensemble mean (thresholded as today).
    - Event support is trained once on the events (not an ensemble).
    - Cost-to-go is computed with the mean step model.
    - `d_ens` enters only as the edge penalty `beta * max(0, d_ens - d0)` and as the surprise trigger.
- **Round 2 only.** The WM becomes a tempered expert in the attribution posterior (Sec. 3.4). A mislabelled edge press predicts a cross that never occurs under the shared relative effect, which the identity-specific WM could not penalise.
- **Unchanged downstream.** Event support (NCE) and cost-to-go (value iteration, about 50-90 min per env; a half-sweep version for screening) are retrained per env on the v2 events.
- **Arms** (E15): current WM, +rel, +rel+soft labels, +soft completion. Keep the current one unless a new arm wins.

---

## 3. Attribution mechanism

### 3.1 Effector track (EffectorNet v2)

**Data.** Frames and recorded actions only. No entity readings, no labels, no proprio.

**What the v1 run already shows (v1, held-out 3x3 job, finished 2026-10-10 01:04).**
- Validation R2 of the commanded translation was x .89, y .98, z .74 for d in {1,2,4}.
- The heatmap peak mass had p50 .0375, so the heatmap is diffuse.
- The median step was 1.26 px per frame.
- This is weaker than the privileged-FK premise (R2 .995) and is the baseline v2 must beat. The v1 loss used soft-argmax on 32x32, d in {1,2,4}, no clipping and no anchor.
- M0.1 scores v1 tip error at privileged press frames (CPU, minutes) before any v2 code is written. The v1 4x4 job is running now and 4x5 follows; no duplicate GPU jobs are launched.

**Net.**
- Input: frame (3x64x64) plus the segmenter probability map (1 channel, used only as an anchor).
- Encoder: conv_block 4->32 at 64, 32->64 stride 2, 2 Res at 32, bilinear up + skip. It outputs heatmap logits H (64x64).
- Keypoint: `kappa = sum softmax(H) * (u, v)` with integer pixel coordinates as in `objects.py`.
- Height: `zeta` is an MLP of features pooled by the heatmap.
- Optional auxiliary gripper-opening head (noise on puzzle, harmless).

**Camera.** `kappa = A q_xyz + b`, with A (2x3) and b (2) learned. The coordinate q is in units of summed action. Inverting gives `q_xy = A_xy^-1 (kappa - A_z zeta - b)` and `q_z = zeta`.
- If the E0 camera check (affine fit on privileged press points) shows residual p90 above 1.5 px, switch to a 3x3 homography on the table plane plus a linear height term. It stays global and low capacity.
- The measured affine residual on 92 presses was median .72 px, p90 1.25, max 1.73 (m), against a button spacing of 6.6-11 px.
- **Yaw.** The tip model ignores the yaw command. E0 reports the yaw range in play data and the FK pinch-site residual against cumulative yaw. If the residual exceeds .5 px, a yaw lever term is added to the integrator.

**Loss.** Sample windows (t, h) with h in {1,2,4,8,16,32} in one episode.
- Integrate `qhat_{i+1} = clip(qhat_i + a_i[:3], lo, hi)` from `qhat_t = q_t`.
- `lo, hi` are 6 learned scalars: workspace box, table floor, ceiling.
- `L_ID = Huber_1(q_{t+h} - qhat_{t+h})`, with gradients into both frames.
- Anchor: `-log sum softmax(H)` inside the dilated agent mask (weight .1 for 2k steps, then .01), plus a small heatmap-entropy term.

**Why this is a tip-like point, and what it does not guarantee.**
- Measured premise (m, privileged FK): effector displacement = 0.022 m x summed action, lag 0, R2 .995 in xy for windows of 1-32 steps (3x3 / 4x4); cube-single R2 .97-.99. z clips at the floor and ceiling, hence the clipped integrator. This is a property of the simulator, not a learned-keypoint result.
- Rigidly translating points of the effector body fit with unit gain. Arm links move at gain below 1 and not linearly. A constant keypoint cannot fit because the summed action is non-zero.
- **The cast shadow also fits.** The shadow is a translating image of q whose displacement from the tip's footprint grows with height. The segmenter anchor mask contains it (the ledger's distal-mask end scored .16 for this reason). The ID loss alone therefore cannot exclude the shadow.
- Remaining gauge freedom: one constant 3-vector plus a height-dependent lateral term (tip vs wrist vs shadow offset). It is resolved in 3.3 by the height-linear offset `delta(zeta)`.
- The shadow is safe for attribution only because tip and shadow coincide at floor contact. What must be validated is `zeta` and the tip position near the floor, not a global zeta correlation (Sec. 3.2, D2).

**Training and outputs.**
- Settings are identical for every family: 12k steps, batch 64 windows, bf16, AdamW 3e-4 cosine, about 10 min per env.
- Inference is about 2 min per 1M frames.
- Outputs: `effector.pt`, `track_{split}.npy` (N x 8: kappa_u, kappa_v, zeta, q_x, q_y, q_z, gripper, ID residual), `effector_report.json`.

**Test time uses one frame, but actions supervise training.** The net takes a single frame at inference. The action windows are a training signal, as in AgentNet and the hindsight-GCRL baselines. The causal contact detector in ContactNet's online twin and the executor's cue use the agent's own commands as inputs. This is stated plainly and not described as "no proprioception". The ablation is the current contact rule on the same events (arm A0).

**Controls.**
- P2: centroid of the opaque purple gripper pixels.
- Anchor-only: the heatmap trained with the anchor loss alone.
- AgentNet action sensitivity (.85 on 3x3, .49 on 4x4 at the contact moment).
- Existing v1 effector.
- Supervised pilot: a heatmap net regressing the FK tip projected to the image (PRIVILEGED upper bound).

### 3.2 Contact moment (contact runs)

Terminology: a **contact run** is a maximal interval of the effector at the table. An "episode" always means an environment episode.

**Default (leg 1): floor-run.**
- A frame is "down" if `zeta` lies in the lower cluster of the 2-means split of `zeta` over TRAIN.
- A contact run is a maximal down run of at least 2 frames.
- `t_c` is the run start.
- The effector "leaves" when kappa moves beyond the reach radius (Sec. 1.3.1).
- Volume: about 3x10^4 contact runs per env for the 1000-episode cap (1000 episodes x about 31 presses).

**Evidence from the privileged premise table (m, FK z on 3 VAL episodes of the scripted puzzle play policy).**
- Every dip to the floor was a press: 88/88 on 3x3 and 87/87 on 4x4.
- The flip coincides with the dip start. The floor-run start matches the privileged press frame to 0 to -1 frames, because the minimum z occurs 1-2 frames after the flip.
- 95.7% / 96.7% of presses lie in a dip.
- This is a property of the puzzle play policy measured with FK z. It is not a learned-zeta result and is re-measured with learned `zeta` in D2.

**D2: the contact-run check (replaces the previous zeta-correlation criterion).**
- On 100 VAL episodes each of 3x3, 4x4 and 4x5, with learned `zeta` and the calibrated offset, against privileged presses (P):
  - recall of presses by contact runs at least .95;
  - precision of contact runs (a run is correct if it contains a press) at least .95;
  - `|t_c - press frame| <= 2` for at least .90 of matched pairs;
  - tip error at press frames: median at most 1.5 px and p95 at most 4 px after delta calibration;
  - `zeta` error by height bin, reported separately for the lowest bin.
- A 12-frame overlay is inspected to confirm the keypoint is on the tip and not on the shadow away from the floor.
- A net with a 2-3 cm height error over a 0-20 cm range would pass a correlation test and fail this one.

**Label-free selector (same code on every env, held-out included).**
- For each entity k, compute the lag `t_window - t_nearest_contact` of its change windows against the nearest contact run.
- The statistic is the KS distance between the observed lag distribution and its null (contact runs circularly shifted by random offsets within each episode, 200 shifts).
- Entity k is "contact-explained" iff its statistic exceeds the 99th percentile of its own null.
- Contact-explained entities use the A1 pipeline. The rest keep A0 events with the geometry-absent posterior, which is the same code path with the geometry term missing.
- The coverage shares are logged next to the decision: the share of change windows with a linked contact run, and the share of contact runs with a linked change.
- No per-family branch exists. E7 validates the selector on DEV against qpos lift onsets (cubes) and `button_states` / joint events (scene), precision and recall at least .9 (P). Carry, drawer and window pulls have no dip, so grasp and release are the contacts that matter on cubes.

**Leg 2: ContactNet (M4, only if the slice shows a deficit that floor-run cannot fix).**
- A 1-D non-causal temporal CNN: 6 residual blocks, kernel 3, dilations 1-32, 48 channels. A causal twin runs online.
- Features per frame: q_xyz with q_z minus floor, central differences of q over +-1 and +-3 frames, `a_xyz`, the gripper command and its change, and the ID residual of the clipped integrator (commanded descent while q_z stays).
- Weak labels: the union of object change windows (`changes_{split}.npz`) and token-memory change windows (`memory_entities`, which covers undiscovered entities). There are no offsets.
- Noisy-OR MIL loss: each window must contain at least one contact, `-log(1 - prod(1 - s_t))`. Frames farther than the window width from every window are negatives.
- Runs are maximal runs with score above a calibrated classifier threshold. It is accepted only if it beats floor-run on D2 metrics and E7 (the comparison E7b runs in M4).
- 3k steps, about 5 min.

### 3.3 Calibrating the tip offset (two estimators, plus one for movers)

The contact point is `c_c = kappa(t_c) + delta(zeta(t_c))`, with `delta(zeta) = delta0 + delta1 (zeta - zeta_floor)`. That is 4 numbers plus 2 for the homography option. `zeta_floor` is the lower edge of the learned height bound (the table). A free CNN heatmap is exactly what failed in `contact.py`; here the capacity is 4-6 shared numbers.

Routes V and C are called **two estimators**, not independent ones: they share the contact runs and the kappa track.

**Route V (vote mode).**
- For every contact run c and every entity k with a reliable flip in c's lag window, vote `d = p_k - kappa_c`.
- Take the 2-D mean-shift mode with bandwidth `h` (Sec. 1.3.1).
- The true offset receives a vote whenever the pressed light flips. A lattice-neighbour offset receives one only when that neighbour exists (a share 1-1/R or 1-1/C of presses), and a diagonal offset never does.
- Margin on 3x3 (design estimate): about .79 vs .57 of runs explained, with about 3x10^4 TRAIN contact runs.

**Route C (contact cloud to place alignment, independent of any reading).**
- Cluster the contact points `kappa_c` of all floor-run contact runs (including those whose pressed light is unreadable) by mean shift with bandwidth `h`. This gives centres `m_j` with counts `w_j`.
- Choose `delta` on a grid of +-1.5 spacings (step `max(.1 px, h/4)`) minimising `sum_j w_j min(tau^2, min_i |m_j + delta - p_i|^2)`. Unmatched clusters pay `tau^2`.
- A one-step lattice shift moves the boundary clusters off the board, so they are unmatched and the score has a unique minimum on any finite board. No lattice is assumed, only that places sit at finite positions.
- Clusters that stay unmatched are candidate undiscovered places (3.8).

**Route M (movers: cubes, scene objects).**
- Where fewer than 2 places exist, `delta` is the mode of `p_k(t) - kappa(t)` over frames in which a mover follows the effector (coupling score above the 2-means/Ashman split).

**Acceptance.** Routes V and C (and M where applicable) must agree within 1 px. Disagreement flags the env for inspection and is not silently resolved. The synthetic test (E2) checks that the true offset is preferred over the one-cell-shifted offset. The lattice-shift diagnostic of 3.4 runs on every EM run.

### 3.4 Fused posterior over the acted entity

For contact run c, the latent acted entity is a_c in {entities, none}.

```
log q_c(j) = log pi_j
           + log N(c_c - p_j ; sigma^2 I)                          [E1 geometry, PRIMARY, weight fixed at 1; movers: min over before/after position]
           + lam_T * sum_k rho_ck [ u_ck log T_j[k] + (1-u_ck) log(1 - T_j[k]) ]   [E2 effect templates, tempered]
           + lam_W * log p_WM(observed after | before, j, x_j)     [E3 WM, round 2 only, tempered]
           + coupling term for movers                               [mover follows effector between consecutive contacts]
```

- **Definitions.** `T` is a K x K Bernoulli matrix, Beta(1,1) prior, with `T[j][k] = P(entity k changes | j acted)`. `u_ck` = entity k changed in c's lag window. `rho_ck` = its reliability (3.5), and missing observations are left out.
- **Temperatures (changed).** The geometry weight is fixed at 1 and is never fitted.
  - `lam_T` is fitted by cross-expert prediction. Let G be the contact runs whose geometry-only posterior is at least .9. E2 alone must predict the geometry label: `lam_T` maximises `sum_{c in G} log q^{E2}_c(j*_geo)` over the grid {0, .25, .5, .75, 1}, fitted on one half of TRAIN episodes and evaluated on the other half (two-fold swap).
  - `lam_W` (round 2) is fitted the same way against the geometry+E2 posterior at least .9. `lam_W = 0` for the first 10 EM iterations.
  - The likelihood of the observed flips is not the fitting objective. That objective favours E2, which models the flips directly, over geometry, the only expert that does not see them. This is what drove EM relabel and the MIL map to the lattice-shift fixed point.
  - The experts are not conditionally independent: they share contact timing and lag-assigned evidence. No independence claim is made. The temperatures absorb the dependence, and calibration is reported empirically (ECE, 7.6).
  - Privileged accuracy over a `lam_T x lam_W` grid on dev is reported in E6 to show sensitivity.
- **Lattice-shift diagnostic on every EM run.** The log-likelihood at the fitted `delta` is logged next to the log-likelihood at `delta` plus each of the 8 one-spacing lattice shifts, together with the Route C score at the same points. A margin below its bootstrap noise flags the run. Runs start from `delta=0`, from the calibration, and from random.
- **EM.** Closed-form M-steps for `delta`, `sigma` (isotropic), `pi` and `T`. 30 iterations on CPU, about 3x10^4 contact runs x 24 entities. Initialisation: `delta` from calibration (3.3), `T` = identity, no WM in round 1.
- **Null slot.** Uniform image density, base-rate templates. It absorbs contact runs whose entity is farther than `r_null` from every known entity, and failed contacts.
- **Outputs.** Top-3 posterior per event, argmax label, confidence = max q. Events below the 2-means split of the confidence log-odds get `target_known = False`. They stay on the timeline but train nothing.

**Geometry-only accuracy, closed form (corrected).** For an isotropic Gaussian localisation error with per-axis sigma and a calibrated offset, a cell with spacing d is assigned correctly when both axes stay inside it:
- interior cell: `[2 Phi(d/2s) - 1]^2`;
- edge cell (one open side): `[2 Phi(d/2s) - 1] * Phi(d/2s)`;
- corner cell: `Phi(d/2s)^2`.

An interior cell has 4 edge neighbours (8 with diagonals), so the earlier "pairwise 5% at sigma = .3d" bound was wrong.

| sigma / d | interior | edge | corner | board average (3x3 / 4x4 / 4x5 / 4x6) |
|---|---|---|---|---|
| .20 | .975 | .982 | .988 | about .98 on all four |
| .25 | .911 | .933 | .955 | .940 / .933 / .931 / .929 |
| .30 | .818 | .861 | .907 | .877 / .862 / .857 / .854 |
| .40 | .622 | .705 | .800 | below .72 |

- Interior accuracy .95 needs `sigma <= .22 d`. For 4x6 (d about 6.5 px, from the env geometry and not a held-out read) that is `sigma <= 1.4 px`, not 2 px. For 3x3 (d about 11 px) it is about 2.4 px.
- At the effector criterion limit (radial p95 of 4 px gives sigma about 1.6 px under a Rayleigh error), with d = 6.6 px: sigma/d about .25, interior about .91, board average about .93. Heavy tails or a residual bias lower this.
- `attr_expect.py` computes this table from each env's `objects_report` spacings and the measured `sigma-hat`. The expectations are written to `PROTOCOL.md` before E6, including the 4x6 expectation. A measured gap above 3 points from the closed form at the measured sigma-hat is reported as heavy tails or bias.
- The templates must carry the rest. E2 is ambiguous only when the observed changes lie in several crosses; E1 is ambiguous only near the localisation limit. The experts fail in different places, which is the reason to expect the product to help. This is unverified until E6 and E2 report geometry-only, template-only and fused accuracy separately.
- **Resolution check.** The 4x6 spacing needs sigma at most about 1.4 px. This is a risk (Sec. 8), and D2/E6 measure it on the top rows of 4x4 and 4x5.

### 3.5 Partial observability and the fix for lagged readings under the gripper

This is the second half of the attribution fix: even with the right contact, the pressed light is read late or not at all.

1. **Reliability.** A reading of entity k at t is an observation only if `rho_k(t) >= rho*` (Sec. 1.3.1).
   - Leg 1 rule: not covered by the dilated agent mask, the effector farther than the reach radius for at least m frames, and SeeThrough confident.
   - Leg 2: `rho` is a logistic regression or small GBM fitted self-supervised on features [coverage of the place disc by the dilated agent mask, SeeThrough probability and path (full view or lookup), effector distance in px, frames since the effector was last within the reach radius, floor flag]. Its target is "equals the next stable reading". Stable means: agent-free, effector at least two kernel widths away for m frames, and no contact with posterior mass on k in between.
2. **Lag model with censoring.** The histogram `G(d)` of reading-flip time minus contact time is estimated by Kaplan-Meier.
   - Pairs are (contact run c, entity k = geometry-only argmax of c). The event is the first reading flip of k after `t_c`.
   - Pairs are censored at the next contact run with geometry mass on k or on k's cross, or at the episode end. A run with no flip within the window is censored and not dropped.
   - Geometry-only attribution is used so that the pairing does not depend on flip timing. Estimating G only from high-confidence pairs would pick the fast-flipping ones and bias L90 low, and the bias would compound over EM iterations.
   - The privileged shape (m): pressed light median 11 / p90 27 frames on 3x3, median 7 / p90 19 on 4x4; neighbours 8 / 21. `L90` = KM p90, re-estimated every EM iteration.
3. **Unknown, not unchanged.** After a contact, "entity k unchanged" counts as evidence with weight `1 - S(span since contact)`. Until the lag has passed it is unknown.
4. **Lag-aware, one-to-one flip assignment.** For each entity k, flips f_1 < f_2 < ... are matched in time order to the contact runs whose posterior has mass on a cross containing k (Hungarian assignment per entity, cost `-log(T_j[k] * G(t_f - t_c))`).
   - Each flip goes to at most one contact run. A contact run may take several flips, one per entity.
   - Unmatched flips go to the null slot and become their own flagged events (`unlinked`). Late flips no longer become separate events because they are matched to the contact that caused them.
5. **Pressed entity's own after-state.**
   - It is the first reliable reading after the effector left the reach radius.
   - If there is none, the after-state is unknown: masked in the WM loss and the support loss, or soft-completed by the arm of Sec. 2.3. It is never defaulted to "unchanged" and never imputed by a hand toggle rule.
   - The learned WM sees the 74-89% of events where it is read, and the label-bias report of E15 tracks what the remaining unread share does to rare and covered lights.
6. **Online** (Sec. 5.3): at contact end the belief receives the WM-predicted transition for the entity actually touched. A reliable reading overwrites it.
7. **Optional (leg 3).** A per-entity HMM over contact runs, with transitions `tau_ck = sum_j q_c(j) T_j[k]` and forward-backward over reliable readings, to recover A-B-A middles that are never read.

### 3.6 Events

**Schema.** Events are written as `events_v2` with `schema_version=2`, loaded only through `events_io.load_events(path, min_version)`, which asserts the required keys. `tests/test_schema.py` runs every consumer (`world_model`, `event_support`, `train_support`, `skill_data`, `closed_loop_objects`) on a tiny v1 and v2 fixture. The existing fields keep their meaning. In particular, `t_core` stays [latest departure, first arrival]. New fields are `acted_post` (N x (K+1)), `conf`, `t_contact`, `t_run` (the contact-run interval [run start, run end]), `contact_xy`, `weight`, `linked`, `unlinked`.

**Leg 1 (arm A1: contact-linked relabel and merge on existing events).**
- Candidate links: for every event from `events_objects.py`, the contact runs with `t_c` in `[t_start - L_pre, t + L90]`. `L_pre` is the TRAIN p90 of the time from the effector entering the reach radius to the floor-run start.
- Each change window is assigned to at most one contact run by the one-to-one rule of 3.5.4. Events merge only when their windows are assigned to the same contact run: union of cores, earliest departure, latest arrival, `before` from the earlier, `after` from the later (re-read at settled frames).
- The acted label is the fused posterior of Sec. 3.4.
- Events with no linked contact run take the geometry-absent posterior (E2/E3 + mover coupling).
- The merge window is not a fixed hard span of width `L90`. With about 32 frames between presses, a hard union over `[t_start - L_pre, t + L90]` would chain adjacent presses and XOR their effects, recreating the earlier problem (11.3 events for 29.6 presses).
- Reported next to `L90`: the inter-contact gap distribution (median, p10, share of gaps below `L90`) and the merged-multi-press rate (P).
- Targets (aligned): events per press at most 1.10 (aspiration about 1.0, versus 49 / 44 / 34 per 31 presses today), merged-multi-press rate at most 3%, press-to-event coverage at least .93.

**Leg 2 (arm A2: `events_contact.py`, contact-anchored).**
- One event per contact run, or a chain linked by coupling.
- Per (run c, entity k): `before` = last reliable observation at or before `t_c`; flips with brackets; no-flip spans, all with `rho` weights.
- `seg_start` = previous `t_c + 1`; `t` = `t_c` + time for the effector to leave the reach radius; `t_core` keeps its meaning; the run interval is stored in `t_run`.
- Coupling link: consecutive runs c, c' form one event when some entity follows the effector between them, `s_k = 1 - |dp_k - dkappa| / |dkappa|`, split by 2-means with Ashman D above 2 (the same rule family as the rest of the code).
- `labels_{split}.npz` and `changes_{split}.npz` are reused unchanged.

### 3.7 Continuous targets and movers

- The event target `x` is the observed after-position: the first reliable reading after release plus settling. It is never a code.
- Release gives a second measurement `x_eff = kappa_rel + delta_k`. The two are fused by inverse-variance weights estimated from at-rest reading noise and EM residuals. Target: cube position error below the front end's 1.5-1.9 cm, checked against qpos (P).
- Mover coupling (E3 in the posterior) selects the carried cube without a template.
- Grasp and release are separate contact runs linked by coupling.

### 3.8 Undiscovered entities and the perception hole

- Contact clusters unmatched by any place (Route C) are clustered by mean shift. A cluster with enough support (above the 2-means split of cluster sizes) becomes a candidate place.
- `objects.py --seeds` accepts a candidate only through the existing persistence and bimodality tests on its pixel patch. Seeds only propose candidate locations.
- **Measure first (E16, P).** The target case is 3x3 button 1 (top-middle, under the arm base): 8 of 9 found today, with a |corr| of .27 for that place. Before any seeding work:
  - measure how often button 1 is visible in dataset frames and in OGBench goal images (separability of its patch, privileged-labelled probe AUC);
  - compute the ceiling: the share of the official 3x3 tasks whose true goal differs from the initial state at button 1.
- If the probe AUC is near .5 on every frame and goal image, no pixel method can read button 1. The ceiling then applies to every method, including D_i. Such tasks are reported separately and excluded from the abstract-only denominator, and the E16 target "9 of 9 with |corr| >= .95" is dropped for that env.
- If button 1 is visible at some frames, the seeded place is accepted only if the persistence and bimodality tests pass.
- If the tests fail on an arm-dominated patch, the entity stays undiscovered. The env is flagged incomplete from unmatched contact clusters (Route C), not from pixel differences, because a never-visible place shows no difference in either frame. The flag enters the arbitration estimate through `q_hidden` (5.4).

### 3.9 Attribution arms, diagnostics and the code-keyed branch

| Arm | Description |
|---|---|
| A0 | Current rule (touched identity nearest centroid of known changes) |
| A1-geo | Effector geometry only, argmax at the floor-run moment |
| A2-codes | Effect templates only (EM from random init), plus a VQ inverse-model variant. Permutation-identifiable only. Used as the code-keyed branch below. |
| A3 | Fused posterior (3.4) |
| A4 | A3 plus the WM expert (round 2) |
| BISCUIT-style | Gated-factor attribution, capped at one week, as a related-work baseline |

**Code-keyed branch (F2, a tested rung of the oracle ladder, Rc).**
- The acted token of the WM and the support model is a latent action code `a_c in {1..K_c}` instead of an entity id.
- Codes come from the effect-pattern EM or a VQ inverse model on the (before, after) change patterns plus the contact geometry where available. `K_c` is chosen by held-out likelihood (data-derived).
- The planner enumerates codes. Their predicted effects come from the WM, and the cost-to-go is unchanged.
- The executor still receives Delta maps (positions and appearance changes) and needs no identity.
- Codes are identifiable only up to permutation. That does not matter for planning, but it makes the branch weaker for scene locks and for any reasoning that needs entity names.
- Rc runs on 4x4 in the oracle ladder (M0), so a D2/effector failure has a measured branch.

**Report per arm:**
- press-level and event-level accuracy (7.6), not only presses inside one event core;
- by board position (corner / edge / interior) and by top-row spacing;
- by visibility;
- ARI for the code arms;
- ECE of the posterior;
- events per press, spurious-event share, merged-multi-press share and press-to-event coverage;
- the unchanged-read share;
- label-free accuracy estimates (mean max posterior; pairwise agreement of E1/E2/WM), validated against privileged accuracy on dev so they can be used on held-out envs.

**Fallback order if the geometry or the effector fails** (each fix capped at one week):
- F0: constant-offset floor-run labels only.
- F1: confident-subset WM training (q at least .9) plus the relative-position WM.
- F2: the code-keyed WM branch (Rc) with the Delta-map executor. Tier 2 remains possible if Rc is competitive in the oracle ladder; otherwise the paper becomes the perception-and-diagnosis paper (Tier 1) plus the executor.

---

## 4. Low level

### 4.1 Goal representation (Delta maps, no acted entity)

**Inputs.** The current memory state S and a target abstract state S_goal, each K x D (D = 2 position + A appearance + 1 covered), with known masks.

**Membership.** `Delta = {k : known in both and (|pos_goal - pos_cur| > tol_pos or max|app_goal - app_cur| > thr_app_id[k] or covered bits differ)}`. The thresholds are the events report's `tol_pos` and `thr_app_id`, so the rule is the same one used for events.

**Channels** (64x64, Gaussian blobs with sigma = w/2 px, w = object width from `front.npz`, about 3 px):
- c0: blob at `pos_cur[k]` for k in Delta ("from").
- c1: blob at `pos_goal[k]` ("to").
- c2: appearance change `min(1, |app_goal - app_cur| / app_unit_id[k])` times blob at `pos_goal[k]`.
- c3: blob at `pos_cur[k]` for members with a covered goal.
- c4 (arm L0+eff): Gaussian at the effector estimate `kappa_t` from EffectorNet on the current frame. Optional scalar inputs are `zeta` and the floor flag, added to the cue vector.
- c5 (arm L0+press-point, new): Gaussian at the press point. In training it is the realised contact point `kappa(t_c) + delta(zeta)` of the contact run that realises the boundary (no attribution needed). At test it is the planned entity's place position, jittered in training by the measured `sigma-hat`. It carries information the planner has and Delta-only goals drop: the Delta of a cross is asymmetric at board edges and incomplete in an estimated 20-30% of training Deltas (E10).
- c4 and c5 are promoted only if T1 on dev shows a gain in exact-press rate or CSR over the Delta-only default.

An empty Delta gives zeros.

**Training perturbations**, to survive WM side-effect errors:
- drop each member with p .15 (never all);
- add one random non-member with p .05;
- jitter the to-positions by up to `tol_pos/2`;
- one random joint shift of +-2 px on frame and maps;
- delay the cur->goal switch by a lag sampled from `G(d)` truncated at its p75, on top of the contact-end switch of 4.3.

**Function.** `goal_maps.py` provides `render` (numpy for the loop, torch on GPU for training), `reached(S, S_goal, rest, rest_count)` and `event_status(...) -> running / success / deviated`.
- Success: every Delta member observed at rest for at least m frames matches the goal, and at least one was observed to change.
- Deviated: a Delta member settled in a state that is neither start nor goal, or an entity outside Delta changed and settled, or any member is still unreached one `L90` after contact end.
- If every member stays hidden, the loop's belief rule applies (5.3).

### 4.2 Observation and history

- Policy input: the current frame only (no history, no velocity), the maps c0-c3 (+c4, +c5), and a 2-number own-action phase cue.
- The cue is the EMA-8 of the commanded gripper channel (normalised by its 2-means midpoint) and the steps since it crossed the midpoint, clipped at 16 and scaled by 1/16. It is enabled only when the gripper command is bimodal on TRAIN (Ashman D above 2, the same rule as elsewhere). On the puzzle, where the gripper is noise, the same code disables it, and the report says which envs use it.
- The cue uses only the gripper channel. It never uses the xyz commands, whose lag-1 autocorrelation of .87-.96 is what makes two-frame policies copy ongoing motion.
- Cue dropout .3. Shift augmentation +-2 px, applied jointly to frame, maps and effector map.
- Trunk: stem conv3x3->32 + GN + GELU at 64x64, ResBlock(32); down conv4x4 s2 ->48, ResBlock(48) at 32x32; down ->96, ResBlock(96) at 16x16; down ->128, ResBlock(128) at 8x8; 1x1 conv ->32, flatten 2048, Linear->512, concat cue, Linear->512. About 1.2M parameters. A half-width trunk is used for V and Q.
- Ablation ladder (chosen on dev T1, then frozen):
  - (a) 1 frame;
  - (b) 1 frame + cue (default);
  - (c) 2 frames gap 2 with history dropout .5;
  - (d) 2 frames + cue;
  - (e) executed chunk E in {4, 8, 16};
  - (f) (b) + c5 press-point channel.
- Copycat diagnostic, condition-sensitivity ratio: `CSR = E|pi(o,g) - pi(o,g')| / E|pi(o,g) - pi(o',g)|`, with g' the Delta of another event at the same frame. Baselines: .125 (two frames), .41 (one frame). Target: at least .6.

### 4.3 Data, hindsight relabelling and chunking (`skill_data.py`)

1. **Interactions.** Merge consecutive events i, i+1 of an episode when the gap `g_i = t_core[i+1,0] - t[i]` is below `delta`. `delta` = the 2-means split (Ashman D above 2, else no merge) of log g over TRAIN.
2. **Boundary timeline** B_0..B_n.
   - B_0 = state before the first interaction; B_j = state after interaction j.
   - `B_j[k] = after[k]` if `after_known`, else `before_{j+1}[k]` if `before_known`, else unknown.
   - Unknown stretches between two equal known values are filled; otherwise they stay unknown and drop out of Delta.
3. **Reach interval** of boundary j: `[lo_j, hi_j]`.
   - `hi_j` = (core departure of interaction j+1) - 1, or the episode end.
   - **`lo_j` = the floor-run contact-run end of interaction j (plus one query period, 4 frames) in both legs**, not the event arrival time `te`. This is the same instant at which the loop applies the belief fill (5.3), so the switch time is identical in training and at test.
   - Interactions with no linked contact run do not define goals. Their effects remain in the boundary timeline. The dropped share is reported (target at most 5%).
4. **Training state from the belief logic.** The current-state input S_t of each row is produced by replaying the episode through `belief.py` (views, hidden rule, age). It uses the same contact-end fill as the loop.
   - In training the fill uses the realised `B_j`, replaced with the WM's prediction with probability equal to the WM's measured event-error rate on VAL, so fill errors are seen.
   - Rows between contact end and the late reading therefore have an empty Delta against goal `B_j`. They are "reached", are dropped from the policy loss, and enter V with target 0. The executor never learns to keep acting through the reading lag.
5. **Rows.** Every 4th frame of the first 1000 episodes (enforced at cache-build time), so every env contributes about 1M frames and cost does not grow with 4x5/4x6/triple/quadruple. Each row stores the boundary index, the 16-action chunk (normalised by TRAIN mean/std, float16, with a validity mask past the episode end), the cue, and the keyframe weight `1 + 2*[|t - core0| <= w_kf or |t - te| <= w_kf]` (`w_kf` from 1.3.1). Frames come from a sorted sequential read of the mmap in 2000-frame blocks, into a compact uint8 cache of about 250k rows (3 GB in CPU RAM, moved per batch; VRAM already holds about 5 GB from other processes and a spill freezes the PC).
6. **Goal mixture** (class (b)) for a row at boundary j(t): j+1 with p .50; j+2 .15; j+3 .07; j+4 .03; the current boundary (Delta empty, reached at once; calibrates V and teaches "nothing to do") .08; a far boundary later in the episode .07; a random boundary of another episode .10 (value only, never reached).
7. **Reward and returns.**
   - Sparse reward: -1 per step, 0 when reached. `reached(u,g)` = `u in [lo_g, hi_g]`.
   - `gamma = .985`, effective horizon about 67 steps. One-interaction goals are 20-60 steps away.
   - Stride-4 value target: `V = 0` if reached. Otherwise, if the interval is entered within 4 steps, `V = -(1 - gamma^d)/(1 - gamma)` with `d = lo_g - t`. Otherwise `V = -(1 - gamma^4)/(1 - gamma) + gamma^4 * Vbar(next row)`.
   - Chunk n-step return: `y = -(1 - gamma^d)/(1 - gamma)` if reached within the chunk, else `-(1 - gamma^16)/(1 - gamma) + gamma^16 * Vbar(row+4, g)`.
8. **Chunking.** H = 16 actions (0.8 s at 20 Hz), executed E = 8. The chunk commits to a phase ("close k steps, then lift"), so replanning mid-phase does not restart it.
9. **Audits (PRIVILEGED, scoring only):**
   - fraction of merged interactions containing exactly one true press or one object move;
   - Delta completeness (identities in Delta equal the toggled lights, mapped through `objects_report` `best_ref`);
   - known fractions;
   - observation-lag and segment-length distributions before and after merging;
   - the dropped share of 3.

### 4.4 Algorithm ladder (`skill_gc.py`)

Each rung must earn its place on single-event T1 against the previous one. L1 and L2 are built only if dev T1 shows a deficit that policy extraction can address: exact presses below the threshold after the L0 ablations of 4.2, or collateral above 2%. The privileged scripted press ties the learned skill on 4x5 (7/30 = 23% for both), so executor gains cannot lift 4x5 above that until the high level improves.

- **L0: flow BC.** Conditional flow matching on the normalised chunk x1 (80-d). Draw `x0 ~ N(0,I)`, `s ~ U(0,1)`, `x_s = (1-s)x0 + s x1`. An MLP `v(x_s, s | h)` (4 layers x 512, `s` via sinusoidal + FiLM) is trained with `loss = mean_i w_i |v - (x1 - x0)|^2` over valid steps, with `w` = the keyframe weight.
  - Rows where the goal is already reached, or Delta is empty, are dropped from the policy loss.
  - Inference: 8 Euler steps from `x0 ~ N(0, tau^2 I)`, `tau` = .5 (swept in {0, .3, .5, 1} on dev T1). EMA .999 weights at test.
  - 20-40k steps, batch 256, about 25 min on 3x3.
  - Head ablation: MSE chunk head (the old skill's loss) vs flow.
- **L1: + IQL value and AWR (conditional).**
  - Value: light trunk on (frame, maps, cue), two heads (ensemble of 2, EMA .995 target, min of targets), expectile tau .9 on the stride-4 TD target, batch 256, 30k steps, about 25 min.
  - Weights: `w_i = keyframe weight * min(exp(beta * A_i / sigma_A), 20)`, with `A_i = y_i - mean(V)(o_i, g_i)`, `sigma_A` the EMA of the batch std, `beta = 1` (ablate 0, .5, 2), normalised to batch mean 1.
  - Checks on VAL: Spearman of V along a segment vs true time-to-go at least .9; AUC of `V(o_t, g_true) > V(o_t, g_other)` at least .95; `V(reached)` near 0.
- **L2: + chunk critic (conditional).** An n-step SARSA critic `Q(o, g, chunk)` regressed to `y`. It is in-distribution because it is trained on data chunks.
  - Inference: draw N = 16 chunks and take the argmax of `min(Q1, Q2)`. Accept a non-first choice only if its Q margin exceeds the 75th percentile of the Q spread among the N samples on VAL (otherwise take sample 0).

All class (b) hyperparameters are logged in `CONSTANTS.md` as "design default; tuned only on dev T1".

### 4.5 Checks

**T0 (offline, no simulator, VAL rows).**
- Flow loss and action MSE.
- CSR at least .6.
- V metrics (L1).
- Q vs realised-return correlation (L2).
- The audits of 4.3.

**T1: single-event replay (PRIVILEGED scoring).**
- Reset the simulator to a dataset state with `env.unwrapped.set_state(qpos, qvel=0, button_states)` taken from the `_priv` VAL arrays (the harness, not the method, reads them).
- **Fidelity check (extended).** Verify the rendered frame matches the dataset frame (mean abs diff below 2/255). Then run a dynamics-fidelity check: replay the recorded 8-16 action chunk from the reset and compare the resulting frames and qpos with the dataset's following frames. The cache holds no qvel, so qvel = 0 is assumed. Abort if the median replay error at h=16 exceeds the p95 of the one-frame-lag error between consecutive dataset frames (a data-derived threshold: the replay must be within one frame of timing). The ogbench `set_control` targets the current pinch pose plus the action, with no hidden integrator, so a velocity mismatch should be small.
- About 300 VAL events per env, stratified: puzzle by pressed cell, cube by cube index and move-distance quantile, scene by object type.
- Build the goal from the dataset's own boundary states B_j -> B_{j+1}, using non-privileged readings. Run up to 150 steps.
- Success = the physical net change equals the dataset's net change: puzzle the same toggled set with no extras; cube within 3 cm of its dataset end position and other cubes moved under 1 cm; scene the same button/drawer/window/cube change.
- Also report: reader-based `event_status` success on 100 events, steps to completion, collateral changes, and gripper toggles per event (more than 2 means dithering).
- Variants: goal = dataset B_{j+1} vs goal = the WM's `M.step` prediction; goals degraded by dropping or hallucinating members at p up to .3; **goals carrying a WM wrong side effect** (a cross that differs from the realised one at an edge), to measure what the loop's frozen-Delta rule of 5.1 is protecting against.
- Cost: no front end is needed for the fast tier, about 60-100 env steps/s, so 300 events take about 5 min; the reader tier runs at about 11 steps/s (the same speed as the full loop).
- **Harness calibration first:** run the existing `obj_skill_h0` on 4x5 (should land near the logged .865 exact presses) and the history-dropout skill on held-out 4x4 (near .35; 4x4 is DEV, so this is allowed). If T1 does not reproduce these, fix the harness before using it.

**T1c.** Chains of 5 consecutive interactions, 40 chains per env, with the reader loop. Count re-presses per chain (the check on the train/test parity of 4.3).

**T2: horizon sweep.** k = 2, 3, 4 and final-goal Delta, 50 multi-event goals each. Success should be non-increasing in k.

**T3 (privileged-oracle tier, puzzle only, reported separately).** The true Lights Out solver plan on the true state, executed with the learned low level through Delta maps. It measures the low level plus perception without the WM.

**Label-floor diagnostic (E9).** Train the old single-frame skill (`skill.py --hist-gap 0 --cond spatial`) twice on 3x3, with the rule's labels and with privileged-correct labels (pressed light from `button_states`, mapped through `objects_report` `best_ref`). If exact presses rise by at least .15, labels are the old skill's floor and Delta goals are the right fix. If not, precision or phase dominates and escalation goes to 4.2 and 4.4.

---

## 5. System integration

### 5.1 Planner to low-level interface

- The planner (WM + support + cost-to-go + weighted A*) returns a plan of events `(e_i, x_i)`. The acted entity is used by the planner, WM and support only.
- The executor receives the next abstract state `S'_k = Model.step(S, plan[:k])` rendered as **frozen intended Delta maps**: the members are the entities the WM predicts to change in event k.
  - At each query only the members' current positions and retirement status are refreshed (c0 blobs at the current position for cubes). Off-plan changes are not added to the Delta, so a residual single-member Delta is never shown to the executor. Such a goal never occurs in training (the perturbations of 4.1 add or drop members, they do not produce it), and for a puzzle no single press can realise a one-light Delta.
  - Members are retired when observed at the goal (reader) or at the contact-end fill.
  - If a member is still unreached one `L90` after contact end, or an off-plan entity changed and settled, `event_status = deviated`. The planner replans from the belief with the failure memory. The belief already contains the side effects of the entity actually touched, so the replan plans an undo as a regular planned event with its own full intended Delta. The same event may be re-issued with a fresh Delta.
- The default goal horizon is k = 1. k = 2-4 and the final goal are measured in T2.

### 5.2 Event loop

Each query happens every 4 frames:
1. Read the entities (views) and update the belief.
2. Render the frozen intended Delta with the members' current positions and query the policy.
3. Execute E = 8 of the 16 chunk actions.

Event end uses `event_status` (4.1), plus the contact signal (5.3). The old "fresh start" trick is dropped, because there is no history. Plan commitment and the recent-state avoid-set stay, but the avoid-set is restricted to states left by AS-PREDICTED events, so a wrong light pressed by mistake can be undone at once.

### 5.3 Belief, contact-end fill and the lag

- `belief.py` is the shared class lifted from `closed_loop_objects.run` (S, `t_obs`, age, covered by the existing `hidden()` rule, fresh within m=5). It is used by the data tables of 4.3 and the online loop, with a parity test: offline replay of 50 VAL episodes equals frame-by-frame online reads.
- EffectorNet and a causal contact detector run on every frame (sub-millisecond).
- At contact end, compute the geometry posterior over the touched entity from the current track.
  - If it equals the planned entity, apply the WM transition (belief fill) for unread members.
  - If it differs, apply the WM transition for the entity actually touched, so the belief reflects what happened.
  - The fill keeps its age, and a reliable reading overwrites it.
- Consequence: Delta empties as soon as the press is done, so the executor does not re-press during the 7-27 frame reading lag. This is the known weakness of an executor with no contact observation, and it also removes the 250-step timeouts caused by a hidden pressed light.
- Unseen entities are assumed at goal until seen (existing rule).

### 5.4 Easy-task path and arbitration (pre-registered rule)

Arms, all in the same loop:
- **D_i:** plain image-goal GCIVL (`gcivl_img.py`) on the raw OGBench goal image. It is also the baseline row (the "flat-raw" row). Reproduction follows OGBench settings (V with upper expectile .9, gamma .99, AWR alpha 3, separate small CNN encoders on concat(frame, goal frame), random-crop p .5, goal mixture .2 current / .5 trajectory geometric / .3 random, batch 256), at 300k steps (about 3-5 h per env here) against 1M published. Both our reproduction and the published number are reported.
- **D_a:** the same executor given the final-goal Delta (`--goal-mode final`) with no search.
- **P:** the waypoint chain (k = 1 by default).
- **Hybrid:** the rule below.

**Rule R, evaluated at start, at every replan and at every event end.** It replaces the stall-count switch of the previous draft. A stall switch with per-event success .9 and about 20 events fires with probability about .3 even in an on-track episode, and sends it to a fallback that is at most 17% on 4x5.

1. **Completeness.**
   - Pixel check: tokens where the goal view and the first-frame view differ but belong to no entity region (and are not hidden tokens) are counted. If the count exceeds the 99.9th percentile of the same statistic on static frame pairs, the abstract goal is incomplete.
   - Contact check (new): the env is flagged incomplete when Route C leaves a supported contact cluster unmatched (3.8). It sets `q_hidden`, the label-free share of VAL dataset segments between two states that contain a contact in an unmatched cluster. This catches a never-visible place that the pixel check cannot.
2. If no plan is found, go to D_i.
3. **Competence comparison.**
   - `c_D(n)` = D_i success on VAL start/goal pairs that differ in n entities (isotropic fit over n bins).
   - `c_P` = (product of measured per-event reader-based success over the plan) x (1 - `q_hidden`) x (budget-model feasibility within the horizon, 5.5).
   - Choose D_i iff `c_D(n_diff) > c_P`. Otherwise choose D_a if the plan length is at most `k_dir`, else P.
   - `k_dir` is the largest k with T2 success at least .95 of the k=1 success. T2 for this rule uses the reader-based `event_status`, not privileged scoring, so the frozen code can compute it on a new env.
4. **Stall switch (ablation only).** After a failed attempt, a binomial test of the observed failures against the measured per-event failure rate (T1) can trigger a switch to D_i for the rest of the budget when `c_D(n_remaining)` also exceeds the updated `c_P`. The headline hard-board configuration has the switch OFF and is reported next to the switch-ON variant.

**Competence tables (E24).** `calib_rollouts.py` runs the arms from simulator resets to VAL dataset states, using the same T1 reset mechanism, and writes `competence.json`. Success is reader-based. The harness is the only code that reads simulator state, and the method sees only the table. It is the same frozen script on every env. The paper describes it as validation-rollout calibration. A rollout-free variant (value-based, from the GCIVL critic) is an ablation.

**Reporting guards.**
- The hybrid is also run with the fallbacks off ("abstract-only"), so the fallback's contribution is isolated.
- "Non-inferior to D_i" on easy envs follows by construction and is not counted as evidence for the method. The evidence is the abstract-only gain on hard boards over D_i and over matched flat-view GCRL.
- 3x3 is reported as abstract-only and hybrid separately, with the E16 ceiling noted.

### 5.5 Replanning, timeouts, planner extras and the budget model

- **Replan on surprise.** On a mismatch, add a failure cost (anchor, quantised target; class (b), in units of the median per-event planner cost) dropped from the candidate set after 3 failures unless no alternative exists. Replan from the belief.
- **Timeouts are data-derived.** Per-event timeout = 3 x the TRAIN median event duration, capped by horizon / (plan length + 1). The old fixed 250 exceeds the whole cube-single horizon of 200. If L1 is on, a stall also triggers when V has not improved by `delta_v` over 48 steps, with `delta_v` the 25th percentile of V improvements along VAL segments.
- **Lambda.** Weighted A* `lam = .6`, raised toward 1 when the budget is tight.
- **Budget model (E25, `budget_model.py`).** A Monte Carlo model of an episode:
  - plan length N from the planner on official tasks;
  - per-attempt success p and steps s from T1, with a collateral probability that costs an extra event;
  - verification delay v (reader lag, or zero with the contact-end fill) and the per-event timeout;
  - horizon H from OGBench.
  - It predicts episode success, steps per event and retries per episode. It is validated against the R3 scripted ceiling (predicted vs measured within 5 points) before it sets thresholds. It sets the G2a-equivalent per-event threshold (default .90) and the 4x5 target of C3.
- **Plan-length diagnostic.** Report plan length against the GF(2)-minimal press count (diagnostic only, never used by the method).
- **Optional, only after C3:** commutation-aware reordering (events i, k commute iff `step(step(S,i),k) == step(step(S,k),i)` per the WM; non-commuting pairs keep their order; the final WM state is asserted unchanged).

### 5.6 Hidden goals and scope

Hidden-goal scene tasks (4-5, cube in the closed drawer) and cube stacks stay out of the claims. The existing covered-bit rule and the D_i fallback remain. No new machinery (occlusion head, container proposals) is built before C3. Per-task results are reported so capped tasks are visible, together with the fraction of official tasks whose goal contains a hidden entity.

---

## 6. Milestones, in execution order

Notation: `local/run_stage.ps1` and `method/run_family.ps1` stage names follow the existing `Stage` helper; new stages `effector`, `contact`, `attr`, `wm_rel`, `sdata`, `gc_value`, `gc_policy`, `gc_eval`, `gcivl`, `loop_gc` are registered there, and `run_family.ps1` takes a `-Method` parameter so a stage cannot silently read the stale view-free perception. Flags below are proposals to be fixed when implemented. Dates assume the late-January ICML deadline, which is not verified. The freeze date is the first thing allowed to slip; the held-out protocol never is.

Structure: there are two decision points (D1, D2). Everything else is a checkpoint (C1, C2, C3) reported while the next step is already running. The vertical slice (M1-M2) is: effector -> contact runs -> geometry attribution -> events -> WM -> the existing scripted loop on 4x4, then 4x5. The synthetic study, the FK ceiling and the supervised pilot are diagnostics attached to the slice and not gates in front of it.

### M0 (Oct 10-14): foundations and the cheapest decisive evidence

1. **M0.1 In-flight work and v1 evidence (CPU, no new GPU job).**
   - Read `JOB_LEDGER.md` and the output dirs. The v1 effector job on held-out 3x3 finished at 01:04; the 4x4 job is running; 4x5 follows. Never run duplicate GPU jobs.
   - Run `events_objects.py --effector` on the existing 3x3 `eff_*.npy` (minutes), and score tip error against FK at privileged press frames with `scripts/priv_effector.py`. This is the v1 baseline and the data point for whether the ID-loss keypoint class can reach the D2 criteria. v2 is written only if v1 does not already meet them.
2. **M0.2 Protocol, firewall refactor and commit (about 1-2 days of CPU work plus about 4 h of scaffolding).**
   - Move all privileged scoring into `diag_privileged.py` and `priv_loop.py`. This covers the sources that match today: `events.py` (`privileged()` reads the qpos and button_states files), `events_objects.py` L502-540, `objects.py` (`privileged()`, `privileged_diagnostic`), `closed_loop.py` and `closed_loop_objects.py` (the scripted arm reads `_cur_button_states` and `site_xpos`; `sim_start`/`sim_goal` logging; the scripted loop consumes `objects_report['privileged_diagnostic']`). Also rewrite docstring mentions.
   - Move `*_qpos.npy` and `*_button_states.npy` into `cache/<env>/_priv/` so method stages get a blind cache. Add `tests/test_blind_pipeline.py` (bit-identical dev outputs on the blind cache) and `tests/test_firewall.py` with its allowlist.
   - Ask the user to commit `method/` (currently untracked). Without a commit, the result JSONs cannot carry a meaningful commit hash.
   - `PROTOCOL.md`: splits (cube-single DEV), the press-to-event matching rule and ECE binning (7.6), decision rule, claim ladder, the pre-registered geometry expectations (`attr_expect.py`), the planned novel boards, gates and dates, NeurIPS fallback. `CONSTANTS.md` with classes (1.3.1) and `tests/test_constants.py`.
   - `freeze.py` skeleton: it hashes every `method/*.py` and also the frozen run-root `source/` copies made by `local/make_run_root.py`, not only repo files.
   - Enforce the 1000-episode cap at cache-build time (a full 5M-frame 64x64 cache would be about 61 GB). E: has 62 GB free (87% used), each 1M-frame env cache is 13 GB and run dirs are 8-31 GB. Ask the user before deleting any stale run dir (`puzzle_grow_*` 31 GB, `scene_grow_*` 15 GB, `unified_local` 2 GB are candidates).
3. **M0.3 E0 premise table (CPU, about 5 min per env).** `scripts/priv_effector.py` (FK of the pinch site from qpos; press matching from `button_states` or cube lift onsets; contact-run scoring). Re-run the premise on 100 VAL episodes of 3x3, 4x4 and cube-single.
   - Expected: gain constant, R2 xy at least .99 for windows 1-32 on puzzle; every dip to the floor is a press (at least .97); affine camera median at most 1 px and p90 at most 1.5 px (else switch to the homography); yaw residual at most .5 px (else add the lever term).
4. **M0.4 Synthetic identifiability, E2 (CPU, about half a day; a unit test inside the slice).** `tests/test_attribute.py` simulates boards 2x2..6x6 with the measured statistics: pressed light unobserved in 11-26% of events, effect dropout, `sigma/d` in {.2, .25, .3, .5}, tip bias in {0, .5, 1} spacings, lagged flips, and a rule-labelled init with 35% edge errors. It runs the exact EM code.
   - It reports geometry-only, template-only and fused accuracy separately, by interior/edge/corner.
   - Geometry-only accuracy must match the closed form of 3.4 within 2 points. This validates the geometry model.
   - The fused accuracy is expected to exceed the geometry-only bound only through the templates. The previous criterion "fused at least .95 for `sigma/d <= .3`" is reported as a fusion result and is not used to validate geometry.
   - The true offset is preferred over the one-cell-shifted offset, and the final accuracy is independent of the init error up to 40%. A failure sends the EM back to redesign before it touches real data.
5. **M0.5 E1 oracle ladder on puzzle-4x4 (GPU, in this order: R1a and R1b first, as they decide D1).** 4x4 is the primary rung because all 16 buttons are found. 3x3 is secondary and its incomplete-goal episodes (button 1) are flagged. 4x5 follows; its partial-label perception already exists (`method_pl`: 20 of 20 buttons, 34.4 events per episode).
   - Every rung uses the scripted PRIVILEGED press (`priv_loop.py` scripted mode), so the numbers are tagged "privileged-oracle executor".

     | Rung | Events | Acted labels | State reading | Transition |
     |---|---|---|---|---|
     | R0 | current events, refreshed perception | rule A0 | learned | learned WM |
     | R1a | current events | privileged, press-matched (7.6): an event with zero or two or more matched presses gets `target_known = False`, counts reported | learned | learned WM |
     | R1b | rebuilt from true presses (window `[press - L_pre, press + L90]`, before/after read at settled frames) | privileged, true | learned | learned WM |
     | R1c | A1 events | learned A3 (after M2) | learned | learned WM |
     | R2 | as R1b | true | privileged | learned WM |
     | R3 | true plan | n/a | privileged | exact simulator Lights-Out transition (an oracle, never the method) |
     | Rc | current events | latent codes (3.9), no acted-entity labels | learned | code-keyed WM |

   - R3 also gives the step-budget ceiling used by the budget model and by C3. It must be near 100% within the horizon, otherwise the loop and budget are the blocker.
   - Cost, re-budgeted: each model-training rung retrains the WM (4 min), support (3 min) and cost-to-go (50-90 min; half-sweeps for screening) and runs a 30-episode scripted loop (30-60 min). That is about 1.5-3 GPU h per rung. R0 + R1a + R1b + R2 + R3 is about 6-9 GPU h per board, Rc adds about 2-3 h, and R1c adds about 2-3 h. The privileged label builder is about 1 h CPU.
   - **D1 (decision, about Oct 17):** R1a on 4x4 at least 60%: go. 30-60%: WM generalisation (relative-position ensemble) comes first, while the attribution slice continues in parallel because it is needed anyway. Below 30%: the formulation cannot support the claim; re-scope to Tier 1 (perception and diagnosis) or arbitrate flat policies with the WM. Rc is read next to R1a, so a later effector failure has a measured branch.
6. **M0.6 E17 (baseline calibration, overnight queue).** `gcivl_img.py` baseline calibration: flat-view and flat-raw GCIVL on 3x3 for 150k then 400k steps (1.6-4 h), queued on nights when no other GPU job runs. It has no dependency and is the fallback for the whole easy-task story. Success: flat-view at least 50% on 3x3 at no more than 400k steps (about 70% of the published 73). If lower, fix the baseline first; if it cannot reach 50%, comparisons use the published numbers and the weaker matched re-implementation separately.
7. **M0.7 Novel boards (E23a, CPU in the background).** Check in 10 min that the installed ogbench registers puzzle-RxC for arbitrary R, C and that the play oracle runs; time a 10k-frame generation. If so, generate two novel boards with the 1000-episode cap: one easy-side (3x4) and one hard-side (5x5), chosen and recorded in `PROTOCOL.md` before generation. They are held-out and are not read before the freeze. If not, they are dropped and Tier 3 rests on 4x6 and cube-double, stated as a limitation.

### M1 (Oct 14-24): vertical slice, part 1 (effector and attribution on the existing events)

1. **M1.1 E3 FK-effector arm (CPU about 20 min after about 8 h of `attribute.py` geometry code).** Use the FK effector (PRIVILEGED control), a fitted affine camera and the floor-run moments, in the A1 pipeline on existing held-out 3x3 and 4x4 events. It gives the attribution ceiling and the first events-per-press numbers for the new assignment and merge. If it falls short (acted below .97 over all presses, or press-to-event coverage below .93), the evidence/event side (lag windows, assignment, reliability) is fixed before an effector is trained.
2. **M1.2 E4 supervised pilot (GPU about 8 min per board), run in the same GPU slot as E5.** Regress the FK-projected tip from frames with a small heatmap net (PRIVILEGED upper bound). If its median error exceeds 1.2 px (p95 3 px), effector-only attribution is out of reach for 4x6 spacing and the templates must carry more weight; the bound is recorded.
3. **M1.3 E5 EffectorNet v2 (GPU about 12 min per board; one job at a time).** Train on 3x3, then 4x4, then 4x5, against P2, anchor-only, v1, AgentNet sensitivity and direct q regression. D2 (3.2) is computed in the same step: contact-run precision/recall/timing and tip error at press frames, plus the 12-frame shadow overlay.
   - **D2 (decision):** pass -> floor-run contact runs go into the slice as the default. Fail -> pull ContactNet forward from M4 for one week; if that also fails, F2 (Rc) becomes the high-level path and the paper is re-scoped as in 3.9.
4. **M1.4** `attribute.py` end to end: calibration (3.3), the fused posterior with the fixed geometry weight, cross-expert temperatures, the KM lag model, the one-to-one assignment, unknown-not-unchanged, the contact-linked merge (arm A1), and export of `events_v2` including `acted_post` and `conf`. Add `attr_eval.py`.
5. **M1.5 E6 (CPU about 20 min per board).** Arms A0-A4, accuracy by corner/edge/interior and top-row spacing against the pre-registered geometry expectations, ECE, the lam grid, label-free estimates, and the lattice-shift margins.
6. **M1.6 E7 (GPU about 5 min per env).** Floor-run contact runs and the label-free selector on cube-single, cube-triple and scene, against qpos lift onsets and `button_states` (P). The ContactNet comparison E7b is deferred to M4.
7. **M1.7 Servo diagnostic (CPU simulator about 45 min).** The analytic unit-gain controller `a = clip(K(q* - q_hat), -1, 1)` using only the learned effector, 10 presses per button. This tests effector sufficiency for control; it is a privileged-scored diagnostic, not a method component. Expect at least .85 exact presses (the scripted privileged arm gets .91-.98).

### M2 (Oct 24-Nov 7): vertical slice, part 2 (WM and the scripted loop on learned labels)

- WM v2 on `events_v2` for 4x4 and then 4x5: relative-position, identity dropout, ensemble, soft labels, soft completion (E15). Support and cost-to-go retrained, with the half-sweep for screening.
- **R1c:** the existing scripted loop on 4x4, then 4x5, with learned A3 labels.
- **E26 ownership analysis for cube-triple and scene (CPU, minutes to hours).** For each failure of the v1 closed-loop logs, classify: plan not found, hidden goal, wrong object, position precision, grasp phase, execution timeout, surprise-replan. Report the fraction of official tasks whose goal contains a hidden entity. Each owner gets a first test: position precision (the oracle-position scripted arm), grasp phase (cube-single T1 in M3), hidden goals (the ceiling).
- **C1 (checkpoint, about Nov 7):**
  - press-level acted accuracy at least .85 on both 3x3 and 4x4 (target .95);
  - 3x3 edge presses 5/7/8 each at least the pre-registered geometry expectation for that cell minus 5 points (now .33/.31/.07);
  - 4x5 bottom row (14/15/17/19) improved over .14/.41/.26/.30;
  - events per press at most 1.10, merged-multi-press rate at most 3%, coverage at least .93;
  - cube-triple at least .87 and scene at least .81 on the label metrics (non-regression within 2 points of .888 / .827);
  - ECE at most .03 (7.6);
  - Routes V and C agree within 1 px;
  - **R1c at least R1a minus 10 points on 4x4 and 4x5.**
- On failure, use the fallback order of 3.9 (F0, F1, F2), each capped at one week.

### M3 (code from Oct 10, GPU from Nov 7, to Nov 28): low level

Code is written on the CPU in parallel with the slice. GPU runs follow the slice, except the executor-limited cube-single run, which takes the GPU in a free slot.

1. **M3.1** `goal_maps.py` + `tests/test_goal_maps.py` (cross pattern for a 3x3 corner press, pick-and-place, empty Delta, shift consistency, `event_status` on synthetic traces, the frozen-Delta rule). About 3 h.
2. **M3.2** `skill_data.py` + `tests/test_skill_data.py` (contact-end `lo_j`, belief-replay rows). About 6 h code, CPU about 10 min per env. **E10 audit** on 3x3, 4x4 and 4x5: at least .90 of interactions contain exactly one true press or move; Delta equals the true toggled set for at least .70 (3x3) and .80 (4x4, 4x5); dropped share at most 5%. If lower, adjust only the data-derived merge rule or the timeline fill, never per-env constants.
3. **M3.3** `skill_eval.py` (T1) and its calibration on `obj_skill_h0` and the hd skill (4.5), including the extended fidelity check. About 6 h code, about 5 min per 300 events.
4. **M3.4 E9 label-floor** (GPU 2 x 30 min on 3x3).
5. **M3.5 L0 training**, in this order: cube-single (executor-limited, dev), 3x3, then 4x4 and 4x5 after C1. Dry-run 2k steps first, logging peak VRAM, RAM and it/s. About 25 min per board.
   - **E12 robustness:** T1 with WM-predicted vs dataset goals within 3 points, success at least .90 with p=.1 degraded goals, and the wrong-side-effect variant.
6. **M3.6 E13 phase ladder** on cube-single (its objects+events chain, about 15 min, must run first), cube-triple and scene: arms a-f of 4.2 with E in {4, 8, 16}. Pick the simplest arm with the criterion, then freeze it on dev.
7. **L1/L2 only on measured T1 deficit** (4.4).
- **C2 (checkpoint, about Nov 28):** thresholds from the budget model (default .90 until E25 validates it):
  - puzzle exact single-event T1 at least .90 on 3x3, 4x4 and 4x5 (target .95; old skills .865 one-frame on 4x5, .35 on 4x4); CSR at least .6; collateral at most 2%;
  - cube-single pick-and-place at least .85 (target .90); cube-triple single moves at least .60; scene at least .60 per object type (target .85); gripper toggles at most 2 per event.
- If oracle-Delta success is below .85, the cause is precision or phase, not labels: escalate (E = 4, a wider 16x16 stage, two frames + cue, L1 value) before building anything else. If still failing by Dec 5, keep the single-frame BC skill for puzzle (exact presses .865 on 4x5) and the GC executor for cube/scene, disclosed as a deviation.

### M4 (Nov 7-28, only if the slice shows a deficit floor-run cannot fix): leg-2 core

- Trigger: events per press above 1.10, merged-multi-press rate above 3%, coverage below .93, or D2 failing.
- `reliability.py`, `events_contact.py` (arm A2, with `t_run` in the v2 schema), ContactNet if E7b says it beats floor-run, contact-seeded discovery (**E16 seeding:** reaches the visible places with |corr| at least .95, no false place on 4x4, CPU about 15 min).
- WM v2 acceptance (**E15**, on the slice's WM): clean presses with all lights right at least .97 (now .90-.96); toggle set right at least .9 under the rare-decile test (80% of the presses of the rarest-decile lights removed; the full hold-out of one light, 4x5 light 2 now about .1, is the stress test and is reported separately); surprise rate at most 20% on the lowest-frequency decile of buttons (now 89% on 4x5 light 2); offline plan found at least .95 (4x4 is .98 now, so it must not regress); ensemble disagreement vs realised surprise Spearman at least .5. Cost: GPU 4 min per WM, cost-to-go 50-90 min per env.
- Dev perception refresh with the same settings: 4x5 is done; cube-triple and scene still need partial-label SeeThrough + views, with privileged diagnostics checked for regression (about 6 GPU h). Overnight queue in `local/run_object_queue.ps1` (uncommitted today; commit with `method/`). One env at a time, with cache eviction after each env finishes.

### M5 (Nov 28-Dec 12): integration and dev closed loops

- `closed_loop_objects.py --low gc` with the frozen-Delta rule, `event_status`, the contact-end belief fill, data-derived timeouts and rule R; `belief.py` with its parity test; E24 competence tables and E25 budget model.
- **Eval harness sharding.** A shard is an evaluation process inside the same logical GPU job. Dry-run ONE shard first and record peak VRAM, RAM and steps/s. The shard count is the largest n with `n x peak VRAM <= 12 GB - 5 GB held by other processes - 1 GB margin` and `n x RAM <= 16 GB`. It never runs next to a training job. The earlier "30 episodes in 10-15 min" is a hypothesis until the dry run.
- **E18 dev loops.** Screening (5 tasks x 10 episodes x 1 seed) for debugging only, never for passing a checkpoint. Order: 3x3 and 4x5 first, then 4x4, cube-single, cube-triple, scene. The checkpoint numbers use the final tier (5 x 20 x 3 seeds for 3x3, 4x5 and cube-triple; 5 x 20 x 1 seed for 4x4, cube-single and scene).
- **C3 (checkpoint and ICML go/no-go).** Evaluated at the final tier, with hierarchical-bootstrap CIs (7.2):
  - **4x5, abstract-only arm (fallbacks off):** at least min(40%, 0.5 x the measured R3 ceiling) and never below 27% (best published pixel 17% plus 10 points); its gain over D_i and over matched flat-view at least 15 points.
  - 3x3: the hybrid is within 10 points of D_i (non-inferiority), with abstract-only reported separately, and 3x3 is a sentinel (any milestone that drops the hybrid below 50% is a regression).
  - 4x4: the hybrid at least 50%, or within 10 points of HIQL's 60.
  - cube-single: within 10 points of HIQL's 89.
  - cube-triple and scene: non-regression against v1 (13%, 23%) within 5 points, with HIQL (21, 49) as aspirational. The E26 ownership table says which losses can move. Scene tasks 4-5 are hidden-goal and likely capped near 60%.
  - **ICML target** if the 4x5 abstract-only criterion holds, 3x3 non-inferiority holds, and the held-out smoke runs pass. The previous rule "any three puzzle criteria on two boards" could be met by the D_i fallback alone and is removed. Otherwise re-target NeurIPS 2027 with the Tier 1 claims and whatever of Tier 2 holds, and spend the extra months on diagnosis, the stretch boards and the hidden-goal problem.

### M6 (Dec 12-Dec 22): evidence build-up and freeze

- **E19 seed noise floor:** 3 pipeline seeds (front end included) on puzzle-4x5, about 14 GPU h. If seed SD is at most 6 points, effects of about 14 points are resolvable with 3 seeds; else 5 seeds on the headline envs and no claims under about 11 points (7.2).
- **E20 ablation grid** (3x3, 4x5, cube-triple; screening 1 seed, then 3 seeds for the six marked *):
  - no view*; no partial-label SeeThrough*;
  - attribution A0 / MIL / relabel / A1 / A3*, and privileged;
  - WM identity vs relative-position*, single WM vs ensemble penalty, masked vs soft completion;
  - executor BC vs L0 vs L1 vs flat-view*, with and without the press-point channel;
  - search: greedy vs A*;
  - arbitration: rule R vs always D_i vs always abstract, stall switch on vs off*, residual fallback off, contact-end belief fill off, recent-state avoidance off, failure memory off;
  - data fraction 25/50/100% (answers the SHARSA 30x-data context).
  - The front end is fixed across downstream ablations and varied only in E19.
- **Baselines on dev** (Sec. 7.3).
- **E21 constants sensitivity** (CPU about 6 h, 0.5x/2x sweeps of the class (a) rules) and `CONSTANTS.md` finalisation.
- **Held-out smoke only** (crash/NaN/wall-clock/disk, no metrics) for 4x6, cube-double, cube-quadruple and the novel boards, one env at a time with eviction. The loop output is discarded; only exit codes, wall-clock and disk are recorded.
- **E21b optional pooled transfer:** train one low level on 4x4 + 4x5 and run T1 on 3x3 with no 3x3 training; success is within 10 points of the per-env executor.
- **FREEZE (Dec 22):** `freeze.py` writes `method/FREEZE.json` (git commit, clean-tree check, sha256 of every `method/*.py` and of the run-root `source/` copies, argparse defaults, constants table) and tags `freeze-v2`. `eval_harness.py --heldout` refuses to start on a mismatch.

### M7 (Dec 22-Jan 12): held-out

- **E22:** full pipeline on puzzle-4x6, cube-double, cube-quadruple and the novel boards (E23b), 3 pipeline seeds x 5 tasks x 30 episodes (450 per env), plus matched baselines on the same seeds. Bugs found afterwards are reported as post-freeze fixes with before/after numbers and never silently patched. Held-out privileged diagnostics (perception counts, acted accuracy) are read here for the first time, which is where the Tier 1 transfer claim comes from.
- The novel boards are the second and third held-out puzzle boards, so the recipe-transfer claim does not rest on 4x6 alone.

### M8 (January): analysis, writing, buffer

Error-budget waterfall, attribution identifiability figure, tables with all seed values, limitations, paper draft; extra seeds on the weakest claims.

### Compute summary (single 12 GB GPU, one job at a time; design estimates, replaced by measured values after the first dry runs)

Per-stage costs:

| Item | Estimate |
|---|---|
| Front end per env | 25 min (0.5-1.5 h on 3M/5M-frame envs; the 1000-episode cap applies to all) |
| Objects + events | about 10 min CPU |
| EffectorNet v2 | about 12 min per env |
| Attribution EM | 5-10 min CPU |
| WM ensemble (3 x 4 min) + support | about 15 min |
| Cost-to-go | 50-90 min (half-sweep for screening) |
| L0 executor | about 25 min (3x3) |
| L1 value + policy (conditional) | about 25 min + 70 min |
| Dev closed loop, 30 episodes | 30-60 min single shard; less only if the shard dry run supports it |
| Per env-seed, full pipeline | about 3 GPU h with L0, 4-5 h with L1 |

Total, re-budgeted with explicit counts:

| Block | Runs | h per run | GPU h |
|---|---|---|---|
| Oracle ladder (E1) R0-R3 on 4x4 and 4x5 | 2 | 8 | 16 |
| Rc + R1c on 4x4 and 4x5 | 2 | 5 | 10 |
| Effector v2 + pilots (E4, E5) | 3 | 0.3 | 1 |
| Executor dev (E9, E12, E13, L0 on 6 dev envs) | | | 25 |
| L1/L2 (only on measured deficit) | up to 6 | 2 | 0-12 |
| Perception refresh, cube-triple and scene | 2 | 3 | 6 |
| Dev final-tier pipeline runs (3x3 x3, 4x5 x3 incl. E19, cube-triple x3, 4x4, cube-single, scene x1) | 12 | 5 | 60 |
| D_i (flat-raw) training: 3x3, 4x5, 4x6 x3 seeds; 6 other envs and 2 novel boards x1 | 17 | 4 | 68 |
| Flat-view GCIVL training: 3x3, 4x5, 4x6 x3 seeds; cube-single and 2 novel boards x1 | 12 | 4 | 48 |
| Flat-view evaluation episodes (need the reader) | 12 | 3 | 36 |
| E24 competence tables (8 envs) | 8 | 1.5 | 12 |
| Ablations E20 (3x3 and 4x5 full grid; cube-triple subset; 3 seeds for the six marked rows) | | | 145 |
| Held-out E22: 5 envs x 3 seeds | 15 | 5.5 | 83 |
| **Total before reruns** | | | **about 520** |
| Reruns and bugs (15%) | | | about 80 |

Cut list if the schedule slips: reduce the ablation grid to the six marked rows at 3 seeds plus screening elsewhere (-55 h); flat-view only on 3x3, 4x5, 4x6 (-20 h); D_i single seed except 4x5 (-16 h); cube-quadruple and novel boards at 1 seed (-22 h); skip L1/L2 (-12 h); competence tables on four envs (-6 h). With the cuts the total is about 400 h before reruns. Both numbers are far below the single GPU's wall-clock capacity from Oct 10 to late January, so the critical path is the calendar and the slice, not GPU hours. The cluster is not part of the plan (about 33 GPU h per month of headroom).

### What changes course

| Event | Consequence |
|---|---|
| D1: R1a below 30% | Formulation cannot support the claim; Tier-1 paper or flat-arbitration re-scope |
| D1: R1a 30-60% | WM generalisation (rel-pos ensemble, soft completion) first; slice continues in parallel |
| D2 fails | ContactNet pulled forward one week; then F2 (Rc, code-keyed WM + Delta-map executor), Tier 1 plus executor, with Tier 2 only if Rc is competitive in the oracle ladder |
| C1 fails (R1c below R1a minus 10, or acted/ECE targets missed) after F0-F2 | WM trained on a confident subset, claims limited to Tier 1 |
| C2 fails | Keep single-frame BC for puzzle, GC for cube/scene (disclosed), or escalate 4.2/4.4 |
| C3 fails the 4x5 abstract-only criterion | NeurIPS 2027 with Tier 1 and the held parts of Tier 2 |

---

## 7. Evaluation protocol for the paper

### 7.1 Splits and freeze

- **DEV** (all design decisions, constants, ablations, privileged scoring allowed): puzzle-3x3, 4x4, 4x5, cube-single, cube-triple, scene. 3x3 and 4x4 are dev because the perception fixes were designed on them. cube-single is dev because its qpos was read for the physics premise and it is the testbed for the cube phase work and the T1 harness.
- **HELD-OUT:** puzzle-4x6, cube-double, cube-quadruple, and the novel boards (E23). Before the freeze they are used only for crash/NaN/wall-clock/disk checks; the loop output is discarded and only exit codes, wall-clock and disk are recorded. No success rate, accuracy or privileged diagnostic is read. The expected 4x6 spacing used in 3.4 comes from the env geometry and not from a held-out read.
- Freeze, firewall and constants ledger as in M6 and Sec. 1.3.
- **Statement for the paper:** the front end is retrained per env with identical settings, so held-out tests recipe transfer. The 1M-frame cap means fewer frames than published baselines on 4x5, 4x6, triple and quadruple. cube-single was dev, and the paper says so.

### 7.2 Seeds, episodes, statistics

- A pipeline seed `s` in {0,1,2} seeds all training (front end, WM, executor) and the evaluation env seeds (`a.seed*10000 + task*100 + ep`).
- Tiers:
  - dev screening: 5 tasks x 10 episodes x 1 seed (SE about 7 points; used for debugging and for effects of at least 15 points, never to pass a checkpoint);
  - dev final: 5 x 20 x 3 seeds on 3x3, 4x5, cube-triple; 5 x 20 x 1 seed on 4x4, cube-single, scene;
  - held-out: 5 x 30 x 3 seeds (450 episodes per env).
- Horizons are the OGBench ones: cube-single 200, cube-double 500, cube-triple and quadruple 1000, puzzle 3x3 and 4x4 500, 4x5 and 4x6 1000, scene 750. Success is `info['success']` at any step.
- Per-episode metrics: success, plan found, plan length, events executed, timeouts, steps per event, retries per episode, replans, arbitration mode, surprise rate, collateral changes.
- **Estimation.** The unit of resampling is the hierarchy (seed, task, episode): resample seeds, then tasks within seed, then episodes within task, with paired seeds and tasks across arms. A bootstrap over 3 seeds underestimates the seed variance, so the seed variance component is also taken from E19 (3 pipeline seeds on 4x5) and pooled across envs as a disclosed assumption.
- **Power (design estimates, to be recomputed from E19).** With a seed SD of 6 points, the SE of a 3-seed mean is about 3.5 points, and the unpaired difference of two arms has SE about 4.9. A minimum detectable effect at 80% power and alpha .05 is about 14 points with 3 seeds and about 11 with 5. The minimum detectable effect is reported with every comparison, and effects below it are labelled "not resolved", not "no effect". A single screening run at n=30 episodes has an SE of 7-9 points and cannot decide any checkpoint.
- **Pre-registered decision rule.** A beats B iff the 95% hierarchical-bootstrap CI of the paired difference excludes 0 and the point gap is at least 5 points. "Competitive" means the CI lower bound of (method minus best baseline) is at least -10. Hybrid vs D_i non-inferiority uses a 10-point margin with paired seeds; if the seed SD exceeds 6, the 3x3 and 4x5 comparisons get 5 seeds.
- Every result JSON stores commit hash, config hash and tier (offline / privileged oracle / learned closed loop).

### 7.3 Baselines and matched controls

| Row | Content |
|---|---|
| B1 published OGBench pixel | GCBC, GCIVL, GCIQL, QRL, CRL, HIQL (copy exact values from the paper table at writing time) |
| B2 matched re-implementations | In our code path and data cap (same 1M frames, same step budget, GPU hours reported): flat-raw (raw frame and raw goal, the OGBench interface; the same model as D_i, trained once and reused as the fallback) and flat-view (agent-free views as observation and goal, "GCIVL/HIQL with our perception"; 3x3, 4x5, 4x6, cube-single and the novel boards only). Calibrated against published numbers in E17. |
| B3 high-level ablations | No search (D_a), greedy one-step event selection by cost-to-go, A*; and a DeepCubeAI-style learned-heuristic search on atomic abstract states without events, to show what events add |
| B4 latent-space planning | A small DINO-WM/PLDM-style MPC on the same play data, on dev 3x3 and 4x5 only, one-week cap, only if C3 passes with margin; otherwise cited |
| B5 attribution baselines | A0 current rule, MIL contact, relabel, and the BISCUIT-style gated-factor arm |
| B6 v1 anchor | 3x3 3%, 4x4 7%, 4x5 23%, cube-triple 13%, scene 23% |
| Context rows only | WMPA (arXiv 2610.10932 as relayed in the task text, not re-verified here; state-based: 3x3 100, 4x4 51-58, 4x5 18-20, 4x6 17-20) and SHARSA (state, 4x5 91 with about 30x the data) |

Flat-raw evaluation does not need the reader and is fast. Training cost is counted explicitly in Sec. 6 (17 D_i runs and 12 flat-view runs).

### 7.4 Ablations and pre-registered expectations

Run on 3x3, 4x5 and cube-triple as in E20. Expectations written in `PROTOCOL.md` before the runs:
- removing the effector-based attribution (back to A0) costs at least 15 points on 4x5;
- removing search (D_a / flat-view) costs at least 15 points on 4x5 and at most 10 on 3x3 (the hybrid is not worse where flat is good);
- removing arbitration costs on 3x3.

A component whose removal changes nothing within the minimum detectable effect is dropped from the paper and the method description, or reported as unresolved.

### 7.5 Claim ladder (recorded in `PROTOCOL.md` at M0)

- **Tier 1 (independent of closed-loop outcome):**
  - agent-free perception: the dev numbers (buttons found 8 of 9 / 16 of 16 / 20 of 20; event timing .96 / .99 / .996) are design evidence only. The transfer claim is made only from held-out envs after the freeze (E22);
  - the error-budget waterfall (R0-R3, Rc);
  - identifiable attribution: accuracy, ECE, synthetic identifiability (geometry-only, template-only, fused), and label-free diagnostics validated on dev.
- **Tier 2:** the abstract-only arm beats matched flat-view GC by at least 15 points on 4x5 and beats the best published pixel number on 4x5 by more than 10 points; the hybrid is within 10 points of D_i on easy envs (3x3, cube-single).
- **Tier 3:** the same frozen code on the held-out envs: 4x6 at least the best published pixel baseline plus 15 and above matched flat-view by at least 15; the novel boards above matched flat-view by at least 15 on the hard-side board and within 10 on the easy-side board; cube-double within 10 of the best published pixel number; cube-quadruple reported without thresholds. If held-out results are worse than the dev trend by more than 15 points, the dev-to-held-out gap is reported as the main finding of the generalization section.
- **Not claimable:** OGBench-wide SOTA; hidden-goal scene tasks and cube stacks; state-based WMPA/SHARSA comparisons other than as context; "no hand rules" (say "data-derived rules"; the memory rule and the 2-means thresholds remain).

### 7.6 Evaluation definitions (written into `PROTOCOL.md`)

- **Press-to-event matching.** A privileged press at frame `t_p` is matched to the event whose core `[latest departure, first arrival]`, dilated by `[-L_pre, +L90]`, contains `t_p`. With several candidates, the one with the nearest core centre wins. An unmatched press is counted wrong at press level.
- **Multi-press events.** An event matched by two or more presses is counted wrong at press level for each of them, and its label is scored against the earliest press at event level. Events with no matched press are "spurious" and counted separately.
- **Metrics.** Press-level accuracy (correct presses over all presses) and event-level accuracy (single-press events with the correct label over all linked events) are both reported, together with events per press, the spurious share, the multi-press share and the coverage.
- **ECE.** Computed at press level on the max posterior, 10 equal-mass bins, with at least 300 presses per bin (100 VAL episodes give about 3100 presses). The 95% bootstrap CI is reported; the C1 criterion applies to the point estimate (at most .03), and the CI shows the noise floor of about .01-.015.
- **Terminology.** "Episode" is an environment episode; "contact run" is an effector-at-table interval.

---

## 8. Risks and mitigations

| Risk | Signal | Mitigation | Cheapest test |
|---|---|---|---|
| Keypoint locks on the shadow, a link or the wrist; the translucent gripper is too faint for 1.5 px at 64x64; v1 is already diffuse (R2 x .89, z .74; peak mass p50 .0375) | tip error above 1.5 px median | Height-linear offset calibrated per env; segmenter anchor; K heatmaps and keep the lowest ID loss; direct regression head; fall back to F2 | M0.1 v1 scoring, M1.2 supervised pilot (8 min), overlay of 12 frames |
| Contact-run detection through a learned `zeta` misses short dips | contact-run recall below .95 or timing worse than 2 frames | ContactNet pulled forward; F2 | D2 on 100 VAL episodes per board |
| Clipping and contacts (floor, bounds, button resistance) corrupt ID windows exactly at contact | zeta poor near the floor | Clipped integrator with learned bounds, Huber loss, short horizons dominate; at contact zeta is the floor by definition, so only xy matters | zeta error by height bin (lowest bin separately) |
| Perspective breaks the affine camera (spacing varies 6.6-9.4 px across rows) | affine residual p90 above 1.5 px | Homography on the table plane plus linear height term | E0 camera check |
| Yaw moves an off-axis keypoint | FK residual vs cumulative yaw above .5 px | Yaw lever term in the integrator | E0 yaw check |
| EM converges to a lattice-shifted solution (the failure of relabelling and effects-only EM) | log-lik margin of fitted vs shifted offsets | Effector track never sees changes; geometry weight fixed; temperatures fitted by cross-expert prediction; two estimators plus Route M; restarts from delta=0 and random; WM expert only after warm-up | M0.4 synthetic test; E6 with three initialisations; per-run margin |
| Geometry accuracy is lower than assumed (interior cell has 4 neighbours) | measured accuracy more than 3 points under the closed form | Pre-registered expectations; the templates and WM take over; supervised pilot gives the bound early | `attr_expect.py`, E6 by cell type and spacing |
| Resolution on 4x6 (about 6.5 px spacing vs about 1.4 px needed for .95 interior) | accuracy by top-row spacing | Effects and WM experts take over; supervised pilot gives the bound early; the 4x6 expectation is stated before the run | E6 on 4x4 and 4x5 top rows |
| Pressed light is the least readable, so vote mode is weakest where it is needed | Route V vs C disagreement | Route C does not use readings; both must agree | Calibration agreement check |
| Contact-run selector wrong off the puzzle (carry, drawer, window) | selector disagrees with qpos/button_states on dev | A0 stays a first-class arm per entity; geometry term absent for unlinked events; compare ContactNet to floor-run | E7 on cube-single, cube-triple and scene |
| Contact-linked merge chains adjacent presses | multi-press share above 3% | One-to-one per-entity assignment; KM lag with censoring; report the inter-contact gap next to L90 | Merged-multi-press rate (P) |
| Delta goals from partial or merged crosses (41% of events show one known change; 26% pressed light unread) teach a mixture | Delta completeness below .70 / .80 | Merge by data-derived gap; unknown entries out of Delta; member dropout and hallucination; interaction audit; press-point arm | E10 audit, E12 degradation curves |
| WM-predicted goals differ from realised-outcome goals (wrong side effects at edges) | T1 gap above 3 points between dataset and WM goals | Train-time perturbations; frozen intended Delta; planner-level replan on deviation; rel-pos ensemble | E12, wrong-side-effect variant |
| Label bias: unread pressed lights are rare in the data, so covered or rare lights stay rare | known-after share vs surprise rate | Soft-completion arm, rare-decile test | E15 |
| Reading lag makes the executor re-press | Delta nonempty after a completed press | Contact-end belief fill; train/test switch parity (4.3); lag-sampled delay augmentation | T1c chains, gripper toggles per event |
| Single-frame precision for 2-3 px buttons (4x6) | exact presses below .9 | E = 4-8, wider 16x16 stage, press-point channel, closed-loop correction | M3.5 on 4x5 / 4x4 |
| Single-frame phase ambiguity on cubes; policy copies the cue | dithering, CSR low | Cue dropout, chunk commitment, Q best-of-N, 2-frame + cue arm | E13 on cube-single (about 25 min per arm) |
| Value over-optimistic or miscalibrated (IQL expectile on stochastic play); AWR weights concentrate | V AUC vs T1 success | Built only on a measured T1 deficit; standardise advantages, clip weights at 20, Q margin rule | beta=0 vs 1 on 3x3 |
| T1 harness mismatch with the real loop (set_state, rendering, velocities, reader lag) | T1 does not reproduce .865 / .35, or the replay differs from the following dataset frames | Extended fidelity abort, calibration on old skills, reader-based tier on 100 events | M3.3 |
| Even with exact attribution, WM plus search does not solve 4x5 | R1a/R1b below 30% | Rel-pos ensemble WM, soft completion, plan-length control, lam toward 1; if still failing, Tier-1 re-scope | M0.5 |
| Perception holes (button 1 under the arm base) are real and pixel methods cannot see them | visibility probe AUC near .5; plan of length 0 | Ceiling stated for every method; unmatched-cluster flag and `q_hidden` in the arbitration; abstract-only denominator reported without those tasks | E16 |
| Arbitration demotes good episodes or picks the wrong arm | competence-table errors | Competence rule instead of a stall count; stall switch off for the headline; ablation rows | E24, E20 |
| Reviewers call the action-integrated effector proprioception | review | State plainly that actions supervise training and some online inputs; ablation against A0 | E6 |
| Reviewers call data-derived rules dev-tuned | review | `CONSTANTS.md` classes, derivation rules, 0.5x/2x sweeps, blind-cache test | E21 |
| Plain-GCIVL reproduction weaker than published (PyTorch, 1M-frame cap, 300-400k steps) | E17 below 50% on 3x3 | Report both numbers; tune alpha on dev only; fix baseline first | E17 |
| Statistical power (n=30, one seed) | seed SD above 6 points | Final-tier gates, hierarchical bootstrap with pooled seed variance, E19 early, 5 seeds on headline envs if needed, minimum detectable effect reported | E19 |
| Compute and infrastructure: one 12 GB GPU, about 5 GB held by other processes, native crashes, 62 GB disk, unmeasured shard speed, baseline training costs | freezes, OOM, no disk | One GPU job at a time via `run_stage.ps1` with CSV-resume and retry loops; chunked batches; torch->decord->torchcodec import order; per-env sequential queue with cache eviction; single-shard dry run before sharding; episode cap at cache build; explicit budget with a cut list | Time one 10k-step executor run and one GCIVL run to extrapolate (30 min) |
| Scope creep (the three source designs sum to over 1000 engineering hours) | schedule | One-week cap per fix; claim ladder fixed in `PROTOCOL.md` before results; leg 2 starts only on a measured deficit; value stack and ensemble only after a single-event number justifies them; two decision points only | Decision points D1, D2 and checkpoints in Sec. 6 |

---

### Appendix A: files and the milestone that creates them

| File | Milestone |
|---|---|
| `PROTOCOL.md`, `CONSTANTS.md`, `tests/test_firewall.py`, `tests/test_blind_pipeline.py`, `tests/test_constants.py`, `diag_privileged.py`, `priv_loop.py`, `freeze.py` skeleton, `attr_expect.py` | M0 |
| `scripts/priv_effector.py`, `tests/test_attribute.py` (E2) | M0 |
| `events_io.py`, `tests/test_schema.py`, `codes.py` (Rc), `local/run_family.ps1 -Method` | M0-M1 |
| `attribute.py` (geometry core and full), `attr_eval.py`, `effector.py --mode id`, `contact_runs.py` (floor-run and selector), `events_v2` export | M1 |
| `world_model.py --rel --complete`, ensemble | M2 |
| `goal_maps.py`, `skill_data.py`, `skill_gc.py` (L0), `skill_eval.py` | M3 |
| `reliability.py`, `events_contact.py`, `objects.py --seeds` | M4 (only on a measured deficit) |
| `belief.py`, `gcivl_img.py`, `budget_model.py`, `calib_rollouts.py`, `competence.json`, `eval_harness.py`, `closed_loop_objects.py --low gc`, `skill_gc.py` (L1/L2 if triggered) | M5 |
| `sensitivity.py`, `aggregate_results.py`, `local/run_shards.ps1`, `FREEZE.json` | M6 |

### Appendix B: critic issues and where each is fixed

| Issue | Fix location |
|---|---|
| Blocker: cube-single split contradiction | Splits paragraph, 7.1, 7.5, M0.2 |
| Blocker: firewall red on today's tree, not a dataflow guarantee, `method/` untracked, FREEZE ignores run-root source | 1.3 (blind cache, tests), M0.2, execution rules |
| Conflict with `AGENTS.md`/`CLAUDE.md` workflow; gate chain | Execution rules, Sec. 6 structure (D1, D2, vertical slice) |
| v1 effector evidence on disk unused | 3.1, M0.1 |
| Geometry accuracy argument wrong | 3.4 (closed form, table, pre-registered expectations), M0.4, E6 |
| G0b tested the wrong quantity; shadow omitted | 3.1, 3.2 (D2), M1.3 |
| Attribution gate a proxy; E1 not clean; budget | M0.5 (R1a, R1b, R1c, Rc, re-budget), C1 |
| Arbitration stall switch hurts hard boards | 5.4 (competence rule, switch off for headline) |
| Reach-interval train/test mismatch | 4.3.3-4.3.4, 5.3 |
| Hard-window merge chains presses; G biased | 3.5.2, 3.5.4, 3.6 |
| Temperatures fitted by flip likelihood | 3.4 (geometry weight fixed, cross-expert fit, lattice diagnostic) |
| Delta-only goals drop planner information | 4.1 (c5 press-point arm), 4.2 (arm f) |
| Contact-run selector privileged/per-family; no owner for cube/scene | 3.2 (label-free selector), E26, C3 |
| Fixed constants against the data-derived rule | 1.3.1, `tests/test_constants.py` |
| Masking unknown after-states biases the WM | 2.3, 3.5.5, E15 |
| F2 under-specified | 3.9 (code-keyed branch), Rc rung, stop table |
| Perception hole on 3x3 | 3.8, E16, 5.4 (`q_hidden`) |
| Decision rule and statistics underpowered | 7.2, C3 (final tier, relative 4x5 target) |
| ICML go/no-go satisfiable by the fallback; one held-out puzzle board | C3, M0.7, 7.1, 7.5 |
| Baseline compute under-counted; sharding conflicts | Compute summary, M5 sharding rule, execution rules |
| Ordering: executor in parallel and E17 late | M3 (after the slice, L1/L2 conditional), M0.6 |
| Evaluation definitions missing or inconsistent | 7.6, 3.6 targets, terminology |
| Interface gaps (`t_core`, `run_family.ps1`, ensemble, schema) | 3.6 (`t_run`, schema v2), 2.2, 2.3 |
| Minor: E7 ordering, WM-side-effect Delta, T1 dynamics fidelity, framing, step budget, resource notes | M1.6/M4, 5.1, 4.5, 1.4 and 3.1, 5.5 (E25), M0.2 and 7.1 |