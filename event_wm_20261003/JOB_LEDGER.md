# event_wm_20261003 job ledger

Method (adopted 2026-10-03): a discrete event world model learned from pixels, a cost-to-go
learned inside that model, heuristic search over events, and a short-horizon skill policy
that executes each event, with replanning after each executed event. Arena: OGBench visual
puzzle (3x3 to 4x6). Run directory: `/mnt/data/nhatnc129/jepa/event_wm`; job prefix `ew_`.

Labels used below:
- **privileged** = reads simulator state (harness checks, evaluation scoring);
- **offline** = scored on stored data/tasks without acting;
- **closed-loop** = the learned method acting in the environment.

## 2026-10-03
GPU quota at submit: 12/24.5 h (5th-ranked user has 49 h). Peer CTA jobs 56915/56916 queued, about 1.6 GPU-h.

| Job | Work | Resources | State |
|---|---|---|---|
| 56924 | Inventory: play-data press statistics; store eval-task init/goal observations; privileged scripted executor | main, 4 CPU, 32 GB, 45 min | COMPLETED 1:36 |
| 56925, 56926 | Data cache (numpy header API mismatch; then superseded by a split submission) | main CPU | FAILED, CANCELLED |
| 56927 | Data cache 4x5: first 1500 train episodes + all val as memory-mappable .npy | main, 2 CPU, 16 GB, 1.5 h | COMPLETED |
| 56928 | Data cache 4x6, 3x3, 4x4 | main, 2 CPU, 16 GB, 1.5 h | COMPLETED 3:52 |
| 56929 | Slow binary event code on frozen P1 encoder (56657) patch tokens, 4x5, K = 32 (smoke first) | mig, 6 CPU, 48 GB, 1 h 15 min; after 56927 | COMPLETED ~10 min |
| 56931 | Event code round 2: (a) linear SFA + ICA; (b) MLP head K = 24, gap 30, slowness x4 | mig, 6 CPU, 48 GB, 50 min | COMPLETED ~12 min |
| 56933 | Pipeline smoke | mig | FAILED: EventWM method named `children` shadowed nn.Module.children (renamed `successors`); the events stage ran |
| 56935 | Event code round 3: SFA, then ICA inside the N slowest directions (N = 32/64/128/256), keep binary components with low binarised flip rate | mig, 6 CPU, 40 GB, 30 min | COMPLETED |
| 56936 | Pipeline smoke rerun from the planner stage (reuses 56933 events) | mig, 35 min | submitted |
| 56937 | Event code round 4: kurtosis ("cube") FastICA, up to 2000 iterations; selection = slow (binarised flip 2-means) AND binary (separation 2-means); N = 48/64/96 | mig, 30 min | COMPLETED |
| 56938 | Event code round 5: band-wise ICA (below / above the eigen-gap), N = 64/96 | mig, 30 min | COMPLETED |
| 56939 | Full pipeline 4x5, training stages (events with debounced/merged detector, planner + offline checks, skill), code 56938/N64 | mig, 55 min | submitted |
| 56939 | (see above) | | COMPLETED |
| 56940 | Full pipeline 4x5, closed loop: arms skill+learned, scripted+learned (privileged), skill+oracle (privileged); 4 episodes x 5 tasks each | mig, 1 h 30 min; after 56939 | CANCELLED after 1 min: the inventory task observations are black (see below), so offline task checks were meaningless; fix the render first |
| 56942 | Render check: the 5 eval tasks rendered by {lewm venv, ogbench venv} x {egl, osmesa}; pixel stats against dataset frames | mig, 15 min | COMPLETED |
| 56943 | Cost-to-go capacity diagnostic: exact-label regression of Lights Out distance with the heuristic's network family ([s,g] vs [s,g,s xor g]; widths 512/2048) | mig, 30 min | COMPLETED 13 min |
| 56944 | Planner v2 on the 56939 events: cost-to-go width 2048 + b xor g features, 100k value-iteration steps; offline checks on the 56942 task frames | mig, 50 min | COMPLETED 27 min |
| 56945 | Closed loop on planner v2 + 56939 skill: 3 arms x 4 episodes x 5 tasks | mig, 1 h 30 min | CANCELLED after 2 episodes: ~4 min per 1000-step episode (rendering-bound), and the code still misread single frames |
| 56950 | Code refinement v1: pseudo-labels = debounced code, frames >= 12 from events, linear readout on all patch tokens | mig, 40 min | COMPLETED (no gain) |
| 56951 | Code refinement v2: segment-majority pseudo-labels (state is constant between events), margin 3 | mig, 40 min | COMPLETED |
| 56953 | Step timing per render backend | mig, 10 min | COMPLETED |
| 56955 | Closed loop, parallel (18 CPU) | | CANCELLED: QOS limit is 16 CPU per job (32 per user) |
| 56965 | Closed loop, parallel (14 workers): refined code 56951 + planner v2 56944 + skill 56939; 3 arms x 8 episodes x 5 tasks | mig, 16 CPU, 96 GB, 1 h | COMPLETED 44 min |
| 56980 | Round 2 training: events from the refined code (count-based vocabulary), planner (width 2048 + xor, 150k steps), skills (tau-conditioned; tau-free on the last 16 frames) | mig, 6 CPU, 48 GB, 1 h | COMPLETED ~47 min |
| 56981 | Round 2 closed loop: 5 arms x 6 episodes x 5 tasks, WM event-consistency filter | mig, 16 CPU, 96 GB, 1 h; after 56980 | running |
| 56986 | Skill v2 training (2-frame history, 8-action chunks, spatial event target): full segments, and the last 21 frames | mig, 6 CPU, 64 GB, 45 min | submitted |
| 56987 | Skill v2 closed loop: {full, tau20} x {oracle plan (PRIVILEGED), learned high level}, 6 episodes x 5 tasks | mig, 16 CPU, 96 GB, 1 h; after 56986 | running (closed_loop.py gained an additive skill-v3 branch while it ran; v2 code path unchanged) |
| 56997, 56998 | Skill v3 (first submission) | | CANCELLED before start: resubmitted with press-and-release segments |
| 56999 | Skill v3 training with press-and-release segments (through the visible change + 3 frames): full segments, and the last 21 frames before the press | mig, 6 CPU, 64 GB, 50 min | submitted |
| 57000 | Skill v3 closed loop: {full_rel3, tau20_rel3} x {oracle plan (PRIVILEGED), learned high level}, 6 episodes x 5 tasks | mig, 16 CPU, 96 GB, 1 h 15 min; after 56999 | submitted |
| 57004 | Recovery test (v2 tau20 + learned) | | FAILED at start (ARMS parsing): `skill2.sh` would also have written into the existing `loop_tau20_learned`. Fixed with semicolon-separated ARMS and a `TAG` output suffix; nothing was overwritten |
| 57005 | Recovery test: v2 tau20 + learned high level, timeout 40, on timeout replay the mean post-event (lift-off) action for 8 steps | mig, 16 CPU, 96 GB, 30 min | COMPLETED 11 min |
| 57013 | Base encoder 4x6 (LeWM recipe, 1M cached frames, 60k steps) | mig, 6 CPU, 40 GB, 1 h 05 min | submitted (GPU quota at submit: 20/28 h) |
| 57015 | 4x6 code stage: SFA + band ICA (N = 64/96, label-free choice), events, segment refinement, events, task frames | mig, 6 CPU, 64 GB, 50 min; after 57013 | submitted |
| 57016 | 4x6 plan stage: event WM + cost-to-go (150k) + offline checks; skill v3 (tau20, release 3) | mig, 6 CPU, 64 GB, 1 h 10 min; after 57015 | submitted |
| 57014 | **Official-protocol eval 4x5:** fully learned method (v3 tau20_rel3 + learned high level), 5 tasks x 20 episodes, held-out env seeds (--seed 1; dev used seed 0) | mig, 16 CPU, 96 GB, 45 min | COMPLETED 19.5 min |
| 56933 | Pipeline smoke on the 56929 code (tiny settings; checks code paths only, not a result) | mig, 6 CPU, 48 GB, 40 min | submitted |

### 56924 inventory
Play data (train split): about 30 press events per 1001-step episode in every size; median 32-34 steps
between presses (5-95%: 29-40); 3-5 lights toggle per event, consistent with single presses.

Privileged scripted executor (minimal GF(2) press set, greedy nearest-button order, gripper closed, no wander):
about 10 steps per press. 18/20 tasks solved: 4x5 task 5 (20 presses) took 198 steps, 4x6 task 5 (24 presses)
221 steps. The two failures (4x5 task 3, 4x6 task 4) are the controller stalling on one press for the rest
of the episode (no retry); 4x4 task 4 stalled for 148 steps and then recovered. The 1000-step limit is not
binding for 20-24 presses, even at the play data's ~33 steps per press.

### 56929 event code, round 1 (MLP head, K = 32, gap <= 10; offline, privileged scoring)
- **Slow bits.** The flip-rate split gives 11 slow bits at 0.0044-0.0061 flips/step (true lights: 0.0061) and 21 fast bits at 0.022-0.063 (arm).
- **Alignment.** Only 7/20 lights have a bit with >= .99 agreement (row nearest the camera, plus lights 9 and 14). The other 13 lights are at chance (.52).
- **Events from slow bits.** Precision .92, recall .60; 17 XOR patterns, mapping to the true button with purity .90.
- **Purity.** Low (.08 slow-only, .55 all bits), because most lights are missing.

Diagnosis: variance and decorrelation do not demand that the code capture *all* slow state. SGD settles on easy arm features. Round 2 (56931) tests a linear SFA + ICA code and a stronger-slowness MLP head.

### 56931 event code, round 2 (offline, privileged scoring)
- **(a) SFA + ICA, eigen-gap rule.**
  - Spectrum: 7 very slow components (slowness .0015-.0031), then a jump to .0083 and a smooth continuum.
  - ICA gives 7 bimodal components (separation 5.8-7.6), which are exactly the same 7 lights as round 1 (9, 14, 15-19; agreement .99-.998).
  - Event recall .44, precision .87.
  - The slow lights' frame-to-frame variance is ~10x below a clean step at the light flip rate. So their visible change spreads over ~10 frames: the gripper covers the button at the press.
  - The other 13 lights (rows farther from the camera, which the arm covers often) are not among the slowest *quadratic* features, although a linear probe reads them at 99.99% (P1 56691).
  - Next: rank by binarised flip rate instead of variance (round 3).
- **(b) MLP head, K = 24, gap 30, slowness x4.** Collapsed to a constant code: the variance hinge stayed at its maximum (.45) and slowness won. The MLP route is fragile under this loss.

### 56933 events stage on the round-1 code (smoke; privileged scoring)
- 17 event types, all train events in the vocabulary.
- 91% of detected events match a true toggle within 3 frames; type-to-button purity .895; 12/20 buttons covered.
- Only 19 events per episode are detected versus 30 true, consistent with the missing lights.

### 56935 event code, round 3 (offline, privileged scoring)
- **N = 64.** 29 components selected; 12 lights at >= .99 (lights 0-8, 10-13: the far rows that rounds 1-2 missed); purity .987.
- **N = 128.** 13 lights at >= .99 (the same far-row lights); purity .983.
- **The 7 near-row lights** (9, 14-19) are lost at N = 64/128 and partially present at N = 32 and 256.
- **FastICA (logcosh, symmetric) hit the 400-iteration cap at every N.** The components are therefore partly mixed, and which lights separate depends on N.
- **Selection.** Low binarised flip rate alone admits non-bimodal components (separation about 2-3: slowly drifting continuous features binarised at their median). Event precision is low (.25-.37) because of them.
- Round 4: kurtosis contrast, more iterations, and a bimodality criterion.

### 56937 event code, round 4 (offline, privileged scoring)
- **N = 64 and 96.** Exactly 13 components pass the slow-and-binary selection (separation threshold about 4). They are the 13 far-row lights (12 at >= .99, light 10 at .988); purity .80.
  - Event precision .87-.88, recall .71-.72; 33-35 patterns; pattern-to-button purity .91-.92.
- **FastICA still hit the iteration cap (1999).**
- **The 7 near-row lights** (9, 14-19), which separate perfectly by ICA inside the 7 slowest directions (round 2, 6 iterations), do not appear when ICA runs on the whole 48-96-dimensional subspace.
- Round 5 runs ICA separately in each slowness band, split at the eigen-gap.

### 56936 pipeline smoke on the round-1 code (code paths only, not a result)
Every stage runs end to end:
- event WM exact 1.0 on the 11-bit code;
- cost-to-go uninformative about true d*, because the code misses most lights;
- the init and goal codes of the evaluation tasks coincide, so plans are empty;
- closed loop 0/2.

Closed-loop speed was about 0.15 s/step, dominated by repeated searches with an uninformative heuristic.

### 56938 event code, round 5: band-wise ICA (offline, privileged scoring)
- **Method.** ICA runs separately on the 7 slowest SFA directions (below the eigen-gap) and on directions 8..N.
- **Selection (label-free; slow and binary) keeps exactly 20 components** at N = 64 and 96.
  - At N = 64: 19/20 lights at >= .99 agreement, light 10 at .989.
  - Code purity .9995 (the code determines the configuration); 5,723 distinct codes for 2,876 configurations (noise variants).
  - Pattern-to-button purity .99.
- **Event detection** with the whole-code stability rule: precision .80, recall .57.
  - Cause: a press reveals its 3-5 lights over several frames (the gripper covers them), so the code changes in steps.
  - Fix used from here on (common.detect_events): per-bit debounce (3 frames), then merge changes closer than 10 frames into one event. Presses are >= 29 steps apart, so merges never join two presses.

### 56939 full pipeline 4x5, training stages (offline; privileged scoring)
- **Events, label-free, with the debounced/merged detector.**
  - Train: 44,478 detected vs 44,398 true presses (29.7 per episode); 99.6% match a true toggle (within 5 frames of the span).
  - Vocabulary: 21 types covering all 20 buttons; type-to-button purity .99995 (val 1.0); 99.2% of events are in the vocabulary.
- **Event WM.** Exact on 100% of 8,826 validation events.
- **Cost-to-go by value iteration in the WM** (40k steps, target copied every 1,000).
  - Mean h by true d* tracks d* up to 4 (1.00, 2.06, 3.07, 4.02), then is flat at about 5 for d* = 6-18.
  - Event-sign accuracy: 1.0 / .98 / .69 / .52 / .47 / .46 for d* 1-2 / 3-4 / 5-6 / 7-8 / 9-12 / 13+.
  - This is the same radius as the P1 GCIVL wall (.99 / .92 / .65 / .50 / .46): imagined Bellman targets alone did not move the wall.
- **Offline search between validation frames** (BWAS, at most 20k expansions), solved fraction by d*: 1.0 (1-2), .93 (3-4), .80 (5-6), .80 (7-8), .40 (9-12), 0 (13+). Plans are near-optimal when found (length/d* about 1.0-1.1).
- **Offline evaluation tasks.** Init and goal codes coincide (empty plans) because **the inventory's task observations are almost black** (mean pixel 1.4 vs 115.8 in the data): the ogbench venv + OSMesa renders black frames. The pipeline-smoke closed loop (lewm venv + OSMesa) did detect true presses, so that renderer produces usable frames.
- **Skill BC.** Validation action MSE (normalised) .22 / .36 / .59 / .25 for tau 0-5 / 6-15 / 16-30 / 31+.

### 56942 render check
On the MIG node all four combinations render the evaluation tasks like the dataset: mean pixel 115.2-115.5 vs 115.6; per-channel means within 1. The task frames differ from one another (no buffer aliasing).
- The black task frames of 56924 came from the CPU node (worker-0, `main`) with OSMesa.
- Closed-loop jobs run on MIG nodes. `closed_loop.py` now aborts if a reset frame or goal frame is dark (mean < 20).
- Offline task checks use `render_56942/torch_egl/tasks_4x5.npz`.

### 56943 cost-to-go capacity (exact labels; diagnostic)
Random 4x5 pairs (d* uniform 0-20), 30k steps x 4096 pairs, 3-hidden-layer MLP.

| Variant | MSE | Single-press sign accuracy |
|---|---|---|
| [s, g], width 512 | 3.10 | .98 / .89 / .73 / .65 / .64 / .84 by d* bin |
| [s, g, s xor g], width 512 | 0.69 | .999 / .97 / .91 / .86 / .82 / .96 |
| [s, g], width 2048 | 0.014 | 1.0 at every distance |
| [s, g, s xor g], width 2048 | 0.002 | 1.0 at every distance |

The Lights Out distance is learnable by a wide MLP. The ~5-press wall of the 56939 cost-to-go (width 512, value iteration) is capacity plus bootstrapping, not an unlearnable function. Planner v2 uses width 2048, xor features and 100k steps.

### 56944 planner v2 (offline; privileged scoring)
- **Cost-to-go** (width 2048, b xor g features, 100k steps; the target mean was still rising at the end, 6.28).
  - Spearman with true d* .73 (v1: -.02). Mean h rises monotonically: 0.97, 1.98, 3.59, 4.23, 5.17, ... 9.70 (d* 12), 12.58 (d* 18); compressed but ordered.
  - Event-sign accuracy .98 / .96 / .94 / .83 / .67 for d* 3-4 / 5-6 / 7-8 / 9-12 / 13+ (v1: .98 / .69 / .52 / .47 / .46). The wall moved from about 5 presses to about 12.
- **Offline search between validation frames**, solved: 1.0 / .93 / .80 / .80 / .40 / .18. The failures at short distances are code misreads of single frames, not search failures.
- **Official tasks (re-rendered frames, 56942).** Tasks 1 and 2 are solved offline with optimal plans (4 and 10 events). Tasks 3-5 produce plans *shorter* than d* (13/11/13 vs 14/16/20): a few code bits of the init or goal frame are wrong, and one wrong light changes a Lights Out solution entirely.

### 56945 closed loop (cancelled)
Two task-1 episodes failed: first plans of 12 and 9 events for d* = 4 (code misreads), 1000 steps each, about 4 minutes per episode.

### 56950 / 56951 code refinement (label-free; privileged scoring)
- **v1** (pseudo-labels = per-bit debounced code, only frames >= 12 from events, 12% of frames): frame-exact .919 vs .937 for the original code. No gain.
- **v2** (segment-majority pseudo-labels: per-bit majority over each inter-event segment, all frames > 3 from events, 73% of frames).
  - Frame-exact (all 20 lights right) **.985** (original .937); per-light .997-1.0.
  - Events: precision 1.0, recall .999, exactly 20 patterns; pattern-to-button purity 1.0; code purity .99998.
  - Bit order is unchanged, so the planner and skill are reused.

### 56953 step timing (MIG node)
`env.step` takes ~0.155 s with either EGL or OSMesa, and rendering alone takes 0.155 s: the cluster has no GPU GL, so both are software. The model calls take 3.5 ms. `closed_loop.py --workers N` now runs episodes in N spawned processes (LP_NUM_THREADS=1, models on the shared GPU).

### 56965 first full closed loop, 4x5 (dev, 8 episodes x 5 tasks per arm)

| Arm | Success | By task (d* 4 / 10 / 14 / 16 / 20) |
|---|---|---|
| Method: learned skill + learned high level | 0/40 | 0 / 0 / 0 / 0 / 0 |
| PRIVILEGED scripted presses + learned high level | 15/40 | 8/8, 7/8, 0, 0, 0 (59 and 232 steps on tasks 1-2) |
| PRIVILEGED learned skill + oracle plan | 0/40 | 0 / 0 / 0 / 0 / 0 |

Attribution:
- **High level.** Tasks 1-2 are solved with optimal first plans.
- **Tasks 3-5 are not a perception error.** The init and goal codes decode to the true states in every episode (decoded through the bit-to-light map from validation data). But the first plans are 13/11/13 events for d* 14/16/20 in every episode.
  - Cause: the event vocabulary has a 21st type that flips **one bit** (92 occurrences vs 2,099-2,269 for each true button). It is code noise that passed the 99% coverage rule.
  - The planner uses it as a shortcut. The scripted executor then presses the mapped button about 230 times while the planner loops.
  - Fix: vocabulary = high-count cluster of a 2-means on log counts.
- **Low level.**
  - When the skill presses, it mostly presses the commanded button.
  - But code changes without a true toggle are frequent (up to 81 per episode): the hovering arm occludes lights for >= 3 frames. Each one triggered a replan and reset the skill's tau countdown, often just before a press. Many episodes never press at all.
  - Fixes:
    - a WM event-consistency filter in the closed loop: accept a debounced change only if some event maps the confirmed code to it; otherwise wait, and force it after 30 frames;
    - a tau-free skill trained on the last 16 frames before each event.

### 56980 round 2 training (offline; privileged scoring)
- **Events from the refined code.** Vocabulary = exactly 20 types (noise patterns with counts <= 6 dropped by the 2-means rule); type-to-button purity 1.0.
  - 44,333 events vs 44,398 true presses; 99.9% matched; 99.8% in the vocabulary.
- **Event WM.** Exact 1.0 on 8,851 validation events.
- **Cost-to-go** (width 2048, b xor g, 150k steps). Spearman with true d* .954.
  - Mean h by d*: 1.0, 2.0, 3.3, 4.1, 5.1, 5.9, 6.9, 7.8, 8.7, 9.6, 10.4, 11.3, 12.3, 13.1, 14.0, 14.9, 16.0, 16.9 (d* 1-18).
  - **Event-sign accuracy .99+ at every distance** (1.0 / 1.0 / .993 / .992 / .994 / .991 for d* 1-2 ... 13+), against the P1 GCIVL value's .99 / .92 / .65 / .50 / .46. The goal-distance wall is gone on this arena.
- **Offline search between validation frames**, solved: 1.0 / .975 / .95 / 1.0 / .925 / .975 by d* bin. Length/d* = 1.00. Median 21-2,771 expansions, 0.11 s or less.
- **All 5 official tasks solved offline with optimal plans** (4, 10, 14, 16, 20 events; 467-4,563 expansions), from re-rendered init/goal frames (56942).
- **Skills.** Validation action MSE at tau 0-5 / 6-15: tau-conditioned .214 / .316; tau-free (last 16 frames) .213 / .379.

### 56981 round 2 closed loop, 4x5 (dev, 6 episodes x 5 tasks per arm; WM event-consistency filter)

| Arm | Success | By task (d* 4 / 10 / 14 / 16 / 20) |
|---|---|---|
| Method: tau-free skill (last 16 frames) + learned high level | 0/30 | 0 / 0 / 0 / 0 / 0 |
| Method: tau-conditioned skill + learned high level | 0/30 | 0 / 0 / 0 / 0 / 0 |
| **PRIVILEGED scripted presses + learned high level** | **29/30 (96.7%)** | 6/6, 6/6, 6/6, 6/6, 5/6 |

- **The learned high level works in closed loop.** Every successful episode uses exactly d* presses (optimal first plans), about 14 steps per press. Steps: ~59 / 139 / 210 / 195 / 280 per task.
  - The filter ignored 0-14 unexplained frames per episode; no event had to be forced.
  - The single failure (task 5) started from an 8-event plan: one misread goal frame. The goal frame is read once, so replanning cannot repair it.
- **The tau-free skill + learned high level** presses rarely: 0-5 presses per 1000 steps, about half on the wrong button. A timeout replan every 80 steps.
  - Spurious events are now rare (filter), so the bottleneck is the learned skill's execution.
  - Reference point: published pixel results on 4x5 are <= 17% (HIQL).

### 56987 skill v2 closed loop (partial: first arm)
**PRIVILEGED oracle plan + skill v2 (full segments): 5/30.** By task: 5/6, 0, 0, 0, 0. This is the first learned skill to solve tasks.
- Most presses now hit the commanded button: tasks 3-4 have 5/5, 6/6, 7/7 right presses.
- But two failure modes remain:
  - **Slow pressing.** 3-10 presses per 1000 steps in many episodes.
  - **Repeated presses.** Up to 60 presses, all "right", without success: the same button is pressed again.
- Mechanism: the gripper covers the button and its neighbours while pressing, so the change is confirmed only after lift-off. The skill for e was trained only up to the press, so it keeps pressing e and toggles it back.
- Fix (skill v3 submission 56999): segments run through the visible change + 3 frames (`--release 3`), so each skill = approach + press + lift-off.

### 56987 skill v2 closed loop (dev, 6 episodes x 5 tasks per arm)

| Arm | Success | By task (d* 4 / 10 / 14 / 16 / 20) |
|---|---|---|
| PRIVILEGED oracle plan + v2 full segments | 5/30 | 5/6, 0, 0, 0, 0 |
| PRIVILEGED oracle plan + v2 last 21 frames (tau20) | 7/30 | 6/6, 1/6, 0, 0, 0 |
| **Method:** learned high level + v2 full | 2/30 | 2/6, 0, 0, 0, 0 |
| **Method:** learned high level + v2 tau20 | **7/30 (23.3%)** | 5/6, 1/6, 1/6, 0, 0 |

- These are the first fully learned, non-privileged closed-loop successes, including one 14-press task.
- Reference: published pixel results on visual-puzzle-4x5 are <= 17% (HIQL; official protocol).
- **This is not a comparable claim yet:** dev seeds, 6 episodes per task, wide CI, and no same-protocol baseline.
- The learned high level is as good as the oracle plan here (7/30 each), so the low-level skill is the remaining bottleneck.

### Analysis of the best method arm (56987, v2 tau20 + learned)
- About 90% of presses hit the commanded button; repeated presses are now rare.
- Inter-press interval: median 44 steps (p25 26, p75 67, p90 144). Up to 12 80-step timeouts per episode.
- The episodes are dominated by stuck periods, which replanning alone does not escape. 20 presses within 1000 steps need <= ~45 steps per press.
- Added `--recover-steps`: on a timeout, replay the play data's mean action over the 5 frames after each detected event (a lift-off; label-free) before retrying.

### 57005 recovery test (dev, 6 episodes x 5 tasks)
**Method (v2 tau20 + learned high level) with timeout 40 + lift-off recovery: 9/30 (30%).** By task: 6/6, 3/6, 0, 0, 0 (without recovery: 7/30).
- Episodes are more consistent, but the press rate did not rise: 6.5 presses per episode, median 63 steps between presses, about 15 recoveries per episode.
- **Per-button breakdown (commanded and correctly pressed).** Buttons 0, 4, 5, 10, 11, 19 are almost never pressed (0-1 times, although 12-18 episodes needed each). These are mostly the left column and the right corners. Centre buttons are pressed reliably: 3 (24), 13 (19), 14 (19), 16 (24).
- Training data is balanced (~2,200 events per button) and the scripted executor presses every button. So the learned skill has a systematic reach deficit at the board edges (likely regression to the mean of MSE behaviour cloning).
- Long tasks need those buttons, which explains 0/18 on tasks 3-5.

### 57000 skill v3 closed loop (dev, 6 episodes x 5 tasks per arm)

| Arm | Success | By task (d* 4 / 10 / 14 / 16 / 20) |
|---|---|---|
| PRIVILEGED oracle plan + v3 full_rel3 | 30/30 (100%) | 6/6 on every task |
| PRIVILEGED oracle plan + v3 tau20_rel3 | 30/30 (100%) | 6/6 on every task |
| **Method (fully learned, non-privileged): learned high level + v3 full_rel3** | **28/30 (93.3%)** | 5/6, 5/6, 6/6, 6/6, 6/6 |
| **Method: learned high level + v3 tau20_rel3** | **29/30 (96.7%)** | 6/6, 5/6, 6/6, 6/6, 6/6 |

- **Skill v3 solves the low level:** CNN on two raw frames trained end to end, plus press-and-release segments.
  - With the oracle plan, every episode uses exactly d* presses at about 30 steps per press (task 5 in ~630 of 1000 steps).
- **The fully learned method solves all 18 long-task episodes** (14, 16 and 20 presses) with optimal first plans.
- **Both failures are a single misread goal light** (light 11 in task 1, light 3 in task 2). The goal frame is read once; the init codes were correct.
- **Reference:** published pixel results on visual-puzzle-4x5 are <= 17% (HIQL).
- **Still to do before any claim:** official-protocol episode counts and held-out seeds, other grid sizes (3x3 / 4x4 / 4x6), same-protocol baselines, and ablations.

### 57014 official-protocol evaluation, visual-puzzle-4x5 (fully learned, non-privileged; held-out env seeds)
**94/100 (94%), 95% CI [89, 98]** (bootstrap stratified by task).
- By task: 1.00 / .95 / .90 / .85 / 1.00 (d* 4 / 10 / 14 / 16 / 20).
- Median steps of successful episodes: 130 / 269 / 387 / 444 / 563 (limit 1000).
- **All 6 failures are a single misread light in the goal frame** (lights 14, 13, 17, 4, 17, 7). They produce wrong first plans (12/13/11/12/11/12 events). Init codes were correct in all 100 episodes.
- Context: published pixel results on visual-puzzle-4x5 are <= 17% (HIQL; OGBench). Our number is one training seed; OGBench reports 8 training seeds.

### 57019 self-training round 2 (from the refined code's events, 1M frames)
- Frame-exact .988 (round 1: .985). Event precision 1.0, recall .999. Diminishing returns.
- The remaining per-light errors are on lights 7, 8, 12 and 13 (.996-.998).
- Goal frames are harder than data frames (6% vs ~1.2% frames with an error), likely because of low arm poses at random places.

| Job | Work | Resources | State |
|---|---|---|---|
| 57019 | Self-training round 2 | mig, 40 min | COMPLETED |
| 57020 | CNN code reader on pixels trained on segment-majority pseudo-labels (round-2 events), shift augmentation, 40k steps; closed_loop.py gains `--reader` | mig, 6 CPU, 64 GB, 45 min | COMPLETED 11 min |
| 57022 | Official-protocol eval 4x5 rerun with the CNN reader (same held-out seeds as 57014) | mig, 16 CPU, 96 GB, 45 min | COMPLETED 15 min |
| 57025 | Official-protocol eval 4x5 with the CNN reader on fresh, never-inspected env seeds (--seed 2) | mig, 16 CPU, 96 GB, 45 min | COMPLETED 15 min |

### 57020 CNN code reader (label-free; privileged scoring)
- **Frame-exact .9977** (linear token readout .985-.988): about 5x fewer frames with a misread light. Per-light .9995-1.0.
- Event precision/recall from the reader's code are reported in `reader_eval.json`.

### 57015 4x6 code stage (partial; label-free, same recipe as 4x5, no tuning)
- **SFA + band-wise ICA selects exactly 24 binary slow components** at N = 64 and at N = 96 (24 lights). N = 64 is chosen by the rule.
  - 23/24 lights at >= .99, code purity .9998; events precision .987, recall .994.
- **Events from the ICA code:** 28 vocabulary types covering all 24 buttons (4 duplicate/noise types; purity .99998). The refined-code events, used downstream, follow in the same job.

### 57015 4x6 code stage (complete)
- Refined code: frame-exact .979 (ICA code .911).
- Events from the refined code: vocabulary exactly 24 types (noise patterns with counts <= 12 dropped). Type-to-button purity 1.0; 43,516 events vs 43,557 true presses; 99.9% matched; all 24 buttons covered.
- Evaluation-task frames render correctly on the MIG node (mean 126.7 vs 126.8 in the data).

| Job | Work | Resources | State |
|---|---|---|---|
| 57015 | 4x6 code stage | mig | COMPLETED |
| 57023 | CNN code reader 4x6 (pseudo-labels from the refined-code events) | mig, 6 CPU, 64 GB, 45 min | submitted |

### 57022 official protocol with the CNN reader, seed-1 episodes (fully learned, non-privileged)
- **100/100**, every task 20/20.
- Median steps 130 / 269 / 387 / 437 / 561. 99/100 episodes use exactly d* presses; one task-3 episode made a wrong press and replanning corrected it (+2 presses).
- Smallest |goal logit| per episode: median 10.2, minimum 6.6 (confident goal reads).
- **Caveat:** the CNN reader was introduced after inspecting the 57014 failures on these same seed-1 episodes. The clean held-out number is the fresh seed-2 run (57025).

### 57025 official protocol, fresh held-out seeds (seed 2), visual-puzzle-4x5 (fully learned, non-privileged)
**99/100 (99%), 95% CI [97, 100].**
- By task 1.00 / 1.00 / 1.00 / .95 / 1.00. Median steps 127 / 265 / 381 / 438 / 554.
- 98 of 99 successes use exactly d* presses.
- The single failure is again a misread goal frame (first plan 9 for d* 16). Its least-confident goal bit has |logit| 2.0 (typical minimum ~10), so the reader's own confidence flags it.
- **This is the clean held-out number for 4x5:** the configuration was fixed before these seeds were run. It is one training seed of every learned component.
- Reference: published pixel results on visual-puzzle-4x5 are <= 17% (HIQL, OGBench; 8 training seeds).

### 57023 CNN reader 4x6
Frame-exact .999 (min light .9998); events from its code: precision 1.0, recall .997.

### 57016 4x6 plan stage (offline; privileged scoring)
- **Event WM.** Exact 1.0.
- **Cost-to-go** (width 2048 + xor, 150k steps). Spearman .895.
  - Mean h by d*: 2.0, 2.9, 3.8, 4.7, 5.5, 6.3, 7.2, 7.9, 8.6, 9.3, 9.9, 10.4, 10.8, 11.2, 11.5, 12.0, 12.1, 12.4, 12.6, 12.9, 13.9 (d* 2-22): ordered, but compressed far away.
  - Event-sign accuracy 1.0 / 1.0 / .974 / .982 / .966 / .852 (d* 1-2 ... 13+).
- **Offline search between validation frames**, solved: 1.0 / 1.0 / 1.0 / .975 / .975 / .975. Length/d* ~1.00.
- **Official tasks offline.** Tasks 1-4 are planned optimally (6, 8, 12, 16 events). Task 5 (d* = 24, the farthest configuration) gets no plan within 50k expansions, because h is compressed at large distances.
- **Skill v3** (tau20, release 3): validation first-action MSE .234.

| Job | Work | Resources | State |
|---|---|---|---|
| 57016 | 4x6 plan stage | mig | COMPLETED |
| 57032 | 4x6 dev closed loop: learned high level and PRIVILEGED oracle plan, skill v3 + CNN reader, 6 episodes x 5 tasks, search budget 200k (greedy-h fallback when no plan) | mig, 16 CPU, 96 GB, 1 h | COMPLETED (learned 30/30; PRIVILEGED oracle plan 30/30) |
| 57033 | **Official-protocol eval 4x6**: fully learned method, 5 tasks x 20 episodes, env seed 1 (never used for 4x6) | mig, 16 CPU, 96 GB, 50 min | COMPLETED 20.5 min |

### 57032 4x6 dev closed loop (fully learned, non-privileged; 6 episodes x 5 tasks)
**30/30 (100%)**, every task 6/6; each episode uses exactly d* presses.
- Steps: ~158 / 217 / 343 / 441 / 682 for d* 6 / 8 / 12 / 16 / 24.
- **Task 5 (d* = 24).** The first search fails within 200k expansions in every episode. The greedy fallback (the event minimising h) then still lowers the true distance at every step, and search finds full plans once closer: 24 presses, ~680 steps.
- **Caveat:** the search budget (200k) was raised after the offline check on the evaluation-task configurations (job 57016). The dev seeds differ from the official run's seeds.
- Reference: published pixel results on visual-puzzle-4x6 are <= 15%; state-based SHARSA with 1B transitions gets 14%.

### 57033 official protocol, visual-puzzle-4x6 (fully learned, non-privileged; env seeds never used for 4x6)
**100/100** (every task 20/20; bootstrap CI [100, 100]: no failure to resample).
- Every episode uses exactly d* presses. Median steps 159 / 218 / 345 / 441 / 685 (max 696 of 1000) for d* 6 / 8 / 12 / 16 / 24.
- Task 5 (d* = 24): the first search never finds a plan within 200k expansions (20/20). The greedy-h steps still reduce the true distance each time, and full plans are found once closer.
- Recipe: identical to 4x5, with no 4x6-specific tuning except the search budget (raised after the offline check on the task configurations, 57016).
- Reference: published pixel results on visual-puzzle-4x6 are <= 15%; SHARSA (state input, 1B transitions) 14%. One training seed of every learned component (OGBench reports 8).

## Cube extension (user priority 2026-10-03: "Domain cube")
Goal: test generality beyond Lights Out, where event type = XOR pattern no longer applies.
- Cube state is continuous (positions).
- An event is "move cube k to place q".

Arena: visual-cube-triple (published pixel SOTA ~21%, HIQL).
- Task goals lie on a coarse grid with stacks up to 3.
- Success: every cube within 4 cm of its target.

| Job | Work | Resources | State |
|---|---|---|---|
| 57041 | Cache visual-cube-triple (1500 train episodes + val, with privileged qpos) + move inventory | main, 2 CPU, 24 GB, 1 h | COMPLETED |
| 57042 | Base encoder visual-cube-triple (LeWM recipe, 1M frames, 60k steps) | mig, 6 CPU, 40 GB, 1 h 05 min; after 57041 | COMPLETED |
| 57043 | Cube step 1: SFA -> ICA sources -> co-change object groups; scored by qpos (R^2 per coordinate, per-coordinate |corr|, within- vs across-cube co-change) | mig, 6 CPU, 48 GB, 30 min; after 57042 | COMPLETED |

Changes for cube data:
- `cache_data.py` and `common.Split` handle optional `button_states` / `qpos`.
- `train_wm.py` loads caches without `button_states`.

### 57041 cube-triple move inventory (privileged qpos; describes the data)
- 17,220 moves in 1500 episodes (11.5 per 1001-step episode).
- Carry duration median 42 steps (5-95%: 10-47); move starts median 88 steps apart.
- Final heights: 78% ground, 19% on one cube, 3% on two cubes.
- 3.3% of moves overlap with another cube's move (knocks, carried stacks).
- The evaluation tasks need 1-4 moves, so search depth is small. The difficulty is the object-centric abstraction and pick-and-place precision.

### 57043 cube step 1: the linear slow-feature recipe does NOT recover cube state (negative)
- **Linear decodability of true cube coordinates** from the N slowest SFA directions (fit on train, R^2 on val):
  - x: .38-.57; y: .60-.79; z (height): ~0 (.004-.04), at N = 16/32/64.
- **ICA sources do not track individual coordinates** (best |corr| per coordinate .07-.51; no source above .5 at N >= 32), so co-change grouping has nothing to group.
- **Why:** puzzle lights sit at fixed image locations, so their state is a linear function of patch tokens. Cubes move, so "where is cube k" is a nonlinear (spatial argmax-like) function of the image. The puzzle state-discovery recipe relies on fixed object locations.
- **Implication:** the cube extension needs a nonlinear, object-centric state discovery step, not a linear readout. Options under discussion with the user.

### Cube direction D (user choice 2026-10-04: "D first, then A")
- **Object state is PRIVILEGED** (qpos cube positions; simulator goal positions).
- **Everything else is learned:**
  - event WM f(s, k, q_xy) -> s' (the resulting height and stacking are learned);
  - affordance a(s) = which cube the play data moves (top cubes only);
  - BFS over moves (k, q), q in {goal xy of each cube} plus 6 data-derived free buffers;
  - pixel CNN pick-and-place skill conditioned on (k, q), trained on release + lift-off segments.
- Scripts: `cube_events.py`, `cube_planner.py`, `train_cube_skill.py`, `cube_closed_loop.py`, `slurm/cube_d.sh`.

| Job | Work | Resources | State |
|---|---|---|---|
| 57116 | Cube D training: move events, event WM + affordance (offline checks), pick-and-place skill (60k steps) | mig, 6 CPU, 64 GB, 1 h | COMPLETED |
| 57118 | Cube D closed loop: 6 episodes x 5 tasks, dev seed 0, final skill checkpoint; BFS vectorized and chunked (task4_cycle needs depth 4) | mig, 16 CPU, 96 GB, 1 h | COMPLETED (5.8 min) |
| 57120 | Cube A step 1: label-free object discovery (background, colour clusters, compact + slow objects, tracks); PRIVILEGED qpos diagnostic only | mig, 6 CPU, 64 GB, 40 min | COMPLETED (0.5 min of compute) |
| 57121 | Cube A prep v1: events with a 2-means move threshold + reader | mig, 6 CPU, 64 GB, 40 min | CANCELLED by me: thr 8.4 px, recall .71, so the reader labels were wrong |
| 57122 | Cube A prep v2: events with thr = one object width (2.72 px) + CNN reader (20k steps) | mig, 6 CPU, 64 GB, 40 min | COMPLETED |
| 57123 | Cube A train: pixel-state WM + affordance (offline checks), pixel-target skill (20k steps, best-val checkpoint) | mig, 6 CPU, 64 GB, 40 min | COMPLETED |
| 57124 | Cube A closed loop, fully label-free (reader perception of current and goal images), 6 episodes x 5 tasks, dev seed 0, skill best checkpoint | mig, 16 CPU, 96 GB, 1 h | COMPLETED (9.4 min) |
| 57128 | Cube A closed loop v2: same models; move end = play-data rest detector (colour track); support constraint (no target inside an occupied place); affordance >= 1/(2K) | mig, 16 CPU, 96 GB, 1 h | COMPLETED (10.7 min) |
| 57130 | Cube A coverage: events_cov (state (u, v, covered); 2-means on visible fraction per static interval), planner_cov (binary dim), reader_cov (attention-pooled coverage head) | mig, 6 CPU, 64 GB, 40 min | COMPLETED; superseded (bad threshold + loss weighting) |
| 57133 | Cube A coverage v2: covered = never visible in a static interval (>= 5 frames, no position-validity requirement); planner position loss in px^2 | mig, 6 CPU, 64 GB, 40 min | COMPLETED |
| 57134 | Cube A closed loop with coverage (events_cov2, reader_cov2, planner_cov2; skill from 57123), rest end rule, support constraints | mig, 16 CPU, 96 GB, 1 h | FAILED at start (occupied-place distance used the 3-dim state); no output written |
| 57135 | = 57134 after the fix | mig, 16 CPU, 96 GB, 1 h | FAILED (planner helper `key` shadowed by a loop variable at depth >= 2; stub test added); no output written |
| 57136 | = 57134 after both fixes (loop_reader_cov2) | mig, 16 CPU, 96 GB, 1 h | COMPLETED (10.1 min) |
| 57138 | = 57136 + `--belief` (WM-consistent belief update at move ends; loop_reader_cov2_belief) | mig, 16 CPU, 96 GB, 1 h | COMPLETED (10.3 min) |
| 57140 | Cube A skill_sup: skill conditioned also on the support object (hindsight: the object whose coverage goes 0 -> 1), 20k steps | mig, 6 CPU, 64 GB, 40 min | FAILED at step 0 (target normalisation used the 3-dim state bounds); no checkpoint |
| 57141 | = 57140 after the fix | mig, 6 CPU, 64 GB, 40 min | COMPLETED (support on 19.8% of train moves; val chunk 0.412 at 10k, 0.411 at 20k) |
| 57142 | Cube A closed loop: skill_sup + belief (matched control = 57138, same models except the skill) | mig, 16 CPU, 96 GB, 1 h | COMPLETED (12.4 min) |

**57116 results (offline):**
- **Moves** (train / val): 17,220 / 3,414; 11.5 per episode; knock 6.5% / 5.9%; stacked 21% / 22%.
- **Event WM, val:**
  - moved-cube error: median 5.6 mm, 99.2% within 4 cm;
  - stacked moves: 98.8% within 4 cm (n = 746);
  - other cubes: median error 3.8 mm.
- **Affordance (mean probability per cube):** 0.047 on covered cubes vs 0.372 on free cubes. So "move only top cubes" is learned, but covered cubes are not fully excluded at aff_min 0.02.
- **Skill:** val chunk MSE is best at step 10k (0.363) and rises to 0.438 at 60k, so it overfits. Only the final checkpoint exists. `train_cube_skill.py` now also saves `cube_skill_best.pt`, which takes effect on the next training run.

**57118 results (PRIVILEGED object state; dev seed 0): 26/30 = 86.7%.**
- By task: 1 6/6, 2 3/6, 3 5/6, 4 6/6, 5 6/6.
- First plan lengths match the optimum (1/3/3/4/3) except task 4 ep 5, which got a 3-move plan by exploiting covered-cube moves.
- All failures are low level:
  - task 2: grasp misses and pushes (eps 0, 2, 4);
  - task 3: unstack topples (ep 4).
- Logged executed moves: 64% within 4 cm, knocks 11%. The final move of a successful episode is not logged because the episode ends first.

### Cube direction A (label-free object state), 2026-10-04
- **57120 discovery:**
  - Background median; foreground threshold 4.4 (2-means, sep 5.4); 16 colour clusters.
  - Object clusters = compact + slow, giving **3 objects** (yellow, light blue, pink top faces).
  - Arm, gripper and shadow clusters are rejected: moving fraction .67–.89, spread 5–14 px.
  - Objects: moving fraction .013–.089, spread 1.1 px.
  - PRIVILEGED diagnostic: the 3 objects match 3 distinct cubes. A held-out quadratic map (u, v) → xy has median error 0.50–0.72 cm, p90 1.7–3.5 cm.
  - Visibility: the colour track sees each cube in only ~54% of frames (64% at rest, 2–8% while carried), because of arm shadow and gripper occlusion.
- **57121 (cancelled):** a 2-means move threshold on run-to-run displacement gave 8.4 px, recall .71, precision .996. It cut through the broad move distribution.
- **57122 events, thr = one object width = √6 × median object spread = 2.72 px** (≈ 3.3 cm; OGBench success radius 4 cm):
  - **train:**
    - 15,249 moves (10.2 per episode; qpos reference 11.5).
    - PRIVILEGED match to the qpos moves: recall .874, precision .987.
    - Recall by true xy distance: < 4 cm .35, 4–8 cm .81, 8–16 cm .90, ≥ 16 cm .91.
  - **val:** recall .879, precision .983.
  - Move end lags the qpos end by a median of 14 frames, because the cube becomes visible only after the gripper leaves.
  - Pseudo-labels are valid on ~70% of frames.
  - Adjacent-run (≤ 2 frames apart) displacement, i.e. pure track noise: p50 1.3, p90 4.0, p99 6.6 px.
- **57122 reader** (heatmap + soft-argmax CNN trained on rest pseudo-labels; val):
  - all valid frames: median error 0.15 px, p95 0.68, 97.4% within 1 px;
  - frames the colour track misses (shadow, gripper): median 0.27 px, p95 1.16, 93.8% within 1 px;
  - PRIVILEGED diagnostic: ground-plane quadratic map to qpos xy has median error 0.41–0.50 cm. The p95 of ~4 cm is mostly stacked cubes, which a ground-plane map cannot fit.
- **57123 pixel planner (val):**
  - moved-object error: median 0.35 px, 99.7% within tol (2.72 px);
  - other objects: median 0.24 px;
  - affordance top-1 accuracy .39 (play data picks among free cubes roughly uniformly);
  - 10,259 free placements available as buffer candidates.
- **57123 skill:** val chunk MSE 0.423 (5k) → 0.407 (10k) → 0.402 (20k). The best checkpoint is the final step; no overfitting at 20k.
- **57124 closed loop, fully label-free (dev seed 0): 14/30 = 46.7%.**
  - By task: 1 6/6, 2 5/6, 3 2/6, 4 0/6, 5 1/6.
  - First plan found 30/30; knocks 26%.
  - **Main failure: move ends detected mid-carry.**
    - A lifted cube projects within tol (2.2 px) of its target, and the arm pauses, which passes the reader stillness test (0.44 px/step).
    - So the planner switches cubes while the gripper still holds one: task 4 ep 0 at z = .26 m, task 5 ep 0 at z = .11 m.
  - **Second failure:** in task 4 every first plan has 3 moves, putting a cube's target inside the place another cube occupies.
    - That target is outside the play-data support (p1 of nearest-object distance at move targets = 1.85 px; stacking offset (0, −2.5) px).
    - The WM's prediction for it is impossible, and the affordance at aff_min 0.02 lets the covered cube move.
- **57128 closed loop v2 (dev seed 0): 14/30 = 46.7%.**
  - By task: 1 5/6, 2 3/6, 3 3/6, 4 3/6, 5 0/6. Knocks 21%.
  - Task 4 first plans are now the correct 4 moves (support constraint works).
  - New problems:
    - Affordance gate 1/(2K) = .167 excludes free cubes in some states, so task 1 eps 0 and 2 got 3-move detour plans.
    - Stacking is ambiguous in pixel space: in task 5 ep 1 the skill placed the cube on the ground next to the bottom cube. It was 2.1 px from the stacked goal, within tol, so the planner believed it was stacked.
  - Task 2 failures are skill grasp failures, as in D.
- **Coverage evidence (PRIVILEGED diagnostic, val):**
  - Covered cubes at rest are visible to the colour track 0.1–1.8% of the time, vs ~75% for free cubes on the ground.
  - Play moves of a covered cube: 1.6%.
  - So "covered" is an observable state bit and "covered ⇒ not moved" is a data-support rule. Implemented as the coverage dim, replacing the affordance gate.
- **57130 (coverage v1), superseded:**
  - 2-means threshold on visible fraction = .56. Agreement with qpos coverage only .87–.89; moved object "covered before" 7.4% (truth ~1.6%).
  - Labels valid on 19% of frames.
  - Planner position loss (/unit, ~0.03) was swamped by the coverage BCE: moved error 1.24 px (was 0.35).
  - Reader coverage: acc .89, recall on covered .65, false positives on free .04.
  - **Fix (PRIVILEGED check of visible fraction per static interval, val):** covered < 1% in 97% of intervals; free > 70% in 85%, never seen 1.3%. So the rule is covered = never visible.
- **57133 coverage v2:**
  - **Labels:** valid on all static frames (23.6%); covered 10.4%. PRIVILEGED agreement with qpos coverage: train .974–.976, val .972–.980.
  - **Planner (val):**
    - moved object .41 px median, 99.2% within tol; other objects .37 px;
    - coverage bit accuracy .92 (.70 where it changes, n = 1,238).
  - Moved object covered before a move: 0.6% (the data-support rule holds).
  - **Reader:**
    - positions unchanged (median .15 px);
    - coverage accuracy .976, recall on covered .865, false positives on free .011.
- **57136 closed loop with coverage (dev seed 0): 17/30 = 56.7%.**
  - By task: 1 6/6, 2 5/6, 3 4/6, 4 2/6, 5 0/6. Knocks 12%.
  - **Task 5:** the skill does stack (ep 0: middle cube at z = .06 on the bottom one). The reader then misreads the stack: the covered bottom cube is localised on top of the upper one and read as uncovered. The planner believes the stack is not built and detours through buffers.
  - **Task 4:** the plans are correct 4-move plans; failures are timeouts, i.e. the skill.
  - **Fix tried in 57138:** belief update.
    - Coverage comes from the WM prediction, overridden by the colour track seeing the top face.
    - Covered objects keep their believed position.
- **57138 with belief update (dev seed 0): 16/30 = 53.3%** (1 6/6, 2 6/6, 3 2/6, 4 2/6, 5 0/6). Same as 57136 within noise.
  - **Task 5 cause, now clear:** the skill places the cube on the ground just in front of the bottom cube (ep 0: 4.6 cm in x), while the reader reports 0.5 px from the stacked target.
  - In this camera view, "on top of B" and "on the ground ~4–5 cm in front of B" project to nearly the same pixel. The target pixel q alone is ambiguous for the skill.
  - The belief correctly shows the bottom cube as uncovered.
  - **Fix (57140):** condition the skill on the support object.
- **57142 support skill + belief: 13/30** (1 4/6, 2 3/6, 3 2/6, 4 3/6, 5 1/6; knocks 26%).

**Cube closed-loop runs side by side (dev seed 0, 6 episodes × 5 tasks; normal-approximation 95% CI):**

| Run | Success | By task (1–5) | Knocks per move |
|---|---|---|---|
| D privileged, 57118 | 26/30 [75, 99]% | 6 3 5 6 6 | .11 |
| A still rule, 57124 | 14/30 [29, 65]% | 6 5 2 0 1 | .26 |
| A rest rule + support constraint, 57128 | 14/30 [29, 65]% | 5 3 3 3 0 | .21 |
| A + coverage, 57136 | 17/30 [39, 74]% | 6 5 4 2 0 | .12 |
| A + belief, 57138 | 16/30 [35, 71]% | 6 6 2 2 0 | .20 |
| A + support skill, 57142 | 13/30 [26, 61]% | 4 3 2 3 1 | .26 |

- **No A variant is distinguishable from another at n = 30.**
  - The fixes are each justified by a specific failure seen in the logs: mid-carry move ends, out-of-support targets, stacking ambiguity.
  - Their effect on success is unmeasured at this sample size.
- **The A-vs-D gap is clear.** Which component causes it is NOT yet attributed.
  - Next: a PRIVILEGED attribution arm (A skill under the privileged high level, targets projected to pixels) vs (A high level with privileged move ends).

| 57146 | PRIVILEGED attribution (matched seeds): (1) privileged high level + A skill; (2) perfect perception projected to pixels + A planner_cov2 + A skill | mig, 16 CPU, 96 GB, 50 min | COMPLETED |

**57146 attribution results:**
- Projection fit (x, y, z) → pixels: held-out median error 0.19 px.
- **Arm 1, privileged high level + A skill: 17/30** (1 6/6, 2 6/6, 3 4/6, 4 1/6, 5 0/6). Reference: D with the D skill 26/30.
- **Arm 2, perfect perception + A planner + A skill: 20/30** (1 5/6, 2 5/6, 3 6/6, 4 4/6, 5 0/6). Reference: A label-free 13–17/30.
- **Reading:**
  - Swapping the skill (D → A) costs about 9/30.
  - Swapping perception (perfect → label-free) costs about 3–7/30, within noise.
  - Every arm with the A skill fails task 5 (stacking) 0/6.
- **Main bottleneck = the A skill** (placement precision, stacking). Next: improve the A skill.

### Cube attribution (PRIVILEGED arms, matched eval seeds = dev seed 0, 6 episodes × 5 tasks)

| Job | Arm | Success | By task |
|---|---|---|---|
| 57118 | D: privileged high level + D skill (metre targets) | **26/30** | 6 3 5 6 6 |
| 57146 | ARM1: privileged high level (D planner, true state) + A skill (pixel targets via fitted projection) | **17/30** | 6 6 4 1 0 |
| 57146 | ARM2: A pixel planner + A skill, perfect perception (true state projected to pixels + true coverage), privileged move ends | **20/30** | 5 5 6 4 0 |
| 57136 | A: fully label-free (reader, rest rule) | 17/30 | 6 5 4 2 0 |
| 57147 | ARM2 with the support-conditioned skill (skill_sup) | **16/30** | 4 4 3 3 2 |

- **Projection fit** (x, y, z) → (u, v): held-out median 0.19 px, p95 0.56 px.
- **Reading:**
  - Swapping the skill under the same privileged high level costs 9/30. The pixel-target skill is the main bottleneck.
  - Label-free perception plus the pixel planner costs ~3/30 more (20 vs 17, within noise).
  - Task 5 (stack) is 0/6 in every arm that uses the A skill, even with perfect perception: the pixel target is ambiguous (on top of B vs in front of B).
- **Caveat:** the ARM2 knock flags are mis-indexed (object vs cube index) in the log. Move-end timing and success are unaffected.
- **57147:** support conditioning gives task 5 2/6 (vs 0/6) but lower tasks 1–3, so 16/30 overall vs 20/30. Neither difference is significant at n = 6 per task.
- **Conclusion:** the pixel skill trained on label-free events is the bottleneck of direction A.
  - Candidate causes, unmeasured:
    - the ambiguous pixel target for stacking;
    - the 12% of moves missed by the label-free events, which merges two moves into one training segment with a wrong hindsight target;
    - segment ends that lag by ~14 frames.

### Cube skill fix (user 2026-10-04: "sửa skill cube trước", i.e. fix the cube skill first)
**Per-move comparison under the same privileged high level** (from existing logs):

| | D skill (57118) | A skill (ARM1 57146) |
|---|---|---|
| Placed within 4 cm | .64 | .43 |
| Median error | 1.7 cm | 5.3 cm |
| Ground placements within 4 cm | .61 | .42 |
| Knocks | .11 | .24 |
| Timeouts per episode | .37 | .73 |

The A skill is imprecise in general, not only when stacking.

**Factors between them:** segments (qpos vs label-free events), target space (metres vs pixels), training steps (60k vs 20k).

| Job | Skill | Resources | State |
|---|---|---|---|
| 57151 | skill_60k: label-free events_cov2, 60k steps (isolates training steps) | mig, 6 CPU, 64 GB, 40 min | COMPLETED (val chunk .412 @10k → .473 @60k) |
| 57152 | skill_qpos2px_60k: PRIVILEGED qpos segments with projected pixel targets, 60k steps (isolates segmentation vs target space) | mig, 6 CPU, 64 GB, 40 min | COMPLETED (val chunk .366 @10k → .455 @60k) |
| 57158 | Privileged high level + skill_60k, then + skill_qpos2px_60k (final checkpoints) | mig, 16 CPU, 96 GB, 45 min | CANCELLED by me seconds after it started (replaced by 57168); no output |
| 57159 | events_refined (reader arrival time ends a move; segments rebuilt; contaminated segments excluded) + skill_refined_60k | mig, 6 CPU, 64 GB, 45 min | COMPLETED |
| 57168 | Privileged high level + {skill_60k, skill_qpos2px_60k, skill_refined_60k} (final checkpoints) | mig, 16 CPU, 96 GB, 55 min | COMPLETED |
| 57173 | Label-free full loop (cov2, rest rule) with skill_refined_60k, episodes 0–11 per task (60) | mig, 16 CPU, 96 GB, 45 min | COMPLETED |
| 57174 | Same with the old skill (57123), episodes 6–11 (combined with 57136's episodes 0–5 = 60) | mig, 16 CPU, 96 GB, 30 min | COMPLETED |

### Puzzle overfitting check: random goals (user request 2026-10-04)
- **Why.** OGBench puzzle has only 5 fixed start/goal problems per size; "fresh seeds" change only the arm's start, and design choices were made while looking at those 5 problems.
- **What.** `closed_loop.py --random-goals N` runs N new problems.
  - Random start; goal = start XOR k random distinct presses; official problems excluded.
  - k cycles over 1..d_max: 4x5 d_max = 20 (5 problems per d*); 4x6 d_max = 24 (4–5 per d*).
  - Passed as OGBench `task_info`, so the goal frame is rendered as for the official tasks.
  - Frozen models and arguments of the official runs (57025 for 4x5; size_4x6 seed-1 for 4x6); env seed 3, never used.

| Job | Work | Resources | State |
|---|---|---|---|
| 57154 | Random-goal check, 100 problems each on 4x5 and 4x6 | mig, 16 CPU, 96 GB, 1.5 h | COMPLETED (13 + 18 min) |

**57154 results (fully learned, non-privileged):**
- **4x5: 100/100.** Every d* bin (1–5, 6–10, 11–15, 16–20, 25 problems each) at 100%. First plan optimal in 100/100. Steps p50 280, max 630.
- **4x6: 100/100.** All bins up to d* 21–24 at 100%. First plan optimal in 91/100. Steps p50 336, max 737.
- **Reading:** no sign of overfitting to the 5 official problems; the method solves random new start/goal pairs at every difficulty.
- Note: `optimal_presses_frac` in the summary is wrong (0.0) because the final press ends the episode before it is logged; executed presses = logged events + 1.

**Segment timing, label-free vs qpos** (PRIVILEGED check, 15,049 matched train moves; lag = label-free time − qpos time):

| | Lag (10/50/90 pct, frames) |
|---|---|
| Move end | 4 / 14 / 18 |
| Segment start | −33 / 13 / 17 |
| Move start (t_start) | −28 / −18 / −9 |

- Segment length 90th percentile: 172 frames vs 101 for qpos, i.e. merged segments where a move was missed.
- **Consequence:** frames right after a release are labelled only with the previous move's target, so the skill never learns to start the next move from there.
- **Label-free fix:** arrival = first frame from which the reader keeps the cube within 1 px of its final rest position. Val check: end lag 10/25/50/75/90 pct = −6 / −5 / −3 / −1 / 5 frames (vs 6 / 12 / 14 / 16 / 18 for the colour track).
- **57159 events_refined (train):**
  - Ends move 17 frames earlier (median).
  - PRIVILEGED lag vs qpos: end −6 / −5 / −3 / −1 / 5; segment start −31 / −5 / −3 / −1 / 5 frames (10/25/50/75/90 pct).
  - Contaminated segments (another object moved inside) 5.2%; excluded in total 6.4%.
  - Skill val chunk .341 @10k (vs .412 for label-free events, .366 for qpos events) → .419 @60k.
- **Peer job 57154** (`ew_rgoals`, puzzle random-goal overfitting check by another session) is running on one of the 2 GPU slots. It is not touched.

**57168: skills under the privileged high level** (dev seed 0, 30 episodes):

| Skill | Success | By task | Placed < 4 cm | Median error | Knocks |
|---|---|---|---|---|---|
| D: qpos segments, metre targets, 60k (57118) | 26/30 | 6 3 5 6 6 | .64 | 1.7 cm | .11 |
| A: colour segments, px, 20k (57146) | 17/30 | 6 6 4 1 0 | .43 | 5.3 cm | .24 |
| A: colour segments, px, 60k | 17/30 | 6 3 4 1 3 | .47 | 4.1 cm | .15 |
| PRIVILEGED qpos segments, px targets, 60k | **25/30** | 6 6 4 5 4 | .59 | 2.1 cm | .14 |
| Reader-refined segments, px, 60k | 21/30 | 6 5 5 5 0 | .54 | 3.8 cm | .05 |

- **Readings:**
  - The pixel target space is not the bottleneck: qpos segments with pixel targets ≈ D.
  - Training steps are not the bottleneck (17 = 17).
  - Segment quality is the bottleneck; reader refinement recovers part of it (17 → 21, not significant at n = 30).
- **Checks:**
  - Projection residual is unbiased at all heights (ground .19, level 2 .18, level 3 .24 px), so the ARM1 test is fair to label-free targets.
  - Stacking moves are not lost from the label-free events: recall .844 vs .882 ground; contamination .054 vs .050.
  - Undetected double moves of the target object: 0.8%.

**57173/57174: label-free full loop, old vs refined skill, matched over 60 seeds (12 per task, dev seed 0):**

| Skill | Success | 95% CI | By task (of 12) | Knocks per move |
|---|---|---|---|---|
| Old (colour segments, 20k; 57136 + 57174) | 36/60 = 60% | [48, 72] | 12 11 8 5 0 | .14 |
| Refined (reader-refined segments, 60k; 57173) | 40/60 = 67% | [55, 79] | 12 11 6 9 2 | .04 |

- Paired over the same seeds: refined-only successes 11, old-only 7, exact McNemar p = .48 (not significant).
- Knocks drop about 3.5×.
- Stacking stays weakest (2/12).
- **Label-free events by stack level:** recall level 1 .88, level 2 .86, level 3 .78.
  - Usable label-free level-3 segments: 371, vs 506 qpos (no knock).
  - Target error is the same at every level (median .20 / .20 / .24 px).

## Unified method (user 2026-10-04: one algorithm for all task families, aim at a top-tier venue, no deadline)
- **Prior art read:**
  - "Better Slots, Better Worlds" (2608.12078): SlotContrast + DINOv3 world model, CEM, on OGBench-Cube **single** only (LeWM 25-step protocol, random policy 48%).
  - Slot-MPC (2605.14937), SOLD, C-JEPA: object-centric WMs, none evaluated on cube-triple / puzzle / scene long-horizon.
  - Visual Robot Task Planning (Paxton 2018): image forward model over high-level actions + tree search, trained with action labels.
  - To check: DCRL (2609.02237, "best prior average 55 → 64 on the five hardest OGBench tasks"; whether state or pixel is unknown, the PDF table was not readable).
- **Candidate unified state:** the agent-free scene image; events = its changes; event parameters = agent contact locations at the start and end.

| Job | Work | Resources | State |
|---|---|---|---|
| 57176 | Probe v1: plain temporal median scene image, W 7/15/31, puzzle-4x5, cube-triple, puzzle-4x6 | mig, 6 CPU, 48 GB, 40 min | COMPLETED, NEGATIVE |
| 57177 | Download and cache visual-scene-play-v0 (HF mirror ryanhoangt/ogbench_data) | main, 2 CPU, 16 GB, 1 h | COMPLETED (cache has button_states + qpos) |
| 57178 | Probe v2: agent = fast colour clusters (slowness rule), masked median, change thresholds from the change distribution | mig, 6 CPU, 48 GB, 40 min | COMPLETED, partial |
| 57179 | Probe 2: deep spatial slow-feature map (temporal VICReg, L1 invariance), same hyperparameters on puzzle-4x5 and cube-triple; PRIVILEGED probes | mig, 6 CPU, 64 GB, 1 h | FAILED in eval (index bug) after training the puzzle map |
| 57180 | = 57179 after the fix (puzzle: eval only from the saved map; cube: train + eval) | mig, 6 CPU, 64 GB, 1 h | COMPLETED |
| 57184 | Keypoint probe (cube xy) + MLP probe (arm joints) on the cube slow map vs raw pixels | mig, 6 CPU, 64 GB, 30 min | COMPLETED |
| 57185 | Slow map + time-contrastive term (w_nce 1, k in [8, 32]), same hyperparameters on cube-triple, puzzle-4x5, scene; probes | mig, 6 CPU, 64 GB, 75 min | COMPLETED |
| 57191 | + same-episode hard negatives (\|d\| in [150, 600] frames), same three domains | mig, 6 CPU, 64 GB, 75 min | COMPLETED |

- **57176 (negative):** the plain temporal median keeps the slow arm base and its large shadow (visible in the PNG). The scene image changes every frame, so every episode becomes one 1000-frame "event" (puzzle recall .01, cube .015).
- **57178 (probe v2, partial):** event precision is high on puzzle (.93 on 4x5, .94 on 4x6), but recall is .57 / .58; cube recall is .43 / .42, precision .53.
  - The scene image still contains the robot base and arm shadow: dark, slowly varying, and classified as slow by the colour rule.
  - Footprints are 200–800 px vs ~45 px expected for a press.
  - Pixel-level rules cannot separate agent from scene cleanly, so the plan moves to a learned separation.
- **57180 slow map, same hyperparameters (C 16, K 8, w 25/25/1, 15k steps):**
  - **puzzle-4x5 (positive):**
    - linear probe on the 8×8-pooled map reads every light: min .987, mean .998;
    - events from map jumps: recall .953, precision .992 (vs the puzzle-specific SFA+ICA code: .989 per light);
    - no puzzle-specific rule.
  - **cube-triple (partial):**
    - pick / place recall .67 / .73, precision .66, 29 events per episode (true ≈ 23 picks + places);
    - the map is very sparse in the PCA view, apparently 1–2 cubes per frame;
    - the linear probe failed numerically (R² −20 on near-constant pooled features), replaced by the keypoint probe in 57184.
- **57184 probes on the cube slow map (VICReg only):**
  - Arm joints 1–4 from the map: R² < 0, i.e. not decodable (pixels: .93–.98). **The map is agent-free.**
  - Cube positions with the keypoint probe: median error 11.6 / 17.0 / 3.7 cm (pixels: 4.2 / 4.1 / 4.3 cm, so the probe itself is coarse). **The map holds 1 of the 3 cubes: incomplete.**
  - Fix: time-contrastive completeness term (57185).
- **57185 contrastive slow map (identical hyperparameters):**
  - **puzzle-4x5:** lights min .998 / mean .9999; events recall .84, precision .96.
  - **cube-triple:** keypoint probe median 8.5 / 3.8 / 12.0 cm (still ~1 cube); pick / place recall .62 / .70, precision .60; arm not decodable.
  - **scene:**
    - buttons .98 / .99;
    - events recall .89 but precision .31;
    - cube keypoint 6.5 cm;
    - arm partly decodable (MLP R² .13–.51), so the map is not fully agent-free.
  - **Diagnosis:** with negatives from other episodes only, one cube already separates scenes. Hence same-episode hard negatives (57191).
- **57191 hard negatives:**
  - **cube:** keypoint 7.5 / 3.6 / 10.9 cm (still ~1 cube); pick / place recall .75 / .76, precision .65; arm absent.
  - **puzzle:** lights .9997, but event recall fell to .42 (the map jitters in 60% of frames).
  - **scene:** cube 4.1 cm, buttons .991; event precision .29; arm leaks more (R² .35–.65).
- **Assessment after 3 objective variants:** each change helps one domain and hurts another, i.e. adjusting the loss per result is itself fitting to these datasets.
  - **Robust:** discrete states (lights, buttons) and agent-freeness on cube.
  - **Not solved:** completeness for small movable objects; clean event detection from map jumps; agent-freeness on scene.
  - **Next:** pre-register success criteria for the state component before trying another design.

**Pre-registered criteria for the unified state component** (set 2026-10-04 before seeing 57195; identical hyperparameters on puzzle-4x5, cube-triple, scene; PRIVILEGED probes):
1. **Discrete state:** linear probe ≥ .98 per light / button. Reference: the puzzle-specific code reached .989.
2. **Completeness:** keypoint-probe median error for every cube ≤ 1.2 × the same probe on raw pixels (~4 cm here; the probe itself is coarse).
3. **Agent-free:** arm-joint MLP R² ≤ .2 (raw pixels .93–.99).
4. **Events from map jumps:** recall ≥ .9 and precision ≥ .8 against the privileged events.
   - puzzle: light toggles;
   - cube: picks and places;
   - scene: any object joint moving or a button toggling (fixes the button-only reference used in 57185 / 57191).

| Job | Work | Resources | State |
|---|---|---|---|
| 57195 | Layered slow map (deep robust-PCA: masked reconstruction + slowness, lam from a 2-means split of per-pixel change), same three domains | mig, 6 CPU, 64 GB, 75 min | COMPLETED, FAILS criteria |
| 57199 | Layered v2: lam = centre of the changing-pixel cluster (cost of an unexplained change) + variance hinge as scale anchor | mig, 6 CPU, 64 GB, 75 min | COMPLETED, FAILS criteria, family stopped |
- **57195 layered v1 fails every criterion:**
  - mask covers 25–55% of pixels;
  - arm-joint R² .83–.99 (the arm is in the map);
  - cube keypoint 17–18 cm (no cube);
  - lights .72 min / .87 mean;
  - scene object events recall .56, precision .75.
  - Cause: lam (the 2-means split, 0.003–0.007) made masking cheaper than encoding, and nothing anchored Phi's scale.
  - 57199 is the last principled fix of this family before falling back to slots.
- **57199 layered v2 fails the criteria:**
  - **cube:** cubes masked; keypoint 9.9 / 16.5 / 13.0 cm; arm absent; pick / place recall .52 / .67.
  - **puzzle:** lights min .963; event recall .30.
  - **scene:** arm leaks (R² .47–.94); object event recall .19.
  - The PNG shows the decoder rendering the arm while the mask covers exactly the cubes.
  - **This failure is by design:** the mask cost scales with area, so small objects always go to the "sparse" part, the classic robust-PCA weakness. The family is stopped as pre-committed.

**State of the unified state component after 5 objective variants (57179–57199):**
- No learned slow map meets all four pre-registered criteria on all three domains.
- Robust: discrete states, and agent-freeness on cube (VICReg variants).
- Unsolved: completeness for 3–4 px movable objects, and clean events.
- **Candidates for a generic front end that handles small objects:**
  - a pretrained foundation segmenter (SAM 2) + mask tracking + slowness;
  - object slots on pretrained features (SlotContrast + DINOv3).
  - Both add pretrained perception, which must be disclosed when comparing with baselines.

**Front-end decision (2026-10-04):** the user left the choice to me ("pick the one you think is best for this research direction"; also "don't worry about usage quota", while the CLAUDE.md top-5 rule still applies).
- **Chosen:** object slots on frozen DINOv2-with-registers-S/14 (SlotContrast style). DINOv3 is gated (manual approval).
- **Reasons:**
  1. Same perception as the 2026 object-centric WMs (C-JEPA, Better Slots, Slot-MPC), so "flat CEM over slots" vs "our event abstraction + search" isolates the contribution.
  2. Pretrained features of the same kind as DINO-WM, so the comparison is fair on prior knowledge.
  3. Per-slot events are natural.
- **Fallback if small cubes do not bind:** SAM 2 + mask tracking.

| Job | Work | Resources | State |
|---|---|---|---|
| 57202 | Slots feasibility on cube-triple: K 8, T 4, stride 3, 10k steps, batch 32 clips; pre-registered criteria | mig, 6 CPU, 64 GB, 75 min | COMPLETED, slots do not bind cubes |
| 57203 | SAM 2.1-hiera-small point-grid segmentation probe (512 px, 24x24 grid) on cube / puzzle / scene | mig, 6 CPU, 32 GB, 30 min | COMPLETED |
- **57202 slots (10k steps):**
  - One slot covers almost the whole image (mean area 200 of 256 patches); no slot binds a cube (best centroid fit 15.6–17.2 cm).
  - Events: recall .96, precision .84. They pass because the dominant slot changes at any scene change, not per entity.
  - The arm MLP probe blew up (R² −10³) on near-constant features: probe bug, needs PCA / clipping.
  - Training was short (Better Slots: 100k steps × 128 clips), but a cube covers about 1 DINO patch at 224 px. Better Slots itself reports that slot metrics are weak for small objects on OGBench-Cube.
  - Per the pre-set fallback: probe SAM 2 (segments at 512 px; a cube is about 24–32 px).
- **57203 SAM 2 probe (24 frames per domain):**
  - **cube:** 17.5 segments per frame (median). PRIVILEGED: each cube centre lies in its own small segment 1.00 / .958 / .958 of frames (misses look like held cubes). Arm parts and shadow are separate segments.
  - **scene:** 32 segments; drawer handle, window, both button pads, cube and arm parts are separate segments.
  - **puzzle-4x5:** 43 segments, but granularity is inconsistent across frames (every light separate in some frames, the whole panel one segment in others).
  - **So:** small objects are separable by a generic segmenter at high resolution. Identity over time and a consistent granularity (lights) still need tracking.

**Front end, SAM 2 entities (2026-10-04):**
- Proposals: the finest valid mask per grid point (fixes panel-vs-light granularity), ≤ 2% of the image.
- Tracking: SAM 2 video propagation.
- Agent vs object entities: slowness rule.
- Pre-registered criteria as diagnostics.
- **Plan to the first closed loop:**
  1. Entity pseudo-labels on a data subset → fast reader.
  2. Plug into the existing cube downstream (event WM, BFS, (entity, target) skill), giving the first closed-loop number.
  3. Then puzzle (lights = entities whose appearance flips), then scene.

| Job | Work | Resources | State |
|---|---|---|---|
| 57233 | SAM 2 entity proposals + video tracking, 10 val episodes × 300 frames per domain (cube, puzzle-4x5, scene) | mig, 6 CPU, 48 GB, 1 h | CANCELLED while pending (mig node CPUs full) |
| 57243 | = 57233, submitted to both partitions (`--partition=mig,main --gres=gpu:1`) | mig or main, 6 CPU, 48 GB, 1 h | FAILED (SAM 2 video API needs start_frame_idx) |
| 57245 | = 57243 after the fix | main (worker-0), 6 CPU, 48 GB, 1 h | COMPLETED |
| 57251 | Front end v2 pilot: sparse SAM 2 frames (stride 5) + identity anchoring (location vs colour) -> u_events; cube, puzzle-4x5, scene; 10 train + 5 val episodes | main, 6 CPU, 64 GB, 2.5 h | CANCELLED by me: ~2.5 s per frame (image re-encoded per 64-point batch, masks post-processed on CPU at 512 px); rewritten as batched GPU proposals |
- **Usage rule changed by the user (2026-10-04):** stay below the 5th-ranked user's hours (no 50% margin).
- **Cluster at 10-04 evening:** main worker-0 has all 8 H100 allocated; the mig node has all 128 CPUs allocated; the other nodes are drained. Jobs now go to both partitions.
- **57245 SAM 2 video entities (10 val episodes × 300 frames):**
  - **cube:** centroid error 1.15 / 1.35 / 1.24 px when tracked, but tracked only .59–.67 of frames; events recall .75, precision .63.
  - **puzzle-4x5:** light state min .77 / mean .89 (some lights never an object entity); events recall .47, precision .76.
  - Video tracking loses small objects after occlusion and is slow (puzzle ~1.7 min per 300 frames).
  - Replaced by sparse per-frame segmentation + identity rules (front end v2).
- **Unified backend written** (`u_events`, `u_reader`, `u_wm`, `u_skill`, `u_closed_loop`, `slurm/u_pipeline.sh`):
  - entity-set transformer WM over (e, x) events; learned cost-to-go; one A*;
  - candidates = goal + rest-state prototypes;
  - data-support rules (covered, occupied place);
  - reader with an at-rest head (generic move-end detection);
  - event-conditioned skill.
| 57257 | Front end v2 pilot (batched GPU proposals) | main, 6 CPU, 64 GB, 2.5 h | FAILED OOM (4 frames × 1024 points in one decoder pass); fixed: cached embeddings + 256-point decoder chunks |
| 57258 | = 57257 after the fix (cube, puzzle-4x5, scene in sequence) | main (worker-0), 6 CPU, 64 GB, 2.5 h | FAILED in u_events on cube (a terminal last frame added an empty episode, index = n); fixed |
- **57258 cube front end (10 train / 5 val episodes, ~0.24 s per frame):**
  - 3 object identities, each matched to a different cube (PRIVILEGED fit: 1.27 / 2.09 / 1.62 cm).
  - But each is present in only .46–.57 of val frames.
  - The colour clusters suggest a lit/shaded split of every cube: bright yellow dropped as rare next to the kept dim yellow; dark red dropped next to red; dark blue classified as agent (moving .30).
  - Changes:
    - segments cached (`segments_{split}.npz`) so identity rules re-run on CPU (`slurm/u_ident.sh`);
    - cluster-link diagnostic (adjacent in the same frame vs alternating at the same place);
    - options `--colour-space chroma` (shading-invariant type clustering; state appearance stays RGB) and `--identity-mask union`.
| 57263–57265 | Front end v2 pilot again with segment caching + link diagnostic; one job per family (cube, puzzle-4x5, scene) | mig or main, 6 CPU / 64 GB (57264–65 lowered to 2 CPU / 24 GB while pending), 1 h | 57263 cube: segments OK, u_events FAILED (no agent mask near a change start -> acted entity None; fixed); 57264–65 PENDING |
- **57263 cube segments (contact sheet + flip statistic):**
  - SAM 2 segments all three cubes in almost every frame; the colour clustering splits each cube into 2–3 clusters (blue lit / shaded; red / dark red; yellow top face / side / tinted under the translucent purple gripper).
  - Colour invariance alone cannot fix it: the bluish-black gripper is as close in chromaticity to the blue cube as its shaded variant.
  - Colour-flip statistic P(B appears where A was | A left the frame), next sampled frame:
    - blue 0→3 .83, yellow 1→4 .87, red 2→8 .73, yellow 4→6 .67;
    - every other pair ≤ .15 (bimodal).
  - **Rule adopted (all families):**
    - clusters merge when the flip link is in the upper group of a bimodal 2-means split;
    - train and val segments labelled by the same nearest-centre rule (val used to drop off-centre segments: red .74 train → .57 val);
    - colour-anchored identities carry constant appearance (the colour is the identity) and use the union of their faces;
    - location-anchored identities keep appearance as state.
| 57268 | Identity stage only on the 57263 cube segments with the flip-merge rule -> u_events | main CPU, 2 CPU, 24 GB, 40 min | COMPLETED |
- **57268 cube, flip merge:**
  - Merged: blue {0, 3}, yellow {1, 4, 6}, red {2, 8} (D = 11.8).
  - Presence: .87 / .94 / .93 train; val visible .95–.97.
  - Events: precision 1.0, recall .46 (train) / .67 (val).
  - Two defects:
    - blue classified as agent (moving .27);
    - static grey bar (cluster 5) is an identity (1 event).
  - Diagnosis:
    - dark-blue cluster 3 also takes bluish arm pieces (23% of frames have two blue segments; 43% of blue "moves" have no cube moving);
    - continuity selection gives blue moving .276 → .113 (red .063, yellow .043, agents .62–.68);
    - excluding agent-touching segments costs coverage (.87 → .64); not used.
| 57270 | Identity stage with continuity selection (nearest previous position within thr_pos, else largest) + 2 cm move-recall diagnostic | main CPU, 2 CPU, 24 GB, 40 min | COMPLETED; continuity fixes blue but makes multi-segment arm types look static (agent test broken) |
| 57264 / 57265 | Puzzle / scene GPU pilots (segments cached; earlier identity code) | main, 2 CPU, 24 GB, 1 h | COMPLETED |
- **Puzzle 57264:**
  - 2 colour clusters: pale panel at fixed places (23 place identities, light state in colour) and dark arm.
  - Events: recall .91 / .91, precision 1.0 (meets criterion 4).
- **Scene 57265:** K 6; recall .53 / .55, precision .97.
| 57271 / 57272 | Identity stage, structural agent rule (multi-instance + not location-anchored = agent; single-instance = agent if moving in > 1/2 of frames) on cube / puzzle | main CPU | COMPLETED |
- **57271 cube:**
  - K 4 (3 cubes + static grey bar).
  - Events: recall .78 / .88, ≥ 2 cm moves .82 / .86, precision 1.0.
- **57272 puzzle:** K dropped to 17 (k-means with random init put two centres on a button); recall .85 / .87.
- **Cube miss analysis (`u_event_misses.py`):**
  - Several large moves are missed with rest labels before and after.
  - Example: 12.7 cm mostly in depth moved the cube only 4.9 px, below thr_pos = one object width (6.6 px), so the two rests were merged.
  - The same thr_pos was the planner's goal tolerance.
| 57273 | `u_pos_jitter.py` PRIVILEGED: resting-cube position noise + px per cm | main CPU | COMPLETED |
- **57273:**
  - Static jitter p50 ≈ 0, p90 0.3–0.7 px, p99 1.1–1.6 px; colour-support centroids are no better than SAM unions.
  - Image u ≈ 0.9 px/cm, v ≈ 0.6–0.7 px/cm: 4 cm ≈ 3.6 / 2.6 px; thr_pos 6.6 px ≈ 7–11 cm (too coarse).
  - Also found: frame-to-frame noise thresholds were computed on adjacent raw frames; entities exist only every 5th frame, so the only pairs were across episode boundaries. Fixed: consecutive sampled frames within an episode.
- **New position tolerance (all families):**
  - r_pos = 2-means noise/motion split of log |d pos| between consecutive sampled frames;
  - tol_pos = 2 r_pos (two rests within r_pos of one place differ by ≤ 2 r_pos) = change threshold and planner goal tolerance;
  - thr_pos stays for occupancy and prototypes.
- **Places for location anchoring:** deterministic density peeling (radius thr_pos / 2, place kept while occupied in ≥ 30% of sampled frames), replacing random k-means.
| 57274–57276 | Identity + events with tol_pos, place peeling, frame-pair fix; cube / puzzle / scene | main CPU, 2 CPU, 24 GB | COMPLETED |
- **57274–57276 (same configuration on all three families):**
  - **cube:**
    - tol_pos 1.69 px; K 4;
    - recall .89 / .95 (≥ 2 cm moves .91 / .93), precision .99 / 1.0;
    - meets criterion 4 on val.
  - **puzzle:**
    - K 26; recall .95 / .93, precision 1.0, but broken;
    - the arm parks at a home pose, so its type is "location-anchored" and there is no agent mask;
    - events are glued into giant groups (3.9 per episode).
  - **scene:**
    - K 19; recall .59 / .39, precision 1.0;
    - the red cube and the red drawer / window handles share one colour type, which the rule labels as agent.
  - Bug: the rest-to-rest appearance split counted exact zeros of constant-appearance identities (cube thr_app 8e-5); fixed (dd > 0 only).
| 57277 / 57278 | Full-scale segments + identity + events (150 train / 30 val episodes), cube / puzzle (`u_pipeline.sh` PARTS="tracks events") | mig, 2 CPU, 32 GB, 4.5 h | RUNNING |
| 57279 | `u_type_stats.py`: candidate motion / structure statistics per colour type, three families | main CPU | COMPLETED |
- **57279:**
  - No single type-level motion statistic separates agent from objects on all three families:
    - any-segment move: puzzle lights .69 (tiles flicker under the arm) vs arms .72–.96;
    - largest-segment move: lights .48, scene lines .48 vs agents .54–.74;
    - per-segment move: scene arm parts .21–.23 = scene lines .21 = blue cube .21.
  - Scene red type (cube + 2 handles) has no fixed place (at_places 0).
  - **Decision (stopping rule):** no more rules on colour types; next structural step = mask-IoU tracklets.
| 57285 | `u_tracklets_probe.py`: IoU-linked tracklets of the cached segments, per-type share in long tracklets and their motion; three families | main CPU | COMPLETED |
- **57285 tracklets:**
  - IoU ≥ .5 links only near-static segments, so tracklets are rest periods.
  - Median length: arm types 1–12, objects 12–196.
  - Scene arm/scenery mixture types (9–12) overlap the blue cube (12); not adopted.
| 57277 / 57278 | (see above) | | 57278 CANCELLED (puzzle identity in flux; would exceed the limit); 57277 CANCELLED (MIG 0.49 s per frame → train alone ~4.1 h) |
| 57286 | Cube val segments only (`--splits val`) | main H100, 27 min | COMPLETED |
| 57287 | `slurm/u_smoke.sh`: CPU smoke test reader → WM → skill → closed loop on pilot cube events (30 steps each) | main CPU, 13 min | COMPLETED; whole chain runs end-to-end (numbers meaningless) |
| 57288 | `u_pixchange_probe.py`: fraction of a type's pixels that change between sampled frames | main CPU | CANCELLED after cube: cubes .21–.44 vs arm .23–.38 (shadows, passing gripper); idea dropped |
| 57289–57291 | Identity: agent = largest segment moves > thr_pos in > 1/2 of frames (all types) | main CPU | COMPLETED |
- **57289–57291:**
  - The agent rule separates every known type on all three families (agents .54–.74, others ≤ .48).
  - Results: cube .89 / .95 (unchanged); puzzle K 21, recall .74 / .75; scene K 7, recall .45 / .53.
  - Puzzle dropped because place identities' rest runs fragment under tol_pos (tile centroids shift when the arm covers them part-way; the dot is a small part of the tile).
- **Rule change:** a place identity has constant position and its state is the mean colour of its place disc (unobserved while an agent segment covers it). This mirrors colour identities (constant appearance, position = state) and replaces the nearest-segment lookup.
| 57292 / 57293 | Cube train segments in two shards (episodes 0–74 / 75–149), `--train-ep-start`, merged by `merge_segments.py` | main H100 / mig | RUNNING |
| 57294 / 57297 / 57296 | Identity with place appearance; cube / puzzle / scene | main CPU | COMPLETED |
- **57294–57297, place appearance:**
  - cube .86 / .93 (≥ 2 cm .90 / .93), precision 1.0; puzzle .84 / .83; scene .59 / .65.
  - Puzzle button probe (`u_button_probe.py`): 18 of 20 buttons ≥ .98; the two back-row buttons are unobserved most of the time (any agent pixel in the disc → unobserved).
| 57298–57303 | Probes + identity with "unobserved only when most of the disc is covered; read the uncovered part" | main CPU | COMPLETED |
- **Results:**
  - Puzzle 19 / 20 buttons ≥ .90; events .87 / .83.
  - Scene per object (`u_scene_recall.py`): cube .55, drawer .56, window .36, buttons .75 / .67.
  - Same-colour instances: red type has ≥ 2 segments in 59% of frames, grey type in 41%; cube types ≤ 20%.
- **Decision:** prioritise cube + puzzle; scene instance identity later.
| 57304–57307 | `u_acted_probe.py`: which acted-entity rule picks the pressed button (privileged pressed button = centre of the toggled cross) | main CPU | COMPLETED |
- **Accuracy (21 events):** nearest arm mask .86, most covered .67, arm tip .48, most central among the changed identities 1.0.
- **Adopted:** most central, ties broken by agent proximity (decides only when ≥ 3 identities change).
- Puzzle front-row lights still never acted. Their change is below thr_app because the place disc is the whole tile, which dilutes the dot.
  - Per-identity thresholds (57308–57309): worse (.54 / .49), reverted.
  - Object-sized discs (57311–57313): no change.
  - Smallest-segment place (57316–57318): only one front-row button has a dot segment.
| 57321–57326 | **Place pixels weighted by temporal variance over train frames** (state shows where pixels change) + probes | main CPU | COMPLETED |
- **57321–57326:**
  - **puzzle:**
    - events .91 / .91, precision 1.0 (meets criterion 4);
    - 19 / 20 buttons ≥ .98 (criterion 1 except back-left button 0: .72, the arm base region);
    - acted rule 44 / 44 correct.
  - **cube:** .87 / .93, ≥ 2 cm .90 / .93, precision 1.0.
  - **scene:** .66 / .60 (cube .55, drawer .63, window .36, buttons .75 / .67), limited by same-colour instances.
| 57327 | Puzzle train segments, shard A (episodes 0–74) | mig, 2.75 h | RUNNING (started 18:10 UTC) |
- **Backend changes before the first full cube run (ported from the cube-specific pipeline that reached 40/60):**
  - **Best skill checkpoint:** validation every 2k steps, `u_skill_best.pt` (the cube skill overfit after ~10–20k steps); the closed loop uses it.
  - **`u_refine.py`, event-timing refinement with the trained reader:**
    - arrival = first frame from which the reader keeps the acted identity at its after-state within r_pos / r_app;
    - segments rebuilt from the refined ends; segments where another identity changed are flagged and excluded from skill training;
    - segment timing was the bottleneck of the cube-specific pipeline.
  - **`slurm/u_train3.sh`:** one GPU; reader and WM concurrently, then refine, then skill on refined segments. The loop reads `events_ref`.
  - Support-object conditioning not ported (13/30 vs 14/30 earlier).
| 57328 | CPU smoke test of `u_refine.py` (pilot cube events, untrained smoke reader) | main CPU | COMPLETED; runs end-to-end |
| 57292 / 57293 | (cube train shards) | | COMPLETED 18:11 / 19:06 UTC (H100 66 min; MIG 2 h) |
| 57329 | `u_shape_probe.py`: mask solidity per colour type | main CPU | COMPLETED; solidity is bimodal even for single cubes (top vs side face, D 2.0–2.3), so it cannot separate same-colour objects without splitting cubes; rejected |
- **Scene, deferred:** the red type = cube (diffuse cloud) + drawer handle (horizontal line) + window handle (vertical line) + button dots (fixed points); the drawer handle changes colour with its lock state. Needs instance identity (tracking / learned); cube and puzzle closed loops first.
| 57330 | Cache cube-double / cube-single (held-out cube families) | main CPU | RUNNING |
| 57338 | Cube full identity + events on merged segments (150 train / 30 val episodes; 410k train segments) | main CPU, 48 GB | RUNNING |
- **57338 cube at full scale (150 / 30 episodes):**
  - Worse than the pilot: K 3, one identity is not a cube (15 cm error, visible .999), blue missing; recall .78 / .76, precision .98.
  - Cause: with 15× more data the colour-flip merge linked the blue cube's dark variant to cluster 8 (82, 88, 114; present .993).
  - Cluster 8 is the table surface / background in many small pieces (plus bar and arm parts). Lifting a cube reveals the table under it: uncovering, not a colour variant.
| 57339–57341 | Free clusters only (fixed-place clusters never merge with movers) | main CPU | COMPLETED; no change at full scale: cluster 8 is not fixed-place (spread everywhere) |
| 57342–57344 | **Variant merging only between free, single-instance clusters** (median one segment per frame where present); full cube + cube / puzzle pilots | main CPU | RUNNING |
- **57342 full cube (free single-instance merge):**
  - K 4: blue 1.23 cm (visible .50), yellow 1.79 cm (.93), red 1.92 cm (.93), plus the background cluster as a 4th identity (acted in 196 train events, wrong labels).
  - Recall .88 / .86 (≥ 2 cm .89 / .88), precision .99. Pilots unchanged.
  - Blue's deepest shade is nearest the background colour (cluster 8) when blue is unseen.
| 57347–57350 | Colour radius from IoU-matched segment pairs (noise / change split) + single-instance colour identities | main CPU | radius collapses (cube .008, puzzle .025; deterministic rendering): thousands of clusters, full cube OOM, pilots worse. **Reverted** to 3 × within-segment deviation |
| 57355 / 57356 | Full cube + puzzle pilot: reverted radius, colour identities only for single-instance types (the background stops being an identity) | main CPU | COMPLETED |
- **57355 cube full-scale front end, used for the first closed loop:**
  - K 3 = the three cubes (1.23 / 1.79 / 1.92 cm; blue visible .50, others .93).
  - Recall .84 / .83 (≥ 2 cm .84 / .84), precision 1.0. Puzzle pilot unchanged (.91 / .91).
| 57357 | **Cube unified backend training** (`u_train3.sh`: reader ∥ WM + cost-to-go, then u_refine, then skill on refined segments; skill --train-frames 160000) on the 57355 events | mig, 1 GPU, 4 h | COMPLETED (82 min) |
- **57357:**
  - Reader: val position error median 0.23 px, 97.5% within 1 px; rest accuracy .90.
  - Refine: ends move a median 2 frames earlier (10th percentile −21); 12–16% of segments contaminated and excluded.
  - WM: acted entity always within tolerance, unchanged .97, knock side effects 0 / 16; offline plans found .97 with data length.
  - **Skill overfits at once:** best val chunk at step 2k (.428 → .479 at 60k). Only 150 episodes (~1050 usable events).
| 57364 | **FIRST UNIFIED CLOSED LOOP**, visual-cube-triple-v0, 5 official tasks × 6 episodes, dev seed 0; SAM 2 front end (150 episodes) → reader + entity event WM + A* + event skill (step-2k checkpoint); no privileged input | mig, 8 CPU, 6 workers, 20 min | COMPLETED |
- **Result: 5/30 (17%).**
  - Task 1: 5/6; tasks 2–5: 0/6 each.
  - First plan found 30/30 with the right number of moves (1 / 3 / 3 / 3–4 / 3–5).
  - Failures: skill timeouts (1–3 moves per episode not completed within 250 steps).
  - References: OGBench best published pixel result on cube-triple 21%; the cube-specific label-free pipeline 40/60.
  - Bottleneck: skill data. Fix = self-training (the reader labels all 1000 train episodes).
- **Self-training round (new scripts):**
  - `u_reader.py --agent-head`: agent-mask head supervised by front-end agent segments.
  - `u_reader_entities.py`: the reader writes the u_events entity table for N episodes (observed = rest head > 0 on sampled frames; agent = agent head).
  - `u_events.py --thresholds-from`: keeps round-1 change thresholds.
  - `slurm/u_self.sh`: reader+agent → entities → events → (WM ∥ refine → skill).
| 57365 | Cube self-training round, NTRAIN = 1000, skill 40k steps | mig, 1 GPU, 6 h | RUNNING |
| 57359 | Puzzle train shard B (episodes 75–149) + val segments | mig or main, 4 h | RUNNING / PENDING |
| 57330 | (cube-double / cube-single cache) | | COMPLETED (1000 train + 100 val episodes each) |
| 57360–57363 | Colour radius 2 × within-segment deviation (`--merge-mult 2`), full cube + 3 pilots | main CPU | COMPLETED; not adopted |
- **57360–57363:**
  - Full cube: three cubes visible .85–.91 (blue .50 → .85), recall .91, but five junk identities (single-instance arm / background fragments, 12–16 cm) and blue's big_move .45 is near the agent threshold.
  - Puzzle pilot .91 → .85; scene .51.
  - Kept 3×.
- **Closed-loop failure inspection (57364):**
  - Many "completed" moves end with the cube 17–45 px from target at v ≈ 33–35 (targets v ≈ 44): the move ended with the cube lifted in the gripper.
  - Some events end after 5 steps.
  - The reader's at-rest head misses 24% of in-transit frames, because front-end rests include the arm pausing with a cube held (≥ 25 steps).
| 57366–57368 | **Contact-free rest observations** (unobserved while an agent pixel is within half an object width) | main CPU | COMPLETED; adopted |
- **57366–57368:**
  - Full cube: 974 events, recall .79 / .76 (from .84 / .83), precision 1.0.
  - Puzzle pilot .90 / .89 (unchanged).
  - Labels now mean "at rest, free of the agent", which the closed-loop move end relies on.
| 57365 | (self-training on the pre-contact-free events) | | CANCELLED after ~10 min (labels superseded) |
| 57369 | Cube self-training round on the contact-free events (57366), NTRAIN = 1000, skill 40k steps | mig, 1 GPU, 6 h | RUNNING |
- `u_closed_loop.py`: with an agent-head reader, "at rest" also requires no agent pixel within half an object width (same definition as the labels).
| 57369 | (above) | | CANCELLED at reader step ~4k: random reads of the memory-mapped observations over contended network storage (2000 steps in 44 min vs 6.6 min before) |
- `u_reader.py`: reads the labelled prefix of the observations into RAM once.
| 57371 | Cube self-training round (57366 contact-free events), reader with RAM prefix | mig, 1 GPU, 6 h | COMPLETED (2 h 17 min) |
- **57371 cube self-training:**
  - Reader + agent head: 0.23 px; in-transit recall of the at-rest head .73.
  - Reader entities on 1000 episodes: observed at rest .71 of sampled frames.
  - **Events: 5865 train (6× the SAM 2 round)**, privileged recall .81 / .81, precision 1.0.
  - Refine: median end shift −9 frames; 29% contaminated, 33% excluded (~3900 usable skill segments vs ~1050).
  - Skill best val .388 at step 4k (round 1: .428 at 2k), then rises to .455.
  - WM: acted within tol 1.0, unchanged within tol .84 (round 1 .97), offline plans .97.
| 57394 | Closed loop v2: cube self-training models (reader+agent head with contact-free rest test, WM, skill best) | mig, 8 CPU, 6 workers | RUNNING (auto-launched 02:12 UTC) |
| 57370 | Puzzle full identity + events (merged shards, 1.31 M train segments) | main CPU, 1 min | COMPLETED |
- **57370 puzzle full:**
  - K 21; recall .91 / .91, precision 1.0; 2665 train events.
  - Probes: 18 / 20 buttons ≥ .98 (57372); acted rule 97.9% on 561 scored events (57373).
  - But 8 identities never acted, although presses per button are uniform (194–259, 57374).
- **57375 (`u_change_count.py`):**
  - thr_app = .376 at full scale; five lights with on/off signal .29–.37 never register a change.
  - Steady pixel-weighted readings rarely split a rest, so the "no change" mode vanished from the rest-to-rest differences and the 2-means split fell between weak and strong lights.
  - **Rule change:** thr_app = 2 r_app (as tol_pos = 2 r_pos); the rest-to-rest split is removed.
| 57376–57379 | thr_app = 2 r_app: full puzzle + three pilots | main CPU | RUNNING |
- **57376–57379 (thr_app = 2 r_app):**
  - **full puzzle:** recall .93 / .92, precision 1.0; buttons 17 / 13 / 12 now acted (106 / 222 / 285 train events). Still never acted: back-row buttons 2–3 (visible .26–.44, the hovering arm hides the back row) and two non-button places.
  - Pilots: cube .81 / .92 (from .87 / .93), puzzle .93 / .90, **scene .30 / .30** (scene r_app .006 → threshold .012 counts shadow-level noise; scene deferred).
  - Adopted.
| 57380 | Puzzle self-training round (57376 events), NTRAIN = 1000, skill 40k | mig, 1 GPU, 6 h | RUNNING |
| 57394 | (closed loop v2, self-training models) | | COMPLETED: **5/30** (task 1 5/6, tasks 2–5 0/6; first plan .90) |
- **57397 inspection of v2 (`u_loop_inspect.py`):**
  - Median final error of "completed" moves 16 px; only 22% within 4 px.
  - Fake completions: the gripper hovering over a static cube shifts the reader position by more than tol_pos (1.7 px), so change + rest ends the move with the cube unmoved.
  - Mid-air ends after 5 steps: the agent colour types miss the translucent purple fingers, so the contact test misses a held cube.
  - The cube-specific pipeline (40/60) ended a move only at rest more than one object width from the start.
- **Rule change in `u_closed_loop.py`:** a move ends at rest (free of the agent) when the acted identity is > thr_pos (one object width) from the event start (place identity: appearance changed) or within tol_pos of its target; knocks likewise at > thr_pos.
| 57402 | Closed loop v3: same self-training models, object-width move-end rule | mig, 8 CPU | RUNNING |
| 57402 | (closed loop v3, object-width end rule) | | COMPLETED: **4/30** (task 1 4/6, tasks 2–5 0/6); fake completions gone, now 2–3 skill timeouts per multi-move episode |
| 57408 | **Attribution:** unified loop (self-trained reader + WM + A*) with the cube-specific skill (skill_refined_60k final, 40/60 in its own loop; identities → its objects by chromaticity) | mig, 8 CPU | RUNNING |
| 57380 | (puzzle self-training) | | CANCELLED at reader step ~6k: NpzFile decompressed the whole label arrays on every batch (K = 21) |
- `u_reader.py`, `u_wm.py`: npz contents decompressed once (`dict(np.load(...))`). This was also the real cause of the slow cube reader in 57369.
| 57410 | Puzzle self-training round (restart) | mig, 1 GPU, 6 h | RUNNING |
- **57408 attribution: 16/30 (53%)** with the cube-specific skill in the unified loop (task 1 6/6, 2 5/6, 3 2/6, 4 2/6, 5 1/6) vs 4/30 with the unified skill.
  - The unified reader, WM, A* and move-end rule support ~50%; **the unified skill is the bottleneck.**
  - Differences: (1) checkpoint: 40/60 used the final 60k, the unified loop the best-val one at step 4k; (2) contamination filter at tol_pos (29% excluded vs 5%); (3) start-state conditioning.
| 57411 | Closed loop v4: unified skill final checkpoint (40k steps) instead of best-val (4k) | mig, 8 CPU | SUBMITTED |
