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

## 2026-10-05 live status and quota audit (14:00 ICT)

- Verified with both squeue and sacct: unified cube-triple evaluation 57411 COMPLETED, 8/30 success (task counts 6/6, 1/6, 1/6, 0/6, 0/6), final 40k skill checkpoint. This is a development result, not a general-method win.
- Puzzle unified self-training 57419 COMPLETED; u_self.sh does not launch closed-loop evaluation. WM validation: side_effect_within_tol 0.0826, offline_plan_found 0.03185. No closed-loop result is implied.
- Family mapping from the submission record: 57420 cube-double, 57421 puzzle-4x6, 57422 puzzle-3x3, 57423 puzzle-4x4, 57424 cube-single.
- Monthly sreport snapshot: nhatnc129 GPU 59 h, CPU 385 h, memory 4,348,122 MB-h. Fifth-ranked users: GPU 94 h, CPU 904 h, memory 7,094,397 MB-h. The respective 90% ceilings are 84.6 h, 813.6 h, 6,384,957.3 MB-h.
- Five 14-hour GPU jobs exceeded the planned-budget rule. On 2026-10-05, scancel 57422/57423/57424 before start; reduce TimeLimit of running 57420/57421 to 10 hours each. Both squeue and sacct confirm the cancellations and new limits. Existing run directories and checkpoints are preserved. This is a resource-budget correction, not a negative scientific result on those families.
- 57420/57421 remain RUNNING in SAM 2 segmentation. At 13:59 ICT, train progress was 18,600/30,150 and 5,800/30,150, respectively; validation segmentation and backend training still follow. Rough first closed-loop ETA, conditional on unchanged throughput and successful stages: cube-double 18:00-20:00 ICT; puzzle-4x6 20:00-22:00 ICT on 2026-10-05. These are estimates, not guaranteed completion or paper-ready comparisons.

## 2026-10-05 unified demo export

- Job 57497: three development roots from unified cube-triple run 57411 (task 1 success, task 2 failure, task 5 failure); read-only observation/video and planner-trace instrumentation around u_closed_loop.run. CPU replay with the same checkpoints/settings; possible numerical differences are disclosed. No new training or policy fix.
- Resources: main, 4 CPU, 16 GB RAM, no GPU, 30-minute limit. Source: docs/demos/unified_cube_20261005/{record_demo.py,demo.sbatch}; frozen source and checkpoint hashes saved in /mnt/data/nhatnc129/jepa/event_wm/demo_unified_cube_57497. Status: SUBMITTED; final artifacts to be verified before claiming completion.
## Demo 57497 final verification (2026-10-05 14:24 ICT)

- Both squeue and sacct verified: COMPLETED, elapsed 00:10:17, exit 0:0. Output /mnt/data/nhatnc129/jepa/event_wm/demo_unified_cube_57497 has three MP4s, initial/final panels, per-episode diagnostic traces, checkpoint hashes, frozen sources and reference_inspection.txt. Local viewing page: docs/demos/unified_cube_20261005/index.html.
- CPU replay: task1 ep0 success 66 steps; task2 ep0 success 472 steps / 4 replans, despite the reference GPU root failing; task5 ep0 failure 1000 steps / 5 replans / 1 timeout. All six task5 planner calls failed to find a complete plan at approximately 20k expansions, then used partial plans. Selected CPU replays do not revise benchmark success 8/30.
- Reference GPU inspection: 57 recorded event endings; reader-space target error p25/median/p75 1.9/3.5/22.2 px; 29.8% <=2px, 52.6% <=4px. Some endings are knock-triggered replans, not successful placements. These are diagnostics, not physical ground-truth errors or isolated causes.


## SAM2/event demo 57538 (2026-10-05)

- Authorized offline visual export: saved SAM2 proposals and entity/event labels for cube-triple and puzzle-4x5. main, 2 CPU, 8 GB, no GPU, 10-minute limit. Both squeue and sacct verified COMPLETED, 00:00:06, exit 0:0. No new inference, physics, training or benchmark.
- Output /mnt/data/nhatnc129/jepa/event_wm/demo_sam2_events_57538: four MP4s, before/during/after storyboards, after panels and exact label metadata. Original cube event index0 101-264; reader clean index1 296-333 (coarse349); reader excluded index0 101-244 (coarse264); puzzle index65, episode9, 9850-9938, changed IDs3/5/7/12, acted5.
- Local gallery docs/demos/sam2_events_20261005/index.html. Mask colours distinguish proposals within each frame, not tracked identity; inferred entity colours use IDs. Clips are recorded play-data illustrations, not learned-policy rollouts.


## SAM2/event demo 57539 presentation correction

- Same four saved-data examples as 57538, re-export with compact RGB strings and explicit 2 sampled frames/sec (stride5), without assuming simulation playback speed. Separate output /mnt/data/nhatnc129/jepa/event_wm/demo_sam2_events_57539; first export preserved. main, 2 CPU, 8 GB, no GPU, 10-minute limit. Both squeue and sacct verified COMPLETED, 00:00:04, exit 0:0. Local gallery now uses job57539.

## 2026-10-05 component audit of the unified pipeline (user request: which parts are weak, what to fix for puzzle/cube/scene)

- Job 57565 (ew_diagref): main CPU, 2 CPU, 32 GB, 30-minute limit; COMPLETED in 00:00:08 per sacct (squeue empty). PRIVILEGED diagnostic reference tables only (no training, no simulator): per-frame cube xyz and motion intervals (cube-triple), per-press table and packed button states (puzzle-4x5), scene val qpos/buttons, compact val entity tables (front end vs self-training) and five raw val episodes. Source: docs/diag_20261005/{build_refs.py,refs.sbatch}. Output: /mnt/data/nhatnc129/jepa/event_wm/diag_20261005/refs_57565 (refs_report.json).
- Reference counts (PRIVILEGED): puzzle-4x5 play data has 29.6 presses per episode, every toggle an exact plus-shaped cross, presses uniform over the 20 buttons; cube-triple has ~11.4 moves >= 2 cm per episode. For comparison the unified events have 6.05 (puzzle self-training) / 15.8 (puzzle round 1) / 5.9 (cube self-training) events per episode, so the overlap-based privileged_recall in report.json overstates event quality. Offline analysis continues locally on these tables; no cluster job beyond 57565.

## 2026-10-05 report demo pack

- User authorized preparing the agreed event, planning, closed-loop success and failure demos. CPU exporter 57577: main, 2 CPU, 16 GB, 20-min limit; COMPLETED 56 s, exit 0. Inputs: cached validation events and CPU replay 57497; no new training or privileged policy input.
- Follow-ups 57578 (3 s), 57579 (9 s), 57580 (7 s), each main CPU 2 CPU/16 GB/5 min, no GPU, all COMPLETED exit 0 verified with both squeue and sacct. Initial covered bits now match Model.plan. Initial reset images for task 3/4 in intermediate exports were corrupt: LP_NUM_THREADS=1 fixed the first context, but OGBench Env.close did not release the renderer; creating a replacement env before the old renderer was destroyed corrupted the next context. Final export reuses one renderer across task resets and validates dtype/shape/mean/black fraction. Corrected task 3/4 image pairs visually inspected. Intermediate outputs preserved, excluded from the report.
- Final assets: /mnt/data/nhatnc129/jepa/event_wm/demo_report_final_57580, local docs/demos/report_20261005/job57580. Three event examples with before/during/after sheets and short clips; WM prediction steps for all five cube tasks; annotated existing CPU rollout task1 success, task5 failure, task2 supplementary (CPU success differs from GPU reference failure). Tasks3/4 are planning only; no new execution claims.
- Report data keeps original GPU reference 8/30 (6,1,1,0,0 successes/6), checkpoint hashes, event provenance, limitations and reader-space diagnostics. SAM2-assisted supervision, data mismatch and one-seed development status disclosed. Root docs/diag_20261005 and peer dirty files untouched.

## 2026-10-05 unified status update (18:00 ICT)

- Verified both squeue and sacct: 57420 COMPLETED, elapsed 06:30:26, exit 0:0; 57421 RUNNING, elapsed 04:48:26 at query.
- Newly completed unified cube-double closed loop: 0/30 success, every task 0/6; first_plan_found 0.5. Raw result saved locally in docs/unified_status_20261005/cube_double_loop_57420.json.
- Cube-double refinement excluded 96.38% of 2019 train events and 97.34% of 188 validation events, leaving 73/5 skill segments. The front end discovered K=4 for two physical cubes; saved privileged diagnostics show two extra identities with approximately 14.4 cm median position error. Their physical identity requires visual audit; do not assert a specific background/arm assignment yet.
- These are new family-transfer failures; no fix, rerun, or baseline superiority is implied. Existing cube-triple 8/30 and puzzle-4x5 offline results remain separate.

## 2026-10-05 event-extraction diagnosis of the unified pipeline (19:30 ICT)

- 57421 (puzzle-4x6 family run) CANCELLED on the user's request at 06:17:32 elapsed (cost-to-go step ~25k of 60k). Reason: its self-training data were already broken: 2.83 events per val episode against 29.1 true presses, 98.3% of skill segments excluded by refine. Verified with squeue (no jobs) and sacct (CANCELLED). Kept: front/ (SAM 2 segments + identities), events/, self1000/{events, events_ref, front, reader, skill}. The WM / cost-to-go was not saved.
- 57586 (ew_diagframes): main CPU, 2 CPU, 8 GB, 10 min; COMPLETED in 00:00:01 (squeue empty, sacct COMPLETED 0:0). PRIVILEGED export of raw val frames (cube-double ep 0-1, puzzle-4x6 ep 0, puzzle-4x5 ep 0) for the visual identity audit. Source docs/diag_events_20261005/{extract_frames.py,extract.sbatch}; output /mnt/data/nhatnc129/jepa/event_wm/diag_20261005/frames_57586.
- **Root cause of the drained skill data and of the puzzle WM side-effect failure (offline, val, PRIVILEGED scoring; local analysis on copied val tables, no cluster compute): events merge consecutive interactions.** Unified self-training events per val episode vs true interactions: cube-triple 5.8 vs 11.3 moves >= 2 cm; cube-double 1.9 vs 11.05; puzzle-4x5 6.2 vs 29.6 presses; puzzle-4x6 2.8 vs 29.1. Share of true interactions inside a clean one-to-one event: 33% / 5% / 7% / 3%. The report.json recall (.66-.95) counts an event spanning several interactions as recalling all of them, so it hid this.
  - Mechanism: a change is the gap between two contact-free rest runs of >= m = 5 SAMPLED frames (stride 5, so >= 21 frames of rest); the next interaction starts ~33 (puzzle) / ~46 (cube) frames later and the agent stays near, so rests between interactions are missed (cube-double per-identity change windows median 133-178 frames against 42-frame moves).
  - Consequences: skill segments hold several interactions, so refine flags them (and should); the puzzle WM learns composite transitions: even on clean one-press val events, the 57419 WM gets side effects right in 8.3% and only 55% of the toggled lights are marked changed in those events.
  - refine's contamination test compares raw single-frame reader outputs of all identities at two frames (occluded and junk identities included, noise-level thresholds): it flags 98% of clean one-press puzzle events, 93% of clean cube-double events and 18% of clean cube-triple events.
- **Cube-double identity audit (PRIVILEGED, visual; docs/diag_events_20261005/audit_cubedouble_identities.png):** id 2 (colour type 9) is a part of the robot arm at the top of the image; id 3 (place, radius 11 px) is the table's centre seam, whose disc colour changes when a cube or the arm's shadow enters it. The agent colour types 6 / 8 also contain static table edges and the seam, and the blue cube's shaded faces fall into agent type 6, so the blue cube is rarely "free of the agent" (front-end visible .47). In the 57420 closed loop, 87% of executed events ended because another identity "changed" (median 12-20 steps against ~42-step moves) and 22% of commanded events targeted ids 2 / 3.
- **Puzzle-4x5 perception (PRIVILEGED):** the self-round reader reads 16 of 20 lights at >= .997 per frame when uncovered; buttons 0-3 (back row) have no usable identity (ids 6/16/19/20 correlate .17-.48 with any button), so 32% of presses are mislabelled; partial occlusion and shadow give long misreads (not separable by run length).
- **Fix 1 implemented: `u_events.py --per-frame`** (old mode unchanged: regression on the cube-double front-round val events and labels is bit-identical). Offline on the same reader tables (val, PRIVILEGED scoring): cube-triple 11.7 events / episode, 86% of 2 cm moves in clean one-to-one events (from 33%); puzzle-4x5 29.6 / episode, 60% (from 7%), toggled lights marked changed in clean events 95% (from 55%), 649 spurious events from misreads remain, acted = pressed button 48% (32% of presses are back-row buttons without identity); cube-double 11.2 / episode, 44% (from 5%), limited by the junk identities.
- 57587 (cube-triple) / 57588 (puzzle-4x5): `slurm/u_pf.sh`, mig or main, 1 GPU, 8 CPU, 48 GB, 3 h. Per-frame events on the existing self-training rounds (57371 / 57419 reader tables, readers, round-1 thresholds) -> WM + cost-to-go || skill (40k steps, final checkpoint) -> closed loop (5 tasks x 6 episodes, seed 0). The only change against 57411 (cube-triple 8/30) and 57419 (puzzle, no loop) is the event extraction. Output: <self1000>/pf_<job>. Quota checked before submission (GPU 70 + 6 planned vs 94 ceiling; mem 5.43M + 0.29M vs 6.82M MB-h). RUNNING.
- `u_closed_loop.py` (synced 2026-10-05 ~20:05 ICT, after 57587 / 57588 started and before their loop stage, so their loops run it): PRIVILEGED logging only, no change to actions, event ends or planning. Per episode `sim_start` / `sim_goal` (cube positions or button states) and the reader states of the start / goal frames; per ended event the step `t`, all reader states `read_end` and `sim_end`; per timeout a `timeout_log` entry. Purpose: per-event execution success of the skill and correctness of the event-end rule (user issues 3 and 5). The source hash recorded at job start (`pf_<job>/src`) predates this change; the new hash is on the next line.
  - u_closed_loop.py sha256 79bc3de350d05d80…
- **Evidence hygiene (user question 2026-10-05: is this hand-engineering to fit the tasks?).** The `--per-frame` design was checked against PRIVILEGED val diagnostics of cube-triple, cube-double, puzzle-4x5 and puzzle-4x6. m = 5 and the place-coverage limit 1/2 were kept after trying m = 10 and coverage 1/10 on puzzle-4x5 under the same diagnostics. Dropping the rest head from per-frame observations followed the cube-double blue-cube analysis. So cube-double and puzzle-4x6 are no longer clean held-out families for this pipeline; the frozen-v1 transfer result (57420, 0/30) stays valid as a v1 result. No unified run or diagnostic is recorded for cube-single, puzzle-3x3 or puzzle-4x4: only data caching, plus the 56924 scripted-executor inventory on the puzzles. Before any new perception rule, decide the dev and held-out families and freeze the code hash.
- **57587 COMPLETED (01:13:36, exit 0; squeue empty for it, sacct COMPLETED). Unified cube-triple with per-frame events: 22/30 = 73% (by task 6/6, 6/6, 6/6, 4/6, 0/6), first plan found .90.** Dev seed 0, 6 episodes x 5 tasks, same reader / thresholds / protocol as 57411 (8/30: 6, 1, 1, 0, 0). References on the same seeds: unified loop with the cube-specific skill 16/30 (57408), privileged D 26/30 (57118); cube-specific label-free pipeline 40/60 over 60 seeds. WM (own val events): acted within tol 1.0, unchanged within tol .991 (57371: .835). Skill final val chunk .419. Task 5 remains 0/6. Pixel track, SAM 2 front end (pretrained; rules developed on these families), not yet held-out.
- 57603 (ew_fetchst): `slurm/fetch_state.sh`, main CPU, 2 CPU, 16 GB, 45 min: official OGBench STATE play datasets cube-triple-play-v0 and puzzle-4x5-play-v0 (+ val) from the HF mirror into $DATA, for a state track compared with published state-based baselines. RUNNING.
- 57604 (ew_ueval): `slurm/u_eval.sh`, mig, 1 GPU, 12 CPU, 48 GB, 1.5 h: official-protocol evaluation (OGBench default 5 tasks x 20 episodes) of the frozen 57587 models on env seed 1, never used for cube-triple (all earlier cube runs used seed 0). Output pf_57587/eval_seed1_57604. RUNNING.

## 2026-10-05 state track of the unified method (user request 21:00 ICT: state-input closed loop, 100 episodes, cube and puzzle, to compare with published baselines)
- 57604 (pixel cube-triple 100-episode evaluation) CANCELLED on the user's request after 21 min (they want the state track, not the pixel number). squeue empty for it, sacct CANCELLED. No result kept.
- 57603 COMPLETED (1 min): official OGBench state play datasets in $DATA: cube-triple-play-v0 (observations 3,003,000 x 46, 3000 episodes; val 300) and puzzle-4x5-play-v0 (3,003,000 x 99, 3000 episodes; val 300), plus qpos (and button_states for puzzle), which only the PRIVILEGED diagnostics read.
- **Design (same backend code and settings for both families; input formatting follows the env's documented object-factored observation):** `scripts/s_entities.py` turns observations into the backend's entity tables. A cube block gives (x, y) and height; a button block gives its discrete state, and its location = median end-effector xy while its own joint is pressed (label-free; on val it recovers the exact 4 x 5 grid, 10 cm spacing). The agent is the end effector as a point (`effector` track; u_events.py now accepts it instead of agent masks, with the same half-object-width contact-free rule). Thresholds: object width = twice the lowest resting height (cubes) or the smallest button spacing; tol_pos = half a width; noise radii by the usual 2-means split. `u_skill.py`: an MLP on two standardised observation vectors when the cache holds vectors (pixel path unchanged). `u_closed_loop.py --state-layout`: entity states from the observation, "at rest" = unchanged since the last step and effector clear (the extraction definition); pixel path unchanged. Regression after these edits: old-mode events/labels on the 57420 cube-double val tables and per-frame pixel events on the cube-triple val tables are bit-identical to before.
- **Offline checks on val (PRIVILEGED scoring, local):** cube-triple 10.8 events per episode vs 11.29 true moves >= 2 cm, 95% of those moves in clean one-to-one events; puzzle-4x5 29.5 vs 29.57 presses, 99.8% of presses in clean one-to-one events, acted = pressed button 97.3%, changed set = true toggled set 97.3%. (The u_events report's puzzle reference here also counts button-joint motion from qpos, about two intervals per press, so its one-to-one numbers for puzzle are not meaningful; scored against button toggles instead.)
- 57613 (cube-triple) / 57614 (puzzle-4x5): `slurm/s_pipeline.sh`, mig or main, 1 GPU, 12 CPU, 48 GB, 4 h. All 3000 train episodes; events -> WM + cost-to-go || state skill (40k, final) -> closed loop: dev (seed 0, 6 x 5 tasks) then OGBench protocol (5 tasks x 20 episodes, seed 1). Output $RUN_ROOT/state_<env>_<job>. Quota before submission: GPU 73 + 9 planned vs 94; mem 5.60M + 0.44M vs 6.92M MB-h. 57613 RUNNING; 57614 PENDING (QOSMaxGRESPerUser) until 57588 releases its GPU.
- 57615 (ew_stsmoke): main CPU, 2 CPU, 8 GB, 20 min: code-path smoke test of the state closed loop with untrained local smoke models (one cube-triple-v0 episode; numbers meaningless).
- 57615 COMPLETED (13:24, exit 0; sacct COMPLETED): the state closed-loop path runs end to end (observation -> entity states, A*, state skill, timeouts, JSON with simulator logs). Smoke models are untrained, so no plan and 0/1 are expected. Coordinate check: sim cube (0.4938, -0.0904, 0.02) m is read as (37.50, 24.77, 0.05) and the goal (0.5, 0.1, 0.02) as (38, 40, 0.05), as defined in s_entities.py.
- Published STATE baselines for the comparison (OGBench paper, Table 2, 8 seeds; 5 evaluation tasks per dataset, 20 episodes per task in the reference implementation): cube-triple-play-v0 GCBC 1, GCIVL 1, GCIQL 3, QRL 0, CRL 4, HIQL 3; puzzle-4x5-play-v0 GCBC 0, GCIVL 7, GCIQL 14, QRL 0, CRL 1, HIQL 4. (The visual numbers of the same table, cube-triple 21 / puzzle-4x5 17, were used only to cross-check the extraction against the ledger.) Our runs use one training seed so far.
- 57588 (puzzle-4x5 pixel, per-frame events) CANCELLED on the user's request at 02:36:54 elapsed (22:22 ICT), to free the GPU for the state track. Its closed loop had started at 21:54 ICT and finished 0 of 30 episodes in 24 min (CPU rendering), so no closed-loop number exists. Saved offline WM checks (own val events): side effects within tol .285 (57419: .083), unchanged within tol .913 (.697), offline plan found .456 (.032), plan length / data length median 1.0. Models are kept in pf_57588/{wm,skill}. 57614 (puzzle state) started right after, verified with squeue and sacct.
- **57613 COMPLETED (00:53:10, exit 0; squeue empty, sacct COMPLETED). STATE TRACK, cube-triple-play-v0 (official state data, 3000 episodes, one training seed):**
  - **OGBench protocol, env seed 1 (never used): 74/100.** By task (of 20): 20, 20, 20, 14, 0. First plan found 1.0. Dev (seed 0, 6 x 5): 24/30.
  - Published state baselines (OGBench Table 2, 8 seeds): best CRL 4; GCIQL 3, HIQL 3, GCIVL 1, GCBC 1, QRL 0.
  - WM (own val events): acted within tol 1.0, unchanged .998, offline plan found 1.0, plan length / data length 1.0. Skill final val chunk about .28 (pixel skill .42).
  - PRIVILEGED per-event check (commanded cube within 4 cm and 2 cm in height of the event target, simulator log): task 2 40/43, task 3 41/48, task 4 50/106, task 5 19/87. 255 of 284 events ended with the acted cube settled.
  - **Task 5 (three-cube tower at (0.425, 0.2)) failure:** the first event is "cube 0 to the top-level position" while no tower exists. It is repeated three times (the cube lands on the table next to the target), then the episode runs out of steps. Cause: the event parameter x is the full target state, including height, and the WM is trained with x = the achieved after-state (hindsight), so it predicts after = x and cannot represent support: an unsupported top placement looks feasible. Direction D avoided this by conditioning on the target xy only and learning the resulting height. Candidate fix (generic): for an entity whose position changes, the event parameter is its target position; the WM predicts the rest (height, covered).
- **Fix for task 5 implemented (user approval 2026-10-05 ~23:10 ICT): `u_wm.py --event-pos-only`** (off by default; the checkpoint records it, and Model.step_batch applies it only when recorded). For an event that moves the acted entity, the model gets the target position and the entity's current appearance (height in the state track), so the landing height and covered bits are predicted, not copied. Local checks: on cube state val events 614 / 1080 targets change (the moving ones); on puzzle pixel events 0 change; on cube pixel self-round events the read colour differs by reader noise, so the flag is not a no-op there, which is why it stays off for the pixel pipelines. Synced before 57614 reaches its loop; 57614's checkpoint has no flag, so its planning is unchanged.
- 57624 (ew_stwm): `slurm/s_wm_rerun.sh`, mig, 1 GPU, 12 CPU, 32 GB, 2 h. Retrains WM + cost-to-go of 57613 with --event-pos-only; keeps 57613's entities, events and skill; loops on seed 0 (dev, 6 x 5), seed 1 (paired with the 57613 model, 20 x 5) and seed 2 (fresh, the clean protocol number, 20 x 5). Output state_cube-triple-play-v0_57613/wm_57624. Quota: GPU 76 + 5.4 planned vs 96; mem 5.76M + 0.23M vs 6.92M MB-h. RUNNING.
- 57626 (ew_strender): main CPU, 4 CPU, 16 GB, 40 min; COMPLETED 06:26, exit 0. Renders val episodes 0-1 of the official cube-triple and puzzle-4x5 STATE datasets from their saved simulator states (qpos, qvel, button states) with the front_pixels camera at 160 px. Output $RUN_ROOT/demo_state_events_57626. Source docs/demos/state_events_20261005/{render_frames.py,render.sbatch}.
- Event demo page (local build `docs/demos/state_events_20261005/build_demo.py` + `template.html` -> `site/`), published as a private artifact: https://claude.ai/artifact/A8Y2DDP3ehRw8TD8Td3HTF. It shows the val events of 57613 / 57614 frame by frame, with the acted object circled and the cube target marked (camera projection checked visually against the rendered cubes, stacked case included), a logic-analyzer style timeline and an event table, against PRIVILEGED ground truth. Shown episodes: cube 11 events vs 12 and 11 true moves >= 2 cm; puzzle 29 vs 29 and 30 presses.

## 2026-10-06 state track: cube fix result, puzzle cost-to-go fix (00:00-00:40 ICT)
- **57624 COMPLETED (00:51:18, exit 0; squeue empty for it, sacct COMPLETED). Cube-triple STATE, --event-pos-only (57613 events and skill, retrained WM + cost-to-go):** dev seed 0 29/30; **seed 1 97/100** (paired with 57613 on the same episodes: 74/100; by task 20, 20, 20, 19, 18); **seed 2 (fresh) 97/100** (20, 19, 20, 19, 19). First plan found 1.0. Published state baselines on cube-triple-play-v0: best CRL 4. One training seed.
- **57614 (puzzle-4x5 STATE, first version): dev seed 0 6/30** (task 1 6/6, tasks 2-5 0/6), first plan found .20. WM (own val events) is good: acted 1.0, side effects within tol .980 (9062 entities), unchanged .994, offline plan found .949 (gaps of 1-3 events). The failure is the cost-to-go: on tasks 2-5 (10-20 presses) A* expands 20k nodes with best h ~29.96 (the cap of 30), i.e. h is flat far from the goal. Cause: value iteration on dataset (S, G) pairs, which on a 20-button board are ~10 presses apart, so near-goal states are almost never sampled. The puzzle-specific pipeline had the same issue and trained on goals from imagined random walks (README, cost-to-go row) with width 2048 (job 56943). Its OGBench-protocol loop (seed 1) is still running at this point; it measures the version before the fix.
- **Fix: `u_wm.py --h-goals walk --h-width 2048`** (defaults unchanged). Half of each cost-to-go batch pairs a dataset state with a goal reached by k ~ U{0..30} imagined random prototype events. A pool of 100k pairs is built once after WM training. h width is stored in the checkpoint. Local smoke test: pool built (mean walk 14.9 events), training runs, checkpoint loads. Running processes of 57614 imported the old module; old checkpoints load with width 1024 as before.
- **Unified state config from here on (both families):** `--event-pos-only --h-goals walk --h-width 2048`. 57630 (puzzle, from 57614's events and skill) and 57631 (cube, from 57613's events and skill, to keep one config across families): `slurm/s_wm_rerun.sh --time=03:00:00`, loops on seed 0 (dev), 1 (paired) and 2 (fresh). Quota: GPU 78 + 7 planned vs 98; mem 5.84M + 0.25M vs 6.92M MB-h. 57630 RUNNING; 57631 PENDING (QOSMaxCpuPerUserLimit) until 57614 ends.
- **57614 COMPLETED (02:20:25, exit 0): puzzle STATE, first version, OGBench protocol seed 1: 20/100** (task 1 20/20, tasks 2-5 0/20), first plan found .20. This is the version before the cost-to-go fixes.
- **57631 COMPLETED (00:58:12, exit 0): cube-triple STATE with `--event-pos-only --h-goals walk --h-width 2048`: dev 30/30, seed 1 100/100, seed 2 97/100** (20, 19, 20, 19, 19). The walk-goal cost-to-go does not hurt cube.
- **57630 (puzzle, same config): still failing on tasks 2-5** (first episodes: no first plan, 1000 steps). Local check of its cost-to-go against GF(2) press distances on 400 val-based pairs (PRIVILEGED scoring): mean h 1.9 / 4.8 / 5.3 / 5.3 / 5.3 / 4.6 for d* 1-3 / 4-6 / 7-9 / 10-12 / 13-16 / 17-20, i.e. right to ~5 presses and flat beyond. The puzzle-specific planner had the same wall and passed it with width 2048, an s XOR g input and 150k steps (README; jobs 56943, 56980).
- **Fix: `u_wm.py --h-absdiff`** (h input also gets |s - g|; stored in the checkpoint; old checkpoints load as before), plus `--h-steps 150000 --h-walk-pool 300000`. Local smoke test passed. Remote hash before the sync was the previous synced version.
- **Final unified STATE config (both families):** `--event-pos-only --h-goals walk --h-width 2048 --h-absdiff --h-steps 150000 --h-walk-pool 300000`. 57636 (puzzle, 5 h) RUNNING; 57637 (cube, 4 h) PENDING until a GPU frees. Both rerun the WM + cost-to-go on the earlier events and skills, with loops on seed 0 / 1 / 2. Quota: GPU 80 + 10.3 planned vs 100; mem 5.90M + 0.34M vs 6.92M MB-h.

## 2026-10-06 cost-to-go example for the existing demo

- Job 57640, ew_costdemo: main CPU, 2 CPU, 8 GB, 2-minute limit. Recompute h(S,G) on the four saved task-2 states in demo report job57580, using the same model SHA256 7eb88a9dd077195f809c98afaefa505d90364faf584d26a03f9e707df3f4bcd8 and frozen u_wm.py from demo57497. No training, search rerun, rendering, physics, or model changes.
- Source: docs/demos/cost_search_example_20261006/{read_cost.py,read_cost.sbatch}; remote distinct dir /mnt/data/nhatnc129/jepa/event_wm/cost_search_example_20261006. Values are CPU recomputations, not logged original priorities. Status at submission: SUBMITTED.

- 57640 final verification: absent from squeue; sacct COMPLETED, 00:00:03, exit 0:0. Saved result_57640.json. Recomputed task-2 h along the saved 3-event plan: 30.00185, 29.93694, 29.92266, 29.90763; the final state passes at_goal despite h remaining near 30. This is a diagnostic of the old demo checkpoint, not of newer per-frame/state-track checkpoints. Original search returned a 3-event plan after 91 expansions; no search rerun.

## 2026-10-06 live verification (03:50 ICT)

- Verified squeue and sacct: 57636 RUNNING ~02:00 elapsed, 5 h limit; 57637 RUNNING ~01:56 elapsed, 4 h limit. Both currently train cost-to-go, not closed loop. Latest logged h steps: puzzle 85000/150000; cube 115000/150000. WM training finished.
- Verified saved JSON summaries directly: pixel cube pf_57587 22/30 (6,6,6,4,0 of 6); state cube wm_57631 seed1 100/100 and seed2 97/100 (20,19,20,19,19 of 20); state puzzle initial 57614 seed1 20/100 (20,0,0,0,0 of 20).
- Latest running unified config uses event-pos-only, walk goals, width2048, absdiff, 150k h steps and 300k walk pool. Earlier state skills and events are reused; no new skill training in these two jobs. Outputs have no final closed-loop result yet.
- State input is the benchmark observation vector with documented object-factored layout, rather than the SAM2/pixel reader. State data use all3000 episodes; pixel runs use self1000. Do not combine the two tracks. One training seed; evaluation env seeds1/2 have already informed prior model iterations and should not be called newly untouched holdouts for the ongoing config.

## 2026-10-06 live verification (04:23 ICT)

- Both scheduler views checked: 57637 COMPLETED, elapsed02:20:51, exit0:0; 57636 RUNNING elapsed02:33:30 of5h.
- Latest unified state cube config (57637) saved JSON summaries verified: seed0 dev29/30 (6,6,6,6,5), seed1 99/100 (20,20,20,20,19), seed2 97/100 (20,19,19,20,19). First plan found1.0. These are learned closed-loop results, one training seed, no new untouched-family claim.
- Puzzle57636 latest log h_step115000/150000, loss0.049. No closed-loop stage or final success result yet. Latest completed puzzle result remains57614 20/100, before the current cost-to-go fix.

## 2026-10-06 live verification (07:11 ICT)

- Verified both schedulers: account squeue empty. 57636 TIMEOUT at05:00:01 elapsed, killed06:49:58 ICT; 57637 remains COMPLETED.
- Puzzle57636 trained and saved model, completed dev seed0 closed loop:6/30 (6,0,0,0,0 of6), first_plan_found0.2,28.7min. Seed1 eval did not finish/save final JSON: log contains96 unique completed episodes,20 successes (task1 20/20; tasks2-4 each0/20; task5 0/16). Seed2 not started. Partial20/96 must not be labelled a complete100-episode score.
- Dev first calls for tasks2-5 exhaust20053 expansions despite best_h approximately0.70/0.63/0.32/0.34; initial planning latency15.65/24.40/23.79/23.81s. The previous flat-at30 diagnosis no longer describes these search outputs; the new config still fails to produce complete plans. Root cause needs separating heuristic optimism, WM rollout error, goal checks, and candidate constraints before another rerun. No new job submitted.
- Local evidence: docs/unified_status_20261006/{ew_stwm_57636.out,puzzle_57636_dev.json}. Latest cube57637 result unchanged:seed1 99/100,seed2 97/100, state track.

## 2026-10-06 puzzle root-cause investigation (user request)

- 57669 ew_puzcause: mig/main,1GPU,2CPU,12GB,15min. Frozen57636 checkpoint; saved dev roots; diagnostic known-press trajectories for5tasks and task2 search controls separating learned/oracle WM and h plus generic nearest-rest-prototype projection. No training, no simulator, no core source changes. Oracle controls use the simulator's puzzle rules only for attribution and are not method results. Source docs/diag_puzzle_20261006; output /mnt/data/nhatnc129/jepa/event_wm/diag_puzzle_cause_20261006/job_57669.
- Before submission squeue empty and sacct checked. Month usage GPU88h/108h ceiling, CPU680h/1094.4h, memory6,195,495MBh/7,969,481.1MBh. Planned15min adds0.25GPUh,0.5CPUh,3072MBh.
- 57636 saved dev traces: all746 ended events in failed tasks2-5 reached their acted-entity target and toggled that entity. Full-effect-set and state-cycle audit saved skill_trace_check.json; simulator information only diagnostic.

- 57669 verified COMPLETED22s,exit0:0 with both schedulers. On known correct press sequences all5task roots, raw WM reaches the goal and thresholded bits remain correct; task2 raw search+learned h fails20053 expansions, h0.699 at a state still9 true presses away. Oracle WM+learned h succeeds2005 expansions; raw WM+oracle h succeeds533; generic nearest-prototype projected WM+same learned h succeeds2005. Controls are offline attribution, not policy benchmark results. All746 ended events of failed tasks have the exact simulator press-effect set, confirming selected-event execution succeeds while planning loops.
- 57670 sensitivity follow-up: sameGPU/2CPU/12GB,5min limit. Changes h input position vs appearance separately on the failing state, then tests raw WM with only projected h input and projected WM on roots for tasks2-5. No training or model changes. Quota rechecked before submission;88GPUh+planned0.0833h stays below108h; CPU/memory also below90% limits.

- 57670 verified COMPLETED1m59s,exit0:0. Correcting only h inputs to nearest prototypes does not solve any task2-5 root. Projecting WM outputs solves task2 but still fails tasks3-5, so heuristic sensitivity to small numerical errors is not a sufficient explanation. The failing task2 board remains h0.867 even after projection although reference distance is9.
- 57671: same resources/5min. Concrete follow-up: raw WM+reference distances vs reference dynamics+learned h on task3-5 roots; task2 search-key-only control and alias counts. Quota rechecked, account scheduler empty before submission; all planned usage below90% caps. No training or core-source change.

- 57671 verified with squeue and sacct: COMPLETED1m13s,exit0:0; account queue empty. Raw learned WM+exact h finds physical-rule-valid plans for tasks3/4/5 with14/16/20 presses and789/917/1173 expansions. Exact dynamics+learned h still exhausts20053 expansions on all three. This identifies learned h as the primary functional bottleneck in long puzzle search, not an inability of this WM to represent these solution paths.
- Task2 raw search generates229186 unique raw keys for3513 logical binary boards (up to3735 keys per board). Changing only duplicate-detection key to nearest-rest-prototype key finds a10-event plan after1941 expansions, vs raw-key failure at20053. This diagnoses state aliases as a secondary cause; the key-only plan was not evaluated closed-loop. Projected WM alone still fails tasks3-5.
- All746 ended events logged in failed dev tasks2-5 have the exact expected physical press effects; this scopes skill evidence to ended events, not every action. Correct long press sequences also reach goal in the learned WM. h underestimates one board's true remaining9 presses as0.699, and0.867 even after prototype projection.
- No core code changed, no retraining or new learned closed-loop score. Puzzle57636 dev remains6/30; oracle controls are offline diagnostics. Evidence and limits recorded in docs/diag_puzzle_20261006/REPORT.md with all3JSON reports and scripts. Next grounded changes concern equivalent-state keys and long-distance cost learning; do not add privileged puzzle rules to the method.

## 2026-10-06 bounded repair investigation (user: find how to fix)

- 57680 ew_puzrepair: mig/main,1GPU,6CPU,16GB,35min. Isolated source under docs/repair_puzzle_20261006/source, remote /mnt/data/nhatnc129/jepa/event_wm/repair_puzzle_20261006; source SHA manifest saved. Core/peer dirty files preserved.
- Concrete hypothesis: real training S vs continuous imagined goals/successors creates inconsistent Bellman representation; canonicalizing train-supported finite attributes at targets, WM outputs, h inputs and search keys should remove aliases and permit useful continued h learning. Support inferred only from train before/after values (<=8 at1e-5 precision); continuous attributes left untouched. This cached pilot requires finite state support and aborts if inapplicable or goal-dependent candidate sets differ. No puzzle dynamics rule, GF2 labels, eval goals or privileged targets used for training/search.
- Keep WM and skill frozen. Same h architecture/checkpoint, continue15000 steps,lr1e-4,batch512, half real goal pairs and half60000 new imagined walk pairs, target copy1000, same20k expansion budget. Candidate successors cached after coverage checks. Then all5saved dev roots and learned state closed loop on1episode/task,seed0,5workers. This is a development pilot, not a benchmark or unseen-family claim. No oracle is inserted into the method.
- Before GPU submission both squeue and sacct checked: account queue empty,57669/70/71 COMPLETED. Monthly usage GPU88h vs108h ceiling,CPU680h vs1094.4h,mem6195495MBh vs7969481.1MBh. Planned maximum adds0.5833GPUh,3.5CPUh,9557.3MBh. All below90% caps.

- 57680 CANCELLED at6m24s (both scheduler views checked) during imagined-goal preparation, before h training/checkpoint/eval. Candidate count20 and cache32768states completed; Python per-state candidate calls were the preparation bottleneck. No method result from this job.
- 57684 replaces it, same1GPU/6CPU/16GB,30min limit; train_repair_v2.py + repair_v2.sbatch. Same repair/training budget and candidate operator, vectorized train-prototype sampling with explicit functional equivalence checks before rollout. Source_V2_SHA256SUMS preserves v1 manifest. Outputs job_57684.
- Before GPU submission squeue/sacct verified57680 cancellation and no peer/duplicate run; sreport still88GPUh/680CPUh/6195495MBh. Counting cancelled job's full prior limit plus new planned0.5GPUh,3CPUh,8192MBh remains below108GPUh/1094.4CPUh/7969481.1MBh ceilings. No repeated scientific configuration or extra training seed was submitted.

- 57684 CANCELLED at4m43s after completing all15000 h training steps and saving job_57684/{u_model.pt,training.json}. Initial search task1 succeeds4events/149expansions; task2 succeeds10events/1749; task3 fails20053. Scalar Python canonicalization made task3 take96.6s, risking evaluation timeout. Frozen source and model retained; no retraining.
- 57685: evaluation continuation only,1GPU/6CPU/16GB,20min. Uses same saved57684 checkpoint and same20k expansion budget. source_fast/u_wm.py vectorizes canonical state/target support lookup; evaluate_fast.py asserts exact equality with previous scalar lookup on random scalar/batched inputs before all5root searches, then learned closed loop1episode/task seed0 using same frozen skill. Output job_57685. This changes implementation speed, not state mapping or training.
- Before submission squeue/sacct verified57684 cancellation, no peer/duplicate scientific run; sreport refreshed, GPU88h/CPU680h/mem6195495MBh. Planned0.3333GPUh,2CPUh,5461.3MBh plus both previous jobs' full limits remains below90% caps. SOURCE_FAST_SHA256SUMS saved separately; old source/manifest preserved.

- 57685 FAILED1m14s,exit1:0 after all5planning roots completed. Vectorized/scalar canonicalization equality check passed. Search: task1 4events/149exp/.40s; task2 10events/1749exp/1.77s; tasks3-5 fail20053exp/~21s each (h0 8.08/8.31/8.69; best_h1.20/1.16/1.19). These establish canonicalization plus continued single-step training is insufficient for long roots.
- Failure was isolated-source dependency packaging: s_entities imports sfa_code, missing from the frozen PYTHONPATH. No simulator episode completed. Added frozen source_support copies as dependency fallback; source_fast remains first and all active method modules unchanged.
- 57686 ew_puzloop: loop continuation only, same57684 checkpoint/source_fast/skill and20k budget,1episode/task seed0,5workers;1GPU/6CPU/16GB,15min. STATE_IMPORT_OK before evaluation. No training or root-search rerun. Output job_57686; source manifest SOURCE_LOOP_SHA256SUMS.
- Before GPU submission squeue empty and sacct57685 FAILED verified. Fresh monthlyGPU89h vs108.9h ceiling (5th121h),CPU681h vs1094.4h,mem6199380MBh vs8146428.3MBh (5th9051587). Planned0.25GPUh/1.5CPUh/4096MBh plus full limits of previous pilot jobs remain below90% caps.

- Latest verified57686 state: RUNNING2m48s by bothsqueue/sacct. Stdout confirms learned closed-loop task1 success119steps, task2 success306steps, task3 success463steps, all0timeouts; task3 had no full initial plan and succeeded by replanning. Each1development episode, seed0. Final task4/5 and finalJSON not yet fetched.
- At02:14:43UTC (09:14:43ICT), repeated configured SSH attempts fail because gateway10.254.152.76:22 refuses connection; fresh ControlMaster-free attempt same. Direct configured target100.96.2.205:22 times out. Current job state unknown after last verified snapshot. No result/state inferred from connection failure. Job has15min limit.
- Local evidence and recommendation finalized under docs/repair_puzzle_20261006. This entry/final recommendation have not been synced to remote due connectivity loss. Core project source remains unchanged; one new trained checkpoint in job57684 is retained, evaluation jobs reused it. LHBL remains a researched recommendation, not an implemented result.

## 2026-10-06 puzzle repair pilot final result (verified 15:54 ICT)
- Verified with both squeue (account queue empty) and sacct: 57686 ew_puzloop COMPLETED00:03:57,exit0:0. SSH recovered; historical network/partial-result entries remain historical.
- Same isolated canonical-state repair + continued SINGLE-STEP h checkpoint job57684/u_model.pt; frozen learned WM and skill. LHBL NOT implemented or used.
- STATE-track learned closed loop, development seed0,1episode/task: success5/5. Task1..5 steps119/306/463/566/640, recorded replans3/9/13/17/19, event timeouts0 for all. Tasks1/2 full first plans4/10; tasks3..5 no complete first plan but repeated partial plans/replanning reached goal. First search0.76/2.88/24.60/24.51/22.28s; loop3.9min with5workers.
- Final source result: /mnt/data/nhatnc129/jepa/event_wm/repair_puzzle_20261006/job_57686/loop_seed0/u_closed_loop.json; local copy docs/repair_puzzle_20261006/loop_57686_final.json; SHA256 e2fd0f08f987b6e417cb644d6cb3e0a4e5f00adfbb814304f9237a4ce7bbbf08.
- This is5development episodes, not a100-episode benchmark, not an8-training-seed baseline comparison. Current h still struggles to find full long root plans; next confirmation is full evaluation of the frozen repair before a matched limited-horizon-search training comparison. No new job submitted for this status/recommendation request.

## 2026-10-06 scene STATE implementation and end-to-end experiment (user authorized)
- Isolated source docs/scene_state_20261006/source; remote /mnt/data/nhatnc129/jepa/event_wm/scene_state_20261006. Existing core/peer dirty files preserved.
- Public40-dimensional scene observation ->5entity tokens(cube,2buttons,drawer,window), same six fields/WM/h/BC skill. Sliders expose joint positions rather than handle xyz; contact geometry is an affine calibration from TRAIN public joint/effector trajectories. No lock rules, prescribed sequences, solver or scripted executor.
- General finite-rest-support canonicalization from TRAIN event states, continuous attributes unchanged; no finite-state successor cache. No LHBL. Semantic tests first failed on unsupported scene / missing finite-support integration, then passed after isolated implementation; local syntax checks passed.
- 57767 ew_fetchst: CPU2/8GB/25min, downloaded official scene STATE play train/val. Verified both squeue/sacct COMPLETED18s,exit0:0.
- 57771 ew_scprep: CPU4/24GB/25min, public state calibration/cache ->same per-frame u_events ->semantic/alignment diagnostics. Output prep_57771. Job ID recorded immediately; results pending.
- Initial quota: GPU89h vs116.1h ceiling(5th129);CPU682h vs1167.3h(5th1297);mem6200677MBh vs9482886.9MBh(5th10536541). No peer/duplicate account jobs running before submission. Training/closed-loop submission requires fresh ranking.

- 57771 preparation verified COMPLETED39s,exit0:0 in both squeue/sacct. Actual official scene data1000train/100val episodes (1,001,000/100,100frames). Events17,858train/1,851val; acted counts train4632/3486/3357/2996/3387. Public-state semantic checks passed. Train handle contact fits median residual3.4mm drawer/2.2mm window; reset alignment diagnostics max18.2mm. Generic qpos reference includes joint press/release and is not a clean event-quality score; do not interpret its30%one-to-one as semantic event precision.
- 57772 ew_scene:1GPU/6CPU/24GB/4h. Same WM30000/h150000,width2048,absdiff,walk pool300000,event-pos-only; general TRAIN finite-support canonicalization added, continuous fields unchanged. State BC skill40000steps final checkpoint. End-to-end pilot1episode/task seed0 is in the same job;20/task seed3 evaluation follows using the saved model after interpreting pilot failures, without retraining unless a concrete bottleneck calls for it. Output job_57772/{wm,skill,loop_dev}; full source manifest recorded.
- Fresh sreport before submission:GPU89h+planned4h<=116.1h;CPU682h+24h<=1167.3h;mem6200677+98304MBh<=9482886.9MBh. Account queue empty and prep COMPLETED before submission. No speculative duplicate array or interactive allocation.

- 57774 ew_sceval submitted afterok:57772:1GPU/6CPU/16GB/1h; same saved scene model/skill,20episodes/task,seed3,5workers. Output eval_57774/loop_seed3. No retraining or speculative duplicate; released automatically when its evaluation ends.
- Dual scheduler verification:57772 PENDING(Priority),57774 PENDING(Dependency); fetch/prep COMPLETED. No scene success score exists yet. Last fresh month ranking before57774 retainedGPU89+bothplanned5<=116.1h,CPU682+30<=1167.3h,mem6200677+114688<=9482886.9MBh.
- Partial h checkpoints every20k and per-episode JSON preserve completed work. State semantic tests/syntax checks passed locally and in compute-node preparation. Source/config/status documented in docs/scene_state_20261006/{PLAN,STATUS}. Queued external compute remains pending; do not claim training or closed-loop completion.

## 2026-10-06 scene STATE final first run + bounded support repair (verified20:52ICT)
- Dual scheduler verification:57772 COMPLETED02:35:29,exit0:0;57774 COMPLETED00:00:47,exit0:0; account queue empty before the repair.
- Learned STATE scene,100episodes seed3:34/100; tasks1..5 successes20/0/14/0/0 of20. Dev seed0:2/5. First plan found100%; initial planning fast. Final model/skill retained; all150k h steps/40k skill steps completed. No LHBL.
- WM own-val events:acted-within-tol99.84%,side-effects84.43%,unchanged99.92%;offline-plan-found100%. These measure observed valid play transitions and do not establish feasibility of counterfactual candidates.
- Trace finding:every task2 first plan manipulates drawer/window while buttons remain locked; example predicted drawer v24.15->11.60 but observed24.15. Every task4/5 first plan places cube into closed drawer before unlocking/opening; actual cube remains outside. Identifies unsupported transition prediction as a concrete bottleneck; not failure to find a search plan. Skill/task3 also retains failures; do not assert all skills are correct.
- FinalJSON local docs/scene_state_20261006/loop_seed3_57774.json,SHA256 a83f5cf15503dfc5ae40b36365ec717906901078b7c879c024946a842065a37e;devJSON andWM-eval also copied locally.
- 57808 ew_scsup:main CPU4/12GB/25min,source_support_v2; learn a generic NCE event-support prior from TRAIN state/event/target pairs with context corruption, calibrate2%quantile on play validation, keep WM/h/skill weights frozen. Low support abstains from predicting progress. Corruptions are density-estimation negatives, NOT verified physical failures. No task/lock relationship/action order encoded in support model.
- Run support repair end to end in the same CPU job:5development episodes seed0,then100episodes seed4. Output support_57808. Different evaluation seed and CPU inference must be disclosed; no claim of paired34->new score or GPU latency from this run. No GPU allocated/no CPU array. Fresh rankingGPU91h vs119.7h ceiling(5th133);CPU697h vs1167.3h;mem6264884MBh vs9721306.8MBh. PlannedCPU1.667h/memory5120MBh safely below caps.
- The wrapper test failed before implementation and passed afterward; source syntax/bash checks passed. Existing source snapshots/core dirty files preserved. New model checkpoint and partial per-episode logs will remain separate.

## Scene support result and matched comparison (2026-10-06T21:10:47+07:00)
- 57808 dual scheduler verified COMPLETED 00:02:31, exit0:0; no GPU allocated. TRAIN support6000 updates completed; valid play-event retention97.76–97.99% per identity, calibrated from separate play validation.
- Development seed0:5/5. New evaluation seed4:87/100, tasks1..5=20/20,20/20,20/20,18/20,9/20. First plans found100%, depths2/6/4/5/8; initial plan median0.615s on CPU. All learned WM/h/skill parent weights unchanged; new generic contrastive support head is the only model addition. No physical negative labels or encoded lock rules.
- Remaining13 failures:2task4 and11task5; all exhaust750 simulator steps. Trace shows inaccurate cube/drawer coupled transitions and repeated reopen/push cycles; skill execution and event termination remain possible contributors, not proven sole causes.
- Generic loop correctness concern: prior event boundary also ends on any resting changed side effect, even while the acted object is still in transit. Isolated source_boundary_v3 adds an opt-in acted-rest-only boundary, preserving original default for matched controls. Semantic test failed before helper implementation and passed afterward; syntax checks passed. Core and previous source snapshots untouched.
- Submitted57810 ew_sccmp:main CPU4/12GB/15min, noGPU/noarray. Frozen parent WM/h/skill+support. Runs100support episodes seed3,100base episodes seed4 on same CPU4 hardware, then40diagnostic episodes task4/5 seed4 with acted-rest-only termination. Outputcompare_57810. Seed4 has now been inspected, so boundary arm is development, not untouched held-out evidence.
- Fresh month-to-date ranking before submission: accountGPU91h vs90% fifth134h=120.6h; CPU697h vs1170.9h; memory6264884MBh vs9780289.2MBh. Planned job adds1CPUh and3072MBh, safely within ceilings. Account queue was empty and accounting confirmed preceding work completed; no peer/duplicate run.
- Local result hashes: {"loop_dev_57808.json": "a3d99742eea1f32b4b652424715533425b154b236412b79744b819fade5ccd5e", "loop_seed4_57808.json": "e8a21d8fb73736f574331f47cce80ffb3014d7970d0d46c04d5379059a3dcab3", "support_report_57808.json": "e420248e3b4c50977053664f77a679f26e8a7c766bca7ea368ea941a857ebe93"}.

## Scene matched evaluation complete and coupled-skill continuation (2026-10-06T21:18:32+07:00)
- 57810 verified by both squeue(empty for job) and sacct:COMPLETED00:04:29,exit0:0.
- Support seed3:85/100, tasks20/20,20/20,18/20,18/20,9/20. Same initial episode identifiers as57774;52fixed,1regressed,33bothsuccess,14bothfail. Original57774GPU inference vs supportCPU hardware difference remains disclosed.
- Fully matched CPU seed4 control:base32/100 vs support87/100; tasks base20/0/12/0/0, support20/20/20/18/9. Identical logged initial states; goals at most0.001canvas units apart at three-decimal log precision (tolerance1.597); same CPU4workers, limits, candidates, skill, WM/h parent weights.55fixed,0regressions. Extra6000support training updates/head inference is the intentional added cost. This is an internal ablation, not a published-baseline comparison.
- Acted-rest-only diagnostic:task4=17/20,task5=9/20,26/40; original27/40.2fixed,3regressed. Does not resolve bottleneck; not adopted as reported method. Kept isolated optional source_boundary_v3.
- Concrete next issue:BCsample construction excludes every knock/multi-object event; this also discards legitimate coupled cube/drawer moves. Inclusion may improve such execution; it is a hypothesis and not asserted to explain all failures. WM coupled-transition prediction error also remains. Generic source_skill_v4 adds optional inclusion and checkpoint initialization; no scene/task-specific control rules.
- 57812 ew_scskill submitted, verifiedRUNNING onworker-mig-3g40gb-0 at21:17:44ICT via both scheduler views.1GPU,CPU6/24GB/45min. Continue57772skill for10000updates atlr3e-5 using all genuineTRAINevent segments, same release10/chunk8/architecture; optimizer reset; normalization/architecture compatibility asserted. Save data counts, parent hash, best and final checkpoints; predeclared usefinal. Frozen support57808WM/h; original event termination retained.
- End-to-end57812 includes40developmentepisodes task4/5 seed4, then matched100episodes each for parent/newskill on fresh envseed5. Seed5 has not been inspected at submission. Outputskill_side_effects_57812. Added training compute and new supervision choice must be disclosed. Semantic eligibility test failed before implementation and passed after; source syntax/bash checks passed. Core and prior snapshots untouched.
- Fresh ranking at submission:accountGPU91+.75planned=91.75h<=120.6h;CPU697+4.5=701.5h<=1170.9h;memory6264884+18432=6283316MBh<=9780289.2MBh. No other account jobs/duplicates remained; GPU allotted only for actual training and evaluation.
- Result hashes:{"loop_support_seed3_57810.json": "259629c56b2f604f59c81bf742e98c45c4dea8fdcc5cd635eb10b7b644659ba3", "loop_base_seed4_57810.json": "2f683a6c7e96c162091d07150063598c833b496c09777495c57cc0db81ef55b5", "loop_acted_rest_seed4_57810.json": "a2038e16af54a8c74d47cc47ce384f88c81f68794254548a34b2745f25faf50c"}.

## Scene skill continuation completed (2026-10-06T21:22:22+07:00)
- 57812 verified by both scheduler views:COMPLETED00:02:19,exit0:0.10000fine-tuning updates and all240evaluation episodes completed. No idle GPU retained.
- Data report:948/17858TRAINmulti-object events and120/1851VALevents had been excluded; inclusion raises frame/event supervision pairs from1079795 to1135223TRAIN,107196 to114186VAL. Same raw1000TRAINepisodes; no new simulator-generated supervision. Final skill validation chunk loss0.23017, not a closed-loop score.
- Newskill development seed4task4/5:18/20,13/20 (31/40 vsparent27/40). Fresh seed5 matched all-task arms on sameGPU5workers:parentskill87/100(tasks20,20,20,17,10), newskill92/100(tasks20,20,20,18,14). Initial logged states identical; goal differences<=0.001canvas units vs1.597tolerance. WM/h/support/candidates/limits unchanged.9fixed,4regressed,83bothsuccess,4bothfail. This is an aggregate improvement with individual regressions.
- Extra10000updates and inclusion of multi-object segments change together; no compute-matched continuation-only control, so do not attribute the gain solely to segment inclusion. One training seed,100evaluation episodes; approximate episode Wilson95 interval [85.0, 95.9]%, paired exact two-sided p=0.2668 for the +5point skill gain. Useful development evidence, not a statistically established or published-baseline win.
- Remaining8failures:2task4 and6task5, all750step horizon exhausted; repeated placement/drawer replanning and skill/WM coupled-motion errors remain. No claim of complete scene robustness or zero-shot transfer. NoLHBL/pixelreader in these STATE results.
- Current reproducible model: support_57808/u_model.pt; skill:skill_side_effects_57812/skill/u_skill.pt; source:source_skill_v4. Original termination retained; optional acted-rest-only arm not adopted. Full logs and checkpoints retained, source SHA manifests and parent checkpoint hashes recorded. Local report/STATUS/latest-run manifest updated. Result hashes:{"loop_old_skill_seed5_57812.json": "3caf3ce7e29fc1e4e469009ea210b1dc7fd651c388d1a4409e7527b7b1738c3a", "loop_new_skill_seed5_57812.json": "6b7415c3536a9407348f1e0a96de9fc79e60f9b2c21bc98fa7d47a89a386e7b1", "loop_new_skill_dev_seed4_57812.json": "ef157f8ed681c545b7bbfb317be7ea91d73281e3ba8921a6c89a5490667b4f87", "skill_data_57812.json": "30c4187b94a44df5a744289bf776e2488e382c318878fce496a3f724c77c050e"}.

## Unified three-family STATE rerun authorized and submitted (2026-10-06T21:43:59+07:00)
- User authorized rerunning puzzle/cube/scene with one unified version. Frozen source/protocol under docs/unified_rerun_20261006 and remote /mnt/data/nhatnc129/jepa/event_wm/unified_rerun_20261006; no core/peer dirty files or historical runs overwritten.
- All3families trained from scratch with the SAME recipe:1000TRAIN/100VALepisodes; WM30k,h150k,width2048,absdiff,imagined walks300k/max30,event-pos-only,TRAINfinite canonicalization,protos8; BC40k at3e-4 then10k at3e-5 including multi-object segments; learned event support6k,width256,seed61006,2%valquantile. Same generic planning/control/candidates; no finite-only successor cache, LHBL, oracle or scripted executor. Support afterh training matches scene recipe. Model dimensions/state adapters/data-derived scales vary by public interface.
- Cube/puzzle formerly used3000TRAINepisodes; this common1000episode run uses less data, so historical results are not directly matched. Scene's available1000episodes sets common amount. Training seed0; fresh environment evalseeds6&7,100episodes each/family=600total, five benchmark tasks. Max20kexpansions,timeout250,execution4,rest5,recovery8,weightedA*lambda.6/maxdepth30. Native cube/puzzle1000steps vs scene750 disclosed.
- Generic alignment correction:clip release supervision at episode terminal; previous recipe could assign an old event target after reset. Semantic test failed before implementation, passed after; full-episode/duplicate-reporting test also RED→GREEN. Public-state/finite-support semantic checks passed; allnewsource compiled, sbatchsyntax passed. Physics/modelinterface/data checks integrated into CPUpreparation.
- Submitted preparation57815(CPU4/24GB/25min),traincube57816/trainpuzzle57817/trainscene57818(each1GPU/CPU6/24GB/4h,afterok57815); evalcube57819/57820,puzzle57821/57822,scene57823/57824(each1GPU/CPU12/24GB/2h,seeds6/7,afterokcorrespondingtrain); aggregation57825(CPU2/4GB/5min,afteranyallevaluations). Noarray or speculative duplicate. End-to-end jobs registered durably in job_registry.json.
- Freshsreport before EACHGPUsubmission andsqueue/sacctpeercheck recorded in9quota audits. Fullplanned24GPUh+91used=115<=120.6ceiling;217.833CPUh+697=914.833<=1170.9;600405.333MBh+6264884=6865289.333<=9780289.2. Reserved entire protocol, not only current submission; noaccountpeer jobs found before submission.
- ProtocolSHA256 5afac26c8cd2b489c27a27f7b354c923f16476a784f1f51d98e9eea42a516b3e;78source/stage/configfiles frozen inSOURCE_SHA256SUMS. Stageentry verifies hashes; checkpoint metadata guards actualh150k/support6k/finalskill, aggregation rejects partial/duplicate protocol. Results must come from these runs, not historical scores. No new success number yet.

## Unified preparation verified and training active (2026-10-06T21:55:19+07:00)
-57815 COMPLETED00:05:21,exit0:0 verified by bothsqueue/sacct. All3actual data sets exactly1001000TRAIN/100100VALframes and1000/100episodes. Model/WM/h/support interfaces and public-state-proprio independence passed, all5tasks reset/read successfully per family; native horizons1000/1000/750 confirmed.
-TRAINevents cube10792,puzzle29474,scene17858; VAL1080/2950/1851. Allobject actors represented. Multi-object positional event counts320/0/948TRAIN (puzzle neighbours change appearance rather than position, so they were never removed by the positional-knock filter). Release clipping needed0/200/57TRAINsegments and0/21/4VALsegments, confirming the reset-alignment issue on realpuzzle/scene data; corrected for allfamilies by the same code.
-57816cube and57817puzzle verifiedRUNNING onworker-mig-3g40gb-0;57818scene PENDING(QOSMaxGRESPerUser), not a method failure. Evaluations57819..57824 andaggregation57825 remain dependency-queued. Same fixedrecipe/source, no speculative duplicate.
-Logs confirm actual WM andBC gradient steps, not empty/idle allocations. CubeWM30k finished andcost-stage began; puzzleWMlog reached15000, BC reached16000 at latest reads. CubeWMvalid side-effect-pos score0/28 is weak offline evidence; keep end-to-end evaluation and diagnose from actualclosed-loop outcomes, not this metric alone. No finalsuccess result claimed.
-Source/training recipe reviewed inline for family-dependent optimization and eval/simulator training inputs; none insharedtrainer/support trainer. Frozen adapter distinctions and density-negative limitations remain explicitly disclosed. Physics/model work stayed onCPU/GPUcompute nodes.

## Unified puzzle h timeout recovery (2026-10-07)
- Both squeue/sacct verified puzzle57817 TIMEOUT04:00:18, cube57816 COMPLETED03:11:24, cube eval57819/57820 COMPLETED00:00:26/00:00:25, scene57818 RUNNING00:51:13. No success score inferred from scheduler completion.
- Cluster denied extending running57817 to5h30m. Latest h partial checkpoint100k preserved; WM30k and BC40k+10k are reused. No scratch duplicate training. Recovery57858(1GPU,CPU6,24GB,2h30m,afterany57817) verified RUNNING00:01:27. Source scripts/resume_unified_h.py and slurm/resume_unified_h.sh; syntax checked locally, numerical recovery not yet complete. Frozen protocol/source unchanged and verified at runtime.
- Cancelled impossible afterok puzzle eval57821/57822 and outdated aggregation57825. Replacement eval57859/57860(seeds6/7,same frozen evaluator,afterok57858); replacement CPU aggregation57861(CPU2,4GB,5min,afterany six current evals). All replacement dependencies recorded in job_registry.json; aggregation explicitly adds recovery limitation to results.
- Resume resets Adam/RNG and regenerates imagined walk pool because original partial checkpoints lack optimizer/RNG/pool state. Final target remains150k effective h updates, with lost trailing work and additional recovery compute disclosed. Cannot claim bitwise-uninterrupted or exact compute-matched training. Original WM and BC weights preserved; recovery run gets distinct directory and checkpoint/source hashes.
- Fresh month-to-date sreport before every recovery/replacement GPU submission. Conservative remaining full limits:22.5GPUh+98used=120.5<=124.2;207.167CPUh+737=944.167<=1199.7;553301.333MBh+6427003=6980304.333<=10016218.8. Original puzzle/scene full4h limits included, even though most elapsed. No speculative array or idle holder.

- Follow-up small final-result metadata read: cube current rerun seed6=81/100, seed7=82/100 (163/200=81.5%), first-plan-found99/100 and100/100. Frozen common1000TRAINepisode recipe. Evaluator already validates the unique100-episode grid; these are learned STATE closed-loop scores, not pixel results or a published-baseline win. High plan-found rate does not establish that predicted trajectories are physically correct; execution/WM coupled failures still require trajectory diagnosis. Recovery57858 verified RUNNING00:02:15 by squeue+sacct and log confirms loading h_step100000 checkpoint; replacement57859/57860 and aggregation57861 PENDING dependencies. No claim that numerical continuation has completed.

## Unified three-family STATE rerun completed (2026-10-07)
- Both squeue and sacct verified all final train/eval/aggregate jobs COMPLETED exit0:0; no training or evaluation allocation remains. Scene57818 02:18:31; puzzle continuation57858 01:53:49; cube eval57819/57820,scene57823/57824,puzzle57859/57860 all completed; aggregation57861 complete.
- Complete frozen protocol results: cube seed6 81/100,seed7 82/100,total163/200=81.5%; puzzle40/100,40/100,total80/200=40%; scene90/100,90/100,total180/200=90%. Task1..5 pooled counts (40each): cube40,37,30,35,21; puzzle40,40,0,0,0; scene40,40,40,40,20. Validated600episode grid; original result SHA256s retained in results.json. No historical scores substituted.
- First plan found cube199/200,puzzle200/200,scene200/200. Plan discovery does not establish physical correctness; puzzle has systematic failures on tasks3..5. All checkpoints haveh150k/support6k/BC40k+10k. One trainseed0,two environment evalseeds6/7,1000TRAINepisodes each,STATEadapters. No published-baseline or pixel superiority claim. Puzzle Adam/RNG reset/pool regeneration and extra recovery compute disclosed by aggregator.
- Submitted focused saved-trajectory CPU diagnostic57874(CPU2,4GB,5min,main; noGPU,noarray), comparing predicted successor vs observed event states and actor target completion. Temporary source outside repo /tmp/event_wm_closedloop_diag_20261007.py; job copies source and input hashes to distinct diagnostics/closedloop_57874. No method change or speculative retraining. Diagnosis must precede attribution/fix.

- Diagnostic57874 FAILED00:00:01 exit2:0 (squeue empty+sacct verified): login-node /tmp source not shared with compute node. Moved source to shared run root and resubmitted57875 with identical CPU2/4GB/5min resources. No model or result changed; no idle allocation left.

- Diagnostic57875 COMPLETED00:00:01 exit0:0 verified by squeue+sacct; no jobs remain. Recorded puzzle event actor target not reached count0 and skill timeout count0 across all tasks. Tasks1/2 have0 WM button mismatches (120/360 settled events). Tasks3/4/5:644/1321,591/1267,633/1352 recorded transitions mismatch predicted button state despite acted object reaching requested target. Mismatch identities are systematic (task3 object13;task4 objects3,13,14;task5 objects7,11,13,17). Tasks3..5 exhaust1000steps and revisit2/4/4 unique end states per episode in aggregate. This is concrete evidence of side-effect WM/transition prediction failure and replanning loops in current run, not the former h-search bottleneck diagnosis. Upstream reason (WM fit vs extraction/canonicalization) still unverified; do not claim fixes or extrapolate skill perfection beyond settled events.

## Focused STATE WM diagnosis/fix authorized (2026-10-07)
- User explicitly authorized diagnosing whether event labels or WM predictions are wrong and applying a focused generic fix, followed by closed-loop. Preserve frozen unified_rerun_20261006 source/protocol and peer dirty files; no task solver or hand-coded puzzle effects.
- Scheduler peer check: squeue account empty; sacct today's prior diagnostic57874 FAILED and57875 COMPLETED. No duplicate or training job active. Monthly sreport current103GPUh/768CPUh/6552522MBh; fifthGPU147=>132.3ceiling,fifthCPU1346=>1211.4,fifthmem11232133=>10108919.7. GPU submissions will each recheck fresh ranking and include full planned limits.
- CPU audit57886 submitted(CPU6,12GB,15min,main), compares all extracted TRAIN/VAL event endpoints to public observations, evaluates rawWM vs canonicalization vs support, and reads development failure examples. All model loading/data analysis on compute; source scripts/diagnose_wm_state.py copied/hash-recorded in distinct remote wm_state_fix_20261007/audit_57886. CPU budget maximum1.5CPUh/3072MBh; noGPU or array. A compact correctness audit resolves the actual implementation decision; no novelty/go-no-go gate.

- Audit57886 verified COMPLETED00:00:10,exit0:0 by both scheduler views. Puzzle boundary mismatch:TRAIN before208/after483 of29474,VAL before24/after55 of2950. RawWM itself has errors before support/canonicalization; canonical-only does not change measured exact-event rate. Follow-up trace57890(CPU2,6GB,5min) inspects concrete label provenance and whether intervals span multiple native observation changes; all work on compute, no GPU yet. This distinguishes endpoint interpolation/visibility leakage from model fit.

## Event label repair and matched WM continuation (2026-10-07)
- Actor trace57891 COMPLETED exit0:0: same-frame public STATE endpoints and nearest agent among actually changing objects produce consistent observed effect signatures for all20puzzle actors. This is a diagnostic only; no effect table or puzzle solver used in extraction/training/planning.
- Two integration regressions57892 fail before patch for observable-state overwrite and missed contacted actor. Generic scripts/u_events.py now preserves readable same-frame endpoint measurements, clips to the same episode, and includes raw changing candidates; contact proximity precedes centrality. Occluded fallback remains. Patch touches no family-specific rule or frozen parent source.
-57893 COMPLETED00:02:28 and57895 COMPLETED00:00:02,exit0:0 verified with both scheduler views. Two regressions pass; public-state semantics, release-boundary and complete-episode checks pass.29,474TRAIN/2,950VAL endpoints have0mismatches;691/79actor labels corrected. All event times/episode/segment starts unchanged; same1000TRAIN/100VALepisodes. Pre-existing constant-cluster noise-estimation warnings occur before fixed thresholds and do not change effective finite thresholds.
- Matched continuation57897 submitted, initiallyPENDING via squeue; sacct ingestion initiallyempty.1GPU,CPU12,24GB,90min,afterok57895. Both arms start identical parentWM and each receives10kextraWMupdates at1e-4,seed0,freshAdam. One keepsoldlabels, onecorrectedlabels; compare oncorrectedVAL andlearnedclosedloop100episodes each seeds6(development)/8(fresh). Originalh/support/BC/candidates/search/native1000step limit unchanged. Extra traincompute is matched between arms; no pixel-result or published-baseline claim.
- Fresh quota and peer check before submission: noaccountlive/duplicate jobs; GPU103+1.5=104.5<=132.3;CPU768+18=786<=1211.4;mem6552522+36864=6589386MBh<=10063331.1. Full time limit counted. Source manifest48ddb9c7927290e346f3935db6356f19aca863179614ac5203a6b56b4c9428ab covers66files and is runtime-verified; separate run /mnt/data/nhatnc129/jepa/event_wm/wm_state_fix_20261007 preserves parent checkpoints/results.

-57897 nowRUNNING verified bysqueue+sacct; log confirms actualWMgradientsteps. Comparison57898(CPU4/8GB/10min,afterok57897) submitted for final rawWMVAL and realtransition diagnostics plus paired success/initial-state verification; no extra training. Maximum additional0.667CPUh/1365.33MBh stays within cap. Comparison source hash d9199f4f5404bfef737ebea7f9a8c8982afa9137dae1cfa8a546c8524858dc06.

-57898 dependency advanced toafterok57895 after WMcheckpoints ready; source comparison split into WM-only then final cached comparison to resolve residualVALerror while physics/search continues.57898 COMPLETED00:00:05,exit0:0 via bothscheduler views. Corrected-input/target VAL rawbinaryexact parent2941/2950,old-labelcontinuation2935/2950,fixed-labelcontinuation2950/2950. Canonical-only identical. Support afterfixedWM abstains66genuineVALevents, yielding2884/2950; keeporiginalsupport forcontrolledloop, do notmisattribute these66toWM. Finalcomparison 57906(CPU4/8GB/5min,afterok57897) reuses57898WMmetrics rather thanrepeatmodelprobe; addsmaximum0.333CPUh/682.67MBh within cap. Sourcehash 4e28cced76fc2195e8f84c38924086fc066d0d31ded56349f7917d7a62318e94.

## Focused STATE WM fix completed and verified (2026-10-07)
- Bothsqueue(emptyaccount) andsacct verify57897 COMPLETED00:52:49,57898 COMPLETED00:00:05,57906 COMPLETED00:00:04,allexit0:0. Noallocation remains. Frozen66file source manifest48ddb9c7927290e346f3935db6356f19aca863179614ac5203a6b56b4c9428ab reverified aftercompletion. Local/server extractor andexperiment driver synced; nopeerfiles orparentrun overwritten.
- Matched learnedSTATEclosed-loop seeds6/8:old-label WMcontinuation20/100+20/100=40/200; corrected-label continuation100/100+100/100=200/200,each task40/40. Same loggedinitialobject states/goals exactly;160pairedfixes,0regressions,40bothsuccess. Same1000TRAIN/100VALrawepisodes,unchangedeventtimes/segments,identicalparentWM initialization and10kadditionalWMupdates/arm. Originalh150k/support6k/BC50k/candidates/planner/native1000step limit keptfixed. Noanalyticpuzzle solver,effecttable,privilegedexecutor ornewfamily-specific methodrule.
- CorrectedVALraw/canonical binarystate exact2950/2950; originalparent2941/2950,matchedold-labelcontinuation2935/2950. Correctedmodel supportfinal2884/2950:66genuine events vetoed byold densitysupport,notWM errors. Recorded settledclosed-loop transitions new0/2520button-state mismatches versusold2477;0actor-target misses and0skilltimeouts inbotharms. Finalnativegoals canterminate beforelastinteraction settles,so2520isrecordedsettledtransitions,notallphysicalinteractions.
- Remainingperformance issue:task3firstsearch median31.865s/14933expansions;task4/5median42.745/42.645s and20053expansions (batched20kbudget overshoot),0/40completefirstplans but40/40success each via partialplan/replanning. Walltimes {"old_labels": {"6": 2.1, "8": 2.1}, "fixed_labels": {"6": 23.1, "8": 22.9}}. Same searchimplementation/caps but actualinferencecompute isgreater forcorrectmodel; do NOTclaim equal totalcompute orpublished-baseline/pixel superiority.
- One trainingseed0,fivebenchmarktasks,tworesetseeds6/8. EpisodeWilson95for200/200=[98.12,100]%,descriptive only;notindependenttrainingseeds ornovelscenario/generalization proof. This run establishes a STATE label-correctness repair andlearnedclosed-loop recovery. No post-fix cube/scene rerun orvisual-reader repair claimed.
- Currentmodel /mnt/data/nhatnc129/jepa/event_wm/wm_state_fix_20261007/repair_57897/fixed_labels/u_model.pt; fixedparent skill /mnt/data/nhatnc129/jepa/event_wm/unified_rerun_20261006/runs/puzzle/train_57858/skill_final/u_skill.pt; source/events /mnt/data/nhatnc129/jepa/event_wm/wm_state_fix_20261007. Source andcheckpoint hashes plusfullgrid/pairedchecks injob_registry.json,repair_57897/results.json andcomparison_57906/comparison.json. Comparison sourceSHA ddb718d40ec2ee55f2aec131848bbd6174cf253109243678d4df8fff1e96c36a; localresultSHA c45f44586815f56103fc63975eb37c434d9b9007c2ecddf9a6d7e689b9ed907c.

## Generic state front end + adapter-input baseline (user request 2026-10-07)
- Question: does the STATE-track result depend on the hand-written layout adapter (s_entities.py: which columns are robot / object blocks, movable vs fixed, on-top `covered` rule, scene column names)? Two parts, run dir /mnt/data/nhatnc129/jepa/event_wm/generic_state_20261007, local docs/generic_state_20261007.
- (1) Fairness control: OGBench reference CRL (impls/main.py unmodified, published cube-triple hyperparameters alpha 3.0, full 3000-episode dataset, 1M steps, 50 eval episodes/task) with raw input (published protocol) vs raw + the adapter's entity features (14 standardised columns: per cube u, v, height, covered; effector u, v). Wrapper baseline/run_baseline.py. Smoke 57946 FAILED (jax 0.4.35 CUDA path: nvidia.cuda_nvcc namespace package; fixed with CUDA_ROOT), smoke 57948 COMPLETED 2m23s (both arms train, ~2 ms/step, eval ~3.8 s/episode). Full run 57949: mig/main, 1 GPU (two arms concurrently), 8 CPU, 48 GB, 4 h, eval every 200k steps. Quota before submission: GPU 104 h + 4 planned vs 90% of 5th (152) = 136.8.
- (2) Generic front end source_generic/g_entities.py replaces s_entities.py with no layout knowledge: agent dims = high change rate or action response (2-means), object dims grouped by co-change (Jaccard, average linkage, 2-means cut), transient parts dropped (rest state never varies), discrete vs continuous dims, contact model by hard EM (agent configuration at change onset predicted from the entity's own state), entity (u, v, a0) = PCA of the contact space, a1/a2 = discrete dims, covered = 0. Thresholds by noise radii / rest-value gaps / same-level pair distances. Backend, recipe and protocol identical to unified_rerun_20261006 (+ fixed u_events of 2026-10-07); u_closed_loop dispatches on layout kind. Synthetic toy check local: agent dims, two objects and contact geometry recovered.
- 57950 ew_gprep: main CPU, 4 CPU, 48 GB, 1 h: g_entities + per-frame u_events for cube / puzzle / scene, adapter entity tables re-extracted with the same u_events, integrated checks, PRIVILEGED compare_events.json (acted accuracy, one-to-one). Training only after inspecting it.
- Generic front-end iterations (all main CPU 4 CPU / 48 GB / 1 h, COMPLETED; earlier prep dirs kept as prep_v1..v7): 57950 v1 (puzzle/scene buttons split into state + spring-joint entities, contact sensor `gripper_contact` kept as an object); 57951 v2 (merge/demote + touching radius; cube radius inflated by wobble / knock onsets); 57956 v3 (real-change filter; velocity dims broke it for drawer / window); 57958 v4 (entity-balanced contact statistics; PRIVILEGED check: true effector-cube distance at attributed onsets p50 5 mm, p90 7-16 cm); 57960 v5 (2-means helper silently refused < 10 samples, so demotion never fired; fixed); 57963 v6 (demotion by localisation ratio > 0.5 instead of a relative split, which had demoted scene cube / drawer / window); 57971 v7 (pruning of never-changing entities read event endpoints, where a pressed spring joint is still down); 57977 v8 = final: prune by contact-free rest-label changes.
- **57977 final generic preparation (val, PRIVILEGED scoring, same current u_events for both arms):** cube K 3 = the three cubes (contact sensor demoted), 10.61 events/episode vs adapter 10.8, acted correct .990 vs .993, 2 cm moves in clean one-to-one events .932 vs .951; puzzle K 20 = the 20 buttons (spring joints pruned), events identical to the adapter's (29.5/episode, acted 1.0); scene K 5 = cube, two buttons, drawer, window, 18.92 vs 18.51 events/episode, acted .929 vs .930, one-to-one .308 vs .302. Generic thresholds (canvas units of the learned contact plane, not comparable to the adapter's): cube tol 3.37, puzzle .71, scene .76.
- Training / evaluation with the unified_rerun recipe unchanged (WM 30k, h 150k, BC 40k+10k, support 6k; 1000 TRAIN episodes; eval seeds 6 / 7, 20 episodes x 5 tasks): train cube 57979 (4 h), puzzle 57982 (6 h), scene 57985 (4 h), each 1 GPU / 6 CPU / 24 GB; eval 57980/57981, 57983/57984, 57986/57987 (1 GPU / 12 CPU / 24 GB / 1 h, afterok). Quota before submission: GPU 106 h + baseline remainder 2.2 + 20 planned = 128.2 vs 136.8; CPU 799 + 174 vs 1211; mem 6.73M + 0.51M vs 10.1M MB-h. Comparison caveat: the adapter numbers of unified_rerun (cube 163/200, puzzle 80/200, scene 180/200) used the pre-fix u_events; a matched adapter rerun with the current source does not fit this month's GPU cap.

## Scene memory v1: learned visual event front-end (2026-10-07, user-authorized)
- Direction agreed with the user: learned scene representation + rule-based event boundaries (report it as such), interaction code c postponed, fixed-H main control = subgoal-conditioned skill on memory state after H steps (the "same command, H boundaries" arm is only a limited ablation). Search key = full code grid + knownness; goal test = only goal tokens judged observed; all-unknown goal -> "insufficient information". Visibility is a separate learned head (synthetic occluders), not alpha. Late-observed effects attach to the previous interaction or stay "uncertain". Skill release clipped before the next interaction as well as at episode end.
- New source: scripts/sm_model.py (16x16 token memory, GateL0RD-style L0 gate, FSQ levels 5^6, fast layer alpha/A from the current frame only, visibility head), scripts/sm_train.py (reconstruction weighted by per-pixel temporal std, lam_alpha 0.01, lam_gate 0.04, synthetic occluders with clean/occluded memory invariance), scripts/sm_diag.py (export codes/gates/alpha/vis; PRIVILEGED probes: buttons, cube keypoint vs pixels, arm R^2, joint R^2; code stability at rest, distinctness across rest periods, synthetic occlusion test), slurm/sm_pipeline.sh, tests/test_scene_memory.py (FSQ round trip, closed gate copies codes exactly, export shapes, CPU training + diagnostics on a synthetic cache: all pass locally).
- Quota at 17:20 ICT: account GPU 106 h; 5th GPU 152 h -> ceiling 136.8 h. Peer session jobs of this account (ew_gtrain x3, ew_geval x6, ew_basecrl) hold ~22 GPU-h at their time limits, so ~8.8 GPU-h remain for new work this month.
- 57994 ew_smsmoke: sm_pipeline.sh, cube-triple, STEPS 500, 100 train episodes, no train export; main/mig 1 GPU, 6 CPU, 48 GB, 30 min. Purpose: measured s/step, GPU memory and export throughput before sizing the three family runs. Not a result. Planned 0.5 GPU-h keeps 106 + 22 + 0.5 <= 136.8.
- Local (Windows RTX 5070, not cluster; no quota) scene run at cluster scale, `local/run_sm.ps1` (STEPS 20000, 1000 episodes, `--accum 2` = 2 micro-batches of 8, `--mmap`, resume every 1000): `E:\jepa-data\event_wm\scene_memory_v1\scene_20261007195148`. Attempt 1 killed by the local RAM watchdog at step ~5400 (49 min; mmap pages grew the working set to 13.6 GB), resumed from step 5000. **Stopped by hand at step ~6000: gate collapse.** Train `open` = 0.0 at every log from step 500 (val_open 0.0 at 2000/4000); step-5000 checkpoint on 3 full val episodes (3003 frames, 19 button toggles, cube moved 8-14 cm): 0 gate openings, 0 frames with any code change, memory = the initial read for all 1001 frames; the per-frame fast layer (alpha ~5-6% of pixels) explains every change, persistent ones included (composite MSE 4e-4..2e-3, memory-only scene MSE 7e-3..1.8e-2). The 200-step smoke diag showed the same (changed_pairs_codes_differ 0.0, probes at chance). Mechanism: g = relu(tanh(s)) gives zero reconstruction gradient to a closed gate; only the L0 straight-through term (pushing s down) reaches it, so once all gates close (by step 500) they cannot reopen. Not evidence about learned event boundaries: the memory never got to make any. resume.pt (step 6000) kept.
- Fix (opt-in, cluster defaults unchanged): `SceneMemory(gate_st=True)` / `sm_train --gate-st`: gate value backward through tanh(s) also where relu clips it (forward identical, closed gate still copies the code exactly; new test `test_gate_st_same_forward_and_gradient_through_closed_gate`, 5/5 tests pass on CPU). `--mmap` now remaps the frame file every 500 steps (same data/RNG; keeps the working set from growing to the file size). `run_sm.ps1 -GateSt`. Check run: scene, 5000 steps (cosine over 5000), other hyper-parameters unchanged, diag on 100 val episodes, no train export: `E:\jepa-data\event_wm\scene_memory_v1\scene_gatest_20261007205954`, started 20:59.
- gate-st check run COMPLETED (train 46.8 min, peak working set 5.7 GB after the remap fix; diag 2 min). Gates reopen (train open 0 at 1000-1500, then 1e-4..7e-3; val_open 4.75e-3 at 5000; v1: exactly 0). Diag on 100 val episodes (PRIVILEGED scoring): open rate rest .0111 vs non-rest .0124 (not selective); rest frames with any code change .020; consecutive rest periods with different true state -> codes differ .435 (v1 smoke 0.0), unchanged -> identical .727 (n 33); occluder test: codes equal after reveal .997, visibility hidden .047 vs other .987. Probes: buttons acc .51/.49 (chance), drawer/window MLP R^2 -.42/-.39, cube keypoint 10.0 cm vs pixels 9.9 cm (the keypoint probe also fails on pixels, so it is uninformative). New `scripts/sm_event_align.py` (memory code-change frames vs PRIVILEGED object-change segments, +-5 frames; control = raw pixel frame difference thresholded to the same number of firing frames): memory fires 28.7/1000 frames, median 1 changed token, precision .63 vs pixel .53; segment recall memory / pixel: cube .18/.12, drawer .49/.89, window .50/.13, buttons .12/.03. panels.png: the memory scene is the empty room (no cube, no button lights, no red handles, drawer shown closed) even at the initial read; every object is carried by the per-frame fast layer. Diagnosis: optimisation lock-in, not the cost balance: where alpha ~ 1 the memory gets no reconstruction gradient, so it never learns objects although a static object is free in memory (initial read) and costs lam_alpha per frame in the fast layer. Not a usable event front end yet.
- Next iteration (opt-in `--w-mem`, default 0): memory-only reconstruction w_mem * weighted (scene - x)^2 on the clean run. w_mem 0.1 chosen so a per-frame gate on the moving arm still costs more than it gains (~1.1e-6 gain vs 5.0e-6 gate cost per token-frame, rough estimate) while a persistent change pays back within a few frames. Matched to the gate-st run (same 5000 steps, seed, data): `E:\jepa-data\event_wm\scene_memory_v1\scene_gatest_wmem01_20261007215112`, started 21:51.
- gate-st + w_mem 0.1 run COMPLETED (train 46 min, diag 2 min). Opposite degenerate solution: train/val open .84 at step 5000 (1.0 at 2000), alpha .011; diag: open rate rest .839 = non-rest .838, every rest frame changes codes (25% of tokens per frame, median 61 changed tokens per frame), unchanged pairs identical 0/33; event_align fires on every frame (trivial). The memory now holds the full state (buttons acc .999/.996, drawer/window MLP R^2 .998/.996; cube keypoint 9.4 cm vs pixels 9.9, probe still uninformative) but also the arm (arm R^2 .98/.96/.91/.94 on 4 of 6 joints); panels: memory scene contains lights, red handles, open drawer and a blurred arm; the fast layer keeps mostly the small cube. Occluder test weaker (.93 hidden tokens equal, visibility hidden .14 vs other .98). Diagnosis: L0 straight-through surrogate sigmoid(4s) has ~zero slope for a far-open gate, so the gate cost (~.034, ~15x the reconstruction loss) cannot close open gates; mirror image of the closed-gate dead zone fixed by gate-st. Representation capacity is not the limit; the sparse gate is.
- Fix (opt-in, default unchanged): `--gate-l0 softplus` (SceneMemory gate_l0): surrogate softplus(4s)/4, derivative sigmoid(4s): constant push on open gates, none on closed gates; forward unchanged. New test `test_gate_l0_softplus_pushes_far_open_gates`; 6/6 tests pass. Matched run (gate-st, w_mem 0.1, softplus, 5000 steps, same seed/data): `E:\jepa-data\event_wm\scene_memory_v1\scene_gatest_wmem01_l0sp_20261007224323`, started 22:43.
- gate-st + w_mem 0.1 + softplus run COMPLETED: still the all-open solution. Train open .94-.99 for steps 1000-4000, .57-.67 at the end (with a loss jump at 4500: rec .001 -> .0037); val_open .62 at 5000. Diag: open rate rest .788 vs non-rest .775; 35% of tokens change per rest frame (median 84 changed tokens per frame); unchanged pairs identical 0/33; buttons .98/.97, drawer/window R^2 .98/.97, arm R^2 .96/.94/.87/.89; occluder test .35 hidden tokens equal after reveal (memory rewrites every frame). So the estimator was not the main reason the gates stay open: per token-frame, w_mem 0.1 puts arm-in-memory (gate ~5e-6) close to arm-in-fast-layer (alpha ~1.2e-6 + stale-memory w_mem penalty ~2-3e-6, rough estimate) and the gradient keeps the arm in memory. Event base rate for reference: firing on every frame gives precision .48 (gate-st run: .63, rate-matched pixel diff .53).
- Bisection point (the one value the same cost arithmetic says favours the intended split on both sides: arm ~1.7e-6 in the fast layer vs 5e-6 gate; a persistent change pays the gate back after ~3 frames): gate-st + w_mem 0.02 + softplus, 5000 steps, same seed/data: `E:\jepa-data\event_wm\scene_memory_v1\scene_gatest_wmem002_l0sp_20261007233352`, started 23:34. All four 5000-step settings are tuned on scene VAL (development); none is a held-out result.
- gate-st + w_mem 0.02 + softplus run COMPLETED (bisection): still all-open. val_open .93 at 5000; diag open rate rest .930 = non-rest .930, every frame changes some code (5.7% of tokens per rest frame, median 12 per frame), unchanged pairs identical 0/33; buttons .99/.99, drawer/window R^2 .994/.994, arm R^2 .97/.94/.87/.80; occluder .68. In every run most "open" gates do nothing in the forward pass (this run: 93% open vs 5.7% of tokens changing code; gate-st run: 1.1% open vs .01%): a partial update g * (u - q) smaller than half an FSQ step rounds back to the same code, while the straight-through gradient still credits it. Hypothesis (not isolated by an experiment): this phantom credit, not the intended cost balance, decides the open rate.
- Rule-based boundary check on the four exports, no training (`sm_event_align.py --debounce 10`: event frame = some token starts a new value held >= 10 frames that differs from its previous held value; control = same rule on raw-pixel tokens, 4x4 patch mean RGB, 8 levels/channel). Precision at chance level is .48 (fire every frame). gate-st memory: 22 fires/1000 frames, precision .57, recall cube .17 / drawer .47 / window .49 / buttons .12. w_mem .1, w_mem .1 + softplus, w_mem .02 + softplus memories: 526-724 fires/1000, precision .42-.49, recall ~1. Pixel tokens: 556/1000, precision .46, recall ~1. A learned memory that contains the objects also contains the arm, and then neither the learned gate nor a persistence rule beats raw pixels. Crux for this direction = separating the agent (arm) from object state without labels, not the object-state representation itself.
- Status: scene_memory_v1 does not yet give events in any of the 4 local configurations (development on scene VAL, PRIVILEGED scoring only). No further runs launched; next design step needs a user decision (see session report 2026-10-08).
- Protocol fact (verified in ogbench 1.2.1 source, 2026-10-08): visual manipulation observations are the 64x64 RGB render only (`compute_observation`, pixels branch); proprio is concatenated only in state mode. The visual .npz files also hold qpos / qvel (/ button_states), but `make_env_and_datasets(add_info=False)` does not load them except for singletask / oraclerep variants. The step `info` dict carries proprio and qpos (not used by baselines). The goal is one rendered image with the arm at a freshly initialised pose (scene_env.py: initialize_arm + 2 random steps, then compute_observation). Pixel mode draws the arm almost transparent (alpha .1) and the gripper purple. Consequences: dataset actions (offline) and the agent's own past actions (online) are the same information every OGBench method has; proprio is not. The goal reader must work on a single frame without action history.
- Design note: scene_memory_v1's memory vs fast layer with an area cost (lam_alpha) has the same structure as the layered slow map (deep robust-PCA, 57195 / 57199) stopped on 2026-10-04 because "the mask cost scales with area, so small objects always go to the sparse part". The gate-st run reproduced that failure (memory = empty room, cube / lights / handles in the fast layer). Further work on the front end should not rely on an area-cost competition between a per-frame layer and the scene state.

## Scene memory v2: action-contingent agent mask + per-patch discrete codes + rule-based memory (user-approved 2026-10-08)
- Decision with the user (2026-10-08): keep the agreed direction "learned scene representation + rule-based event boundaries"; separate the agent by action contingency (pixel analogue of g_entities.py's agent rule "the agent responds directly to the action"; prior art: contingency-aware exploration, Choi et al. ICLR 2019; Iso-Dream, NeurIPS 2022; robot self-detection by motor contingency), using only what every OGBench method has (offline observations + actions; the agent's own past actions online). Proprio is not used. No per-frame layer with an area cost (the robust-PCA family stopped on 2026-10-04).
- Source (new, v1 files untouched): scripts/sm2_model.py (AgentNet next-frame predictor with action dropout; teacher rule `contingent`: token pixels change (exact, deterministic rendering) and R = clip(1 - e_action / e_free, 0, 1) above its 2-means split over changing TRAIN tokens, 0.5 if not bimodal; Segmenter one frame -> per-token agent logit; SceneCodes per-patch encoder (receptive field = its 4x4 patch, no cross-position normalisation, so unchanged pixels keep their code exactly) + FSQ 5^6 + decoder with context; `agent_masks` = teacher pairs (t-1,t)|(t,t+1) | segmenter p > .5, dilated 1 token; `memory_rule` k = 3 consecutive agent-free frames with the same code; `merge_onsets` as slowmap.py), scripts/sm2_train.py (stages agent / label / seg / segpred / scene, resume for the long ones; reconstruction weighted as sm_train.py and counted only outside the agent mask), scripts/sm2_diag.py (VAL: memory over learned codes and over raw-pixel tokens with the same mask and rule as control; pre-registered 2026-10-04 criteria c1-c4 with the slowmap.py / slowmap_probe.py reference definitions, window 15, gap 10; stability / distinctness; event lag after the end of the object change; agent-mask diagnostics incl. arm R^2 from the mask; panels; VAL export mem/changed/agent/codes), tests/test_sm2.py (memory rule, merging, per-patch locality, dilation, CPU smoke of all stages + diag: 5/5 pass), local/run_sm2.ps1.
- Synthetic check of the teacher (CPU, 20 episodes, action-driven blob + a block that jumps once, AgentNet 2000 steps): contingent fraction on blob tokens .41 per pair (only its moving edges change), static block .002, block at its jump 0 (a non-action change is excluded), elsewhere .0025. A first version (2-means on absolute gain) marked the static block 42% of the time (edge noise); a second (model error as "changing", unclipped R) put the split at R = -9.9 (outliers). Both replaced before any real run.
- Scene run (local RTX 5070): `E:\jepa-data\event_wm\scene_memory_v2\scene_20261008085549`, started 08:55. AgentNet 10k steps in 3.5 min: val error action 3.3e-4 vs action-free 4.9e-4; median R on the top-5% changing tokens .37 (rising: .20 / .34 / .37 at 4k / 6k / 8k). Teacher fit: 25% of tokens change per pair; R quantiles over changing tokens 10/25/50/75/90% = 0 / 0 / .18 / .42 / .61; split .307 (Ashman D 3.38); 8.9% of tokens contingent per pair.
- Scene run 1 COMPLETED in ~15 min total (agent 3.6, label 3.2, seg 1.5, segpred 0.7, scene 4.2, diag 0.8 min); peak working set 3.9 GB. Seg vs teacher on VAL recall/precision .77/.76; AE 15k steps val loss 2.3e-4, 5122 distinct codes on VAL. Mask (teacher | segmenter, dilate 1) 31.5% of tokens; arm R^2 from the mask .97/.89/.89/.84. VAL, PRIVILEGED scoring, learned codes: c3 agent-free PASS (arm R^2 max .196; v1: .8-.98); c2 nominal PASS (keypoint ratio .89, but the probe itself is weak: pixels 10.6 cm); c1 FAIL (buttons .917/.974; settled .914/.957); c4 FAIL (w15 recall .82 precision .77; w30 .96/.98; event start - reference end p10/p50/p90 = -34/-12/+24 frames); drawer/window R^2 .91/.94; memory changes in 46% of object-rest frames (~1 token per frame). Analysis: 96% of rest changes lie within 2 tokens of the agent footprint of the last 6 frames (31% under it, 47% at 1, 18% at 2), a third return to the previous value within 100 frames: the faint transparent arm / shadow fringe leaks past the mask (the teacher only marks tokens whose change the action explains, R > .31). Matched control (raw-pixel tokens, same mask and rule) was BETTER than the learned codes: w15 .93/.78; per-type recall w15 cube .95 vs .79, buttons .93 vs .87, drawer .94 vs .94, window .84 vs .65; buttons .95/.99. So far the mask and the memory rule do the work, not the learned codes.
- Scene run 2 (only change: teacher grown through changing tokens; same AgentNet copied from run 1): `E:\jepa-data\event_wm\scene_memory_v2\scene_grow_20261008091629`. Teacher 24.4% of tokens per pair; segmenter vs teacher .96/.95; arm R^2 from the mask .99/.96/.94/.87; AE val loss 8e-5. Current-frame raw-pixel probe ceiling (contains the arm): buttons .9997/.9995, drawer/window R^2 .999/.999, arm .98-.995.

| scene run 2, VAL | mask | rest frames w/ change | events w15 rec/prec | w30 | buttons (settled) | drawer/window R^2 (settled) | arm R^2 max |
|---|---|---|---|---|---|---|---|
| learned, dilate 1 | 44% | 11.8% | .884/.813 | .971/.970 | .84/.97 (.87/.98) | .63/.59 (.72/.68) | -.31 |
| learned, dilate 0 | 28% | 28.5% | .939/.772 | .995/.973 | .93/.98 (.93/.98) | .90/.95 (.90/.95) | .18 |
| pixel tokens, dilate 1 | 44% | 11.8% | .891/.818 | .970/.968 | .90/.98 (.92/.98) | .68/.56 (.77/.66) | -.22 |
| pixel tokens, dilate 0 | 28% | 29.8% | .915/.768 | .991/.972 | .95/.99 (.97/.98) | .94/.96 (.94/.96) | .40 |

- Per-type recall w15 (learned, d1 / d0): cube .72/.98, buttons .94/.97, drawer .99/.92, window .99/.82. Events per episode 37 (d1) / 51 (d0) vs 18.7 reference segments: one interaction gives several memory events (e.g. a pick shows "cube gone from A" and later "cube at B"; tokens of one object become visible at different times). Settled = object rest and no object change in the previous 30 frames. Trade-off: a larger mask cleans the memory but leaves objects under the shadow stale for long; a smaller one updates in time but lets the fringe in. Root cause: the arm is drawn ~90% transparent and casts a large shadow; the scene stays visible through them, only tinted. Criteria (pre-registered) not met on scene in any variant: c1 (per-frame probe) and c4 at window 15.
- Cube-triple and puzzle-4x5 caches made locally (1000 train / 300 val episodes, 100 val used). Same-hyper-parameter runs (growth on, AE dilate 1; diag at dilate 1 and 0) launched 09:34: cube then puzzle, log `E:\jepa-data\event_wm\logs\run_sm2_cube_puzzle_20261008093455.log`.
- Cube-triple `E:\jepa-data\event_wm\scene_memory_v2\cube_grow_20261008093456` and puzzle-4x5 `...\puzzle_grow_20261008095020` COMPLETED (same hyper-parameters, ~15 min each). The teacher split self-fits almost identically on all three families: R threshold .307 / .308 / .332 (scene / cube / puzzle), Ashman D 3.38 / 3.46 / 3.49; grown teacher 24 / 22 / 25% of tokens per pair; segmenter vs teacher .96/.95, .94/.93, .97/.96; AgentNet median R on the top-5% changing tokens .37 / .40 / .49.

| VAL, learned codes | mask | events w15 rec/prec | w30 | events/episode (reference) | rest frames w/ change | arm R^2 max | discrete min (settled range) |
|---|---|---|---|---|---|---|---|
| cube, dilate 1 | 42% | .915/.636 | .981/.995 | 42.1 (11.8) | 10.1% | -.28 | n/a |
| cube, dilate 0 | 26% | .932/.710 | .991/.995 | 37.8 | 39.8% | .43 | n/a |
| puzzle, dilate 1 | 44% | .860/.918 | .985/1.0 | 45.7 (29.6) | 15.6% | -.05 | .62 (.57-.96) |
| puzzle, dilate 0 | 28% | .945/.937 (c4 met) | .998/1.0 | 53.4 | 25.2% | .11 | .82 (.77-.98) |

- Raw-pixel-token control with the same mask and rule is again about equal (cube d0 w15 .981/.707; puzzle d0 .940/.944, lights min .83). Current-frame raw-pixel ceiling: puzzle lights min .9996, arm R^2 .98-.999 (cube .95-.99). Puzzle lights are read worst where the arm works most (settled accuracy per light .57-.96 at dilate 1): the memory waits for agent-free views, but the scene stays visible through the ~90% transparent arm (cf. the puzzle SFA/ICA code, which read all 20 lights at >= .989 from current frames). Cube keypoint probe (c2) is unreliable here: the same probe on raw pixels gives 7-15 cm median error vs 4.1-4.3 cm in 57184, so c2 is not judged.
- Summary of v2 against the pre-registered criteria (same hyper-parameters on three families): c3 agent-free met on all three at dilate 1 (v1: failed); c4 met on puzzle at dilate 0, not met at window 15 elsewhere (recall .86-.95, precision .64-.94), while at window 30 recall .97-1.0 and precision .97-1.0 everywhere: events are found, but late (the rule waits for the agent to leave) and fragmented (1.3-3.6x the reference count); c1 not met (scene buttons .84-.93, puzzle lights .62-.82; current-frame ceiling .9996): staleness under the transparent arm / shadow; learned per-patch codes do not beat raw-pixel tokens yet. No closed-loop result; all numbers are VAL development with PRIVILEGED scoring.
- See-through stage (user-approved 2026-10-08): `SeeThrough` (frame -> per-token code digits with context, sm2_model.py) trained on self-generated targets (`see_targets`: agent-free frames give their confirmed code; agent-covered frames of a visit after which the memory confirms the same code as before give that code; no labels). Same hyper-parameters on all families (10k steps, width 128, 2 code spaces: learned 5^6 and raw-pixel 8^3). Targets: 44% of token-frames agent-covered (dilate 1), 46% of those get a see-through target (scene; cube / puzzle similar). Scene VAL at step 2000: clean tokens .987 code accuracy, covered-unchanged tokens .962 (.980 when p > .5). Run dirs: the `*_grow_*` runs above, outputs `train/seethru_{learned,pixel}.pt`, `diag_see/`.
- Memory with see-through codes (trusted = p > .5, no agent mask), VAL: fresher state and earlier events, but more false changes under the hovering arm.

| VAL | variant | events w15 rec/prec | w30 | rest frames w/ change | arm R^2 max | state probes (settled) |
|---|---|---|---|---|---|---|
| scene | masked learned | .884/.813 | .971/.970 | 11.8% | -.31 | buttons .84 (.87/.98), joints .63/.59 (.72/.68) |
| scene | see learned | .932/.793 | .994/.973 | 29.6% | .12 | buttons .96 (.93/.995), joints .86/.88 (.88/.95) |
| scene | see pixel | .907/.785 | .981/.970 | 30.6% | .22 | buttons .97 (.996/.998), joints .93/.91 (.95/.95) |
| cube | masked learned | .915/.636 | .981/.995 | 10.1% | -.28 | (keypoint probe unreliable) |
| cube | see learned | .983/.709 | .999/.995 | 32.5% | -.11 | |
| puzzle | masked learned | .860/.918 | .985/1.0 | 15.6% | -.05 | lights min .62 (settled min/mean .57/.83) |
| puzzle | see learned | .983/.969 (c4 met) | 1.0/1.0 | 26.3% | -.03 | lights min .81 (settled .91/.99) |
| puzzle | see pixel | .987/.973 (c4 met) | 1.0/1.0 | 27.5% | .36 | lights min .89 (settled .97/.99) |

- False changes of the see-through memory at rest: 71-73% lie under the agent footprint of the last 3 frames, 25-27% one token away; ~46% return to the previous value within 100 frames (confident see-through errors while the arm hovers). Assembly (`assemble_events`, gap fitted on 100 TRAIN episodes by 2-means on log gaps: 26-51 frames, Ashman D 2.9-3.2) over-merges through these chains (scene see-learned 11 assembled events per episode vs 18.7 reference). Next: events only from changes confirmed after the agent left (memory over agent-free frames with see-through codes), backdated to when the fresh see-through memory first showed the confirmed value (`backdate`); state read from the fresh memory. Diag rerun `diag_conf/` launched 11:03.
- Confirmed events (`conf_*` variants, `diag_conf2/`): events = changes of a memory over agent-free frames with see-through codes (mask | p <= .5), backdated (`backdate`) to the first frame after the last agent-free confirmation of the old value at which the fresh see-through memory showed the confirmed value; state = the fresh see-through memory. Backdated share of confirmed changes: scene .84, cube .67, puzzle .98 (learned codes). A first version backdated to the start of the last fresh run (`diag_conf/`) gave the same event metrics within .01.

| VAL, conf_learned (events) + fresh see-through state | events w15 rec/prec | w30 | events/ep (ref) | false-change rest frames (events / state memory) | arm R^2 max | state probes |
|---|---|---|---|---|---|---|
| scene | .893/.833 | .968/.976 | 40.6 (18.7) | 12.3% / 29.6% | .12 | buttons .956 (settled min/mean .934/.965); drawer/window .86/.88 (settled .88/.95) |
| cube | .950/.667 | .988/.996 | 45.8 (11.8) | 9.4% / 32.5% | -.11 | c2 keypoint probe unreliable |
| puzzle | .927/.961 (c4 met) | .990/1.0 | 52.0 (29.6) | 15.4% / 26.3% | -.03 | lights min .815 (settled min/mean .908/.989) |

- Per-type recall w15 (scene conf_learned): cube .72, buttons .97, drawer .98, window .99 (fresh see-through events: cube .98, drawer/window .89; open question why backdating does not recover the cube timing). Interaction assembly is not reliable yet: assembled events per episode 22 / 19 / 18 vs reference 18.7 / 11.8 / 29.6; span recall/precision w15 .71/.88, .76/.78, .49/.96 (w30 .89/.98, .90/1.0, .64/1.0). Raw-pixel-token codes with the same machinery: state probes higher (scene buttons settled .996/.997, puzzle lights settled .973/.993) but the arm leaks (R^2 .22 / .13 / .36), events about equal.
- Status after v2 (dev, VAL, PRIVILEGED scoring, one seed, local, identical hyper-parameters on three families; no closed loop): c3 agent-free met everywhere with learned codes; c4 at window 15 met on puzzle, scene .893/.833 (recall .007 short), cube .950/.667 (fragmentation); at window 30 recall .97-.99 / precision .98-1.0 everywhere; c1 not met (scene buttons .956, puzzle lights min .815); c2 not judged (probe). Report: docs/scene_memory_v2_20261008/REPORT.md.

## Scene memory v2 -> event-WM backend: events, state reader, puzzle closed loop (user-approved 2026-10-08)

All local (Windows RTX 5070, `local/run_stage.ps1`), dev VAL, PRIVILEGED scoring only where stated; no held-out claims yet. Same hyper-parameters for every family; `--family` only selects the privileged scoring.
- `scripts/sm2_events.py` (label-free): events = temporal clusters of backdated confirmed changes on dynamic tokens; state bits = fresh see-through memory on dynamic tokens; thresholds are 2-means splits on TRAIN. Puzzle (`puzzle_grow_20261008095020/events`): 47 dynamic tokens (1 bit each, all 20 lights), interaction gap 14.4 frames, 26 co-change objects, 23.1 events/episode (reference 29.6 presses), pattern vocabulary 813 types (in-vocab .79).
- Acted-object cue for state-dependent families (PRIVILEGED check, puzzle VAL single presses, chance .26): teacher distance .30, segmenter coverage .45, teacher fraction .45, first change .18, AgentNet action sensitivity over 6 frames before onset .73 (xy) / .71 (all dims) -> `sm2_events.py --types acted`.
- `scripts/sm2_reader.py` (CNN state reader on one raw frame, BCE on pseudo-labels; events from its bits as build_events.py). v1 `--labels segment` (`reader/`): labelled 15%, lights (linear map, PRIVILEGED) mean .944 / min .82, all-20 frames .46; 15.5 events/episode, span w15 recall .63 precision .99; 1615 pattern types.
- Pseudo-label candidates vs lights (PRIVILEGED, VAL): fresh memory .930, confirmed .839, backdated confirmed .901, end-of-interval .913, per-frame SeeThrough bits without memory .961 (min .945, all-20 .585): the memory's p-gating only adds lag -> v2 `--labels smooth` (per-bit majority of per-frame SeeThrough bits over +-7 frames, kept where every bit agrees >= .8).
- **Reader v2 (`reader_smooth/`, 10.8 min): lights mean .978 (min .959), all-20 frames .73; 31.3 events/episode vs 29.6, span w15 recall .972 precision .988 (w30 .988/1.0); 384 pattern types, type -> pressed-button purity .967, all 20 buttons covered.**
- Error analysis (PRIVILEGED): 44.7% of VAL frames lie inside reader event spans (median 11 frames), where all-20 is .53; at >= 4 frames from an event all-20 is .91-.96 (per light .992-.997). Per-segment majority of the reader's own bits gives the same accuracy as the raw bits (errors persist within a segment), so a segment self-training round was not run. 38% of single-press events carry a pattern other than the dominant one for their true toggle set, 68% of those one bit off.
- `sm2_reader.py --vocab modes` (dev change after the puzzle VAL analysis above; default stays `patterns`): every pattern climbs to its most frequent Hamming-1 neighbour (TRAIN counts) until none is more frequent; types = modes above the 2-means split of log mode counts over TRAIN events. Puzzle (`reader_smooth_modes/`, same reader bits via `--reuse`): 2628 patterns -> 1275 modes -> **37 types covering 89% of TRAIN events**; VAL typed .894, all 20 buttons covered, purity mean .906 (min .25); types per button mostly 1 (buttons 0/1/4/15/19: 3/2/5/4/8). Unit test `test_pattern_modes_absorbs_one_bit_variants`.
- `scripts/sm2_loop.py`: closed loop (reader -> BWAS in the event WM + cost-to-go of `sm2_planner.py` -> skill v3 of `sm2_skill.py` -> debounced, WM-consistent event detection -> replan); PRIVILEGED arms `--low scripted` / `--high oracle`. Smoke with random-weight models: 0.3 min per 1000-step episode; scripted presses detected one-to-one (5-12 bits per press).
- Frozen-config check (no change after the puzzle runs): `sm2_events` + `sm2_reader --labels smooth` on scene and cube queued 13:15 (`logs/run_sm2_events_reader_sc_20261008131531.log`). Puzzle backend chain queued after it (`logs/run_sm2_puzzle_backend_20261008133328.log`): planner -> loop with scripted low level (PRIVILEGED attribution) -> skill v3 -> loop with the learned skill, dev seed 0, 6 episodes x 5 tasks.
- **Frozen-config check result (13:15-13:48, no change after the puzzle runs): the reader stage does not transfer.** `sm2_events` (memory, no reader) on VAL, PRIVILEGED span scores: scene 14.1 interactions/episode (reference 18.7), w15 recall .70 precision .92 (w30 .86/.99; per type cube .71, buttons .67, drawer .76, window .71); cube 15.1 (11.8), w15 .81/.93 (w30 .92/1.0), 65 pattern types (top type 31% of events); puzzle 23.3 (29.6), w15 .81/.996. `sm2_reader --labels smooth` (frame mask): scene labelled 4.9% of frames (89 dynamic tokens, one-hot bits), 6.1 events/episode, w15 recall .27; cube labelled 0.14%, 6.6 events/episode, w15 recall .47. Cause: the frame mask requires every bit to agree, which almost never holds with many bits; the setting was chosen on puzzle (47 bits). Generic fix queued for all three families after the puzzle backend chain (`logs/run_sm2_bitmask_c2_20261008134206.log`): `--label-mask bit` (each bit labelled where it agrees, masked BCE) + `--vocab modes`; then `sm2_diag` with the c2 MLP probe (cube, scene).
- `sm2_diag.py` c2: DEVIATION from the pre-registered keypoint probe, documented in the script: c2 is judged with an MLP (2 x 512) on 8 x 8-pooled features for the memory and raw pixels (keypoint probe on raw pixels 3.5-15 cm over seeds / lengths on the same cube data, MLP 4.4-5.6 cm); keypoint numbers still reported.
- Carry signal for pick/place pairing (PRIVILEGED check, cube VAL; truth = some cube moving > 2 mm/frame, 45% of frames): segmenter agent area 61.9 tokens while a cube moves vs 63.4 otherwise, teacher area 55.8 vs 57.1, so the agent mask does not reveal carrying; pairing needs another signal (open).

## Bit backend cancelled; general method folder `method/` (user 2026-10-08)
- User: the bit planner would need code / method changes for cube and scene, so cancel it; the general version must be good on every task, otherwise review and optimise the method. Cancelled (local processes stopped): the puzzle bit planner (h step ~15k of 100k), the queued bit loop / skill / loop and the bit-mask readers. The c2 diag reruns were kept.
- Investigation (user question: was a general backend tried before?): yes, the unified entity backend (`u_events`, `u_wm`, `u_skill`, `u_closed_loop`, support prior; latest frozen source `docs/generic_state_20261007/base_source`). STATE adapter: cube 163/200, puzzle 80/200 -> 200/200 after the event-label fix, scene 180/200; generic STATE front end `g_entities.py`: training / eval jobs 57979-57987 submitted, results not recorded locally; PIXEL via SAM 2 entities: cube-triple 22/30 (dev seed 0), puzzle no loop number, scene not run. The scene memory had never been connected to it; the bit branch had reused the old puzzle pipeline instead.
- `method/` (single source of the method, README lists components 1-17, sources with sha256, changes, what is not part of the method): `frontend.py` / `frontend_train.py` (identical definitions to scripts/sm2_*), `memory_entities.py` (new: every token whose agent-free memory changes in TRAIN is a fixed-place entity; no identity extraction), `events.py` (port of u_events --per-frame), `world_model.py` (port of u_wm: D from data, unknown goal entities, frequent-code prototypes, candidates ranked by the support prior, stages wm / h), `event_support.py`, `train_support.py`, `skill.py`, `skill_segments.py`, `closed_loop.py` (online front end), `run_family.ps1`. `tests/test_method.py` (6 pass): vectorized rest runs and transitions equal the verbatim u_events functions on random data.
- Changes forced by token entities (evidence local): appearance thresholds (>= 99.9% of consecutive agent-free readings identical, 40-50% of real changes 1 digit-step; the u_events split of non-zero differences would drop ~70% of real changes -> quantized rule r_app = step / 4, thr_app = step / 2); transitions read every frame as u_events does (a readability test stretched windows to p50 / p90 16 / 29 frames and gave 11.3 puzzle events / episode for 29.6 presses; without it 1 / 13 frames).
- Entities (tokens with a confirmed change in TRAIN): puzzle 50, scene 116, cube 152.
- **First results, same settings for all families (VAL, PRIVILEGED scoring, sm2_diag reference):** events / episode puzzle 27.5 (presses 29.6), scene 20.8 (reference 18.7), cube 10.4 (11.8); span w15 recall / precision puzzle .76 / .99, scene .64 / .84, cube .46 / .87 (cube: an event starting near the pick .31, near the place .28); puzzle presses in clean one-to-one events .57, acted token -> pressed button .56. Puzzle world model on these events (30k steps): acted entity within tol .988, side effects .48, unchanged entities .93, all 50 entities right 1.9%. c2 with the MLP probe (all frames): memory cube xy error 9.6-11.7 cm vs pixels 3.8-5.3 cm (cube), 4.8 vs 1.6 cm (scene), c2 not met (carried cubes are left out of the memory by design; a rest-frame check is still to do).
- Diagnosis: the token memory records many changes near the arm (cube VAL episode 0, frames 208-333: ~40 token changes over r6-r11 / c4-c13 around 2 true moves, windows up to 120 frames where the readings under arm and cube match neither rest code); overlap grouping then chains interactions, and the acted token is wrong on puzzle 44% of the time. Short A-B-A reversions are not the cause (90-96% lie near true interactions; middle state p50 107-175 frames). Longer rests (m 10 / 20) merge more (cube 8.7 / 7.6 events per episode).
- **Interaction assembly (user-approved step 1, 2026-10-08 evening), local, VAL, PRIVILEGED scoring:** contact flags are uninformative on puzzle (99.7% of token changes have the agent within one token); space-constrained grouping splits Lights-Out crosses (puzzle 47.7 events / episode); the acted-token cues "most covered by the agent" / "occluded longest" are worse than u_events' "closest to the agent, then most central" (.46 / .37 vs .61 on single-press events). Complete linkage on change arrivals (`events.py --group span`) fixes the timing: w15 span recall / precision with G = 16 frames puzzle .94 / 1.0, scene .91 / .92, cube .97 / .91 (overlap chains: .76 / .99, .64 / .84, .46 / .87), at 32.7 / 29.8 / 33.0 events per episode (references 29.6 / 18.7 / 11.8). G from data failed: 2-means of arrival gaps = 4.4-4.7 frames (62 events per episode), a label-free repeatability criterion prefers ever smaller events; G = 16 is a DEV CHOICE on VAL with privileged metrics, documented in events.py.
- c2 at rest (cubes still within +-5 frames, MLP probe, cm): cube memory 10.1-12.8, current SeeThrough code 7.9-10.3, raw pixels 4.1-5.8; unpooled 16 x 16 code grid: 10.5-12.9 / 9.1-10.2 / 4.7-6.1; scene 5.3 / 3.2 / 1.9. The learned per-patch codes lose about half of the cube position precision even at rest (front-end limitation, not the probe and not only carried cubes).
- Label check (puzzle, PRIVILEGED per-light decoders from the tokens that show each light): readings of tokens decode lights 100% right where readable (73% of light-frames, through the transparent arm) or readable and free (36%). Event labels: nearest-rest state (u_events) all 20 lights right 34% before / 42% after, toggled set 21%; change-window state rule 31 / 34%, 25%; + readable override 39 / 46%, 27%; + readable rest observations (`--rest-obs readable`) 56 / 56%, 41% (events per episode then 41.5 puzzle, 46.3 cube, 43.9 scene). `events.py` now uses the change-window state rule with the readable override (documented); rest observations stay readable + free by default.
- World models on span-16 events (old label rule, 30k steps, VAL): acted entity within tol / side effects / unchanged / all entities exact: puzzle .987 / .487 / .913 / .013; scene .820 / .359 / .971 / .008; cube .968 / .093 / .973 / 0.0 (overlap events: puzzle .988 / .480 / .931 / .019, scene .831 / .297 / .980 / .035). Token-entity world models do not learn side effects: on cube a pick or place changes several tokens whose exact codes depend on the sub-token position.

## Direction A: object entities from pixels -> u_events-style events (user-approved 2026-10-08/09; option (a): data-driven rules, held-out envs as the test)

All local CPU (no GPU stage yet), dev VAL, PRIVILEGED scoring only where stated; no held-out or closed-loop result yet.
- `method/objects.py` (component 7, object entities; every threshold a 2-means split or the majority 1/2):
  - changed pixel = leaves its typical (mode) colour bin with a chromaticity change > tau_c (shadows and shading are not changes; a brightening test made the arm's shadow look like an object);
  - PLACES = components of (masks of free rest instances whose exact centroid recurs in >= 2 episodes and in more than half of the episodes with any instance within one object width) | (pixels changed in more than half of their free observations), kept if PERSISTENT (flip rate between consecutive discovery frames < 1/2 of independent draws: the arm's unsegmented parts flicker);
  - MOVERS = untouched rest instances off places; K from the median per-episode maximum seen at once, grown while one more colour cluster halves the share of frames where one cluster is seen twice; clusters seen in < 1/2 of episodes dropped;
  - AGENT COLOURS: inside the (dilated token) agent mask a changed pixel is the agent's if its colour bin is more frequent inside than at >= w outside; other changed pixels are objects seen in the margin / arm shadow (movers and the stored agent mask use this);
  - place readings: full view (chromaticity, exact) or, under the agent, the SeeThrough codes of the place's token entities (`--tokens` = memory_entities.py output) looked up to the reading most often seen in full with those codes (2 code tuples per light on puzzle).
- Identity checks (VAL, PRIVILEGED): puzzle-4x5 20 places = buttons 0-19 in order, |corr| 1.0, visible 45-92%; cube-triple 3 movers (K by colour co-occurrence; the count alone said 2) at 1.5-1.9 cm, visible 75-88%; scene cube |corr| .996 (visible 57%), drawer .985, window .975, buttons 1.0, plus 3-4 one-pixel junk places (window fragments / a robot pixel).
- `method/events_objects.py` = u_events (sha 38fab593) + changes, each forced by tables that hide identities: identities observed only where the table shows them (unobserved frames formed rests at (0,0): cube 25 events/ep, acted cube moved in 0.4%); noise radii from consecutive OBSERVATIONS (across gaps); place readings need no agent-clear test; `--group stab` (minimum stabbing of change windows, no parameter; overlap chains gave 3-5 events/ep on puzzle); states read BETWEEN events with known masks (before/after_known, target_known); acted = among identities the agent touches at the contact moment, the nearest to the centroid of the known changes.
- Puzzle-4x5 VAL (PRIVILEGED per-press scoring): 30.6 events/ep for 30.6 presses; 74.7% of presses inside exactly one event core; pressed light = acted .895; over known (event, light) pairs label accuracy .9995, known changed set = known toggled set .991; 94% of (event, light) states known.
- Diagnostic arms (PRIVILEGED, not methods): perfect light readings everywhere -> 100% one press per event and correct sets; the remaining gap was reading through the translucent arm, place position noise, rest observations needing agent-clear, and sub-pixel distance ranking for the acted light (.26-.40).
- 2026-10-09 (local, dev VAL): two-state places snap to their states (2-means along the principal axis of full-view readings, Ashman D > 2; lookup readings snap the same way), so one state is one exact value; places whose changed pixels are as agent-coloured as the agent mask's own changed pixels are dropped (cube robot pixel, scene 1-px window fragments: shares .94-1.0 vs .72-.89 threshold), and mover instances likewise; mover K grows only on a halving of co-occurrence violations beyond 2 Poisson SD (cube-triple: 967 -> 2 -> 0 violating frames, K = 3). events_objects: identities with <= 8 distinct readings are exact (any change counts) and get an appearance unit of half their smallest value gap; the shared split runs over inexact identities only (with exact ones it fell between light colours: .48 for flips of .3). world_model.py: per-entity appearance units in the loss, prototypes and scores.
- Puzzle-4x5 (`method\objects`, `obj_events`, `obj_model`; VAL, PRIVILEGED where stated): pressed light = acted .896, known toggled sets .992, per-light labels .9995; world model acted .988 / side effects .915 / unchanged .978 / event exact .685 (token track .487 side effects; STATE track .98), 2 prototypes per light. `local/run_object_queue.ps1` started 02:54: puzzle support / h / skill / loop, then cube and scene full chains, then held-out cube-single, puzzle-3x3, puzzle-4x4 (front end via run_sm2.ps1, same settings, then the same chain). Held-out data: visual-cube-single-play-v0 downloaded with the user's approval (HF mirror ryanhoangt/ogbench_data, 1.85 GB); caches built for the three held-out envs.
- 2026-10-09: acted rule refined (touched AND not known to stay unchanged, then nearest the centroid of the known changes): VAL acted = true object, PRIVILEGED: puzzle .867 / cube .888 / scene .827 (was .878 / .887 / .681; scene button presses otherwise picked the cube or the window, the arm mask touches 3-4 identities). Scene per type: buttons .88 / .70, drawer .92, window .95, cube .58 with only 18% of its moves in one event core (the scene cube is hidden in the drawer / under the arm). Puzzle objects/events/model were built before this change (puzzle .878 vs .867 with the new rule); puzzle will be rebuilt with the final rule for the reported numbers.
- `world_model.Model.candidates_batch` (vectorized candidates, identical events and order: 0 mismatches on 300 VAL states x 2 known masks, local check) now used by the planner, the cost-to-go walks and its training loop: puzzle walk pool 35 -> 2.2 min, cost-to-go steps 2.4x faster (150k steps ~ 105 min). Queue restarted 03:36 (`logs/object_queue_20261009033625.log`).
- 2026-10-09 closed loop (object track, `method/closed_loop_objects.py`, same per-frame reading as the tables via `objects.read_frame`): smoke runs on CPU with a random skill complete 1000-step episodes. Goal image: place tokens read from any looked-up SeeThrough code (a static frame): dev puzzle goals (30 loop seeds, PRIVILEGED check) lights known .81 with p > .5 (min .70) vs .988 (min .90), all correct; cube-triple goals show .89 of the cubes (task 5 .50: stacked). Hidden-goal semantics: a mover unseen in the goal is a hidden goal (covered bit; `Model.at_goal` then checks only the bit, the heuristic ignores its position / appearance, no goal candidate); online, a mover unseen while the agent is clear of its last position is covered. Unseen entities start at their goal state until first seen. Scene tasks 4 / 5 put the cube in the closed drawer (hidden in the goal image), cube-triple task 5 stacks (lower cubes hidden).
- 2026-10-09 puzzle diagnosis (local, VAL, PRIVILEGED scoring only). Closed loop object track, 30 episodes (6 x 5 tasks): learned skill 0/30, spatial-conditioned skill 0/30, scripted press arm (PRIVILEGED diagnostic, 91% exact presses) 1/30; episodes end 4-15 lights wrong. The world model's errors are SYSTEMATIC per light, not noise: on clean single presses its predicted toggle set is right for lights 0, 9, 10, 12, 13, 16-19 (>= .98) but .00 for light 3 and .21 for light 2; light 15 had 24 training events for 1494 true presses, light 3 had 3699 for 1496. Cause: acted labels (pressed -> label confusion: 15 -> 3 / 1, 3 -> 8 / 4, 19 -> 14, 2 -> 1). Two reasons: (1) a two-state place snapped its appearance but not its position, so light 3 read (39.0, 25.0) in full view and (37.5, 24.5) through the SeeThrough lookup in one state, a 1.6 px "move" above tol_pos 1.33 that made spurious one-light events, cut the state windows of real ones (presses of 15 then had no known change) and joined the acted candidates; (2) presses whose lights are all hidden for 20 frames (bottom row under the arm). Event-exact rate of the round-2 model on clean presses: .79 with lights whose before-state is unknown, .93 over lights known before and after (the loop always knows its state). Relabelling by world-model explanation (`relabel_events.py`) lowers acted accuracy (.896 -> .820): it fits the systematic errors.
- 2026-10-09 state windows bounded by the earliest departures instead of the contact moments (`obj_events_v2`, against after-states showing the next press): per-light labels .9995 -> .9997 but toggled lights known .829 -> .747 and acted .896 -> .851; reverted. Offline acted-rule variants on the saved events (PRIVILEGED): touched at the contact moment / first arrival / any core frame / no touch filter: puzzle .867 / .849 / .885 / .885, cube .888 / .822 / .875 / .873, scene .827 / .815 / .827 / .669: no variant wins on all three; kept the contact moment.
- 2026-10-09 fix: `objects.py` two-state places snap the whole reading (appearance and position) to the mean of its side, in full view and through the lookup (`snap_reading`; tables built before keep the old 5-field snap and unsnapped positions). Puzzle rebuilt from objects (`method\objects`, `obj_events`; the previous outputs are `*_run2`).
- 2026-10-09 (cont.) puzzle with positions snapped per state (`*_run3`): spurious events gone (28.8 events/ep for 30.6 presses), presses in exactly one core .747 -> .788, light 15 acted .00 -> .81; per-frame light readings vs true button states (VAL, PRIVILEGED): 100% correct in full view (857k light-frames) and through SeeThrough codes (707k). But exact positions made the position noise radius 0.066 px (tol_pos .13 px): world model event-exact .45 (positions must be regressed within .13 px; appearance toggles right .92). Remaining label errors came from GROUPING: every light near a press is hidden by the arm (change windows 15-35 frames), and the first-moment stabbing (slack 2) put a light covered while the arm left for the next button into the previous press, moving the contact moment off the press (TRAIN target-known events: 17% had no press inside their core).
- fixes (dev VAL scored offline on the saved change windows, PRIVILEGED): no slack (`--frame-gap 0`) and each change assigned to the stab moment nearest its window centre (`events_objects.stab_groups`): presses / interactions inside exactly one core puzzle .789 -> .909 (cores with one press .837 -> .960), cube .711 -> .714, scene .612 -> .629. A place's position is its centre (fixed; its state is its appearance). Offline acted-rule variants (largest change first / known changes first) help cube (.888 -> .917) and scene (.827 -> .867) but break puzzle (.43 / .76); not adopted. Puzzle rebuilt from objects (run 4).
- 2026-10-09 run 4 (puzzle, grouping fix, VAL PRIVILEGED): presses inside exactly one core .908, acted .875 (top row: light 3 .23 -> mostly labelled 8: with light 2 unknown the centroid of {3, 4, 8} is nearest 8; corners 19 / 15: the refined agent mask at the contact moment is > half an object width from the light), round-1 model: clean presses all lights right .903. Per-frame light readings stay 100% correct; reading covered lights with ANY looked-up SeeThrough code (as for goal images) adds 33k readings at .898 accuracy, so the table keeps readable codes only (100%). Offline acted variants (touch radius one object width, touch duration over the core / span) do not win on all families (puzzle .884 / .801 / .381; scene .781 with the wider touch).
- EM relabelling (`relabel_events.py`, round-1 model): best-candidate relabel .809 (trim) / .809 (no trim) acted accuracy; conservative relabel (change a label only when its loss is above the 2-means split of log best losses and another candidate's is below) .912. Training labels of the target-known events: round 1 .858, trimmed relabel .993 (14.6k of 24.0k events kept), conservative .908. Round-2 models, clean presses all lights right: trimmed .964, conservative .932, no-trim best .883; but the trimmed model misses toggles the first model missed (pressing light 1 toggles light 2 in 14% of probes): trimming keeps only events the first model already explains. `relabel_events.py` now trims only with `--trim` (default off) [puzzle chain below uses the trimmed relabel, `obj_events_r2trim`].
- bug fixed: `events_objects.stab_groups` looped forever on a change met within one frame (transition() can return t0 = t1 + 1; with slack 0 it was never removed): cube events grew to 15.6 GB and were killed twice; windows are now (min, max) (test added). `run_family.ps1` stops the chain when a stage leaves no output (a killed events stage had let the world model start on missing events). Never run two objects / events stages at once (cube events + puzzle events + puzzle objects got both events stages killed at low free RAM).
- `objects.py` place rule refined: only DISCRETE places snap (2-means bimodal, Ashman D > 2, AND each side dominated by one exact reading, > 1/2; dev VAL dominant shares: puzzle lights .72-1.0, scene buttons .99-1.0, scene window .20 / .25 and drawer .20 / .13) and take their centre as position; sliding places keep the continuous reading and the changed-pixel centroid (centre positions for every place had dropped scene privileged |corr| window .98 -> .72, drawer .97 -> .62, acted .833 -> .772). Puzzle tables are unchanged by this (all lights discrete); scene rebuilt (run 4).
- puzzle chain started 08:27 (`logs/method_puzzle_chain_r2trim_*.log`): support -> cost-to-go -> skill (`--cond spatial`) on `obj_events_r2trim` / `obj_model_r2trim`, then closed loop 30 episodes with the scripted press arm (PRIVILEGED diagnostic) and with the learned skill (method), both with the new belief update (closed_loop_objects.py: entities predicted to change but unseen since the first observed change take the predicted state when all seen entities agree).
- 2026-10-09 puzzle closed loop, round-2 trimmed model (`obj_model_r2trim`: cost-to-go 150k steps 89.7 min, offline plan found .948 vs .84 before), 30 episodes seed 0: scripted press arm (PRIVILEGED diagnostic) 4/30 (task 1 4/6; task 4 ended 1-3 lights wrong in 4 of 6), exact presses .93, WM match .94-.98. Failure: 65% of events repeat the previous acted light (one light pressed 30+ times): replanning after every event with weighted A* (lam .6, learned h) returned plans whose first step undid the last event. Fix (`closed_loop_objects.py`): the plan continues while outcomes agree with the prediction (replan on a surprise, a timeout or an empty plan), and a new plan may not pass through the last 3 states left (`Model.plan(avoid=...)`). Scripted and skill loops rerun with it (`obj_loop_*_r2trim_pc`); cube and scene chains queued after them (`logs/method_gpu_queue_*.log`).
- 2026-10-09 10:30 loop fixes (closed_loop_objects.py): (1) PRIVILEGED scripted press arm travels at a safe height (a low straight path pressed other buttons on the way); (2) an event also ends when other entities changed and settled while the acted entity is hidden (the pressed light stayed under the arm at its rest pose: 2-3 timeouts of 250 steps per failed episode), the belief update then gives the hidden one its predicted state if the seen ones agree. Partial scripted run with (1) + plan commit (CPU, stopped at 9 episodes for the new rule): task 1 5/6 successes, task 2 0/3 (timeouts with the acted light hidden). Learned skill on the trimmed labels: exact presses .454 (spatial skill on round-1 labels .288, full conditioning .146), no-press .033; loop 0/30 (before the fixes; light 2 trained on 23 events: exact .03). Rerun of both arms with all fixes on CPU: `obj_loop_{scripted,skill}_r2trim_pc3`.
- 2026-10-09 11:15 puzzle loops with all fixes (CPU, `obj_loop_*_r2trim_pc3`, 30 episodes seed 0): scripted press arm (PRIVILEGED) task 1 6/6 (150-300 steps, no timeouts), task 2 1/6, tasks 3-4 0/12 (task 5 running); learned skill task 1 1/6 (first success of the method without privileged input), tasks 2-3 0/12. Failure of the high level on tasks 2-5: light 2 — of 177 exact planned presses of light 2, 157 came out differently from the round-2 model's prediction (the trimmed relabel kept 23 training events acting on light 2), then 3 (24/42), 7 (19/38), 4 (13/23). Relabel policy changed for every family: an event trains round 2 if the round-1 model explains it OR its best candidate is the contact rule's choice (puzzle TRAIN labels .931, 23.2k events, 160 on light 2; explained only .993, 14.5k, 23 on light 2). Skill on these confirmed labels (`obj_skill_u`): val chunk loss .356 (trimmed .361), loop running. GPU queue: cube (relabel .. loop) -> scene (all) -> puzzle (relabel .. loop + scripted) with the new policy.
- 2026-10-09 11:20 final puzzle loops (seed 0, 30 episodes, round-2 trimmed model): scripted press arm (PRIVILEGED) with all fixes 7/30 = task 1 6/6 (200 steps), task 2 1/6, tasks 3-5 0/18, exact presses .977 (`obj_loop_scripted_r2trim_pc3`). Learned skill on confirmed labels (`obj_skill_u`, 23.2k events): exact presses .56 (trimmed-label skill .45), loop 0/30 (`obj_loop_skill_u`, GPU); 241 of 291 wrong single presses lie outside the box between the previous press and the target (mis-targeting, not presses on the way). Cube relabel with the agreement policy keeps 94% of TRAIN events as targets (trimmed: 57%).
- 2026-10-09 12:20 puzzle round-2 model with the agreement relabel (`obj_model_r2agree`, GPU 4.2 min): clean presses all lights right .947 (trimmed .964); probes on VAL states (toggled set right): light 1 .26 (trimmed .12), 7 .33 (.10), 2 .57 (.84), 8 .50 (.71), 4 .71 (.81): no clear gain. In the scripted loop, of 229 exact presses of light 2 the trimmed model predicted 5 changed lights (true 4) in ~113 and 2-3 in others: spurious / missed toggles on the states the loop visits, which the VAL probe (.84 right) does not show; the remaining high-level errors are generalization of the world model to loop states, not labels. Keypoint skill (`skill.py --arch keypoints`, spatial softmax over 32 maps at 16 x 16) training on the confirmed labels (`obj_skill_kp`), then its loop; cube skill / loop and the scene chain wait for it.
- 2026-10-09 12:45 keypoint skill (`obj_skill_kp`, spatial softmax): val chunk .364 (CNN .356), loop exact presses .537 (CNN .56), 1/30 (task 1 1/6): not better; the CNN stays the default (`--arch cnn`). GPU queue 4 (`logs/method_gpu_queue4_*.log`): cube skill + loop (round-2 model with the agreement relabel, cost-to-go done in 46 min) -> scene full chain -> puzzle chain with the agreement relabel (relabel .. loop, loop_scripted).
- 2026-10-09 13:35 cube-triple closed loop (round-2 agreement model, cost-to-go 46 min, CNN skill 22 min; GPU, code as of 12:40): task 1 2/6 successes (848 and 377 steps) — the method's first cube successes — with "no plan found" in most episodes. Diagnosis (CPU replay of task 1-3 episode 0): (1) goal image and first frame read movers with the near-agent test, so a cube seen in the goal image next to the arm became a "hidden goal" and cubes seen at the start became "unknown, assumed at goal" -> static frames now observe every visible mover (`read(..., static=True)`); (2) the planner's successor kept the model's sub-threshold drift of untouched cubes and its approximate placement of the acted one, so goal tests (tol_pos .99 px) failed one move from the goal -> `Model.step_batch` now applies the event's definition (acted entity = x; a predicted change within the change thresholds = no change). With both, task 1 plans the one move ([(2, goal)]), task 3 a two-move plan; task 2 still finds no plan (a goal cube unseen in the goal image, a stack: the hidden goal needs the model to predict covering). Event support does not block (scores pass after the sigmoid). Cube loop rerun with these changes on CPU (`obj_loop_seed0_v2`); the scene chain and the puzzle chain use them.
- 2026-10-09 13:15 the skills ignore their condition (offline, VAL segment frames one third into the approach; swap the event condition for another entity's): puzzle `obj_skill_u` predicted xyz actions change by .095 vs a spread of .759 across frames, action MSE vs data .092 (true condition) / .133 (swapped); cube `obj_skill` .092 vs .760, MSE .222 / .226. Two stacked frames (gap 2) show where the arm is already heading in play data, so behaviour cloning copies the motion (copycat) and the target condition carries little; at an event start in the loop there is no such motion and the arm goes elsewhere (puzzle wrong presses spread over the board; cube moves the wrong cube, mostly pushing). Cube loop with the new reading / transitions (CPU): task 1 first plans of 1-2 moves in every episode, 1 success in the first 5 (timeouts from the skill). Next: skill from the current frame only (`--hist-gap 0`) on puzzle, then cube and scene.
- 2026-10-09 13:40 RESULT puzzle-4x5 closed loop, METHOD (no privileged input; round-2 trimmed model, skill on the confirmed labels from the current frame only, `obj_skill_h0` / `obj_loop_skill_h0`, seed 0, 30 episodes): 7/30 = task 1 6/6 (364-721 steps), task 2 1/6, tasks 3-5 0/18; exact presses .865 (two-frame skill .56). Equal to the PRIVILEGED scripted-press arm (7/30) on the same model: the low level is no longer the bottleneck on puzzle; tasks 2-5 fail on the world model (lights near the top row). The single-frame skill: condition swap changes predicted actions .312 (two frames .095), action MSE true / swapped condition .116 / .431. `skill.py --hist-gap` default 0. GPU queue 5: cube skill + loop (new reading and transitions), scene chain, puzzle chain (agreement relabel).
- 2026-10-09 14:20 cube with the single-frame skill (`obj_skill_h0`): it now goes to the planned cube (condition swap .196 vs .092 with two frames) but does not grasp and carry it (task 1 0/6, cube touched and shifted 2 cm or not reached); the two-frame skill had task 1 2/6 (old planner) and 1/26 overall with the new planner (CPU run `obj_loop_seed0_v2`). Pick-and-place needs the phase cue of two frames, button presses do not. `skill.py --hist-dropout p` (two frames, the current frame twice with probability p during training) tested at p .5 on puzzle and cube (`obj_skill_hd`, `obj_loop_skill_hd`) before the scene skill; OGBench pixel baselines for reference (OGBench paper, Table 2): visual-puzzle-4x5 best 17% (GCIVL), cube-triple 21% (HIQL), scene 49% (HIQL), puzzle-3x3 73%, 4x4 60%, cube-single 89%.
- 2026-10-09 15:00 history dropout (two frames, current frame twice with p .5 in training): puzzle condition swap .112, loop 0/30, exact presses .542 (like the two-frame skill); cube swap .130, MSE true/swapped .247/.250. Not adopted. Test running: cube task 1 with the single-frame skill executing whole 8-action chunks (`--exec-steps 8`, CPU, `obj_loop_h0_exec8`), the hypothesis being that without history the policy dithers between grasp phases.
- 2026-10-09 15:20 cube task 1 (6 episodes, CPU) with the single-frame skill executing whole 8-action chunks: 2/6 (199 and 699 steps) vs 0/6 executing 4 of 8. Puzzle check of the same setting running (`obj_loop_h0_exec8`, CPU) before making `--exec-steps 8` the loop default.
- 2026-10-09 15:30 cube-triple closed loop with the history-dropout skill (`obj_skill_hd`, two frames at test): 4/30 = task 1 4/6, tasks 2-5 0/24 (first plan found .73) — best cube result so far; the same skill type on puzzle 0/30. Puzzle with the single-frame skill executing 8-action chunks: task 1 4/6 (exec 4: 6/6). To keep one setting for all families: `closed_loop_objects.py` now feeds the current frame twice at an event's first skill query (`--no-fresh-start` to disable; the motion left over from the previous event is unrelated to the new target), testing hd skill + fresh start on puzzle and cube (`obj_loop_skill_hd_fresh`).
- 2026-10-09 15:25 hd skill + fresh start (current frame twice at an event's first query): puzzle 3/30 (task 1 2/6, task 4 1/6; exact presses .559), cube 4/30 (task 1 4/6). Skill settings so far (seed 0, 30 episodes): single frame exec 4 -> puzzle 7/30, cube task 1 0/6; single frame exec 8 -> puzzle running (>= 4 successes), cube task 1 2/6; two frames + history dropout .5 + fresh start -> puzzle 3/30, cube 4/30. Puzzle needs precise targeting (one frame), cube a phase cue for grasping (two frames). Candidate next: the last executed gripper command as a skill input (the agent's own action, a phase cue without the motion direction).
- 2026-10-09 15:45 puzzle single-frame skill executing 8-action chunks: 4/30 (task 1 4/6), exact presses .785. Method default set to the balanced skill (two frames, history dropout .5; loop: current frame twice at an event's first query, exec 4): puzzle 3/30, cube-triple 4/30 (cube outputs `obj_skill` / `obj_loop_seed0` = the hd skill and its fresh-start loop). Full queue started (`logs/method_full_queue_*.log`): scene chain (wm .. loop), held-out cube-single, puzzle-3x3, puzzle-4x4 (front end with run_sm2.ps1, then the same chain, no code or setting changes), then the puzzle chain with the agreement relabel.
- 2026-10-09 16:57 RESULT scene closed loop (method defaults: agreement relabel, round-2 model event-exact .885 / side effects .827, cost-to-go 49 min, offline plan found .64; hd skill + fresh start; seed 0, 30 episodes, 750 steps): 7/30 = task 1 4/6, task 3 3/6, tasks 2 / 4 / 5 0/6; first plan found .50 — tasks 4 and 5 never find a plan (goal: the cube in the closed drawer, hidden in the goal image; the model would have to predict the cube becoming covered). OGBench pixel HIQL 49%. Today's dev summary (seed 0): puzzle-4x5 7/30 (single-frame skill; 3/30 with the default skill) vs best baseline 17%; cube-triple 4/30 vs HIQL 21%; scene 7/30 vs HIQL 49%. Held-out queue started: cube-single front end 16:57 (`heldout_cube_single`).
- 2026-10-09 17:20 held-out cube-single front end trained (agent 3.6 min, label 3.6, seg 0.8 (val recall .946 / precision .932), segpred 1.0, scene 4.1, stargets 2.2, seethru learned 3.2 / pixel 3.0) but the PRIVILEGED front-end diagnostic crashed (`scripts/sm_diag.privileged_state` assumed three cubes) and run_sm2.ps1 threw, so the queue skipped the cube-single chain. Fixed: the diagnostic takes the cube slices from the qpos width (1-4 cubes), and a failing diagnostic no longer stops run_sm2.ps1 (scoring only). Cube-single resumes after the current queue (puzzle-3x3, puzzle-4x4, puzzle chain): `logs/method_cube_single_followup_*.log`. No method code or setting changed for the held-out runs.
- 2026-10-09 17:55 HELD-OUT puzzle-3x3 (no code or setting change): front end 25 min; objects found 7 of 9 buttons (privileged |corr| 1.0 for buttons 2-8; buttons 0 and 1 not discovered; visible fractions .30 / .38 / .017 (button 4) / .64 / .72 / .58 / .75: the arm covers the small board most of the time, and place discovery needs agent-free rest instances); all 7 discrete (snapped). Events VAL: 21.4 / episode, PRIVILEGED recall .944, precision 1.0. Chain continues (wm .. loop).
- 2026-10-09 18:50 RESULT HELD-OUT puzzle-3x3 closed loop (seed 0, 30 episodes, 500 steps): 1/30 (task 1 1/6); first plans of length 0 in 14 episodes (at goal for the model: the undiscovered buttons 0 / 1 carry the goal difference); episodes end 1-8 lights wrong. OGBench pixel HIQL 73%. Failure is perception (place discovery under near-constant occlusion), not the event model. Puzzle-4x4 front end started 18:50.
- 2026-10-09 19:27 HELD-OUT puzzle-4x4: front end 25 min; objects found 15 of 16 buttons (button 2 missing; button 1 visible .053, button 5 .257, others .40-.89; privileged |corr| 1.0); events VAL 27.3 / episode, recall .987, precision 1.0. The same failure as 3x3, milder: top-row buttons near the arm's rest pose are rarely seen agent-free, and place discovery needs agent-free rest instances.
- 2026-10-09 21:20 RESULT HELD-OUT puzzle-4x4 closed loop (seed 0, 30 episodes, 500 steps; offline plan found .983): 2/30 (task 5 2/6); exact presses .349 (default skill: two frames + history dropout + fresh start), wrong .587; first plans 3-6 events, 7-9 replans per episode. OGBench pixel HIQL 60%. Held-out summary so far: the event layer transfers (event recall .94-.99, precision 1.0; offline planning .98 on 4x4), the failures are place discovery under near-constant occlusion (3x3: 7 of 9 buttons, 4x4: 15 of 16) and the low-level skill (35% exact presses with the default; the single-frame skill reached 86% on 4x5 but fails to grasp on cube).
- 2026-10-09 21:25 queue stopped at the user's request to rethink how the method generalizes to new environments (puzzle-4x5 agreement-relabel chain and held-out cube-single not run). No jobs running.
- 2026-10-09 22:00 status and direction written to `method/STATUS.md` (results table vs OGBench pixel baselines, diagnosis: occlusion > hand rules > BC skill > hidden goals > model generalization; agreed plan: pragmatic first = SeeThrough-inpainted agent-free perception + offline GCRL low level + frozen-code held-out protocol over the full cube / puzzle suites; then v2 = events inferred in a learned factored model with agent-conditioned binary interaction gates, go / no-go on puzzle-3x3 + cube-single). Data: user approved fetching all OGBench cube and puzzle datasets; `local/fetch_data.py` groups `visual_cube_all`, `state_cube_puzzle_all` added (visual-cube-double / quadruple play + state play / noisy fetching, ~21 GB new); pixel noisy datasets (~43 GB) left out for disk (94 GB free before the fetch).
- 2026-10-09 22:25 data fetch finished (`logs/fetch_cube_puzzle_all_20261009.log`, ALL_DOWNLOADED, 48 files): visual-cube-double / quadruple play (+ val) and state cube single..quadruple / puzzle 3x3..4x6 play + noisy (+ val); every new file's npy headers read back (visual-cube-quadruple 5.005M frames, qpos 42 = 4 cubes). 74 GB free on E:. No caches built yet (built on first use by `local/run_sm2.ps1`).
- 2026-10-09 22:30-22:45 PRAGMATIC STEP 1 (agent-free perception), held-out puzzle-3x3, offline only. New `method/view.py` (component 5b): a token in the dilated agent mask shows the EXEMPLAR of its SeeThrough code (the 4 x 4 patch seen most often with that code at that token agent-free in TRAIN; 306 keys, purity .99) when p > .5, else hidden; `objects.py --view` runs every rule on views, contact masks stay on raw frames; `closed_loop_objects.py` builds the same view online from `front.npz`. VAL: agent-masked tokens 41.7% -> hidden 17.7%. PRIVILEGED view diagnostic (scratchpad `view_diag.py`): on agent-covered button tokens the decoded state is right .924 (confident) / .814 (not), but split by time: >10 frames from a toggle of that button .984 / .926, within 10 frames .73 / .66 (40% of covered frames). Discovery on views (`objects_view_disc`): 8 places, top row still one merged non-discrete place (26.2, 27.9), button 4 only 2 px. Cause found in the views: the per-pixel TYPICAL colour (mode) mixes the two states of a place switching ~50/50, so changed pixels are fragments (top-row instances median area 2 px vs 11 for button 3). Added PLACES (iii) to `objects.py` (change instances: pixels changed between consecutive discovery frames, exact recurring centroids, same rule as (i)); on 3x3 views only 7,692 change instances, top-row toggles rarely seen in both frames of a pair (gripper hides them): no gain (`objects_view_disc2`). Root cause: SeeThrough has no target while an object changes under the agent (58% of covered token frames). Fix under test: partial-label SeeThrough (`frontend.see_targets(partial=True)`, `frontend_train.py --partial`: covered frames of a visit that changed the token get the set {before, after}, loss -log(p(before) + p(after))); front end `heldout_puzzle3x3/train_pl` (hard links to the base files, new stargets + seethru).
- 2026-10-09 22:45-23:40 PRAGMATIC STEP 1 results (held-out, offline, PRIVILEGED scoring only). Partial-label SeeThrough (`frontend_train.py --partial`; front ends `heldout_puzzle{3x3,4x4}/train_pl` = hard links to the base files + new stargets / seethru; set targets 5.9% / 8.5% of covered token frames; VAL set acc .986 / .938, behind confident .978 / .977) + views (`method/view.py`) + `objects.py --view`, outputs `heldout_*/method_pl/`. View diagnostic, covered button tokens right: 3x3 .85 -> .91 (near toggles confident .73 -> .83), 4x4 .96 / .99. Discovery: 3x3 8 of 9 buttons (button 1 still missing: a 2-px place with |corr| .27), 4x4 16 of 16 (was 15). Events: 3x3 49.4 / episode (presses 31), timing (press within +-15 frames of an event) .768 -> .962; 4x4 44.4 / episode, .922 -> .987. ACTED (pressed = label, presses inside exactly one event core): 3x3 .465 -> .582, 4x4 .714 -> .688; the touched + centroid rule fails at board edges (3x3 pressed 5 / 7 / 8: .33 / .31 / .07, the inner neighbour chosen). Attempts to fix attribution, all offline (scratchpad scripts): EM relabel with the round-1 model (3x3 .464 -> .366 of presses; original runs also worse: 3x3 .281 -> .230, 4x4 .601 -> .583); round-1 model trained on fully observed events only (`world_model.py --full-only`, 17.5% of 3x3 events; contact rule right on .906 of them): .366; effect-pattern consensus: .37 (the arm mask touches every member); learned MIL contact map (`method/contact.py`): .10 (not identifiable: a pressed light's neighbours always change with it; it learned "the light below"); distal end of the arm mask: .16 (the mask holds the shadow); AgentNet action-sensitivity effector, nearest identity at contact - 2 frames: 3x3 .853, but 4x4 .49 (sensitivity spreads over the whole arm; distance to the pressed light p50 2 px on 3x3, 5-9 px on 4x4). Pressed light read as unchanged by its own event: 3x3 26%, 4x4 11% (read under the gripper). Conclusion: perception gains are general (both held-out envs), attribution of the acted entity is the bottleneck and needs an effector model or the learned event model (v2). PLACES (iii) (change instances) gave no measurable gain (3x3 views, old SeeThrough).
- 2026-10-10 user decision: v2 = HYBRID (high level: event WM + heuristic search over object states; low level: offline goal-conditioned RL given one-event goals), attribution learned instead of the contact rule, perception gains kept; first tests puzzle-3x3 (easy) + puzzle-4x5 (hard) with the same code. Literature check (2026-10-10): no published method strong on both easy and long combinatorial pixel puzzles (OGBench pixel 4x5 <= 17%, 4x6 <= 15%; WMPA arXiv 2610.10932 state-based 4x5 18-20%; SHARSA state 4x5 91% with 100M transitions). Plan drafted by a design panel workflow (Sonnet subagents: 4 designs, 3 judges, synthesis, critic) -> `method/V2_PLAN.md`. Started: partial-label SeeThrough + views + objects/events for dev puzzle-4x5 (`puzzle_grow_20261008095020/train_pl`, `method_pl/`).
- 2026-10-10 00:20-00:51 dev puzzle-4x5 with the agent-free front end (`puzzle_grow_20261008095020/train_pl` partial-label SeeThrough, `method_pl/` views + objects + events; offline, PRIVILEGED scoring): 20 of 20 buttons (|corr| >= .99; button 2 visible .28, the others .70-1.0), events VAL 34.4 / episode (31 presses; recall .998, precision 1.0), timing .977 -> .996, acted (pressed = label, presses in one core) .875 -> .883 (of presses .823 -> .829); errors on the bottom row (pressed 14 / 15 / 17 / 19: .14 / .41 / .26 / .30, the light above chosen: the arm body covers it). New `method/effector.py` (effector keypoint by inverse dynamics: soft-argmax keypoint whose displacement predicts the mean commanded translation) and `events_objects.py --effector` (acted = identity not known to stay nearest the effector at the contact moment; identities within half an object width of the effector are not observed). Training effectors on held-out 3x3 / 4x4 and dev 4x5 (`method_pl/effector`).
- 2026-10-10 00:52-01:50 EFFECTOR attempts (offline, PRIVILEGED scoring at true press frames / presses in one event core). v1 `method/effector.py` (keypoint whose 1-4-step displacement predicts the mean command through a free MLP; val R2 x .89 y .98 z .74; diffuse map): acted at the contact moment 3x3 .90 (rule .58) but 4x4 .40, 4x5 .33; the keypoint sits ~16 px from the pressed light (not on the tip). v2 `method/effector2.py` (keypoint + height, learned affine camera inverse, actions integrated over 1-32 steps with workspace clipping, V2_PLAN Sec 3.1): run 1 (`heldout_puzzle3x3/method_pl/effector2_run1_offarm`) failed: keypoint 33 px from the pressed light, 22% of its mass on the arm -- with a learned camera map the keypoint is ANY affine image of the effector position (a gauge freedom the plan missed), and the clip bounds (init +-20) never acted. Fix: heatmap restricted to the dilated arm mask ("the effector is part of the agent") + bounds set from q quantiles at step 1000; retraining on 3x3. Oracle rung R1a on 4x4 queued after it (`scratchpad/oracle_labels.py`: press-matched PRIVILEGED labels on `method_pl/obj_events` -> `obj_events_oracle`: 32% of events hold no press (split / late-flip events), 1 event with several; rule labels agree on .70 of target-known matched events) -> WM -> support -> cost-to-go -> scripted loop (`scratchpad/rung.ps1`, PRIVILEGED-oracle tier).
- 2026-10-10 02:05 effector v2 run 2 (arm-constrained heatmap, learned clip bounds; `heldout_puzzle3x3/method_pl/effector2_run2_clipcollapse`): keypoint on the arm and stable (step p50 .64 px, peak .96, window loss .02) but the learned x bounds collapsed to .61 / .62, so q_x was constant and the keypoint did not follow the effector vertically in the image (offset to the pressed light by row -6 / +1 / +9.5 px; acted .32; zeta carried no floor signal). Clipping is now off by default (`--clip` keeps it); the no-clip retrain on 3x3 runs after the 4x4 oracle rung (GPU busy).
- 2026-10-10 03:16 ORACLE RUNG R1a, held-out-designed (now DEV) puzzle-4x4, PRIVILEGED-oracle tier (press-matched acted labels from button_states on `method_pl/obj_events` + learned views/objects/WM/support/cost-to-go + scripted privileged press arm, seed 0, 30 episodes): 18/30 = 60% (tasks 1.0 / .33 / .5 / .5 / .67), first plan found 30/30 (5-8 events). V2_PLAN decision D1 (R1a >= 60%: go) is met at its edge. All 12 failures time out at 500 steps with 2-3x the planned events: at event end the loop sees only part of the toggled lights (e.g. [12] of [4, 8, 9, 12]) and lagged readings under the arm contradict the prediction (outcomes as predicted 57% in successes, 34% in failures), so it replans. Effector v2 run 3 without bounds (`effector2_run3_noclip`): window loss stayed .7 (pushing into the table does not fit), keypoint 29 px from the pressed light. Changes: `effector2.py` one-sided floor on height (running .5% quantile of q_z; a constant q cannot fit upward motion); `closed_loop_objects.py --confirm clear` (a reading settles an event or contradicts the WM only with no agent pixel within half an object width). Running: R1a on dev 4x5 (oracle labels: 91% of events match one press; rule labels agree .85), then the floor effector on 3x3; the 4x4 oracle loop with --confirm clear on CPU.
- 2026-10-10 05:15-05:45 closed-loop fixes on the 4x4 ORACLE tier (CPU runs, seed 0, 30 episodes, PRIVILEGED-oracle labels + scripted arm): `--confirm clear` (a reading confirms only with no agent pixel within half an object width): 1/30, 29 timeouts -- the dilated arm mask is too large, entities near the arm never settle (flag kept, default off). `--grace -1` (after the event-end condition keep reading for the 90th percentile of within-event arrival spreads of TRAIN events, 23 frames on 4x4 / 21 on 4x5 / 19 on 3x3, stop early once every predicted change is seen as predicted): 21/30 = 70% (60% without), events per episode 20.8 -> 10.9, outcomes as predicted .48 -> .64, timeouts 3 -> 2. Remaining not-as-predicted events (117 of 326): WM mispredicts with all lights seen right 48, toggled lights still unseen 47, a light seen changed that did not toggle 20, ended before the press 2; all 9 failures time out at 500 steps (scripted press ~25-35 steps + grace; plans of 5-7 events). Also tested and rejected: effector = opaque moving part of the agent (hidden view tokens minus the always-hidden base): nearest light at the press .37-.44 on 3x3. Seed 1 with / without grace running on CPU.
- 2026-10-10 05:30-05:45 more effector probes (offline, PRIVILEGED scoring at true press frames): AgentNet sensitivity to the YAW command (the gripper rotates, the effector position does not): 3x3 nearest light .70 / .61 (offsets -2 / 0), 4x4 .42 / .58 (top row right, lower rows wrong) -- worse than the translation sensitivity on 3x3 (.958); the gripper command is not used in puzzle play (std .089 vs .254 on cube-single), so it cannot locate the effector on puzzles. New low level L0 (V2_PLAN Sec 4): `method/goal_maps.py` (Delta maps: from / to / appearance-change blobs of the entities that must change, the events' change rule) + `method/skill_delta.py` (one frame + Delta maps -> 16-action chunk, BC, shift augmentation; trained on event segments with every known before/after, no acted-entity label) + `closed_loop_objects.py` support (checkpoint kind "delta": goal = the WM's predicted next state from the event start). Queued on the GPU after the 4x5 rung and effector run 4: Delta executor on 4x4, then the 4x4 loop (oracle-tier WM + learned executor + grace).
- 2026-10-10 05:55 4x4 ORACLE tier seed 1 (CPU): with grace 21/30 = 70%, without 19/30 = 63%; over seeds 0-1 (60 episodes) grace 42/60 = 70% vs 37/60 = 62%.
- 2026-10-10 07:04 ORACLE RUNG R1a, dev puzzle-4x5 (PRIVILEGED-oracle tier, no grace, seed 0, 30 episodes): 4/30 = 13% (task 1 4/6, tasks 2-5 0/6) -- below the v1 scripted run (7/30 with rule labels and raw-frame perception). First plans 2-9 events (found 30/30), but 31.7 events per episode, 19 timeouts, outcomes as predicted .52; events not as predicted: toggled lights unseen 272, WM differs 134, nothing toggled (ended early on a false change) 97; repeated timeouts on button 2 (top row under the arm's rest pose, visible .28) and runs pressing one button again and again. The 4x5 loop with grace is running on CPU. Effector v2 run 4 (one-sided running floor): degenerate again (floor drifted 19 -> 81, q_z inflated, val window loss .84, no vertical tracking, acted .32). New `method/effector3.py`: no position in the prediction (fully convolutional, RF ~15 px, per-cell estimate of the mean commanded translation from the pair (x_t, x_{t+d}) + confidence; prediction = softmax-weighted mean; effector = confidence-weighted cell position), so no gauge freedom. Delta executor run 1 crashed (float64 shift tensor), fixed and relaunched; effector v3 on 3x3 / 4x4 queued after it.
- 2026-10-10 07:30-07:45 4x5 ORACLE loop with grace (CPU, seed 0): 3/30 = 10% (13% without). Not-as-predicted events now mostly "WM differs" (443 of 1117). Diagnosis: the oracle WM is right OFFLINE (VAL event-exact .918; the loop's agreement rule on fully known VAL events .932) and the loop's perception is right (debug run `method_pl/obj_loop_debug`, new per-event logging of predicted / disagreeing ids: seen changes = true toggles), but the WM mispredicts presses of bottom-row lights 14 / 17 / 19 (predicts neighbour toggles in .60 / .67 / .79 of their TRAIN events vs .93-1.0 elsewhere; 14 has 2005 oracle events vs ~1400 for others). Cause: the pressed light under the arm reads its old state late, its change becomes a separate event, so the event that holds the press says "light 14 stays, neighbours toggle" -- contradictory data even with oracle acted labels. The lag under the gripper is the root of both the attribution and the segmentation errors: the fix needs the effector (readings under the effector unobserved), which effector3.py targets.
- 2026-10-10 07:53-08:16 EFFECTOR v3 WORKS (`method/effector3.py`, offline, PRIVILEGED scoring; `method_pl/effector3`): at true press frames the nearest discovered light is the pressed one 1.0 (3x3, offsets -4 / -2) and 1.0 (4x4, offsets -2 / 0); distance to the pressed light p50 1.7 / 1.4 px, p90 3.1 / 2.0 px. Acted at the event contact moment (presses in one core; effector point at core0, nearest identity among ALL identities): 3x3 .905, 4x4 .969 (core0 + 2: .889 / .992), vs the rule .582 / .688 (restricting to identities not known to stay hurts, .67 / .83: the lagged pressed light reads "unchanged"). Delta-executor L0 (`method_pl/skill_delta`, 4x4, 40k steps, val chunk loss .567, condition sensitivity .35 < target .6) in the oracle-WM loop with grace: 0/30 -- it presses a neighbour of the planned light (planned 9 -> 13, 4 -> 9, 11 -> 6) and takes 32-190 steps per event; training Deltas often lack the lagged pressed light, so the cross is ambiguous. `events_objects.py --effector` now reads effector3 track files; running: effector v3 on 4x5, then events with --effector (identities under the effector unobserved; acted = nearest the effector) for 3x3 / 4x4 / 4x5 + content checks.
- 2026-10-10 08:16-08:45 effector v3 on dev 4x5: nearest light at true press frames 1.0 (offsets -4 / -2), .996 (0), distance p50 1.5-1.7 px; acted at core0 among all identities .997 (core0 - 2: 1.0). First integration `events_objects.py --effector` (`method_pl/obj_events_eff`; acted among identities not known to stay; effector-clear radius thr_pos / 2): acted .569 / .703 / .837 of presses (3x3 / 4x4 / 4x5), events per episode unchanged (48.9 / 44.1 / 34.4): the lagged pressed light still reads "unchanged" and is excluded. Second integration (`--effector-lag L`): an entity the effector came within one object width of (last-seen position) is unobserved for L frames afterwards (L = p90 of within-event arrival spreads of the first pass), acted = the entity nearest the effector at the contact moment among ALL entities -> `method_pl/obj_events_eff2`, running with content checks.
- 2026-10-10 09:05 LEARNED ATTRIBUTION (no privileged input; PRIVILEGED scoring only): `events_objects.py --effector method_pl/effector3 --effector-lag L` (L = 23 / 21 / 19 for 4x4 / 4x5 / 3x3, the p90 rule) -> `method_pl/obj_events_eff2`. Acted = pressed (presses in one event core / of all presses): 4x5 .996 / .952 (rule on the same views .883 / .829; bottom row 14 / 15 / 17 / 19 now .99 / 1.0 / 1.0 / .98 from .14 / .41 / .26 / .30), 4x4 .959 / .812 (rule .688 / .556; remaining errors at the corners under the arm base, button 0 .59, 3 .62), 3x3 .859 / .741 (rule .582 / .464; button 1 not discovered = 10% of presses). Events per episode 33.4 / 39.2 / 39.5 (31 presses; was 34.4 / 44.4 / 49.4); privileged recall .995-.998, precision 1.0. Cost: target-known share of presses .63 / .30 / .33 (the pressed light is unobserved for L frames after the visit and the next press often comes first). Running: rung R1c = WM / support / cost-to-go on these learned-label events + scripted (PRIVILEGED) press arm + grace, 4x5 then 4x4.
- 2026-10-10 10:53 RESULT RUNG R1c, dev puzzle-4x5. High level on LEARNED labels, scripted PRIVILEGED press arm + grace, seed 0, 30 episodes (`obj_loop_scripted_eff2_seed0`).
  - Pipeline: `method_pl/obj_events_eff2` -> WM `obj_model_eff2` (VAL event-exact .838, side effects .978, after known .979) -> support -> cost-to-go (99 min).
  - Result: 8/30 = 27% (task 1 6/6, task 2 2/6, tasks 3-5 0/18). R1a (PRIVILEGED press-matched labels on the old events) gave 13%, or 10% with grace. OGBench pixel best is 17%.
  - All 22 failures time out at 1000 steps after 5-13 replans.
  - Per event (836 events; PRIVILEGED scoring of 806 consecutive pairs): outcome as predicted .76; acted light changed .98; WM change set = simulator flips .66.
  - Error (i), perception: light 2 flips not read in 141 events, lights 3 and 7 in 18 each, never a false change.
  - Error (ii), WM omits neighbours, depending on state:
    - press 14 -> {13, 14} in 74 of 139 (omits 9 and 19);
    - press 1 omits 2 in 70 of 115;
    - press 17 omits 16 / 18 in ~55 of 100.
  - Cause of (ii) in the TRAIN events:
    - Lights 14 / 15 / 17 / 19 get 2233-2620 events each, against ~1450 for every other light.
    - For acted 14, 1003 of 2620 events read 9 and 19 unchanged. Their known change set is empty (559) or only {14} (427), and their length is p50 1 frame (other 14-events: 2).
    - Light 17 is the same (540 of 2452).
    - Interpretation: late re-reads after the lag shadow, and changes of entities unknown on one side, become one-frame events, attributed to the light nearest the effector.
  - Fix candidates:
    - Drop events with an empty known change set from WM / support / cost-to-go training.
    - Merge an event whose only change is the late re-read of an entity shadowed in the previous event into that event (this fills its after-state).
  - Running next: 4x4 R1c (cost-to-go since 10:58), then the fully learned 4x4 loop. The 4x4 learned-label WM is weaker offline than the oracle-label one: event-exact .621 vs .858, side effects .884 vs .961, 1193 vs 1925 VAL events.
- 2026-10-10 11:30 GENERALITY CHECK queued (user question: are the view / objects rules fitted to the puzzles?).
  - v1 event attribution (rule, raw frames, outputs of 10-08/09; PRIVILEGED scoring with `scratchpad/ev_eval_cs.py`):
    - cube-triple: interactions in one event core .728, acted right .875, known .982.
    - scene: one core .679, acted right .776, known .872; the scene cube is worst (.239 / .541 / .430).
  - Arms, all scored by `scratchpad/cs_compare.py`:
    - (B) the current objects.py / events_objects.py on raw frames (`method_pl/objects_raw`, `obj_events_raw`). CPU, running now; it checks that the puzzle-time code changes did not move cube / scene.
    - (C) partial-label SeeThrough + views + objects --view + events. Same code and settings as the puzzles.
    - (D) C + effector3 + lag shadow (`obj_events_eff2`).
  - C and D need the GPU. They start after the fully learned 4x4 chain (`scratchpad/cs_v2_chain.ps1`).
- 2026-10-10 12:09 RESULT RUNG R1c, puzzle-4x4. LEARNED labels (`obj_events_eff2`) -> WM / support / cost-to-go (h 68 min); scripted PRIVILEGED press arm + grace; seed 0, 30 episodes.
  - Result: 18/30 = 60% (tasks 5 / 6 / 1 / 4 / 2 of 6). R1a with PRIVILEGED labels on the same seed gave 21/30; OGBench pixel HIQL gives 60%.
  - The 12 failures are timeouts. Events per episode 9.8; outcome as predicted .58; WM change set = simulator flips .34.
  - Perception misses (lights flipped but not read): lights 1 (60), 2 (44), 6 (21), 11 (18). These are top-row and centre lights near the arm base.
- 2026-10-10 12:15 EVENT DATA DEFECT, all boards (TRAIN `obj_events_eff2`, PRIVILEGED scoring on VAL).
  - Share of events with an EMPTY known change set: 3x3 .160, 4x4 .262, 4x5 .147. They concentrate on a few lights (4x4: 7 / 11 / 13 at .40-.48; 4x5: 14 / 15 / 17 / 19 at .30-.45), and those lights get 1.3-2x the median event count.
  - Among empty VAL events (PRIVILEGED button states in the core +-2 frames):
    - 4x5: 55% have no flip at all (spurious); 44% are real presses of the acted light whose changes were read stale.
    - 4x4: 47% spurious; 52% real presses read stale.
  - Hypothesis: lights under the arm body read through the views (partial-label SeeThrough accepts the old state while covered) count as KNOWN but stale at event end, and the late read becomes a separate event attributed to the light nearest the effector.
  - Fix to test, general and with no per-board rule:
    - (1) Events with an empty known change set do not train WM / support / cost-to-go.
    - (2) An entity whose reading came through SeeThrough (covered) after the contact is not KNOWN in the after-state until it has been seen agent-free, or until L frames have passed.
    - (3) A late read becomes the previous event's after-state, not a new event.
- 2026-10-10 12:19 GENERALITY CONTROL (B): the current objects.py / events_objects.py on raw frames, cube-triple and scene. CPU; PRIVILEGED scoring `scratchpad/cs_compare.py`.
  - cube-triple: IDENTICAL to the 10-08/09 outputs on every number (3 cubes, 1.5-1.9 cm; events 6.35/ep; one core .728; acted .875; known .982). The puzzle-time code changes do not touch cube.
  - scene: one silent regression, in the window.
    - privileged |corr| .981 -> .915; visible .838 -> .771.
    - window acted .938 -> .855; overall acted .776 -> .759; one core .679 -> .650.
  - Cause: PLACES (iii) (change instances, added 10-09 for puzzle-3x3, no measurable gain there). On puzzles it adds 0-7 place pixels; on scene it adds 62, which join the window place (44 -> 105 px, centre (59.8, 18.6) -> (58.3, 28.1)).
  - Change: `objects.py` PLACES (iii) is now OFF by default (`--change-places` keeps it as an ablation; the report records "used").
    - Scene outputs with (iii) are kept as `method_pl/objects_raw_iii` / `obj_events_raw_iii`; control B is re-running on scene without (iii).
    - The puzzle `method_pl/objects` (built with (iii): +4 px on 3x3, 0 on 4x4, +7 on 4x5) must be rebuilt with the final code before frozen runs.
- 2026-10-10 12:26 control B on scene re-run with PLACES (iii) OFF (`method_pl/objects_raw`, `obj_events_raw`): IDENTICAL to the 10-08/09 outputs on every number (window |corr| .981, acted .938; overall acted .776). The current object / event code without (iii) reproduces v1 exactly on cube-triple and scene. Rule from now on: any change to a perception rule is re-checked on all three families (puzzle, cube, scene) before it is kept.
- 2026-10-10 12:39 RESULT FULLY LEARNED v2 loop, puzzle-4x4. No privileged input; seed 0, 30 episodes, 500 steps, exec 8, grace (`method_pl/obj_loop_learned_seed0`).
  - Components: WM `obj_model_eff2` + executor `skill_delta_pp` (Delta maps + press-point channel; 40k steps, 27.7 min, val chunk .555, condition sensitivity .37).
  - Result: 0/30. First plans found 30/30 (6 events); 3.3 events per episode at p50 83 steps per event.
  - Planned light pressed in 26% of events (PRIVILEGED scoring), another light or several in 53%, nothing in 21%.
  - The WM's predicted change set (the executor's Delta goal) is wrong in 84 of 99 events.
    - With a wrong goal: planned light pressed 20 of 84.
    - With the true cross as goal: 6 of 15.
  - Diagnosis:
    - (1) Delta conditioning ties the executor to WM accuracy; the 4x4 eff2 WM is poor (event-exact .62).
    - (2) The executor is weak even with right goals (n = 15).
    - The scripted arm uses only the planned light's position and reaches 60% on the same WM.
  - Next, after the cube / scene check (GPU): the v1 single-frame spatial skill (`skill.py --cond spatial --hist-gap 0`: planned entity + target; it reached .865 exact presses on 4x5 with rule labels), trained on effector-attributed events, in the same loop.
- 2026-10-10 12:52 EVENTS v3 (`method_pl/obj_events_eff3`). Offline; PRIVILEGED scoring `scratchpad/heldout_event_content2.py` + `ev_stats.py`.
  - Change: `events_objects.transition(obs=)` times a change only from observed frames (outside the effector shadow).
  - Old cause: a reading in the shadow is the old state read late, so the pressed light's change was timed where its reading caught up. That made a separate one-frame event; on 4x5 it explained 99.6% of the events with no simulator flip in their core.
  - Results, eff2 -> eff3:

    | Measure | 4x5 | 4x4 | 3x3 |
    |---|---|---|---|
    | VAL events / episode (presses ~30) | 33.4 -> 29.5 | 39.2 -> 33.4 | 39.5 -> 34.8 |
    | Spurious (no flip in core) | .085 -> .001 | .299 -> .208 | .308 -> .221 |
    | Presses in exactly one core | .956 -> .995 | .846 -> .853 | .862 -> .860 |
    | Acted right (of all presses) | .952 -> .990 | .812 -> .814 | .741 -> .742 |
    | Known changes right | .940 -> .976 | .701 -> .696 | .618 -> .588 |
    | Target known | .626 -> .654 | .299 -> .235 | .331 -> .363 |
    | Empty known change set | .145 -> .081 | .255 -> .263 | .148 -> .133 |

  - 4x5: per-light TRAIN event counts are now uniform (1366-1538; before, up to 2620).
  - Remaining spurious events on the small boards: long change windows of lights covered by the arm BODY (outside the effector shadow; 4x4 lights 7 / 11 / 10, 3x3 identity 1). The contact moment (latest departure) then falls long before the press, and corner presses get the wrong acted label (4x4 buttons 0 / 3: .60 / .43).
  - Next test: `--contact closest` (contact = the effector's closest approach to the changed entities between the latest departure and the first arrival) -> `obj_events_eff4`, slotted between the cube / scene chain's GPU stages (`scratchpad/effev4_chain.ps1`).
- 2026-10-10 13:05 HIGH-LEVEL DIAGNOSTIC on the R1c loops (PRIVILEGED scoring only).
  - Method: minimum number of presses from the simulator start / goal buttons (GF(2) solve, minimal weight over the null space), against the planner's first-plan length.
  - 4x5, task 1-5:
    - optimal 4 / 10 / 14 / 16 / 20;
    - first plan 6 / 7 / 10 / 7 / 7;
    - events executed 6 / 34 / 32 / 35 / 39;
    - replans 1 / 10 / 8 / 9 / 12;
    - success 1.0 / .33 / 0 / 0 / 0.
  - 4x4:
    - optimal 4 / 6 / 6 / 6 / 7;
    - first plan 6 / 4 / 6 / 6 / 5;
    - success .83 / 1.0 / .17 / .67 / .33.
  - Reading: on long tasks the eff2 WM's plans are SHORTER than any real solution (4x5 task 5: 7 vs 20). The model has shortcuts (wrong side effects), so execution contradicts it, and the loop replans until the 1000-step budget runs out.
  - Steps per executed event p50: 30.5 (4x5), 35.7 (4x4). A 20-press task needs about 610 steps even with a perfect plan, so long 4x5 tasks leave room for very few surprises.
  - Next decisive test: R1c on 4x5 with the WM on `obj_events_eff3` (clean events). The checks are first-plan length ~ optimal and in-loop WM accuracy.
- 2026-10-10 13:46 GENERALITY CHECK RESULT (C / D arms): the v2 perception developed on puzzles does NOT transfer to cube-triple / scene. Same code and settings as the puzzles; PRIVILEGED scoring `scratchpad/cs_compare.py`. Note: cube / scene `obj_events*` here already include the `transition(obs=)` fix (eff3 code).
  - cube-triple:
    - views: cube fit improves 1.5-1.9 -> 1.29-1.32 cm, but visibility drops .75-.88 -> .61-.64; one core .728 -> .625; acted .875 -> .830; known .982 -> .950.
    - + effector lag: L = 88 frames (the p90 arrival spread is the carry time, not a reading lag); acted .875 -> .522; one core .586.
  - scene:
    - views: 10 identities instead of 5 (cube x3, window x3, drawer x2; object width 4.65 vs 5.86); events / ep 51.7 vs 12.5; precision .878; acted .776 -> .220.
    - + effector (L = 5): acted .593.
  - Causes:
    - (1) View exemplars assume exact recurrence. Weighted exemplar purity is scene .845 (13.5% of key mass below .5) against puzzles / cube .974-.978 (1.4-1.8%): sliding window / drawer parts.
    - (2) The effector3 point carries an environment-dependent offset from the contact (PRIVILEGED):
      - scene button presses: 6.8 px p50, p90 7.7, never within 4 px;
      - cube grasps: 14.8 px p50, p90 22; the moved cube is the nearest cube only .76 of the time;
      - puzzles: 1.4-1.7 px.
    - (3) Lag L = p90 within-event arrival spread mixes reading lag with interaction duration.
  - Needed for one general formulation, then a re-check on all three families against v1:
    - (a) views only where the exemplar recurs exactly (purity >= 1/2, the majority rule used elsewhere), hidden otherwise;
    - (b) a data-derived contact-offset calibration of the effector point (V2_PLAN 3.3);
    - (c) L from the reading lag after the effector leaves, not from event durations.
  - Also, eff5 (`--contact closest`, cores unchanged): 4x4 acted .814 -> .822, 3x3 .742 -> .734. Not adopted.
  - GPU queue started (`scratchpad/next_gpu_chain.ps1`): spatial single-frame skill on 4x4 `obj_events_eff3` + learned loops (exec 4 / 8), then rung R1c on 4x5 with `obj_events_eff3`.
- 2026-10-10 14:00 GENERAL FIXES for the v2 perception (code; tests 13/13).
  - (a) `view.py --min-purity .5` (default): an exemplar is kept only when it is MORE than half of its key's agent-free patches (exact recurrence); otherwise the tokens stay hidden.
  - (b) New `method/effector_calib.py`. From a first `--effector` events pass:
    - OFFSET = median(centroid of the changed entities before the event - effector at t_core0);
    - LAG = ceil(p90) of (arrival - last frame the calibrated effector was within thr_pos of the entity's NEW rest position).
    - `events_objects.py --effector-offset DU DV` applies the offset; events npz now store `effector_offset`, `effector_lag`, `t_contact`.
  - First-pass calibration (no privileged input):

    | Board | Offset (du, dv) | L |
    |---|---|---|
    | puzzle 3x3 | (-0.1, +1.6) | 18 |
    | puzzle 4x4 | (+0.4, +1.3) | 20 |
    | puzzle 4x5 | (+0.4, -0.6) | 11 |
    | cube-triple | (-0.4, +19.8) | 20 (was 88) |
    | scene | (-0.4, +4.1) | 29 |

    - Cube: residual p50 6.1 px after the offset; the PRIVILEGED estimate was (+1.0, +16.1).
    - Scene: distance p50 20 px before and after; its first pass came from the broken ungated views. It will be recalibrated on gated views.
  - Running `scratchpad/cs_v21_chain.ps1` (CPU) on scene then cube: view_g -> objects_g -> obj_events_g -> obj_events_g_eff1 -> effector_calib_g -> obj_events_g_eff; scored by cs_compare.py.
- 2026-10-10 14:25 RESULTS.
  - (1) EXECUTOR v1-style on 4x4: `skill.py --cond spatial --hist-gap 0` on `obj_events_eff3`; 499k train frames; 24 min; train loss .30, val chunk .62 (overfits).
    - Learned loop with the eff2 WM: exec 4 1/30, exec 8 0/30 (the scripted arm on the same WM: 18/30).
    - PRIVILEGED scoring: planned light pressed .41 / .40 (scripted .99), another light .35, nothing .24-.27; steps per event p50 47-74 (scripted 33).
    - Wrong presses land 1-2 cells away (19 of 36), 3-4 cells away (9), or press several lights (8).
    - Two executor designs (Delta maps, spatial BC) now fail on 4x4. BC from play data is too imprecise here. The low level is an open problem, not a detail.
  - (2) v2.1 perception on cube / scene (`scratchpad/cs_v21_chain.ps1`, scored by cs_compare.py). Recalibrated: scene offset (+1.8, +14.1) L 23; cube (-0.2, +19.4) L 17.
    - cube, gated views + rule acted: acted .902 (v1 .875), but one core .654 (v1 .728), visibility .57-.67 (v1 .75-.88).
    - cube, + calibrated effector + reading lag: acted .675, 3.9 events / ep (merges).
    - scene, gated views: 6 identities (5 exist; one duplicate window), 28 events / ep (v1 12.5), acted .544 (buttons .03: the rule fails on gated views).
    - scene, + effector: acted .596. Buttons .85 / .96 (v1 .81 / .65), window .07, drawer .41, cube .50.
    - Verdict: still below v1 on cube and scene. The effector attribution helps static presses (puzzle lights, scene buttons) and hurts movers and sliding parts.
  - Running: rung R1c on 4x5 with `obj_events_eff3` (started 14:17; cost-to-go about 100 min).
- 2026-10-10 14:40 LOW LEVEL L1 (user decision: image-goal GCIVL + one-event subgoal images). New code; smoke-tested end to end on CPU.
  - `method/gcivl.py`: OGBench GCIVL for pixel tasks ported to PyTorch.
    - Value: two heads on an Impala-small encoder of concat(s, g); expectile .9; Polyak .005.
    - Actor: AWR, alpha 10, fixed-std Gaussian.
    - Goals: value cur / traj-geometric / random = .2 / .5 / .3; actor uniform future.
    - Random crop p .5; batch 256; 500 TRAIN episodes in RAM.
  - `method/subgoal.py`: per discrete place and state, the median of TRAIN frames with the place's own disc agent-free; region = window pixels whose state medians differ beyond their 2-means split; `render()` pastes target states.
    - 4x4: 14 of 16 places (light 2 under the arm base has too few agent-free frames; light 10 not snapped). Panels look right.
  - `closed_loop_objects.py --low gcivl --gcivl --subgoals`: one subgoal per event, rendered from the event-start frame with every rendered place in the WM's predicted next state; one action per step.
  - `method/gcivl_eval.py`: flat baseline (actor on the task goal image, closed-loop seeds).
  - Queued after the 4x5 rung: `scratchpad/gcivl_chain.ps1` on 4x4 (150k steps, flat eval, planner + GCIVL loop with `obj_model_eff2`, comparable with the scripted 60%).
  - 4x5 eff3 WM offline (own VAL events): event-exact .862 (eff2 .838; oracle-label .918), side effects .986.
- 2026-10-10 16:06 RESULT RUNG R1c, puzzle-4x5, CLEAN events (`obj_events_eff3`). WM event-exact .862 -> support -> cost-to-go -> scripted PRIVILEGED press arm + grace; seed 0, 30 episodes (`obj_loop_scripted_eff3_seed0`).
  - Result: 13/30 = 43% (tasks 5 / 6 / 2 / 0 / 0 of 6). eff2 events gave 27%, R1a oracle labels on the old events 13%, the best published pixel baseline is 17%.
  - Events 682 (eff2: 836); outcome as predicted .75; WM change set = simulator flips .73 (eff2 .65).
  - WM misses now sit on the lights near the arm base: (7, 12) 36, (8, 7) 34, (7, 2) 21, (1, 2) 17.
  - First plans are still shorter than the optimum on long tasks (PRIVILEGED GF(2) optimum):
    - task 3: 10 vs 14; task 4: 9 vs 16; task 5: 8 vs 20.
    - The WM keeps shortcuts, and tasks 4-5 time out after 9-10 replans.
  - Reading: clean events lift the high level a lot. The remaining limit is the WM's state-dependent effects, so a structured, relative-position effect model is the next high-level step.
- 2026-10-10 17:10 USER DECISION: re-scope the claim to combinatorial manipulation with discrete-state objects (puzzle boards of every size + scene buttons / locks); cube and sliding parts become stated limitations. Next high-level step: a structured world model. Recorded in `method/STATUS.md` section 0.
- 2026-10-10 17:10 STRUCTURED WORLD MODEL: `world_model.py --wm-arch rel` (default stays `entity`).
  - Per-entity MLP on [own state, acted entity's state, target, offset to the acted entity in place spacings (RBF per axis + raw + distance), is-acted, is-place]. No identity embedding, no attention across entities.
  - Data-derived constants: places = entities whose 95th-percentile position spread is <= tol_pos; h_sp = median nearest-neighbour place spacing (4x5: 7.25 px).
  - CPU check on 4x5 `obj_events_eff3`, 8k steps: VAL event-exact .856 (entity .862), side effects .990 (.986), unchanged .991.
  - PRIVILEGED cross probe (`scratchpad/wm_cross_probe.py`): press every light from 150 data states and 150 random patterns; the predicted change set must equal the true cross.

    | WM | Data states | Random patterns |
    |---|---|---|
    | rel (wm stage) | 1.000 | 1.000 |
    | entity eff3, wm stage | .973 | .990 |
    | entity eff3, + event support (model in the 43% loop) | .947 | .980 |
    | entity eff2 (the 27% loop) | .798 | .949 |

  - Queued after the GCIVL chain (`scratchpad/relwm_chain.ps1`): rung R1c with `--wm-arch rel` on 4x5 (`obj_events_eff3`), then 4x4.
- 2026-10-10 17:30-17:55 USER: keep the GENERAL claim (no narrowing); the method must work on cube and scene. `STATUS.md` section 0 and the memories were reverted to the general claim.
  - Plan G1-G4 (each change checked on all three families against v1):
    - G1 views only where they belong:
      - discovery on raw frames, plus the discrete places that only the views show (snapped, more than w from raw places);
      - movers read on raw frames (agent colours), places through purity-gated views;
      - code: `objects.py` (`--view-discovery` restores the old behaviour);
    - G2 effector only where it belongs: `events_objects.py --shadow-places-only --acted-among not-staying`;
    - G3 structured WM for movers: offset to the target position, translation invariance;
    - G4 subgoal rendering for movers and sliding places.
  - Running (CPU): `scratchpad/cs_g1_chain.ps1` (scene, cube), then `scratchpad/g12_chain.ps1` (all five boards: view_g -> objects_g1 -> effector pass 1 -> effector_calib -> obj_events_g12; scoring).
- 2026-10-10 17:33 RESULT GCIVL on puzzle-4x4 (`method_pl/gcivl`: 150k steps, 500 episodes in RAM, 82 min).
  - Flat baseline (actor on the task goal image, `gcivl_flat_seed0`): 11/30 = 37% (tasks 6 / 0 / 2 / 0 / 3 of 6). OGBench pixel GCIVL with more steps and data: 60%.
  - Planner + GCIVL loop (`obj_loop_gcivl_seed0`, WM eff2): 1/30. Planned light pressed .36 (with a correct WM goal 18/40), other / several lights .37, nothing .27.
  - PRIVILEGED single-event harness (`scratchpad/exec_probe.py`, 40 task-start states, random light, same reset seed): exact cross .88 with the REAL subgoal (simulator lights set to the target and rendered with the arm in place), .90 with the SYNTHESIZED subgoal (subgoal.py; all places of the cross rendered in 47.5% of them).
  - Reading: the executor and the subgoal synthesis work. The loop drew EVERY place from the belief state, so the subgoal differed from the frame wherever a reading was stale.
  - Fix: `closed_loop_objects.py` draws only the entities the event is predicted to change. Re-running the 4x4 loop on CPU (`obj_loop_gcivl2_seed0`).
- 2026-10-10 17:45 RESULT planner + GCIVL loop on puzzle-4x4 with changed-places-only subgoals (`obj_loop_gcivl2_seed0`, CPU, WM eff2): 2/30.
  - Planned light pressed .58 (before the fix .36), nothing .17, wrong .25.
  - Split by WM goal (PRIVILEGED scoring):
    - WM change set = true cross: planned light pressed 32 of 36 = .89 (single-event harness .88-.90);
    - WM wrong: 65 of 131 = .50. The eff2 WM is wrong in 78% of loop events.
  - The executor is no longer the limit on 4x4; the WM is. The structured WM fixes exactly this (cross probe 1.000).
  - Queued (`scratchpad/learned_chain.ps1`, after the structured-WM rungs):
    - FULLY LEARNED 4x4 = `obj_model_eff3rel` + GCIVL;
    - then GCIVL on 4x5 (train, subgoals, flat baseline, learned loop with `obj_model_eff3rel`).
- 2026-10-10 17:48 RESULT G1 (`objects_g1` = raw discovery + discrete places only the views show; movers read raw; places through the purity-gated views; events with the rule acted `obj_events_g1`; PRIVILEGED scoring cs_compare.py).
  - cube-triple: IDENTICAL to v1 on every number (3 cubes 1.5-1.9 cm, visible .75-.88, acted .875, known .982).
  - scene:
    - 5 identities (v1 parity; ungated views gave 10, gated 6); window / drawer |corr| .975 / .968, visible .861 / .804 (v1 .838 / .777).
    - But the buttons read through the views lag: |corr| .953 / .951 (v1 1.0 / .997); 17.3 events / ep (v1 12.5); button acted with the rule .04 / .02 (v1 .81 / .65); overall acted .514 (v1 .776).
    - This is the reading lag that the effector shadow addresses (G2, running in `scratchpad/g12_chain.ps1`).
  - G3 started: `world_model.py --wm-arch rel2`, translation invariant (no absolute positions). Inputs: offsets of k from the acted entity AND from its target, plus the event displacement x - e.
    - It is a new arch name, so the in-flight `rel` rungs are untouched.
    - CPU check on 4x5 running (cross probe must stay 1.000), then cube.
- 2026-10-10 18:05 RESULT G1 + G2 on cube / scene (`obj_events_g12`).
  - Setup: objects_g1; first effector pass -> effector_calib (scene offset (0.15, +7.4) L 12, cube (0.70, +16.1) L 25; PRIVILEGED estimates scene buttons +6.8, cube +16.1) -> events with `--effector-offset`, `--effector-lag L`, `--shadow-places-only`, `--acted-among not-staying`.
  - scene, against v1:
    - acted .812 (v1 .776): button0 .939 (.81), button1 .711 (.646), window .945 (.938), drawer .856 (.862), cube .486 (.541);
    - events / ep 12.88 (12.53); one core .662 (.679); known .852 (.872).
  - cube-triple: acted .843 (v1 .875; cube1 .803 vs .880); events, recall and one core identical to v1; target known .937 (.98).
  - Scene is above v1; cube is 3 points below on acted. The puzzle boards are still running in `g12_chain.ps1`.
  - G3 check: `--wm-arch rel2` on 4x5 keeps the cross probe at 1.000 / 1.000 (data / random states). Comparing entity vs rel2 on cube / scene events now (`scratchpad/wm_arch_cs.sh`, CPU).
- 2026-10-10 18:45 WORLD-MODEL ARCHITECTURE across families (CPU 8k steps). Planner-style eval `scratchpad/wm_plan_eval.py`: Model.step (acted = x, sub-threshold snap) on canonical VAL states; other known entities.
  - The raw wm_eval "event_exact" counts the acted entity's own prediction, which planning overwrites, so this eval replaces it.

    | Board / WM | Unchanged kept | Side effects within tol | Changed set exact |
    |---|---|---|---|
    | 4x5 entity | .999 | .990 | .967 |
    | 4x5 rel2 | .999 | .994 | .976 (cross probe 1.000) |
    | scene entity | .966 | .380 | .852 |
    | scene rel2 | .930 | .046 | .775 |
    | cube entity | .858 | .037 | .696 |
    | cube rel2 | .787 | .170 | .618 |

  - Scene side effects are the LOCKS: button1 -> window appearance 177, button0 -> drawer appearance 160. They are pair-specific, and the window's offset to its button varies as it slides, so the identity-free rel2 misses them.
  - New `--wm-arch pair`: k and the acted entity only (states, target, offsets in place spacings and image units, displacement) + identity embeddings of k and e with dropout .5. No third entity enters, hence no global-state shortcut. Evaluating on 4x5 / cube / scene (`scratchpad/wm_pair_all.sh`).
- 2026-10-10 18:55 RESULTS.
  - (1) `--wm-arch pair`, planner-style eval:
    - 4x5: changed-set exact .972; cross probe .988 data / 1.000 random (light 1 .80 on data states).
    - scene: .829 (entity .852, rel2 .775); lock side effects still .044.
    - cube: .629 (entity .696, rel2 .618).
    - No structured arch matches entity on cube / scene yet. Next: `pairabs` = pair + absolute positions of k, e and the target (the puzzle shortcut came from third-entity states, not absolute positions). Running `scratchpad/wm_pairabs_all.sh`.
  - (2) G1 + G2 on puzzle-4x5 (`objects_g1` + `obj_events_g12`; calibrated offset (0.11, -0.86), reading lag 9).
    - 20 / 20 places, no view places needed.
    - Acted .988 (eff3 .990); presses in one core .993; known changes right .955 (eff3 .976).
    - TARGET KNOWN .885 (eff3 .654).
    - 4x4 / 3x3 still running.
- 2026-10-10 19:15 `--wm-arch pairabs` (pair + absolute positions), planner-style eval: 4x5 .973 (probe .991 / 1.000), cube .555, scene .828. No gain.
  - The pairwise models' cube weakness is "unchanged kept" (.73-.81 vs entity .86): unaffected cubes drift beyond tol_pos.
  - New `--wm-arch pairg` = pair + a sparse CHANGE gate. A per-entity change logit, trained with BCE against the data's changed labels (beyond tol_pos / app_unit_id); at inference the residual applies only where the gate is open. Evaluating on 4x5 / cube / scene (`scratchpad/wm_pairg_all.sh`).
- 2026-10-10 19:38 RESULT G1 + G2 on ALL FIVE boards, one code and setting (`scratchpad/g12_chain.ps1`; PRIVILEGED scoring). Before = eff3 for puzzles, v1 for cube / scene.

  | Board | Places / objects | Acted, before -> after | Target known, before -> after | Other | Calibration |
  |---|---|---|---|---|---|
  | puzzle-3x3 | 9 / 9 (was 8 / 9; the views added 2 places raw frames never show) | .742 -> .794 | .363 -> .379 | known changes .588 -> .628 | offset (-0.31, +2.34), L 18 |
  | puzzle-4x4 | 16 / 16 (views added 1) | .814 -> .800 | .235 -> .327 | known changes .696 -> .732 | offset (0.27, +1.16), L 18 |
  | puzzle-4x5 | 20 / 20 | .990 -> .988 | .654 -> .885 | | offset (0.11, -0.86), L 9 |
  | scene | 5 / 5 | .776 -> .812 | | | |
  | cube-triple | 3 / 3 | .875 -> .843 | | | |

  - The general perception (G1 + G2) is at or above the earlier per-board best except cube (-.03) and 4x4 (-.014).
- 2026-10-10 19:31 RESULT `--wm-arch pairg` (pair + sparse change gate), planner-style eval:
  - 4x5: .973; cross probe .989 data / 1.000 random.
  - cube: .817 (entity .696, pair .629); unchanged kept .992.
  - scene: .872 (entity .852); unchanged kept .983; lock side effects still .059 within tol.
  - pairg is the best single architecture across the three families. Adopted for the next rungs: `scratchpad/pairg_chain.ps1` (G1 + G2 events, objects_g1, subgoals_g1; 4x4 rung + fully learned loop, then 4x5 rung + GCIVL + learned loop).
- 2026-10-10 19:49 RESULT RUNG R1c, puzzle-4x5, STRUCTURED WM (`--wm-arch rel` on `obj_events_eff3`; `obj_model_eff3rel`; scripted PRIVILEGED press arm + grace; seed 0, 30 episodes).
  - Result: 23/30 = 77% (tasks 6 / 6 / 5 / 4 / 2 of 6). The entity WM on the same events gave 43%; old events 27%; pixel baseline 17%.
  - Loop events 427; as predicted .88; WM change set = simulator flips .80 (entity .73).
  - On long tasks, A* within 20k expansions often finds no full plan: first plan None in all 6 task-4 episodes and 4 of 6 task-5. The loop then executes the best partial plan and replans; tasks 4-5 still reach 4 / 6 and 2 / 6.
  - The remaining high-level limit is search depth / cost-to-go on 16-20-press tasks, not the WM.
- 2026-10-10 19:50 GPU queue reordered (`scratchpad/final_chain.ps1`). The 4x4 rel rung (WM stage just started) and `learned_chain.ps1` were stopped.
  - (1) FULLY LEARNED 4x5 = GCIVL (train 150k, subgoals, flat baseline) + loop with `obj_model_eff3rel`.
  - (2) pairg on 4x5 with G1 + G2 events (rung + learned loop).
  - (3) the same on 4x4.
- 2026-10-10 21:37 FIRST COMPLETE FULLY LEARNED RESULT (no privileged input), puzzle-4x5, seed 0, 30 episodes, 1000 steps (`method_pl/obj_loop_gcivl_eff3rel_seed0`).
  - Components:
    - perception + events: eff3, objects of 10-10;
    - structured WM `obj_model_eff3rel` (support + cost-to-go);
    - executor: GCIVL (`method_pl/gcivl`, 150k steps, 82 min) on one-event subgoal images (`method_pl/subgoals`, 20 places; changed places only).
  - Result: 11/30 = 37% (tasks 4 / 5 / 1 / 1 / 0 of 6).
  - Matched flat baseline (same GCIVL actor on the task goal image, same seeds, `gcivl_flat_seed0`): 6/30 = 20% (task 1 6/6). OGBench best pixel 17%.
  - Ceiling with the scripted PRIVILEGED arm on the same model: 77%.
  - Loop events, PRIVILEGED scoring:
    - planned light pressed .85 (scripted .98); WM change set right .80 (same as scripted); planned pressed when the WM is right .85;
    - timeouts 1.33 / episode (scripted .57), each costing 250 steps; events / episode 11.8 (14.2).
  - Gap to the ceiling: executor misses (15%) and per-event timeouts. Levers:
    - a data-derived event timeout (250 is fixed by hand);
    - more GCIVL training (OGBench uses 500k-1M steps, here 150k; 500 of 1000 episodes in RAM).
  - Running: pairg rung on 4x5 (h since 21:41), then the pairg learned loop, then 4x4.
- 2026-10-10 23:00 USER: use the PC's resources in parallel instead of one job at a time. Measured with one job (pairg 4x5 cost-to-go): GPU 20% / 1.6 of 12.2 GB VRAM, CPU 34% of 20 threads, 13.2 of 23.7 GB RAM free.
  - New `scratchpad/sched.py` (jobs: `scratchpad/jobs.json` from `make_jobs.py`; PowerShell wrappers of `local/run_stage.ps1`, which keep the RAM watchdog).
    - A job starts when its dependencies are done, VRAM used + ramping reservations + its estimate <= 10 GB, and free RAM - ramping reservations - its estimate >= 3 GB; at most 4 jobs.
    - Logs: `scratchpad/sched_logs/`.
  - Started 22:59 next to `final_chain.ps1` (pairg 4x5 rung):
    - lever1_4x5: fully learned loop rel + GCIVL with `closed_loop_objects.py --timeout -1` (new: p99 of TRAIN event segments, 120 on 4x5, instead of a fixed 250);
    - model_cube / model_scene: pairg WM + support + cost-to-go on G1 + G2 events.
  - Waiting on RAM: gcivl_cube, gcivl_scene (150k steps each), gcivl_long_4x5 (500k steps, lever 2), lever12_4x5.
- 2026-10-10 23:20 G4 (user: approach (a), compositional rendering from data exemplars): `method/subgoal.py` draws every entity kind objects.py tells apart.
  - Discrete places: state medians, as before.
  - MOVERS: median crop at rest with no agent within 2 object widths. Mask = the connected part around the centre where the median differs from the typical-colour image beyond half the 2-means split (side faces and shadow included; the top-face split left dark rims). Drawn by erasing at the current position with the typical colours (1-px dilated mask) and pasting at the next position (not if covered next).
  - CONTINUOUS places: a bank of frames with the agent clear of the place's own disc. Region = std > 2-means split within the sliding window, minus pixels the agent covers in more than half of the frames (a static robot part next to the scene window). Bank frames keep the region agent-free and mover-free (read position or mover colour; one used to copy a cube into the drawer). Drawn from the nearest bank reading (standardised).
  - Built from `objects_g1`: cube-triple 3 movers (mask 23 / 18 / 26 px); scene 2 buttons + cube (22 px) + window (168 px, bank 821) + drawer (166 px, bank 1482). Panels checked by eye.
  - `closed_loop_objects.py --low gcivl` now uses `render_state` (changed entities incl. the covered flag). CPU smoke runs pass on cube-triple and scene. Data-derived event timeouts: cube 564, scene 442, 4x5 120.
  - Second scheduler (`scratchpad/sched.py` on `jobs2.json`, logs `sched_logs2`): flat GCIVL baselines + FULLY LEARNED loops on cube-triple / scene. Each waits for its pairg model and GCIVL from the first scheduler.
- 2026-10-11 00:05 Results (puzzle 4x5, seed 0, 6 episodes x 5 tasks):
  - PRIVILEGED scripted arm on the pairg WM (G1 + G2 events, `obj_loop_scripted_g12pairg_seed0`): 29/30 = 97% (tasks 5/6/6/6/6), up from 77% with the rel WM. Events per episode 19.2, as predicted .94, no timeouts.
  - Lever 1, LEARNED rel + GCIVL with the data-derived timeout (120, `obj_loop_gcivl_eff3rel_to_seed0`): 9/30 = 30%, versus 37% with 250. Timeouts rise to 3.23 per episode (1.33 before). The learned executor needs longer than the TRAIN event p99, so a short timeout does not help; keep 250 for 4x5 until the executor is faster.
  - LEARNED pairg + GCIVL with G1 subgoal images (`obj_loop_gcivl_g12pairg_seed0`): 2/30 = 7%. Acted .80 (rel learned .82), as predicted .73 (.77), timeouts 1.30 (1.33). 28 of 28 failures reach the 1000-step limit. Steps per acted event match (median 31 versus 30). The 30-point drop is not explained by these averages.
  - Diagnosis running: the PRIVILEGED single-event harness (`scratchpad/exec_probe.py`, 60 trials) with old versus G1 subgoal images.
- 2026-10-11 00:40 Diagnosis of the 7% (LEARNED pairg 4x5). PRIVILEGED scoring throughout.
  - Executor harness with G1 subgoal images (`scratchpad/exec_probe_g1.py`, 60 trials, 150 steps): real goal image .92 exact cross, synthesized .92. The images are not the cause.
  - Event outcomes in the loop are the same for rel and pairg learned (acted and as predicted .61 vs .58, only others changed .12 vs .15, no change .06 vs .05).
  - The difference is the number of events. Solved episodes need 18.8 events with pairg (scripted) versus 14.8 with rel. At ~55-70 learned steps per event (timeouts included) that is past the 1000-step budget.
  - Cause: the PLAN, not the WM.
    - At the end of the first plan, the belief equals the simulator (0 wrong lights in every checked episode).
    - In 6 of 19 checked scripted episodes the simulator was one light off the goal: lights 3, 6 and 7, which sit under the arm's start position.
    - An entity unseen at the start is assumed at its goal. When first seen during an event, its reading silently becomes the event baseline (`start_state[newly]`), so the as-predicted test never flags it. The plan runs to its end one light off, and Lights Out needs ~10 more presses to fix one light.
  - Fix (general, `closed_loop_objects.py`, default on, ablation `--no-replan-on-surprise`):
    - an entity first seen during an event whose reading differs from what the WM predicts from the belief the plan assumed (`belief0`) forces a replan at the event end;
    - events record `unexpected_ids`.
  - Running (`scratchpad/surprise_4x5.ps1`): LEARNED pairg + GCIVL + G1 subgoals (`obj_loop_gcivl_g12pairg_surp_seed0`), then the scripted ceiling (`obj_loop_scripted_g12pairg_surp_seed0`).
  - Note: the cube / scene LEARNED loops queued in `sched_logs2` will start with this code.
- 2026-10-11 01:00 Replan-on-surprise, LEARNED pairg 4x5 (`obj_loop_gcivl_g12pairg_surp_seed0`): 4/30 = 13% (tasks 1/3/0/0/0), versus 7% without. Within noise.
  - Events per episode drop from 14.2 to 8.6, but timeouts rise from 1.30 to 1.80 per episode.
  - Timeouts by planned light:
    - light 3: 21 of 54 (pairg without surprise 11/39, rel 6/31);
    - lights 15 and 7: 9 and 8.
- Root causes found (PRIVILEGED inspection, no task rule involved):
  - F1, perception. On raw frames, G1 discovery found puzzle-4x5 light 3 as a 2-px FRAGMENT: (24,36) background and (25,39) edge. Its centre is (37.5, 24.5) versus (40.1, 24.9) for the light. The view discovery saw it whole (8 px).
    - Lights 2, 3, 6 and 7 sit under the arm's rest pose (visible 59-80% of frames).
    - G1 keeps raw places and adds only view places farther than w, so the fragment won.
    - Effects: state contrast .17 (others ~.27), readings that flip 0.267 <-> 0.433 while the simulator light stays, WM inputs off its training distribution.
    - Fix (`objects.py`, default on, ablation `--keep-fragments`): a raw place takes the pixels of a view place within w that has more than twice its area. On the five dev boards (`scratchpad/fragment_check.py`) only this place changes; scene, cube, 4x4 and 3x3 are unchanged.
  - F2, subgoal regions. `subgoal.py` took a discrete place's region from a 2w window around it. The per-pixel median of state-split frames turns any correlation of a neighbour's state, or of the arm's whereabouts, into full contrast. Regions held neighbour pixels: 4x5 G1 9 of 20 lights, 4x4 7 of 15; scene buttons 0.
    - Drawing one light also redrew a neighbour in a state the event does not reach.
    - Fix (default on, ablation `--region-window`): region within the place's own disc (objects.py) dilated by 1 px.
- 2026-10-11 01:00 Scheduling.
  - Two schedulers would both have seen the RAM freed by gcivl_cube and started ~14 GB at once; run_stage watchdogs kill below 1.2 GB free.
  - Scheduler 2 was stopped (nothing started). Scheduler 1 keeps only its running gcivl_cube; its pending jobs are no-ops.
  - Everything pending is in ONE scheduler (`scratchpad/sched.py` on `jobs4.json`, logs `sched_logs4`), in priority order:
    1. flat_cube, learned_cube;
    2. g1b_prep: objects_g1b (F1) + effector pass + calibration + obj_events_g12b;
    3. g1b_4x5_rest: pairg rung with the scripted ceiling, subgoals_g1b (F2), LEARNED loop `obj_loop_gcivl_g12bpairg_seed0`;
    4. gcivl_scene, flat_scene, learned_scene;
    5. f2_4x4 and f2_4x5 (F2 alone, `subgoals_g1r`);
    6. gcivl_long_4x5, then lever12_4x5.
  - lever12 now isolates lever 2: timeout 250, `--no-replan-on-surprise`, output `obj_loop_gcivl_long_seed0`.
  - The cube / scene LEARNED loops run with replan-on-surprise (the default).
- 2026-10-11 00:56 PRIVILEGED scripted ceiling with replan-on-surprise, same pairg model (`obj_loop_scripted_g12pairg_surp_seed0`): 30/30 = 100%, up from 97%.
  - Events per solved episode 18.8 -> 14.6; steps 588 -> 465; as predicted .94 -> .96.
  - This confirms the surprise diagnosis.
  - The learned loop does not show the gain yet (13%): light-3 timeouts dominate; see F1 / F2 above.
- 2026-10-11 01:20 puzzle 4x4 with pairg on G1 + G2 events (`final_chain.ps1` step 3). The loop code includes replan-on-surprise.
  - PRIVILEGED scripted ceiling (`obj_loop_scripted_g12pairg_seed0`): 30/30 = 100% (old eff2 entity-WM ceiling: 60%). Events per episode 5.9, steps 247, first plan found 1.0.
  - Flat GCIVL: 11/30 = 37% (unchanged).
  - LEARNED pairg + GCIVL + `subgoals_g1` (old 2w-window regions; `obj_loop_gcivl_g12pairg_seed0`): 5/30 = 17%.
    - Acted .71, as predicted .60.
    - Events: acted with a different change set .41, acted and set as predicted .30, only others changed .23, no change .06.
    - Steps per event: median 47 (scripted 31). Timeouts .43 per episode. All 25 failures end at the 500-step limit of 4x4.
  - Open issue (4x4, general): place 15 (button2, added by the views) has no subgoal exemplar. Only 9 of 23,756 sampled frames are agent-free on raw frames: it sits under the arm's rest pose and is read only through the views. Place 1 (button1) is visible in 7.5% of frames but has 7,725 agent-free samples.
  - Running: f2_4x4 (F2 subgoal regions, same model), started 01:20 in `sched_logs4`.
- 2026-10-11 01:45 Results.
  - 4x4 F2 subgoal regions (`f2_4x4`, `obj_loop_gcivl_g12pairg_f2_seed0`, same model): 6/30 = 20%, versus 17% with the 2w-window regions.
    - Acted .71 -> .80, as predicted .60 -> .70, only others changed .23 -> .15.
    - Executor accuracy is better; success is within noise.
  - cube-triple, first numbers.
    - GCIVL 150k steps: 106.8 min (shared GPU).
    - Flat GCIVL: 3/30 = 10% (task 1 3/6).
    - FULLY LEARNED loop: pairg WM on G1 + G2 events, GCIVL on G4 subgoal images (mover erased and pasted at its next position), data-derived timeout 564, replan-on-surprise (`obj_loop_learned_g4_seed0`): 3/30 = 10% (task 1 3/6).
    - 0.2 events per episode: the executor almost never completes a planned event.
    - PRIVILEGED look at the 26 timeouts (`scratchpad/cube_fail.py`): 12 idle (no cube moved > 2 cm), 12 a cube moved without the planned event completing, 2 a cube left in the air.
    - The objects report has no identity -> cube mapping, so "planned vs other cube" is not separated.
    - First plan found in 77% of episodes.
    - The cube low level does not do the events; the high level is not the limit here.
  - Scheduler 1 stopped (its gcivl_cube was done; only no-ops were left). g1b_prep started 01:40 in `sched_logs4`.
- 2026-10-11 02:10 USER: build the low level as GCIVL with STATE goals.
  - `method/gcivl.py --goal state --events <obj_events dir>`. Image mode is unchanged (same nets / keys / RNG use).
  - Goal: the entity state after the event (K x 6: position, appearance, covered); the WM's predicted next state in the loop. No subgoal image.
  - Inputs: the current frame (Impala-small, 3 channels) and an MLP on [belief, goal, goal - belief] per entity (positions / 64), joined (1024) before the heads.
  - States per frame: the closed loop's memory = latest valid reading of the episode (labels_train.npz valid / cov_valid), else the episode's first valid one.
  - success(s, g): every entity within the event thresholds (tol_pos, thr_app_id + 1e-4, covered bit; a hidden goal entity only needs to be hidden).
  - Goals: value current .2 / next event .2 / trajectory geometric .3 / random .3; actor next event .5 / uniform future .5.
  - Random crops shift the state positions with the image.
  - `closed_loop_objects.py --low gcivl` detects a state-goal checkpoint and feeds goal_vector(belief S, predicted_next). `--subgoals` is not needed.
  - CPU smoke runs: gcivl state mode on 4x5 / cube / scene; loop on cube (untrained actor: timeouts as expected).
- Schedule (`scratchpad/state_runner.ps1`, task bapgx9ns0). One GCIVL in RAM at a time: each training waits for >= 13 GB free twice a minute apart. Per family: train (150k steps, 500 episodes, as the image version), then the FULLY LEARNED loop with the same WM / objects / events / seed / timeout as that family's image-goal loop:
  1. cube-triple: obj_events_g12, obj_model_g12pairg, timeout -1, `obj_loop_gcivl_state_seed0`;
  2. scene: same settings;
  3. puzzle-4x5 on the F1 pipeline: obj_events_g12b, obj_model_g12bpairg, objects_g1b, timeout 250, `obj_loop_gcivl_state_g12b_seed0`;
  4. puzzle-4x4: g12, timeout 250.
  - Then `state_runner_done.flag`. sched_logs4's image GCIVL slots (gcivl_scene, then gcivl_long_4x5) now wait for that flag and for free RAM.
- 2026-10-11 02:15 USER: first make sure the HIGH LEVEL is good on every task.
  - New PRIVILEGED diagnostic low level `method/oracle_low.py` (`closed_loop_objects.py --low oracle`, arm "PRIVILEGED DIAGNOSTIC: OGBench plan-oracle low level"). OGBench's plan oracles (the controllers that generated the play data, noise 0) execute each planned event.
    - Entity -> object: best_ref of objects_report (scene: cube_y / window / drawer / button0-1), else the block whose projection into 'front_pixels' is nearest the believed position.
    - Block target: table point under the target pixel, or the top of a block within w/2.
    - Drawer / window: the slide end (closed / open) whose handle pixel displacement matches the event's.
    - Camera check (`scratchpad/cam_check.py`): projected block centres match the frame to ~0.5 px.
    - Episodes record `oracle_events` (mapped / unmapped).
  - CPU smoke run on cube-triple: oracle executes events; task 3 solved (647 steps). Task 2: no first plan, and every event "not as predicted" with all three cubes predicted changed.
- 2026-10-11 02:25 HIGH-LEVEL BUG (general, numeric). `world_model.Model.app_tol` = thr_app_id ~2e-6 for exact identities, but the support values are rounded to 4 decimals while a mover's reading keeps its identity colour at full precision (0.42519668 vs 0.4252).
  - Every cube therefore differed from its own prediction by ~4e-6 and counted as changed.
  - The loop's as-predicted test and belief fill failed on every cube event.
  - `scratchpad/wm_changed_check.py`, cube VAL events: exact predicted change set .015 against raw states vs .762 against canonical.
  - Fix: app_tol floor 1e-4 (rounding error <= 5e-5; state changes > .05). After the fix, raw .762 = canonical .762; scene .537 (unchanged); puzzle 4x5 .797 (unchanged). This simple metric is stricter than the planner-style one.
- Ceiling runs:
  - `scratchpad/ceil_runner.ps1` (task bh5aj8uwe): oracle ceiling cube (started 02:13 with the OLD tolerance = a "before fix" point), oracle ceiling scene (fixed code), then the puzzle 3x3 pairg rung (scripted ceiling).
  - `scratchpad/ceil_runner2.ps1` (task b6mom660p): cube oracle ceiling again with the fix (`obj_loop_oracle_g12pairg_fix_seed0`).
  - Correction: world_model.py was saved at 02:15:01 and the cube oracle loop started at 02:15:20 (log local20261011021520), so `obj_loop_oracle_g12pairg_seed0` HAS the fix. The rerun (`ceil_runner2.ps1`) was stopped.
- 2026-10-11 02:20 Scene WM by acted entity (`scratchpad/scene_lock_check.py`, VAL events, exact predicted change set on known entities):
  - window .94, drawer .96, button0 (drawer lock) .82, button1 (window lock) .67;
  - cube .33.
  - Data, for cube-acted events: true change sets {window} 48, {cube} 34, {button1} 31, {drawer} 23 of 187. ATTRIBUTION errors: a carried cube is the entity nearest the effector, so other changes go to it.
  - WM on cube moves: predicts no change for 96 / 187.
  - Lock mechanics (scene_env.py): button state 0 = red = locked, and it also turns that drawer / window handle red. So a button press has a visible side effect on the drawer / window appearance.
  - Tasks 4-5 put the cube INTO the drawer and close it (a hidden goal).
- 2026-10-11 02:25 cube-triple ORACLE CEILING (first oracle version; `obj_loop_oracle_g12pairg_seed0`): 16/30 = 53% (tasks 6/5/3/2/0). Events per episode 1.7, as predicted .54, first plan .73.
  - Two failure causes belonged to the DIAGNOSTIC oracle, now fixed in `oracle_low.py`:
    - The oracle's last pose is random and could leave the arm over the moved cube, so events never ended within 564 steps (task 4: the cube was at its goal, then a timeout). Fix: return to the episode's start effector position after each event.
    - Identity -> block by projection picked a lower block of a 3-stack (task 3). Fix: the objects report's colour mapping (identity -> cube).
  - Re-run queued (`scratchpad/ceil_runner3.ps1`, task b0xj24mfl -> `obj_loop_oracle_g12pairg_v2_seed0`).
  - The scene oracle ceiling (started 02:25:18) already uses the corrected oracle (files saved 02:25:09 / 02:25:11).
- HIGH-LEVEL gap found: HIDDEN GOAL entities.
  - cube-triple task 5 = a 3-stack at (0.425, 0.2). Only the top cube is read in the goal image; the others are "hidden there".
  - The planner proposes no target for a hidden goal entity (candidates = its visible goal + absolute data prototypes), so no plan within 20k expansions in 6/6 (best_h ~ 0).
  - Same class: scene tasks 4-5 (cube inside the closed drawer).
  - The cost-to-go itself is fine: on VAL pairs it rises with event distance (cube 1.45 / 2.39 / 2.70 at 1 / 3 / 6 events; puzzle 5.5 / 10.0 / 10.9; scene 6.9 / 8.6 / 14.7).
- Scene events, PRIVILEGED truth per object (`scratchpad/ev_eval_cs.py`):
  - one-core / acted-right: button0 .82 / .94, button1 .98 / .71, drawer .74 / .86, window .78 / .95;
  - cube .26 / .49 (cube visible 57% of frames; its passive moves inside the drawer also count as interactions).
- cube-triple per object: one-core .82 / .60 / .78; acted right .85 / .80 / .87; all .728 / .843.
- 2026-10-11 02:40 Scene high level, diagnosed with the oracle loop. The oracle opens the drawer 0 -> -0.16 and the window 0 -> 0.18 in ~50 steps when asked directly (`scratchpad/oracle_slide_test.py`).
  - DIAGNOSTIC bug: the oracle chose the slide end by the nearest pixel displacement. The read place moves less than the handle (window 5.7 px read vs 15 px handle when open), so it picked "closed" for open requests: scene task 1 0/6 with no events. Fix: the end the asked displacement points to. The corrected v2 ceilings (scene, cube) run in `scratchpad/ceil_runner4.ps1` (task b126n8dbs), then the 3x3 rung.
  - The loop now logs, per episode, `start_belief`, `goal_read`, `start_rest`, `first_plan_events`; sim_state also records the drawer / window slides; oracle `slides` decisions.
  - HIGH-LEVEL bug, scene LOCKS (task 2: unlock, close, lock). The belief was right (both buttons red = locked), but the first plan was close drawer, close window, close drawer (depth 3; 6 events needed). The oracle could not move the locked drawer: timeout 442 of 750 steps.
    - Event support for "close the open drawer while locked" = sigmoid .41 > threshold .27, so the WM predicts the drawer closes.
    - In general the model knows the locks: TRAIN drawer / window events with the button flipped to locked get median support logits -5.2 / -4.7 and predicted slide motion .01 / .00 (unlocked .76 / .76) (`scratchpad/lock_support_check.py`). This state is an uncertain rare combination.
    - TRAIN drawer events with a red button before: 151 / 2071, of which only 16 moved the drawer > 1 px (window 5 / 30): label noise, not real locked motion.
- 2026-10-11 02:55 scene ORACLE CEILING v2 (corrected slide ends, colour mapping, home after events; confirm rest; `obj_loop_oracle_g12pairg_v2_seed0`): 5/30 = 17% (tasks 4/0/1/0/0). As predicted .36, first plan .57.
  - Task 2: plans close the LOCKED drawer first, timeout 442.
  - Task 3: oscillates between press button1 / close window.
  - Tasks 4-5: no first plan (cube inside the closed drawer = hidden goal).
  - Root cause of task 3 (`--trace-steps` debug option: per-step reading of the planned entity):
    - With the arm 0.4 px over button1, the place's reading flips to "white" (0.333) while the simulator button is still 0 (red). The reading under the agent comes from the SeeThrough lookup.
    - The event ends "as predicted" at step 12 before the oracle presses (~25). When the arm leaves, the reading returns to red, so the button is pressed again in the next plan, and so on.
  - Fixes (loop, general):
    - FAILURE MEMORY (default on, ablation `--no-failure-memory`): an event that timed out is forbidden as the first event of plans from the same belief key (`world_model.plan(forbid=...)`, `Model.same_target`).
    - `--confirm clear`: settle / contradict only with the agent > w/2 away (an existing option; to become the default if it also holds on puzzle).
  - Fixes (PRIVILEGED oracle):
    - fail-fast: done + home + m + grace idle steps without the event ending counts as a failure;
    - the gripper opens on the way home (it held the locked handle);
    - the home latch (the arm drifts around the 2 cm mark);
    - home = high and back in the arm sampling bounds (the start pose covered the scene buttons).
  - Single-episode checks (CPU, confirm clear): scene task 3 solved in 305 steps (both slides moved, buttons right).
- cube-triple STATE-GOAL GCIVL trained (150k steps, 68.3 min, 7.5 GB). Its loop started 03:05. cube oracle v2 started 02:57:45 with all fixes above but confirm rest.
- 2026-10-11 03:15 Results.
  - cube-triple ORACLE CEILING v2 (corrected oracle + failure memory, confirm rest; `obj_loop_oracle_g12pairg_v2_seed0`): 22/30 = 73% (tasks 6/5/5/6/0). Task 5 = 3-stack, hidden goal.
  - cube-triple FULLY LEARNED with the STATE-GOAL GCIVL (`obj_loop_gcivl_state_seed0`): 0/30 (image goals: 3/30).
    - 0.7 events per episode; timeouts: another cube moved 16, idle 11.
    - Training log: succ_share .45-.57 of value goals. Unchanged cube states make most relabeled goals already reached: a weak signal (to fix: goals that differ from the current state).
- 2026-10-11 03:20 g1b_prep had CRASHED at 01:43 (and the job hung with no child until 03:18). F1 replaced only disc and centre; the view place's per-pixel weights `w` stayed those of the 2-px fragment (place_reading broadcast (2,1) vs (8,3) in see_lookup).
  - Fix: the whole view place dict is copied. Checked with `--discover-only` (3.1 min): light 3 at (40.12, 24.88), 8 px.
  - Re-run `scratchpad/g1b_prep2.ps1` (task becf42ia1). sched_logs4's g1b_4x5_rest starts when obj_events_g12b exists. gcivl_scene (after the failed g1b_prep) will not start from sched_logs4.
- 2026-10-11 03:30 More HIGH-LEVEL fixes (scene diagnosis, single CPU episodes):
  - GOAL OCCLUSION (`closed_loop_objects.py`, ablation `--no-goal-occlusion`): a mover unseen in the goal image is "hidden there" unless the goal image's agent lies within w of its current position (then unknown). Scene task 1 read "hide the cube" with the unmoved cube under the goal arm.
  - FAILURE MEMORY: no longer keyed on the exact belief; cleared by the next as-predicted event (a locked drawer still moves slightly, so the key changed after every attempt: task 2 tried 7 times).
  - PRIVILEGED oracle: slide end = the nearer of the place's two read clusters (2-means of its events' after positions; the open one is further along the handle's closed -> open projection). A partly opened window read past its open end, so the displacement rule toggled it open / closed 12 times in task 1.
- HIDDEN GOALS (`world_model.py`, defaults on; ablations `--no-hidden-protos`, `--cov-known-only`):
  - The covered-bit loss now includes entities hidden after the event (the WM predicted hiding in 0/205 TRAIN drawer-closings and 0/43 VAL stackings).
  - `hidden_goal_support`: movers get placement and pre-hiding prototypes (cube-triple 8 -> 16-17 per cube; scene cube 3 -> 12), plus COVER relations (m put covers j: mode of target - j position, >= 5 events and >= 25% within 2 tol_pos). cube-triple: all 6 pairs, offsets (0, -2.4 to -3.0) px; scene: none.
  - Planner: `hidden_goal_targets(G)` puts a hidden goal entity under the visible goal entity that covers it, through chains (3-stack: middle and bottom candidates). `candidates_batch` adds relative targets (on a visible entity + cover offset) and these.
  - Re-training cube / scene pairg + support + h (`obj_model_g12pairgh`) and oracle ceilings with confirm clear: `scratchpad/hl_runner.ps1` (task b528p9o07). The v3 ceilings (old model, corrected loop) run first.
- 2026-10-11 03:45 puzzle 3x3 SCRIPTED CEILING with pairg on G1 + G2 (`heldout_puzzle3x3 ... obj_loop_scripted_g12pairg_seed0`): 3/30 = 10% (tasks 3/0/0/0/0).
  - Acted .55, as predicted .43.
  - Place 8 (view-added, top-middle light = button1) was planned 54 times and changed twice.
  - PRIVILEGED: its reading |corr| .099, full view in 19% of frames, never read through the agent.
  - Its disc was a 2-px FRAGMENT (27,32),(27,33): light pixels (|corr| .89/.55), but the light's other pixels (corr .8-.94) are agent-free in only 2-50% of frames. The robot's resting column covers it, and both raw and view discovery saw only the edge, so F1 (view replaces raw) could not help.
  - The other boards: the only places below half the median area are this one and 4x5 light 3 (fixed by F1).
  - Fix F1b (`objects.py grow_fragments`, off with `--keep-fragments`): a location place below half the median place area grows by the pixels within one object width whose chromaticity, along the fragment's main axis of colour change and in frames where both are agent-free (refine_agent pixel mask), follows its own (|corr| >= the 2-means split, floor .5).
    - Intensity does not work (.12-.42): the states differ in colour. With the coarse token mask the fragment was agent-free in 28 / 8040 frames.
    - 3x3 place 8: 2 -> 9 px (rows 27-29, cols 30-33 = the privileged light pixels), centre (32.5, 27.0) -> (31.3, 27.8), snapped (D 3e8).
  - Re-run: `scratchpad/p3x3_runner.ps1` (task b07xvf72e) -> objects_g1c, obj_events_g12c, rung g12cpairg.
- 2026-10-11 03:47 4x5 g1b prep done (objects 8.3 min, peak 9.5 GB; effector pass 9.7 min; calib offset (0.139, -0.794) lag 9; events g12b 10 min). sched_logs4 started g1b_4x5_rest at 03:51.
- 2026-10-11 04:00 ORACLE CEILINGS v3: old pairg models, corrected loop (confirm clear, failure memory, goal occlusion, oracle slide clusters and high-back home).
  - cube-triple `obj_loop_oracle_g12pairg_v3_seed0`: 21/30 = 70% (tasks 6/5/4/6/0).
  - scene `obj_loop_oracle_g12pairg_v3_seed0`: 10/30 = 33% (tasks 4/0/6/0/0), up from 17%. Task 3 is 6/6.
  - Scene task 2 still tried the locked drawer 6 times: each replan chose a slightly different closed target (several prototypes within a few px), so the (e, x) memory did not match. Fix: failure memory per ENTITY (no plan may start with an event on an entity whose event failed), cleared by any ended event that observed a change. A t_ref ordering bug (cleared after the reset, i.e. never) was fixed before any run.
  - Scene tasks 4-5: no plan or an empty best plan, then the loop idled for hundreds of steps (hidden goal; the gh models are the fix under test).
- 2026-10-11 05:09 puzzle 3x3 SCRIPTED CEILING with F1b (objects_g1c, obj_events_g12c, pairg; calib offset (-0.379, 2.136) lag 19): 1/30 = 3%.
  - Place 8 is now |corr| .83 (was .10), visible .31.
  - Per planned entity (events / acted / as predicted): centre light 2 = 69 / 43 / 0; light 8 = 25 / 4 / 4.
  - Cause: the PRIVILEGED scripted arm returns to the episode's start pose, which covers the 3x3 top-middle and centre lights. Their presses were never seen, so the planner pressed light 8 again and again (the simulator toggled back and forth).
  - Fix (diagnostic arm, `--home-start-pose` restores the old one): the scripted arm waits high and back in the arm sampling bounds, as the oracle. Re-run `scratchpad/p3x3_home2.ps1` (task bk5vessp0, LoopTag _home2).
  - Note: the learned executor has no retreat; perception under the arm stays an open issue for the METHOD.
- cube gh WM (`obj_model_g12pairgh`; WM 30k steps in 2.2 min; support 0.3 min; cost-to-go running for 80+ min):
  - wm_eval: event exact .736, acted .978, side effects within tol .035.
  - VAL stackings predicted to hide the lower cube .209 (old model 0), false hiding on visible staying cubes .006.
  - Prototypes 17 / 15 / 17, cover relations 6.
- 2026-10-11 05:20 HIDING in the WM, VAL F1 of "an entity becomes hidden" (`scratchpad/hiding_check.py`; transitions are 2.6% of scored covered bits on cube, 1.3% on scene).
  - cube, unweighted (gh): P .56 R .21 F1 .31; weight N_still/N_flip (gh2, x20): P .17 R .56 F1 .26 (false hiding .10); weight sqrt (gh3): P .36 R .47 F1 .40 (false hiding .031).
  - scene, old model: R 0; gh2 (x47): P .07 R .83 F1 .13 (false hiding .12); gh3: P .41 R .53 F1 .46 (false hiding .009).
  - Default now `--cov-balance-power .5` (ablation `--no-cov-balance`), chosen by VAL F1 on both families.
  - Note: the scene "gh" WM that hl_runner trains after 05:28 therefore has the gh3 setting.
  - cube gh3 cost-to-go + oracle ceiling: `scratchpad/gh3_cube.ps1` (task bc4w7vm8o).
- 2026-10-11 05:30 puzzle 3x3 SCRIPTED CEILINGS (g12c model):
  - start-pose home 3%; high-back home 0% (it covers the top row; see-through readings stale at that pose);
  - confirm clear 0% / 7% (timeouts: the event never ended while the pressed light stayed under the arm).
  - The arm cannot wait clear of the 3x3 board: z <= .25; automatic park pose (below) still covers 27 place px.
  - Fixes:
    - (loop, general) with confirm clear, the "acted entity hidden" end condition uses the confirmable flag (conf_) instead of rest. A light read through the arm counts as rest but can never settle.
    - (PRIVILEGED arms) automatic PARK POSE: five poses at the top of the arm sampling bounds, each reached in a separate env and read by the method's perception; the pose leaving the most place pixels clear (within w/2 of the agent) wins.
  - After these, 3x3 events end and fills happen, but the 3x3 WM predicts WRONG crosses (centre press -> [1, 2, 7] or [2, 3, 5]; true {1, 2, 3, 5, 8}). wm_eval event exact .51.
  - Cause in the data: for centre presses the acted light's after state is unknown in 70% of TRAIN events and the top-middle in 40% (under the arm); 12% of events have no known change.
  - OPEN: after-state reading under the arm in events_objects.py (3x3).
- 2026-10-11 05:35 State-goal runner stopped (it waited for 13 GB of free RAM since 03:14; user priority is now the high level). Cube state-goal results kept. Its done flag was never written, so sched_logs4's image GCIVL slots stay parked.
- 2026-10-11 05:45 CONTINUITY in events (`events_objects.py`, default on, ablation `--no-continuity`).
  - Nothing changes between consecutive events of an episode (an event holds every change). So an entity unknown after event i takes its known state before event i+1, and one unknown before event i+1 takes its state after event i.
  - Applied after the acted attribution, which is unchanged.
  - 3x3 test: `scratchpad/p3x3_cont.ps1` (task bxj8wihee): obj_events_g12d (objects_g1c, g1c calibration) -> pairg rung (scripted ceiling, confirm clear + automatic park pose).
  - Other boards keep their events until this is checked.
- 2026-10-11 05:38 puzzle 4x5 with F1 (objects_g1b, obj_events_g12b, pairg; `obj_loop_scripted_g12bpairg_seed0`, run with replan-on-surprise, failure memory, confirm rest): PRIVILEGED scripted ceiling 30/30 = 100%. Events per episode 14.7, steps 458, as predicted .94, first plan .67.
  - sched_logs4 g1b_4x5_rest continues: subgoals_g1b (F2), then the LEARNED loop.
- 2026-10-11 05:43 3x3 continuity: 0 states filled on TRAIN and VAL. An entity unknown after event i is also unknown before event i+1: the arm stays over it between the events.
  - WM g12dpairg = g12cpairg (event exact .504 vs .508, side effects .80).
  - 3x3 stays OPEN. The resting arm covers the top-middle / centre lights in the play data (incomplete labels) and in the loop. Fixing it needs reading those states later than the next event without assuming which events could change them.
  - The rung's h + scripted loop (confirm clear, park pose) still run for the record.
- 2026-10-11 06:00 Hidden goals, offline planner probe (`scratchpad/plan_probe.py`: logged start belief / goal of the v3 cube task-5 episodes, gh3 WM + support, cost-to-go borrowed from g12pairg).
  - Three fixes in `world_model.py`:
    1. Hidden-goal targets skip coverers that are unknown or unread at (0, 0); relative targets only on read visible entities; placements only from events with a known target.
    2. `at_goal`: a hidden goal entity with hidden-goal targets must be hidden AND within 2 tol_pos of one of them. "Hidden anywhere" let the planner hide the 3-stack's blocks elsewhere.
    3. BUG: `hidden_goal_targets` did `ok = np.asarray(known)` then `ok &= ...`, changing the CALLER's `known` in place. Hidden-goal entities (goal position (0, 0)) became "unknown", so the planner ignored them.
       - It affected every cube / scene loop started after ~03:20, including the v3 ceilings (cube task 5, scene tasks 4-5 understated). Puzzle loops are unaffected (all goal positions non-zero).
       - Fixed with `np.array` (copy).
  - After the fixes, task 5 plans found 6/6:
    - ep 5: block 0 bottom, block 2 middle, block 1 top (the 3-stack; final covered = goal);
    - ep 2: depth 4, stacked;
    - ep 0: 2-stack only, because the bottom block was unread in the goal image (under the arm; goal occlusion -> unknown).
- 2026-10-11 05:58 puzzle 4x5 FULLY LEARNED with every fix so far (`obj_loop_gcivl_g12bpairg_seed0`): 13/30 = 43% (tasks 6/4/1/1/1). Best METHOD number on 4x5 so far.
  - Setup: objects_g1b (F1), obj_events_g12b, pairg WM, subgoals_g1b (F2 regions), image-goal GCIVL 150k, replan-on-surprise, failure memory, goal occlusion, confirm rest, timeout 250.
  - Acted .889, as predicted .851, timeouts .8 per episode, events 13.0, steps 784.
  - Earlier: rel 37%, pairg-G1 7% / 13% (surprise); flat GCIVL 20%; OGBench best pixel 17%; scripted ceiling 100%.
- 2026-10-11 06:20 GOAL READING, colour rule (`closed_loop_objects.py`; replaces the 03:30 agent-distance rule, ablation `--no-goal-occlusion`).
  - A mover unseen in the goal image is hidden there unless its own colour still shows in more than half of its footprint at its current place (the arm is transparent). Then its goal is its current state.
  - Evidence (`goal_occlusion` per episode, oracle, 3 episodes per scene task):
    - task 1 (cube unmoved under the goal arm): agent .44, colour .80 -> stays;
    - tasks 4-5 (cube inside the closed drawer): agent .75-1.0 (the arm's shadow), colour .00-.07 -> hidden. The old rule had made them "unknown", so the planner ignored the cube.
  - With the scene gh3 WM (model_hborrow.pt = gh3 WM + support + cost-to-go from g12pairg), tasks 4-5 now read the cube hidden. First plans put the cube at the in-drawer prototype (21.8, 17.0), press button0, close the drawer, but never OPEN it: the WM / support miss the "drawer open" precondition. Oracle runs time out.
- 2026-10-11 06:25 The gh / gh3 cube cost-to-go trainings were STOPPED: ~32 candidates per state (old models ~9) made h steps very slow (14k steps in 139 min).
  - Ceilings now use the cost-to-go borrowed from g12pairg (`obj_model_g12pairgh3/model_hborrow.pt`, cube and scene): `scratchpad/gh3_ceilings.ps1` (cube bj6cbrxu0, scene b7c9r75lz), confirm clear, `obj_loop_oracle_g12pairgh3b_seed0`.
  - The 3x3 g12d rung continues (h at 96k steps).
- 2026-10-11 06:15 ORACLE CEILINGS with the gh3 WMs (cost-to-go borrowed from g12pairg; confirm clear; goal colour rule; known-copy fix; failure memory per entity; park pose) = `obj_loop_oracle_g12pairgh3b_seed0`.
  - scene: 10/30 = 33% (tasks 4/0/6/0/0), the same as v3.
    - Task 2: unlock with button0 as predicted, then press button0 AGAIN (relock), and close-drawer attempts fail. Repeats.
    - Tasks 4-5: the cube is put at the in-drawer place while the drawer is closed (no "open first" precondition), so oracle failures.
  - cube-triple: 17/30 = 57% (tasks 6/5/1/5/0), BELOW v2 73% / v3 70%.
    - Task 5 is still 0 although offline plans are right. Closed-loop stacks: events alternate as-predicted / not, 1000 steps run out.
    - Task 3 (unstack a start 3-stack) 4/6 -> 1/6: one block is moved back and forth.
    - Suspects: (a) the colour goal rule calls a block hidden when the goal arm covers its NEW place (the rule only looks at its current place); (b) the gh3 WM's spurious hide / appear predictions (false hiding .031).
  - Conclusion for now: the hidden-goal machinery works offline but not yet in closed loop. Best cube ceiling remains v2 / v3 (70-73%).
- 2026-10-11 06:20 Checks on the gh3b cube failures.
  - Task 3 goals read all three blocks visible in 5 / 6, so the goal colour rule is not the cause. The regression comes from the gh3 WM: spurious hide / appear while a start 3-stack is taken apart; its lower blocks are unseen at the start and assumed at their goals.
  - Task 5 goals now read 1-2 blocks hidden (the colour rule finds no block colour at their old places).
  - Closed-loop stacking disagrees: the WM predicts the lower block hidden while perception still reads its side faces (not as predicted -> replans) until 1000 steps.
- OPEN high-level items, by family:
  - scene task 2: a double button press after unlocking. The drawer appearance mixes the handle colour (lock) with the drawer; goal tests on it.
  - scene tasks 4-5: precondition "drawer open" before putting the cube in.
  - cube task 5: a stacked block's visibility (side faces) vs the covered bit learned from events.
  - cube task 3 with gh3: spurious hide / appear.
  - 3x3: after-states under the resting arm.
- 2026-10-11 06:22 ABLATION F2 alone on 4x5 (`obj_loop_gcivl_g12pairg_f2_seed0`; old G1 pairg model, light 3 still a fragment; subgoals_g1r = own-disc regions; loop with surprise + failure memory): LEARNED 12/30 = 40% (tasks 5/4/3/0/0).
  - Acted .815, as predicted .765, timeouts 1.2.
  - Same model with the 2w-window regions: 13% (with surprise), 7% (without).
  - F2 is the large step for the learned executor; F1 adds the rest (43%).
- 2026-10-11 06:19 3x3 g12d rung (continuity no-op, confirm clear, park pose): scripted 0/30 (WM event exact .504). As expected; 3x3 stays open.
- 2026-10-11 07:30 scene task 2, plan logging (`--log-plans`):
  - The planner circumvented the "first event only" failure memory with a NO-OP PAIR (press button0 twice: unlock, relock) before the locked drawer.
  - Fix: failure memory keyed on (state key of the belief at the failure, entity), applied at every search node, kept for the episode. A no-op pair returns to the same key; one press (unlocked) is a new key.
  - Still 0/2 on task 2: the WM / support believe the locked drawer AND window can close, so the planner tries the window next, then button presses in odd orders. The scene cost-to-go rates this 6-event task at 2.8.
  - Root cause upstream, in the EVENTS: of 151 drawer-acted TRAIN events with a red (locked) button, 135 did not move the drawer. These are reading flicker attributed to the drawer, which teaches the support model that drawer events happen while locked.
- 2026-10-11 08:00 Event quality checks (PRIVILEGED, VAL).
  - `scratchpad/false_events.py`: events whose core misses every true interaction by more than the margin.
    - Margin +-2 frames: scene .120 (cube-acted .29, button1 .18), cube-triple .014, 4x5 g12b .000, 3x3 g12c .083.
    - Margin +-10: scene .011; +-30: .001.
    - So the scene events are REAL, offset from the simulator toggles by the reading lag. The "flicker events" hypothesis is wrong.
  - `scratchpad/place_reliability.py` (no privileged input): place readings that disagree with clear full-view readings on both sides inside a stable span.
    - Adjacent-agent full-view readings: wrong .000-.007 (scene, 3x3, 4x5).
    - See-through readings: wrong .000 (puzzles), .049 (scene window), .125 (scene drawer).
    - Place readings are reliable; the scene lock problem is not label noise from readings.
  - Remaining scene lock cause: attribution / support. The support model gives sigmoid .41 > .27 (the 2% VAL quantile) to closing the open drawer while locked.
- 2026-10-11 08:20 EVENT SUPPORT is the weak link in scene (`scratchpad/support_threshold.py`: VAL genuine events vs context-corrupted copies, the training recipe).
  - Default support (6k steps, width 256): at its thresholds (2% genuine quantile) it rejects only .21-.43 of corrupted contexts; at the balanced thresholds .35-.51, keeping .79-.94 of genuine.
  - Longer / wider (60k steps, width 512, 3.5 min; `obj_model_g12pairg_sup2`): close-drawer-while-locked .38 -> .00, unlocked .56 -> .69; close-window-while-locked .31 -> .02.
  - But the 2% quantile thresholds collapse to ~0 (overconfident), so abstention stops.
  - NEW (`world_model.py plan`, `closed_loop_objects.py --feas-weight`, default 1): A* priority += feas_weight * sum of -log(event support). Plans avoid implausible events softly; the hard threshold stays.
  - Testing on scene task 2 (CPU).
- 2026-10-11 09:40 HIGH-LEVEL fixes and diagnoses (scene task 2 and up).
  - A* (`world_model.py plan`), two general bugs once plans carry a feasibility cost:
    - the goal test ran when a goal node was GENERATED, so the first goal found won whatever its cost. Now a goal is returned when it is POPPED (nodes popped before it in a batch are expanded first);
    - states were deduplicated by plan length, so a cheap long path to a state was pruned by a short expensive one (closing the locked drawer reached "closed" in 1 event and pruned the 3-event unlock / close / lock path). States now keep their cheapest path cost (lam * length + feasibility cost).
    - Without a feasibility cost (feas_weight 0) the search is unchanged.
  - PREDICTION TOLERANCE (`Model.set_prediction_tolerance`, loop `--pred-tol-q`, default .75): a WM prediction is compared with a reading (goal test on imagined states, the loop's as-predicted and surprise tests) within the WM's own precision.
    - Per entity: the .75 quantile of the WM's VAL appearance error on entities changed as a SIDE effect; at least the event threshold, at most half the median appearance change (distinct modes stay apart).
    - Values: scene window .0039 -> .0160, drawer .0039 -> .0148 (a button press recolours the handle: WM error median .002-.011, VAL after-reading IQR .01-.08); cube, 4x5, 4x4 unchanged; 3x3 lights 2/7/8 1e-4 -> .11-.13 (half their colour gap).
    - Before: every scene button press was "not as predicted" and the loop pressed again (gh3b task 5: btn1 x8 per episode).
  - SUPPORT variants on scene (`scratchpad/support_cf.py`, VAL counterfactuals; `t2_support_cmp.py`, the task-2 start beliefs):
    - every model drops unlocked drawer / window moves ~100x when the button is set red;
    - but in the task-2 start context the default model lets "drawer -> closed with the locked look" pass (.38 > threshold .27);
    - the 60k-step models (sup2 mixed negatives, sup3 = new `--p-one 1` single-entity swaps) reject the locked closes (<= .013) but give legal presses .03-.32. sup3 is not better than sup2.
  - EVENT RECALL is the main scene data problem (`scratchpad/cube_recall.py`, PRIVILEGED: sim cube moves of > 2 cm between rests):
    - scene VAL: 550 true cube moves, found by an event .33, with the cube as the acted entity .17;
    - cube-triple VAL, per cube: found .76-.89, acted .53-.75.
    - Cause: a mover is a rest observation only with no agent pixel within half an object width. The scene agent mask is coarse (~38% of the image with the arm's shadow) and the play arm hovers over the cube between its manipulations. Example: TRAIN episode 0, t 500-836, four cube moves, no event.
    - Consequence: the cube placed INTO the drawer (scene tasks 4/5) is visible after an event in 8 TRAIN events. The WM cannot learn "cube -> drawer". Before-states of events also jump: the scene cube moves > 6 px between consecutive events in 19.5% of event pairs (cube-triple 5.6-10.3% per cube).
  - Running (CPU, parallel): oracle ceilings with these fixes, `obj_loop_oracle_c5_seed0` (scene and cube, gh3 WM with the borrowed cost-to-go), and the 4x5 scripted ceiling `obj_loop_scripted_c5_seed0` (regression check, was 100%).
- 2026-10-11 10:50 Ceilings c5 / c6 and fixes (PRIVILEGED oracle / scripted low levels, seed 0, 6 episodes x 5 tasks).
  - c5 (A* fixes + prediction tolerance; gh3 WM, default joint support, borrowed cost-to-go):
    - scene 11/30 = 37% (tasks 5/0/6/0/0);
    - cube 20/30 = 67% (6/6/1/6/1): first 3-stack success; task 3 is the gh3 regression (1/6);
    - 4x5 scripted: tasks 1-2 12/12 but task 3 0/5 (no first plan; stopped). REGRESSION.
  - REGRESSION CAUSE: the feasibility cost (default since 08:20) on EVERY event made weighted A* nearly breadth-first (the cost-to-go ignores it), and the pop-time goal test needs the cheapest goal popped: no plan in 20k expansions.
  - FIX (`world_model.py`, loop `--feas-margin-q` .9): only the event cost above the .9 quantile of GENUINE VAL event costs of that acted entity enters plans; a goal generated with zero feasibility cost returns at once. Margins: 4x5 .75-.87, cube .82-.88 nats.
  - FAILURE MEMORY now matches states within the event / prediction tolerances (was the exact state key): handle-colour reading noise gave each belief a new key, so the locked window was retried 4-6 times per episode.
  - PAIRWISE SUPPORT (new, `train_support.py --kind pair`, `event_support.make_pair_support`): one NCE logit per (event, other entity); event cost = summed evidence against compatibility beyond chance. Displacement features are magnified (the drawer travels 3 px).
    - scene task-2 contexts: close the locked drawer 4.1-4.2 nats, presses .2-.7;
    - but closing the locked window toward the goal's LOCKED look stays cheap (.4-.6): a big move with a locked-look target never occurs in the data, positive or negative;
    - genuine unlocked drawer moves have cost q90 4.5, so a q90 margin erases the lock signal. The failure memory carries the locks.
  - c6 scene (pairwise support + tolerance memory): 10/30 = 33% (5/0/5/0/0). Task 2 now fails on the HANDLE COLOUR: after a press the WM's predicted drawer / window look misses the reading, so the loop replans; plans then insert no-op press pairs to "fix" the look.
  - EVENT RECALL FIX (`events_objects.py --mover-rest effector`, new default): a mover is a rest observation when the effector contact point is more than one object width away (was: no agent pixel within half a width).
    - PRIVILEGED rest-run check (`scratchpad/rest_criteria.py`), true rest intervals with a rest run: scene .19 -> .76; cube-triple .62-.76 -> .92-.97; runs that are mostly carried frames <= .010.
    - Events g13 (same thresholds as g12), cube moves found: scene .33 -> .83 (acted entity right .17 -> .75); cube-triple .76-.89 -> .97-.98 (acted .53-.75 -> .79-.84).
    - Cube moves ending raised (stacks): cube-triple .69-.88 -> .97.
    - g13 pairg WMs (VAL of their own events): scene event_exact .52 -> .65, cube-triple .73 -> .81.
  - SCENE CUBE IN THE DRAWER is never read. `objects.read_frame` drops changed-pixel components that touch ANY place, and the drawer's place pixels surround its interior.
    - Allowing continuous places (`scratchpad/mover_rule.py`): the cube in the drawer is read in .85 of frames.
    - But on table frames the reading jumps (q50 18 px): the red LOCKED handles pass the cube colour test (cube chroma .479/.224 vs lock red .504/.223). Not adopted; scene tasks 4-5 stay blocked on perception.
  - Running: c7 = g13 WMs + pairwise support + borrowed cost-to-go for scene and cube; 4x5 scripted with the margin fix.
- 2026-10-11 11:30 c7 results and further fixes.
  - 4x5 scripted ceiling with the feasibility margin (`obj_loop_scripted_c7_seed0`): 29/29 so far (tasks 1-5). The c5 regression is fixed.
  - c7 = g13 WMs + pairwise support + borrowed cost-to-go:
    - scene 10/30 = 33% (5/0/5/0/0);
    - cube 18/30 = 60% (6/6/3/3/0): task 3 1 -> 3/6, task 4 6 -> 3/6.
  - Cube task-5 failures are mostly WRONG STACK ORDER: the goal reading sees only the top cube. Either order of the two hidden cubes satisfied the hidden-goal test (cover chains of depth 1 and 2 both allowed).
    - Yet the goal image shows their side faces: 11-16 px of a 29 px median area (`scratchpad/goal_stack.py`). The reader drops anything less than half visible.
  - PARTIAL GOAL HINTS (`objects.read_partial_movers`, loop default, ablation `--no-goal-hints`): hidden goal movers with a side face visible in the goal image (more than 1/4, at most 1/2 of their median area) are ranked by image height; a lower face is deeper. `hidden_goal_targets(depth=...)` keeps only cover chains of that depth.
  - DERIVED APPEARANCES (`Model.set_derived_appearance`, loop `--derived-r2` .5): an appearance predictable from the other entities' states and the entity's own position (kNN R^2 on VAL) is ignored by the goal and prediction tests. Greedy flagging: of two appearances that predict each other, only the more predictable is flagged.
    - R^2 values: scene drawer .91, window .73 (both flagged), buttons .61-.64 (not flagged once the handles leave the predictors); puzzle lights -.25 to -.02; cube constant appearance.
  - c8 scene (derived appearances): 9/30 = 30% (5/0/4/0/0). New failure mode: a target that differs only in a derived appearance was "arrived" at once, and the plan replanned every 5 steps (25-41 replans per episode).
    - Fix 1: derived appearances are no event targets (`candidates_batch`).
    - Fix 2: an event that ends with NOTHING changed enters the failure memory, like a timeout.
  - Running: scene c9 with these fixes; cube task 5 with partial goal hints (`obj_loop_oracle_c8t5_seed0`).
- 2026-10-11 12:30 Scene task 2 solved for the first time; loop / planner fixes.
  - c9 scene (derived-only targets dropped, no-change events remembered): 11/30 = 37% (5/0/6/0/0). Task 2 plans then inserted no-op pairs (press a button twice, open and close the window): the failure memory still compared the derived handle colour, so a pair changed the predicted colour and re-allowed the locked drawer.
    - Fix: failure matches and the search state key ignore derived appearances.
  - c10 scene: 13/30 = 43% (tasks 5/2/6/0/0). First task-2 successes.
  - Remaining task-2 failure: after a failed event, the plan unlock / close / lock had to pass a recently left state (the avoid list), and no plan was found.
    - Fix: a failure (timeout or no-change event) clears the avoid list; the failure memory stops the repeat.
    - Also: event targets of an entity with a derived appearance keep its current look; only the position moves. The support model saw an impossible "locked look while unlocked" target before.
  - 4x4 scripted ceiling with all loop changes up to c9 (`obj_loop_scripted_c9_seed0`): 28/30 = 93% (task 3 4/6; was 100% on 2026-10-11 01:45).
  - Cube task 5 with partial goal hints (`obj_loop_oracle_c8t5_seed0`): 1/6. The ranks are read correctly, e.g. depth {2: 1, 1: 2} matches the sim. But:
    - plans placed the top cube before the middle one: "a cube must be below" is an EXISTENCE precondition, which a pairwise support cannot express;
    - one episode found no plan.
    - Now running: cube with the g13 WM and a JOINT support (`obj_model_g13pairg_joint`).
  - 3x3 label check (PRIVILEGED, VAL): known after-state labels are right .83-.93 per light; stale (true change missed) .06-.10; spurious .01-.06. Press-to-neighbour change rates are far below 1 for lights under the resting arm (centre press: neighbours .45-.66), so the WM's cross for the centre is wrong (event exact .10). Source not yet resolved.
- 2026-10-11 13:30 Failure relevance and a learned stacking direction; scene c11 / c12; 4x4 without feasibility cost.
  - c11 scene (avoid list cleared on failure, derived look kept in targets): 10/30 = 33% (4/0/6/0/0).
  - FAILURE RELEVANCE (`Model.set_failure_relevance`, loop `--failure-relevance` .3 nats): a failed event is remembered against the entities RELEVANT to it only. Relevance of k to events of e = the mean pairwise-support cost added by giving k the state of another event, over genuine VAL events of e.
    - Learned scene relevances: window <- btn1 4.6, cube .71; drawer <- btn0 3.0, cube .78; btn0 <- drawer 2.2; btn1 <- window 2.0; cube <- drawer 1.45.
    - Cube-triple: every cube relevant to every other (4.1-5.2).
    - Motivation: the planner re-allowed a failed event by changing an UNRELATED entity first (move the cube away and back, press the other button; debug log `scratchpad/dbg_t2`).
  - Partial goal hints are now ranked along the LEARNED cover direction (mean offset of the cover relations), replacing the hand-written "lower in the image = deeper". Same order with this camera.
  - c12 scene: 10/30 = 33% (4/1/5/0/0). Scene runs c9-c12 range 33-43%, with task 2 at 0-2/6: six episodes per task cannot separate these variants.
    - Remaining task-2 loop: press btn0 twice, then the locked drawer. The WM's predicted state of the HIDDEN cube changes over the presses, the cube is relevant to the drawer, so the failure no longer matches.
  - 4x4 scripted without the feasibility cost (`obj_loop_scripted_c12f0_seed0`): 28/30 = 93% (task 5 4/6), the same total as with it (c9: task 3 4/6). Not attributable to the feasibility cost.
  - STATUS REPORTED TO THE USER (Vietnamese), with an audit of hand-set choices made while looking at failures on the seed-0 evaluation episodes:
    - derived R^2 threshold lowered .8 -> .5 after seeing the scene window at .73;
    - pairwise magnification constants 16 / 10;
    - the effector rest criterion chosen with PRIVILEGED VAL recall;
    - q75 / q90 / .3 nats.
    - No per-task code. The reported ceilings are optimistic; the proposed next step is to freeze the configuration and evaluate fresh seeds, held-out families and the end-to-end loop.
