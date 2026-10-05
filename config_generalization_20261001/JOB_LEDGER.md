# Job ledger: config_generalization_20261001

Renamed 2026-10-01 from compositional_wm_20261001 (run dir compositional_wm -> config_generalization) to avoid confusion with the closed compositional trajectory-JEPA branch (latent_scope_20260909); log paths of 56541/56544/56545 moved with it.

User decision 2026-10-01: test P1 offline on OGBench visual puzzle and cube-triple before any method work.

- **RQ.** Offline GCRL collapses as compositional complexity grows (puzzle-4x6 14% even with 1B state transitions; pixel cube-triple ~21%, quadruple 14%). Does a world model give the compositional generalization a goal-conditioned value function lacks?
- **P1 (offline).** A world model predicts event outcomes (button press, block placement) in unseen configurations almost as well as in seen ones. A goal-conditioned value function on the same representation degrades with configuration novelty.

Outputs are in /mnt/data/nhatnc129/jepa/config_generalization/.

## 2026-10-01
Quota at submit:
- GPU: 4/5.5 h (5th-ranked user has 11 h in October), so GPU work waits.
- CPU: 17/41 h.

| Job | Work | Resources | State |
|---|---|---|---|
| 56541 | FAILED at the cube step (env model not built before reset); downloads and puzzle stats completed (in log). Download visual-puzzle-{3x3,4x4,4x5,4x6}-play and visual-cube-triple-play (train and val, HF mirror ryanhoangt/ogbench_data). Config novelty: puzzle Hamming distance of val configs/events to the nearest train config (capped at 4); cube stack structure and nearest-neighbour block-position distance | main, 4 CPU, 64 GB, 2 h | FAILED (cube part) |
| 56544 | p1.sh SMOKE on visual-puzzle-3x3 (200 WM steps, 300 value steps): code path only | mig, 6 CPU, 160 GB, 40 min | COMPLETED 3:18 (200 steps = 0.17 min; 3x3: 3051 clean events, all configs seen) |
| 56545 | Cube-triple stats only (fixed: gymnasium.make + reset before reading qpos addresses) | main, 4 CPU, 64 GB, 1 h | COMPLETED 18:46 |

### Puzzle configuration novelty (56541 log)
| env | train steps | unique train configs / all | val press events | pre-event novelty 0 / 1 / 2 / 3 |
|---|---|---|---|---|
| 3x3 | 1.0M | 512 / 512 (100%) | 3092 | 3092 / 0 / 0 / 0 |
| 4x4 | 1.0M | 23,412 / 65,536 (36%) | 3001 | 1104 / 1897 / 0 / 0 |
| 4x5 | 3.0M | 83,420 / 1.05M (8%) | 8868 | 660 / 6670 / 1538 / 0 |
| 4x6 | 5.0M | 143,251 / 16.8M (0.9%) | 14533 | 130 / 2690 / 10487 / 1226 |
A press toggles 3-5 lights; about 31 presses per 1000-step episode. Held-out configurations are unseen
but close (Hamming 1-3) to training ones, so novelty is a graded axis within 4x5 and 4x6.
GPU cap reached at 2026-10-01 16:30 UTC (own 5 h incl. the CTA job 56504 vs cap 6 h): full P1 runs wait.

### Cube-triple arrangement novelty (56545)
3.0M train steps. Steps by blocks resting on another block: 0 = 72.5%, 1 (two-block stack) = 25.5%,
2 (three-block tower) = 1.9% (train) / 1.7% (val). Validation nearest-neighbour distance to any training
step (max over blocks): median 3.8 cm, 90% 5.2 cm, 99% 6.5 cm. Towers are rare but seen, so cube
novelty is a rarity/distance axis, weaker than puzzle's unseen-configuration axis. qpos block slices 14/21/28.

## 2026-10-02
GPU quota at submit: 8/10.5 h (5th-ranked user has 21 h).

| Job | Work | Resources | State |
|---|---|---|---|
| 56657 | P1 puzzle-4x5: world model 60k steps (batch 128); probes; event prediction by novelty; GCIVL value on frozen latents, 100k steps | mig, 6 CPU, 160 GB, 1 h 40 min | COMPLETED 1:02 — INVALID as a P1 test (see below) |
| 56691 | probe_diag.py on the 56657 WM: linear/MLP probes for button states from pixels (16x16), encoder patch tokens, CLS before projector, latent | mig, 6 CPU, 160 GB, 35 min | COMPLETED 5:00 |
| 56695 | p1_patch.sh SMOKE (puzzle-4x5, base 56657 encoder; 300 predictor steps, 40 train episodes, 300 value steps) | mig, 6 CPU, 160 GB, 30 min | COMPLETED 4:59; patch probe ceiling: val exact config 98.7% (vs 0% for the CLS latent) |
| 56701 | P1 patch mode, puzzle-4x5: predictor 30k steps on the frozen 56657 encoder; probes/value on 1000 training episodes; value 60k steps | mig, 6 CPU, 160 GB, 40 min | COMPLETED 27:15 (GPU 9/10.5 h per sreport at submit) |

### 56657 result (puzzle-4x5): measurement invalid
WM trained well on its own objective (val prediction MSE .0011 vs copy .28), but a linear probe reads
button states from the latent at only 68% bit accuracy on training frames (chance 50%; exact
configuration ~0%), so the decoded ceiling ("real target") is as low as the WM's prediction
(effect accuracy .67 both; copy .33). Every novelty bucket is flat because the readout is at floor.
The GCIVL value on the same latents: Spearman with true presses .47/.51/.51 (novelty 0/1/2), event
sign agreement .70/.69/.68. Neither the WM nor the value comparison is interpretable until the
representation carries the button state. Diagnosis job 56691.

### 56691 probe diagnosis (puzzle-4x5, held-out val frames; bit accuracy / exact configuration)
pixels 16x16: 99.99% / 99.95%; encoder patch tokens: 99.99% / 99.78%; CLS before projector: 69% (MLP 71%) / 0%;
world-model latent: 68% / 0%. The encoder sees every button; the LeWM CLS bottleneck discards the
combinatorial object state. Fix for the P1 test: patch-token world model on the frozen encoder
(scripts/patch_wm.py, train_patch_wm.py; p1_puzzle.py --patch-wm). Value net in patch mode: per-token
linear 192->16, flatten, MLP.

### 56701 P1 result, puzzle-4x5, patch-token latents (valid measurement)
Readout ceiling: probe on real latents 99.4% exact configuration at every novelty level; predictor val
MSE .0037 vs copy .207 (standardised tokens). By post-event novelty (Hamming distance 0 / 1 / 2 to the
nearest training configuration; n = 657 / 6595 / 1520 events), mean [95% CI]:
- WM effect accuracy (toggled buttons): .846 [.828,.864] / .846 [.841,.851] / .844 [.833,.856];
  persistence .9997 / .9997 / .9995; exact configuration .61 / .59 / .60 (two-block rollout the same).
- GCIVL value on the same latents: Spearman with -d* .69 / .72 / .71; event sign agreement
  .756 [.723,.788] / .755 [.745,.766] / .736 [.713,.758].
Reading: neither the world model nor the value degrades with natural configuration novelty; both are
flat within CIs. P1 as stated (differential degradation) is NOT supported on this axis. Natural
held-out configurations are only 1-2 buttons away from training ones, too weak to separate
interpolation from compositional generalization. Side finding: the LeWM CLS latent discards the
button state (68% bits) while patch tokens keep it (99.99%).

## 2026-10-03
GPU quota at submit: 10/24.5 h (5th-ranked user has 49 h).

| Job | Work | Resources | State |
|---|---|---|---|
| 56899 | Re-evaluate puzzle-4x5 with the 56657 encoder and 56701 patch predictor (no retraining). New breakdowns: value event-sign by goal distance d* (1-2, 3-4, 5-6, 7-8, 9+); mean V by d* | mig, 6 CPU, 160 GB, 40 min | COMPLETED 7:47 |
| 56900 | Structured held-out region, puzzle-4x5. Configurations with buttons 6, 7, 11, 12 all ON (central 2x2 block, 1/16 of configurations) removed from base-WM/encoder training (60k), patch predictor (30k), probes and value (60k). Test split: events/pairs inside vs outside the region | mig, 6 CPU, 160 GB, 2 h 15 min | COMPLETED 1:24:56 |

### 56899 goal-distance breakdown (puzzle-4x5, same models as 56701)
Value event-sign agreement (does V move the right way after a press) by true distance to goal d*:
1-2 presses .993 [.989,.996] (n 2193); 3-4 .923 [.911,.934]; 5-6 .645 [.623,.667]; 7-8 .504 [.477,.528]
(chance); 9+ .450 [.422,.477]. Mean V by d*: -1.5, -4.5, -7.9, -10.4, -12.3, -13.3 for d* = 0-5, then flat at
about -14 for d* = 6-13. The value is reliable only within ~4 presses and blind beyond ~6. The WM's
per-event accuracy is local by construction (effect .848, exact .585; unchanged).
- OGBench puzzle evaluation tasks, true minimal presses init -> goal (GF(2)): 3x3 [2,5,8,9,7]; 4x4 [4,6,6,6,7];
  4x5 [4,10,14,16,20]; 4x6 [6,8,12,16,24]. Four of the five 4x5 tasks lie beyond the ~4-press radius where
  the value is informative, consistent with model-free success <=17% on 4x5/4x6.

### 56900 structured held-out region (puzzle-4x5; buttons 6,7,11,12 all ON never seen in any training)
8.4% of training windows removed. Events whose post-event configuration is in the region (n 522) vs
outside (n 8250): WM effect .890 [.876,.903] vs .896 [.892,.899]; exact .62 [.58,.66] vs .66 [.65,.67];
readout ceiling 1.00 both. Value: Spearman .667 (state or goal in region, n 672) vs .710; event sign
.733 [.708,.758] vs .753 [.743,.763]; goal-distance wall unchanged (.99/.92/.65/.50/.46 for d* 1-2 ...
9+). Neither model breaks on an unseen combination of individually seen button states; the dominant
failure is the value's goal-distance wall, not configuration novelty.

### Hamming distance vs true distance (math only, no model; login node, seconds)
Puzzle-4x5, 6000 pairs with goals placed k = 1..20 random presses away. Hamming distance between the
state and the goal (what any image or latent L2 measures) by true minimal presses d*: 5.5 (d* 1-2), 8.8
(3-4), then flat at ~10 = n/2 (random) for d* 5-6, 7-8, 9-12 and 13-20. Spearman(Hamming, d*) = .33. For
far goals (d* >= 7) only 43% of optimal presses reduce Hamming. Beyond ~5 presses a board is as different
from the goal as a random board, so neither latent L2 nor a perceptual distance can guide search there.
The GCIVL value's wall (sign .65 at d* 5-6, .50 at 7-8) sits at the same radius.
