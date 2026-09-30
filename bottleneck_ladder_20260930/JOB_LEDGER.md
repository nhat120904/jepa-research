# Job ledger: bottleneck_ladder_20260930

Oracle ladder for released LeWM planning on Cube/Reacher dev roots 0-99 (same
manifest as cem_stopping / consensus_planning; test roots untouched). For the 20
recorded plan-1 candidates per root: L0 WM cost history 1 (released), L1 WM cost
history 3 (the context length the predictors were trained with; the released
eval config leaves PlanConfig.history_len at 1), L2 latent cost of the true
rendered endpoint, L3 physical. Closed loop CEM-30 with history 3 under the same
10 CEM seeds as consensus single30 (history 1): paired.
Outputs /mnt/data/nhatnc129/jepa/bottleneck_ladder/.

## 2026-09-30
Quota before: GPU 210/243 h, CPU 1788/1945 h; queue empty.

| Job | Work | Resources | State |
|---|---|---|---|
| 56052_[0-1] | Smoke roots 0-1 per task | mig, 4 CPU, 32 GB, 15 min | COMPLETED; h1 recompute rel err 4e-6; Reacher dataset-action replay error 1e-8 (alignment correct); Cube replay error 0.0004 / 0.29 (contact, frames are restored not replayed) |
| 56055 | Full dev, 4 shards/task | mig, 4 CPU, 32 GB, 30 min, %2 | COMPLETED (11-12 min Cube, 4 min Reacher shards) |
| 56058 | Analysis, afterok 56055 | main, 2 CPU, 8 GB, 10 min | COMPLETED |

### Ladder result (analysis_56058.json; 93/100 roots have 10 steps of history)

| | Cube | Reacher |
|---|---|---|
| Spearman(cost, physical) over 20 candidates: h1 / h3 / true endpoint | -.08 / -.07 / .29 | .17 / .34 / .58 |
| Top-1 success among 20: h1 / h3 / true / random / any | .76 / .76 / .81 / .61 / .82 | .32 / .41 / .66 / .32 / .86 |
| Informative roots top-1 h1 / h3 / true | .85 / .85 / .97 (n=33) | .37 / .47 / .76 (n=79) |
| Median predicted-vs-true endpoint error h1 / h3 | 156 / 153 | 48 / 41 |
| **Closed loop CEM-30, 10 paired seeds: h1 / h3** | .675 / .669 (-0.6 [-2.4,+1.0]) | **.658 / .853 (+19.5 [+15.1,+24.0])** |

Reacher h3 beats h1 on every one of the 10 seeds (+10 to +30 pp). Reading: on
Reacher the binding bottleneck is the truncated planning context (velocity),
then rollout error (h3 -> true endpoint), then the latent metric (true .66 vs
any .86). On Cube context does not matter; rollout error does (informative
top-1 .85 -> .97 with true endpoints).

| Job | Work | Resources | State |
|---|---|---|---|
| 56078 | Upstream evaluator (le-wm eval.py logic) Reacher, history_len 1 vs 3, seeds 42-44 x 50 episodes | mig, 4 CPU, 32 GB, 45 min | submitted |
| 56079_[0-3%2] | Reacher controls, 93 roots: static3 (3 identical frames, zero action history), h2 (10 seeds); replan-every-block h1/h3 (3 seeds) | mig, 4 CPU, 32 GB, 50 min | submitted |
Quota before: GPU 210/243 h, CPU 1789/1945 h; planned +4.1 GPU-h, +16 CPU-h.
| 56081 | Cube replan-every-block, history 1 (3 seeds, 93 roots) | mig, 4 CPU, 32 GB, 40 min | submitted |

### Verification results (2026-09-30)
- 56078 COMPLETED 11 min. Upstream evaluator (le-wm eval.py logic), Reacher, seeds
  42-44 x 50 identical episodes: history_len 1 = 76/80/74 (mean .767), history_len 3
  = 84/86/88 (mean .860); paired +9.3 pp [+2.0, +16.7], 24 vs 10 discordant. Only
  the second plan gets history (upstream warm-up), so this is a lower bound.
- 56079 COMPLETED (4 x ~21 min). Reacher controls, 93 roots, paired seeds:
  static3 (three copies of the current frame, zero action history) .666 vs h1 .658
  (+0.8 [-3.0,+4.4]); h2 .862 (+20.4 [+15.7,+25.2] vs h1); h3 .853 (h3-h2 -1.0).
  => gain is motion information, one past frame suffices. Replan every block with
  terminal-only cost collapses: h1_rh1 .280, h3_rh1 .276 (3 seeds): the fixed-horizon
  terminal cost keeps moving the arrival 25 steps ahead.
- 56081 COMPLETED (4 x ~5 min). Cube replan-every-block, history 1, 3 seeds, 93
  roots: .613 vs released receding-5 .663 (-5.0 [-9.7,-1.1]). Feedback with the
  fixed-horizon terminal cost hurts on both tasks.
- Note: DINO-WM's plan.py also feeds a single frame (obs_0 = arr[:, 0]) to a
  predictor trained with num_hist 3, so context truncation is not LeWM-specific.

| Job | Work | Resources | State |
|---|---|---|---|
| 56106 | Feedback vs objective timing, two-frame context, roots 0-49 x 3 seeds, Cube + Reacher: rh5_term, rh1_term, rh1_min (arrive at any predicted block), rh1_shrink (terminal at min(5, blocks left)), rh1_min_c6 (6 CEM iterations, same WM evaluations as rh5) | mig, 4 CPU, 32 GB, 70 min, %2 | submitted |
- 56106 (feedback objectives, two-frame context, 3 seeds): Cube 46 roots rh5_term .681,
  rh1_term .645, rh1_min .674 (-0.7 [-7.2,+5.8]), rh1_shrink .645, rh1_min_c6 .659;
  Reacher 36/47 roots at read time rh5_term .843, rh1_term .296, rh1_min .843 (+0.0),
  rh1_shrink .694, rh1_min_c6 .778 (-6.5). Freeing the arrival time removes the
  collapse but per-block feedback adds nothing; compute-matched is worse. Closed.
