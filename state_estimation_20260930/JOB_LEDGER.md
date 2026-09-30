# Job ledger: state_estimation_20260930

Direction 1 (user-approved 2026-09-30): estimate the planning start state of a frozen
latent world model from the executed history, instead of planning from the single
current frame. Released LeWM evaluators (le-wm/config/eval) plan with history_len 1
although every predictor is trained with 3 frames; the first plan of an episode sees
one frame even with history_len 3 (the history buffer starts empty).
Outputs under /mnt/data/nhatnc129/jepa/state_estimation/.

Closest prior art (read 2026-09-30): AdaJEPA 2606.32026 (test-time weight updates,
history-3 frozen baseline already strong under dynamics shift), Sandwich-Residuals
2609.21740 (online residuals around a frozen predictor), FIRM-WM 2609.22816 (trained
recurrent state with a dynamic fibre; Reacher 92.7 vs LeWM 88.0), DALI 2508.20294,
PLUME 2606.11396, Implicit State Estimation via Video Replanning 2510.17315.

## Phase 1: where does the start state matter? (2026-09-30)
GPU before: 214/243 h (5th-ranked 486 h). Planned ~10 GPU-h.

| Job | Work | Resources | State |
|---|---|---|---|
| 56115_[0-3] | Smoke: official_eval.py per task (reacher, pusht, cube, tworoom), 4 episodes, h1 and h3+prefill | mig, 4 CPU, 48 GB, 25 min | submitted |
| 56116_[0-3%2] | afterok 56115. _1 (pusht) FAILED at m10 seed 43: a start of exactly 10 made load_prefill ask for row -1 (fixed: that unused action is NaN-padded; runs with start > 10 unchanged). Resumed by 56128. Per task: official sampling h1/h3 seeds 42-44; min-start-10 sampling h1/h3/h3+prefill seeds 42-44; 50 episodes each | mig, 4 CPU, 48 GB, 90 min | submitted |
| 56117_[0-3%2] | pred_error.py per task: 600 dataset windows, rollout error h1/h2/h3, innovation offset, MHE smoothing | mig, 4 CPU, 48 GB, 40 min | submitted |
| 56124_[0-3%2] | innovation.py per task: 3000 windows (strided row reads), history-3 rollout error vs innovation corrections (post-hoc, in-loop disturbance, EMA, per-dim gain, ridge on held-out episodes) | mig, 4 CPU, 48 GB, 45 min | submitted |

| 56128_1 | PushT resume of 56116_1 (MODE=m10, same run dir, existing outputs skipped) | mig, 4 CPU, 48 GB, 40 min | submitted |
| 56130_0 | (cancelled before start: dist 1.0 diverges offline, replaced by 56138) | | CANCELLED |
| 56138_0 | innov_closed.sh Reacher: min-start 16, seeds 42-44: h3, h3+prefill, h4+prefill (control = h3+prefill), h4+prefill+innovation dist 0.25 / post 0.5 / dist 0.5 | mig, 4 CPU, 48 GB, 90 min | submitted |
| 56139_[0-1] | Released sampling (min-start 0), history 3 + partial prefill (envs with <10 prior steps get what exists), reacher + pusht, seeds 42-44, into the 56116 run dir. load_prefill/PrefillPolicy now append per env (unit-checked identical to the batched append for full prefill). | mig, 4 CPU, 48 GB, 40 min | submitted |

### Early results
- 56115 smoke OK on all four tasks (prefill path runs).
- 56117_0 Reacher (600 windows): latent MSE by horizon 1..5 — h1 .069/.100/.111/.114/.118; h3 .0069/.0109/.0137/.0163/.0193 (h2 = h3); no-motion reference .53-1.49. Innovation is persistent: cos(nu_t, next h3 error) = .58; adding 0.5 nu cuts the 1-block error 29% and the 5-block error 7%. MHE smoothing of clean encodings: no gain.
- 56116_0 Reacher (released evaluator, 3 seeds x 50): official h1 76.7, h3 86.0 (+9.3 [+2.0,+16.7]). Min-start-10 sampling: h1 74.0, h3 84.7, **h3+prefill 93.3** (h3p-h3 +8.7 [+2.0,+16.0], 21 vs 8; h3p-h1 +19.3 [+11.3,+27.3], 36 vs 7). The blind first plan costs ~9 pp even with history_len 3.
- 56116_1 PushT: official h1 88.7, h3 88.0 (no change: only the second plan gets history). Min-start seed 42: h1 78, h3 80, h3+prefill 88 (5 vs 1). Seeds 43-44 resumed in 56128.
- 56117_1-3 (600 windows, latent MSE at horizon 5): PushT h1 .071 / h3 .066; Cube h1 .039 / h3 .030; TwoRoom h1 .133 / h3 .135. MHE smoothing never helps.
- 56124 (3000 windows, held-out episodes, change in history-3 error at horizon 1 / 5): Reacher cos(nu, next error) .59; best correction ridge(10) on nu_t, nu_t-5 -40% / -15%, post(0.5) -30% / -8%, in-loop dist(1.0) diverges. PushT cos .30, best -7% / -2%. Cube cos .34, best -11% / -2%. Innovation feedback is only material on Reacher; it is also prior art (Feedback World Model 2605.15705, a latent observer).
- 56116/56128 final, min-start-10 sampling, 3 seeds x 50 (h1 / h3 / h3+prefill): Reacher 74.0 / 84.7 / 93.3; PushT 82.7 / 84.0 / 88.0 (h3p-h1 +5.3 [0,+10.7]); Cube 68.0 / 68.0 / 69.3 (n.s.); TwoRoom no effect. Released sampling h3 vs h1: Reacher +9.3, PushT -0.7, Cube 0.0 (identical episodes), TwoRoom -0.7.
- 56139 released sampling + partial prefill (history 3): **Reacher 96.7 (90/100/100) vs released h1 76.7: +20.0 [+13.3,+26.7], 33 vs 3**; vs h3 +10.7. PushT 90.7 vs 88.7 (+2.0, n.s.).
- 56138 Reacher closed loop, min-start 16, 3 seeds x 50: h3 86.7, h3+prefill 96.7 (+10.0 [+4.7,+16.0]), h4+prefill 97.3 (control, = h3p within noise), +innovation post 0.5 98.0 (+0.7 [-2.7,+4.0]), dist 0.25 95.3 (-2.0), dist 0.5 94.7 (-2.7). Innovation feedback adds nothing once the start state is complete; Reacher is at ceiling.

### Phase 1 verdict (2026-09-30)
The only large lever is giving the first plan the executed history (Reacher released 76.7 -> 96.7; PushT +2 to +5; Cube, TwoRoom 0). Estimators on top (MHE, innovation observer) add nothing in closed loop, and the observer / test-time adaptation / recurrent-state mechanisms are prior art (2605.15705, 2606.32026, 2609.21740, 2609.22816). Not a standalone method; h3+prefill is the corrected LeWM baseline. No jobs running.
