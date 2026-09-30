# CTA in a DINO-WM planner: PushT end-to-end comparison (2026-09-27)

Question: inside the released DINO-WM planning protocol (Zhou et al., ICML 2025), does a planner that scores each
candidate action sequence with CTA (one pass C, A → Ŝ, then reader D(C, Ŝ, goal)) reach the same success as
DINO-WM's frame-by-frame latent rollout with a terminal L2 cost, at lower planning cost? "Same quality, lower
compute" is a hypothesis, not a claim.

## Fixed across arms
Official PushT data (`pusht_noise`), action interface (frameskip 5, 10-D macro actions), horizon H = 5, goal source
= dataset state goal_H = 5 macro steps ahead, MPC-CEM budget (300 samples, top 30, 30 iterations, 5 actions taken),
50 evals per seed, same seeds. Success = the environment's own criterion (position < 20 px, angle < π/9).

## Arms
| Arm | Scores a candidate by |
|---|---|
| DINO-WM (official checkpoint) | autoregressive per-frame rollout, L2 of final latent to goal latent |
| ENDPOINT | one-pass predictor of the final latent only, then the same reader as CTA (only the content of the code differs) |
| DIRECT | network (C, A, goal) → score, trained on the same labels as the reader |
| CTA | one-pass Ŝ + reader D(C, Ŝ, goal) |
Metrics: success rate (bootstrap over evals and seeds), planning wall-clock per MPC step and per episode, total
pipeline time. ENDPOINT at equal parameter/compute budget is the control for any claim that intermediate
information helps; without it, a CTA win only shows that a compact one-pass representation plans well.

## Steps
1. Reproduce DINO-WM with the official checkpoint and config (job 55275; earlier attempts 55259 (no /usr/bin/time), 55273 (gym missing), 55274 (Hydra launcher override; single runs use no launcher) failed before planning; `scripts/dinowm/slurm_dinowm.sh repro`),
   with timing. Compare to the paper's PushT number. The only change is a stub for the PointMaze import (d4rl);
   `max_iter` is capped at 5 because the released config loops until every eval succeeds.
2. Build CTA / ENDPOINT / DIRECT on DINO-WM's frozen DINOv2 features and the same training trajectories; train
   labels come from the dataset (future latents / states), not from the planner's own samples, in round 1.
3. Plug each scorer into the same CEM loop; run matched seeds; report success and time.
Risk known from earlier programs in this repo: CEM exploits scorer error on off-distribution action sequences
(reward hacking). Report the scorer's error on CEM elites against the simulator, as a diagnostic, next to success.
GPU cost is set from the timing in step 1, not assumed.
