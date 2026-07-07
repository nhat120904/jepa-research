# E1 — Amortized control (remove the adversary) + search-dose response

**Follows E0** (`2026-07-06-overoptimization-sweep-design.md`, results in
`results/overopt_analysis.md`): E0 established the three-mode picture — honest cost → search
helps (reach×l2 2/8→8/8); exploitable cost → proxy↓ −17%, true flat, elite decode error ↑24%
(push×stateprobe); no-signal cost → search inert (push×l2 flat at 0.252 across all budgets).

E1 tests the framing's constructive corollary: **if test-time search is the adversary, removing
it should not hurt — and re-adding it in increasing doses should corrupt a good plan.**

## Controller

`GC-IDM` from existing pieces (no training): `h_inv(z_t, Δobj) → 20-dim raw action chunk`
(`checkpoints/inverse_proposal_dino_wm_metaworld.pt`, scripts/28; val action-MSE 0.419 vs 0.571
baseline, obj_scale 0.028 m) + the off-policy spatial probe for `Δobj = g(z_goal) − g(z_t)`.
Same frozen representation, same probe readout the hacked stateprobe cost uses — so any
gcidm-vs-CEM difference is attributable to the *optimizer*, not to information content.

## Arms (script `43_e1_amortized_control.py`, latent-oracle env, mw-push, paired seeds 10000+)

1. **`gcidm`** — pure amortized: every model step (5 raw steps), encode → probe → request one
   typical-scale object step toward goal (`Δ` clipped to `step_scale`, default obj_scale) →
   head → execute. No cost, no search, replans 3× more often than CEM (that per-decision
   cheapness is the point — cite "Latent Geometry Beyond Search").
2. **`cemseed_it{0,2,6,12,24}`** — the dose-response axis: CEM on the latent oracle
   (cost=stateprobe), horizon 6, `init_mean` = the gcidm proposal tiled over the horizon
   (Δ = (g_goal−g_t)/plan_h per model step, the scripts/18 `inv`-seed convention), CEM cadence
   (commit 3 model steps per replan). `it=0` executes the seed with **zero** search — the
   same-cadence no-search control (separates "no search" from gcidm's faster replanning).
   Mean-inclusion is on for seeded runs (scripts/30 `init_mean`), so elites can *retain* the
   seed — search only replaces it when the cost approves the swap.

## The E1 crown measurement: seed-vs-chosen, per replan (`results/e1_seed_vs_chosen.csv`)

At every replan of every seeded arm, roll BOTH the seed plan and the CEM-chosen plan on the
sim (perfect dynamics), score both with the proxy AND with true state:
`(seed_proxy, seed_true) vs (chosen_proxy, chosen_true)`. The hacking signature, quantified:

- **corruption event** = chosen_proxy < seed_proxy (search "improved" the plan by its own
  cost) AND chosen_true > seed_true (the plan got truly worse). Report the corruption rate
  and its dose-response in iterations.
- Honest-cost prediction: corruption ≈ 0, chosen_true ≤ seed_true.
- Adversary prediction: corruption rate grows with iterations; across-arm episode outcome
  (`obj_goal_dist`) degrades (or stays flat) from it=0 → it=24 while proxy improves.

## Interpretation matrix

| gcidm / it0 outcome | dose-response | reading |
|---|---|---|
| crosses push (≥ frozen 0–2/16 baseline clearly) | true degrades with iters | **headline positive**: grounding + no search beats grounding + search; adversary confirmed |
| fails (~baseline) | true degrades with iters | adversary confirmed on the *plan level* even though the proposal is too weak to cross alone — E0+E1 still carry the claim |
| fails | true improves with iters | search genuinely repairs weak proposals → adversary framing WRONG at plan level; report honestly, fall back to E0 + representation story |

Note: mw-reach is excluded — h_inv is object-conditioned (Δobj ≡ 0 on reach → degenerate input).
The honest-search control for reach is already in E0 (reach×l2).

## Cost

gcidm ≈ 1 min/ep; it0 ≈ 1 min; seeded CEM arms ≈ 0.83 min/iter/ep → per 8-episode paired sweep
≈ 5.5 h. One SLURM job (`slurm_e1_amortized.sh`), resume-safe like scripts/41.
