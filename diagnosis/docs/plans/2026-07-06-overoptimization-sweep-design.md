# E0 — Cost-overoptimization sweep (Goodhart curves for latent planning)

**Goal (paper repositioning, see `2026-07-05-novel-methods-survey.md` §3):** turn the Phase-A
reward-hacking observation into the paper's headline *quantitative* result — the world-model
analogue of Gao et al.'s "Scaling Laws for Reward Model Overoptimization". Claim under test:
**test-time search is an adversary of any learned cost** — as CEM optimization pressure grows, the
*proxy* cost keeps improving while the *true* task outcome stalls or degrades, and the planner
converges onto the cost's residual-error pockets.

## Why the latent oracle is the right harness

`scripts/30_latent_oracle.py` gives **perfect dynamics** (sim-step → render → encode) with only the
cost swapped. Any proxy/true divergence is therefore attributable to the *cost under optimization*
— not to predictor rollout error (the classic MBRL "model exploitation" confound). This is the
isolation no other paper has.

`cem_plan_latent`'s `on_elites` hook already exposes, per CEM iteration, the elite population's
proxy costs **and their true sim states** (`raw_elite`, 39-dim MetaWorld state). So the
within-search Goodhart curve is directly measurable.

## Measurements (script `41_overoptimization_sweep.py`)

Sweep grid: CEM `iterations ∈ {2, 6, 12, 24}` × `samples ∈ {100}` (upstream ladder used 6×100),
costs `{stateprobe, l2}`, tasks `{mw-push, mw-reach}`, 8 paired episodes (`seed0=10000`,
strict success, upstream-parity protocol as scripts/29/30).

Per (cost, task, budget, episode) → `results/overopt_episodes.csv`:
`success`, `success_end`, `obj_goal_dist`, `ee_dist`, `steps`, `minutes` — the **across-budget**
closed-loop panel ("does more search help?").

Per CEM iteration (every replan) → `results/overopt_curves.csv`:
- `proxy_min`, `proxy_med` — elite cost values (what the planner believes);
- `true_obj_min_believed` — true object→goal distance of the **argmin-proxy** candidate (the plan
  the planner most trusts) + `true_obj_med` over elites;
- `true_sp_min_believed` / `true_sp_med` — the true-state analogue of the stateprobe cost
  (‖obj−goal‖ + w_hand·‖hand−obj‖) so proxy and true are unit-comparable for that arm;
- `decode_err_med_cm` — object-probe decode error on the elites (pocket-depth mechanism: does the
  error of the readout *on the frames CEM trusts* grow with optimization pressure? extends
  Phase-A's 91.5%→24% single-point result into a curve).

## Predictions (falsifiable)

1. **push × stateprobe**: `proxy_min` ↓ monotonically with iteration; `true_sp_min_believed` stops
   tracking it (flat/↑) after few iterations; `decode_err_med_cm` ↑ with iteration. Across budgets,
   `obj_goal_dist` does **not** improve (possibly worsens) from 2→24 iterations.
2. **reach × (l2, stateprobe)**: proxy and true track each other; more budget helps or saturates —
   the honest-cost control that makes panel 1 interpretable.
3. **push × l2**: flat true outcome at all budgets (no minimum at task success — geometry, not
   search). Distinguishes "cost has no signal" (l2) from "cost has signal but is exploitable"
   (stateprobe) — two different failure modes, one figure.

If instead push×stateprobe *improves* with budget, the reward-hacking reading of Phase 3b/A is
wrong and the repositioning collapses back to the representation story — also worth knowing now.

## Runtime budget

Latent oracle measured ≈5 min/episode at 6×100 (job 22489), linear in iterations. Push job
(grid {2,6,12,24} × 2 costs × 8 eps) ≈ 10 h; reach job ({2,6,24} × 2 × 8) ≈ 7 h. Two SLURM jobs
(`slurm_overopt_sweep.sh`, parameterized) so they run in parallel. Resume-safe: partial
(task, seed) pairs are dropped and redone whole (same convention as scripts/18).

## Checkpoints used (existing, no training)

- `checkpoints/spatial_object_probe_dino_wm_metaworld_offpolicy.pt` (Phase-3 3b off-policy-robust)
- `checkpoints/ee_probe_dino_wm_metaworld_offpolicy.pt`

## Follow-ups queued behind this result

- E1 amortized GC-IDM control (removes the search adversary; `inverse_proposal` ckpt exists).
- Samples-axis sweep (`samples ∈ {50,300}` at 6 iters) if the iterations axis shows the effect.
- WM-closed-loop (scripts/18) budget sweep — deployed-stack panel, dynamics error included.
