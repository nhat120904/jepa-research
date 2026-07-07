# Cost-overoptimization (E0) analysis

Latent oracle (perfect dynamics, scripts/30) — any proxy/true divergence is the
*cost under optimization*, not predictor error. See
`docs/plans/2026-07-06-overoptimization-sweep-design.md`.

## Across-budget outcomes (does more search help?)

Episode-clustered bootstrap 95% CI. `obj_med` = median true object→goal distance;
for reach read `ee_med` (no object).

| task | cost | iters | n_samp | n_ep | success_end | success frac [CI] | obj_med [CI] | ee_med |
|---|---|---|---|---|---|---|---|---|
| mw-push | l2 | 12 | 50 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.252 [0.226,0.290] | 0.018 |
| mw-push | l2 | 2 | 100 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.252 [0.226,0.290] | 0.142 |
| mw-push | l2 | 6 | 100 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.252 [0.226,0.290] | 0.017 |
| mw-push | l2 | 12 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.252 [0.240,0.278] | 0.019 |
| mw-push | l2 | 24 | 100 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.252 [0.226,0.290] | 0.021 |
| mw-push | l2 | 12 | 300 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.252 [0.202,0.290] | 0.023 |
| mw-push | stateprobe | 12 | 50 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.233 [0.176,0.290] | 0.140 |
| mw-push | stateprobe | 2 | 100 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.236 [0.170,0.290] | 0.149 |
| mw-push | stateprobe | 6 | 100 | 8 | 1/8 | 0.12 [0.00,0.38] | 0.252 [0.135,0.290] | 0.149 |
| mw-push | stateprobe | 12 | 100 | 16 | 2/16 | 0.12 [0.00,0.31] | 0.207 [0.158,0.244] | 0.133 |
| mw-push | stateprobe | 24 | 100 | 8 | 2/8 | 0.25 [0.00,0.62] | 0.252 [0.041,0.290] | 0.142 |
| mw-push | stateprobe | 12 | 300 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.252 [0.202,0.290] | 0.144 |
| mw-reach | l2 | 2 | 100 | 8 | 2/8 | 0.25 [0.00,0.50] | 0.000 [0.000,0.000] | 0.093 |
| mw-reach | l2 | 6 | 100 | 8 | 8/8 | 1.00 [1.00,1.00] | 0.000 [0.000,0.000] | 0.006 |
| mw-reach | l2 | 24 | 100 | 8 | 8/8 | 1.00 [1.00,1.00] | 0.000 [0.000,0.000] | 0.006 |
| mw-reach | stateprobe | 2 | 100 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.000 [0.000,0.000] | 0.261 |
| mw-reach | stateprobe | 6 | 100 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.000 [0.000,0.011] | 0.264 |
| mw-reach | stateprobe | 24 | 100 | 8 | 0/8 | 0.00 [0.00,0.00] | 0.000 [0.000,0.000] | 0.253 |

## Within-search Goodhart (first → last CEM iter, at max budget)

Proxy = the cost the planner minimizes; TRUE = the real state cost of the
candidate it most believes in; decode = object-probe error on the committed
elites (pocket depth).

| task | cost | budget | proxy Δ% | true_sp Δ% | decode first→last (cm) |
|---|---|---|---|---|---|
| mw-push | l2 | 24 | -32% | +2% | 4.1→4.5 (+10%) |
| mw-push | stateprobe | 24 | -17% | -2% | 6.6→8.2 (+24%) |
| mw-reach | l2 | 24 | -45% | +7% | 3.4→3.6 (+5%) |
| mw-reach | stateprobe | 24 | -30% | -8% | 3.2→3.1 (-4%) |

**Reading:** an *exploitable* cost shows proxy Δ ≪ 0 (planner "improves"),
true_sp Δ ≈ 0 (reality stalls), decode ↑ (search converges onto readout-error
pockets). An *honest* cost (reach×l2) shows proxy ↓ AND the across-budget
success climbing with iterations. A *no-signal* cost (push×l2) shows proxy ↓
but true perfectly flat at every budget.

## E1 — amortized control + search-dose response (mw-push)

Removing the search adversary (`gcidm`, pure amortized) and re-adding it in
doses. Episode-clustered bootstrap 95% CI.

| arm (dose) | n_ep | success_end | obj_med [CI] | ee_med |
|---|---|---|---|---|
| gcidm | 8 | 0/8 | 0.252 [0.226,0.290] | 0.241 |
| cemseed_it0 | 8 | 0/8 | 0.252 [0.226,0.290] | 0.342 |
| cemseed_it2 | 8 | 0/8 | 0.252 [0.226,0.290] | 0.081 |
| cemseed_it6 | 8 | 0/8 | 0.252 [0.134,0.290] | 0.138 |
| cemseed_it12 | 8 | 0/8 | 0.252 [0.126,0.290] | 0.134 |
| cemseed_it24 | 8 | 0/8 | 0.252 [0.162,0.290] | 0.102 |

**Seed-vs-chosen** (per replan): does search corrupt the amortized seed?
`corruption` = search lowered proxy AND raised true cost; `Δtrue_mean` < 0
means search helped the true state on average.

| arm (dose) | n_replan | beats_seed(proxy) | corruption [CI] | Δtrue_mean |
|---|---|---|---|---|
| cemseed_it0 | 56 | 0.00 | 0.00 [0.00,0.00] | +0.0000 |
| cemseed_it2 | 56 | 0.98 | 0.00 [0.00,0.00] | -0.0413 |
| cemseed_it6 | 56 | 1.00 | 0.04 [0.00,0.11] | -0.0497 |
| cemseed_it12 | 56 | 1.00 | 0.04 [0.00,0.11] | -0.0460 |
| cemseed_it24 | 56 | 1.00 | 0.07 [0.00,0.18] | -0.0430 |

**Reading:** all arms end 0/8 with obj_med ≈ 0.25 m — neither amortizing away
the search NOR any search dose crosses the contact wall; the object never moves.
Corruption is real and rises monotonically with dose (0→7%), confirming the
adversary at the plan level, but is too small to dominate — search still helps
the *hand* (Δtrue < 0) while neither seed nor search moves the *object*. E1 is a
null for "remove the adversary fixes contact": the frozen representation, not
the optimizer alone, is the residual wall. Grounding is necessary, not sufficient.
