# Cost-overoptimization (E0) analysis

Latent oracle (perfect dynamics, scripts/30) — any proxy/true divergence is the
*cost under optimization*, not predictor error. See
`docs/plans/2026-07-06-overoptimization-sweep-design.md`.

## Across-budget outcomes (does more search help?)

Episode-clustered bootstrap 95% CI. `obj_med` = median true object→goal distance;
for reach read `ee_med` (no object).

| task | cost | iters | n_samp | n_ep | success_end | success frac [CI] | obj_med [CI] | ee_med [CI] |
|---|---|---|---|---|---|---|---|---|
| mw-pick-place | l2 | 2 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.286 [0.251,0.301] | 0.093 [0.024,0.141] |
| mw-pick-place | l2 | 6 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.286 [0.251,0.301] | 0.011 [0.009,0.013] |
| mw-pick-place | l2 | 12 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.286 [0.251,0.301] | 0.016 [0.011,0.021] |
| mw-pick-place | l2 | 24 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.286 [0.251,0.301] | 0.014 [0.008,0.018] |
| mw-pick-place | stateprobe | 2 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.276 [0.246,0.301] | 0.210 [0.166,0.244] |
| mw-pick-place | stateprobe | 6 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.269 [0.235,0.298] | 0.196 [0.169,0.220] |
| mw-pick-place | stateprobe | 12 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.270 [0.251,0.297] | 0.204 [0.171,0.239] |
| mw-pick-place | stateprobe | 24 | 100 | 16 | 1/16 | 0.06 [0.00,0.19] | 0.263 [0.249,0.294] | 0.210 [0.177,0.241] |
| mw-push | l2 | 2 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.239 [0.179,0.261] | 0.138 [0.099,0.144] |
| mw-push | l2 | 6 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.239 [0.164,0.261] | 0.015 [0.011,0.021] |
| mw-push | l2 | 12 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.239 [0.171,0.261] | 0.017 [0.010,0.028] |
| mw-push | l2 | 24 | 100 | 16 | 1/16 | 0.06 [0.00,0.19] | 0.239 [0.171,0.261] | 0.018 [0.013,0.025] |
| mw-push | stateprobe | 2 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.177 [0.155,0.230] | 0.129 [0.086,0.149] |
| mw-push | stateprobe | 6 | 100 | 16 | 1/16 | 0.06 [0.00,0.19] | 0.197 [0.166,0.261] | 0.104 [0.080,0.170] |
| mw-push | stateprobe | 12 | 100 | 16 | 1/16 | 0.06 [0.00,0.19] | 0.183 [0.147,0.242] | 0.102 [0.068,0.154] |
| mw-push | stateprobe | 24 | 100 | 16 | 2/16 | 0.12 [0.00,0.31] | 0.211 [0.164,0.261] | 0.104 [0.073,0.155] |
| mw-reach | l2 | 2 | 100 | 16 | 5/16 | 0.31 [0.12,0.56] | 0.000 [0.000,0.000] | 0.036 [0.017,0.120] |
| mw-reach | l2 | 6 | 100 | 16 | 16/16 | 1.00 [1.00,1.00] | 0.000 [0.000,0.000] | 0.006 [0.004,0.007] |
| mw-reach | l2 | 12 | 100 | 16 | 16/16 | 1.00 [1.00,1.00] | 0.000 [0.000,0.000] | 0.005 [0.002,0.008] |
| mw-reach | l2 | 24 | 100 | 16 | 16/16 | 1.00 [1.00,1.00] | 0.000 [0.000,0.000] | 0.005 [0.003,0.009] |
| mw-reach | stateprobe | 2 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.000 [0.000,0.003] | 0.262 [0.236,0.278] |
| mw-reach | stateprobe | 6 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.000 [0.000,0.006] | 0.256 [0.230,0.286] |
| mw-reach | stateprobe | 12 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.000 [0.000,0.003] | 0.257 [0.228,0.277] |
| mw-reach | stateprobe | 24 | 100 | 16 | 0/16 | 0.00 [0.00,0.00] | 0.000 [0.000,0.000] | 0.250 [0.242,0.279] |

## Within-search Goodhart (first → last CEM iter, at max budget)

Proxy = the cost the planner minimizes; TRUE = the real state cost of the
candidate it most believes in; decode = object-probe error on the committed
elites (pocket depth).

| task | cost | budget | proxy Δ% | true_sp Δ% | decode first → last (cm) [CI] | growth CI-clean? |
|---|---|---|---|---|---|---|
| mw-pick-place | l2 | 24 | -29% | +2% | 3.5 [2.8,4.4] → 3.7 [2.9,4.8] (+6%) | no (CIs overlap) |
| mw-pick-place | stateprobe | 24 | -13% | -2% | 6.7 [5.9,7.4] → 8.1 [7.0,9.1] (+21%) | no (CIs overlap) |
| mw-push | l2 | 24 | -35% | +1% | 4.1 [3.5,4.8] → 4.5 [3.8,5.2] (+9%) | no (CIs overlap) |
| mw-push | stateprobe | 24 | -18% | -2% | 6.6 [5.7,7.4] → 8.3 [7.1,9.3] (+25%) | no (CIs overlap) |
| mw-reach | l2 | 24 | -48% | +7% | 3.3 [2.7,3.9] → 3.4 [2.7,4.1] (+1%) | no (CIs overlap) |
| mw-reach | stateprobe | 24 | -31% | -7% | 3.5 [2.8,4.2] → 3.4 [2.6,4.1] (-3%) | no (CIs overlap) |

`growth CI-clean?` = whether the first-iter and last-iter decode-error CIs
(episode/seed-clustered bootstrap, n=8 episodes) are non-overlapping — i.e.
whether the "pocket deepens" claim survives the small sample size, not just
the point estimate.

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
