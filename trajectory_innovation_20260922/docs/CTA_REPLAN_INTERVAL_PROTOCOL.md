# Replan interval: does trajectory-level selection matter more when the planner replans less often?

Pinned 2026-10-01, before any data for it. User decision: wrap up CTA for CVPR with this test.

## Why

On PushT with the native 8 executed actions per decision, the closed loop absorbs selection errors
(55877–55879 on 55666): success-critical "crossing" decisions are 1% of decisions, and about 70% of roots
with a missed crossing still succeed a few decisions later. So the offline advantage of CTAV2 (retained gap
.68 vs DIRV2 .48, DINO-WM .33, CI-clean) yields only +2 to +5 closed-loop successes over them.

Replanning less often removes part of that recovery. It is also a real deployment constraint: one decision
costs about 770 ms of policy sampling at K = 8 (55763), so fewer decisions per episode is the compute lever.
Hypothesis: as the executed chunk grows, P0 loses success, the oracle's headroom grows, and the success gap
between CTAV2 and the direct and per-step scorers widens.

## Design

- Same frozen `lerobot/diffusion_pusht`, K = 8 seeded candidates (`candidate_seed`), same clone and native
  termination. Executed actions per decision L ∈ {8, 15}. L = 15 is every future action of one sample: the
  policy predicts 16 actions and index 0 is the step of the older observation (`executed_slice`).
- L = 8 is the existing v2 closed loop 55666 (roots 2200–2399). No new L = 8 run.
- L = 15 networks are retrained with the v2 recipe on L = 15 data only; nothing is reused from 8-step models.
  - Collection `cta_collect_plus.py --n-exec 15` (Round-4 design: standard + perturbed bank; even roots follow P0,
    odd roots execute the geometry-best standard candidate on a seeded half of decisions). Train roots
    34000–35199 (1,200 episodes, unused before). Selection roots 2000–2099 (dev offline range).
  - Intermediate frames at the quarter points (4, 8, 11) instead of (2, 4, 6); end frame at step 15.
  - Encode with the original PCA (`cta_feat_54490`). Train `cta_train_v2.py --only-r4 --with-perturbed
    --select-r4 <dev features>`, seed 0, default steps.
  - Disclosed difference: the L = 8 v2 models also used Round-0 and on-policy banks (more data). Comparisons
    between methods are made within each L, where data and selection are identical.
- Closed loop at L = 15 on roots 2200–2399: arms P0, GEOM8 (geometry oracle over the 8 candidates; privileged),
  CTAV2, ENDV2, DIRV2, DINOWM (official checkpoint, 3 macro steps = 15 actions); FULLV2 and CODEV2 logged.
  Learned arms never read simulated futures.

## Pre-registered read-out (fixed before any L = 15 result)

1. P0 and GEOM8 success at L = 15 against L = 8 (122 and 160 of 200): the headroom.
2. Crossing share, capture and recovery after a missed crossing at L = 15, with the definitions of 55877–55879.
3. Primary: CTAV2 − DIRV2 and CTAV2 − DINOWM success at L = 15, root-paired bootstrap and exact McNemar.
4. Difference in differences across L (same roots in both): (CTAV2 − X at L = 15) − (CTAV2 − X at L = 8) for
   X ∈ {DIRV2, DINOWM, ENDV2}, root bootstrap.
5. Planning compute per episode (decisions × ms per decision) for every arm at both L.

Reading, not a kill rule: if the gaps widen, the paper's claim is that trajectory-level selection pays off when
replanning is sparse; if they do not, that is reported as measured. Development roots only; a sealed
confirmation (roots 3000–3399, unopened) with more training seeds is decided with the user after this read-out.

## Compute

Estimates from Round 4 (collection 2.9 GPU-h for 1,200 episodes, encode 0.6 h, v2 training 2.3 h) and 55666
(3.7 GPU-h for 7 arms × 200 roots at L = 8): about 10 GPU-h in total at L = 15. October's top-5 cap is small at the
start of the month, so submissions are staged and each one is checked with `sreport` first.
