# CTA benchmark decision and LIBERO continuation protocol (2026-09-27)

## Benchmark decision (supersedes all earlier benchmark plans)

CTA's claim that needs proof: a code S of the whole chunk, conditioned on context C, ranks policy proposals better
than endpoint-vs-goal scoring **when task success depends on what happens inside the chunk**. The test task must
make that dependence part of its own success criterion; no query is invented by us.

| Task | Role | Label |
|---|---|---|
| **LIBERO-Goal + SmolVLA** (existing, exact clone, P0 ≈ 71–75 %) | Main task for the trajectory claim | Episode success when the frozen policy continues from the chunk end |
| PushT + Diffusion Policy (Round 4 chain 55149 → 55170) | Development task, second table | Coverage/geometry; continuation labels later if useful |

Dropped, do not resume: RECON / NWM / CompACT navigation (its standard task scores only the final image against the
goal, so it cannot test the trajectory claim), LeWM suite (arXiv preprint, not an anchor), CALVIN/DynaGuide
(assets unreachable from the cluster). Downloaded RECON/CompACT files in `/mnt/data/nhatnc129/jepa/nav` are unused.

## Why LIBERO, and what L3 did and did not show

L3 (54670): P0 .75 vs ORACLE8 .74, where ORACLE8 picks the chunk with the highest privileged goal progress at the
chunk end. That is the endpoint-vs-goal criterion; it gives no closed-loop gain. It does not show that the choice of
chunk has no effect on success: grasp, lift, drawer opening and detours can leave goal distance unchanged while
deciding the episode. L1 found 50/100 roots both succeed and fail across policy seeds.

## L4: continuation-value collection (`scripts/libero/l4_continuation.py`)

- Roots: tasks 0–9 × init 0–1 (20 roots, development inits; inits 10–49 stay untouched for test).
- Live env follows P0. Anchors at pre-registered env times t = 20, 60, 120 (first decision at or after); an anchor
  the episode never reaches is recorded as missing, never replaced.
- At each anchor: K = 8 seeded SmolVLA chunks (candidate 0 = P0) are branched from the exact clone. Each branch runs
  its chunk, then P0 continues to the episode end under 4 noise streams; stream s uses the same seeds for every
  sibling (common random numbers).
- Saved per candidate: chunk frames (both cameras, every step), qpos trace, scene qpos and eef at chunk end, goal
  progress at chunk end, 4 continuation outcomes. These are also CTA training data if the scale-up goes ahead.

Analysis (aggregate mode, bootstrap over roots):
- held-out gain over candidate 0 of (a) selection by continuation value on streams 0–1 and (b) selection by
  chunk-end goal progress, both evaluated on streams 2–3;
- fraction of anchors whose candidates differ in continuation value;
- sibling pairs with chunk-end progress within 0.02 whose continuation values differ by ≥ 0.5: the direct count of
  cases where endpoint-vs-goal cannot rank but the task outcome differs.

Next step fixed in advance: if (a) is above zero with a CI that excludes 0, scale the same collector to more anchors
and train CTA vs endpoint-latent vs DIRECT readers on the continuation label, then evaluate closed-loop on held-out
inits. If (a) is not above zero, the K = 8 SmolVLA bank has no reachable continuation headroom at these anchors;
report that and change the proposal bank (larger K or noise scale), not the label.

Limitation stated up front: in simulation the full end state determines the future given the policy. The trajectory
argument is about observations: frames at the chunk end do not show grasp stability, velocity or occluded contact,
while the chunk's frames do. Privileged-state analysis cannot settle that; only learned readers on images can.

Jobs: 55231 (collect, main, array 0–19%10, 4 CPU, 24 GB, 12 h each), 55232 (aggregate, afterany).

## Interim reading (13/20 roots, 2026-09-27)

16 of 30 present anchors are saturated (all candidates succeed or all fail). On the 14 others, a two-way split of the
8 × 4 outcome table gives candidate 16 %, stream 36 %, residual 48 % of the variance; with pure noise the
candidate share would be about 23 %. The present configuration therefore shows no reliable ranking signal. With 4
continuations per candidate this does not show that chunks have no effect, nor why; it is a resolution limit of
this label configuration. Collection is not expanded; the aggregate (55232) is read with those limits. Main
end-to-end test moves to the DINO-WM planner (docs/CTA_DINOWM_PLANNING_PROTOCOL.md).
