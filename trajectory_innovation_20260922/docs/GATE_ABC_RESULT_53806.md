# PushT gates A–C: result (jobs 53803 → 53806 → 53811)

2026-09-22. Pre-registered protocol: `GATE_ABC_PROTOCOL.md` (plus Amendment 1). The run used
100 qualification roots (seeds 1000–1099), with all intervals paired by root.
Machine-readable result: `/mnt/data/nhatnc129/jepa/trajectory_innovation/gate_agg_53806/summary.json`.

**Verdict:**
- A PASS
- B PASS
- **C FAIL**

The frozen policy's K=8 bank carries real selection headroom. The pinned DINOv2
goal-image score cannot find that headroom. Per the protocol, the control contract stops here for this
scorer. No world model is trained on it, and the kernel is not re-tuned post hoc.

## Success by arm (closed loop, same 100 roots)

| Arm | Success | Mean max coverage |
|---|---:|---:|
| OFFICIAL (unmodified `select_action`) | 65% | .917 |
| P0 (candidate 0, our sampler + clones) | 64% | .924 |
| PHYS8 (oracle: best 8-step coverage) | **79%** | .927 |
| VIS8 (DINOv2 goal-map score on ACTUAL future) | 67% | .912 |
| MEDOID8 (no future information) | 68% | .919 |

## Gates

**A: runtime reproduction (PASS).**
- OFFICIAL succeeded on 65/100, Wilson CI [55.3, 73.6]%. This deviates −0.4 pp from the card's 65.4%.
- On the same 100 seeds, the published run got 70/100. Our difference from it is −5 pp [−15, +5], McNemar p = .44.
- P0 − OFFICIAL is −1 pp [−11, +9], so there is no runtime blocker. Our sampler plus branch cloning reproduces the policy.

**B: headroom (PASS).**
- PHYS8 − P0 is **+15 pp, CI [+5, +25]**. There are 21 roots that only PHYS8 solves and 6 that only P0 solves, exact McNemar p = .006.
- The gain comes from repeated myopic selection. The average per-decision coverage gain is small, +.0074 [.0066, .0083] at K=8, rising to .0105 at K=32. A strictly better candidate exists at 54% of decisions.

**C: visual-score alignment (FAIL).**
- VIS8 − P0 is +3 pp [−8, +14].
- Retention is 0.20, with a bootstrap CI of [−1.0, 0.85]; 17 resamples had a nonpositive denominator. This is below the 0.80 bar.
- The offline secondary tells the same story. The retained per-decision coverage gap is 0.33 [0.26, 0.39]. The within-bank Spearman(visual, physical) averages 0.06 over 1,742 decisions: close to no ranking signal.

## Diagnostics

**H_rep (continuation oracle, split seeds).**
- The split estimate is +2 pp [−3, +7.5]; default continuation succeeds at .59.
- The naive max-over-seeds estimate is +22.75 pp [+17.5, +28.3]. That is almost entirely winner's curse, as the protocol anticipated.
- So a single swapped chunk rarely changes the final outcome. PushT's headroom comes from compounding many small, myopically better choices, not from one decisive chunk.

**MEDOID8 − P0** is +4 pp [−8, +17]: not distinguishable from the default.

## What this does and does not show

It shows three things:
- The proposal side has headroom under a fair closed-loop oracle.
- The selector needs a scorer sensitive to small task progress.
- A global DINOv2 patch-feature distance to a mean goal map is not such a scorer here.

This is the repository's main lesson again: the binding constraint is the cost, not proposals or prediction, and here it holds on policy support, not only under CEM.

It does not show:
- that every visual or self-supervised query fails;
- that a progress-type query (e.g. learned temporal distance) fails;
- anything about the trajectory-abstraction world model, which was not trained.

Why the kernel fails is a hypothesis, not a measurement. Candidates in a bank mostly differ in the agent
disk's position, which dominates whole-image patch differences. Coverage changes of less than 1% are small, local block motions.

Next step (not run): any new scorer or query family needs a new pre-registered protocol and
must be re-qualified with the same PHYS8 ceiling and the same seeds rule. Qualification roots
1000–1099 are now used and must not serve as confirmation roots.
