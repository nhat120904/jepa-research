# HyS-JEPA pre-gate — latent curvature by contact regime (MetaWorld)

Date: 2026-08-21. 720 trajectories, 12 MetaWorld tasks, both frozen checkpoints.
n = 69,840 curvature samples per model. Trajectory-clustered bootstrap CIs.
Curvature `c_t = 1 - cos(z_{t+1}-z_t, z_{t+2}-z_{t+1})`, the ICML-2026 straightening form.

## Headline

**The mode-gating premise is refuted. The straightening opportunity is confirmed and large.**

| quantity | dino_wm | jepa_wm |
|---|---|---|
| chance (cross-trajectory displacements) | 0.9998 | 0.9993 |
| latent, all transitions | **1.0955** [1.0930, 1.0979] | **1.1062** [1.1037, 1.1087] |
| latent, free_space | 1.0815 | 1.0937 |
| latent, contact_manipulation | 1.0926 | 1.1003 |
| latent, pre_grasp | 1.1086 | 1.1201 |
| physical end-effector, within-mode | 0.0863 | 0.0863 |
| physical end-effector, within-mode, large moves | **0.0314** | 0.0317 |
| physical object, within-mode | 0.3204 | 0.3204 |

## 1. The frozen latent is more curved than chance

Chance curvature is 1.0 (random directions in ~1e5 dims are orthogonal). Every latent group
sits **above** it. Consecutive latent displacements are systematically *anti-correlated*: the
latent trajectory zig-zags. Meanwhile the physical trajectory it encodes is genuinely straight
(end-effector 0.03-0.09, object 0.32).

The encoder destroys temporal straightness that is present in the physics. The gap the
straightening objective would have to close is roughly **1.10 -> 0.03-0.32**, and it is
consistent across both checkpoints and all 12 tasks.

## 2. Physics does NOT kink at contact-mode switches -- the latent does

| switch minus within | dino_wm | jepa_wm |
|---|---|---|
| latent | **+0.0287** [+0.0249, +0.0327] CI-clean | **+0.0237** [+0.0196, +0.0277] CI-clean |
| physical object | **-0.1370** [-0.1618, -0.1137] CI-clean | **-0.1370** [-0.1618, -0.1137] CI-clean |
| physical end-effector | -0.0138 [-0.0193, -0.0083] CI-clean | -0.0138 [-0.0193, -0.0083] CI-clean |

The signs are opposite. At transitions where the contact mode changes, the **object's real
trajectory is smoother than average**, while the **latent is kinkier than average**.

This directly contradicts the motivating assumption behind `(1 - s_t) L_curve`. Gating the
curvature loss off at mode switches would exempt from correction exactly the transitions where
the latent is most wrong and the underlying physics is smoothest. The gate is backwards.

## 3. The excess curvature is structural, not encoder jitter

Curvature vs `||dz||` decile (dino_wm; jepa_wm identical in shape):

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| curvature | 0.984 | 1.110 | 1.145 | 1.141 | 1.131 | 1.119 | 1.106 | 1.088 | 1.071 | 1.060 |

Additive frame-to-frame encoder noise would make curvature **highest** at the smallest
displacements and fall monotonically. The observed pattern is the opposite: the smallest
displacements sit at chance (0.984) and curvature *peaks* in the middle deciles. The zig-zag
appears when the scene actually moves. It is a property of the representation's response to
motion, which is what a straightening objective acts on.

## 4. What this means for the proposal

- **Drop the `(1 - s_t)` gating.** Its premise does not hold on this data.
- **Plain temporal straightening remains well-motivated here**, and by a much larger margin than
  the mode-aware refinement ever offered: 1.10 vs a physical reference of 0.03-0.32.
- If a mode-conditioned term is kept at all, the evidence points the *opposite* way: the latent
  carries a spurious kink at boundaries that could be penalised, not protected.
- The contribution shifts from "mode-aware straightening" to **the first evaluation of temporal
  straightening on contact-rich manipulation**. Its own paper evaluates only Wall, PointMaze-UMaze,
  PointMaze-Medium and PushT -- precisely the suite `CLAUDE.md` designates
  "saturated sanity checks only, never thesis evidence".

## Caveats

1. Regime labels are the **object-displacement proxy** (`stratification/metaworld_regimes.py:3`),
   not MuJoCo contact ground truth. A true contact sensor could place switches elsewhere, though
   "the object started/stopped moving" is arguably the boundary that matters for planning cost.
2. `dino_wm_metaworld` and `jepa_wm_metaworld` share one frozen DINOv2 encoder, so the agreement
   between the two columns is **not** an independent-representation replication (n=1).
3. All of this is measured on demonstration trajectories. The documented CEM failure is
   `rho_final` going CI-clean negative *under search*, i.e. off the data manifold
   (`diagnosis/docs/CURRENT_STATUS.md:71`). Straightening constrains geometry only along observed
   trajectories, so this pre-gate does not address that risk at all. It remains the #1 threat.

---

# CORRECTION (2026-08-21, later same day)

**Section 2 of this document is wrong. The premise it claimed to refute is in fact supported.**

The original comparison of physical-object curvature at mode switches versus within modes was
confounded by stationary frames. Curvature is the angle between consecutive displacements; when
the object barely moves, that angle is noise and the curvature sits near chance.

Object displacement per step:

| group | n | median | fraction < 1 mm |
|---|---|---|---|
| at mode switch | 4,585 | 4.77 mm | 0.180 |
| within mode | 65,255 | **0.71 mm** | **0.522** |

Over half of within-mode transitions have an essentially stationary object. The original
"within 0.320 vs switch 0.184" was therefore comparing mostly-still frames against
mostly-moving ones, not smooth physics against kinked physics. The `|big` variant did not fix
this: that filter thresholded the *latent* displacement, which does not imply the object moved.

Recomputed with the object required to move at least 3 mm on both legs of the angle:

| group | n | curvature |
|---|---|---|
| at mode switch | 3,099 | **0.0704** |
| within mode | 20,833 | **0.0291** |
| **switch − within** | | **+0.0413, CI [+0.0349, +0.0481] — CI-clean POSITIVE** |

The physical object trajectory kinks **2.4x more** at contact-mode switches. That is exactly the
premise behind gating the curvature loss with `(1 - s_t)`: at those steps the real dynamics
genuinely turn a corner, so forcing the latent straight through them asks the representation to
erase a real event.

This also reconciles the rho audit (`RHO_GATE_RESULT.md`), where the mode-gated arm was the only
one with CI-clean positive rho_final. Two independent lines now agree.

What still stands from the original analysis:
- the frozen latent is more curved than chance (1.096 vs 1.000) while physics is straight;
- the latent kinks more at switches than within modes (+0.029, CI-clean);
- the excess curvature is structural rather than encoder jitter.

What does not stand: the claim that gating protects the wrong transitions, and the
recommendation to drop it. That recommendation was withdrawn.
