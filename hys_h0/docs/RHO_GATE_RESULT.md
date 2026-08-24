# HyS-JEPA gate result — does straightening survive CEM search?

Date: 2026-08-21. DINO-WM MetaWorld, oracle-dynamics CEM, 16 episodes/cell, seed0=41000,
strict success. Config copied from `scripts/slurm_cem_preselection_audit.sh` so the new arms
are directly comparable to the l2 dumps already on disk. 112 candidate populations per cell.

## Validation of the analysis path

The analyzer reproduces the published baseline numbers (`diagnosis/docs/CURRENT_STATUS.md:71`):

| cell | published rho_init | here | published rho_final | here |
|---|---|---|---|---|
| DINO push l2 | 0.25 | **+0.247** | -0.08 | **-0.085** |
| DINO pick l2 | 0.02 | **+0.022** | -0.09 | **-0.094** |

## Result

| arm | rho_init | rho_final | verdict |
|---|---|---|---|
| push l2 | +0.247 [+0.163, +0.321] | **-0.085 [-0.135, -0.033]** | FAIL — anti-aligned under search |
| push straight `none` | -0.077 [-0.138, +0.000] | -0.011 [-0.040, +0.018] | anti-alignment removed, no ordering |
| push straight `off` | -0.035 [-0.094, +0.028] | -0.026 [-0.063, +0.012] | anti-alignment removed, no ordering |
| push straight `switch` | +0.198 [+0.107, +0.274] | **+0.071 [+0.020, +0.118]** | **PASS** |
| pick l2 | +0.022 [-0.048, +0.105] | **-0.094 [-0.136, -0.051]** | FAIL |
| pick straight `none` | +0.051 [-0.017, +0.119] | -0.003 [-0.046, +0.039] | anti-alignment removed |
| pick straight `off` | +0.002 [-0.072, +0.076] | +0.001 [-0.045, +0.050] | anti-alignment removed |
| pick straight `switch` | **+0.099 [+0.010, +0.185]** | +0.010 [-0.033, +0.059] | anti-alignment removed |

## Two findings

**1. Any learned projection removes the CI-clean negative rho_final — including the arm with
no curvature term at all.** The `off` control (prediction + reconstruction + VICReg, zero
straightening) moves push from -0.085 to -0.026 and pick from -0.094 to +0.001, both spanning
zero. So the anti-alignment that CEM induces on raw latent L2 is not specifically a curvature
problem; a learned low-dimensional bottleneck is enough to stop it. That weakens any claim that
*straightening* is the operative ingredient.

**2. Only the mode-gated arm achieves positive ordering that survives search.** `switch` is the
only cell anywhere with CI-clean positive rho_final (push +0.071), and the only one with
CI-clean positive rho_init on pick. This CONTRADICTS the pre-gate recommendation in
`PREGATE_CURVATURE.md`, which argued the gating was backwards because the physical trajectory
is smoother, not kinkier, at contact-mode switches.

That pre-gate measurement stands — it described the frozen latent and the physical state. What
it could not predict is what *excluding* those transitions does to the learned projector. The
geometry of the frozen representation does not determine the best training signal.

## Why this is not yet trustworthy

- **Single training seed.** `none` and `switch` have nearly identical curvature (0.484 vs 0.527)
  and object decodability (6.79 vs 6.73 cm) yet differ by 0.082 in rho_final. Two representations
  that match on every summary statistic should not diverge that much downstream. This is the
  shape of a seed artefact. Replication launched (job 44339, seeds 1 and 2, both arms).
- **One task.** The PASS is push only; on pick `switch` is +0.010, spanning zero.
- **Small magnitude.** +0.071 is a weak ordering in absolute terms — better than anti-aligned,
  far from the +0.5 or so that would make an L2-style cost dependable.
- **R_sel moves the wrong way.** Selection regret in object-goal distance is *higher* for
  `switch` (1.14 cm init, 0.42 final) than for the other arms (0.03-0.11). rho is computed
  against the shaped cost (object + 0.5x hand-approach), R_sel against object distance alone,
  so the two can diverge — but it means `switch` is not straightforwardly picking better
  candidates by object progress.
- **Object information is still degraded.** Both straightening arms read object position at
  ~6.7-6.8 cm against a 5 cm success radius (`off` keeps it at 2.79 cm). Improved ordering was
  bought at the cost of object precision.

## What the replication decides

If seeds 1 and 2 reproduce `switch` > `none` on rho_final, the mode-gating has a real effect and
the pre-gate's mechanistic story was simply the wrong predictor of it. If they do not, the single
PASS was noise and the honest conclusion is finding 1 alone: a learned bottleneck removes CEM's
anti-alignment, but nothing here produces a dependable planning cost.
