# Single controlled target correction — result of job 52644

Date: 2026-09-15. Protocol: `COMP_PILOT_TARGET_FIX_PROTOCOL.md`, locked before training.

- Job 52644: 1 GPU, COMPLETED in 00:31:11. Both spatial-anchor arms trained the full 12k steps; there was no test evaluation.
- Profile run 52643 passed first.
- Result: `/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/target_fix_full_52644/target_fix_result.json`.
- Decision set: `val_decide`, 38 validation episodes (264 / 409 / 607 contact windows at 32 / 64 / 128). Test episodes were excluded from loading. One seed.

## Locked decision: `TARGET_NOT_IMPROVED_STOP_SCRUB_FOR_THIS_DIRECTION`

The composition criterion also fails. Even if the target rule is read leniently, the second
row of the decision table applies: close this version on Scrub. Both readings end the
Scrub line for this direction.

## 1. Target adequacy: true-future-summary readout, S2, Pearson r of count_empty (contact windows)

| Arm | Old anchor (32 / 64 / 128) | Spatial anchor (32 / 64 / 128) | Δr by length | Mean Δr | Rule (≥ +0.05 for both arms) |
|---|---|---|---|---:|---|
| no composition | 0.34 / 0.40 / 0.49 | 0.39 / 0.47 / 0.52 | +0.04 / +0.07 / +0.02 | **+0.043** | not met |
| composition | 0.25 / 0.29 / 0.54 | 0.39 / 0.45 / 0.47 | +0.14 / +0.16 / −0.07 | +0.076 | met |

- The spatial anchor helps short-window summaries, mainly in the composition arm: at 64 steps
  Δr +0.16 [+0.04, +0.27]. It does not help at 128.
- The no-composition arm misses the threshold narrowly, with every interval spanning zero.
- Absolute adequacy stays moderate: r 0.39–0.52 at best, against constant MAE 0.74 / 1.16 / 1.77.

## 2. Composition within the corrected target (S2, count_empty, contact windows)

Δr is `segment_comp_spatial:composed` minus the reference; positive favors composition.

| Split | vs no-composition additive | vs no-composition direct whole |
|---|---:|---:|
| 64 = 24+40 (primary) | +0.02 [−0.14, +0.17] | −0.05 [−0.19, +0.07] |
| 128 = 64+64 (primary) | −0.02 [−0.15, +0.10] | −0.02 [−0.14, +0.10] |
| 64 = 32+32 | +0.04 [−0.11, +0.17] | −0.04 [−0.18, +0.12] |
| 128 = 32+32+64 | −0.03 [−0.12, +0.07] | −0.04 [−0.18, +0.08] |

- **Composition rule not met:** point estimates are mixed and near zero, no interval excludes
  zero, and several estimates are negative.
- In the hard stratum the estimates swing widely with overlapping intervals:
  64=32+32 is +0.24 [−0.02, +0.43], while 64=24+40 is −0.30 [−0.64, +0.05].
  With 22–28 episodes this is noise-dominated.
- Composition is again not better than the matched no-composition system. It still does not
  approach the supervised `map_union` reference: its composed-split r is 0.52–0.63, against
  0.36–0.49 for the composition arm.

## 3. Reference points (52634 checkpoints, same `val_decide` windows)

| Arm | Composed or native r, 64 = 24+40 | Composed or native r, 128 = 64+64 |
|---|---:|---:|
| `map_union` native heads (supervised map) | 0.52 | 0.58 |
| `frame_rollout` composed | 0.36 | 0.45 |
| old `segment_comp` composed | 0.39 | 0.44 |
| new `segment_comp_spatial` composed | 0.36 | 0.49 |

## Final judgement for this programme version

Across the pilot (52634), the corrected re-evaluation (52642) and the single target correction
(52644), the learned composer never beats the matched no-composition system on held-out
composition. Correcting the target to keep spatial structure gives only a modest,
length-dependent improvement in how much contact information the summaries hold.

- **Closed:** compositional trajectory-JEPA (learned segment summary plus learned composer) on
  Scrub demonstrations with DINOv2-S 4×4 features, at one seed.
- **Not established:** that composition cannot help anywhere. Absolute readability of contact
  counts stays moderate for every representation, including supervised ones, so the ceiling
  on this arena is low.
- **Not done, deliberately:**
  - more seeds of this configuration;
  - patch 16×16 encoding;
  - board-pose probes;
  - a different arena;
  - further target variants.

Per the protocol, this ends the Scrub line for this direction. Any future return to the idea
would need a new, separately justified setting, not a continuation of this chain.
