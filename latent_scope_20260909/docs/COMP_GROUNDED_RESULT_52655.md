# Grounded shared-teacher composition prototype — result of job 52655

> **Scope correction, 2026-09-15, after review.** The composer verdict stands, but three readings below are narrowed:
> 1. **"Grounding helped" is not a controlled before/after comparison.** The target changed from empty-history counts to history-conditioned increments.
> 2. **The teacher is frozen**, so composition cannot have degraded the *target* representation. The lower correlation of `segment_comp` sequential may reflect the student; the mechanism is not identified.
> 3. **The teacher reading the true future is not an MAE ceiling.** At 128 steps its MAE (2.103) is worse than the sequential student's (1.424), so length extrapolation and calibration effects remain.
>
> Next step, locked separately: `SEGMENT_VS_FRAME_EVAL_PROTOCOL.md`.


Date: 2026-09-15. Protocol: `COMP_GROUNDED_PROTOCOL.md` (locked before training).

- Job 52655: 1 GPU, COMPLETED in 00:36:14, exit 0.
  - Teacher: 8k steps. Students: 12k steps each. All training completed.
  - Profile runs: 52652 failed on the cuDNN RNN backward issue, which was fixed; 52653 passed.
- Result: `/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/grounded_full_52655/grounded_result.json`.
- Data: test episodes dropped before loading. Train 345 episodes; `val_select` 38; decision set
  `val_decide` 38 (409 contact windows at 64, 607 at 128). One seed.
- `val_decide` was read in earlier rounds, so this is a development result.

## Locked decision: `COMPOSITION_NOT_BETTER_THAN_MATCHED_CONTROLS`

### Primary metric: history-conditioned count increment, MAE on contact windows (lower is better)

| Split | `segment_comp` composed | Best matched control | Relative MAE reduction | Composed − best, 95% CI |
|---|---:|---|---:|---:|
| 128 = 64+64 | 1.623 | `segment_nocomp` sequential, 1.424 | **−14.0%** (worse) | +0.199 [−0.005, +0.402] |
| 64 = 24+40 | 1.003 | `segment_nocomp` sequential, 0.977 | −2.7% | +0.026 [−0.052, +0.101] |
| 128 = 32+32+64 | 1.643 | `segment_nocomp` sequential, 1.419 | −15.8% (worse) | **+0.224 [+0.046, +0.411]** |
| 64 = 32+32 | 0.993 | `segment_nocomp` direct whole, 1.002 | +0.9% | −0.009 [−0.069, +0.051] |

The rule required at least 10% improvement on both primary splits. Instead, composition is
worse than the sequential memory rollout on long windows. On the three-part 128-step split
the interval excludes zero **in the unfavorable direction**.

### All inference paths (count increment; contact windows)

| Path | 64 = 24+40 MAE / r | 128 = 64+64 MAE / r |
|---|---:|---:|
| `segment_nocomp` sequential (fixed teacher update U) | **0.977** / 0.44 | 1.424 / **0.55** |
| `segment_comp` sequential (same arm, no composer) | 1.003 / 0.40 | 1.430 / 0.51 |
| `segment_comp` composed (learned composer) | 1.003 / 0.40 | 1.623 / 0.49 |
| `frame_teacher` sequential | 1.040 / 0.44 | 1.531 / 0.43 |
| `frame_teacher` composed (teacher encoder over rolled frames) | 1.053 / 0.36 | 2.121 / 0.44 |
| `segment_nocomp` direct whole (128 unseen) | 1.002 / 0.43 | 1.941 / 0.50 |
| Teacher reading the true future (diagnostic only) | 0.848 / 0.61 | 2.103* / 0.66 |

\* The teacher's heads were trained on 32/64-step windows. At 128 its correlation stays
highest, but its scale is miscalibrated, hence the high MAE.

Readings:

- **The composer adds nothing on top of its own sequential path.** Sequential and composed
  are equal at 64 steps; at 128 the composer is worse by +0.19 MAE [−0.01, +0.39].
- **Composition does beat one-shot length extrapolation**: against direct whole at 128,
  −0.32 MAE [−0.43, −0.21]. But memory-update rollout gets that benefit without a learned
  composer, and gets more of it.
- **Composition training slightly hurts the student's representation.** `segment_comp`
  sequential correlates below `segment_nocomp` sequential (0.51 vs 0.55 at 128). This
  matches the concern that composition pulls summaries toward what is easy to merge.

## Did grounding help? (secondary; different targets, so indicative only)

- **Teacher reading the true future:** r 0.61 (64) and 0.66 (128) for history-conditioned
  count increments. The closed self-supervised spatial-anchor summaries reached r 0.47 / 0.52
  for empty-history counts on the same `val_decide` windows.
- **Student predictions from prefix and actions:**
  - `segment_nocomp` sequential: r 0.44 (64 = 24+40) and 0.55 (128 = 64+64).
  - The closed spatial-anchor no-composition arm (additive readout): 0.32 and 0.49.
- **Absolute accuracy stays modest:** about 1 contact of MAE at 64 steps, with a large gap
  to the teacher reading the true future.

This corresponds to the protocol row *"rule fails and the controls improve over the
closed version"*: grounding helps, but the learned composer should not be defended on this
design.

## Conclusion

Across four evaluations the learned composer has never beaten the matched system without
composition:

- 52634, the self-supervised pilot;
- 52642, the corrected re-evaluation;
- 52644, the spatial target fix;
- 52655, the grounded shared teacher.

With a grounded, frozen teacher, the best way to predict long-horizon history-conditioned
effects is **sequential memory rollout with a fixed learned update**. A learned composition
operator does worse.

**The learned-composer claim is closed on this data.** What remains is a grounded latent
memory world model (privileged teacher, distilled student, sequential update). It has
moderate accuracy and is close to existing privileged-distillation and recurrent
world-model work. It is a possible engineering baseline, not a composition result.
