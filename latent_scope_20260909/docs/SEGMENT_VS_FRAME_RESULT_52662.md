# Segment vs frame prediction from the 52655 checkpoints — result of job 52662

Date: 2026-09-15. Protocol: `SEGMENT_VS_FRAME_EVAL_PROTOCOL.md` (locked before measurement).
Evaluation only, no retraining.

- Job 52662: 1 GPU, COMPLETED in 00:00:19. Profile run 52661 passed first.
- Result: `/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/segment_vs_frame_full_52662/segment_vs_frame_result.json`.
- Data: `val_decide` (38 episodes, 788 windows, 409 / 607 contact windows at 64 / 128). Test excluded.
  Constants fitted on train windows. One seed, previously read validation data.

## Locked decision: `CLOSE_SCRUB_BRANCH_AS_METHOD_DIRECTION`

Neither the accuracy rule nor the cost rule passed.

## 1. Accuracy: `count_increment`, contact windows, segment minus frame (paired episode bootstrap)

| Split | Segment MAE | Frame MAE | Train-fitted constant MAE | Relative MAE reduction | Segment − frame MAE, 95% CI | Δr, 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 64 = 24+40 (primary) | 0.977 | 1.040 | 1.093 | 6.0% | −0.063 [−0.177, +0.054] | +0.00 [−0.14, +0.15] |
| 128 = 64+64 (primary) | 1.424 | 1.531 | 1.724 | 7.0% | −0.108 [−0.236, +0.028] | **+0.12 [+0.04, +0.19]** |
| 64 = 32+32 | 1.015 | 1.075 | 1.093 | 5.6% | −0.060 [−0.158, +0.030] | +0.04 [−0.06, +0.17] |
| 128 = 32+32+64 | 1.419 | 1.551 | 1.724 | 8.5% | **−0.132 [−0.246, −0.031]** | **+0.11 [+0.05, +0.19]** |

- **Accuracy rule failed.** MAE is lower by 6–7% on the primary splits, but neither primary
  interval excludes zero.
- At 128 steps, segment prediction has a clear **correlation** advantage (Δr about +0.12,
  both intervals exclude zero), and on the secondary three-part split its MAE advantage is
  significant. That was not the locked criterion, and it is a single-seed development result.
- **Both students barely beat a constant at 64 steps** on contact windows (0.98–1.08 vs
  1.09). At 128 the gain over the constant is about 17% for segment and 11% for frame.

## 2. Inference cost (bf16, median of 20, CUDA-synchronized)

| Setting | Segment | Frame | Frame / segment |
|---|---:|---:|---:|
| Batch 256, horizon 128, excluding shared context | 5.9 ms | 10.0 ms | **1.70×** |
| Batch 256, horizon 128, including context | 24.0 ms | 28.5 ms | 1.19× |
| Batch 256, horizon 64, excluding context | 4.2 ms | 7.4 ms | 1.76× |
| Batch 32, horizon 128, excluding context | 3.7 ms | 6.5 ms | 1.76× |

- **Cost rule failed:** 1.70× is below 2×, and the MAE upper bounds are above zero.
- The shared history context (fusing up to 96 history frames) dominates end-to-end cost, so the
  practical speedup is only about 1.2×.
- Parameter counts are matched: 4.74M segment vs 4.81M frame; the teacher has 7.09M.

## 3. Long-segment decomposition (teacher reading observed frames, contact windows)

| Total | Whole window at once | Two halves, observed memory | Two halves, learned update U | Constant |
|---:|---:|---:|---:|---:|
| 64 | MAE 0.848, r 0.61 | 0.852, 0.60 | 0.847, 0.61 | 1.093 |
| 128 | **MAE 2.103**, r 0.66 | **1.093, r 0.71** | **1.092, r 0.71** | 1.724 |

- The teacher's bad 128-step MAE comes from reading an **untrained length in one shot**.
  Splitting into trained lengths removes it (MAE 2.10 → 1.09) and raises correlation.
- **The learned update U is as good as continuing the observed memory** (1.092 vs 1.093).
  Advancing memory with segment summaries loses essentially nothing at this horizon.
- The students' sequential paths remain well short of this diagnostic: 1.42 vs 1.09 at 128,
  and 0.98 vs 0.85 at 64. Most of the remaining error is in predicting summaries from actions.

## Conclusion

By the locked rule, this Scrub branch is closed as a **method direction**:

- the learned composer is closed (four evaluations);
- segment-effect prediction is not clearly better or cheaper than frame prediction under
  the pre-set thresholds.

Kept as engineering facts, not claims:

1. Chunking long horizons into trained-length segments with a learned memory update is what
   makes long-horizon effect prediction work, for teacher and students alike.
2. Segment prediction shows a consistent but sub-threshold edge over frame prediction at
   128 steps (correlation +0.12, MAE −7 to −9%, 1.7× faster predictor), from one seed.

The code (`comp_pilot/grounded.py`, `segment_vs_frame.py`) stays as a baseline. Any new
method question should be chosen fresh, not as another variant on this Scrub chain.
