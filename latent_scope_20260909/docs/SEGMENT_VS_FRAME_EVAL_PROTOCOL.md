# Segment vs frame prediction from the 52655 checkpoints (evaluation only)

Locked 2026-09-15, before measurement. No retraining, no new variants, no arena change.
The learned composer is closed; it is not re-tested here.

## Question

> Does predicting a segment's effect summary directly, then advancing memory, predict
> history-conditioned effects better or more cheaply than predicting every frame and reading
> the effect from the predicted frames?

## Inputs

- Checkpoints: the frozen grounded teacher, `segment_nocomp` and `frame_teacher` from
  `runs/grounded_full_52655`.
- Data: the same splits. Test episodes are dropped; decisions use `val_decide`, and train
  windows fit the constants.

## Measurements

1. **Accuracy.**
   - Paths: the sequential paths of both students on 64=32+32, 64=24+40, 128=64+64 and
     128=32+32+64.
   - Metrics: `count_increment` MAE and Pearson r (primary), `area_increment` (secondary).
   - Strata: all windows and contact windows.
   - Constants fitted on train windows of the same total length.
   - Paired episode bootstrap (1000 resamples) of segment minus frame.
2. **Inference cost.**
   - Same windows, batch sizes (32, 256) and horizons (64 = 32+32, 128 = 64+64), bf16, after warmup.
   - Median wall time of 20 repetitions, with CUDA synchronization and peak memory.
   - Reported with and without the shared history context (both students use the same
     frozen teacher context).
   - Includes predictor, memory advance and teacher readout heads; the frame path also includes
     the teacher encoder over predicted frames. cuDNN is enabled for both at inference.
3. **Long-segment decomposition (teacher reading observed frames, diagnostic).** For 128 and 64:
   - the whole window read at once;
   - two halves with the observed memory continuation;
   - two halves with the learned update U.

## Locked decision

Keep segment prediction as the basis for the next method step if **either** holds on both
primary splits (64=24+40 and 128=64+64), contact windows:

- **(a) Accuracy:** segment MAE at least 5% lower than frame, with the paired bootstrap MAE
  interval excluding zero on both primary splits.
- **(b) Cost:** segment at least 2× faster (batch 256, horizon 128, excluding shared context),
  with MAE no worse than frame (upper interval bound of segment − frame ≤ 0).

Otherwise close this Scrub branch as a method direction; keep the code only as an
engineering baseline. One seed and a previously read `val_decide`, so a pass only permits
designing the next step (candidate ranking from shared prefixes). It is not a method result.
