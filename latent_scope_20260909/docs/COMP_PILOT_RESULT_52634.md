# Offline composition pilot — result of job 52634

> **Correction, 2026-09-15 (job 52642, `COMP_PILOT_REEVAL_RESULT_52642.md`):**
> - The readout used `LayerNorm(1)` on `map_union` map tokens, which zeroes every input, so the "map_union collapsed" finding was an evaluation bug. With the fix, its native heads reach r 0.41–0.56.
> - The "constant" baselines below were computed from test labels.
> - The direct-128 "length-generalization hint" disappears with a calibrated readout.
> - Raw-input probes show the inputs themselves read contact counts only moderately (r 0.44–0.68), with no gain from vision over proprio.
>
> Read the sections below with these corrections.


Date: 2026-09-15. Protocol: `COMP_PILOT_PROTOCOL.md` (locked before training).

| Job | What | Status |
|---|---|---|
| 52634 | Full pilot: 1 GPU, one seed, four arms, 12k steps each | COMPLETED in 00:46:07 |
| 52638 | Sanity baselines: CPU, constant predictor and Pearson r | COMPLETED |

Artifacts: `/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/full_52634/`
(`pilot_result.json`, `sanity_baselines.json`, `*/test_rows.json`, checkpoints).
Test set: 83 episodes. Training ran all 12k steps for every arm; no arm hit its time budget.

## Verdict: inconclusive about composition, because the target itself carries little contact information

The protocol's third decision rule applies: *all arms poor, including the true-summary
readout ceiling → target or input problem; do not conclude against composition.*

### 1. No representation predicts contact counts well, not even the true future summary

Test windows that contain contact. The constant MAE uses the test-stratum mean, so it is an
optimistic baseline.

| Window | Constant MAE | Best arm MAE | Pearson r, predicted (range over arms) | r, readout of the **true future** summary |
|---|---:|---:|---:|---:|
| direct 32 | 0.83 | 0.94 (frame) | 0.25–0.28 | 0.26–0.35 |
| direct 64 | 1.33 | 1.35 (frame, comp) | 0.35–0.38 | 0.39–0.48 |
| direct 128 | 2.01 | 2.13 (comp) | 0.09–0.42 | 0.43–0.51 |

In MAE, no arm beats a constant. Correlations are weak. The readout of the EMA summary of
the **actually observed** future clip reaches only r ≈ 0.3–0.5. At this operating point the
learned latent (frozen DINOv2-S, 4×4 pooled, fused without contact supervision) keeps little
count information. Better prediction or composition cannot recover information the target
does not contain.

### 2. Composition comparisons (count_empty, hard stratum, paired episode bootstrap)

| Split | comp:composed − nocomp:additive | comp:composed − nocomp:direct whole | comp:composed − frame:composed |
|---|---:|---:|---:|
| 64 = 32+32 | −0.31 [−0.62, −0.01] | −0.08 [−0.32, +0.16] | −0.01 [−0.22, +0.21] |
| 64 = 24+40 | −0.16 [−0.40, +0.11] | −0.02 [−0.26, +0.22] | +0.09 [−0.09, +0.30] |
| 128 = 64+64 | **+0.65 [+0.26, +1.05]** (worse) | **−0.93 [−1.20, −0.68]** (better) | −0.20 [−0.43, +0.00] |
| 128 = 32+32+64 | +0.40 [−0.11, +0.92] | −0.94 [−1.17, −0.70] | −0.14 [−0.34, +0.06] |

Negative means the composition arm has lower error. These MAE differences mostly reflect
readout calibration rather than information:

- The readout was trained on 32/64-step windows and applied unchanged to 128-step composed
  summaries. Composed predictions stay near the 32/64 scale: predicted mean about 2.4 versus
  a true mean of 5.5 in the hard stratum. Additive sums of part readouts get the scale right
  by construction.
- Correlation, which ignores scale, is similar across methods. On 128 = 64+64 contact windows:
  comp composed r 0.47, comp additive 0.47, nocomp additive 0.46, frame composed 0.47.

**One hint worth a replication, not a claim:** direct 128-step prediction, a length never
trained, holds up in the composition-trained arm (r 0.34, MAE 2.13) but collapses without
composition (r 0.09, MAE 2.77). Composition may regularize length generalization. One seed,
weak signal overall.

### 3. Evaluation defects found (fix before any rerun)

1. **`map_union` collapsed** to a constant (r ≈ 0, identical predictions). BCE on a 144-cell
   map with very sparse positives, and no positive weighting, drove the map to zero. It is
   not a valid control in this run.
2. **Map F1 is 0 for every arm**: no readout probability exceeds 0.5 on sparse maps. Use
   average precision or a calibrated threshold.
3. **Readout scale extrapolation**: readouts must be trained on the total lengths they
   evaluate (including 128), or composition must be compared with scale-free metrics
   (correlation, ranking) as primary.
4. **Stratum mismatch**: readouts are trained on all windows but evaluated on contact
   windows, which biases predictions toward zero. Report both strata, or train on the
   evaluated stratum.
5. **Parameter counts differ** (comp 13.4M, nocomp 11.9M, frame 5.6M, map 6.6M).

## What this means

The pilot did what it was supposed to do cheaply: it found the binding constraint before any
planning work. It is **not** evidence against composition. The next question is which
of these is true:

- **the inputs lack the signal**: 4×4-pooled DINOv2 frames plus proprio cannot express where
  and how often the sponge touched the board; or
- **the self-supervised target discards it**: the information is in the inputs but not in the
  learned latent.

The cheapest discriminating test is a supervised readout of contact counts directly from the
**raw observed** future inputs (pooled DINO sequence + proprio), and from proprio alone,
using the same windows and splits. No world model is needed. If raw inputs read well, fix
the target; if not, change the input resolution or the task before rerunning the comparison.
