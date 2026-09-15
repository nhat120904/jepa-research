# Re-evaluation of the 52634 checkpoints and raw-input probes — job 52642

> **Scope correction, 2026-09-15, after review.** Three conclusions below are narrower than first written:
> 1. "The inputs are the main limit" is **not established.** Probes measure one reader (a `FrameFuser` that compresses all patches and cameras to one token per frame) at one training budget. That is not an information ceiling. At 128 steps, proprio+action cuts MAE from 2.00 to 1.48 (about 26%), which is useful signal.
> 2. "Images add nothing" should read: **no benefit of images was seen with this reader and budget.** Vision beating proprio is not a requirement for the composition hypothesis; it matters only for a vision-specific claim.
> 3. The 128-step extrapolation gap is **readout-dependent**: present with the S1 readout, absent with S2. That does not show a general representation advantage, but it does not prove that no effect exists either.
>
> Unchanged: composition is not better than the matched no-composition arm. The composer beats additive readout on its own representation only.
> Decision taken: one controlled target correction for both segment arms, then close. See `COMP_PILOT_TARGET_FIX_PROTOCOL.md`.


Date: 2026-09-15. Protocol: `COMP_PILOT_REEVAL_PROTOCOL.md` (locked before outcomes).
1 GPU, COMPLETED in 00:11:43. Profile run 52641 passed first.
Result: `/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/reeval_full_52642/reeval_result.json`.
One seed; frozen checkpoints from 52634; test set 83 episodes, 1689 windows (1344 with contact at 128 steps).

## 1. The readout bug is confirmed and fixed

The 52634 readout applied `LayerNorm(1)` to `map_union`'s one-dimensional map tokens, and
that outputs zero for every input. With train-set standardization:

- `map_union` readout on predicted maps reaches Pearson r 0.36 / 0.50 / 0.48 for contact
  counts (32 / 64 / 128, contact windows), instead of ≈0.
- `map_union` native heads, before any readout, reach r 0.41 / 0.56 / 0.49.

The 52634 claim that `map_union` collapsed was an evaluation artifact.

## 2. Raw-input probes: the observed inputs themselves carry only moderate count information, and images add nothing beyond proprio

Supervised probes on **observed** history plus observed window frames (no prediction).
Contact windows, `count_empty`:

| Length | DINO+proprio+action r | DINO+action r | proprio+action r | Probe MAE (best) | Train-fitted constant MAE | Map micro-AP (best probe) | Cell-prior AP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.37 | 0.32 | **0.44** | 0.85 | 0.83 | 0.049 | 0.027 |
| 64 | 0.55 | 0.52 | **0.57** | 1.15 | 1.32 | 0.072 | 0.039 |
| 128 | 0.67 | 0.65 | **0.68** | 1.48 | 2.00 | 0.088 | 0.057 |

- **Even reading the observed window with full supervision, count correlation is only
  0.44–0.68**, and at 32 steps MAE does not beat a constant. Map AP is under 2× the prior:
  where on the board contacts happened is barely readable.
- **Proprio+action alone matches or beats every DINO variant.** At this feature
  configuration (DINOv2-S, 4×4 pool), images add no count information beyond the
  end-effector trajectory. The premise that Scrub's hidden contact history makes vision
  necessary is not supported at this resolution.

## 3. Learned summaries versus the probe ceiling

Pearson r, contact windows, `count_empty`:

| Length | Best probe | `map_union` native | `segment_nocomp` true-future summary, S2 | `segment_comp` true-future summary, S2 |
|---:|---:|---:|---:|---:|
| 32 | 0.44 | 0.41 | 0.34 | 0.28 |
| 64 | 0.57 | 0.56 | 0.46 | 0.41 |
| 128 | 0.68 | 0.49 | 0.62 | 0.53 |

The self-supervised summaries lose some information relative to the inputs (about 0.1 in r),
but the dominant limit is the inputs themselves.

## 4. Composition after correction (paired episode bootstrap, count_empty)

Primary setting S2 (readout calibrated on the same inference path; the predictor never saw 128).
Δr > 0 means `segment_comp:composed` correlates better than the comparison.

| Split | vs `nocomp` additive, contact | vs `nocomp` direct whole, contact | vs `comp` additive, contact | vs `frame` composed, hard | vs `map_union` native, contact |
|---|---:|---:|---:|---:|---:|
| 64 = 32+32 | −0.14 [−0.25, −0.03] | −0.14 [−0.25, −0.01] | +0.04 [−0.03, +0.12] | +0.05 [−0.09, +0.18] | −0.20 [−0.30, −0.08] |
| 64 = 24+40 | −0.11 [−0.21, −0.01] | −0.15 [−0.24, −0.05] | +0.01 [−0.06, +0.08] | +0.06 [−0.10, +0.29] | −0.22 [−0.31, −0.12] |
| 128 = 64+64 | −0.11 [−0.20, −0.02] | −0.03 [−0.12, +0.05] | +0.08 [+0.01, +0.15] | −0.19 [−0.37, −0.01] | −0.21 [−0.29, −0.12] |
| 128 = 32+32+64 | −0.10 [−0.20, +0.01] | −0.03 [−0.10, +0.06] | +0.08 [+0.00, +0.15] | −0.08 [−0.24, +0.11] | −0.20 [−0.26, −0.13] |

MAE differences in S2 are within about ±0.15 with intervals spanning zero, except against
`map_union` native (composition worse).

- The learned composer beats the same arm's additive readout of its own parts (small, several
  intervals exclude zero). **But composing is not better than the matched no-composition arm:**
  its correlation is lower in all four splits, with intervals excluding zero in three.
  It is also worse than the supervised map control everywhere.
- **Retraction.** The 52634 "length-generalization hint" (direct 128: comp r 0.34 vs nocomp
  0.09) was a readout calibration artifact. Under S1 the gap is Δr +0.28 [+0.10, +0.44];
  with a calibrated readout (S2) it is −0.01 [−0.08, +0.06].

## Verdict against the locked decision rules

- *"Existing checkpoints read clearly better than 52634"*: only `map_union`, from the bug
  fix. The segment arms are essentially unchanged.
- *"Raw inputs read well, summaries poorly"*: **no**. Summaries trail the inputs by about
  0.1 r, but the inputs themselves read only moderately.
- *"Raw inputs also read poorly"*: **closest match.** Probes reach r 0.44–0.68, map AP is
  under 2× the prior, and at 32 steps a constant does as well. That gives no basis to keep
  investing in this feature configuration.
- *"Controls read well, composition adds nothing"*: composition adds nothing, and is
  somewhat worse than the matched no-composition arm. The controls are only moderate, so
  this is not a clean test of the idea.

**Decision: stop investing in the Scrub + DINOv2-S 4×4 configuration.** Do not run three
seeds of it. The composition method has no positive evidence at this operating point, but the
operating point is also a weak test: contact effects are poorly observable from these inputs,
and what is observable comes from proprioception rather than vision.

## What would make a further test worthwhile

A further test is informative only in a setting where both hold:

1. the accumulated effect is well recoverable from observations (probe r clearly above
   these values, well above the constant);
2. vision carries information beyond proprio.

Two cheap checks on the existing data would settle whether Scrub can meet them:

- a probe with proprio plus the privileged board pose, testing whether missing board
  location explains the ceiling;
- a probe on full 16×16 patch features for a subset of episodes.

If neither raises the ceiling substantially, close Scrub as an arena for this idea.
