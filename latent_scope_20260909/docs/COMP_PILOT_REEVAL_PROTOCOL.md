# Re-evaluation of the 52634 checkpoints and raw-input probes

Locked 2026-09-15, before any re-evaluation outcome. It covers one bounded batch:

- no retraining of the four world-model arms;
- no new seeds;
- no planning.

## Why

The 52634 evaluation had a confirmed defect. The readout normalized each token with
`LayerNorm(token_dim)`. For `map_union` the tokens are 1-dimensional, so `LayerNorm(1)`
outputs zero for every input. That readout never saw the predicted map, and its
"collapse" is not evidence about the map model. Other weaknesses:

- map F1 at a 0.5 threshold;
- constants computed from test labels;
- readouts trained on 32/64 but applied to 128;
- readouts trained on all windows but reported on contact windows.

The "true-summary ceiling" result also cannot separate three causes: the DINO pooling loses
information, the target encoder discards it, or the readout is mismatched.

## Changes (evaluation only)

1. **Readout:** same attention-pool architecture; inputs standardized with train-set per-feature
   mean and standard deviation instead of LayerNorm over the token dimension. Retrained for
   every arm on frozen representations.
2. **Map quality:** average precision, per window and pooled over contact windows, next to the
   average precision of a train cell-frequency prior.
3. **Constants:** fitted on train windows (overall mean for all windows; contact-window mean
   for contact strata, which uses label information and is flagged as such).
4. **`map_union` native heads:** map and scalar heads evaluated directly, before any readout.
5. **Two readout settings, always reported separately:**
   - **S1 (unseen):** readout trained on direct 32/64 predictions only. The whole pipeline
     never saw 128-step totals. This matches the 52634 lock.
   - **S2 (calibrated):** readout trained on train-episode representations from the same
     inference path, including 128-step totals. The predictor still never saw 128; only
     the readout is calibrated.
6. **Primary composition comparisons use Pearson r (scale-free) and MAE**, with a paired
   episode bootstrap (1000 resamples), for contact and hard strata.

## Raw-input probes (Step B)

Supervised models trained directly on observed inputs: stride-4 history plus every frame
of the window, same episode split, targets and window starts, lengths 32/64/128.

| Variant | Inputs |
|---|---|
| `dino_proprio_action` | pooled DINO + proprio + actions |
| `dino_action` | pooled DINO + actions |
| `proprio_action` | proprio + actions |

They measure how much contact information the pilot's inputs contain for a supervised
reader. They say nothing about prediction of unexecuted actions.

## Decision rules

| Outcome after correction | Decision |
|---|---|
| Existing checkpoints read clearly better than in 52634 | Re-assess composition on these checkpoints before changing architecture |
| Raw inputs read well, learned summaries read poorly | Allow **one** target correction, applied equally to both segment arms |
| Raw inputs also read poorly | No basis to keep investing in this feature configuration |
| Controls read well and composition still adds nothing | Direct negative evidence at this operating point; confirm, then close this version |

"Well" is judged against the train-fitted constant and the probe ceiling on the same
windows, not against an absolute threshold. One seed throughout: the output decides the next
bounded step, not a method claim.

## Budget

One GPU, 3 h walltime cap for the full batch, after a short profile run. The 52634
checkpoints, features and labels are reused unchanged.
