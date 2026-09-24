# Gate C2 result (jobs 53938, 53992, 54139)

Protocol: `GATE_C2_PROGRESS_READER_PROTOCOL.md` (pinned before any C2 data).
Roots 1200–1299, 100 closed-loop episodes per arm, one reader seed.
Aggregation was run twice (54139, and 54144 from a peer session); the two summaries are identical.

## Verdict

| Rule | Result | Verdict |
|---|---|---|
| 1. B replication: PHYS8 − P0 ≥ 10 pp and CI lower bound > 0 | +10 pp, CI [−1, +22], McNemar p = 0.13 | **EXTEND (not replicated)** |
| 2. C2 retention ≥ 0.80 | ratio 1.40, CI [0.43, 7.0] | **NOT INTERPRETED** (rule 1 failed) |

Under the pre-registered rules, C2 has no PASS/FAIL verdict: the myopic physical oracle's
headroom did not replicate CI-clean on these roots, so retention against it is undefined in the protocol's sense.

## Numbers

| Arm | Success (n = 100) |
|---|---|
| P0 | 0.65 |
| PHYS8 | 0.75 |
| PROG8 | 0.79 |

- PROG8 − P0: **+14 pp, CI [+3, +25]**, McNemar 23 vs 9, p = 0.020 (reported as a raw paired difference; not the primary).
- PROG8 − PHYS8: +4 pp, CI [−8, +16].
- Offline retained gap on P0 decisions (vs 8-step coverage): 0.13 [0.05, 0.20].
- Within-bank Spearman(progress score, 8-step coverage): 0.04 over 1840 decisions.

Reader training (53938): held-out Spearman(predicted, true log-distance) 0.854; goal-only control 0.169.
Train loss 0.09 vs val MSE 0.39 (some overfitting).

## Diagnostic for gate C's failure (not a gate)

Mean within-bank Spearman over P0 decisions:

| Score | agent displacement | block displacement | block rotation | 8-step coverage |
|---|---|---|---|---|
| gate-C L2 (VIS) | −0.01 | 0.06 | 0.05 | 0.07 |
| progress (PROG) | 0.02 | −0.03 | −0.02 | 0.04 |

The hypothesis "agent position dominated the L2 score" is **not supported**: the L2 score is
close to uncorrelated with agent displacement as well as with everything else.

## What this does and does not show

- It shows: selecting among the policy's own 8 candidates by a label-free, goal-conditioned
  reader on the actual future raised closed-loop success over the policy's default draw on fresh roots,
  with a CI that excludes zero. Because the 8 candidates are i.i.d. policy draws, P0 is also
  the distribution of a random pick, so this is not a "more samples" artifact of choosing at random.
- It does not show a C2 PASS: that comparison was pre-registered against PHYS8 and PHYS8's gap was not CI-clean.
- The reader and the 8-step coverage oracle almost never agree within a bank (ρ ≈ 0.04), yet both help.
  Consistent with the protocol note that the reader targets long-horizon progress while the oracle is myopic, but
  **we do not yet know what the reader is keying on**; none of the logged displacement features explains it.
- One reader seed, one set of 100 roots, and the PROG8 vs P0 contrast was secondary. Repo history (encoder-LoRA 5/16)
  says single-run wins must be replicated before they are relied on.

## Next step (not run; needs a new pre-registered protocol)

Confirm PROG8 vs P0 as the **primary** contrast on fresh roots (the reserved 1100–1199 are B-extension roots for gate B, not for this),
with a second reader seed, before building a predictor on top of the reader.
