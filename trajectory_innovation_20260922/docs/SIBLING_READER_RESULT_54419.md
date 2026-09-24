# Gate S1 result (jobs 54418 collection, 54419 train + offline eval)

Protocol: `SIBLING_READER_OFFLINE_PROTOCOL.md` with Amendment 1 (S1-b is a privileged diagnostic only).

**Data:**
- Training: roots 30000–30249, 7,049 decisions.
- Held-out: roots 1500–1599, 2,778 decisions, of which 1,753 have cov8 spread across siblings.

## Verdict (pre-registered): **WEEK1_KILL**

S1-a does not advance.

S1-a's failure is **uninformative about the idea**. It never learned its own training task: training
loss sat at ln 8 = 2.0794 (chance for 8-way) from step 1,000 to step 10,000. Held-out top-1 sibling
identification was 0.115, which is chance (1/8). Its outputs were effectively constant across siblings.
This is an optimisation or target failure. It is not evidence that label-free sibling contrast cannot work.
CPU job 54431 (peer session) checked whether sibling identity is recoverable from goal16 at all:
pixel nearest neighbour gets top-1 0.197 against 0.125 chance. So the signal exists but is weak: 16 policy steps later, sibling futures
have largely converged. Within-bank Spearman(cov8, cov24) is only 0.26. The S1-a target was low-signal *and* the run
did not even reach the pixel baseline.

## Held-out within-bank numbers (P0 decisions)

| Scorer | Spearman vs cov8 [95% CI] | Retained gap vs cov8 [95% CI] | Spearman vs cov24 |
|---|---|---|---|
| gate-C L2 (fixed) | 0.079 [0.055, 0.103] | 0.287 [0.207, 0.359] | 0.008 |
| C2 reader r0 | 0.039 [0.015, 0.062] | 0.141 [0.050, 0.226] | 0.022 |
| **S1-a** (label-free) | 0.024 [0.001, 0.047] | 0.063 [−0.017, 0.135] | −0.007 |
| **S1-b** (cov8 labels, diagnostic) | **0.258 [0.231, 0.285]** | **0.549 [0.479, 0.611]** | 0.052 |

Advance thresholds: Spearman ≥ 0.30 **and** gap ≥ 0.40. S1-b meets the gap threshold but misses the Spearman one. It could not advance in any case (Amendment 1).

## What this does show

1. **The image features are not hopeless for within-bank ranking.** With the right training signal (S1-b), frozen
   DINOv2 at 16×16, conditioned on the decision-time frame, recovers about 55% of the myopic oracle's per-decision gain offline.
   That is 4× the C2 reader (0.14) and about 2× the fixed L2 score (0.29).
   The encoder-resolution explanation ("96 px DINOv2 cannot see sibling differences") is therefore at most partial.
2. **Every scorer's agreement with 24-step coverage is near zero**, including S1-b's (0.05).
   Differences created by one chunk mostly wash out after 16 more policy steps.
   This matches the single-decision H_rep of +2 pp: gains come from being slightly right at every decision.
3. **The fixed L2 score is better offline than the C2 learned reader** (gap 0.29 vs 0.14). Closed loop it was also weak (+3 pp).
   Offline retained gap above 0.3 has not yet been shown to translate into closed-loop gain.

## Not concluded

- Nothing about label-free sibling contrast: that run failed to train.
- Nothing closed-loop: S1 is offline only.
- Per rule 3, readers, features and thresholds are not retuned on roots 1500–1599.
  Any fixed S1-a, and any closed-loop test of an S1-b-style reader, needs a new protocol and fresh held-out roots.
