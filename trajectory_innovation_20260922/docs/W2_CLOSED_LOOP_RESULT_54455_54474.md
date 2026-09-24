# W2 result (jobs 54455 tasks 0-3, 54474 tasks 4-7, aggregation 54475)

Protocol: `W2_CLOSED_LOOP_PROTOCOL.md`. Fresh roots 1600–1999 (n = 400). Frozen S1-b reader
(`s1_train_54419/reader_s1b.pt`), trained on task reward (cov8) under `RESEARCH_DESIGN.md` §5a.
Preflight: the closed-loop scorer reproduced the S1 held-out scores (argmax agreement 0.94, relative diff 0.04).
Tasks 4–7 of 54455 failed at 5 s because of an unrelated unit-test bug. They were rerun as 54474 with byte-identical W2 code.

## Verdict: **PASS** (not STRONG)

| Arm | Success (n = 400) |
|---|---|
| P0 (policy default) | 0.625 |
| PHYS8 (privileged 8-step coverage oracle) | 0.765 |
| **RANK8 (learned reader on actual futures)** | **0.710** |

- **Primary:** RANK8 − P0 = **+8.5 pp, CI [+2.75, +14.25]**, McNemar 88 vs 54, p = 0.005.
- **Oracle:** PHYS8 − P0 = +14.0 pp [+8.75, +19.25], p = 1e-6.
- RANK8 − PHYS8 = −5.5 pp [−11.0, 0.0].
- **Retention:** 0.61, CI [0.24, 1.00]. Below the 0.80 STRONG line.
- **Within bank:** Spearman(reader, cov8) 0.23; the reader's choice matches the oracle's 27% of the time.

## What this does and does not show

- **It shows (ladder level 1):** a learned image reader with no simulator input at test time, conditioned on
  the decision-time frame and trained on sibling branches, selects among the policy's own candidates well
  enough to raise closed-loop success CI-clean on fresh roots, with the model frozen before these roots were seen.
  Earlier scorers had failed to do this: fixed L2 (gate C), the context-free hindsight reader (C2, confirm), and label-free contrast (S1-a).
- **It does not show a world-model result.** Candidates were scored on their *actual* futures from simulator
  branches, so dynamics were privileged. The code (level 2) and predicted code (level 3) are untested.
- **Caveats:**
  - The reader's label is the task's own dense reward (coverage), so the claim is not self-supervised.
  - PushT only; one reader seed.
  - About 40% of the oracle headroom remains untaken.
