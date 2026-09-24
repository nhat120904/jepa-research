# Gate S1 result (jobs 54418, 54419, 54431)

Protocol: `SIBLING_READER_OFFLINE_PROTOCOL.md`, with Amendments 1–2 (report numbers only; S1-b is a diagnostic).
Held-out roots 1500–1599: 2,778 decisions, 1,753 of which have a cov8 spread across the 8 siblings.
Training roots 30000–30249: 7,049 decisions.

## Numbers (held-out, within-bank, root-bootstrap 95% CI)

| Scorer | Spearman with cov8 | Retained gap (cov8) | Spearman with cov24 |
|---|---|---|---|
| gate-C L2 to goal map (fixed) | 0.079 [0.055, 0.103] | 0.287 [0.207, 0.359] | 0.008 |
| C2 reader r0 (hindsight steps-to-goal, no C) | 0.039 [0.015, 0.062] | 0.141 [0.050, 0.226] | 0.022 |
| **S1-a** (label-free sibling contrast, with C) | 0.024 [0.001, 0.047] | 0.063 [−0.017, 0.135] | −0.007 |
| **S1-b** (cov8 ranking labels; privileged diagnostic) | **0.258 [0.231, 0.285]** | **0.549 [0.479, 0.611]** | 0.052 |

## S1-a did not train

- The training loss stayed at ln 8 = 2.079 from step 1,000 to step 10,000, i.e. identical scores for all siblings.
- Held-out contrastive top-1 was 0.115, against chance 0.125.

CPU check 54431 (`s1_signal_check.py`) asked whether the task carries signal at all:
- **Weak signal.** Pixel nearest neighbour from goal16 back to the sibling end frames reaches top-1 **0.197** against chance 0.125.
  The same holds on differences from the decision frame.
- **The data are not degenerate.** Sibling end frames and goal frames are distinct: a median of 8 distinct frames of each per decision.
- **8-step differences only partly persist.** Within-bank Spearman(cov8, cov24) = 0.26.

So the label-free task defined in S1 (identify the sibling from a frame 16 policy steps later) carries only a weak signal.
The continuation, with independent seeds per sibling, washes most of the chunk difference out.
The reader then collapsed to a constant instead of learning even that weak signal.
This is a failure of this particular self-supervised target and its optimisation. It is not evidence about label-free readers in general.

## What the numbers show

1. **The features can resolve siblings.** With a short-horizon progress signal, the same DINOv2 (96 px → 224, PCA-128)
   features and architecture reach within-bank ρ 0.26 and retain 55% of the offline 8-step oracle gap. That is 2× gate-C L2 and 4× C2.
   Most of the "encoder cannot see it" worry is ruled out for PushT at this level.
2. **The label-free S1-a target, as defined, does not carry the needed information.**
3. **All scorers barely predict cov24.** One chunk's advantage mostly does not survive two more policy chunks,
   consistent with the small single-decision H_rep (+2 pp).
4. **Offline gap does not map linearly to closed-loop gain.** Gate-C L2 had an offline gap of 0.29 but a closed-loop retention of 0.20 (+3 pp).
   S1-b's closed-loop value is unmeasured.

## Candidate next steps (to be decided with the user)

- **S1-b closed loop** on fresh roots (P0 vs PROG8 with S1-b vs PHYS8). This measures what a well-trained short-horizon
  progress reader buys in closed loop, and gives the upper value of the "relax labels" route.
- **A better label-free target:**
  - common random numbers (the same continuation seed for all siblings);
  - a nearer goal offset (+8 steps);
  - a contrastive loss with a temperature and an auxiliary term, or a pixel/higher-resolution input.
