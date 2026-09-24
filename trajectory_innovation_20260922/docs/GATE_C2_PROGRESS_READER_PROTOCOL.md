# Gate C2: self-supervised progress reader on actual futures (pre-registered)

Pinned 2026-09-23, after gate C failed (`GATE_ABC_RESULT_53806.md`) and before any C2 data
was collected. C2 replaces gate C's fixed scorer (whole-image DINOv2 L2 to a goal map) with the scorer the method actually proposes: a **learned, goal-conditioned progress reader**.
Nothing is predicted by a world model here. The reader sees ACTUAL future frames. If it cannot rank
candidates from real futures, a world model that predicts its input cannot either.

## Question

Can a reader trained without success labels, only by hindsight relabeling, estimate "how far this state is from a goal image"? And does selecting the
policy's candidates by that estimate take most of the physical-oracle headroom?

## Reader

- **Input:**
  - the last two frames of a candidate's executed 8-step segment, plus agent proprio;
  - one goal image.
  - All images go through frozen DINOv2 ViT-S/14 (`x_norm_patchtokens`, 96→224 resize, ImageNet normalization), and the 16×16 patch grid is average-pooled to 8×8 (64 tokens per image).
- **Architecture (fixed):**
  - token projection to width 256, learned type/position embeddings, CLS token;
  - 4 Transformer layers, 4 heads;
  - MLP head that outputs one scalar, the predicted log(1 + steps to goal).
- **One reader for every goal.** The goal is an input, not a separate head or model. A new task means a new goal image and the same reader.

## Self-supervised training data and target

- **Data:** P0 policy rollouts on fresh training roots 20000–20799 (800 episodes) and
  validation roots 20800–20999 (200 episodes). Episodes are split by root before any
  sampling. Candidate-0 draws come from the pinned sampler. No branches are needed: bank
  candidates are i.i.d. policy draws, so on-policy segments match the test distribution of
  single candidates.
- **Frames stored:** only those at decision boundaries (t ≡ 0 or 7 mod 8, plus the terminal frame).
- **Training pair:**
  - segment end at a decision time u of an episode;
  - goal = a stored frame at time g ≥ u of the SAME episode, with g drawn uniformly (terminal frame included);
  - target = log(1 + (g − u)).
- **No labels from the simulator:** no success flag, coverage or reward enters training.
- **Optimisation:** AdamW, lr 3e-4, weight decay 0.05, batch 256, 20,000 steps, one seed.
  Hyperparameters are not tuned on qualification roots.
- **Held-out check (reported, not a gate):** on validation episodes, report the Spearman correlation between
  predicted and true log-distance, and the same statistic for a goal-only baseline (reader with the segment frames blanked).

## Test-time score

- The goal images are the same 16 dev success frames used for gate C (`gate_smoke_53803/goal_frames.npz`).
- Score of a candidate = −mean over those 16 goals of the predicted log-distance, computed from that candidate's actual executed segment.
- Tie rule as before: candidate 0 wins any tie it is part of.

## Closed-loop arms on NEW qualification roots 1200–1299

The same runtime is used as in gate A–C: sampler, `deepcopy_fixed` clones, adopt-the-branch, K=8 from
the nested 32-bank. Seeds 1000–1099 are spent. Seeds 1100–1199 stay reserved as B-extension roots.

| Arm | Selection |
|---|---|
| P0 | Candidate 0 |
| PHYS8 | Best 8-step coverage (physical oracle) |
| PROG8 | Best progress-reader score on the actual future |

## Pre-registered rules

1. **B replication:** PHYS8 − P0 ≥ 10 pp with paired CI lower bound > 0 on these roots. If this
   fails, C2 is not interpreted (report "headroom did not replicate").
2. **C2 PASS:** retention = (PROG8 − P0) / (PHYS8 − P0) ≥ 0.80.
   - Also reported: raw paired differences, the root-clustered bootstrap CI for retention, and exact McNemar.
3. **Secondary metrics (they cannot rescue a failed primary):**
   - offline retained gap on P0 decisions;
   - within-bank Spearman(progress, 8-step coverage).

   Note that the reader targets long-horizon progress while the oracle is myopic, so disagreement between them is not automatically a reader error.
4. **Reading PROG8 against PHYS8:** PROG8 may exceed PHYS8. If so, report it; do not treat it as a violation.
5. **If C2 fails:** do not retune the architecture, pooling, target or goal set on these roots.
   Any new reader needs a new protocol and fresh roots.

## Diagnostic for gate C's failure (not a gate)

On P0 decisions of the C2 run, record each candidate's agent displacement and block
pose change. Report the correlation of (a) the gate-C L2 score and (b) the progress score with
each of them. This tests the hypothesis that the agent's position dominated the L2 score.

## Compute (bounded)

- Collection: 1 GPU, batched rollouts, ≤ 45 min.
- Encoding and training: 1 GPU, ≤ 60 min.
- Closed loop: 2 shards × 50 roots, ≤ 2 h each.
- Estimated total ≈ 4 GPU-h. Together with the ≈ 3.5 GPU-h already used, this reaches the programme's 8 GPU-h
  cap, so it needs explicit approval before submission.
