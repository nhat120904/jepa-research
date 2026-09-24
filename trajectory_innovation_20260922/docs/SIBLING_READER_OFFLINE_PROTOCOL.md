# Week-1 gate S1: can a reader rank siblings from the same state? (offline, pre-registered)

Pinned 2026-09-23, after `GATE_C2_CONFIRM_RESULT_54269.md` and before any S1 data. Budget: the
user approved 60 GPU-h for the CVPR plan on 2026-09-23. S1 is expected to use ≈ 4 GPU-h.

## Diagnosis being tested

The C2 reader was accurate across states (held-out Spearman 0.85) but blind within a bank.
Its within-bank agreement with the 8-step coverage oracle was 0.04, and closed-loop gains did not replicate.
Three causes are hypothesised, and S1 changes all three at once:

1. **Input:** it saw only the end frame, 2×2-pooled to 8×8 tokens. The small block-pose
   differences between siblings are exactly what pooling removes.
   → S1 readers see the full 16×16 grid, and the candidate's end tokens *and* their difference from the decision-time tokens.
2. **Data:** it never saw two outcomes from the same state.
   → S1 trains on **siblings**: all 8 candidates branched from each decision state.
3. **Target:** long-horizon success barely depends on one decision (split H_rep +2 pp), so it is a
   near-noise label for ranking one chunk.
   → S1 targets short-horizon differences.

S1 does not attribute the effect to one cause. It only asks whether the three together recover within-bank resolution.

## Data (new roots only)

- **Training:** roots 30000–30249 (250 episodes). **Held-out:** roots 1500–1599 (100 episodes). Neither has been used before.
- The P0 trajectory is followed (candidate 0). At every decision, all K = 8 candidates are drawn (per-candidate seeded
  generators, `candidate_seed`) and each is executed for its 8-step chunk on a `deepcopy_fixed` clone.
- Each sibling is then continued by the policy for 2 more chunks (16 steps), with seeds from a new namespace `ti-v1-sib`.
- **Stored per decision:**
  - decision-time frame and agent positions;
  - per sibling: end frames (current and previous), agent positions, coverage at the end (cov8), frame and coverage after the continuation (goal16, cov24), and physical state at the end.

## Features

- Frozen DINOv2 ViT-S/14 `x_norm_patchtokens`, 16×16 grid.
- Channels are reduced 384 → 128 by one PCA basis, fitted on training decision-time frames only (unsupervised) and then frozen.

## Readers (same architecture, two training signals)

**Architecture:**
- Tokens: sibling end (256), end − decision (256), goal (256), plus one proprio token (end and previous agent position).
- Width 256, 4 layers, 4 heads, CLS, scalar head.

**S1-a, label-free (sibling contrast):**
- For a decision, pick a sibling k at random; the goal is its goal16 frame.
- Score all 8 siblings against that goal; the loss is cross-entropy with target k.
- Siblings whose end physical state equals k's (max abs diff < 1e-3) share the target mass.
- No coverage, reward or success enters training.

**S1-b, privileged at training time:**
- Pairwise logistic ranking loss on cov8 over sibling pairs with a cov8 difference > 1e-3.
- The goal input is a random one of the 16 dev goal frames.
- Coverage is used only as a training label; at test time the reader sees only images and positions.

**Optimisation, both readers:** AdamW, lr 3e-4, wd 0.05, 16 decisions (128 siblings) per step, 10,000 steps, seed 0. No tuning on held-out roots.

## Test-time score and reference scorers

- Sibling score = mean over the 16 dev goal frames (`gate_smoke_53803/goal_frames.npz`) of the reader output.
- For S1-a this is the logit; the contrast is over siblings, so the logit is used directly.
- Reference scorers, computed on the same held-out siblings:
  - C2 reader r0 (`c2_train_53938/reader.pt`);
  - gate-C L2 score (`VisualScorer` with `goal_map.pt`).

## Metrics (held-out, P0 decisions)

- **Primary:** mean within-bank Spearman(score, cov8), over decisions where cov8 varies (ptp > 0).
- **Primary:** offline retained gap = Σ(cov8[argmax score] − cov8[0]) / Σ(max cov8 − cov8[0]), with the same tie rule (candidate 0 wins ties).
- Both are reported with root-bootstrap 95% CIs (10,000).
- **Secondary:** Spearman with cov24; S1-a held-out contrastive top-1 accuracy; the number of informative decisions.

## Pre-registered rules

1. A reader **ADVANCES** to the week-2 closed-loop test iff within-bank Spearman(cov8) ≥ 0.30 **and** retained gap ≥ 0.40.
2. If both advance: choose S1-a (label-free) if its retained gap ≥ 0.8 × S1-b's; otherwise choose S1-b.
3. If neither advances, **week-1 kill**: CVPR is off (agreed with the user 2026-09-23). Results are
   reported as they are; readers, features and thresholds are not retuned on roots 1500–1599.
4. Closed loop is not run in S1. Week 2 needs its own protocol on fresh roots.

## Compute

- **Collection:** 7 array tasks × 50 episodes (5 training, 2 held-out), 1 GPU each, ≤ 1 h limit. Frames are stored as compressed uint8.
- **Train + evaluate:** 1 GPU, ≤ 2 h limit.
- Estimated ≈ 4 GPU-h.

## Amendment 1 (2026-09-23, before any S1 training output existed)

`RESEARCH_DESIGN.md` §5 forbids simulator contact/pose/coverage labels in WM, codec and
query-reader training. S1-b is trained on cov8, so it cannot be the method's reader.

- S1-b is kept as a **privileged diagnostic only**. It answers "is within-bank ranking learnable
  from these image features at all?". It is never selected to advance.
- Rule 2 is replaced as follows. Only S1-a can ADVANCE, under the unchanged thresholds of rule 1. If S1-a does not
  advance, it is a week-1 kill regardless of S1-b. S1-b's numbers are reported to locate the failure:
  - S1-b advancing while S1-a does not means the features can resolve siblings, but the label-free signal does not find it;
  - neither advancing means the image features themselves do not resolve siblings.
- The training code is unchanged. Only the verdict reading changes. The `verdict` field written by
  `s1_train_eval.py` is to be read with this amendment.

## Amendment 2 (2026-09-23, user decision, before any S1 training output existed)

- The ADVANCE thresholds (Spearman ≥ 0.30, retained gap ≥ 0.40) and the week-1 kill are **withdrawn**.
  They were judgment calls that the user had not reviewed.
- S1 reports numbers only: within-bank Spearman, retained gap, CIs, the S1-b diagnostic, and the reference scorers. What happens
  next is decided together with the user after reading them. The method is not abandoned on an offline gate
  before it has been implemented and tested end to end.
- The `verdict` field that `s1_train_eval.py` writes is ignored.
