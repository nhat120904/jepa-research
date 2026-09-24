# Gate D: does a compact conditional code of the actual future keep the reader's ranking? (pre-registered)

Pinned 2026-09-24, while W2 (job 54455) is running and before any gate-D data exists.
**Gate D runs only if W2 PASSES**, or if the user explicitly decides otherwise. If the reader does not help
closed loop on actual futures, compressing its input is moot.

## Question (step 2 of 3)

Step 1 (S1, W2) scores candidates with a reader that sees the candidate's full actual end-frame tokens. D asks:

> If the actual future is first squeezed into a small discrete code S, conditioned on the context C, how much of
> that reader's within-bank ranking survives?

Three secondary questions come from the design and from the user's concerns:

- Does conditioning on C help at a matched bit budget?
- Does adding the path (intermediate frames) help beyond the endpoint?
- Does a learned code beat simply pooling the end frame down to the same number of bits?

Still no world model: S is encoded from the actual future. Gate E will predict S from (C, A).

## Data (new roots, collected by `scripts/d_collect.py`)

- **Training:** roots 30250–30649 (400 episodes). **Held-out:** roots 2000–2099 (100 episodes). None have been used before.
- **Same runtime as S1:** the P0 trajectory is followed, and at every decision all 8 seeded candidates are executed for their 8-step chunk on
  `deepcopy_fixed` clones. No continuations.
- **Stored per decision:**
  - decision-time frame (and the previous frame), agent positions;
  - the 8 action chunks (reused for gate E);
  - per sibling: frames after steps 2, 4 and 6, end frame, end agent positions, cov8, physical end state.

## Features

- Frozen DINOv2 ViT-S/14, 16×16 patch tokens, PCA 384 → 128. The basis is fitted on training decision-time frames only.
- Intermediate frames are 2×2-pooled to 8×8 (64 tokens).

## Arms (all trained on the same data, same steps, same seed)

| Arm | S given to the reader | Bits per candidate |
|---|---|---|
| FULL | end-frame tokens, uncompressed (256 × 128 fp16) | reference (≈ 524k) |
| COND16 | 16 FSQ tokens; encoder sees end + C | 16 × 15.97 ≈ 256 |
| COND4 | 4 FSQ tokens; encoder sees end + C | ≈ 64 |
| UNCOND16 | 16 FSQ tokens; encoder sees the end frame only | ≈ 256 |
| TRAJ16 | 16 FSQ tokens; encoder sees end + C + frames at steps 2, 4, 6 | ≈ 256 |
| POOL16 | end tokens average-pooled to 4×4, linear to 6 dims, FSQ (no attention, no C) | ≈ 256 |

**Encoder:** 2-layer cross-attention from M learned queries. FSQ levels (8, 8, 8, 5, 5, 5).

**Reader D(C, S, q):**
- Tokens: C (256), S (M, or 256 for FULL), goal (256), and an end-proprio token.
- 4 layers, width 256.
- The reader always sees C, in every arm.

**Losses:**
- **Reader:** pairwise logistic ranking on cov8 over sibling pairs, exactly as S1-b. Task-reward label, allowed for the reader under `RESEARCH_DESIGN.md` §5a. Goal input = a random dev goal frame.
- **Code arms, added distortion term:** normalised MSE of a 2-layer decoder that predicts the end-frame tokens from (C, S), weight λ = 1.
  It stops S from collapsing to a rank-only bottleneck, the user's "handcrafted-query bottleneck" concern. It is reported as R².

**Optimisation:** AdamW, lr 3e-4, wd 0.05, 16 decisions (128 siblings) per step, 10,000 steps, seed 0,
encoder, reader and decoder trained jointly. No tuning on held-out roots.

## Test-time score and metrics (held-out roots 2000–2099)

- **Score:** mean over the 16 dev goal frames of D(C, S_k, goal); candidate 0 wins ties.
- **Per arm:**
  - within-bank Spearman(score, cov8);
  - offline retained gap vs cov8 (as S1);
  - decoder R² (code arms).
- **Retention vs FULL** = Σ(gain of arm's choice) / Σ(gain of FULL's choice) over held-out decisions, with root-bootstrap CI (10,000).
- **Paired gap differences** (root bootstrap): COND16 − UNCOND16, TRAJ16 − COND16, COND16 − POOL16.

## Pre-registered rules

0. **Reference check:** FULL's retained gap must be ≥ 0.40 (S1-b had 0.55). Otherwise D is NOT INTERPRETED:
   the reader itself did not reproduce on the new data.
1. **D PASS** iff COND16 retention vs FULL ≥ 0.80 **and** its CI lower bound ≥ 0.50.
2. **Secondary claims.** Each is stated only if its paired CI lower bound is > 0:
   - COND16 > UNCOND16: "conditioning on C helps at matched bits";
   - TRAJ16 > COND16: "the path adds information beyond the endpoint". Otherwise: "endpoint suffices on PushT";
   - COND16 > POOL16: "the learned code beats per-frame pooling at matched bits".
3. COND4 is reported as a point on the rate curve; there is no rule.
4. **If D fails:** the compact code loses the ranking. No retuning of M, levels, λ or architecture on roots 2000–2099.

## Compute

- **Collection:** 10 shards × 50 roots (8 training, 2 held-out), 1 × MIG 3g.40gb each, ≈ 6 min per shard
  (no continuations), plus a 2-root smoke.
- **Training and evaluation:** 6 models × ≈ 10 min on one MIG slice, ≤ 2 h 30 min limit.
- Estimated ≈ 3 GPU-h (MIG).
