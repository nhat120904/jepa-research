# Week 2 (W2): closed-loop selection with the frozen S1-b reader on PushT (pre-registered)

Pinned 2026-09-23, after `SIBLING_READER_RESULT_54419.md` and the user's decision to relax the
label rule (`RESEARCH_DESIGN.md` §5a). No W2 data exists yet.

## Question

Does selecting among the policy's own 8 candidates with a reward-trained, context-conditioned
reader (S1-b), scoring the **actual** 8-step futures, raise closed-loop success over the policy's default
draw? No world model is involved. W2 is the upper bound that any predicted code must later approach.

## Fixed inputs (nothing retrained)

- Reader: `s1_train_54419/reader_s1b.pt`, together with its PCA basis. It is frozen and its sha256 is recorded per root.
- **Score of candidate k:**
  - C = decision-time frame; end = candidate k's frame after its 8-step chunk; proprio = its last two agent positions.
  - Score = mean over the 16 dev goal frames of the reader output, exactly as in S1 evaluation.
  - Ties go to candidate 0.
- Runtime identical to C2-confirm:
  - `deepcopy_fixed` clones;
  - each arm draws only the candidates it needs from the nested seeded bank (P0: candidate 0; PHYS8 and RANK8: candidates 0–7);
  - the selected branch is adopted.

## Roots

Fresh roots **1600–1999** (n = 400); none have been used. The size is chosen for power. The earlier contrasts were
+4.5 pp (confirm) and +14 pp (C2). At n = 400 the paired 95% CI half-width is about ±6 pp,
while at n = 200 it was about ±8.5 pp.

## Arms

| Arm | Selection |
|---|---|
| P0 | Candidate 0 (policy default; also the distribution of a random pick) |
| PHYS8 | Best 8-step coverage (privileged oracle, the reference headroom) |
| RANK8 | Best S1-b reader score on the actual future |

## Pre-registered rules

1. **Primary:** RANK8 − P0, root-paired bootstrap (10,000). **PASS** iff the 95% CI lower bound > 0.
2. **Strength (reported, not a gate):**
   - retention = (RANK8 − P0) / (PHYS8 − P0), with root-clustered CI;
   - **STRONG** if retention ≥ 0.80 and PHYS8 − P0 has CI lower bound > 0.
3. Also reported:
   - McNemar for RANK8 vs P0 and PHYS8 vs P0;
   - within-bank Spearman(reader, cov8) on RANK8 decisions;
   - agreement rate of RANK8's choice with PHYS8's choice at the same decisions.
4. **Bar change, stated openly.** Earlier the plan said week 2 needs ≥ 80% retention. That was set before
   S1 measured the offline ceiling (0.55), and C2 showed retention is uninterpretable when the oracle's own gap is noisy.
   The primary is therefore a CI-clean gain over the policy, which is the claim a paper needs. Retention is
   reported as strength. This change is made before any W2 data exists.
5. **If PASS fails:** the reward-trained reader is not qualified closed-loop on PushT. No retuning on roots 1600–1999.
   The CVPR decision is revisited with the user.

## Compute

- 8 shards × 50 roots, 1× MIG 3g.40gb each, ≤ 1 h 15 min limit per shard.
- Expected ≈ 45–50 s per root for 3 arms, so ≈ 5.5 GPU-h (MIG slices).
- Wall time depends on the per-user cap of 2 GPUs, which is shared with the LIBERO jobs.
- Aggregation: CPU.
