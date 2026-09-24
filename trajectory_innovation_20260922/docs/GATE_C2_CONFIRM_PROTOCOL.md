# Gate C2-confirm: does progress-reader selection beat the policy default? (pre-registered)

Pinned 2026-09-23, after `GATE_C2_RESULT_53992.md` and before any confirmation data.

## Why

In C2 (roots 1200–1299), PROG8 − P0 was +14 pp [+3, +25]. That contrast was **secondary**: the
pre-registered primary compared against the myopic physical oracle PHYS8, whose own gap did not
replicate CI-clean, so C2 was NOT INTERPRETED. This protocol makes PROG8 − P0 the **primary**
contrast, on fresh roots, with the already-trained reader frozen, plus a second reader seed.

## Fixed inputs

- Reader r0: `c2_train_53938/reader.pt`, unchanged (sha256 recorded per root).
- Reader r1: same code, data and hyperparameters, `--seed 1`, no goal-only control. The
  collection is seed-deterministic, so r1 differs from r0 only in initialisation and pair sampling.
  If r1's held-out Spearman is < 0.70, report "r1 training failed"; this does not affect the primary.
- Goal images, sampler, clone (`deepcopy_fixed`), K = 8, tie rule: as in C2.
- Runtime change (declared): each arm draws only the candidates it needs from the nested bank
  (P0: candidate 0; PROG8: candidates 0–7), with the same per-candidate seeds. The smoke test
  showed K8 vs K32 nested draws differ by ≤ 0.008 px. This is a cost change only.

## Roots and arms

Fresh roots **1300–1499** (200 episodes). None have been used before.

| Arm | Selection |
|---|---|
| P0 | Candidate 0 (the policy default; also the distribution of a random pick) |
| PROG8_r0 | Best score from reader r0 on the actual 8-step future |
| PROG8_r1 | Best score from reader r1 |

## Pre-registered rules

1. **Primary:** PROG8_r0 − P0, root-paired bootstrap (10,000). **CONFIRMED** iff the 95% CI lower
   bound > 0. Otherwise **NOT_CONFIRMED**.
2. **Seed robustness (secondary):** the same rule for PROG8_r1 − P0. Reported as ROBUST / NOT_ROBUST.
   It cannot rescue a NOT_CONFIRMED primary.
3. Also reported: exact McNemar for both contrasts; pooled C2 + confirm estimate for r0 (300 roots),
   labelled as pooled, not as a test.
4. If NOT_CONFIRMED: the +14 pp in C2 is treated as not established. No retuning of the reader,
   goals or K on these roots.

## Compute (bounded)

- r1 training: 1 GPU, ≤ 1 h 45 min limit (expected ≈ 30 min: collect + encode + one reader).
- Closed loop: 4 shards × 50 roots, 1 GPU each, ≤ 1 h limit each (expected ≈ 20 min each).
- Aggregation: CPU.
- Expected ≈ 1.8 GPU-h. With ≈ 5.6 GPU-h used, total stays under the 8 GPU-h cap.
- Eval is submitted with `afterok` on r1 training; aggregation with `afterok` on the eval array.
