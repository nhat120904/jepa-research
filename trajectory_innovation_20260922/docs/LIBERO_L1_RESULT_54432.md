# LIBERO L1 result (job 54432)

Unmodified `lerobot-eval` (lerobot 0.6.1, hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.3.7, CPU OSMesa rendering).
Policy: `HuggingFaceVLA/smolvla_libero` at 6721902, `n_action_steps=10`. Suite: libero_goal, 10 tasks × init states 0–9 = 100 roots,
policy seeds 0, 1 and 2 (300 episodes). Six array tasks (seed × task half), 47–57 min each.

## Reproduction

| Seed | Success |
|---|---|
| 0 | 74% (Wilson 95% CI ≈ [65, 82]) |
| 1 | 76% |
| 2 | 64% |
| Mean | **71.3%** |

- The published figure (SmolVLA paper, `n_action_steps=10`) is 91%. The run is **17 pp below it, so it does not reproduce.**
- The documented config items from lerobot#4614 are all already satisfied:
  - camera names match the checkpoint (`image`, `image2`), so no rename is needed;
  - fps is 20;
  - MuJoCo is 3.3.7, below 3.8.1;
  - `n_action_steps` is 10.
- The gap is in line with other public reports: lerobot#2354 and #3264, and 77–84% on other suites in #4614.
- Operating point for all later LIBERO gates: **≈ 71% P0 success**. It is stated as measured, not tuned.

Per task (mean of 3 seeds, %):

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Success | 70 | 93 | 80 | 37 | 90 | 77 | 63 | 97 | 70 | 37 |

Tasks 3 and 9 are the weakest: task 3 is "open the top drawer and put the bowl inside", task 9 is "put the wine bottle on the rack".

## Headroom shape (reported, not a gate)

| Root type | Count (of 100) |
|---|---|
| Succeed on all 3 seeds | 45 |
| Fail on all 3 seeds (systematic) | 5 |
| Mixed (stochastic) | 50 |

- There are 86 failures in total, and **71 of them (83%) are on stochastic roots**. The same start state succeeds under other policy noise.
- Only stochastic failures can be fixed by choosing among the policy's own samples. This is the precondition for selection headroom.
  It is not yet a measurement of that headroom: L3 measures it with the every-decision oracle.

## Next

- **L2:** clone fidelity for robosuite/LIBERO state save and restore, plus candidate diversity. The branching harness is still to be written.
- **L3:** closed-loop ORACLE8 vs P0, with privileged BDDL goal progress (Amendment 2).
