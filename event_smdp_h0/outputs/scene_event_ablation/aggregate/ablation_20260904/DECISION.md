# Scene event-observer input ablation

Verdict: **DEAD_RECKONING_REFUTED**

128 paired resets at K=112.  Reproduction of `frame_full` and `history_full` against the confirmatory factorial: 768 rows compared, 0 mismatches.

| Arm | mean success | 95% CI | seed 0 | seed 1 | seed 2 | task 4 | task 5 |
|---|---:|---|---:|---:|---:|---:|---:|
| `oracle_event` | 87.50% | [81.25, 92.97] | n/a | n/a | n/a | 98.44% | 76.56% |
| `abstract_terminal` | 2.34% | [0.00, 5.47] | n/a | n/a | n/a | 0.00% | 4.69% |
| `openloop_transition` | 80.47% | [73.44, 86.72] | n/a | n/a | n/a | 98.44% | 62.50% |
| `frame_full` | 69.01% | [61.20, 76.04] | 67.19% | 71.88% | 67.97% | 83.85% | 54.17% |
| `obs_history_full` | 93.49% | [89.06, 97.14] | 94.53% | 92.97% | 92.97% | 97.40% | 89.58% |
| `action_only_full` | 80.73% | [73.95, 86.98] | 81.25% | 79.69% | 81.25% | 98.44% | 63.02% |
| `history_full` | 90.62% | [85.68, 95.05] | 88.28% | 89.84% | 93.75% | 97.92% | 83.33% |

| Contrast | points | 95% CI |
|---|---:|---|
| `VISION_GIVEN_ACTIONS` = history_full - action_only_full | +9.90 | [+4.69, +15.62] |
| `ACTIONS_GIVEN_VISION` = history_full - obs_history_full | -2.86 | [-6.25, +0.26] |
| `ACTION_ONLY_VS_OPENLOOP` = action_only_full - openloop_transition | +0.26 | [-1.30, +1.82] |
| `OBS_HISTORY_VS_FRAME` = obs_history_full - frame_full | +24.48 | [+17.45, +32.03] |
| `OPENLOOP_VS_HISTORY_FULL` = openloop_transition - history_full | -10.16 | [-15.89, -4.69] |
| `HISTORY_FULL_VS_FRAME_FULL` = history_full - frame_full | +21.61 | [+15.36, +28.39] |
| `OPENLOOP_VS_FRAME_FULL` = openloop_transition - frame_full | +11.46 | [+5.47, +17.71] |
