# Scene event-observer coverage/history factorial

Verdict: **HISTORY_REQUIRED_CONFIRMED**

128 paired fresh resets at K=112.

| Arm | mean success | 95% CI | seed 0 | seed 1 | seed 2 |
|---|---:|---|---:|---:|---:|
| `oracle_event` | 87.50% | [81.25, 92.97] | n/a | n/a | n/a |
| `abstract_terminal` | 2.34% | [0.00, 5.47] | n/a | n/a | n/a |
| `frame_canonical` | 74.22% | [66.67, 80.99] | 79.69% | 70.31% | 72.66% |
| `frame_full` | 69.01% | [61.20, 76.04] | 67.19% | 71.88% | 67.97% |
| `history_canonical` | 82.29% | [75.78, 88.02] | 84.38% | 78.91% | 83.59% |
| `history_full` | 90.62% | [85.68, 95.05] | 88.28% | 89.84% | 93.75% |

| Contrast | points | 95% CI |
|---|---:|---|
| `COVERAGE` = frame_full - frame_canonical | -5.21 | [-12.24, +1.82] |
| `HISTORY` = history_full - frame_full | +21.61 | [+15.36, +28.39] |
| `HISTORY_UNDER_CANONICAL` = history_canonical - frame_canonical | +8.07 | [+3.39, +13.28] |
| `COVERAGE_UNDER_HISTORY` = history_full - history_canonical | +8.33 | [+2.34, +14.58] |
| `FRAME_FULL_VS_TERMINAL` = frame_full - abstract_terminal | +66.67 | [+57.81, +74.74] |
| `HISTORY_FULL_VS_TERMINAL` = history_full - abstract_terminal | +88.28 | [+81.77, +93.75] |
| `FRAME_FULL_VS_ORACLE` = frame_full - oracle_event | -18.49 | [-25.00, -12.50] |
| `HISTORY_FULL_VS_ORACLE` = history_full - oracle_event | +3.12 | [-1.82, +8.33] |
