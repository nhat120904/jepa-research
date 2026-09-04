# Scene event-observer coverage/history factorial

Verdict: **HISTORY_REQUIRED_CONFIRMED**

32 paired fresh resets at K=112.

| Arm | mean success | 95% CI | seed 0 | seed 1 | seed 2 |
|---|---:|---|---:|---:|---:|
| `oracle_event` | 87.50% | [75.00, 96.88] | n/a | n/a | n/a |
| `abstract_terminal` | 0.00% | [0.00, 0.00] | n/a | n/a | n/a |
| `frame_canonical` | 75.00% | [61.46, 87.50] | 84.38% | 65.62% | 75.00% |
| `frame_full` | 68.75% | [53.12, 83.33] | 62.50% | 75.00% | 68.75% |
| `history_canonical` | 87.50% | [76.04, 96.88] | 87.50% | 84.38% | 90.62% |
| `history_full` | 87.50% | [76.04, 96.88] | 87.50% | 87.50% | 87.50% |

| Contrast | points | 95% CI |
|---|---:|---|
| `COVERAGE` = frame_full - frame_canonical | -6.25 | [-20.83, +8.33] |
| `HISTORY` = history_full - frame_full | +18.75 | [+8.33, +31.25] |
| `HISTORY_UNDER_CANONICAL` = history_canonical - frame_canonical | +12.50 | [+3.12, +23.96] |
| `COVERAGE_UNDER_HISTORY` = history_full - history_canonical | -0.00 | [-11.46, +10.42] |
| `FRAME_FULL_VS_TERMINAL` = frame_full - abstract_terminal | +68.75 | [+53.12, +83.33] |
| `HISTORY_FULL_VS_TERMINAL` = history_full - abstract_terminal | +87.50 | [+76.04, +96.88] |
| `FRAME_FULL_VS_ORACLE` = frame_full - oracle_event | -18.75 | [-31.25, -7.29] |
| `HISTORY_FULL_VS_ORACLE` = history_full - oracle_event | -0.00 | [-9.38, +9.38] |
