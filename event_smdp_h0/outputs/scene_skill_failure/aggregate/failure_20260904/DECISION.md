# Scene skill-failure dose response

Verdict: **MECHANISM_CONFIRMED**

64 task-5 resets.  Reproduction of the p=0.00 column against the ablation run: 896 rows, 0 mismatches.

| State source | p=0.00 | p=0.10 | p=0.20 | p=0.30 | degradation (0.00 - 0.30) |
|---|---:|---:|---:|---:|---|
| `oracle_event` | 76.56% | 75.00% | 76.56% | 75.00% | +1.56 [-10.94, +15.62] |
| `openloop_transition` | 62.50% | 28.12% | 10.94% | 3.12% | +59.38 [+46.88, +71.88] |
| `frame_full` | 54.17% | 57.81% | 61.98% | 54.69% | -0.52 [-12.50, +11.98] |
| `action_only_full` | 63.02% | 29.69% | 14.06% | 3.12% | +59.90 [+47.92, +71.35] |
| `obs_history_full` | 89.58% | 89.58% | 88.02% | 78.65% | +10.94 [+2.60, +19.79] |
| `history_full` | 83.33% | 68.75% | 61.46% | 37.50% | +45.83 [+34.38, +57.29] |

| Contrast | points | 95% CI |
|---|---:|---|
| `EXTRA_DEGRADATION_ACTION_ONLY_FULL` | +48.96 | [+34.38, +63.02] |
| `EXTRA_DEGRADATION_OPENLOOP_TRANSITION` | +48.44 | [+33.33, +63.02] |

| State source | p | exact-q | over-reads | under-reads | timeout |
|---|---:|---:|---:|---:|---:|
| `oracle_event` | 0.00 | n/a | 0.00 | 0.00 | 23.4% |
| `oracle_event` | 0.10 | n/a | 0.00 | 0.00 | 25.0% |
| `oracle_event` | 0.20 | n/a | 0.00 | 0.00 | 23.4% |
| `oracle_event` | 0.30 | n/a | 0.00 | 0.00 | 25.0% |
| `openloop_transition` | 0.00 | 83.8% | 1.62 | 0.00 | 37.5% |
| `openloop_transition` | 0.10 | 52.0% | 5.77 | 0.00 | 71.9% |
| `openloop_transition` | 0.20 | 35.0% | 8.45 | 0.00 | 89.1% |
| `openloop_transition` | 0.30 | 22.3% | 11.66 | 0.00 | 96.9% |
| `frame_full` | 0.00 | 74.8% | 0.98 | 1.54 | 45.8% |
| `frame_full` | 0.10 | 74.4% | 1.26 | 1.80 | 42.2% |
| `frame_full` | 0.20 | 76.7% | 1.20 | 1.81 | 38.0% |
| `frame_full` | 0.30 | 76.7% | 1.36 | 2.13 | 45.3% |
| `action_only_full` | 0.00 | 79.6% | 1.89 | 0.16 | 37.0% |
| `action_only_full` | 0.10 | 49.5% | 5.74 | 0.30 | 70.3% |
| `action_only_full` | 0.20 | 34.1% | 8.32 | 0.23 | 85.9% |
| `action_only_full` | 0.30 | 22.0% | 11.66 | 0.04 | 96.9% |
| `obs_history_full` | 0.00 | 91.8% | 0.01 | 0.77 | 10.4% |
| `obs_history_full` | 0.10 | 91.6% | 0.03 | 0.85 | 10.4% |
| `obs_history_full` | 0.20 | 90.8% | 0.06 | 1.03 | 12.0% |
| `obs_history_full` | 0.30 | 91.2% | 0.20 | 1.03 | 21.4% |
| `history_full` | 0.00 | 93.2% | 0.01 | 0.64 | 16.7% |
| `history_full` | 0.10 | 83.7% | 0.78 | 1.11 | 31.2% |
| `history_full` | 0.20 | 80.8% | 1.26 | 1.18 | 38.5% |
| `history_full` | 0.30 | 65.9% | 3.54 | 1.51 | 62.5% |
