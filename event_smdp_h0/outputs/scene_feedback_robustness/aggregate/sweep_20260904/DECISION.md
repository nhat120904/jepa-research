# Scene feedback-robustness sweep

Verdict: **PARTIAL**

64 task-5 resets.  Anchor reproduction against the 3x2 grid: 896 rows, 0 mismatches.

| Feedback | oracle | `obs_history_full` | `frame_full` | gap (obs - oracle) | preserves |
|---|---:|---:|---:|---|:--:|
| `branch_w030` | 93.75% | 92.71% | 82.29% | -1.04 [-3.12, +0.00] | yes |
| `branch_w040` | 95.31% | 92.71% | 81.77% | -2.60 [-6.77, +0.00] | no |
| `branch_w050` | 76.56% | 89.58% | 54.17% | +13.02 [+4.17, +22.40] | yes |
| `branch_w056` | 40.62% | 25.00% | 10.42% | -15.62 [-28.12, -3.12] | no |
| `branch_w062` | 75.00% | 2.08% | 10.94% | -72.92 [-83.33, -61.98] | no |
| `branch_w070` | 98.44% | 8.85% | 9.38% | -89.58 [-95.85, -81.77] | no |
| `anti_livelock` | 71.88% | 85.94% | 51.04% | +14.06 [+6.25, +23.44] | yes |
| `shaped_gamma09` | 92.19% | 92.19% | 74.48% | +0.00 [-3.65, +4.17] | yes |

| Feedback | source | timeout rate | repeated-skill rate | exact-q |
|---|---|---:|---:|---:|
| `branch_w030` | `oracle` | 6.2% | 27.6% | n/a |
| `branch_w030` | `obs_history_full` | 7.3% | 28.0% | 98.6% |
| `branch_w040` | `oracle` | 4.7% | 27.4% | n/a |
| `branch_w040` | `obs_history_full` | 7.3% | 28.0% | 98.6% |
| `branch_w050` | `oracle` | 23.4% | 30.2% | n/a |
| `branch_w050` | `obs_history_full` | 10.4% | 30.4% | 91.8% |
| `branch_w056` | `oracle` | 59.4% | 35.2% | n/a |
| `branch_w056` | `obs_history_full` | 75.0% | 51.1% | 54.7% |
| `branch_w062` | `oracle` | 25.0% | 29.4% | n/a |
| `branch_w062` | `obs_history_full` | 97.9% | 56.1% | 40.5% |
| `branch_w070` | `oracle` | 1.6% | 25.5% | n/a |
| `branch_w070` | `obs_history_full` | 91.1% | 48.2% | 53.4% |
| `anti_livelock` | `oracle` | 28.1% | 28.9% | n/a |
| `anti_livelock` | `obs_history_full` | 14.1% | 26.5% | 95.0% |
| `shaped_gamma09` | `oracle` | 7.8% | 28.0% | n/a |
| `shaped_gamma09` | `obs_history_full` | 7.8% | 28.6% | 96.8% |
