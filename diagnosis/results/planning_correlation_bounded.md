# Planning Action-Error vs CRA_eff correlation

## Per-transition correlation (headline)

| subset | n | Pearson r | Spearman r | perm p (Spearman) |
| --- | --- | --- | --- | --- |
| all | 155 | -0.084 | -0.068 | 0.4163 |
| H=1 | 78 | -0.023 | -0.003 | 0.9834 |
| H=3 | 77 | -0.031 | -0.061 | 0.6215 |

Expected sign: **negative** (higher Action Error ↔ lower CRA_eff).
Observed CRA_eff positives: **6/155 (4.2%)**; this severe class imbalance limits correlation power.

## Run configuration

- Maximum transitions per regime/horizon: 30
- CEM: 100 samples, 15 iterations, 10 elites

## Per-regime means

| horizon | regime | n | Action Error | Action Score | CRA_eff (plan probe) |
| --- | --- | --- | --- | --- | --- |
| 1 | contact_manipulation | 30 | 1.0963 | 0.582 | 0.100 |
| 1 | gripper_actuation | 18 | 1.1362 | 0.567 | 0.000 |
| 1 | pre_grasp | 30 | 1.0154 | 0.613 | 0.067 |
| 3 | contact_manipulation | 30 | 1.6393 | 0.375 | 0.017 |
| 3 | gripper_actuation | 17 | 1.8472 | 0.295 | 0.000 |
| 3 | pre_grasp | 30 | 1.8106 | 0.309 | 0.033 |

- Regime-level (H=1, 3 regimes) Spearman(Action Error, CRA_eff) = -0.500
- Regime-level (H=3, 3 regimes) Spearman(Action Error, CRA_eff) = -0.500

## Cross-check: main-diagnostic hard_nn CRA_eff per regime

| regime | CRA_eff (05) |
| --- | --- |
| contact_manipulation | 0.059 |
| free_space | 0.000 |
| gripper_actuation | 0.035 |
| pre_grasp | 0.072 |

![Figure C](figures/figure_c_planning_vs_cra_bounded.pdf)
