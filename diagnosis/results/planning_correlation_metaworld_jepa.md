# Planning Action-Error vs CRA_eff correlation

## Per-transition correlation (headline)

| subset | n | Pearson r | Spearman r | perm p (Spearman) |
| --- | --- | --- | --- | --- |
| all | 66 | -0.091 | -0.103 | 0.4197 |
| H=1 | 31 | -0.124 | -0.211 | 0.2655 |
| H=3 | 35 | -0.041 | -0.069 | 0.7147 |

Expected sign: **negative** (higher Action Error ↔ lower CRA_eff).
Observed CRA_eff positives: **40/66 (60.6%)**; this severe class imbalance limits correlation power.

## Run configuration

- Maximum transitions per regime/horizon: 20
- CEM: 64 samples, 15 iterations, 10 elites

## Per-regime means

| horizon | regime | n | Action Error | Action Score | CRA_eff (plan probe) |
| --- | --- | --- | --- | --- | --- |
| 1 | contact_manipulation | 14 | 8.9551 | 0.653 | 0.714 |
| 1 | free_space | 13 | 9.0734 | 0.648 | 0.615 |
| 1 | pre_grasp | 4 | 10.2459 | 0.602 | 0.500 |
| 3 | contact_manipulation | 17 | 19.4489 | 0.245 | 0.588 |
| 3 | free_space | 13 | 16.0081 | 0.379 | 0.462 |
| 3 | pre_grasp | 5 | 19.5092 | 0.243 | 0.800 |

- Regime-level (H=1, 3 regimes) Spearman(Action Error, CRA_eff) = -1.000
- Regime-level (H=3, 3 regimes) Spearman(Action Error, CRA_eff) = 1.000

## Cross-check: main-diagnostic hard_nn CRA_eff per regime

| regime | CRA_eff (05) |
| --- | --- |
| contact_manipulation | 0.530 |
| free_space | 0.601 |
| pre_grasp | 0.491 |

![Figure C](figures/figure_c_planning_vs_cra_metaworld_jepa.pdf)
