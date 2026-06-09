# Planning Action-Error vs CRA_eff correlation

## Per-transition correlation (headline)

| subset | n | Pearson r | Spearman r | perm p (Spearman) |
| --- | --- | --- | --- | --- |
| all | 200 | 0.073 | 0.011 | 0.8778 |
| H=1 | 98 | -0.045 | -0.014 | 0.8932 |
| H=3 | 102 | 0.137 | 0.020 | 0.8300 |

Expected sign: **negative** (higher Action Error ↔ lower CRA_eff).
Observed CRA_eff positives: **7/200 (3.8%)**; this severe class imbalance limits correlation power.

## Run configuration

- Maximum transitions per regime/horizon: 40
- CEM: 64 samples, 15 iterations, 10 elites

## Per-regime means

| horizon | regime | n | Action Error | Action Score | CRA_eff (plan probe) |
| --- | --- | --- | --- | --- | --- |
| 1 | contact_manipulation | 40 | 1.1729 | 0.605 | 0.000 |
| 1 | gripper_actuation | 18 | 1.0779 | 0.637 | 0.000 |
| 1 | pre_grasp | 40 | 1.0787 | 0.637 | 0.087 |
| 3 | contact_manipulation | 40 | 1.6650 | 0.440 | 0.075 |
| 3 | gripper_actuation | 22 | 2.2575 | 0.241 | 0.045 |
| 3 | pre_grasp | 40 | 1.6276 | 0.453 | 0.000 |

- Regime-level (H=1, 3 regimes) Spearman(Action Error, CRA_eff) = 0.000
- Regime-level (H=3, 3 regimes) Spearman(Action Error, CRA_eff) = 0.500

## Cross-check: main-diagnostic hard_nn CRA_eff per regime

| regime | CRA_eff (05) |
| --- | --- |
| contact_manipulation | 0.059 |
| free_space | 0.000 |
| gripper_actuation | 0.035 |
| pre_grasp | 0.072 |

![Figure C](figures/figure_c_planning_vs_cra_safe.pdf)
