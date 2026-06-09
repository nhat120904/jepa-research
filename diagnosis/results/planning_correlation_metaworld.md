# Planning Action-Error vs CRA_eff correlation

## Per-transition correlation (headline)

| subset | n | Pearson r | Spearman r | perm p (Spearman) |
| --- | --- | --- | --- | --- |
| all | 117 | 0.037 | 0.050 | 0.6057 |
| H=1 | 59 | 0.229 | 0.204 | 0.1270 |
| H=3 | 58 | -0.168 | -0.168 | 0.2082 |

Expected sign: **negative** (higher Action Error ↔ lower CRA_eff).
Observed CRA_eff positives: **68/117 (58.1%)**; this severe class imbalance limits correlation power.

## Run configuration

- Maximum transitions per regime/horizon: 40
- CEM: 64 samples, 15 iterations, 10 elites

## Per-regime means

| horizon | regime | n | Action Error | Action Score | CRA_eff (plan probe) |
| --- | --- | --- | --- | --- | --- |
| 1 | contact_manipulation | 30 | 9.6080 | 0.673 | 0.433 |
| 1 | free_space | 22 | 8.8835 | 0.698 | 0.636 |
| 1 | pre_grasp | 7 | 6.8804 | 0.766 | 0.714 |
| 3 | contact_manipulation | 32 | 22.0062 | 0.251 | 0.531 |
| 3 | free_space | 24 | 18.8230 | 0.359 | 0.708 |
| 3 | pre_grasp | 2 | 13.4051 | 0.544 | 1.000 |

- Regime-level (H=1, 3 regimes) Spearman(Action Error, CRA_eff) = -1.000
- Regime-level (H=3, 3 regimes) Spearman(Action Error, CRA_eff) = -1.000

## Cross-check: main-diagnostic hard_nn CRA_eff per regime

| regime | CRA_eff (05) |
| --- | --- |
| contact_manipulation | 0.530 |
| free_space | 0.601 |
| pre_grasp | 0.491 |

![Figure C](figures/figure_c_planning_vs_cra_metaworld.pdf)
