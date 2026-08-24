# HyS-JEPA rho audit — does straightening survive CEM search?

Populations grouped by (task, seed, replan); CIs are bootstrap clustered on episode seed. `rho` = Spearman(proxy cost, true shaped cost) over the full candidate population. **The gate is rho_final.**

| arm | phase | rho | 95% CI | recall@10% | R_sel (cm) | n pops |
|---|---|---|---|---|---|---|
| dino_pick_l2 | init | +0.022 | [-0.048, +0.105] | 0.072 | 0.01 | 112 |
| dino_pick_l2 | final | -0.094 | [-0.136, -0.051] **CI-clean neg** | 0.012 | 0.00 | 112 |
| dino_pick_straight_none | init | +0.051 | [-0.017, +0.119] | 0.078 | 0.11 | 112 |
| dino_pick_straight_none | final | -0.003 | [-0.046, +0.039] | 0.031 | 0.00 | 112 |
| dino_pick_straight_off | init | +0.002 | [-0.072, +0.076] | 0.081 | 0.08 | 112 |
| dino_pick_straight_off | final | +0.001 | [-0.045, +0.050] | 0.054 | 0.01 | 112 |
| dino_pick_straight_switch | init | +0.099 | [+0.010, +0.185] **CI-clean pos** | 0.087 | 0.30 | 112 |
| dino_pick_straight_switch | final | +0.010 | [-0.033, +0.059] | 0.041 | 0.09 | 112 |
| dino_push_l2 | init | +0.247 | [+0.163, +0.321] **CI-clean pos** | 0.155 | 0.11 | 112 |
| dino_push_l2 | final | -0.085 | [-0.135, -0.033] **CI-clean neg** | 0.022 | 0.05 | 112 |
| dino_push_straight_none | init | -0.077 | [-0.138, +0.000] | 0.047 | 0.03 | 112 |
| dino_push_straight_none | final | -0.011 | [-0.040, +0.018] | 0.027 | 0.02 | 112 |
| dino_push_straight_off | init | -0.035 | [-0.094, +0.028] | 0.084 | 0.05 | 112 |
| dino_push_straight_off | final | -0.026 | [-0.063, +0.012] | 0.051 | 0.03 | 112 |
| dino_push_straight_switch | init | +0.198 | [+0.107, +0.274] **CI-clean pos** | 0.158 | 1.14 | 112 |
| dino_push_straight_switch | final | +0.071 | [+0.020, +0.118] **CI-clean pos** | 0.069 | 0.42 | 112 |

## Gate verdict

- **dino_pick_l2**: rho_final -0.094 [-0.136, -0.051] -> FAIL (still CI-clean negative)
- **dino_pick_straight_none**: rho_final -0.003 [-0.046, +0.039] -> inconclusive (CI spans zero)
- **dino_pick_straight_off**: rho_final +0.001 [-0.045, +0.050] -> inconclusive (CI spans zero)
- **dino_pick_straight_switch**: rho_final +0.010 [-0.033, +0.059] -> inconclusive (CI spans zero)
- **dino_push_l2**: rho_final -0.085 [-0.135, -0.033] -> FAIL (still CI-clean negative)
- **dino_push_straight_none**: rho_final -0.011 [-0.040, +0.018] -> inconclusive (CI spans zero)
- **dino_push_straight_off**: rho_final -0.026 [-0.063, +0.012] -> inconclusive (CI spans zero)
- **dino_push_straight_switch**: rho_final +0.071 [+0.020, +0.118] -> PASS (CI-clean positive)
