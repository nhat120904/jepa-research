# HyS-JEPA rho audit — does straightening survive CEM search?

Populations grouped by (task, seed, replan); CIs are bootstrap clustered on episode seed. `rho` = Spearman(proxy cost, true shaped cost) over the full candidate population. **The gate is rho_final.**

| arm | phase | rho | 95% CI | recall@10% | R_sel (cm) | n pops |
|---|---|---|---|---|---|---|
| dino_push_l2 | init | +0.247 | [+0.163, +0.321] **CI-clean pos** | 0.155 | 0.11 | 112 |
| dino_push_l2 | final | -0.085 | [-0.135, -0.033] **CI-clean neg** | 0.022 | 0.05 | 112 |
| dino_push_straight_none | init | -0.077 | [-0.138, +0.000] | 0.047 | 0.03 | 112 |
| dino_push_straight_none | final | -0.011 | [-0.040, +0.018] | 0.027 | 0.02 | 112 |
| dino_push_straight_none_s1 | init | +0.028 | [-0.050, +0.096] | 0.044 | 0.04 | 112 |
| dino_push_straight_none_s1 | final | +0.021 | [-0.032, +0.069] | 0.028 | 0.02 | 112 |
| dino_push_straight_none_s2 | init | +0.047 | [-0.023, +0.113] | 0.061 | 0.07 | 112 |
| dino_push_straight_none_s2 | final | -0.099 | [-0.128, -0.072] **CI-clean neg** | 0.035 | 0.00 | 112 |
| dino_push_straight_switch | init | +0.198 | [+0.107, +0.274] **CI-clean pos** | 0.158 | 1.14 | 112 |
| dino_push_straight_switch | final | +0.071 | [+0.020, +0.118] **CI-clean pos** | 0.069 | 0.42 | 112 |
| dino_push_straight_switch_s1 | init | +0.066 | [-0.017, +0.145] | 0.104 | 0.37 | 112 |
| dino_push_straight_switch_s1 | final | +0.014 | [-0.021, +0.052] | 0.045 | 0.05 | 112 |
| dino_push_straight_switch_s2 | init | +0.174 | [+0.096, +0.247] **CI-clean pos** | 0.122 | 0.55 | 112 |
| dino_push_straight_switch_s2 | final | +0.035 | [+0.001, +0.071] **CI-clean pos** | 0.054 | 0.03 | 112 |

## Gate verdict

- **dino_push_l2**: rho_final -0.085 [-0.135, -0.033] -> FAIL (still CI-clean negative)
- **dino_push_straight_none**: rho_final -0.011 [-0.040, +0.018] -> inconclusive (CI spans zero)
- **dino_push_straight_none_s1**: rho_final +0.021 [-0.032, +0.069] -> inconclusive (CI spans zero)
- **dino_push_straight_none_s2**: rho_final -0.099 [-0.128, -0.072] -> FAIL (still CI-clean negative)
- **dino_push_straight_switch**: rho_final +0.071 [+0.020, +0.118] -> PASS (CI-clean positive)
- **dino_push_straight_switch_s1**: rho_final +0.014 [-0.021, +0.052] -> inconclusive (CI spans zero)
- **dino_push_straight_switch_s2**: rho_final +0.035 [+0.001, +0.071] -> PASS (CI-clean positive)
