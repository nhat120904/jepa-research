# HyS-JEPA rho audit — does straightening survive CEM search?

Populations grouped by (task, seed, replan); CIs are bootstrap clustered on episode seed. `rho` = Spearman(proxy cost, true shaped cost) over the full candidate population. **The gate is rho_final.**

| arm | phase | rho | 95% CI | recall@10% | R_sel (cm) | n pops |
|---|---|---|---|---|---|---|
| dino_push_l2 | init | +0.247 | [+0.163, +0.321] **CI-clean pos** | 0.155 | 0.11 | 112 |
| dino_push_l2 | final | -0.085 | [-0.135, -0.033] **CI-clean neg** | 0.022 | 0.05 | 112 |
| dino_push_v2_random_s0 | init | +0.307 | [+0.220, +0.370] **CI-clean pos** | 0.140 | 0.25 | 112 |
| dino_push_v2_random_s0 | final | +0.067 | [+0.021, +0.105] **CI-clean pos** | 0.039 | 0.07 | 112 |
| dino_push_v2_random_s1 | init | +0.055 | [-0.030, +0.139] | 0.104 | 0.41 | 112 |
| dino_push_v2_random_s1 | final | +0.018 | [-0.015, +0.050] | 0.062 | 0.03 | 112 |
| dino_push_v2_random_s2 | init | +0.192 | [+0.125, +0.249] **CI-clean pos** | 0.172 | 1.19 | 112 |
| dino_push_v2_random_s2 | final | +0.068 | [+0.028, +0.105] **CI-clean pos** | 0.062 | 0.53 | 112 |
| dino_push_v2_switch_s0 | init | -0.131 | [-0.200, -0.045] **CI-clean neg** | 0.029 | 0.13 | 112 |
| dino_push_v2_switch_s0 | final | -0.003 | [-0.031, +0.026] | 0.024 | 0.02 | 112 |
| dino_push_v2_switch_s1 | init | +0.047 | [-0.019, +0.112] | 0.102 | 0.38 | 112 |
| dino_push_v2_switch_s1 | final | +0.000 | [-0.029, +0.029] | 0.052 | 0.06 | 112 |
| dino_push_v2_switch_s2 | init | +0.192 | [+0.120, +0.254] **CI-clean pos** | 0.146 | 1.27 | 112 |
| dino_push_v2_switch_s2 | final | +0.036 | [+0.007, +0.060] **CI-clean pos** | 0.052 | 0.30 | 112 |

## Gate verdict

- **dino_push_l2**: rho_final -0.085 [-0.135, -0.033] -> FAIL (still CI-clean negative)
- **dino_push_v2_random_s0**: rho_final +0.067 [+0.021, +0.105] -> PASS (CI-clean positive)
- **dino_push_v2_random_s1**: rho_final +0.018 [-0.015, +0.050] -> inconclusive (CI spans zero)
- **dino_push_v2_random_s2**: rho_final +0.068 [+0.028, +0.105] -> PASS (CI-clean positive)
- **dino_push_v2_switch_s0**: rho_final -0.003 [-0.031, +0.026] -> inconclusive (CI spans zero)
- **dino_push_v2_switch_s1**: rho_final +0.000 [-0.029, +0.029] -> inconclusive (CI spans zero)
- **dino_push_v2_switch_s2**: rho_final +0.036 [+0.007, +0.060] -> PASS (CI-clean positive)
