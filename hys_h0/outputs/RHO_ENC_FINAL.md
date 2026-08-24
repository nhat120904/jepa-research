# HyS-JEPA rho audit — does straightening survive CEM search?

Populations grouped by (task, seed, replan); CIs are bootstrap clustered on episode seed. `rho` = Spearman(proxy cost, true shaped cost) over the full candidate population. **The gate is rho_final.**

| arm | phase | rho | 95% CI | recall@10% | R_sel (cm) | n pops |
|---|---|---|---|---|---|---|
| dino_push_enc_random_s0 | init | +0.048 | [-0.031, +0.128] | 0.114 | 0.11 | 112 |
| dino_push_enc_random_s0 | final | -0.182 | [-0.225, -0.138] **CI-clean neg** | 0.015 | 0.01 | 112 |
| dino_push_enc_random_s1 | init | -0.042 | [-0.145, +0.055] | 0.096 | 0.17 | 112 |
| dino_push_enc_random_s1 | final | -0.049 | [-0.112, +0.011] | 0.030 | 0.02 | 112 |
| dino_push_enc_random_s2 | init | -0.030 | [-0.111, +0.052] | 0.032 | 0.05 | 112 |
| dino_push_enc_random_s2 | final | -0.009 | [-0.066, +0.040] | 0.016 | 0.00 | 112 |
| dino_push_enc_switch_s0 | init | -0.360 | [-0.439, -0.258] **CI-clean neg** | 0.023 | 0.16 | 112 |
| dino_push_enc_switch_s0 | final | -0.175 | [-0.228, -0.116] **CI-clean neg** | 0.023 | 0.03 | 112 |
| dino_push_enc_switch_s1 | init | +0.122 | [+0.037, +0.203] **CI-clean pos** | 0.080 | 0.01 | 112 |
| dino_push_enc_switch_s1 | final | +0.075 | [+0.030, +0.119] **CI-clean pos** | 0.041 | 0.00 | 112 |
| dino_push_enc_switch_s2 | init | +0.087 | [+0.003, +0.162] **CI-clean pos** | 0.130 | 0.21 | 112 |
| dino_push_enc_switch_s2 | final | -0.153 | [-0.214, -0.093] **CI-clean neg** | 0.015 | 0.16 | 112 |
| dino_push_l2 | init | +0.247 | [+0.163, +0.321] **CI-clean pos** | 0.155 | 0.11 | 112 |
| dino_push_l2 | final | -0.085 | [-0.135, -0.033] **CI-clean neg** | 0.022 | 0.05 | 112 |

## Gate verdict

- **dino_push_enc_random_s0**: rho_final -0.182 [-0.225, -0.138] -> FAIL (still CI-clean negative)
- **dino_push_enc_random_s1**: rho_final -0.049 [-0.112, +0.011] -> inconclusive (CI spans zero)
- **dino_push_enc_random_s2**: rho_final -0.009 [-0.066, +0.040] -> inconclusive (CI spans zero)
- **dino_push_enc_switch_s0**: rho_final -0.175 [-0.228, -0.116] -> FAIL (still CI-clean negative)
- **dino_push_enc_switch_s1**: rho_final +0.075 [+0.030, +0.119] -> PASS (CI-clean positive)
- **dino_push_enc_switch_s2**: rho_final -0.153 [-0.214, -0.093] -> FAIL (still CI-clean negative)
- **dino_push_l2**: rho_final -0.085 [-0.135, -0.033] -> FAIL (still CI-clean negative)
