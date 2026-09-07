# Scene progress-WM: goal-offset ladder

| Arm | Δ=25 | Δ=50 |
|---|---|---|
| _ceiling (dataset replay)_ | 100% | 98% |
| `latent_l2` | 28.0% [16.0, 40.0] | 36.0% [24.0, 50.0] |
| `prog_ssl_w0.25` | 40.0% [26.0, 54.0] | 26.0% [14.0, 38.0] |

## Paired difference vs `latent_l2` (points)

| Arm | Δ=25 | Δ=50 |
|---|---|---|
| `prog_ssl_w0.25` | +12.0 [+0.0, +26.0] | -10.0 [-22.0, +2.0] |

Bold marks an interval excluding zero.
