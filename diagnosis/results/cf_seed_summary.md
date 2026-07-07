# Phase-H hardening: counterfactual predictor A/B, seed sweep

Model **dino_wm_droid**, CF seeds `['lora', 's1', 's2', 's3']` vs frozen baseline.
Pooled over 157 planned DROID transitions (weighted by `n_planned`).

| metric | frozen | CF seed-mean ± sd | range | all seeds beat frozen? |
|---|---|---|---|---|
| AE (↓) | 1.468 | 1.076 ± 0.012 (n=4) | [1.059, 1.086] | **yes** |
| AS (↑) | 0.466 | 0.544 ± 0.019 (n=4) | [0.525, 0.570] | **yes** |
| CRA (↑) | 0.061 | 0.610 ± 0.034 (n=4) | [0.583, 0.659] | **yes** |

Per-seed pooled values:

| seed | AE | AS | CRA |
|---|---|---|---|
| lora | 1.086 | 0.525 | 0.605 |
| s1 | 1.059 | 0.542 | 0.659 |
| s2 | 1.079 | 0.540 | 0.592 |
| s3 | 1.079 | 0.570 | 0.583 |

## Second model: jepa_wm_droid (single-seed A/B)

| tag | AE ↓ | AS ↑ | CRA ↑ | all better? |
|---|---|---|---|---|
| frozen | 1.386 | 0.476 | 0.076 | — |
| lora | 1.410 | 0.526 | 0.229 | partial |
| r16_l1p0 | 1.303 | 0.532 | 0.293 | **yes** |
| r16_l3p0 | 1.269 | 0.514 | 0.315 | **yes** |
