# GFPR H0-A report

**Locked verdict: `STOP_GFPR_FORMULATION`**

Four-fold outer predictions cover 32 distinct snapshots/episodes. No candidate from a held-out snapshot was used to fit its scorer.

| arm | success | success gain vs native | physical gain (cm) | corrective | switches | harmful / switches |
|---|---:|---:|---:|---:|---:|---:|
| action_diverse_oracle8 | 0.594 [0.438, 0.750] | 0.062 [0.000, 0.156] | 2.908 [1.333, 4.831] | 0.281 [0.125, 0.438] | 27 | 0 / 27 |
| action_only_gated | 0.500 [0.312, 0.656] | -0.031 [-0.125, 0.062] | -0.201 [-1.754, 0.838] | 0.125 [0.031, 0.250] | 11 | 2 / 11 |
| action_only_ungated | 0.500 [0.312, 0.656] | -0.031 [-0.125, 0.062] | 0.387 [-1.466, 2.062] | 0.188 [0.062, 0.344] | 32 | 2 / 32 |
| dino_best | 0.562 [0.406, 0.719] | 0.031 [-0.094, 0.156] | -0.528 [-2.999, 1.690] | 0.188 [0.062, 0.344] | 32 | 2 / 32 |
| latent_context_gated | 0.531 [0.344, 0.688] | 0.000 [-0.125, 0.125] | -0.143 [-2.474, 2.225] | 0.094 [0.000, 0.219] | 22 | 2 / 22 |
| latent_context_ungated | 0.531 [0.375, 0.688] | 0.000 [-0.125, 0.125] | 0.237 [-1.756, 2.284] | 0.094 [0.000, 0.188] | 32 | 2 / 32 |
| native | 0.531 [0.344, 0.719] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 | 0 / 0 |
| physical_oracle_full | 0.781 [0.625, 0.906] | 0.250 [0.125, 0.406] | 7.054 [4.243, 10.100] | 0.500 [0.312, 0.688] | 28 | 0 / 28 |
| proxy_action_gated | 0.500 [0.344, 0.688] | -0.031 [-0.125, 0.062] | -0.422 [-1.862, 0.473] | 0.062 [0.000, 0.156] | 9 | 2 / 9 |
| proxy_action_ungated | 0.562 [0.375, 0.719] | 0.031 [-0.094, 0.156] | 1.404 [-0.960, 3.877] | 0.188 [0.062, 0.312] | 32 | 2 / 32 |

`action_diverse_oracle8` and `physical_oracle_full` inspect physical outcomes and are upper bounds, not deployable zero-query selectors.

The primary gate is defined in `PROTOCOL.md`. A GO verdict licenses a fresh frozen-model endpoint; it is not final paper evidence.
