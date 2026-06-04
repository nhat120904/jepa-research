# CAI-JEPA Diagnostic Decision Report

**Decision:** CONDITIONAL_GO

**Justification:** Moderate pathology in at least one dataset: effect-conditioned CRA — MW(hard contact-regimes)=0.651 [hi 0.703]; DROID(contact-regimes)=0.047 [hi 0.072]

## How To Read The Numbers

- CRA top-1 chance is approximately `0.059` because each factual action is ranked against 16 negatives.
- `CRA_eff` is the main decision signal: CRA restricted to transitions whose latent state actually changed.
- `AUG` is the factual-vs-counterfactual prediction gap; positive means factual actions predict closer next latents.
- `ECS` is `AUG` on effectful transitions only.
- `random` negatives test coarse action sensitivity, `opposite` negatives are usually easy, and `hard_nn`/`hard_effect` are the strict tests because they keep the current state similar while changing the action/effect.

## Sanity Gates

**DROID sanity gate:** PASSED: scripts\terver_gripper_test.py checks loader/cache/action plumbing; gripper delta max error 5.96e-08, cache-loader error 0, and model predictions are action/gripper-sensitive.

## Run Coverage

| dataset | models | rows | ok | insufficient |
| --- | --- | --- | --- | --- |
| Metaworld | dino_wm_metaworld, jepa_wm_metaworld | 384 | 256 | 128 |
| DROID | dino_wm_droid | 16 | 16 | 0 |

Metaworld `gripper_actuation` cells are mostly below the minimum transition count in this 60-trajectory/task diagnostic subset, so the Metaworld conclusion leans on `pre_grasp` and `contact_manipulation`.

## Strategy By Regime

### Metaworld

| strategy | regime | rows | transitions | CRA | CRA_eff | AUG | ECS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | free_space | 22 | 9094 | 0.765 | 0.858 | +0.0839 | +0.0955 |
| random | pre_grasp | 20 | 10508 | 0.414 | 0.664 | +0.0480 | +0.0997 |
| random | contact_manipulation | 22 | 7660 | 0.698 | 0.753 | +0.0807 | +0.0892 |
| opposite | free_space | 22 | 9094 | 0.966 | 0.992 | +0.0834 | +0.0934 |
| opposite | pre_grasp | 20 | 10508 | 0.857 | 0.966 | +0.0482 | +0.1008 |
| opposite | contact_manipulation | 22 | 7660 | 0.963 | 0.978 | +0.0803 | +0.0891 |
| hard_nn | free_space | 22 | 9094 | 0.564 | 0.601 | +0.0850 | +0.0959 |
| hard_nn | pre_grasp | 20 | 10508 | 0.368 | 0.491 | +0.0481 | +0.0998 |
| hard_nn | contact_manipulation | 22 | 7660 | 0.507 | 0.530 | +0.0814 | +0.0900 |
| hard_effect | free_space | 22 | 9094 | 0.564 | 0.601 | +0.0830 | +0.0934 |
| hard_effect | pre_grasp | 20 | 10508 | 0.368 | 0.491 | +0.0507 | +0.1078 |
| hard_effect | contact_manipulation | 22 | 7660 | 0.507 | 0.530 | +0.0812 | +0.0893 |

### DROID

| strategy | regime | rows | transitions | CRA | CRA_eff | AUG | ECS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | free_space | 1 | 998 | 0.167 | 0.500 | +0.0371 | +0.0294 |
| random | pre_grasp | 1 | 518 | 0.371 | 0.371 | +0.0171 | +0.0171 |
| random | gripper_actuation | 1 | 400 | 0.203 | 0.306 | +0.0289 | +0.0587 |
| random | contact_manipulation | 1 | 415 | 0.436 | 0.435 | +0.0031 | +0.0026 |
| opposite | free_space | 1 | 998 | 0.394 | 0.000 | +0.0371 | +0.0294 |
| opposite | pre_grasp | 1 | 518 | 0.044 | 0.044 | +0.0171 | +0.0171 |
| opposite | gripper_actuation | 1 | 400 | 0.100 | 0.048 | +0.0289 | +0.0587 |
| opposite | contact_manipulation | 1 | 415 | 0.171 | 0.171 | +0.0031 | +0.0026 |
| hard_nn | free_space | 1 | 998 | 0.082 | 0.000 | +0.0371 | +0.0294 |
| hard_nn | pre_grasp | 1 | 518 | 0.072 | 0.072 | +0.0171 | +0.0171 |
| hard_nn | gripper_actuation | 1 | 400 | 0.022 | 0.035 | +0.0289 | +0.0587 |
| hard_nn | contact_manipulation | 1 | 415 | 0.059 | 0.059 | +0.0031 | +0.0026 |
| hard_effect | free_space | 1 | 998 | 0.082 | 0.000 | +0.0371 | +0.0294 |
| hard_effect | pre_grasp | 1 | 518 | 0.072 | 0.072 | +0.0171 | +0.0171 |
| hard_effect | gripper_actuation | 1 | 400 | 0.022 | 0.035 | +0.0289 | +0.0587 |
| hard_effect | contact_manipulation | 1 | 415 | 0.059 | 0.059 | +0.0031 | +0.0026 |

## Hard-Negative Breakdown

### Metaworld strict negatives by model/regime

| model | strategy | regime | rows | transitions | CRA | CRA_eff [95% CI] | AUG | ECS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dino_wm_metaworld | hard_nn | free_space | 11 | 4547 | 0.527 | 0.568 [0.500, 0.635] | +0.0841 | +0.0952 |
| dino_wm_metaworld | hard_nn | pre_grasp | 10 | 5254 | 0.329 | 0.452 [0.358, 0.546] | +0.0455 | +0.0949 |
| dino_wm_metaworld | hard_nn | contact_manipulation | 11 | 3830 | 0.461 | 0.486 [0.423, 0.549] | +0.0753 | +0.0829 |
| dino_wm_metaworld | hard_effect | free_space | 11 | 4547 | 0.527 | 0.568 [0.500, 0.635] | +0.0808 | +0.0915 |
| dino_wm_metaworld | hard_effect | pre_grasp | 10 | 5254 | 0.329 | 0.452 [0.358, 0.546] | +0.0476 | +0.1005 |
| dino_wm_metaworld | hard_effect | contact_manipulation | 11 | 3830 | 0.461 | 0.486 [0.423, 0.549] | +0.0751 | +0.0823 |
| jepa_wm_metaworld | hard_nn | free_space | 11 | 4547 | 0.602 | 0.634 [0.569, 0.698] | +0.0859 | +0.0965 |
| jepa_wm_metaworld | hard_nn | pre_grasp | 10 | 5254 | 0.408 | 0.530 [0.432, 0.625] | +0.0507 | +0.1046 |
| jepa_wm_metaworld | hard_nn | contact_manipulation | 11 | 3830 | 0.552 | 0.574 [0.510, 0.639] | +0.0874 | +0.0972 |
| jepa_wm_metaworld | hard_effect | free_space | 11 | 4547 | 0.602 | 0.634 [0.569, 0.698] | +0.0852 | +0.0952 |
| jepa_wm_metaworld | hard_effect | pre_grasp | 10 | 5254 | 0.408 | 0.530 [0.432, 0.625] | +0.0538 | +0.1151 |
| jepa_wm_metaworld | hard_effect | contact_manipulation | 11 | 3830 | 0.552 | 0.574 [0.510, 0.639] | +0.0872 | +0.0963 |

### DROID strict negatives by model/regime

| model | strategy | regime | rows | transitions | CRA | CRA_eff [95% CI] | AUG | ECS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dino_wm_droid | hard_nn | free_space | 1 | 998 | 0.082 | 0.000 [0.000, 0.000] | +0.0371 | +0.0294 |
| dino_wm_droid | hard_nn | pre_grasp | 1 | 518 | 0.072 | 0.072 [0.051, 0.096] | +0.0171 | +0.0171 |
| dino_wm_droid | hard_nn | gripper_actuation | 1 | 400 | 0.022 | 0.035 [0.015, 0.061] | +0.0289 | +0.0587 |
| dino_wm_droid | hard_nn | contact_manipulation | 1 | 415 | 0.059 | 0.059 [0.039, 0.083] | +0.0031 | +0.0026 |
| dino_wm_droid | hard_effect | free_space | 1 | 998 | 0.082 | 0.000 [0.000, 0.000] | +0.0371 | +0.0294 |
| dino_wm_droid | hard_effect | pre_grasp | 1 | 518 | 0.072 | 0.072 [0.051, 0.096] | +0.0171 | +0.0171 |
| dino_wm_droid | hard_effect | gripper_actuation | 1 | 400 | 0.022 | 0.035 [0.015, 0.061] | +0.0289 | +0.0587 |
| dino_wm_droid | hard_effect | contact_manipulation | 1 | 415 | 0.059 | 0.059 [0.039, 0.083] | +0.0031 | +0.0026 |

## Interpretation

- Metaworld shows a large strategy gap: `opposite` negatives are near-saturated, `random` is intermediate, and `hard_nn` drops substantially. That means the models can react to gross action changes, but struggle when the counterfactual action is paired with a similar latent state.
- On Metaworld, `pre_grasp` is the weakest hard-negative regime and `contact_manipulation` remains only moderate. `free_space` is easier, which is expected because action effects are smoother and less contact-dependent.
- On Metaworld, `hard_effect` mirrors `hard_nn` in CRA/CRA_eff for this fixed candidate pool, so effect-aware candidate scoring does not rescue the ranking signal.
- `jepa_wm_metaworld` is consistently stronger than `dino_wm_metaworld`, but both still lose margin under `hard_nn`.
- On DROID, after the pipeline gate passes, `random` negatives are still separable in some regimes, while `hard_nn` and `hard_effect` are near chance in `gripper_actuation` and `contact_manipulation`. This is the sharpest action-grounding failure in the rerun.

## Decision Logic

- `GO` requires strong pathology in both datasets: Metaworld hard-task contact-regime `CRA_eff < 0.60` and DROID contact-regime `CRA_eff < 0.65`.
- `ABANDON` requires the upper confidence bounds to be high in both datasets: both contact-regime upper CIs at least `0.85`.
- `CONDITIONAL_GO` is used when at least one dataset shows moderate pathology below `0.75`, but the evidence is not strong enough for full `GO`.

Observed decision inputs: Metaworld hard-task contact-regime `CRA_eff=0.651` with upper CI `0.703`; DROID contact-regime `CRA_eff=0.047` with upper CI `0.072`.

Therefore the decision is `CONDITIONAL_GO`: DROID is strongly pathological and the pipeline now passes sanity checks, while Metaworld is clearly below the abandon threshold but not below the stricter full-GO threshold.

## Figures

![Figure A](figures/figure_a_cra_per_regime.pdf)
![Figure B](figures/figure_b_metaworld_per_task.pdf)
