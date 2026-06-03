# CAI-JEPA Diagnostic Decision Report

**Decision:** CONDITIONAL_GO

**Justification:** Moderate pathology in at least one dataset: effect-conditioned CRA — MW(hard,contact)=0.648 [hi 0.701]; DROID(contact)=nan [hi nan]

## Critical cells

### Metaworld

| model | regime | CRA top-1 [95% CI] | CRA (effect-cond.) | AUG | ECS |
|---|---|---|---|---|---|
| dino_wm_metaworld | contact_manipulation | 0.459 [0.405, 0.512] | 0.483 | +0.0754 | +0.0833 |
| jepa_wm_metaworld | contact_manipulation | 0.550 [0.492, 0.612] | 0.572 | +0.0861 | +0.0957 |

## Figures

![Figure A](figures/figure_a_cra_per_regime.pdf)
![Figure B](figures/figure_b_metaworld_per_task.pdf)
![Figure C](figures/figure_c_correlation_planning.pdf)
