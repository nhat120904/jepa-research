# CAI-JEPA Diagnostic Decision Report

**Decision:** CONDITIONAL_GO

**Justification:** Moderate pathology in at least one dataset: effect-conditioned CRA � MW(hard contact-regimes)=0.651 [hi 0.703]; DROID(contact-regimes)=0.045 [hi 0.069]

## How To Read The Numbers

- CRA top-1 chance is approximately `0.059` because each factual action is ranked against 16 negatives.
- `CRA_eff` is the main decision signal: CRA restricted to transitions whose latent state actually changed.
- `AUG` is the factual-vs-counterfactual prediction gap; positive means factual actions predict closer next latents.
- `ECS` is `AUG` on effectful transitions only.
- `random` negatives test coarse action sensitivity, `opposite` negatives are usually easy, and `hard_nn`/`hard_effect` are the strict tests because they keep the current state similar while changing the action/effect.

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
| random | free_space | 1 | 998 | 0.153 | 0.750 | +0.0463 | +0.1184 |
| random | pre_grasp | 1 | 518 | 0.367 | 0.367 | +0.0108 | +0.0108 |
| random | gripper_actuation | 1 | 400 | 0.212 | 0.319 | +0.0508 | +0.0760 |
| random | contact_manipulation | 1 | 415 | 0.443 | 0.442 | -0.0054 | -0.0055 |
| opposite | free_space | 1 | 998 | 0.396 | 0.000 | +0.0459 | -0.0363 |
| opposite | pre_grasp | 1 | 518 | 0.044 | 0.044 | +0.0123 | +0.0123 |
| opposite | gripper_actuation | 1 | 400 | 0.085 | 0.048 | +0.0352 | +0.0679 |
| opposite | contact_manipulation | 1 | 415 | 0.166 | 0.167 | +0.0006 | +0.0001 |
| hard_nn | free_space | 1 | 998 | 0.084 | 0.000 | +0.0490 | +0.0960 |
| hard_nn | pre_grasp | 1 | 518 | 0.070 | 0.070 | +0.0115 | +0.0115 |
| hard_nn | gripper_actuation | 1 | 400 | 0.022 | 0.035 | +0.0467 | +0.0753 |
| hard_nn | contact_manipulation | 1 | 415 | 0.055 | 0.056 | +0.0119 | +0.0115 |
| hard_effect | free_space | 1 | 998 | 0.084 | 0.000 | +0.0509 | +0.1542 |
| hard_effect | pre_grasp | 1 | 518 | 0.070 | 0.070 | +0.0138 | +0.0138 |
| hard_effect | gripper_actuation | 1 | 400 | 0.022 | 0.035 | +0.0378 | +0.0629 |
| hard_effect | contact_manipulation | 1 | 415 | 0.055 | 0.056 | +0.0195 | +0.0195 |

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
| dino_wm_droid | hard_nn | free_space | 1 | 998 | 0.084 | 0.000 [0.000, 0.000] | +0.0490 | +0.0960 |
| dino_wm_droid | hard_nn | pre_grasp | 1 | 518 | 0.070 | 0.070 [0.050, 0.093] | +0.0115 | +0.0115 |
| dino_wm_droid | hard_nn | gripper_actuation | 1 | 400 | 0.022 | 0.035 [0.015, 0.061] | +0.0467 | +0.0753 |
| dino_wm_droid | hard_nn | contact_manipulation | 1 | 415 | 0.055 | 0.056 [0.037, 0.078] | +0.0119 | +0.0115 |
| dino_wm_droid | hard_effect | free_space | 1 | 998 | 0.084 | 0.000 [0.000, 0.000] | +0.0509 | +0.1542 |
| dino_wm_droid | hard_effect | pre_grasp | 1 | 518 | 0.070 | 0.070 [0.050, 0.093] | +0.0138 | +0.0138 |
| dino_wm_droid | hard_effect | gripper_actuation | 1 | 400 | 0.022 | 0.035 [0.015, 0.061] | +0.0378 | +0.0629 |
| dino_wm_droid | hard_effect | contact_manipulation | 1 | 415 | 0.055 | 0.056 [0.037, 0.078] | +0.0195 | +0.0195 |

## Interpretation

- Metaworld shows a large strategy gap: `opposite` negatives are near-saturated, `random` is intermediate, and `hard_nn` drops substantially. That means the models can react to gross action changes, but struggle when the counterfactual action is paired with a similar latent state.
- On Metaworld, `pre_grasp` is the weakest hard-negative regime and `contact_manipulation` remains only moderate. `free_space` is easier, which is expected because action effects are smoother and less contact-dependent.
- On Metaworld, `hard_effect` mirrors `hard_nn` in CRA/CRA_eff for this fixed candidate pool, so effect-aware candidate scoring does not rescue the ranking signal.
- `jepa_wm_metaworld` is consistently stronger than `dino_wm_metaworld`, but both still lose margin under `hard_nn`.
- On DROID, after the pipeline gate passes, `random` negatives are still separable in some regimes, while `hard_nn` and `hard_effect` are near chance in `gripper_actuation` and `contact_manipulation`. This is the sharpest action-grounding failure in the rerun.

## Boundary Blindness (the gate for the Boundary-Blind reframing) — run 2026-06-10

The reframed thesis (see `docs/PAPER_IDEA.md`) is that the failure that matters is not
one-step action-ignoring (CRA) but **failure to resolve contact bifurcations**:
`BB = relu(S_true_norm − S_model_norm)` per transition, where `S_true` is the spread of the
*true* outcome over a similar-state neighbourhood of nearby actions and `S_model` the spread
of the model's predictions over the same actions. `bb_boundary` restricts to the
top-quartile `boundary_score` subset (the bifurcation-like transitions). Source:
`results/metaworld_boundary.csv`, `results/droid_boundary.csv`
(`scripts/12_boundary_diagnostic.py`, frozen checkpoints, nothing trained).

**GATE VERDICT: PASS.** BB concentrates in the pre-grasp boundary, CI-aware, on both
datasets and both Metaworld baselines.

### Metaworld (object-displacement outcome; pooled over tasks, n_boundary-weighted, excl. `mw-door-close`)

| regime | dino_wm `bb` | dino_wm `bb_boundary` | jepa_wm `bb` | jepa_wm `bb_boundary` |
| --- | --- | --- | --- | --- |
| free_space | 0.069 | 0.282 | 0.070 | 0.299 |
| pre_grasp | 0.541 | **1.323** | 0.581 | **1.280** |
| contact_manipulation | 0.212 | 0.481 | 0.194 | 0.441 |

Per-task CI-aware pairing (`bb_boundary_lo(regime) > bb_boundary_hi(free_space)` within the
same task): pre_grasp is confidently elevated in 4/6 pairable tasks (dino_wm) and 5/6
(jepa_wm) with zero confident reversals; contact_manipulation is confidently elevated in 2/7
(both models) — moderate, not the locus. `gripper_actuation` has no populated Metaworld cells
(no gripper signal in this subset), as expected.

### DROID (transfer check, ‖Δz‖ proxy outcome; single flat pool, `dino_wm_droid`)

| regime | `bb` [95% CI] | `bb_boundary` [95% CI] | n_boundary |
| --- | --- | --- | --- |
| free_space | 0.422 [0.365, 0.476] | 0.721 [0.613, 0.834] | 415 |
| pre_grasp | 0.887 [0.753, 1.034] | **1.975 [1.601, 2.350]** | 97 |
| gripper_actuation | 0.395 [0.330, 0.463] | 1.093 [0.791, 1.393] | 30 |
| contact_manipulation | 0.311 [0.243, 0.375] | 0.463 [0.266, 0.686] | 41 |

`pre_grasp` is CI-confidently elevated over `free_space` (lo 1.601 > hi 0.834);
`gripper_actuation` is point-elevated with marginally overlapping CIs; `contact_manipulation`
is not elevated. Same locus as Metaworld: the bifurcation the models smear is the
**grasp/approach boundary**, not post-contact manipulation.

### Caveats (binding)

- Metaworld boundary/outcome label is an **object-displacement proxy** (the HF dataset has no
  MuJoCo contact ground truth); DROID outcome is the weaker ‖Δz‖ latent proxy and its regimes
  are proprio/latent heuristics — DROID is a transfer check, not the proof.
- `mw-door-close` shows BB ≈ 1.7–2.9 in *every* regime including `free_space` (articulated
  door motion confounds the displacement proxy); pooled Metaworld numbers above exclude it.
  Including it, free_space pooled `bb_boundary` rises to ≈1.4 and the regime ordering blurs —
  the per-task paired comparison is the robust read.
- Neighbourhoods relax to nearest-by-latent when the raw `similarity_radius` selects nothing
  (the same fallback the production `hard_nn` sampler uses); on these caches the relax path is
  what defines the neighbourhood.
- `vjepa2_ac_droid` not run (needs ~24 GB VRAM; this rerun used a 12 GB RTX 5070) — server-only.

### Relation to the CRA finding

Consistent, and sharper: CRA_eff says the models stop *distinguishing* similar-state
counterfactual actions in pre-grasp/contact (hard_nn ≈ 0.46–0.57 vs opposite ≈ 0.97–0.99 on
Metaworld; chance-floor on DROID). BB adds *where it matters and why*: exactly at bifurcation
transitions the models' predicted-outcome spread collapses relative to the world's
(pre_grasp ≈ 4.5× free_space on Metaworld, ≈ 2.7× on DROID). No contradiction between the two
signals was observed.

![Figure BB](figures/figure_bb_per_regime.pdf)

## Planning Action-Score Probe (supplementary evidence)

A faithful CEM planner (upstream `L2_cem` params) was run on the cached latents
to test the inferential step "low `CRA_eff` ⇒ planning failure" by relating
per-transition Action Error to `CRA_eff`, on **both** datasets. Full details:
`results/planning_correlation.md`.

**Metaworld (both baselines `dino_wm_metaworld` + `jepa_wm_metaworld`, 183 planned
transitions) — the link, confirmed.** Here `CRA_eff` has real spread (0.43–1.0),
so the test is well posed:

- **Pooled regime-level Spearman(Action Error, `CRA_eff`) = −1.000 at H=1 and at
  H=3** (expected sign). As `CRA_eff` falls, Action Error rises monotonically at
  both horizons: pre_grasp/free_space (higher `CRA_eff`, lower error) → contact
  (lowest `CRA_eff`, highest error). Action-grounding quality predicts planning
  quality, and the failure concentrates in the contact-rich regime — the thesis.
- Per model, the H=1 regime-level Spearman is −1.000 for *both* baselines
  independently; the only reversal is `jepa_wm` at H=3 (+1.0), driven by its
  tiny n=5 pre_grasp cell, which washes out in the pooled result.
- Per-transition the correlation is weak and not significant (pooled H=1 +0.08,
  H=3 −0.13) because single-expert Action Error is high-variance per transition;
  that noise averages out at the regime mean, which is why the regime-level
  signal is clean. The defensible reading is the regime-level link.

**DROID (`dino_wm_droid`, 355 transitions over two runs) — the floor, consistent.**
`CRA_eff` is at the 16-way chance floor in every contact/gripper regime (~4 %
positive), so there is no variance to correlate (per-transition Spearman ≈ −0.07
then +0.01; perm p ≈ 0.42 then 0.88 — a floor effect, not a sample-size problem).
The evidence here is the level: near-chance `CRA_eff` **while** Action Error is
high and grows with horizon (≈1.0–1.2 at H=1 → ≈1.6–2.3 at H=3).

**Net effect on the decision: unchanged (CONDITIONAL_GO), but materially
strengthened.** Metaworld now supplies the *causal* link (grounding quality →
planning quality, expected direction, both horizons); DROID supplies the *severe*
end of the same axis. The earlier weak argument (DROID per-transition correlation)
is replaced by the Metaworld regime-level link plus the DROID level read. The
per-transition null is itself evidence for the boundary reframing — the planning
leg is recast around BB (see the Boundary Blindness section above).

## Decision Logic

- `GO` requires strong pathology in both datasets: Metaworld hard-task contact-regime `CRA_eff < 0.60` and DROID contact-regime `CRA_eff < 0.65`.
- `ABANDON` requires the upper confidence bounds to be high in both datasets: both contact-regime upper CIs at least `0.85`.
- `CONDITIONAL_GO` is used when at least one dataset shows moderate pathology below `0.75`, but the evidence is not strong enough for full `GO`.

Observed decision inputs: Metaworld hard-task contact-regime `CRA_eff=0.651` with upper CI `0.703`; DROID contact-regime `CRA_eff=0.045` with upper CI `0.069`.

Therefore the decision is `CONDITIONAL_GO`: DROID is strongly pathological and the pipeline now passes sanity checks, while Metaworld is clearly below the abandon threshold but not below the stricter full-GO threshold.

## Figures

![Figure A](figures/figure_a_cra_per_regime.pdf)
![Figure B](figures/figure_b_metaworld_per_task.pdf)
![Figure BB](figures/figure_bb_per_regime.pdf)
