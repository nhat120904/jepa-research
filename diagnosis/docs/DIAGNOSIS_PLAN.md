# Diagnosis plan (plan-of-record, 2026-06-09)

Current, consolidated plan for the go/no-go diagnostic. Complements
`docs/METHODOLOGY.md` (concepts + code map) and
`docs/plans/2026-06-09-action-identifiability-fix-design.md` (fix design). Where the
original `diagnostic_implementation_plan_v2.md` and this file disagree, **this file
wins** for the boundary leg.

---

## 0. The question and the deliverable

Do frozen action-conditioned JEPA world models exhibit measurable **action-grounding
failures** — and specifically, do they **fail to resolve contact bifurcations**?
Deliverable: `results/decision_report.md` + the BB-per-regime table. Nothing is
trained in the diagnostic; everything runs on pretrained, frozen checkpoints.

## 1. Two questions, two metrics (the key distinction)

| Question | Metric | What it catches | What it misses |
|---|---|---|---|
| Does the model use actions *at all*? | CRA / AUG / **CRA_eff** (ECS) | one-step action-ignoring | a model that *responds* to actions but smears the boundary |
| Does the model *resolve the sharp boundary*? | **Boundary Blindness (BB)** | smearing of contact bifurcations | (this is the new signal) |

The thesis: failures concentrate in **contact-rich, high-sensitivity** regimes, and
the second question is the one that matters for planning. BB is the headline number.

## 2. The data / regime / metric matrix

- **Datasets:** Metaworld (primary, object-state → can label the boundary),
  DROID (secondary, transfer only — pose+gripper, no object GT, no force).
- **Models:** `jepa_wm_metaworld`, `dino_wm_metaworld` (+ `vjepa2_ac_droid`,
  `dino_wm_droid` on DROID). Frozen.
- **Regimes:** `free_space`, `pre_grasp`, `gripper_actuation`, `contact_manipulation`
  (Metaworld proxy from the 39-dim state; DROID from proprio + latent-change).
- **Boundary regime:** a *cross-cutting* selection within the above — transitions
  whose similar-state neighbourhood is a bifurcation (`boundary_score` high).
- **Metrics:** CRA(top1/MRR), AUG, ECS/CRA_eff, **BB**; all with trajectory-clustered
  bootstrap CIs. Distance = L2 everywhere (planning configs are `L2_cem`).

## 3. Pipeline (scripts, in order)

```
01_setup_environment.sh        clone upstream + uv sync
02_download_checkpoints.py     pull frozen checkpoints
03_extract_latents.py          encode every frame once → HDF5 (z, proprio, state, gripper)
04_classify_regimes.py         per-transition regime sidecar (JSON)
05_run_diagnostic.py           CRA / AUG / ECS / CRA_eff per (model×strategy×regime×task)
12_boundary_diagnostic.py      Boundary Blindness per regime  → results/{dataset}_boundary.csv   [GATE]
08_planning_probe.py           faithful CEM Action Error per transition
09_correlate_planning.py       Action Error ⇄ counterfactual sensitivity
06_analyze_results.py          CSVs → figures → decision_report.md
07_validate_synthetic.py       synthetic-model metric validation (offline)
```

Offline (no GPU/data): `pytest tests/` (56 tests) + `07_validate_synthetic.py`. The
boundary metric is validated in `tests/test_boundary_diagnostic.py` (action-ignoring
model → boundary-blind; perfect model → not).

## 4. The Boundary Blindness gate (procedure)

1. Ensure caches (`03`) + regimes (`04`) exist (Metaworld: done).
2. `python scripts/12_boundary_diagnostic.py --config configs/diagnostic_metaworld.yaml`.
3. For each anchor: draw similar-state neighbours from the hard_nn pool (within
   `hard_nn.similarity_radius`); true outcome = object displacement (Metaworld) /
   ‖Δz‖ (else); `boundary_score = std(outcome)/mean(‖Δaction‖)`;
   `S_true` = neighbourhood outcome spread; `S_model` = spread of `F(z_t, a')`.
4. Standardise `S_true`, `S_model` over the whole model; `BB = relu(S_true−S_model)`.
   Report `bb` (all) and `bb_boundary` (top-`quantile` boundary-score subset) per
   (task, regime) with bootstrap CIs.

**Gate decision (CI-aware):**
- **PASS** — `bb_boundary` confidently elevated in `pre_grasp` / `gripper_actuation` /
  `contact_manipulation` vs. `free_space` → gap proven, proceed to the fix (C1).
- **FAIL** — not elevated → the contact-boundary reframing is wrong; stop and report.

## 5. Status (2026-06-09)

| Leg | Status |
|---|---|
| Metaworld CRA/AUG/ECS (`05`) | **complete** — CONDITIONAL_GO; CRA_eff collapses in contact regimes |
| Boundary diagnostic (`12`, BB) | **coded + tested offline; run pending on server** (the gate) |
| DROID main metric (`05`) | set up & cached; finish `vjepa2_ac_droid` then run |
| Planning correlation (`08`/`09`) | run done; per-transition CRA_eff correlation **null** (class imbalance) → recast around BB |
| DROID data question (direction D) | `inspect_droid_observation_keys.py` pending (5 min) |
| The fix (C1 / D) | **not started** — gated on the BB result |

## 6. Honesty constraints (must appear in any result)

- Metaworld boundary label = object-displacement **proxy** (no MuJoCo contact GT).
- DROID = transfer check only: pose+gripper, **no force/joint** — no tactile claim.
- AUG is not cross-model comparable (latent-scale dependent); CRA/BB are.
- Push-T / PointMaze are sanity checks, never thesis evidence.

## 7. Immediate next actions

1. Server: run STEP 1 (`inspect_droid_observation_keys.py`) and STEP 2
   (`12_boundary_diagnostic.py` on Metaworld) — see `docs/HANDOFF_BOUNDARY_FIX.md`.
2. Read `results/metaworld_boundary.csv`, apply the §4 gate, fold the BB table +
   figure into `decision_report.md`.
3. If PASS: start C1 (`models/heads/mixture_predictor.py` + `train_predictor_head.py`).
