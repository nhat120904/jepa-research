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
12_boundary_diagnostic.py      Boundary Blindness per regime  → results/{dataset}_boundary.csv   [GATE — PASSED]
08_planning_probe.py           faithful CEM Action Error per transition
09_correlate_planning.py       Action Error ⇄ counterfactual sensitivity
06_analyze_results.py          CSVs → figures → decision_report.md
07_validate_synthetic.py       synthetic-model metric validation (offline)
--- fix legs (post-gate) ---
train_predictor_head.py        head-level C1 training (measured null — kept as the ablation)
13_eval_fix_boundary.py        BB before/after a trained head
14_train_object_probe.py       object probe g(z) + V1–V3 gates   [metric fix, step 1]
15_eval_metric_boundary.py     BB under the state-grounded φ metric → {dataset}_boundary_metric.csv
16_planning_metric_compare.py  paired CEM Action Error: L2 vs φ cost → {dataset}_planning_metric.csv
```

Offline (no GPU/data): `pytest tests/` (68 tests) + `07_validate_synthetic.py`. The
boundary metric is validated in `tests/test_boundary_diagnostic.py` (action-ignoring
model → boundary-blind; perfect model → not); the fix legs in
`tests/test_mixture_predictor.py` and `tests/test_object_probe.py`.

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

**VERDICT (run 2026-06-10, both Metaworld baselines + DROID dino_wm transfer): PASS.**
Pooled `bb_boundary` (n_b-weighted, excl. the `mw-door-close` proxy anomaly):
pre_grasp **1.323/1.280** (dino/jepa) vs free_space 0.282/0.299; contact 0.481/0.441.
Per-task CI-aware pairing: pre_grasp confidently elevated in 4/6 (dino) and 5/6 (jepa)
tasks, zero confident reversals. DROID (‖Δz‖ proxy): pre_grasp 1.975 [1.601, 2.350] vs
free_space 0.721 [0.613, 0.834] — CI-confident. The locus is the **pre-grasp boundary**
(contact_manipulation only moderate — post-grasp dynamics are smooth, few bifurcations).
Sources: `results/{metaworld,droid}_boundary.csv`; full analysis + run log (incl. two
disclosed fixes: chunked predict, hard_nn-fallback neighbourhoods) in
`results/boundary_gate_report.md`. Metaworld `gripper_actuation` cells empty as expected.

## 5. Status (2026-06-10)

| Leg | Status |
|---|---|
| Metaworld CRA/AUG/ECS (`05`) | **complete** — CONDITIONAL_GO; CRA_eff collapses in contact regimes |
| Boundary diagnostic (`12`, BB) | **RUN — gate PASSED** (2026-06-10, local 12 GB box): pre-grasp locus, CI-aware, both baselines + DROID transfer; see §4 verdict + `results/boundary_gate_report.md` |
| DROID main metric (`05`) | **dino_wm_droid complete** (rerun 2026-06-10; gripper sanity gate PASS; hard_nn at chance floor in contact regimes); `vjepa2_ac_droid` still server-only (needs 24 GB) |
| Planning correlation (`08`/`09`) | run done; per-transition CRA_eff correlation **null** (class imbalance) → recast around BB |
| DROID data question (direction D) | `inspect_droid_observation_keys.py` pending (5 min) |
| The fix — head-level C1 | **NULL, measured** (4 variants; ablation). `docs/FIX_C1_EXPLAINER.md` §6 |
| The fix — metric-only (φ-probe) | **NULL, measured** (redistributes BB; ablation). `results/metaworld_boundary_metric.csv` |
| The fix — **grounded dynamics channel h(z,a)→Δobj** | ✅ **WORKS (2026-06-10)**: counterfactual tracking corr +0.035 → **+0.682**; pre_grasp `bb_boundary` 1.323 → **0.660** (−50%); pre_grasp-vs-free gap 1.04 → 0.32. `results/metaworld_boundary_dynamics.csv`, explainer §7 |
| Planning A/B (CEM: L2 vs grounded cost) | **done** — no-harm/no-gain on open-loop Action Error (all paired CIs ∋ 0; n=6 pre_grasp); metric rewards full-arm mimicry → closed-loop success rate is the declared next experiment (server). `results/metaworld_planning_metric.csv` (+ `_buggy_scale.csv` disclosed) |

## 6. Honesty constraints (must appear in any result)

- Metaworld boundary label = object-displacement **proxy** (no MuJoCo contact GT).
- DROID = transfer check only: pose+gripper, **no force/joint** — no tactile claim.
- AUG is not cross-model comparable (latent-scale dependent); CRA/BB are.
- Push-T / PointMaze are sanity checks, never thesis evidence.

## 7. Immediate next actions

1. ~~Run STEP 2 (`12_boundary_diagnostic.py`), apply the §4 gate, fold BB into
   `decision_report.md`~~ — **done 2026-06-10 (PASS)**; BB table + figure are in
   `results/decision_report.md` and `results/figures/figure_bb_per_regime.pdf`.
2. ~~Start C1~~ — **done 2026-06-10, head-level null with measured causes** (§5
   table). Next fix task: **encoder/metric-level D** — train a supervised latent
   projection (object-displacement label, cached latents only) that re-amplifies
   the boundary subspace; re-run BB through `13_eval_fix_boundary.py` on top of it.
3. Server: STEP 1 (`inspect_droid_observation_keys.py`, 5 min) and `vjepa2_ac_droid`
   (`03` → `05` → `12`; needs 24 GB).
