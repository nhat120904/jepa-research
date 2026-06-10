# Handoff — Action-identifiability "fix" leg (contact-boundary reframing)

**Updated:** 2026-06-10 (**STEP 2 — the gate — has been RUN: PASS**).
**For:** the server agent (A5000) / local 12 GB box.
**Read first:** `docs/plans/2026-06-09-action-identifiability-fix-design.md` (full
design + rationale) and `docs/DIAGNOSIS_PLAN.md` (the current plan-of-record).
This file is the operational "what to run, in what order."

---

## 0. TL;DR — your immediate task

**UPDATE 2 (2026-06-10, later the same day): C1 was built and run — head-level
NULL, measured.** Four head variants (soft-NLL, WTA, +state-slice context,
supervised mode assignment) all improve NLL over K=1, none move BB
(`results/metaworld_boundary_fix.csv`). Probes pin two causes: the
"moves/doesn't" conditional future-means differ by only 9.9 L2 vs a 106 residual
(the latent metric compresses the boundary subspace ~10×), and expert data has no
counterfactual action coverage at the boundary (π action-flip ≈ 0). **Next task:
encoder/metric-level D** — supervised latent projection on the cached latents —
then re-run `scripts/13_eval_fix_boundary.py`. Read `docs/FIX_C1_EXPLAINER.md` §6
first. The §4 plan below is preserved for context; its head-level step 1 is done
(code: `models/heads/mixture_predictor.py`, `scripts/train_predictor_head.py`).
The boundary diagnostic was run on the local 12 GB box (Metaworld both baselines +
DROID dino_wm transfer): `bb_boundary` is CI-confidently elevated at the
**pre-grasp boundary** — Metaworld pooled 1.323/1.280 (dino/jepa) vs free_space
0.282/0.299 (excl. mw-door-close proxy anomaly); DROID 1.975 [1.601, 2.350] vs
0.721 [0.613, 0.834]. Full tables, run log, and two disclosed implementation fixes:
`results/boundary_gate_report.md`; gate verdict recorded in `DIAGNOSIS_PLAN.md` §4.

Still open on the server: STEP 1 below (5 min, direction D data question) and the
`vjepa2_ac_droid` leg (`03` → `05` → `12`, needs 24 GB).

```bash
cd <repo>/diagnosis && source .venv/bin/activate && export $(grep -v '^#' .env | xargs)

# STEP 1 (5 min, no GPU): resolve the one open data question for direction D.
python scripts/inspect_droid_observation_keys.py \
    --paths-csv data/droid_subset/droid_paths.csv --n 5
```

---

## 1. Why we are here (context in 6 lines)

- Diagnostic (CounterfactualBench + correlation study) stands. The **fix** leg changed.
- A critique killed the easy fixes: classifier-free action guidance AND the original
  one-step counterfactual margin loss only *amplify* existing action-sensitivity; they
  cannot *create* resolution of a sharp action→outcome boundary (gripper centred → cup
  lifts vs. 2–3° off → no lift).
- The real gap = **high-sensitivity / bifurcation regimes near contact boundaries**,
  where unimodal latent prediction (point + L2) averages across the two outcomes.
- New plan: **C1** = distributional/multimodal latent prediction + boundary-supervision
  head (hero, force-free); **D** = state-grounded latent (supporting, Metaworld only);
  plus the **boundary diagnostic** (now coded) that ships first on frozen baselines.

---

## 2. STEP 1 — the one open data question (direction D only)

We verified at the upstream source (`droid_dset.py:259-265,316-322`) that DROID
proprio/state is **7-dim = cartesian_position(6) + gripper_position(1)** with **no
force/torque and no joint state** — so direction D's force-grounded form is **off the
table on DROID**. The only remaining unknown: do raw episodes expose
`joint_position/velocity` (richer proprio, still not force) that a loader patch could add?

`scripts/inspect_droid_observation_keys.py` answers this. Interpreting its output:

- **force/torque FOUND** → unexpected; revisit the design, D-force may revive.
- **joint state FOUND, force absent** → DROID-D uses `[pose ‖ gripper ‖ joints]`
  (tiny loader patch + a proprio-only cache re-pass, no visual re-encode). Update design §2/§5.
- **only pose+gripper** (most likely) → DROID-D stays `[pose ‖ gripper-width]`; DROID is
  transfer-only. No change needed.

Either way the boundary *proof* is on Metaworld (the only dataset with object state →
a boundary label). DROID is a transfer check; do not over-claim tactile/force grounding.

---

## 3. STEP 2 — boundary diagnostic on frozen baselines (NO training) — **the gate — DONE: PASS (2026-06-10)**

Goal: prove the reframed gap is real and measurable before building any fix. Runs on
the existing **Metaworld** caches with frozen checkpoints — same cost profile as
`05_run_diagnostic.py`.

**Status: implemented.** No code to write — just run it.

- `stratification/boundary_regime.py` — boundary-regime selection. For each transition,
  similar-state neighbours (the hard_nn pool within `hard_nn.similarity_radius`) define
  a local neighbourhood; `boundary_score = std(true outcome) / mean(‖Δaction‖)` is high
  exactly when a small action change fans the outcome out (a bifurcation).
- `metrics/boundary_blindness.py` — `BB = relu(S_true_norm − S_model_norm)`, where
  `S_true` is the spread of the true outcome across the neighbourhood (object Δ on
  Metaworld; ‖Δz‖ proxy on DROID) and `S_model` the spread of the model's
  `F(z_t, a')` predictions over the same actions. `BB` large ⇒ the world bifurcates
  here but the model predicts ~the same future for every action.
- `scripts/12_boundary_diagnostic.py` (+ importable `scripts/_boundary_diagnostic.py`) —
  streams one (task, regime) cell at a time, standardises sensitivities over the whole
  model, and writes `results/{dataset}_boundary.csv` with bootstrap CIs.
- Tests: `tests/test_boundary_diagnostic.py` (11 tests) — incl. the synthetic proof
  that an action-ignoring model is boundary-blind and a perfect model is not.

Run also on DROID once its caches/regimes exist (transfer check, ‖Δz‖ proxy):

```bash
python scripts/12_boundary_diagnostic.py --config configs/diagnostic_droid.yaml
```

**GATE — read `results/metaworld_boundary.csv`:**
- Compare `bb_boundary` (BB on the boundary-flagged subset) across regimes.
- **Pass** (expected): `bb_boundary` is elevated, CI-aware, in `pre_grasp` /
  `gripper_actuation` / `contact_manipulation` relative to `free_space` → the gap is
  real and measurable; proceed to STEP 3.
- **Fail**: if `bb_boundary` is *not* elevated in the contact-rich regimes, the
  contact-boundary reframing is wrong — **stop and report** before building C1/D.

**RESULT (2026-06-10): PASS.** Pooled `bb_boundary` (n_b-weighted, excl. mw-door-close):
pre_grasp 1.323/1.280 (dino/jepa) vs free_space 0.282/0.299; contact 0.481/0.441;
per-task CI-aware pairing 4/6 and 5/6 confidently elevated, no confident reversals.
DROID transfer agrees (pre_grasp 1.975 [1.601, 2.350] vs free_space 0.721 [0.613, 0.834]).
Metaworld `gripper_actuation` cells empty as expected. Two implementation fixes were
needed and are disclosed (chunked predict for the 12 GB box; the hard_nn
relax-to-nearest fallback ported into `state_neighbours` — the first run was degenerate
without it). Sources: `results/{metaworld,droid}_boundary.csv`,
`results/boundary_gate_report.md`, figure `results/figures/figure_bb_per_regime.pdf`.

Deliverable: ~~the `bb` / `bb_boundary` per-regime table + a figure for the paper~~ — delivered.

---

## 4. STEP 3+ — the fix. STATUS UPDATE (2026-06-10, evening)

Build order as designed (design §4–§6), with measured outcomes:

1. **C1 head-level — DONE, NULL (measured).** `models/heads/mixture_predictor.py` +
   `scripts/train_predictor_head.py` + `scripts/13_eval_fix_boundary.py`. Four
   variants (soft-NLL K3, WTA K3, C1+D-at-the-head WTA K3, supervised-assignment
   K2): all beat K=1 on NLL, none move BB. Causes quantified (mode separation 9.9
   vs residual 106; no counterfactual action coverage). `docs/FIX_C1_EXPLAINER.md` §6.
2. **C1 + D at the head — DONE, NULL** (subsumed in 1; state-slice context did not
   rescue the head because the failure is in the *metric*, not the context).
3. **CURRENT METHOD-OF-RECORD: state-grounded latent metric** (encoder/metric-level
   D): `models/probes/object_probe.py` (probe `g(z)→obj xyz`, φ-metric adapter,
   boundary-aware CEM cost) + `scripts/14_train_object_probe.py` (V1–V3 gates) +
   `scripts/15_eval_metric_boundary.py` (BB under φ) +
   `scripts/16_planning_metric_compare.py` (paired Action-Error A/B). Success =
   BB drops under φ in pre_grasp AND paired Action Error improves with the φ cost.
4. **Transfer to DROID** — unchanged caveat: no state label → the exact probe form
   is Metaworld-only; DROID transfer of the *principle* needs a proxy label (open).
5. Remaining ablations: β sweep, probe capacity, mixture-head-on-top-of-φ (the C1
   retry in the re-weighted space, where mode separation is no longer ~9% of noise).

---

## 5. Guardrails (carry over from CLAUDE.md)

- Metric code is validated with synthetic models first (`tests/test_boundary_diagnostic.py`,
  `tests/test_mixture_predictor.py`, `tests/test_object_probe.py` follow the
  `07_validate_synthetic.py` pattern). Keep `pytest tests/` green (68 tests).
- L2 distance everywhere (planning configs are `L2_cem`).
- Metaworld boundary label is a **proxy** (object displacement, no MuJoCo contact GT) —
  state this explicitly in any result.
- Don't report Push-T / PointMaze as thesis evidence.

---

## 6. Files (boundary diagnostic — delivered & tested)

- `stratification/boundary_regime.py` — boundary-regime selection (state neighbours,
  boundary score, threshold/mask).
- `metrics/boundary_blindness.py` — the `BB` metric (S_true/S_model, standardise+relu).
- `scripts/_boundary_diagnostic.py` — importable runner core (true outcome, per-cell
  accumulation, global standardisation, per-cell bootstrap).
- `scripts/12_boundary_diagnostic.py` — CLI.
- `tests/test_boundary_diagnostic.py` — 11 unit/integration tests.
- `configs/diagnostic_{metaworld,droid}.yaml` — `boundary:` block.
- `docs/plans/2026-06-09-action-identifiability-fix-design.md` — full design.
- `scripts/inspect_droid_observation_keys.py` — STEP 1 tool.

**Still to code (STEP 3):** `models/heads/mixture_predictor.py`,
`train_predictor_head.py`, and the D latent-augmentation path.
