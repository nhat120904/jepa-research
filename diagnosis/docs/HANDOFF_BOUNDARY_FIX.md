# Handoff — Action-identifiability "fix" leg (contact-boundary reframing)

**Updated:** 2026-06-09 (boundary diagnostic now **coded, tested, runnable**).
**For:** the server agent (A5000, where the latent caches + DROID subset live).
**Read first:** `docs/plans/2026-06-09-action-identifiability-fix-design.md` (full
design + rationale) and `docs/DIAGNOSIS_PLAN.md` (the current plan-of-record).
This file is the operational "what to run, in what order."

---

## 0. TL;DR — your immediate task

The boundary diagnostic (the **gate** that proves the reframed gap before any fix
is trained) is implemented and unit-tested offline. Two things on the server, in
this order:

```bash
cd <repo>/diagnosis && source .venv/bin/activate && export $(grep -v '^#' .env | xargs)

# STEP 1 (5 min, no GPU): resolve the one open data question for direction D.
python scripts/inspect_droid_observation_keys.py \
    --paths-csv data/droid_subset/droid_paths.csv --n 5

# STEP 2 (the GATE): run the boundary diagnostic on the frozen Metaworld baselines.
# Caches (03) + regimes (04) already exist for Metaworld, so this runs immediately.
python scripts/12_boundary_diagnostic.py --config configs/diagnostic_metaworld.yaml
```

Then inspect `results/metaworld_boundary.csv` and apply the **gate** (§3). Do **not**
start training any fix until the gate passes.

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

## 3. STEP 2 — boundary diagnostic on frozen baselines (NO training) — **the gate**

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

Deliverable: the `bb` / `bb_boundary` per-regime table + a figure for the paper.

---

## 4. STEP 3+ — the fix (only after the §3 gate passes)

Build order (design §4–§6). Metaworld first throughout; DROID transfer last.
**None of this is coded yet** — these are the next code tasks.

1. **C1 — `models/heads/mixture_predictor.py` + `train_predictor_head.py`.**
   Mixture-density head (`K`≈2–4) over `z_{t+1}`, NLL loss, on cached Metaworld
   latents (frozen encoder + base trunk; DINO-WM scale → cheap on A5000). Add the
   **boundary-supervision head** `g_{t+1}` (grasp-success / object-moves from state).
   Success = `BB` drops in boundary regimes (re-run `12_boundary_diagnostic.py` on the
   retrained head) + planning improves vs baseline AND vs the original one-step
   contrastive loss (now a *fix baseline*, not the hero).
2. **C1 + D — latent augmentation.** `z̃_t = [z_t^vis ‖ φ(state-slice)]` on Metaworld
   (ee–object geometry + gripper). Cache already stores `state`/`proprio` → re-wire,
   **no visual re-encode**. Ablate D's marginal contribution.
3. **Transfer to DROID.** C1 on DROID latents (force-free → unaffected by §2).
   D-on-DROID limited to pose+gripper(+joints per STEP 1). Report as transfer.
4. Ablations: head type (mixture / flow / diffusion), `K`, boundary head on/off,
   sensitivity-supervision on/off, C1-only / D-only / C1+D.

---

## 5. Guardrails (carry over from CLAUDE.md)

- Metric code is validated with synthetic models first (`tests/test_boundary_diagnostic.py`
  follows the `07_validate_synthetic.py` pattern). Keep `pytest tests/` green (56 tests).
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
