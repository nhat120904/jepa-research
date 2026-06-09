# Handoff — Action-identifiability "fix" leg (contact-boundary reframing)

**Date:** 2026-06-09
**For:** the server agent (A5000, where the latent caches + DROID subset live).
**Read first:** `docs/plans/2026-06-09-action-identifiability-fix-design.md` (the full
design + rationale). This file is the operational "what to run, in what order."

---

## 0. TL;DR — your immediate task

The "fix" contribution of the paper was reframed (see §1). Before any fix is built,
**two things must happen on the server, in this order:**

```bash
cd <repo>/diagnosis && source .venv/bin/activate && export $(grep -v '^#' .env | xargs)

# STEP 1 (5 min, no GPU): resolve the one open data question.
python scripts/inspect_droid_observation_keys.py \
    --paths-csv data/droid_subset/droid_paths.csv --n 5
```

Then report the STEP-1 output back (it decides one design parameter — §2) and proceed
to STEP 2 (the boundary diagnostic, §3). Do **not** start training any fix until the
boundary diagnostic gate (§3) passes.

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
  plus a **boundary diagnostic** that ships first on frozen baselines.

---

## 2. STEP 1 — the one open data question

We verified at the upstream source (`droid_dset.py:259-265,316-322`) that DROID
proprio/state is **7-dim = cartesian_position(6) + gripper_position(1)** with **no
force/torque and no joint state** — so direction D's force-grounded form is **off the
table on DROID**. The *only* remaining unknown: do raw episodes expose
`joint_position/velocity` (richer proprio, still not force) that a loader patch could
add?

`scripts/inspect_droid_observation_keys.py` answers this. Interpreting its output:

- **force/torque category FOUND** → unexpected; revisit the design, D-force may revive.
- **joint state FOUND, force absent** → DROID-D uses `[pose ‖ gripper ‖ joints]`. Note
  this requires a tiny loader patch + a proprio-only cache re-pass (no visual
  re-encode). Update design §2/§5.
- **only pose+gripper** (most likely) → DROID-D stays `[pose ‖ gripper-width]`; DROID is
  transfer-only. No change needed.

Either way the boundary *proof* is on Metaworld (the only dataset with object state →
a boundary label). DROID is a transfer check; do not over-claim tactile/force grounding.

---

## 3. STEP 2 — boundary diagnostic on frozen baselines (ship first, NO training)

Goal: prove the reframed gap is real and measurable before building any fix. Runs on
the existing **Metaworld** cache (diagnostic already complete there) with frozen
checkpoints — same cost profile as the current `05_run_diagnostic.py`.

To build (design §3; not yet coded — these are the next code tasks):

1. `stratification/boundary_regime.py` + extend `04_classify_regimes.py`:
   select **boundary-regime** transitions = similar-state neighbourhoods whose *true*
   outcome (object displacement from the 39-dim Metaworld state) is **bimodal under
   small action differences**. Reuse the existing `hard_effect` pooling
   (`pool_size`, `similarity_radius` already in the DROID config; add equivalents to
   the Metaworld config).
2. `metrics/boundary_blindness.py` + wire into `05_run_diagnostic.py`:
   **Boundary Blindness** `BB = relu(S_true_norm − S_model_norm)`, where over a
   neighbourhood of near-by actions, `S_true` = spread of the true outcome and
   `S_model` = spread of the model's `F(z_t, a')` predictions. (Full def: design §3.2.)

**GATE:** run `BB` on the baselines. If `BB` is *not* elevated in boundary regimes
relative to free-space, the reframing is wrong — **stop and report** before building
C1/D. If it is elevated (expected: concentrated in pre-grasp / gripper-actuation), the
gap is proven and you proceed to the fix.

Deliverable: a `BB`-per-regime table added to the diagnostic CSVs + a figure.

---

## 4. STEP 3+ — the fix (only after the §3 gate passes)

Build order (design §4–§6). Metaworld first throughout; DROID transfer last.

1. **C1 — `models/heads/mixture_predictor.py` + `train_predictor_head.py`.**
   Mixture-density head (`K`≈2–4) over `z_{t+1}`, NLL loss, on cached Metaworld
   latents (frozen encoder + base trunk; DINO-WM scale → cheap on A5000). Add the
   **boundary-supervision head** `g_{t+1}` (grasp-success / object-moves from state).
   Success = `BB` drops in boundary regimes + planning improves vs baseline AND vs the
   original one-step contrastive loss (now a *fix baseline*, not the hero).
2. **C1 + D — latent augmentation.** `z̃_t = [z_t^vis ‖ φ(state-slice)]` on Metaworld
   (ee–object geometry + gripper). Cache already stores `state`/`proprio` → re-wire,
   **no visual re-encode**. Ablate D's marginal contribution.
3. **Transfer to DROID.** C1 on DROID latents (force-free → unaffected by §2).
   D-on-DROID limited to pose+gripper(+joints per STEP 1). Report as transfer.
4. Ablations: head type (mixture / flow / diffusion), `K`, boundary head on/off,
   sensitivity-supervision on/off, C1-only / D-only / C1+D.

---

## 5. Guardrails (carry over from CLAUDE.md)

- Validate metrics with synthetic models first (`scripts/07_validate_synthetic.py`
  pattern) — new `BB` metric should get a synthetic test in `tests/`.
- Keep `pytest tests/` green; add tests for `boundary_regime` + `boundary_blindness`.
- L2 distance everywhere (planning configs are `L2_cem`).
- Metaworld boundary label is a **proxy** (object displacement, no MuJoCo contact GT) —
  state this explicitly in any result.
- Don't report Push-T / PointMaze as thesis evidence.

---

## 6. Files delivered with this handoff

- `scripts/inspect_droid_observation_keys.py` — STEP 1 tool (done, compiles, run on server).
- `docs/plans/2026-06-09-action-identifiability-fix-design.md` — full design.
- `docs/HANDOFF_BOUNDARY_FIX.md` — this file.

Everything in STEP 2+ (`boundary_regime.py`, `boundary_blindness.py`,
`mixture_predictor.py`, `train_predictor_head.py`) is **not yet coded** — those are
your next tasks, in the order above.
