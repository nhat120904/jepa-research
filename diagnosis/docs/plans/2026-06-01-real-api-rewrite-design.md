# Design & Decisions — Real-API Rewrite of the CAI-JEPA Diagnostic

Date: 2026-06-01. Status: implemented (integration code + metrics + tests). The
GPU/data path runs on the server (see `RUNBOOK.md`).

## Why this rewrite

The first implementation was written against a *guessed* API for
`facebookresearch/jepa-wms` and would have failed on first run. After cloning
the real repo into `external/jepa-wms` and reading the source, the integration
layer (model adapters + dataset loaders + stratification) was rewritten against
the genuine API, and several methodology gaps were closed.

## Ground truth (verified against upstream source)

| Concern | Reality | File |
|---|---|---|
| Model object | `EncPredWM` wrapping `VideoWM`; `torch.hub.load` returns `(model, preprocessor)` | `hubconf.py`, `app/vjepa_wm/modelcustom/.../vit_enc_preds.py` |
| Encode | `EncPredWM.encode(obs)` — raw `[0,255]` visual, does /255+transform+encoder internally; latent `(B,T,V,H,W,D)` | `vit_enc_preds.py` |
| Predict | `EncPredWM.unroll(z_ctxt, act_suffix=(T,B,A))` — the planner's primitive; handles ctxt window, proprio, per-`pred_type` predictor | `vit_enc_preds.py`, `evals/unroll_decode/eval.py` |
| Action norm | `preprocessor.normalize_actions` (plural); stats from hardcoded `DATA_STATS` | `preprocessor.py`, `datasets/__init__.py` |
| Datasets | `MetaworldHFDataset`/`DROIDVideoDataset`/`RoboCasaDataset` in `app/plan_common/datasets`; return `(obs, act, state[, reward, info])`, `obs={"visual","proprio"}` | `*_dset.py` |
| Distance | every config is `L2_cem` → planner uses **L2** | `hubconf._MODEL_CONFIGS` |
| Metaworld GT | **no contact GT**; 39-dim `state` carries ee/object/goal positions | `metaworld_hf_dset.py`, `DATA_STATS` |

The old code's worst bug: it checked `normalize_action` (singular, nonexistent)
→ silently fell back to identity → Metaworld actions were never normalized (the
plan's "Note 1" #1 bug). Fixed.

## Key decisions (confirmed with the user)

1. **Verification target = local + runbook.** This environment has no GPU, no
   ~200 GB datasets, no checkpoints. We (a) clone upstream + read real source,
   (b) rewrite all integration code against it, (c) unit-test everything that
   doesn't need GPU/data (metrics, samplers, bootstrap, cache, stratification,
   loader schema with mocks), (d) hand off the heavy path via `RUNBOOK.md`.
2. **Metaworld stays primary, stratified by proxy from the 39-dim `state`.**
   `ee=state[0:3]`, `gripper=state[3]`, `object=state[4:7]`. Regimes:
   gripper Δ → `gripper_actuation`; object displacement > 5 mm → (proxy)
   `contact_manipulation`; ee↔object < 10 cm → `pre_grasp`; else `free_space`.
   Honest framing: "contact" = "object measurably moved", not a MuJoCo sensor.
3. **Decision metric = CRA headline + effect-conditioned CRA primary + CTD
   support, CI-aware.** A low *raw* 1-step CRA in contact regimes can just mean
   the one-step latent barely moved; the *effect-conditioned* CRA (restricted to
   `‖Δz‖>τ`) is what shows the model fails to use actions when they matter. The
   GO/ABANDON logic keys off effect-conditioned CRA; ABANDON requires the upper
   CI bound to be confidently high, so a single noisy number can't trigger it.

## Other methodology fixes

- **Cluster bootstrap.** CIs resample whole trajectories, not iid transitions
  (within-trajectory correlation made the old iid CIs over-confident).
- **Proprioception threaded** through `predict`/CRA/AUG/ECS, respecting
  `model.use_proprio` (DROID checkpoints are `_noprop`).
- **Runner calls the metric functions** (`cra_per_transition`, …) instead of
  re-implementing them inline, so `07_validate_synthetic.py` and the unit tests
  exercise the exact production path.
- **Tie-aware CRA top-1.** An action-ignoring model produces a K-way tie; fair
  top-1 must be `1/(K+1)` (chance), not 0. (Caught by synthetic validation.)
- **AUG/ECS compared within-model only** (raw latent MSE isn't comparable across
  ViT-S vs ViT-L latents); CRA (ranking) is the cross-model metric.
- **`check_action_normalization`** implemented (the #1-bug gate) +
  **Terver gripper test** (`terver_gripper_test.py`) for the easy-case sanity.

## Known simplifications (documented, not hidden)

- One-step counterfactual uses a **tau=1 context**; CEM planning uses 2 frames.
  The ranking signal is preserved; full 2-frame context is a future refinement
  (the cache stores whole trajectories, so it can be added without re-encoding).
- **CTD** is wired (module + proprio threading) but not yet looped in the runner
  by default (`--ctd` flag is a stub); it was always "optional" in the plan.
- **RoboCasa** has no published checkpoint, so it runs on the DROID-trained
  checkpoints (shared 7-dim action format) and uses the DROID latent heuristic.

## What is verified vs pending

- Verified locally (no GPU/data): all metric math on synthetic models, negative
  samplers, cluster bootstrap, HDF5 cache roundtrip (incl. `task/idx` ids),
  Metaworld stratification, loader schema adaptation. `pytest tests/` = 23 green.
- Pending on server: `smoke_test.py` (real checkpoints load/encode/predict),
  `check_normalization.py` (MSE within 2× of eval loss), the full pipeline.
