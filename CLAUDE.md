# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This repository contains the CAI-JEPA research and a **fully implemented diagnostic** in `diagnosis/`:

- `cai_jepa_paper_proposal.md` — the research proposal for "Counterfactual Action-Identifiable JEPA World Models for Robot Planning" (CAI-JEPA). Defines the problem, the four diagnostic metrics, and the proposed training objectives.
- `diagnostic_implementation_plan_v2.md` — the phased plan to validate the idea. **Section 12 records the v2.1 adjustments** made after reading the real upstream API.
- `diagnosis/` — the implemented go/no-go diagnostic. Code is correct against the real `facebookresearch/jepa-wms` API and unit-tested offline (`pytest diagnosis/tests`, 23 tests). The GPU/data path runs on a server per `diagnosis/RUNBOOK.md`.

**Execution status (2026-06-04):**
- **Metaworld (primary): complete** — `diagnosis/results/metaworld_diagnostic.csv`, decision **CONDITIONAL_GO**. Gap is visible: `opposite` CRA ~0.97–0.99 but `hard_nn` ~0.46–0.57 in pre-grasp/contact regimes.
- **DROID (secondary): set up & cached, metric step pending** — env built, 333-episode subset downloaded, latents encoded (`03`), regimes recalibrated (`04`). `05` was blocked by a GPU fall-off-the-bus fault (hardware/driver, not a code bug). Finish with `05`+`06` on a healthy GPU or `eval.device: cpu`.

**Start here for orientation, in order:**
1. `diagnosis/docs/METHODOLOGY.md` — concepts + code map + the dataset/task/regime/strategy matrix and how it proves the gap (read this first).
2. `diagnosis/docs/plans/2026-06-01-real-api-rewrite-design.md` — what the upstream API actually is + key design decisions.
3. `diagnosis/docs/HANDOFF.md` (Metaworld) and `diagnosis/docs/HANDOFF_DROID.md` (DROID) — operational "how to run / how to finish" handoffs.

## The Research Goal

A go/no-go validation study: quantitatively determine whether existing action-conditioned JEPA world models (DINO-WM, V-JEPA-2-AC, JEPA-WM/Terver) exhibit measurable **action-grounding failures** — i.e. they produce near-identical latent predictions for different actions from the same state — especially in contact-rich Franka manipulation. If failures are real → pursue the full paper; if not → pivot or abandon. The deliverable is `diagnosis/results/decision_report.md`.

## Architecture of the Diagnostic (`diagnosis/`)

Operates entirely on **pretrained, frozen checkpoints** — nothing is trained. Data flow:

1. **Adapters** (`models/adapters/`) — `WorldModelAdapter` ABC + one unified `EncPredWMAdapter` for all three baselines. They all load via `torch.hub.load('facebookresearch/jepa-wms', hub_id, trust_repo=True)` returning `(EncPredWM, preprocessor)`. The adapter drives the model through its own `EncPredWM.encode` (raw `[0,255]` visual in → `(B,T,V,H,W,D)` latent) and `EncPredWM.unroll` (the planner's prediction primitive) — **never** `.encoder`/`.predictor` directly.
2. **Latent extraction** (`scripts/03_extract_latents.py`) — encode every frame once, cache to HDF5 under `data/precomputed_latents/` (z + proprio + raw state + gripper). All metrics run on the cache.
3. **Regime stratification** (`stratification/`) — `free_space`, `pre_grasp`, `gripper_actuation`, `contact_manipulation`. Metaworld uses a **proxy** from the 39-dim `state` vector (ee/object positions; object displacement = contact proxy) — the HF dataset has no MuJoCo contact GT. DROID/RoboCasa use proprioception + latent-change heuristics. Metrics are per-regime; the thesis is failures concentrate in contact-rich regimes.
4. **Metrics** (`metrics/`) — CRA, AUG, ECS, CTD (optional). Each exposes a *per-transition* function the runner calls directly (so synthetic validation tests the production path). The primary decision signal is **effect-conditioned CRA** (CRA on transitions with `‖Δz‖>τ`). CIs are **trajectory-clustered** bootstrap.
5. **Analysis** (`scripts/06_analyze_results.py`) — CSVs, figures, and the decision report. Decision logic is CI-aware (ABANDON needs the upper CI bound confidently high).

## Key Implementation Pitfalls

- **Action normalization is the #1 bug.** The real method is `preprocessor.normalize_actions` (plural); the adapter calls it. Validate on the server with `scripts/check_normalization.py` (predict a real transition; MSE within ~2× the model's eval loss). DROID = identity (mean 0/std 1); Metaworld = real shift+scale.
- **Always sanity-check against a published number** and run `scripts/terver_gripper_test.py` (open vs close gripper on DROID; expect 2-way CRA > 0.90).
- **Validate metrics with synthetic models** first: `python scripts/07_validate_synthetic.py`.
- All planning configs are `L2_cem` → CRA uses **L2** for every baseline.
- Push-T / PointMaze are sanity checks only — never report them as thesis evidence.

## Environment & Commands

`uv` for dependency management. Offline (no GPU/data) you can run the unit tests; the full pipeline needs a server (see `diagnosis/RUNBOOK.md`).

```bash
cd diagnosis
# Offline: metric/code correctness (no GPU, no data, no checkpoints)
.venv/bin/python -m pytest tests/
python scripts/07_validate_synthetic.py

# Server: clone upstream + run the pipeline (see RUNBOOK.md for the full sequence)
bash scripts/01_setup_environment.sh          # clones external/jepa-wms + uv sync
python scripts/smoke_test.py                  # real checkpoints load + encode + predict
python scripts/check_normalization.py --config configs/diagnostic_metaworld.yaml --model jepa_wm_metaworld --ref-eval-loss <L>
python scripts/03_extract_latents.py  --config configs/diagnostic_metaworld.yaml
python scripts/04_classify_regimes.py --config configs/diagnostic_metaworld.yaml
python scripts/05_run_diagnostic.py   --config configs/diagnostic_metaworld.yaml
python scripts/06_analyze_results.py  --metaworld_csv results/metaworld_diagnostic.csv --droid_csv results/droid_diagnostic.csv
```

If `torch.hub.load` returns 503s, delete `external/jepa-wms/uv.lock` and re-run `uv sync`.
