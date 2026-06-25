# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CAI-JEPA research. A **go/no-go validation study**: quantitatively determine whether existing action-conditioned JEPA world models (DINO-WM, V-JEPA-2-AC, JEPA-WM/Terver) exhibit measurable **action-grounding failures** — i.e. they produce near-identical latent predictions for *different* actions from the same state, especially in contact-rich Franka manipulation. If failures are real → pursue the full paper; if not → pivot or abandon. The deliverable is `diagnosis/results/decision_report.md`.

`AGENTS.md` (for Codex) covers the same project; keep both roughly in sync when changing orientation facts.

### Top-level layout
- `diagnosis/` — the implemented diagnostic (all the code). **This is where you work.**
- `cai_jepa_paper_proposal.md` — the research proposal: problem, the four diagnostic metrics, proposed training objectives.
- `diagnostic_implementation_plan_v2.md` — phased validation plan. Section 12 records v2.1 adjustments made after reading the real upstream API. Section 4.2 defines the decision threshold.
- `PROJECT_OVERVIEW_VI.md` — long-form overview (Vietnamese).
- `paper/` — LaTeX writeup (`main.tex`, `refs.bib`). `world_model/` — reference PDFs.

The most important orientation doc is `diagnosis/docs/plans/2026-06-01-real-api-rewrite-design.md` — it records what the upstream `facebookresearch/jepa-wms` API actually is and the key design decisions. The `docs/plans/*.md` files are dated design docs for each successive metric/fix (read the latest ones for current direction).

## Architecture of the diagnostic (`diagnosis/`)

Operates entirely on **pretrained, frozen checkpoints** — nothing here is trained except small linear probes / predictor heads (`models/probes/`, `models/heads/`, the `1[4-9]_*`/`2[0-5]_*` scripts). Data flow:

1. **Adapters** (`models/adapters/`) — `WorldModelAdapter` ABC (`base.py`) is the *only* interface the rest of the code touches a model through. One concrete `EncPredWMAdapter` (`enc_pred_adapter.py`) serves all baselines; `factory.build_adapter(model_name)` dispatches via the `_HUB_ID_INFO` registry (the canonical list of loadable checkpoints: `jepa_wm_metaworld`, `dino_wm_metaworld`, `*_droid`, `vjepa2_ac_droid`, etc.). All load via `torch.hub.load('facebookresearch/jepa-wms', hub_id, trust_repo=True)` → `(EncPredWM, preprocessor)`. The adapter drives the model through `EncPredWM.encode` (raw `[0,255]` visual → `(B,T,V,H,W,D)` patch-token latent) and `EncPredWM.unroll` — **never** `.encoder`/`.predictor` directly. `synthetic.py` provides fake adapters for offline metric validation.
2. **Latent extraction** (`scripts/03_extract_latents.py`) — encode every frame once, cache to HDF5 under `data/precomputed_latents/` (z + proprio + raw state + gripper). All metrics run on the cache, not the GPU model. Cache I/O is `data/latent_cache.py`; trajectory iterators wrapping the upstream loaders are `data/loaders.py`.
3. **Regime stratification** (`stratification/`) — labels each transition `free_space` / `pre_grasp` / `gripper_actuation` / `contact_manipulation`. Metaworld uses a **proxy** from the 39-dim `state` vector (object displacement = contact proxy) — the HF dataset has no MuJoCo contact GT. DROID/RoboCasa use proprioception + latent-change heuristics. The thesis is that failures concentrate in contact-rich regimes.
4. **Metrics** (`metrics/`) — CRA, AUG, ECS, CTD, and **Boundary Blindness** (`boundary_blindness.py`, the current core metric; see its module docstring). Each exposes a *per-transition* function the runner calls directly, so synthetic validation tests the production path. Primary decision signal is **effect-conditioned CRA** (CRA on transitions with `‖Δz‖>τ`). CIs are **trajectory-clustered** bootstrap (`bootstrap.py`). Negative-sampling strategies live in `negative_samplers.py` (`random`, `opposite`, `hard_nn`, `hard_effect`).
5. **Analysis** (`scripts/06_analyze_results.py`) — CSVs, figures, decision report. Decision logic is CI-aware (`make_decision`, per plan §4.2: ABANDON needs the upper CI bound confidently high).

Scripts are **numbered and run in order, per dataset/config**; each is idempotent and standalone. `planning/cem_planner.py` is the Action-Score planning probe (scripts `08`/`09`/`16`). One YAML config per dataset in `configs/` selects models, regimes, negative strategies, and dataset paths.

## Commands

`uv` for dependency management. Offline (no GPU/data) you can run the unit tests and synthetic validation; the full pipeline needs a GPU server.

```bash
cd diagnosis

# Offline — metric/code correctness (no GPU, no data, no checkpoints)
.venv/bin/python -m pytest tests/          # full suite
.venv/bin/python -m pytest tests/test_metrics_synthetic.py::test_name   # single test
python scripts/07_validate_synthetic.py    # validate metrics on synthetic models first

# Server — full pipeline (see diagnosis/RUNBOOK.md for the authoritative sequence)
bash scripts/01_setup_environment.sh        # clones external/jepa-wms + uv sync
python scripts/smoke_test.py                # every checkpoint loads + encode + predict
python scripts/check_normalization.py --config configs/diagnostic_metaworld.yaml \
    --model jepa_wm_metaworld --ref-eval-loss <EVAL_LOSS>
python scripts/03_extract_latents.py  --config configs/diagnostic_metaworld.yaml
python scripts/04_classify_regimes.py --config configs/diagnostic_metaworld.yaml
python scripts/05_run_diagnostic.py   --config configs/diagnostic_metaworld.yaml
python scripts/06_analyze_results.py \
    --metaworld_csv results/metaworld_diagnostic.csv --droid_csv results/droid_diagnostic.csv
```

SLURM batch scripts for the H100 cluster are `scripts/slurm_*.sh`; PowerShell sweep drivers are `scripts/run_*.ps1`. The upstream repo is cloned to `external/jepa-wms` (gitignored, with its own `.venv`) — adapters add it to `sys.path` lazily via `data.loaders.add_upstream_to_path`.

## Critical pitfalls (these cause silently-wrong numbers)

- **Action normalization is the #1 bug.** The real upstream method is `preprocessor.normalize_actions` (plural); the adapter wraps it as `normalize_action`. Always gate with `scripts/check_normalization.py` (predicts a real transition; MSE must be within ~2× the model's eval loss). DROID = identity (mean 0/std 1); Metaworld = real shift+scale. If MSE ≫ 2× eval loss → STOP.
- **Always sanity-check against a published number** and run `scripts/terver_gripper_test.py` (open vs close gripper on DROID; expect 2-way CRA > 0.90). Use the Terver-fixed `vjepa2_ac_droid` — the original Meta release shipped with an action-norm bug.
- **Validate metrics on synthetic models first** (`07_validate_synthetic.py`) before trusting any real-model number.
- All upstream planning configs are `L2_cem` → CRA and planning distance use **L2** for every baseline.
- ECS thresholds are **calibrated per model** automatically (median `‖z_{t+1}−z_t‖` over the eval set); the YAML `fallback_threshold` is a fallback only.
- **Push-T / PointMaze / Wall are saturated sanity checks only — never report them as thesis evidence.**
- `jepa_wm_robocasa` has a hub entrypoint but no checkpoint; run RoboCasa via the droid-trained checkpoints (shared 7-dim action format).
- `torch.hub.load` returning 503s → `rm external/jepa-wms/uv.lock && uv sync`.
- The ViT-G V-JEPA-2 / heavy-model paths are guarded against accidental small-GPU runs; set `CAI_JEPA_ALLOW_HEAVY_MODEL=1` (and `JEPAWM_OSSCKPT`) only on the intended 24 GB+ server.

## Result so far (2026-06-22)

Scaling study across 4 models (22M→1B): action-grounding does **not** scale away; all baselines remain boundary-blind (`results/droid_scaling_curve.md`, `results/decision_report.md`). This is the central finding driving the paper.
