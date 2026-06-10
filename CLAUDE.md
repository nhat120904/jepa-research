# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This repository contains the CAI-JEPA research and a **fully implemented diagnostic** in `diagnosis/`:

- `cai_jepa_paper_proposal.md` — the research proposal for "Counterfactual Action-Identifiable JEPA World Models for Robot Planning" (CAI-JEPA). Defines the problem, the four diagnostic metrics, and the proposed training objectives.
- `diagnostic_implementation_plan_v2.md` — the phased plan to validate the idea. **Section 12 records the v2.1 adjustments** made after reading the real upstream API.
- `diagnosis/` — the implemented go/no-go diagnostic **plus the fix legs**. Code is correct against the real `facebookresearch/jepa-wms` API and unit-tested offline (`pytest diagnosis/tests`, 68 tests). The GPU/data path runs on the server A5000 (24 GB) per `diagnosis/RUNBOOK.md` **or locally on a 12 GB GPU** (everything except `vjepa2_ac_droid`; run GPU scripts through `diagnosis/scripts/run_with_watchdog.ps1` on the 16 GB-RAM Windows box).

**Execution status (2026-06-10):**
- **Metaworld (primary): complete** — `diagnosis/results/metaworld_diagnostic.csv`, decision **CONDITIONAL_GO**. Gap is visible: `opposite` CRA ~0.97–0.99 but `hard_nn` ~0.46–0.57 in pre-grasp/contact regimes.
- **Boundary Blindness gate (`12`): RUN — PASS (2026-06-10, local 12 GB RTX 5070)** — `results/metaworld_boundary.csv` + `results/droid_boundary.csv`. Pooled `bb_boundary` (n_b-weighted, excl. the `mw-door-close` proxy anomaly): pre_grasp **1.323/1.280** (dino/jepa) vs free_space 0.282/0.299, confirmed CI-aware per task; DROID transfer (‖Δz‖ proxy): pre_grasp **1.975 [1.601, 2.350]** vs free_space 0.721 [0.613, 0.834]. The locus is the **pre-grasp boundary**, as the Boundary-Blind framing predicted → **fix C1 green-lit**. Full run log + caveats: `results/boundary_gate_report.md`. Two fixes shipped with the run: chunked BB predict (`CAI_JEPA_BB_PREDICT_ROWS`; an unchunked B×M unroll froze a 16 GB Windows box via driver sysmem fallback) and the hard_nn relax-to-nearest fallback in `state_neighbours` (without it BB degenerates to 0 on real caches — the radius never matches in raw latent units).
- **DROID (secondary): dino_wm_droid complete** — gripper sanity gate PASS; `05` rerun 2026-06-10 (`results/droid_diagnostic.csv`): hard_nn/hard_effect at the 16-way chance floor in pre-grasp/gripper/contact. `vjepa2_ac_droid` still pending — needs ~24 GB VRAM (server A5000 only; run `03` for it first, then `05`+`12`).
- **Planning Action-Score probe: run done** — per-transition CRA_eff correlation null (class imbalance); the regime-level link is clean on Metaworld. Per the idea-of-record (`diagnosis/docs/PAPER_IDEA.md`) the planning leg is recast around BB. Design: `diagnosis/docs/plans/2026-06-05-planning-action-score-design.md`; ops: HANDOFF_DROID §8.
- **Fix leg: complete on Metaworld (2026-06-10) — the grounded object-dynamics channel WORKS; everything else measured-null.** Full narrative `diagnosis/docs/FIX_C1_EXPLAINER.md` (§6 nulls, §7 the fix). The ladder, every rung from a CSV: (1) head-level mixture C1, 4 variants — null (mode separation 9.9 vs residual 106; π action-flip ≈ 0); (2) φ-metric re-weighting (`models/probes`, `15`) — null, redistributes BB; (3) probe chain V1✓/V2✓/V3✗ localizes the bottleneck to the predictor's *counterfactual action→object* channel (spread corr +0.035); (4) **`h(z,a)→Δobject`** (`scripts/17`, 0.5M params, frozen everything, cache-only): counterfactual tracking corr **+0.682**, pre_grasp `bb_boundary` **1.323→0.660 (−50%)**, boundary-vs-free gap 1.04→0.32 (`results/metaworld_boundary_dynamics.csv`). Planning A/B (`16`, paired CEM, `traj_cost_fn` hook): grounded cost is no-harm/no-gain on open-loop Action Error (metric rewards arm mimicry; closed-loop success rate = declared next experiment, server-side). One scale bug disclosed (`metaworld_planning_metric_buggy_scale.csv`).

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
python scripts/12_boundary_diagnostic.py --config configs/diagnostic_metaworld.yaml   # BB gate
python scripts/06_analyze_results.py  --metaworld_csv results/metaworld_diagnostic.csv --droid_csv results/droid_diagnostic.csv

# Fix legs (frozen trunk, cached latents; see docs/FIX_C1_EXPLAINER.md):
python scripts/train_predictor_head.py --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld --K 3 --objective wta   # head-level C1 (measured null)
python scripts/13_eval_fix_boundary.py --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld --ckpt checkpoints/<head>.pt
python scripts/14_train_object_probe.py --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld                        # state-grounded metric fix
python scripts/15_eval_metric_boundary.py --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld --probe checkpoints/object_probe_dino_wm_metaworld.pt
python scripts/16_planning_metric_compare.py --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld --probe checkpoints/object_probe_dino_wm_metaworld.pt
```

If `torch.hub.load` returns 503s, delete `external/jepa-wms/uv.lock` and re-run `uv sync`.
