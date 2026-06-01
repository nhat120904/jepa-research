# Server Runbook — running the CAI-JEPA diagnostic end-to-end

This repo is correct against the real `facebookresearch/jepa-wms` API and fully
unit-tested for the parts that don't need a GPU. The steps below run the heavy
path (real checkpoints + datasets + GPU) on a server.

> **Hardware:** 1–2 GPUs (ViT-L for DROID), ~250 GB disk
> (Metaworld ~5 GB + DROID ~150 GB + latents ~50 GB).

## 0. Environment

```bash
# from repo root
bash scripts/01_setup_environment.sh          # clones external/jepa-wms + uv sync
cd diagnosis
uv venv && source .venv/bin/activate
uv pip install -e .                            # diagnostic deps
uv pip install -e ../diagnosis/external/jepa-wms  # OR add it to PYTHONPATH
# upstream extras the loaders/model need:
uv pip install einops tensordict omegaconf datasets imageio imageio-ffmpeg decord lpips
huggingface-cli login                          # for facebook/jepa-wms weights + data
```

The adapters import the upstream lazily and put `external/jepa-wms` on
`sys.path` automatically (`data.loaders.add_upstream_to_path`).

## 1. Checkpoints + data

```bash
python scripts/02_download_checkpoints.py      # pre-warms torch.hub / HF cache
# Datasets (from external/jepa-wms):
python external/jepa-wms/src/scripts/download_data.py --dataset metaworld robocasa pusht
# DROID: follow the upstream README gsutil instructions (left camera only).
```

Set dataset paths in the configs (`configs/diagnostic_*.yaml → dataset.root`).

## 2. Gate checks — DO THESE BEFORE TRUSTING ANY NUMBER

```bash
python scripts/smoke_test.py                   # every checkpoint loads + encode + predict
python scripts/07_validate_synthetic.py        # metric code is correct (also runs offline)

# The #1-bug gate: action normalization. ref-eval-loss = the model's reported
# eval MSE (JEPA-WMs paper / checkpoint logs). MSE must be within ~2×.
python scripts/check_normalization.py --config configs/diagnostic_metaworld.yaml \
    --model jepa_wm_metaworld --ref-eval-loss <EVAL_LOSS>
```

If `check_normalization` shows MSE ≫ 2× eval loss → STOP, normalization is wrong.

## 3. Pipeline (numbered; run in order, per dataset)

```bash
# Metaworld (primary)
python scripts/03_extract_latents.py --config configs/diagnostic_metaworld.yaml
python scripts/04_classify_regimes.py --config configs/diagnostic_metaworld.yaml
python scripts/05_run_diagnostic.py  --config configs/diagnostic_metaworld.yaml

# DROID (secondary)
python scripts/03_extract_latents.py --config configs/diagnostic_droid.yaml
python scripts/04_classify_regimes.py --config configs/diagnostic_droid.yaml
python scripts/05_run_diagnostic.py  --config configs/diagnostic_droid.yaml

# Easy-case sanity (Terver gripper test) on DROID
python scripts/terver_gripper_test.py --config configs/diagnostic_droid.yaml --model jepa_wm_droid
```

## 4. Sanity + decision

```bash
python scripts/sanity_check.py \
    --metaworld_csv results/metaworld_diagnostic.csv \
    --droid_csv     results/droid_diagnostic.csv
python scripts/06_analyze_results.py \
    --metaworld_csv results/metaworld_diagnostic.csv \
    --droid_csv     results/droid_diagnostic.csv
```

Deliverable: `results/decision_report.md` (GO / CONDITIONAL_GO / PIVOT /
ABANDON) — keyed on **effect-conditioned** CRA with trajectory-clustered CIs.

## Troubleshooting

- `torch.hub` 503s → `rm external/jepa-wms/uv.lock && uv sync`.
- `ImportError: app.plan_common...` → upstream repo not on path; re-run step 0.
- `jepa_wm_robocasa` "unknown model" → expected; RoboCasa has no checkpoint, run
  `jepa_wm_droid`/`dino_wm_droid` on the robocasa config instead.
- Out of memory on DROID latent extraction → lower `eval.batch_size`.
