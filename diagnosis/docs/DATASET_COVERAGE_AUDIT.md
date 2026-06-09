# Dataset Coverage Audit

Current state as of 2026-06-08. "Covered" here means there is a diagnostic
config, a loader path in `data/loaders.py`, latent caches for the configured
models, regime sidecars, and a result CSV produced by `05_run_diagnostic.py`.

| Dataset | Requested config name | Current repo support | Current result state | Remaining work |
|---|---|---|---|---|
| Metaworld | `metaworld` | Implemented: `configs/diagnostic_metaworld.yaml`, `iterate_metaworld_trajectories`, regimes, metrics, analysis | `results/metaworld_diagnostic.csv` exists for `jepa_wm_metaworld` and `dino_wm_metaworld` | Full 42-task run would require expanding the config beyond the current 12-task subset and re-encoding/re-scoring |
| DROID | external download | Partially implemented: `configs/diagnostic_droid.yaml`, `iterate_droid_trajectories`, DROID subset CSV, regimes, metrics | `results/droid_diagnostic.csv` exists for `dino_wm_droid` only | Run `vjepa2_ac_droid` encode/score on a 24 GB GPU server; `jepa_wm_droid` remains blocked until gated DINOv3 `.pth` weights are available |
| RoboCasa | `robocasa` | Implemented: `configs/diagnostic_robocasa.yaml`, `iterate_robocasa_trajectories`, DROID-format Franka actions | No `results/robocasa_diagnostic.csv` or RoboCasa latent cache in this workspace | Download RoboCasa data, encode configured DROID-trained baselines, classify regimes, score metrics |
| franka_custom | `franka_custom` | Implemented: `configs/diagnostic_franka_custom.yaml`, `iterate_franka_custom_trajectories` via upstream MPK/HF DROID loader, DROID-style regimes | No cache or CSV | Download/locate franka_custom data, encode DROID-trained baselines, classify regimes, score metrics |
| Push-T | `pusht` | Implemented: `configs/diagnostic_pusht.yaml`, `iterate_pusht_trajectories`, free-space sanity regime | No cache or CSV | Download Push-T data, encode, classify, score |
| PointMaze | `point_maze` | Implemented: `configs/diagnostic_point_maze.yaml`, `iterate_point_maze_trajectories`, free-space sanity regime | No cache or CSV | Download PointMaze data, encode, classify, score |
| Wall | `wall` | Implemented: `configs/diagnostic_wall.yaml`, `iterate_wall_trajectories`, free-space sanity regime | No cache or CSV | Download Wall data, encode, classify, score |

## DROID Server Finish Commands

The local desktop has an 11.9 GiB GPU and now refuses to load
`vjepa2_ac_droid` by default. Run the remaining DROID step on the intended
24 GB GPU server:

```bash
cd diagnosis
export JEPAWM_OSSCKPT=/path/to/pretrained_opensource_encoders
export CAI_JEPA_ALLOW_HEAVY_MODEL=1
python scripts/03_extract_latents.py --config configs/diagnostic_droid.yaml
python scripts/04_classify_regimes.py --config configs/diagnostic_droid.yaml
python scripts/05_run_diagnostic.py --config configs/diagnostic_droid.yaml
python scripts/06_analyze_results.py \
  --metaworld_csv results/metaworld_diagnostic.csv \
  --droid_csv results/droid_diagnostic.csv
```

Or use the guarded runner:

```bash
python scripts/11_run_droid_completion.py --allow-known-blockers
```

Expected DROID completion evidence:

- `data/precomputed_latents/droid__vjepa2_ac_droid.h5`
- `data/precomputed_latents/droid__vjepa2_ac_droid.h5.regimes.json`
- `results/droid_diagnostic.csv` containing both `dino_wm_droid` and
  `vjepa2_ac_droid`
- `results/decision_report.md` regenerated from the completed Metaworld and
  DROID CSVs

`jepa_wm_droid` is not expected in the DROID CSV until the gated DINOv3 ViT-L
`.pth` dependency is obtained and wired through `JEPAWM_OSSCKPT`.

## Coverage Gate

After all server jobs finish, run:

```bash
python scripts/10_audit_coverage.py
```

This strict gate fails unless every requested dataset has its config, latent
caches, regime sidecars, and diagnostic CSV rows. It also enforces the user's
"all three baselines" DROID request: `dino_wm_droid`, `vjepa2_ac_droid`, and
`jepa_wm_droid`. If the DINOv3 gated-weight blocker is being explicitly waived
for a progress report, use:

```bash
python scripts/10_audit_coverage.py --allow-known-blockers
```

For just the DROID completion step, use:

```bash
python scripts/10_audit_coverage.py --dataset droid --allow-known-blockers
```
