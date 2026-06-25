# `regime_visualization.ipynb` — how to run the full notebook

Three parts: **1** Metaworld (frame galleries + state-space + CRA/BB), **2** DROID
(real wrist-camera frames + gripper-primary regimes), **3** cross-dataset regimes +
the DROID encoder-scaling curve. Parts 1–2 need frames (data below); Part 3 is
frame-free (sidecars + caches + result CSVs only).

## Data dependencies
| part | needs | how to get it |
| --- | --- | --- |
| 1 | `metaworld__dino_wm_metaworld.h5` + `.regimes.json` | `sbatch scripts/slurm_metaworld_cache.sh` (03+04, GPU) |
| 1 | Metaworld parquet `data/hf_mw/metaworld/data/train-*.parquet` (126 shards) | `hf_hub_download` from `facebook/jepa-wms` (dataset) — see `docs/HANDOFF.md §4` |
| 2 | `droid__dino_wm_droid.h5` + `.regimes.json`, `data/droid_subset/` | DROID pipeline (`scripts/slurm_droid_pipeline.sh`) + the staged subset |
| 3 | `{droid,franka_custom,pusht,point_maze,wall}__*.h5.regimes.json` + `results/*_diagnostic.csv` + `results/droid_boundary.csv` | the per-dataset diagnostic jobs |

Caches/parquet live on `/mnt/...` (symlinked under `data/`) and are git-ignored.

## Execute end-to-end (embeds all outputs)
```bash
cd diagnosis
# one-time: jupyter stack in the uv venv
uv pip install --python .venv/bin/python nbconvert nbclient ipykernel
# run every cell, write outputs back in place (~7 min; reads the 26 GB MW cache)
BOTO_CONFIG=/dev/null HF_HUB_OFFLINE=1 \
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3 \
  notebooks/regime_visualization.ipynb
```

## Rebuild the notebook structure (drops outputs)
The notebook is generated from `_build_regime_viz_nb.py`; Part 3 lives in
`_part3_cells.py` (shared source of truth). To regenerate then re-execute:
```bash
.venv/bin/python notebooks/_build_regime_viz_nb.py   # emits unexecuted .ipynb
# then the nbconvert --execute line above
```
To refresh **only** Part 3 onto an already-executed notebook (preserves the
Part 1/2 frame galleries), use `.venv/bin/python notebooks/_append_part3.py`.
