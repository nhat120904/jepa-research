# CAI-JEPA Diagnostic

Action-grounding diagnostic for pretrained JEPA world models. Implements the
go/no-go validation study described in `../diagnostic_implementation_plan_v2.md`.

> **One core question:** do pretrained JEPA-WMs (DINO-WM, V-JEPA-2-AC, Terver
> JEPA-WM) produce measurably different latent rollouts under different actions
> from the same state — especially in contact-rich Franka regimes?

Nothing in this directory trains anything. The diagnostic operates entirely on
**pretrained, frozen** checkpoints.

> **Running on a server?** Follow [`RUNBOOK.md`](RUNBOOK.md) for the exact
> GPU/data steps. The integration is written against the **real**
> `facebookresearch/jepa-wms` API (see
> [`docs/plans/2026-06-01-real-api-rewrite-design.md`](docs/plans/2026-06-01-real-api-rewrite-design.md)
> for what the API actually is and why the first draft was wrong). The metric
> code is unit-tested offline: `pytest tests/` (23 tests, no GPU/data needed).

## Pipeline

Run scripts in order. Each one is idempotent and writes to `results/` or
`data/precomputed_latents/`.

```bash
# 0. Set up env + clone upstream jepa-wms repo (provides checkpoints + loaders).
bash scripts/01_setup_environment.sh

# 1. Pre-warm torch.hub cache.
python scripts/02_download_checkpoints.py

# 2. Verify every adapter loads + encodes + predicts.
python scripts/smoke_test.py

# 3. (Recommended) Validate metric code on synthetic models before real eval.
python scripts/07_validate_synthetic.py

# 4. Encode every frame once → HDF5 cache (most expensive step).
python scripts/03_extract_latents.py --config configs/diagnostic_metaworld.yaml
python scripts/03_extract_latents.py --config configs/diagnostic_droid.yaml

# 5. Tag every cached transition with its regime label.
python scripts/04_classify_regimes.py --config configs/diagnostic_metaworld.yaml
python scripts/04_classify_regimes.py --config configs/diagnostic_droid.yaml

# 6. Compute CRA / AUG / ECS per (model × strategy × regime × task) cell.
python scripts/05_run_diagnostic.py --config configs/diagnostic_metaworld.yaml
python scripts/05_run_diagnostic.py --config configs/diagnostic_droid.yaml

# 7. Sanity-check the results.
python scripts/sanity_check.py

# 8. Generate figures and the decision report.
python scripts/06_analyze_results.py \
    --metaworld_csv results/metaworld_diagnostic.csv \
    --droid_csv     results/droid_diagnostic.csv
```

The final artifact is `results/decision_report.md`, which contains a
GO / CONDITIONAL_GO / PIVOT / ABANDON verdict with justification, the critical
CRA values, and embedded figures.

## Layout

| Path | Purpose |
|---|---|
| `models/adapters/` | `WorldModelAdapter` ABC + one concrete adapter per baseline. The adapter ABC is the only place the rest of the code touches a model. |
| `metrics/`         | CRA, AUG, ECS, CTD, negative samplers, bootstrap CI. |
| `stratification/`  | Per-dataset regime classifier (MuJoCo ground truth on Metaworld, proprioception heuristics on DROID, object-grasp flags on RoboCasa). |
| `data/`            | HDF5 latent cache + trajectory iterators that wrap the upstream `external/jepa-wms` loaders. |
| `scripts/`         | Numbered pipeline scripts; each is runnable standalone. |
| `configs/`         | YAML configs — one per dataset. |
| `results/`         | CSVs, figures, decision report (gitignored). |

## Key gotchas (from the plan)

1. **Action normalization is the #1 source of bugs.** The real method is
   `preprocessor.normalize_actions` (plural) — the adapter calls it. Validate
   on the server with `scripts/check_normalization.py` (predicts a real
   transition; MSE must be within ~2× the model's eval loss). For DROID this is
   identity (mean 0/std 1); for Metaworld it is a real shift+scale.
2. **Always use the Terver-fixed `vjepa2_ac_droid`** — the original Meta
   release shipped with an action normalization bug.
3. **Calibrate ECS thresholds per model** — the runner does this automatically
   as the median ||z_{t+1} − z_t|| over the eval set.
4. **Never report Push-T / PointMaze as evidence** — they are saturated
   sanity checks only.
5. **Bootstrap CIs on every cell** — the decision logic in
   `06_analyze_results.py` reads them.

## Decision logic

The decision threshold (`scripts/06_analyze_results.py::make_decision`)
follows Section 4.2 of the plan: it focuses on JEPA-WM (the strongest
baseline) on `hard_nn` negatives in contact-rich regimes on the four hardest
Metaworld tasks. If aggregate CRA there is below 0.60 on Metaworld **and**
below 0.65 on DROID, the pathology is strong → **GO**.
