# CROD H0 job ledger

The checkpoint advertised for the LeWM baseline suite is currently
inaccessible, so this run trains the official `stable-worldmodel`
`scripts/train/prejepa.py` reproduction rather than silently substituting an
unverified model. The auxiliary is DINOv2-Small, action-only, history 3,
frame-skip 5, batch 128, and 10 epochs.

Jobs are appended here immediately after submission and reconciled with both
`squeue` and `sacct`.

## Submitted 2026-08-18 UTC

| Job | Exact command | Dependency | Output | State at submission check |
|---|---|---|---|---|
| `42385` | `sbatch crod_h0/scripts/slurm_manifest.sh` | none | `crod_h0/outputs/h0/manifest.json`; log `/mnt/data/nhatnc129/jepa_runs/logs/crod_manifest_42385.out` | `COMPLETED`, exit `0:0`, 128 states, 288 prior episodes excluded |
| `42386` | `sbatch crod_h0/scripts/slurm_train_dinowm_cube.sh` | none | `$STABLEWM_HOME/checkpoints/crod_dinowm_cube_seed42/weights_epoch_10.pt`; log `/mnt/data/nhatnc129/jepa_runs/logs/crod_train_dino_42386.out` | `RUNNING` |
| `42388` | `sbatch --dependency=afterok:42385:42386 crod_h0/scripts/slurm_h0_smoke.sh` | manifest and DINO-WM train succeed | `crod_h0/outputs/h0/smoke/0/`; log `/mnt/data/nhatnc129/jepa_runs/logs/crod_h0_smoke_42388.out` | `PENDING (Dependency)` |
| `42387_[0-31]` | `sbatch --dependency=afterok:42386 crod_h0/scripts/slurm_complementarity_array.sh` | DINO-WM train succeeds | `crod_h0/outputs/complementarity/shards/{0..31}/`; logs `/mnt/data/nhatnc129/jepa_runs/logs/crod_comp_42387_%a.out` | `PENDING (Dependency)` |
| `42389` | `sbatch --dependency=afterok:42387 crod_h0/scripts/slurm_complementarity_aggregate.sh` | all complementarity shards succeed | `crod_h0/outputs/complementarity/aggregate/`; log `/mnt/data/nhatnc129/jepa_runs/logs/crod_comp_aggregate_42389.out` | `PENDING (Dependency)` |
| `42390_[0-31]` | `sbatch --dependency=afterok:42388 crod_h0/scripts/slurm_h0_array.sh` | H0 smoke succeeds | `crod_h0/outputs/h0/shards/{0..127}/`; logs `/mnt/data/nhatnc129/jepa_runs/logs/crod_h0_42390_%a.out` | `PENDING (Dependency)` |
| `42391` | `sbatch --dependency=afterok:42390 crod_h0/scripts/slurm_h0_aggregate.sh` | all H0 shards succeed | `crod_h0/outputs/h0/aggregate/`; log `/mnt/data/nhatnc129/jepa_runs/logs/crod_h0_aggregate_42391.out` | `PENDING (Dependency)` |

The fresh manifest SHA-256 is
`c9d1f62276c2ec299c326826a4117890851511bf46747b0e208fb61cababdb06`.
The queue and accounting states above were checked with both `squeue` and
`sacct` immediately after submission.
