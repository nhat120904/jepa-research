# Slurm job ledger — 2026-07-13

This is the exact submission ledger for the ICLR-hardening run. Live state was
last checked with both `squeue` and `sacct` at **2026-07-13 09:20 UTC**:
`26481`, `26485`, `26497`, `26502_0`, and `26502_1` completed successfully;
`26482` and `26502_2` were running; `26498` failed with an OOM exit 125 and was
resubmitted as `26610` with 128G RAM. All other research jobs were pending
quota/dependencies. Always re-check before acting. All commands were submitted
from `diagnosis/`.

| Job | Submission / script | Dependencies | Primary outputs | State at check |
|---|---|---|---|---|
| `26481` | `sbatch scripts/slurm_jepa_pipeline.sh` | none | `results/*_jepa.csv`, log `/mnt/data/nhatnc129/jepa_runs/logs/jepa_pipeline_26481.out` | completed, exit 0 |
| `26482` | `sbatch scripts/slurm_planner_generality.sh` | none | `results/latent_oracle_*_{mppi,shooting}.csv`, log `.../planner_generality_26482.out` | running on `worker-2` |
| `26485` | `sbatch scripts/slurm_encoder_infoUB.sh` | none | `results/encoder_info_upperbound.{csv,md}`, probes; log `.../jepa_infoUB_26485.out` | completed, exit 0 |
| `26493` | `P4H_MODEL=dino_wm_droid sbatch --array=0-3 --dependency=afterany:26481:26482:26485 scripts/slurm_phaseH_cf_heldout.sh` | `26481`,`26482`,`26485` | held-out manifests, checkpoints, frozen/LoRA CSVs; log `.../phaseH_cf_heldout_26493_%a.out` | pending dependency |
| `26494` | `P4H_MODEL=jepa_wm_droid sbatch --array=0-3 --dependency=afterany:26481:26482:26485 scripts/slurm_phaseH_cf_heldout.sh` | `26481`,`26482`,`26485` | same for JEPA-WM; log `.../phaseH_cf_heldout_26494_%a.out` | pending dependency |
| `26495` | `sbatch --dependency=afterok:26493:26494 scripts/slurm_phaseH_heldout_analysis.sh` | both held-out arrays successful | `results/cf_heldout_{dino,jepa}_summary.md` | pending dependency |
| `26491` | `sbatch --array=0-9%2 scripts/slurm_confirmatory_locked.sh`, then dependency updated to `afterany:26493:26494` | held-out arrays terminal | ten `results/confirmatory_*.csv` cells, 64 locked unseen seeds per cell | pending dependency |
| `26492` | `sbatch --dependency=afterok:26491 scripts/slurm_confirmatory_analysis.sh` | confirmatory array successful | `results/confirmatory_{summary,contrasts,report}.*` | pending dependency |
| `26497` | `sbatch scripts/slurm_same_state_intervention.sh` | none | `results/metaworld_same_state*`; log `.../same_state_26497.out` | completed, exit 0 |
| `26498` | `sbatch scripts/slurm_shared_scaling_protocol.sh` | none | partial `results/droid_shared_scaling*`; log `.../shared_scaling_26498.out` | failed OOM, exit 125 |
| `26499` | `sbatch scripts/slurm_paper_build.sh` | none | `paper/main.pdf` (19 pages); log `.../paper_build_26499.out` | completed, exit 0 |
| `26504` | `sbatch scripts/slurm_paper_build.sh` | none | post-literature-correction `paper/main.pdf` (19 pages); log `.../paper_build_26504.out` | completed, exit 0 |
| `26502` | `sbatch scripts/slurm_exploitation_instrumented.sh` | none | three `results/exploitation_instrumented_{episodes,curves}_*.csv` cells; log `.../exploit_instrumented_26502_%a.out` | queued/pending quota |
| `26503` | `sbatch --dependency=afterok:26502 scripts/slurm_exploitation_components_analysis.sh` | exploitation array successful | `results/exploitation_components_instrumented*`; log `.../exploit_components_26503.out` | pending dependency |
| `26505` | `sbatch scripts/slurm_oracle_coverage_selection.sh` | none | ten `results/oracle_covsel_*_{iterations,candidates,episodes}.*` cells; log `.../oracle_covsel_26505_%a.out` | queued/pending quota |
| `26506` | `sbatch --dependency=afterok:26505 scripts/slurm_oracle_coverage_selection_analysis.sh` | coverage array successful | `results/oracle_coverage_selection*`; log `.../oracle_covsel_analysis_26506.out` | pending dependency |
| `26507` | `sbatch scripts/slurm_trm_train.sh` | none | six split-safe TRM-style heads (2 models × 3 head seeds); log `.../trm_train_26507_%a.out` | queued/pending quota |
| `26508` | `sbatch --dependency=afterok:26507 scripts/slurm_trm_eval.sh` | all TRM heads successful | 34 held-out oracle cells, 64 seeds each; log `.../trm_eval_26508_%a.out` | pending dependency |
| `26509` | `sbatch --dependency=afterok:26508 scripts/slurm_trm_analysis.sh` | TRM eval array successful | `results/trm_heldout_{summary,report}.*`; log `.../trm_analysis_26509.out` | pending dependency |
| `26510` | `sbatch scripts/slurm_acid_idm_train.sh` | none | two split-safe approximate inverse-dynamics verifiers; log `.../acid_idm_26510_%a.out` | queued/pending quota |
| `26511` | `sbatch --dependency=afterok:26510 scripts/slurm_acid_baseline_eval.sh` | both ACID verifiers successful | 8 paired terminal/ACID cells on learned/oracle dynamics; log `.../acid_eval_26511_%a.out` | pending dependency |
| `26512` | `sbatch --dependency=afterok:26511 scripts/slurm_acid_analysis.sh` | all eight ACID eval cells successful | `results/acid_paired_{summary,report}.*`; log `.../acid_analysis_26512.out` | pending dependency |
| `26610` | `SHARED_SCALE_OUT_PREFIX=results/droid_shared_scaling_retry sbatch scripts/slurm_shared_scaling_protocol.sh` | none | retry outputs `results/droid_shared_scaling_retry*`; log `.../shared_scaling_26610.out` | pending quota |

The user-reported original jobs `26166` and `26400` were cancelled on
2026-07-12; `26481` and `26482` are their resume-safe replacements. Do not
submit duplicates merely because a queued job is waiting on quota.

The two analysis jobs use `afterok` deliberately: a partial/failed upstream
array must not silently generate a headline summary. Phase-H training uses
`afterany` on the three long-running predecessors to avoid deadlocking the
research queue if an unrelated predecessor fails; its own artifacts and split
provenance still determine validity.
