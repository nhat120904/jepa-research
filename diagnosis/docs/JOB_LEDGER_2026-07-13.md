# Slurm job ledger — 2026-07-13

This is the exact submission ledger for the ICLR-hardening run. Live state was
last checked with both `squeue` and `sacct` at **2026-07-16 17:35 UTC**.
Core locked oracle, same-population, shared-branch, coverage, held-out Phase-H,
and permutation-null artifacts are complete. TRM evaluation tasks 8--9 were
running with 10--33 pending; ACID smoke `27982` and shared scaling `26610` were
pending quota. Always re-check before acting. All commands were submitted from
`diagnosis/`.

| Job | Submission / script | Dependencies | Primary outputs | State at check |
|---|---|---|---|---|
| `26481` | `sbatch scripts/slurm_jepa_pipeline.sh` | none | `results/*_jepa.csv`, log `/mnt/data/nhatnc129/jepa_runs/logs/jepa_pipeline_26481.out` | completed, exit 0 |
| `26482` | `sbatch scripts/slurm_planner_generality.sh` | none | `results/latent_oracle_*_{mppi,shooting}.csv`, log `.../planner_generality_26482.out` | completed, exit 0; push CEM/MPPI/shooting = 1/16, 0/16, 2/16 |
| `26485` | `sbatch scripts/slurm_encoder_infoUB.sh` | none | `results/encoder_info_upperbound.{csv,md}`, probes; log `.../jepa_infoUB_26485.out` | completed, exit 0 |
| `26493` | `P4H_MODEL=dino_wm_droid sbatch --array=0-3 --dependency=afterany:26481:26482:26485 scripts/slurm_phaseH_cf_heldout.sh` | `26481`,`26482`,`26485` | held-out manifests, checkpoints, frozen/LoRA CSVs; log `.../phaseH_cf_heldout_26493_%a.out` | completed, exit 0 |
| `26494` | `P4H_MODEL=jepa_wm_droid sbatch --array=0-3 --dependency=afterany:26481:26482:26485 scripts/slurm_phaseH_cf_heldout.sh` | `26481`,`26482`,`26485` | same for JEPA-WM; log `.../phaseH_cf_heldout_26494_%a.out` | completed, exit 0 |
| `26495` | `sbatch --dependency=afterok:26493:26494 scripts/slurm_phaseH_heldout_analysis.sh` | both held-out arrays successful | `results/cf_heldout_{dino,jepa}_summary.md` | completed, exit 0 |
| `26491` | `sbatch --array=0-9%2 scripts/slurm_confirmatory_locked.sh`, then dependency updated to `afterany:26493:26494` | held-out arrays terminal | ten `results/confirmatory_*.csv` cells, 64 locked unseen seeds per cell | completed, all tasks exit 0 |
| `26492` | `sbatch --dependency=afterok:26491 scripts/slurm_confirmatory_analysis.sh` | confirmatory array successful | `results/confirmatory_{summary,contrasts,report}.*` | completed, exit 0 |
| `26497` | `sbatch scripts/slurm_same_state_intervention.sh` | none | `results/metaworld_same_state*`; log `.../same_state_26497.out` | completed, exit 0 |
| `26498` | `sbatch scripts/slurm_shared_scaling_protocol.sh` | none | partial `results/droid_shared_scaling*`; log `.../shared_scaling_26498.out` | failed OOM, exit 125 |
| `26499` | `sbatch scripts/slurm_paper_build.sh` | none | `paper/main.pdf` (19 pages); log `.../paper_build_26499.out` | completed, exit 0 |
| `26504` | `sbatch scripts/slurm_paper_build.sh` | none | post-literature-correction `paper/main.pdf` (19 pages); log `.../paper_build_26504.out` | completed, exit 0 |
| `26502` | `sbatch scripts/slurm_exploitation_instrumented.sh` | none | three `results/exploitation_instrumented_{episodes,curves}_*.csv` cells; log `.../exploit_instrumented_26502_%a.out` | completed, all tasks exit 0 |
| `26503` | `sbatch --dependency=afterok:26502 scripts/slurm_exploitation_components_analysis.sh` | exploitation array successful | `results/exploitation_components_instrumented*`; log `.../exploit_components_26503.out` | failed exit 1: optional `tabulate` missing |
| `26505` | `sbatch scripts/slurm_oracle_coverage_selection.sh` | none | ten `results/oracle_covsel_*_{iterations,candidates,episodes}.*` cells; log `.../oracle_covsel_26505_%a.out` | completed, all tasks exit 0 |
| `26506` | `sbatch --dependency=afterok:26505 scripts/slurm_oracle_coverage_selection_analysis.sh` | coverage array successful | `results/oracle_coverage_selection*`; log `.../oracle_covsel_analysis_26506.out` | failed exit 1; corrected analysis artifacts were subsequently produced 2026-07-14 |
| `26507` | `sbatch scripts/slurm_trm_train.sh` | none | six split-safe TRM-style heads (2 models × 3 head seeds); log `.../trm_train_26507_%a.out` | queued/pending quota |
| `26508` | `sbatch --dependency=afterok:26507 scripts/slurm_trm_eval.sh` | all TRM heads successful | 34 held-out oracle cells, 64 seeds each; log `.../trm_eval_26508_%a.out` | pending dependency |
| `26509` | `sbatch --dependency=afterok:26508 scripts/slurm_trm_analysis.sh` | TRM eval array successful | `results/trm_heldout_{summary,report}.*`; log `.../trm_analysis_26509.out` | pending dependency |
| `26510` | `sbatch scripts/slurm_acid_idm_train.sh` | none | two split-safe approximate inverse-dynamics verifiers; log `.../acid_idm_26510_%a.out` | queued/pending quota |
| `26511` | `sbatch --dependency=afterok:26510 scripts/slurm_acid_baseline_eval.sh` | both ACID verifiers successful | 8 paired terminal/ACID cells on learned/oracle dynamics; log `.../acid_eval_26511_%a.out` | pending dependency |
| `26512` | `sbatch --dependency=afterok:26511 scripts/slurm_acid_analysis.sh` | all eight ACID eval cells successful | `results/acid_paired_{summary,report}.*`; log `.../acid_analysis_26512.out` | pending dependency |
| `26610` | `SHARED_SCALE_OUT_PREFIX=results/droid_shared_scaling_retry sbatch scripts/slurm_shared_scaling_protocol.sh` | none | retry outputs `results/droid_shared_scaling_retry*`; log `.../shared_scaling_26610.out` | pending quota |
| `26746` | `sbatch --dependency=afterok:26502 scripts/slurm_exploitation_components_analysis.sh` | exploitation array successful | replacement analysis outputs `results/exploitation_components_instrumented*` | completed, exit 0 |
| `27982` | `ACID_EPISODES=1 sbatch --array=0 scripts/slurm_acid_baseline_eval.sh` | none; renderer-recovery smoke test | `results/acid_dino_wm_metaworld_learned_mw-push_seed22000_n1.csv`, log `.../acid_eval_27982_0.out` | submitted 2026-07-15; pending `QOSMaxGRESPerUser`. `26511` and dependent `26512` were cancelled after tasks 0--2 exposed an EGL renderer lifecycle failure. |
| `27990` | `sbatch scripts/slurm_cem_preselection_audit.sh` | none | eight GPU cells under `results/cem_preselection_<tag>_{episodes,iterations,candidates}.*`; logs `.../cem_preselection_27990_%a.out` | submitted 2026-07-15; locked same-population raw-vs-proxy-elite error/optimism audit, 2 models x 2 contact tasks x 2 costs, seeds 41000--41015 |
| `27991` | `sbatch --dependency=afterok:27990 scripts/slurm_cem_preselection_analysis.sh` | `27990` | `results/cem_preselection_audit.{md,csv}` and population/first-final CSVs; log `.../cem_preselection_analysis_27991.out` | submitted 2026-07-15; pending dependency |
| `27994` | `sbatch scripts/slurm_shared_population_branch.sh` | none | four shared-noise GPU cells under `results/shared_branch_<tag>_*`; logs `.../shared_branch_27994_%a.out` | submitted 2026-07-15; Stage-B pilot, 2 models x push/pick, proxy-vs-true branches, true-state carrier, seeds 42000--42007 |
| `27995` | `sbatch --dependency=afterok:27994 scripts/slurm_shared_population_branch_analysis.sh` | `27994` | `results/shared_population_branch_audit*`; log `.../shared_branch_analysis_27995.out` | submitted 2026-07-15; pending dependency |
| `28009` | `sbatch --dependency=afterok:27990 scripts/slurm_release_shared_branch.sh` | successful Stage-A GPU array | no research artifact; Slurm control-plane log `.../release_shared_branch_28009.out` | submitted 2026-07-15; releases held `27994` only after Stage A succeeds |
| `28010` | `sbatch --dependency=afterany:27994 scripts/slurm_restore_deprioritized_jobs.sh` | terminal Stage-B GPU array | no research artifact; Slurm control-plane log `.../restore_deprioritized_28010.out` | submitted 2026-07-15; releases held `26508`, `26610`, and `27982` after Stage B ends |
| `28322` | `sbatch scripts/slurm_residual_permutation_null.sh` | none | `results/cem_residual_permutation_null{,_populations,_summary}.{md,csv}`; log `.../residual_permutation_null_28322.out` | completed 2026-07-16, exit 0; 1000 within-population permutations and seed-clustered bootstrap |
| `28324` | `sbatch scripts/slurm_paper_build.sh` | none | rewritten `paper/main.pdf`; log `.../paper_build_28324.out` | completed 2026-07-16, exit 0; 19-page generic article build, no undefined references/citations in final log |
| `28326` | `sbatch scripts/slurm_paper_build.sh` | none | final rewritten `paper/main.pdf`; log `.../paper_build_28326.out` | completed 2026-07-16, exit 0, after appendix provenance/line-break cleanup |

**Priority intervention (2026-07-15):** to prioritize the locked pre-selection
audit `27990`, the pending TRM evaluation remainder `26508_[4-33]`, shared
scaling retry `26610`, ACID renderer smoke test `27982`, and Stage-B follow-up
`27994` were put on a reversible user hold with `scontrol hold`. The already
running TRM tasks `26508_2` and `26508_3` were not cancelled. Stage B will be
released automatically after successful Stage A; the unrelated holds will be
released after Stage B reaches a terminal state.

The user-reported original jobs `26166` and `26400` were cancelled on
2026-07-12; `26481` and `26482` are their resume-safe replacements. Do not
submit duplicates merely because a queued job is waiting on quota.

The two analysis jobs use `afterok` deliberately: a partial/failed upstream
array must not silently generate a headline summary. Phase-H training uses
`afterany` on the three long-running predecessors to avoid deadlocking the
research queue if an unrelated predecessor fails; its own artifacts and split
provenance still determine validity.
