# Job ledger

| UTC | Job ID | Stage | Exact command/config | Dependency | Output | State |
|---|---:|---|---|---|---|---|
| 2026-08-19 07:55 | 43088 | fresh manifest | `sbatch --parsable scripts/slurm_00_manifest.sh`; n=128, seed=20260819, excludes Phase-0d/1a/1av2 | none | `outputs/h0/manifest.json`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_manifest_43088.out` | COMPLETED 0:0 |
| 2026-08-19 07:55 | 43089 | collection smoke | `sbatch --parsable --dependency=afterok:43088 scripts/slurm_01_collect_smoke.sh`; snapshot 1, CEM 96x12, topk10, record {0,11} | afterok:43088 | `outputs/h0/populations_smoke/snapshot_001`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_collect_smoke_43089.out` | COMPLETED 0:0; repeat gate pass |
| 2026-08-19 07:56 | 43090 | collection A | `sbatch --parsable --dependency=afterok:43089 --array=0-63%2 scripts/slurm_01_collect.sh`; indices 0--63 | afterok:43089 | `outputs/h0/populations/snapshot_000..063`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_collect_43090_*.out` | RUNNING |
| 2026-08-19 07:56 | 43091 | training smoke (superseded) | `sbatch --parsable --dependency=afterok:43090 scripts/slurm_02_train_smoke.sh` | afterok:43090 | log path reserved | CANCELLED before start; collection B was required |
| 2026-08-19 07:56 | 43094 | full training (superseded) | `sbatch --parsable --dependency=afterok:43091 scripts/slurm_02_train.sh` | afterok:43091 | log path reserved | CANCELLED before start |
| 2026-08-19 07:56 | 43095 | eval smoke (superseded) | submitted with incorrect dependency while assigning the chain | incorrect | log path reserved | CANCELLED before start |
| 2026-08-19 07:56 | 43096 | eval array (superseded) | submitted with incorrect dependency while assigning the chain | incorrect | log path reserved | CANCELLED before start |
| 2026-08-19 07:56 | 43097 | analysis (superseded) | submitted with incorrect dependency while assigning the chain | incorrect | log path reserved | CANCELLED before start |
| 2026-08-19 07:57 | 43098 | eval smoke (superseded) | `sbatch --parsable --dependency=afterok:43094 scripts/slurm_03_eval_smoke.sh` | afterok:43094 | log path reserved | CANCELLED before start |
| 2026-08-19 07:57 | 43099 | eval array (superseded) | `sbatch --parsable --dependency=afterok:43098 scripts/slurm_03_eval.sh` | afterok:43098 | log path reserved | CANCELLED before start |
| 2026-08-19 07:57 | 43100 | analysis (superseded) | `sbatch --parsable --dependency=afterok:43099 scripts/slurm_04_analyze.sh` | afterok:43099 | log path reserved | CANCELLED before start |
| 2026-08-19 07:59 | 43106 | submit remainder | `sbatch --parsable --dependency=afterok:43090 scripts/slurm_submit_remainder.sh`; submits collection B and the gated train/eval/analysis chain only after collection A frees QOS slots | afterok:43090 | `/mnt/data/nhatnc129/jepa_runs/logs/perd_submit_rest_43106.out` | PENDING |
| 2026-08-19 08:18 | 43106 | submit remainder terminal update | Slurm controller retried the first nested `sbatch` for two minutes | afterok:43090 | same log | FAILED 1:0: `Resource temporarily unavailable`; no nested job was created |
| 2026-08-19 08:25 | 43183 | collection B retry | `sbatch --parsable scripts/slurm_01_collect_second.sh`; array 64--127 `%2` | none; A already complete | `outputs/h0/populations/snapshot_064..127`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_collect_b_43183_*.out` | RUNNING |
| 2026-08-19 08:25 | 43186 | training smoke retry | `sbatch --parsable --dependency=afterok:43183 scripts/slurm_02_train_smoke.sh`; operator/metric, seed11, 1 epoch | afterok:43183 | `outputs/h0/checkpoints_smoke`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_train_smoke_43186.out` | PENDING |
| 2026-08-19 08:25 | 43187 | matched full training retry | `sbatch --parsable --dependency=afterok:43186 scripts/slurm_02_train.sh`; 5 arms x seeds {11,23,47}, max250 epochs | afterok:43186 | `outputs/h0/checkpoints`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_train_43187.out` | PENDING |
| 2026-08-19 08:25 | 43188 | zero-query eval smoke retry | `sbatch --parsable --dependency=afterok:43187 scripts/slurm_03_eval_smoke.sh`; heldout index0, all arms | afterok:43187 | `outputs/h0/eval_smoke`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_eval_smoke_43188.out` | PENDING |
| 2026-08-19 08:25 | 43189 | full held-out eval retry | `sbatch --parsable --dependency=afterok:43188 scripts/slurm_03_eval.sh`; 16-task array, each evaluates test indices `i` and `i+4`, `%2` | afterok:43188 | `outputs/h0/eval/snapshot_*`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_eval_43189_*.out` | PENDING |
| 2026-08-19 08:25 | 43190 | locked paired analysis retry | `sbatch --parsable --dependency=afterok:43189 scripts/slurm_04_analyze.sh`; 10k paired bootstrap | afterok:43189 | `outputs/h0/decision.json`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_analyze_43190.out` | PENDING |
| 2026-08-19 09:35 | 43186 | training smoke terminal update | training loader reached deployable feature construction | afterok:43183 | `/mnt/data/nhatnc129/jepa_runs/logs/perd_train_smoke_43186.out` | FAILED 1:0: NumPy compatibility typo `residual.square()`; no training result consumed |
| 2026-08-19 10:16 | 43277 | corrected training smoke | `sbatch --parsable scripts/slurm_02_train_smoke.sh`; one-line fix to `np.square`, operator/metric seed11, 1 epoch | none; all 128 shards present | `outputs/h0/checkpoints_smoke`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_train_smoke_43277.out` | COMPLETED 0:0 |
| 2026-08-19 10:16 | 43278 | corrected full training | `sbatch --parsable --dependency=afterok:43277 scripts/slurm_02_train.sh`; 5 arms x seeds {11,23,47} | afterok:43277 | `outputs/h0/checkpoints`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_train_43278.out` | RUNNING |
| 2026-08-19 10:16 | 43279 | corrected zero-query eval smoke | `sbatch --parsable --dependency=afterok:43278 scripts/slurm_03_eval_smoke.sh` | afterok:43278 | `outputs/h0/eval_smoke`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_eval_smoke_43279.out` | PENDING |
| 2026-08-19 10:18 | 43281 | resilient eval-chain submitter | `sbatch --parsable --dependency=afterok:43279 scripts/slurm_submit_eval_remainder.sh`; retries controller submission, then queues full eval and analysis | afterok:43279 | `/mnt/data/nhatnc129/jepa_runs/logs/perd_submit_eval_43281.out` | PENDING |
| 2026-08-19 10:20 | 43278 | full training terminal update | all 5 arms x 3 seeds produced checkpoints | afterok:43277 | `outputs/h0/checkpoints/training_summary.json` | COMPLETED 0:0 |
| 2026-08-19 10:20 | 43279 | eval smoke terminal update | first learned-cost CEM call exposed CPU/CUDA normalization-buffer mismatch | afterok:43278 | `/mnt/data/nhatnc129/jepa_runs/logs/perd_eval_smoke_43279.out` | FAILED 1:0; no scientific result consumed |
| 2026-08-19 11:21 | 43291 | corrected eval smoke | `sbatch --parsable scripts/slurm_03_eval_smoke.sh`; normalization buffers moved to checkpoint device | none; trained checkpoints reused | `outputs/h0/eval_smoke`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_eval_smoke_43291.out` | COMPLETED 0:0; repeat gate pass |
| 2026-08-19 11:22 | 43292 | eval-chain submitter retry | `sbatch --parsable --dependency=afterok:43291 scripts/slurm_submit_eval_remainder.sh` | afterok:43291 | `/mnt/data/nhatnc129/jepa_runs/logs/perd_submit_eval_43292.out` | COMPLETED 0:0; created 43293/43294 |
| 2026-08-19 11:22 | 43293 | full held-out zero-query eval | 16-task array; 32 test states, all six objectives, one post-hoc execution per returned plan | afterok:43292 | `outputs/h0/eval/snapshot_*`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_eval_43293_*.out` | COMPLETED 0:0 for all tasks |
| 2026-08-19 11:26 | 43294 | locked H0 analysis | 10k paired state bootstrap and preregistered decision rule | afterok:43293 | `outputs/h0/decision.json`; `/mnt/data/nhatnc129/jepa_runs/logs/perd_analyze_43294.out` | COMPLETED 0:0; `STOP_OPERATOR_NOVELTY` |

The protocol was locked on 2026-08-19 before submission.  Update this table after every `sbatch`,
and verify final state with both `squeue` and `sacct`.

One attempted 128-task array submission and one attempted simultaneous second 64-task array were
rejected by `QOSMaxSubmitJobPerUserLimit` and received no job ID.  Collection is therefore split
into sequential arrays A (0--63) and B (64--127), each with a `%2` GPU cap.  Job 43106 will print
all remainder job IDs in its Slurm log; copy them into this ledger after it runs.
