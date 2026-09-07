# Job ledger

All simulation, model, training, and result analysis runs are submitted to
Slurm compute nodes.  Login-node activity is limited to inspection, editing,
syntax checks, submission, and reading small summary metadata.

| Stage | Job ID | Exact submission | Dependency | Outputs | State |
|---|---:|---|---|---|---|
| Gate 1 smoke | 48231 | `sbatch moment_wm_h0/scripts/slurm_gate1_smoke.sh` | none | `moment_wm_h0/outputs/gate1_smoke/{summary.json,episodes.csv,actions.pt}` | COMPLETED, 00:00:06, exit 0:0; technical smoke GO |
| Gate 1 full | 48232 | `sbatch moment_wm_h0/scripts/slurm_gate1_full.sh` | smoke job 48231 completed; logical dependency, submitted after validation | `moment_wm_h0/outputs/gate1_full/{summary.json,episodes.csv,actions.pt}` | COMPLETED, 00:00:07, exit 0:0; GO (relative cost reduction 0.793, 95% CI [0.745, 0.837]) |
| Gate 2 smoke v1 | 48233 | `sbatch moment_wm_h0/scripts/slurm_gate2_smoke.sh` | Gate 1 GO (48232) | `moment_wm_h0/outputs/gate2_smoke/` | FAILED, 00:00:17, exit 1:0; RAFT requires >=128px input, observed 64px |
| Gate 2 smoke v2 | 48234 | `sbatch moment_wm_h0/scripts/slurm_gate2_smoke.sh` | code fix: resize native frames to RAFT-required 128px | `moment_wm_h0/outputs/gate2_smoke/` | COMPLETED, 00:00:18, exit 0:0; end-to-end technical validation |
| Gate 2 data | 48235 | `sbatch moment_wm_h0/scripts/slurm_gate2_generate.sh` | Gate 2 smoke 48234 completed | `moment_wm_h0/outputs/gate2_full/glides.{pt,json}` | COMPLETED, 00:00:05, exit 0:0 |
| Gate 2 anchors | 48236 | `sbatch --dependency=afterok:48235 moment_wm_h0/scripts/slurm_gate2_extract.sh` | afterok:48235 | `moment_wm_h0/outputs/gate2_full/anchors.{pt,json}` | COMPLETED, 00:01:20, exit 0:0 |
| Gate 2 probes | 48237 | `sbatch --dependency=afterok:48236 moment_wm_h0/scripts/slurm_gate2_fit.sh` | afterok:48236 | `moment_wm_h0/outputs/gate2_full/certificates/` | COMPLETED, 00:02:14, exit 0:0; recurrent-only executed hash `8b779f...` from wrapper |
| Gate 2 finalize v1 | 48238 | `sbatch --dependency=afterok:48237 moment_wm_h0/scripts/slurm_gate2_finalize.sh` | afterok:48237; adds analytic pixel-centroid certificate because 48237 started on the prior recurrent-only hash | `moment_wm_h0/outputs/gate2_full/certificates/summary.json` | COMPLETED, 00:00:03, exit 0:0; Gate 2 GO |
| Gate 2 finalize provenance rerun | 48239 | `sbatch moment_wm_h0/scripts/slurm_gate2_finalize.sh` | all Gate 2 jobs completed; idempotent correction of parent executed hash | same as above, originals preserved | COMPLETED, 00:00:03, exit 0:0 |
| Gate 3 smoke v1 | 48240 | `sbatch moment_wm_h0/scripts/slurm_gate3_smoke.sh` | Gate 2 GO | `moment_wm_h0/outputs/gate3_smoke/` | COMPLETED, 00:00:32, exit 0:0; technical path passed, but smoke reference tasks mismatched Gate-1 prefix due batch-size-dependent RNG |
| Gate 3 smoke v2 | 48241 | `sbatch moment_wm_h0/scripts/slurm_gate3_smoke.sh` | fix: generate locked n=128 task batch before smoke slicing | `moment_wm_h0/outputs/gate3_smoke/` | COMPLETED, 00:00:30, exit 0:0; clean end-to-end validation |
| Gate 3 data | 48242 | `sbatch moment_wm_h0/scripts/slurm_gate3_generate.sh` | Gate 3 smoke 48241 completed | `moment_wm_h0/outputs/gate3_full/control.{pt,json}` | COMPLETED, 00:00:10, exit 0:0 |
| Gate 3 features | 48243 | `sbatch --dependency=afterok:48242 moment_wm_h0/scripts/slurm_gate3_prepare.sh` | afterok:48242 | `moment_wm_h0/outputs/gate3_full/features.{pt,json}` | COMPLETED, 00:03:03, exit 0:0; decoder test R2 0.859 |
| Gate 3 baseline array | 48244 | `sbatch --dependency=afterok:48243 moment_wm_h0/scripts/slurm_gate3_train_array.sh` | afterok:48243 | `moment_wm_h0/outputs/gate3_full/models/` | COMPLETED, all 9 tasks exit 0:0 (00:00:39--00:00:46) |
| Gate 3 evaluation | 48245 | `sbatch --dependency=afterok:48244 moment_wm_h0/scripts/slurm_gate3_evaluate.sh` | afterok:48244 (entire array) | `moment_wm_h0/outputs/gate3_full/evaluation/` | COMPLETED, 00:00:04, exit 0:0; CONTINUE_TO_MMR, best MSE recovers 0.149 oracle gap |
| Gate 4 smoke | 48255 | `sbatch moment_wm_h0/scripts/slurm_gate4_smoke.sh` | Gate 3 CONTINUE_TO_MMR | `moment_wm_h0/outputs/gate4_smoke/` | COMPLETED, 00:00:10, exit 0:0; end-to-end technical validation |
| Gate 4 MMR array | 48256 | `sbatch moment_wm_h0/scripts/slurm_gate4_train_array.sh` | Gate 4 smoke 48255 completed | `moment_wm_h0/outputs/gate4_full/models/` | COMPLETED, all 9 tasks exit 0:0 (00:00:47--00:00:50) |
| Gate 4 lambda selection | 48257 | `sbatch --dependency=afterok:48256 moment_wm_h0/scripts/slurm_gate4_select.sh` | afterok:48256 (all nine MMR runs) | `moment_wm_h0/outputs/gate4_full/selection.json` | COMPLETED, 00:00:01, exit 0:0; validation selected lambda 1.0 |
| Gate 4 evaluation | 48258 | `sbatch --dependency=afterok:48257 moment_wm_h0/scripts/slurm_gate4_evaluate.sh` | afterok:48257, validation-only lambda selection | `moment_wm_h0/outputs/gate4_full/evaluation/` | COMPLETED, 00:00:04, exit 0:0; STOP (both preregistered improvement criteria failed) |

## Decision chain

1. Gate 1 full must return `GO` before any anchor/model training is submitted.
2. On `STOP`, the method branch terminates and the reason is recorded here.
3. A passing Gate 1 licenses the anchor recoverability certificate next; it
   does not license a positive method claim.
4. Gate 4 returned `STOP`; no larger-dataset or robot-world-model extension was
   submitted for this branch.
