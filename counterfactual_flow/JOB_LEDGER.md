# Counterfactual-flow job ledger

## Phase 0 — OGBench-Cube same-state mining

| Job | Exact command | Dependency | Output | State |
|---|---|---|---|---|
| `41150` | `sbatch counterfactual_flow/scripts/slurm_phase0_smoke.sh` | none | `counterfactual_flow/outputs/ogbench_cube_phase0/smoke_0/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_smoke_41150.out` | `COMPLETED`, exit `0:0`, `00:01:26` |
| `41151_[0-31]` | `sbatch --dependency=afterok:41150 counterfactual_flow/scripts/slurm_phase0_array.sh` | smoke pass | `counterfactual_flow/outputs/ogbench_cube_phase0/locked_shards/{0..31}/`; logs `/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_mine_41151_%a.out` | all 32 tasks `COMPLETED`, exit `0:0` |
| `41152` | `sbatch --dependency=afterok:41151 counterfactual_flow/scripts/slurm_phase0_aggregate.sh` | all 32 array tasks pass | `counterfactual_flow/outputs/ogbench_cube_phase0/locked/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_aggregate_41152.out` | `COMPLETED`, exit `0:0`, `00:00:00` |

The Phase-0 executor imports the full-state reset routine from
`diagnosis/scripts/76_ogb_true_endpoint_corrected.py`.  It is an exploratory
dataset-construction gate and should not be cited as policy evidence.

Terminal states were verified with both `squeue` and `sacct` on 2026-08-16.
The locked aggregate reports `GO`: 28/32 snapshots contain a low-proxy-cost,
high-physical-regret candidate and 27 have a regret-matched control within
1 cm.  This is an availability gate for candidate pools, not evidence that
ordinal-inversion acquisition improves a trained policy.

## Phase 0b — same-population pairwise inversion verification

| Job | Exact command | Dependency | Output | State |
|---|---|---|---|---|
| `41484` | `sbatch counterfactual_flow/scripts/slurm_phase0b_inversions.sh` | Phase-0 locked shards available | `counterfactual_flow/outputs/ogbench_cube_phase0/phase0b/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_phase0b_41484.out` | `COMPLETED`, exit `0:0`, `00:00:01` |
| `41485` | `sbatch counterfactual_flow/scripts/slurm_phase0b_inversions.sh` | Phase-0 locked shards available | same output; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_phase0b_41485.out` | `COMPLETED`, exit `0:0`, `00:00:01`; terminology-only rerun replacing “non-inverted control” with “proxy-rejected hard control” |

The verifier uses rank margins (so proxy-cost units do not affect the score),
requires a 2 cm same-population physical reversal, and matches controls within
1 cm.  Its gate is `GO`: 26/32 final-CEM populations and 28/32 initial
populations contain a verified inversion; 25/32 final populations also contain
a same-population, regret-matched proxy-rejected hard control.

## Phase 0c — initial-to-final population selection enrichment

| Job | Exact command | Dependency | Output | State |
|---|---|---|---|---|
| `41511` | `sbatch counterfactual_flow/scripts/slurm_phase0c_selection_enrichment.sh` | Phase-0 locked shards available | `counterfactual_flow/outputs/ogbench_cube_phase0/phase0c/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_phase0c_41511.out` | `FAILED`, exit `1:0`, `00:00:01`: undefined all-tied-population inversion fraction was not filtered before bootstrap |
| `41512` | `sbatch counterfactual_flow/scripts/slurm_phase0c_selection_enrichment.sh` | Phase-0 locked shards available | same output; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_phase0c_41512.out` | `COMPLETED`, exit `0:0`, `00:00:01` after finite-snapshot bootstrap fix |

## Phase 0d — exact CEM-returned-plan audit

| Job | Exact command | Dependency | Output | State |
|---|---|---|---|---|
| `42047` | `sbatch counterfactual_flow/scripts/slurm_phase0d_smoke.sh` | none | `counterfactual_flow/outputs/ogbench_cube_phase0/phase0d_smoke/0/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_p0d_smoke_42047.out` | `COMPLETED`, exit `0:0`, `00:00:27`; returned plan replay passed exactly |
| `42048_[0-31]` | `sbatch --dependency=afterok:42047 counterfactual_flow/scripts/slurm_phase0d_array.sh` | successful Phase-0d smoke | `counterfactual_flow/outputs/ogbench_cube_phase0/phase0d_shards/{0..31}/`; logs `/mnt/data/nhatnc129/jepa_runs/logs/cf_p0d_audit_42048_%a.out` | all 32 tasks `COMPLETED`, exit `0:0`, 43–61 sec/task |
| `42049` | `sbatch --dependency=afterok:42048 counterfactual_flow/scripts/slurm_phase0d_aggregate.sh` | successful completion of all array tasks | `counterfactual_flow/outputs/ogbench_cube_phase0/phase0d/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_p0d_aggregate_42049.out` | `COMPLETED`, exit `0:0`, `00:00:00`; `GO_MINIMAL_POLICY_PILOT` |

Phase 0d is deliberately isolated in `counterfactual_flow/`. It records the
actual `CEMSolver.solve(...)["actions"]` final elite-refitted mean, replaying
that plan twice from a complete MuJoCo reset. The analysis compares its physical
outcome with final-population alternatives. It passed: returned-plan physical
selection regret is 7.05 cm (95% bootstrap CI [4.24, 10.12]) and the
proxy-rejected-corrective rate is 50.0% ([34.4, 65.6]). This is the mechanism
gate that authorizes the minimal matched-budget BC pilot, not policy evidence.

## Phase 1a — locked, matched-budget acquisition test

| Job | Exact command | Dependency | Output | State |
|---|---|---|---|---|
| `42231` | `sbatch counterfactual_flow/scripts/slurm_phase1a_manifest.sh` | none | `counterfactual_flow/outputs/ogbench_cube_phase1a/manifest.json`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_p1a_manifest_42231.out` | `COMPLETED`, exit `0:0`, `00:00:08`; exactly 128 episodes, disjoint from the 32 Phase-0d audit episodes |
| `42232` | `sbatch --dependency=afterok:42231 counterfactual_flow/scripts/slurm_phase1a_smoke.sh` | successful manifest | `counterfactual_flow/outputs/ogbench_cube_phase1a/smoke/0/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_p1a_smoke_42232.out` | `COMPLETED`, exit `0:0`, `00:00:31`; selection locked pre-physics and exact repeat gate passed |

Slurm rejected the first early 128-task array submission with
`QOSMaxSubmitJobPerUserLimit`; no array job or physical rollout was created by
that rejected command. The submitted full run uses 32 tasks, each handling four
snapshots sequentially, so the scientific 128-snapshot protocol is unchanged.

| `42234_[0-31]` | `sbatch counterfactual_flow/scripts/slurm_phase1a_array.sh` | successful manifest and smoke | `counterfactual_flow/outputs/ogbench_cube_phase1a/shards/{0..127}/`; logs `/mnt/data/nhatnc129/jepa_runs/logs/cf_p1a_acquire_42234_%a.out` | all 32 grouped tasks `COMPLETED`, exit `0:0`, 77–100 sec/task; all 128 snapshot shards present |
| `42237` | `sbatch --dependency=afterok:42234 counterfactual_flow/scripts/slurm_phase1a_aggregate.sh` | successful completion of all grouped array tasks | `counterfactual_flow/outputs/ogbench_cube_phase1a/aggregate/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_p1a_aggregate_42237.out` | `COMPLETED`, exit `0:0`, `00:00:01`; verdict `HOLD_NO_CI_CLEAN_ACQUISITION_GAIN_VS_RANDOM` |

## Phase 1a-v2 — proxy-instability/disagreement fresh replication

| Job | Exact command | Dependency | Output | State |
|---|---|---|---|---|
| `42314` | `sbatch counterfactual_flow/scripts/slurm_phase1av2_manifest.sh` | completed Phase 1a | `counterfactual_flow/outputs/ogbench_cube_phase1av2/manifest.json`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_p1av2_manifest_42314.out` | `COMPLETED`, exit `0:0`, `00:00:07`; 128 episodes disjoint from all 160 prior audit/acquisition episodes |
| `42315` | `sbatch --dependency=afterok:42314 counterfactual_flow/scripts/slurm_phase1av2_smoke.sh` | successful fresh manifest | `counterfactual_flow/outputs/ogbench_cube_phase1av2/smoke/0/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_p1av2_smoke_42315.out` | `COMPLETED`, exit `0:0`, `00:00:23`; model-only scoring and exact physical replay passed |
| `42316_[0-31]` | `sbatch counterfactual_flow/scripts/slurm_phase1av2_array.sh` | successful manifest and smoke | `counterfactual_flow/outputs/ogbench_cube_phase1av2/shards/{0..127}/`; logs `/mnt/data/nhatnc129/jepa_runs/logs/cf_p1av2_acquire_42316_%a.out` | all 32 grouped tasks `COMPLETED`, exit `0:0`, 76–104 sec/task; all 128 shards present |
| `42319` | `sbatch --dependency=afterok:42316 counterfactual_flow/scripts/slurm_phase1av2_aggregate.sh` | all grouped array tasks pass | `counterfactual_flow/outputs/ogbench_cube_phase1av2/aggregate/`; log `/mnt/data/nhatnc129/jepa_runs/logs/cf_p1av2_aggregate_42319.out` | `COMPLETED`, exit `0:0`, `00:00:01`; verdict `STOP_NO_ROBUST_MODEL_SIGNAL_BEYOND_DIVERSITY` |
