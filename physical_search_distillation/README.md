# Physical Search Distillation

This folder implements the preregistered H0 pilot in [PROTOCOL.md](PROTOCOL.md).  The simulator is
used to label same-state candidate populations during training only.  Deployment planning is
zero-query: a learned cost is evaluated inside ordinary CEM, and only the returned plan is run.

Pipeline:

1. `slurm_00_manifest.sh`: create a fresh 128-state manifest.
2. `slurm_01_collect.sh`: collect iteration-0/11 populations and exact physical labels.
3. `slurm_02_train.sh`: fit matched pointwise/listwise/elite/operator/operator-metric arms.
4. `slurm_03_eval.sh`: zero-query held-out CEM evaluation.
5. `slurm_04_analyze.sh`: paired bootstrap and locked decision.

Every submission and terminal state is recorded in [JOB_LEDGER.md](JOB_LEDGER.md).
