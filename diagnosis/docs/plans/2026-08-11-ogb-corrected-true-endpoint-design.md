# Corrected OGBench true-endpoint audit

Date: 2026-08-11. Status: **complete; all gates passed, strong support**.

## Question

The original Stage-0 audit claimed that replacing LeWM's predicted terminal
embedding by an embedding of the simulator's true terminal image left residual
physical selection regret. That decomposition is retracted because the old
candidate evaluator called `CubeEnv.reset()` before every rollout. Goal-image
construction inside that reset executes random actions and restores only
`qpos/qvel`, so controller and contact-solver state leaked between rollouts.

This audit asks the narrower corrected question:

> On the exact persisted final CEM populations, does a frozen LeWM encoder rank
> reproducibly rendered true endpoints in a way that leaves physical selection
> regret after dynamics prediction error has been removed?

This is a diagnostic only. Simulator state and images are never available to a
deployed selector or proposed method.

## Fixed inputs

- checkpoint: `quentinll/lewm-cube`;
- dataset: `ogbench/cube_single_expert.h5`;
- the 32-snapshot Stage-0 manifest;
- the 300 persisted final CEM candidates per snapshot;
- goal offset 25, horizon 5, action block 5;
- success threshold 4 cm;
- snapshot is the unit of uncertainty; final intervals use 10,000 paired
  snapshot-bootstrap draws.

No CEM search, action normalization, candidate set, checkpoint, snapshot, or
goal image is regenerated.

## Corrected replay and renderer protocol

Each audit process creates two independent `World`/`MjModel` instances. Each is
reset exactly once with visual variation disabled, after which camera, light,
material, texture, and geometry arrays are hashed and must agree across the two
instances. MuJoCo solver warm-start is disabled.

Before every state render or candidate rollout the evaluator:

1. calls `mujoco.mj_resetData`;
2. restores the dataset `qpos/qvel` (or the future row's `goal_qpos/goal_qvel`);
3. restores the future cube pose as the target;
4. calls `mujoco.mj_forward`;
5. aligns the environment's finite-difference previous-state fields;
6. calls `post_step`.

The two independently compiled worlds replay all 300 action sequences. Physical
distance must agree within `1e-5` m; success and executed-step arrays must agree
exactly. Rendered initial, goal, and candidate-endpoint pixels must agree exactly
(`uint8` maximum absolute difference zero). Frozen-encoder endpoint costs must
agree within `1e-5`, and the selected candidate must agree exactly. Any failure
invalidates that shard.

## Two true-endpoint arms

Both arms use the same corrected endpoint image and differ only in the goal
image:

- `true_dataset_goal`: endpoint rendered by the evaluator, goal image from the
  dataset. This reproduces the semantic quantity attempted by historical
  Stage 0, but crosses renderer domains.
- `true_rendered_goal`: endpoint and exact future goal state rendered by the
  same locked evaluator. This removes renderer-domain offset from the cost and
  is the primary representation diagnostic.

The learned predicted-endpoint L2 selector and the physical best-in-population
candidate are retained as reference arms.

## Precommitted domain-match gate

Rendering can be repeatable while still being out of domain for the released
encoder. For the exact initial and future goal states define squared encoder
distances

```
delta_init = d(rendered_init, dataset_init)
delta_goal = d(rendered_goal, dataset_goal)
task_data  = d(dataset_init, dataset_goal)
task_sim   = d(rendered_init, rendered_goal)
domain_ratio = max(delta_init, delta_goal) / min(task_data, task_sim)
```

with a denominator floor of `1e-12`. A snapshot passes only if:

1. `domain_ratio <= 0.25`;
2. rendered initial state is closer to dataset initial than dataset goal;
3. rendered goal state is closer to dataset goal than dataset initial.

The 0.25 threshold is fixed before the smoke jobs: a same-state renderer shift
larger than one quarter of the task separation is not treated as negligible for
candidate ranking. Pixel dataset-vs-render differences are reported but are not
an absolute gate because EGL rasterization can differ across hardware.

## Staged execution and stop rules

1. Run snapshots 3, 6, and 9, the hard cases that exposed the old dynamic reset
   failure.
2. Continue to locked `n=32` only if all three pass physical, renderer,
   encoder-repeat, visual-signature, and domain-match gates.
3. If any smoke domain gate fails, stop. The historical 21/32 success and
   3.13 cm residual representation regret remain retracted; no full array is
   licensed.
4. If smoke passes, run the unmodified script as a 32-task Slurm array. The
   aggregate is valid only with all 32 shards and all gates passing.

## Interpretation boundary

A positive result would show a residual failure of this released encoder plus
terminal squared-L2 cost on one OGBench checkpoint and task. It would not show
that representation learning in general is broken, that scaling cannot help,
or that simulator-assisted selection is a fair deployable method. A negative or
blocked result removes OGBench true-endpoint representation evidence; it does
not erase the separately reproduced learned-cost selection failure and
candidate-coverage result under the physical evaluator.

## Execution ledger

Jobs, exact commands, dependencies, outputs, hashes, and terminal states are
recorded here as they are submitted.

| Job | Command | Dependency | Output | State |
|---|---|---|---|---|
| 38298 | `sbatch --export=ALL,TRUE_ENDPOINT_SNAPSHOT_INDEX=3,TRUE_ENDPOINT_OUT_TAG=smoke_3 scripts/slurm_ogb_true_endpoint_smoke.sh` | none | `diagnosis/results/ogb_true_endpoint_corrected/smoke_3/`; log `/mnt/data/nhatnc129/jepa_runs/logs/ogb_true_endpoint_smoke_38298.out` | **COMPLETED** in 1:45; all gates pass, domain ratio `1.8877e-4`, repeat errors zero |
| 38297 | `sbatch --export=ALL,TRUE_ENDPOINT_SNAPSHOT_INDEX=6,TRUE_ENDPOINT_OUT_TAG=smoke_6 scripts/slurm_ogb_true_endpoint_smoke.sh` | none | `diagnosis/results/ogb_true_endpoint_corrected/smoke_6/`; log `/mnt/data/nhatnc129/jepa_runs/logs/ogb_true_endpoint_smoke_38297.out` | **COMPLETED** in 1:48; all gates pass, domain ratio `1.4257e-4`, repeat errors zero |
| 38296 | `sbatch --export=ALL,TRUE_ENDPOINT_SNAPSHOT_INDEX=9,TRUE_ENDPOINT_OUT_TAG=smoke_9 scripts/slurm_ogb_true_endpoint_smoke.sh` | none | `diagnosis/results/ogb_true_endpoint_corrected/smoke_9/`; log `/mnt/data/nhatnc129/jepa_runs/logs/ogb_true_endpoint_smoke_38296.out` | **COMPLETED** in 1:48; all gates pass, domain ratio `7.8242e-5`, repeat errors zero |
| 38299_[0-31] | `sbatch --dependency=afterok:38296:38297:38298 scripts/slurm_ogb_true_endpoint_array.sh` | all three locked-gate smokes must succeed | `diagnosis/results/ogb_true_endpoint_corrected/locked_shards/$TASK_ID/`; logs `/mnt/data/nhatnc129/jepa_runs/logs/ogb_true_endpoint_array_38299_%a.out` | **COMPLETED**; 32/32 exit 0 and all locked gates pass |
| 38300 | `sbatch --dependency=afterok:38299 scripts/slurm_ogb_true_endpoint_aggregate.sh` | all 32 locked shards must succeed | `diagnosis/results/ogb_true_endpoint_corrected/locked/`; log `/mnt/data/nhatnc129/jepa_runs/logs/ogb_true_endpoint_aggregate_38300.out` | **COMPLETED**; `strong_support` after 10,000 bootstrap draws |

## Final result

All 32 array tasks completed with exit code 0 and every shard passed every
locked gate. Worst repeat error was zero for physical outcomes, endpoint
pixels, true-endpoint costs, and reference-evaluator reproduction. The worst
domain ratio was `6.5906e-4` against the locked 0.25 maximum.

The primary same-renderer true-endpoint arm succeeds on 21/32 snapshots and
has mean physical regret 3.286 cm [1.495, 5.672]. Physical-oracle candidate
availability is 25/32, leaving a paired success gap of 4/32 [1/32, 8/32]. The
representation gate is `strong_support`. Full interpretation and hashes are in
`../../results/ogb_true_endpoint_corrected/TRUE_ENDPOINT_DECISION.md`.
