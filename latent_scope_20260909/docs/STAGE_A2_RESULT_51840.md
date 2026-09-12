# Stage A2 result: likely 2x action-repeat mismatch

Slurm job `51840` completed successfully on 10 September 2026 in 1 minute 20 seconds.
The machine verdict is `A2_OPEN_LOOP_INCOMPATIBLE`, but the component-wise measurements
localize a concrete leading explanation that the initial classifier did not test: the
released state interval is twice the current environment control interval.

## Findings

| Check | ScrubCuttingBoard | RinseSinkBasin | Interpretation |
|---|---:|---:|---|
| Initial state load max error | 0 | 0 | Initial reset is exact |
| Direct released state 1 load max error | 0 | 0 | State carrier and indexing are valid |
| Released state 0 to 1 time delta | 0.10 s | 0.10 s | Recorded state interval |
| One current `env.step` time delta | 0.05 s | 0.05 s | Current control interval |
| Canonical action 0 error to state 1 | 0.195823 | 0.05 | One step is temporally short |
| Raw LeRobot action 0 error to state 1 | 0.388914 | 0.389260 | Raw ordering is worse; keep the official reorder |

Using canonical action index 1 instead of index 0 did not materially fix the target-state
error. Thus neither the official LeRobot-to-robosuite reorder nor a simple +1 action index
shift explains the Stage A failure.

The runtime's robosuite `1.5.2` and MuJoCo `3.3.1` versions match the dataset metadata.
The metadata records environment version `0.5.1`, while the installed RoboCasa package is
`1.0.1`; this remains a secondary compatibility risk, especially for the residual Scrub
velocity error.

## Decision

Do not reinterpret Stage A as a scientific failure and do not start Stage B. First run a
minimal A2b probe that compares repeat factors 1, 2, and 3 from the same released state,
including both holding action 0 twice and applying actions 0 then 1. If repeat factor 2
matches state 1, amend the replay convention explicitly and rerun Stage A. If it does not,
investigate the RoboCasa `0.5.1` versus `1.0.1` environment compatibility next.

Machine-readable result:
`/mnt/data/nhatnc129/jepa/latent_scope_stage_a/outputs/a2_alignment_51840/stage_a2_result.json`.
