# Stage A3 result: 10 Hz functional replay failed

Slurm job `51847` completed successfully on 10 September 2026 in 6 minutes 57 seconds.
The experimental verdict is `A3_FUNCTIONAL_FAIL`.

## Functional result

| Task | Native successes | Episodes with history changes |
|---|---:|---:|
| ScrubCuttingBoard | 0/5 | 0/5 |
| RinseSinkBasin | 0/5 | 0/5 |
| **Total** | **0/10** | **0/10** |

Every episode diverged from its released state trajectory at step 0. Across the full
rollouts, the 10 Hz clock accumulated between 16.2 and 30.2 seconds of error, depending on
episode length. The 0.10-second gap between released states 0 and 1 that motivated A2b is
therefore an initial-transition anomaly, not the trajectory-wide control interval.

This also explains why the one-step 10 Hz probe looked promising while the full replay
failed completely. The 20 Hz Stage A run retained 5/10 successes and history changes for
both tasks; switching the entire environment to 10 Hz made functional reproduction worse.

## Decision

Reject both proposed timing fixes: do not repeat every action twice at 20 Hz and do not
run the trajectories at 10 Hz. Do not weaken the state tolerance or start Stage B.

The current released demonstrations were recorded with `env_version=0.5.1`, while the
installed public RoboCasa runtime reports `1.0.1`. The next branch is now compatibility,
not further action-index permutations:

1. recover an exact or explicitly supported runtime for the recorded `0.5.1` environment;
2. if that runtime is unavailable, recollect task transitions under the current `1.0.1`
   runtime while retaining the native tasks and success predicates;
3. rerun the execution gate before training any world model.

This failure invalidates the current execution substrate only. No compositional JEPA model
has been trained or evaluated, so it is not evidence that the research idea fails.

Full result:
`/mnt/data/nhatnc129/jepa/latent_scope_stage_a/outputs/a3_replay_51847/stage_a3_result.json`.
