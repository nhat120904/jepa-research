# Stage A4 result — jobs 52073 and 52074

## Decision

`A4_LOCAL_TRANSITION_GATE_PASS_WITH_LONG_HORIZON_REPLAY_LIMITATION`.

Both Slurm jobs completed successfully. The raw calibration artifact reports
`A4_REPLAY_FAIL` because the implementation added a bit-exact final-state check after
replaying 1,200--1,350 actions. That extra long-horizon check failed and is retained as a
real limitation. It is not one of the locked A4 local-transition gate conditions; A4 was
introduced specifically because long open-loop replay is unstable.

## Collection

The official GR00T policy produced five successful current-runtime trajectories per task:

- `ScrubCuttingBoard`: 5 successes in 7 attempts;
- `RinseSinkBasin`: 5 successes in 9 attempts.

## Locked local-transition gate

| Condition | Scrub | Rinse | Result |
|---|---:|---:|---|
| Anchors loaded with all three cameras | 8/8 | 8/8 | PASS |
| Fresh-carrier repeated branches exact | 8/8 | 8/8 | PASS |
| Recorded action plus four perturbations produce outcome diversity | 8/8 | 8/8 | PASS |
| Distinct progress labels | 7 | 5 | PASS |

Across 80 repeated-branch comparisons, simulator state and pre/post images were exactly
equal (`max_abs = 0`). Minimum candidate outcome diversity was `0.06765` for Scrub and
`0.05199` for Rinse, both far above the locked `1e-6` threshold.

## Long-horizon limitation

Only 4/5 action replays per task reproduced success, and none reproduced the collected
terminal simulator state bit-exactly. Therefore these data support local one-step
calibration, not claims of exact full-trajectory action replay. Stage B may use the saved
before/action/after branch tuples and native history-derived progress labels, but must not
treat a long open-loop replay as ground truth.

Raw artifacts:

- collection: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/a4_collect_52073/collection_result.json`;
- calibration: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/a4_calibrate_52074/stage_a4_result.json`.
