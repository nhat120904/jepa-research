# Stage A2b result: timing is real but not a complete replay fix

Slurm job `51843` completed successfully on 10 September 2026 in 1 minute 39 seconds.
Its verdict is `A2B_TIMING_NOT_GENERAL`.

## Result

| Task | Candidate | Timestamp error | Physical max error | Interpretation |
|---|---|---:|---:|---|
| ScrubCuttingBoard | 20 Hz, action 0 once | 0.05 | 0.1958233 | Stage A baseline |
| ScrubCuttingBoard | **10 Hz, action 0 once** | **0** | **0.0078367** | Strong improvement, still fails `1e-5` |
| ScrubCuttingBoard | 20 Hz, hold action 0 twice | 0 | 0.0219534 | Worse than 10 Hz |
| RinseSinkBasin | 20 Hz, action 0 once | 0.05 | **0.0045254** | Best physical error but wrong timestamp |
| RinseSinkBasin | **10 Hz, action 0 once** | **0** | 0.0076783 | Correct timestamp, slightly worse physical error |
| RinseSinkBasin | 20 Hz, hold action 0 twice | 0 | 0.0217990 | Worse than either one-step option |

Sequential action variants were worse. Therefore neither repeating action 0 twice nor
advancing to action 1 on the second 20 Hz step reproduces released state 1. Setting the
environment to 10 Hz is the semantically cleaner timing correction because the delta
controller receives the command once, but it does not recover exact released dynamics.

## Decision

Do not change Stage A to a two-step action repeat and do not start Stage B. The next useful
gate is not another one-step permutation. Run a camera-free 10 Hz native replay across the
same ten episodes and measure task success plus the full history trajectory. If 10 Hz
recovers all native successes and the expected history events, the strict state-equality
gate can be reconsidered using the protocol's predeclared amendment rule. If it does not,
the recorded RoboCasa `env_version=0.5.1` versus installed `1.0.1` compatibility gap must
be resolved before model work.

Full result:
`/mnt/data/nhatnc129/jepa/latent_scope_stage_a/outputs/a2b_repeat_51843/stage_a2b_result.json`.
