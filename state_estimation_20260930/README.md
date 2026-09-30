# Planning-time state estimation for frozen latent world models

Direction 1 (user-approved 2026-09-30). A latent world model is planned from a start
state built out of observations. Released LeWM evaluators build it from the current
frame only, although the predictors are trained on three frames, and every episode's
first plan has no history at all. This folder measures where that start state limits
planning and builds training-free estimators that use the executed history.

- `se/tasks.py`: released eval settings for reacher / pusht / cube / tworoom.
- `se/policy.py`: `PrefillPolicy` seeds the history buffer with the frames that preceded the start.
- `se/models.py`: innovation feedback, which adds `gamma * nu` to the rollout. `nu` is the newest frame's encoding minus its one-step prediction.
- `scripts/official_eval.py`: upstream evaluator with these switches; the defaults reproduce the release.
- `scripts/pred_error.py`, `scripts/innovation.py`: offline rollout-error decomposition on dataset windows.
- `slurm/*.sh`: batch wrappers. The job ledger is `JOB_LEDGER.md`.

Evidence rules: every arm is paired on the same episodes and seeds. Released-sampling
numbers (min-start 0) are the ones compared to published LeWM results. Prefill arms
use min-start sampling, and the same sampling is applied to every arm they are compared with.
