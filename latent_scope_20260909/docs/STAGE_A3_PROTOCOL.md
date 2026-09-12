# Stage A3: 10 Hz functional replay gate

A3 is the decision gate following A2b. It replays the same first five released episodes
for each selected task at `control_freq=10`, matching the 0.10-second interval between
released simulator states.

The run is camera-free and reads success only from the native reward returned by
`env.step`. It records every change to the task-owned history fields and summarizes
released-state errors by component and quantile. It does not call `_check_success`, change
the task success predicates, test a learned model, or render images.

Verdicts:

- `A3_FUNCTIONAL_PASS`: all 10 episodes reach native success and both tasks exhibit their
  expected history changes;
- `A3_HISTORY_SIGNAL_FAIL`: all episodes succeed but at least one task has no history
  change;
- `A3_FUNCTIONAL_FAIL`: at least one episode does not reach native success.

A functional pass does not silently waive state divergence. It permits a separate,
documented decision about the Stage-A protocol's predeclared amendment rule. A failure
blocks Stage B and triggers runtime-compatibility or transition-recollection work.

The job is CPU-only, uses no cameras, and is capped at 45 minutes.
