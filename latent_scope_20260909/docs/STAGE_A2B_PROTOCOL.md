# Stage A2b: repeat-factor and control-frequency check

A2b tests the timing explanation exposed by job `51840` without rendering or running a
full episode. On episode 0 of each task it compares released state 1 against:

- one, two, and three `env.step` calls at 20 Hz while holding action 0;
- one, two, and three sequential actions at 20 Hz;
- one action at 10 Hz, which applies a delta controller goal once while integrating for
  the released 0.10-second state interval.

The 10 Hz candidate is distinct from repeating a delta action twice at 20 Hz: the latter
can update the relative controller goal twice. The test is CPU-only, uses no cameras, and
is capped at 30 minutes. It does not change Stage A or authorize Stage B.
