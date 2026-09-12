# Stage A2: replay-alignment diagnostic

Stage A2 is a bounded execution diagnosis triggered by Stage A job `51835`. It does not
train a model or change any Stage A tolerance.

For episode 0 of each selected task, the diagnostic:

1. verifies exact initial and direct next-state loading;
2. applies the official reordered action at index 0 and compares it with released states
   0 through 3, with errors split into time, `qpos`, and `qvel`;
3. repeats from a fresh reset with the raw LeRobot action ordering, action index 1, and a
   zero-action control;
4. records dataset/runtime versions, control frequency, action bounds, and composite
   controller action slices.

The alternatives distinguish a direct state-carrier failure, LeRobot-to-robosuite action
ordering error, one-step action/state indexing error, and a residual open-loop dynamics or
release-compatibility mismatch. No images are rendered. The job requests CPU only and is
capped at 30 minutes.
