# Scene learned event-state perception gate

Date locked: 2026-09-04 UTC, after the shared-model factorial PASS and before
observer training.

## Question

Can the event-progress planner retain a material physical-success advantage
when its current automaton state `q` is inferred from the current observation,
rather than supplied by the simulator at every replan?

This is the missing deployment-facing gate after H1b.  The event automaton and
its training labels remain hand specified; only current-state observation is
learned here.

## Models

- `latent`: frozen LeWM-Cube 192-dimensional visual embedding, task id, and
  goal are mapped to current cube/window stage and stable-success state.
- `privileged`: the same observer architecture receives qpos, qvel, and button
  states.  It is an information upper bound, not the deployable method.
- Both observers are trained only on unique canonical roots from the existing
  H1 train split; the seven duplicated root-by-skill rows are deduplicated.
- Train/validation/test reset seeds remain disjoint.  The H1b abstract
  transition checkpoint is frozen and unchanged.

## Evaluation

- Same 16 test resets, skill library, UCT, horizon 4, exploration 0.55, search
  seeds, and decision limits as H1b/H2.
- Budgets K=14,28,56,112; primary K=112.
- At each physical replan the observer predicts current `q` from the newly
  rendered observation or privileged state.  The planner never receives true
  simulator `q`.  True `q` is retained only for evaluation and for advancing
  the physical skill controller's bookkeeping.
- Compare physical success to the shared-model abstract terminal arm and report
  retention relative to the simulator-monitored event arm from H2.
- Report current-q cube/window/stable and exact-match accuracy on states visited
  by each planner, not only on canonical validation roots.

## Locked pilot gate

At K=112, latent observation PASS requires:

1. pooled physical success at least 75%;
2. paired gain over the shared-model terminal arm at least 50 points with
   reset-bootstrap 95% lower bound above zero; and
3. no more than a 25-point loss relative to the simulator-monitored event arm.

If privileged passes but latent fails, verdict is visual representation
ceiling.  If both fail, current `q` is not learnable from a single observation
under this dataset/model and a recurrent belief-state formulation is required.
A positive latent pilot licenses three observer seeds and a 64-reset-per-task
replication; it is not itself paper-level evidence.

## Locked replication after pilot PASS

The pilot passed on 2026-09-04.  Before launching the licensed replication, we
lock the following confirmatory design:

- Freeze the seed-0 abstract transition checkpoint and every planner/skill
  hyperparameter.  Vary only the latent observer initialization over seeds
  0, 1, and 2.
- Evaluate 64 **fresh** resets per task: task 4 uses seeds 84400--84463 and
  task 5 uses 84500--84563.  These do not overlap train, validation, or pilot
  resets.
- Use the primary budget K=112 and run three arms from the identical restored
  physical root: learned latent `q`, simulator-monitored event `q`, and the
  shared-transition terminal-probability arm.  Oracle and terminal arms are
  evaluated once per reset; each latent observer seed is evaluated separately.
- Bootstrap resets, not observer-seed rows.  The aggregate learned statistic
  first averages the three observer outcomes within each reset and then
  resamples the 128 physical resets.

Replication PASS requires all of:

1. every latent observer seed has pooled success at least 75%;
2. mean success over observer seeds is at least 65% on each task;
3. the reset-clustered mean gain over the terminal arm is at least 50 points
   and its 95% bootstrap lower bound is above zero; and
4. no observer seed loses more than 25 points pooled relative to the
   simulator-monitored event arm.

This replication still learns only event-state perception.  The event
automaton, skill vocabulary, and event supervision remain hand specified.

## Replication result (2026-09-04)

The locked replication verdict is
`LEARNED_EVENT_PERCEPTION_REPLICATION_FAIL`.  Mean learned success was 84.11%
over observer seeds, compared with 1.56% for the shared-transition terminal
arm and 88.28% for simulator-monitored event planning.  Three of four checks
passed, but observer seed 1 reached only 95/128 = 74.22% pooled success, one
episode below the locked 75% per-seed threshold.  The threshold is not changed
post hoc.

The failure is structured rather than a broad collapse.  Seed-1 task-5 success
was 35/64 and its visited exact-q accuracy was 65.62%, whereas seed 0 reached
56/64 and seed 2 reached 50/64.  The post-hoc diagnostic found 177 seed-1
visits where true `(cube_stage, window_stage)=(1,2)` was read as `(0,1)`; the
single-frame observer was perfect for its first two decisions and then lost
history.  This motivates a separately labeled, fresh-reset belief-filter gate;
it does not revise this replication verdict.
