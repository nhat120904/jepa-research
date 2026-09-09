# Direction A pilot: skill handoff state and time

Status: completed through A2; STOP on AntMaze-medium. See [`RESULTS.md`](RESULTS.md). This protocol does not assert
novelty or a positive result.

## Estimand

With a frozen state-based skill policy `pi(a | s, u)`, does selecting a termination time from the current handoff state
improve task success over the best validated fixed duration and an ordinary reach-the-subgoal termination rule? The
first causal estimand is oracle headroom, not model likelihood.

## Frozen choices

- OGBench 1.2.1; `antmaze-medium-navigate-v0` for development and `antmaze-large-navigate-v0` only after the medium
  gate passes.
- Training and validation NPZ files are separate. The low-level XY-conditioned GCBC skill samples future goals only
  within the same trajectory. Final trajectory rows can be goals but never action targets.
- Full 29-dimensional Ant state is observed. The skill parameter is a desired XY location. Pixels are out of scope.
- Standard OGBench task IDs 1--5, paired reset seeds, 1,000 deployed primitive steps.
- The BFS waypoint proposal rule is shared by every switching arm and is rerun only after an arm exhausts its current
  sequence without success. It is privileged task geometry and therefore makes this a mechanism diagnostic, not a
  deployable system result.
- Candidate durations are 10, 20, 40, and 80 primitive steps. Fixed durations, heuristic termination, and the physics
  oracle use the same frozen policy and waypoint sequence.

## Gates

### A0: skill competence

Evaluate held-out validation states whose future state is 10, 20, 40, or 80 steps away in the same trajectory. Restore
the complete simulator state, execute the frozen skill toward the future XY, and report reach success and minimum XY
distance. A failure here is a low-level-policy/reproduction failure, not evidence against handoff modeling.

Pre-registered screen rule: do not interpret A1 unless success is at least 70% for horizons 10 and 20 and at least 50%
pooled over all four horizons. These are engineering sufficiency thresholds, not paper claims.

### A1: handoff headroom

Compare every fixed duration, `heuristic` (stop on first entry into the 0.5 radius), and `oracle`. All unsuccessful arms
may replan until the same 1,000-step cap, so short fixed durations are not assigned a smaller execution budget. At each handoff the
oracle restores the same physics state, tries each candidate duration, then probes the next skill for 40 steps. It
chooses the candidate with best downstream physical distance. Simulator search steps are recorded separately from
deployed steps.

The direction passes A1 only if oracle success exceeds the strongest non-oracle arm by at least 10 percentage points
on the development screen. Report the paired episode outcomes as well as final distance; do not select the strongest
fixed duration on the confirm episodes.

### A2: simple-baseline kill

The reach-radius heuristic is already included in A1. If it captures the oracle gain, stop. Only if residual headroom
remains should a standard learned state-dependent termination classifier be trained from a common rollout set and
evaluated on disjoint starts.

### A3: joint model (conditional on A0--A2)

Only after A2 passes, collect at most 100,000 additional primitive transitions per task from the frozen policy. Each
record must contain `(s_start, u, b, outcome, tau, s_exit, censored, trajectory_id)`. Compare:

1. fixed-duration transition model;
2. standard option model with learned termination;
3. matched-capacity factorized `p(s_exit | s,u,b) p(tau,outcome | s,u,b)`;
4. joint model `p(tau,outcome | s,u,b) p(s_exit | tau,outcome,s,u,b)`.

Timeouts are right-censored and do not receive a fabricated exit-state target. Held-out likelihood/RMSE are diagnostic;
the decision metric is paired success/return in the same planner.

### A4: confirm

Freeze all choices, use at least five training seeds and held-out reset/goal sets, then test medium plus large. The target
is at least +5 success points over the strongest baseline with a paired interval excluding zero. Otherwise report STOP
or INCONCLUSIVE according to interval width.

## Accounting

Every artifact records dataset/checkpoint SHA-256, Slurm job ID, deployed primitive steps, and privileged simulator-query
steps. Additional online rollouts, reset/branch calls, GPU time, CPU time, storage, and planning latency remain separate
ledger columns. Results from this pilot must not be described as strict-offline OGBench.
