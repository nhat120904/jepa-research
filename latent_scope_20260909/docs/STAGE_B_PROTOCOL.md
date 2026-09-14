# Stage B: component-headroom protocol

## Question

Before training a new JEPA, determine whether choosing among eight action chunks from a
common frozen proposal can improve eventual native task completion, and whether the
candidate outcomes contain visual/history variation that a learned selector could in
principle exploit.

This is a one-intervention oracle experiment. It is not repeated oracle MPC and it is not
evidence for the proposed compositional target by itself.

## B0 profiling gate

Run two independent canonical prefixes per task at half the native horizon. At each prefix:

1. sample eight eight-step chunks from the same frozen GR00T policy using locked proposal
   seeds;
2. reconstruct the canonical prefix from its current-runtime XML, state and action prefix;
3. execute one candidate chunk, then use the same frozen continuation policy to the native
   horizon;
4. reset the continuation RNG to the same seed for all candidates from a prefix;
5. record eventual native completion, progress history, candidate post-state diversity,
   three-camera before/post/final observations and runtime.

Every candidate branch must reproduce the canonical pre-intervention simulator state and
task history. Long-horizon equality to the original source rollout is neither assumed nor
required; all candidates are compared from the same reconstructed canonical carrier.

B0 is capped at four GPU hours. It passes operationally only if all branches finish, prefix
reconstruction is exact, all camera keys are present, and candidate chunks create state
diversity above `1e-6`. The small profile estimates cost and catches integration failures;
its success-rate difference is descriptive and cannot accept or reject scientific
headroom.

### B0.1 event-aligned repair

The fixed-half-horizon B0 can be non-diagnostic when the task is still untouched or its
latched milestones are already complete. In that case, run one bounded repair profile per
task. Generate a separate successful source rollout, identify its last increase in native
nonterminal progress between 10% and 90% of the horizon, and branch immediately before
that event. Candidate seeds remain independent of the source rollout; the source action is
not inserted into the candidate set.

The event carrier must be restored directly from the source simulator snapshot together
with the task-owned history fields. Simulator state and task history must match exactly,
and all reconstructed candidate branches must render identical canonical observations.
Source-rollout screenshots are diagnostic rather than a gate: an independent five-restore
calibration showed pixel-identical cross-restores, while two later unseen prefixes exceeded
the predeclared source-to-restore MAE tolerance despite exact physical state/history, before
any candidate was evaluated. Candidate proposals and outcomes therefore both use the same
freshly rendered canonical restored observation; source-to-restore image metrics remain in
the artifact and cannot be used to filter prefixes after outcomes are observed.

Report two distinct ceilings: eventual native success and maximum native progress reached
under the common continuation. For Scrub, normalized progress is half capped contact
milestone and half sweep milestone. For Rinse, it is washed-region count divided by three.
This profile may establish that task-relevant alternatives exist, but its event-conditioned
sampling is an availability diagnostic, not an unbiased native-success estimate for B1.

### B0.2 nested action-horizon check

At each exact event carrier, draw one eight-candidate bank of the native 16-step GR00T
output. Evaluate the same candidates after the first 8 actions and after all 16 actions;
the 8-step sequence must therefore be an exact prefix of the 16-step sequence. Continue
with a fresh GR00T query every 8 native steps in both conditions. This isolates intervention
horizon from candidate identity and continuation cadence.

## B1 screening experiment

If B0 fits the locked budget, expand to one validation prefix from each of 40 independent
episodes per task and the same eight candidates. Candidate 0 is the predeclared behaviour
choice. The availability ceiling is whether any of eight candidates succeeds; report the
paired difference against candidate 0 with an interval. The target worth pursuing is at
least 10 percentage points.

Never train on these validation branches. Visual/compressed-target readouts require a
separate training-rollout split with the same supervision available to every arm. Stage C
remains blocked until B1 establishes useful candidate headroom and permitted-observation
readability.

## Candidate-count contingency

Do not interpret a null result from eight candidates automatically as zero headroom. Use
the following predeclared diagnosis:

| Observation with 8 candidates | Next action |
|---|---|
| Candidate post-states are not diverse | Stop; drawing more samples from the same collapsed proposal is not useful. |
| Post-states are diverse, but all eight have the same eventual result | Run one nested `N=32` sensitivity check on the same locked prefixes and seeds. The first eight candidates must be unchanged. |
| Eventual outcomes already vary, but the paired interval is wide | Increase independent prefixes, not candidate count. Use 40 first and at most the predeclared 200-prefix primary-task expansion. |
| More candidates still do not affect eventual outcome, while eight steps cause no task-relevant intervention | Consider the separately costed 32-step proposal extension; do not conflate action horizon with candidate count. |

The `N=32` check is diagnostic, not a new default. Report both `N=8` and nested `N=32`
results and account for the larger oracle search set. Do not keep increasing candidate
count after this one extension. Any full screening run with 32 candidates must still fit
the 24 additional GPU-hour cap; otherwise revise the execution design before launch.

## Budget and stopping

- B0: at most four GPU hours.
- B1 initial tranche: at most 24 additional GPU-hours and 96 CPU-hours.
- Do not silently shorten continuation or replace eventual native completion with local
  progress.
- If projected B1 cost exceeds the cap, stop and revise the execution design; this is an
  infrastructure result, not a failure of the compositional-JEPA hypothesis.
