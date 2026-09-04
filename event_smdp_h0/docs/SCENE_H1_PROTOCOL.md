# OGBench-Scene H1 learned-event protocol

Date locked: 2026-09-04 UTC

## Question

Can a learned skill-level event evaluator retain a material part of the oracle
event-feedback advantage under the same seven skills and receding-horizon
Skill-UCT used by H0?

H1 does not test SearchCal.  Search calibration is licensed only after H1
passes and H2 shows that selected-candidate error grows with search effort.

## Counterfactual dataset

- Scene tasks 4 and 5; train, validation, and test reset seeds are disjoint.
- Canonical paths create roots at every task milestone.  Task 5 includes both
  drawer-first and window-first paths.
- From every root snapshot, all seven skills are executed and the root is
  exactly restored between interventions.
- Each row stores current/next frozen LeWM-DINO latent, privileged simulator
  vector, task goal, current/next automaton state, endpoint predicates,
  duration, success, and no-effect/censoring label.
- A repeated root-skill transition must match in duration, automaton state,
  simulator signature, and rendered pixels.

## Models

For each feature view (`latent`, `privileged`), fit one shared skill dynamics
model.  Freeze the fitted dynamics when comparing three readouts:

1. `terminal`: endpoint native-success BCE;
2. `event_bce`: endpoint predicate BCE followed by the fixed task automaton;
3. `event_time`: joint next cube/window automaton-state likelihood, success,
   no-effect, and log-duration losses.

The privileged feature view is an information upper bound, never the main
deployable method.  Current automaton state at each real replan is supplied by
the simulator monitor for all event arms; imagined transitions come only from
the learned model.  This isolates transition-model learnability from event
perception and must remain explicit in every claim.

Primary H1 planning sets duration cost to zero so it matches what H0 actually
established.  Duration is auxiliary until a separate oracle experiment makes
elapsed time enter the planning objective.

## Evaluation and gate

- Fresh test resets only; paired by task, reset, model seed, and search budget.
- Same skills, horizon, UCT implementation, exploration constant, and model
  query budget for all heads.
- Search seed is independent of budget, so at a fixed physical root the
  smaller-budget search trace is a prefix of the larger-budget trace.
- Pilot budgets: 14 and 28 simulations per replan.
- Pilot GO: at budget 28, the best frozen-latent event head has at least 50%
  pooled physical success, paired gain over terminal of at least 10 points,
  and bootstrap 95% lower bound above zero.
- `privileged` passes but `latent` fails: representation ceiling.
- both fail: no learned-event advantage under this data/model protocol.

The initial pilot uses 16 training resets per task, 4 validation resets per
task, 8 test resets per task, and model seed 0.  A positive pilot must be
replicated with three training seeds and 64 test resets per task before being
used as paper evidence.

## Locked pilot result

The initial H1 run completed as Slurm jobs `49058`--`49061` after replacing one
task-5 training reset whose window-first scripted support path failed.  All 16
fresh test resets completed.  The preregistered verdict was
`H1_NO_LEARNED_EVENT_ADVANTAGE`: at budget 28, latent `event_bce` succeeded on
1/16 resets (6.25%, paired gain +6.25 points, bootstrap interval [0, 18.75]),
while latent `event_time`, both privileged event heads, and both terminal heads
succeeded on 0/16.  Budget-14 success was 0/16 for every arm.

This result is preserved as the answer for the model specified above.  It is
not an encoder-ceiling result because the privileged arm also failed.  The
large gap between held-out one-step prediction and recursive planning instead
licenses one bounded mechanism audit: test whether closing the learned
transition directly in automaton state removes recursive feature-dynamics
drift.  That follow-up is H1b, not a relabeling or overwrite of H1.  SearchCal
remains blocked unless the closed abstract model first recovers useful physical
success.
