# OGBench-Scene Skill-UCT H0 protocol

## Why the substrate changed

The OGBench-Cube snapshot gate used future demonstration chunks as proposals.
Those chunks made terminal success dense enough that both arms solved every
snapshot.  OGBench-Scene tasks 4 and 5 instead contain real prerequisite and
temporary-regression structure: buttons lock the drawer/window, while the
final state can require those controls to be relocked.

## Locked comparison

- Support: seven task-agnostic closed-loop skills backed by the official
  OGBench Markov controllers (toggle either button, open/close drawer,
  open/close window, place cube in an open drawer).
- Planner: receding-horizon Skill-UCT for both arms.
- Oracle model: exact MuJoCo transition under each skill.
- Terminal arm: three-step-stable native task success only.
- Event arm: a task-conditioned automaton that records ordered prerequisites;
  task 5 has independent drawer/cube and window branches.
- Controls: identical skill set, node-local expansion order, UCT parameters,
  horizon, simulations per replan, maximum replans, and initial state.

This is a causal-room gate, not evidence for a learned world model.  A positive
gate licenses the next experiment: learn a skill prior and event-transition
head from Scene play data, then compare against a terminal/value baseline on
held-out task resets.

## Stop/go rule

- GO only if the known fixed skill composition succeeds and the event arm
  improves task success across fresh seeds at matched budgets.
- STOP or redesign support/search if the known composition fails.
- STOP the event-state thesis on this substrate if a well-powered paired run
  shows no success-rate room across meaningful finite budgets.

