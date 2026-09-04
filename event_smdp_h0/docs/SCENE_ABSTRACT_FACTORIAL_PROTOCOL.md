# Scene learned abstract-model factorial

Date locked: 2026-09-03 UTC, after the wide latent-terminal sweep and before
running the abstract-terminal arm.

## Question

Does intermediate event feedback improve physical success when both arms use
the exact same learned Event-SMDP transition, rather than comparing abstract
dynamics against recursive latent dynamics?

## Locked comparison

- Both arms use the identical H1b seed-0 checkpoint (verified by SHA-256),
  seven skills, UCT implementation, horizon 4, exploration 0.55, candidate
  budgets K=7,14,28,56,112, search seeds, reset seeds, and decision limits.
- `abstract_event` uses milestone-progress reward from predicted `q`.
- `abstract_terminal` uses only the predicted final stable-success probability
  from the same transition head.  It receives no intermediate milestone reward.
- Event outcomes are the already completed H2 closed-loop results.  Only the
  terminal arm is newly executed.
- Primary endpoint: pooled paired physical-success difference at K=112.
- PASS requires at least +10 points, reset-bootstrap 95% lower bound above
  zero, and exact McNemar p below .05.

This factorial still uses a hand-specified, simulator-monitored current event
state and one training seed.  It isolates planner feedback under the learned
transition; it is not an end-to-end visual world-model result.

## Result

Jobs `49131` and `49134` completed all 16 paired resets and verified identical
checkpoint SHA-256 across arms.  Verdict:
`EVENT_FEEDBACK_CAUSAL_GAIN_UNDER_SHARED_MODEL`.  The terminal-probability arm
was 0/16 at every K.  Event-progress success was 0/16, 7/16, 12/16, 14/16, and
16/16 at K=7,14,28,56,112.  The primary K=112 paired difference was +100
points with bootstrap interval [100, 100] and exact McNemar p=0.0000305.

This establishes a clean pilot causal effect of event-progress feedback under
the learned abstract transition.  It does not remove the stated current-q
monitor, task, sample-size, or one-training-seed limitations.
