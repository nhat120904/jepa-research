# Scene matched wide-budget terminal baseline

Date locked: 2026-09-03 UTC, after reading the H2 aggregate and before running
the terminal sweep.

## Question

H2 found that Event-SMDP success rises to 16/16 at K=112.  Does that remain a
method advantage when the learned frozen-latent terminal head receives exactly
the same wider search budget?

## Protocol

- Reuse the already trained H1 seed-0 latent dynamics and terminal checkpoint;
  do not retrain after seeing H2.
- Same 16 test reset seeds, seven skills, horizon 4, UCT exploration 0.55,
  search seed formula, and physical decision limits as H1b/H2.
- Evaluate K=7,14,28,56,112.  Event-SMDP outcomes are the closed-loop outcomes
  already recorded by H2; terminal outcomes are fresh deterministic paired
  reruns.
- Primary endpoint is pooled paired success difference at K=112.
- The wide-budget advantage is retained only if the difference is at least 10
  points, its reset-bootstrap 95% lower bound is above zero, and exact McNemar
  p is below .05.

This is still a pilot with one training seed and simulator-monitored event
state for Event-SMDP.  A positive result must be replicated and compared to
dense automaton/value baselines before a paper-level claim.

## Result

Jobs `49114` and `49117` completed all 16 paired resets.  The latent terminal
head remained at 0/16 physical success for every K from 7 through 112.  At
K=112 the structured Event-SMDP was 16/16, for a paired +100-point difference
with bootstrap interval [100, 100] points and exact McNemar p=0.0000305.
Verdict: `EVENT_SMDP_ADVANTAGE_RETAINS_AT_WIDE_SEARCH`.

Because this comparison changes both transition architecture and feedback, it
is supporting evidence rather than the clean causal ablation.  The subsequent
shared-checkpoint factorial in `SCENE_ABSTRACT_FACTORIAL_PROTOCOL.md` isolates
feedback alone.
