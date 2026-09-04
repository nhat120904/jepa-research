# OGBench-Scene H1b abstract-closure mechanism audit

Date specified: 2026-09-03 UTC, after reading the locked H1 aggregate and
before training H1b.

## Motivation and scope

H1 produced accurate held-out one-step event predictions but near-zero physical
planning success for both visual-latent and privileged recursive feature
dynamics.  H1b tests one mechanism only: whether repeatedly feeding a learned
feature prediction into the next event transition caused that collapse.

H1b does not replace, relabel, or invalidate the negative H1 result.  It uses
the same H1 train/validation transitions and the same 16 fresh test resets.
SearchCal remains out of scope.

## Intervention

Fit a categorical Event-SMDP transition

`p(q[k+1], success[k+1], no_effect[k], tau[k] | q[k], task, skill[k])`.

Here `q` is the same history-bearing Scene automaton used in H0/H1.  During an
imagined skill sequence the model recursively feeds back only its predicted
`q`; it does not roll a visual or privileged physical feature.  Current `q` at
each real replan is still supplied by the simulator monitor, exactly as in the
H1 event arms.  Consequently H1b is an abstraction-closure audit, not a claim
of learned event perception or an end-to-end deployable world model.

Before evaluation, report how often identical `(q, task, skill)` inputs have
conflicting next-stage/success labels in train and validation.  This diagnoses
whether the chosen event state is approximately Markov for the fixed skill
library.

## Locked evaluation and decision

- Same test reset seeds `83400..83407` and `83500..83507` as H1.
- Same Skill-UCT, horizon 4, exploration 0.55, budgets 14 and 28, and maximum
  physical decisions (6 for task 4, 10 for task 5).
- Search seeds use the exact H1 formula, so candidate ordering is matched.
- The paired baseline is the already completed H1 frozen-latent terminal head;
  no baseline is retrained after observing H1.
- Primary budget: 28.
- PASS requires pooled H1b success at least 50%, paired gain at least 10 points,
  and bootstrap 95% lower bound above zero.

PASS isolates recursive feature drift as a material failure mechanism and
licenses a later search-width audit of the structured model.  FAIL means the
event abstraction/model itself does not retain enough oracle advantage, and
the direction stops before SearchCal.

## Result

Jobs `49078`--`49080` completed the locked audit.  Verdict:
`H1B_ABSTRACT_CLOSURE_PASS`.  At budget 28, pooled physical success was 12/16
(75%, bootstrap interval [50%, 93.75%]) versus 0/16 for the frozen-latent
terminal baseline, a paired +75-point difference with the same interval and
exact McNemar p=0.000488.  Task 4 was 8/8 and task 5 was 4/8.  At budget 14,
pooled success was 7/16 (43.75%).

The abstraction audit found conflicting next-stage/success labels for only
3/147 `(task,q,skill)` cells in train and 1/147 in validation.  The locked H1
latent contextual event-time model deployed an event-advancing action on
29.69% of decisions, versus 84.91% for H1b; first-action progress was 56.25%
versus 100%.  These observations support recursive feature drift as a material
mechanism, but H1b still assumes simulator-monitored current `q` and therefore
does not establish learned event perception.

This PASS licenses H2 on the structured model.  It does not by itself license
SearchCal: H2 must first show that selected-model error or overconfidence
increases with search width.
