# Research positioning: search-calibrated event world models

## Decision after the September 2026 literature audit

Do **not** claim that an event world model composed with a temporal automaton is
new.  `hint^2` already predicts action-induced atomic-proposition transitions,
propagates their probabilities through an LTL automaton, and uses expected
automaton potential for inference-time robot-policy guidance.  EV-WM also
establishes task-grounded event prediction and verification.  H-WM separately
combines logical and visual world models.

The revised paper object is a **planner-conditional conformal event world
model** (working name: `SearchCal-EventWM`).  The method targets a different
failure: after a planner evaluates many candidates, the selected candidate is
drawn from the model's optimistic error tail.  Marginally calibrated event
probabilities or ensemble variance do not automatically remain valid after
this adaptive selection.

Primary neighbors checked in this audit:

- hint^2: <https://arxiv.org/abs/2608.13678>
- Event-Aware World Models: <https://arxiv.org/abs/2606.13053>
- Horizon-Calibrated Uncertainty World Model:
  <https://openreview.net/forum?id=pZuZWRuPyi>
- Conformal Risk Control: <https://arxiv.org/abs/2208.02814>
- Conformal Decision Theory: <https://arxiv.org/abs/2310.05921>

## Proposed method

For a root observation `o`, a frozen proposal mechanism produces the same
candidate set `C_K(o)` that the planner will search.  An event head predicts a
joint next-event distribution, no-effect/failure probability, and skill
duration for every candidate.  On each calibration root, define one grouped
nonconformity score as the **maximum error over the complete candidate set**:

```text
R_i = max_{a in C_K(o_i)} r(event_i(a), p_theta(. | o_i, a)).
```

The split-conformal quantile of `{R_i}` gives simultaneous, root-level
coverage of all candidates under the locked proposal/search protocol.  The
planner composes the resulting transition intervals with the task automaton
and maximizes a lower bound on acceptance probability.  Receding-horizon risk
is handled either by calibrating a complete search trace as one group or by a
predeclared finite-horizon risk allocation.  A guarantee must never be stated
for a different candidate generator, search width, horizon, or online
distribution than the calibration protocol.

This imports two principles rather than a new scalar loss:

1. simultaneous/post-selection inference from conformal prediction; and
2. robust probabilistic model checking on the product of an event SMDP and a
   task automaton.

The intended mechanism claim is: **candidate-set calibration removes the
growth of selected event-model error with search width, and robust product
planning converts that reduction into higher physical task success.**

## Required baselines

All learned comparisons must share the encoder, action/skill proposal support,
training split, compute budget, and receding-horizon controller.

1. terminal/value head with sparse task success;
2. uncalibrated event head with expected automaton potential (hint^2-style
   high-level baseline);
3. marginally calibrated event head;
4. ensemble pessimism / uncertainty penalty;
5. SearchCal-EventWM with grouped candidate-set calibration;
6. oracle event transitions as an upper bound.

Report both nominal prediction quality and the **selected-candidate** error as
search width grows.  The crucial curve is true task success versus candidate
count, not event AUROC alone.

## Staged falsification plan

### H0: substrate and causal room

Use OGBench-Scene tasks 4 and 5 with exact MuJoCo skill transitions and the
official task-agnostic Markov skill controllers.  Compare sparse terminal
feedback against history-bearing automaton feedback under exactly matched
Skill-UCT search.  This is only a positive-control gate; it is not evidence for
the learned method.

### H1: post-selection failure exists

Collect fresh Scene play roots and execute all locked candidate skills from
snapshot/restore.  Train the event head on trajectory-disjoint data.  On held-
out roots, test whether selected event error or false-positive rate increases
with `K` for the nominal and marginal-calibration baselines.

Stop if search does not amplify event-model error: there is then no causal room
for candidate-set calibration.

### H2: calibration fixes the selected tail

Fit the grouped conformal quantile on calibration roots only.  Verify
simultaneous candidate-set coverage and selection-conditional error on a
locked test split for several predeclared `K` values.  Do not tune the
nonconformity score on test coverage.

### H3: planning success

Run robust product planning on held-out task resets.  The paper direction
continues only if SearchCal-EventWM improves physical success over the nominal
event planner and terminal/value baseline at matched proposals and model
queries.  Include search-width sweeps, calibration-size sweeps, seed-level
paired intervals, and a second planner family.

## Scope limits

- The temporal automaton and event prediction are borrowed components, not the
  claimed novelty.
- Split-conformal coverage is marginal over exchangeable grouped roots; it is
  not a per-instance certificate.
- The guarantee is tied to the frozen candidate/search protocol unless a
  stronger anytime-valid construction is proved.
- Oracle H0 cannot support claims about learned representations, uncertainty,
  or real-robot generalization.
