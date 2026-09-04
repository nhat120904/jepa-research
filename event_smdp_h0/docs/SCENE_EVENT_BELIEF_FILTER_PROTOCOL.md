# Scene event-belief filter gate

Date locked: 2026-09-04 UTC, after the learned-perception replication and its
post-hoc error audit, before any belief-filter rollout.

## Mechanistic hypothesis

The learned current-q replication missed its locked per-seed threshold because
a single rendered frame aliases history-bearing automaton states.  In
particular, observer seed 1 repeatedly regressed task-5 state `(1,2)` to
`(0,1)` after being correct at the first decisions.  The proposed fix imports
the prediction--correction principle from state estimation into event-SMDP
planning.

For filtered arm `s`, after executing skill `a_t`, form the transition prior

`qbar_(t+1) = MAP P_theta(q' | q_t, a_t)`.

At the next frame, observer `s` produces `qobs_(t+1)`.  Because the two event
branches are ordinal and non-decreasing, correction is their product-lattice
join:

`q_(t+1) = qbar_(t+1) join qobs_(t+1)`.

Terminal stability is not latched by the join; it always comes from the current
observation/physical outcome.  The filter receives no simulator event state.

## Locked pilot

- Freeze the same abstract transition, three latent observers, Skill-UCT,
  skill library, K=112, horizon 4, and exploration 0.55.
- Use 16 new resets per task: task 4 seeds 85400--85415 and task 5 seeds
  85501--85516.  They are disjoint from training, validation, perception
  pilot, and replication resets.
- From the identical restored root compare, for every observer seed:
  single-frame `fresh_q` and prediction--correction `filtered_q`.  Also run
  simulator-q event and shared-transition terminal arms once per reset.
- Primary analysis is task 5.  Average observer outcomes within each reset,
  then bootstrap resets.

Pilot PASS requires all of:

1. filtered mean task-5 success at least 75%;
2. filtered-minus-fresh task-5 gain at least 15 points with reset-bootstrap
   95% lower bound above zero;
3. seed-1 filtered task-5 gain at least 25 points; and
4. every filtered observer seed has pooled success at least 75%.

A PASS licenses a larger fresh-reset confirmatory run and probabilistic filter
variants.  It does not make the hand-specified automaton novel or learned.

Protocol note: seed 85500 was used for plumbing smoke before the full array and
all learned arms succeeded on it.  It is therefore excluded and replaced by
the unseen seed 85516 before full evaluation.  This conservative replacement
keeps 16 task-5 resets blind and cannot manufacture a filtered-over-fresh gain
from the inspected all-success smoke.

## Pilot result (2026-09-04)

Verdict: `EVENT_BELIEF_FILTER_PILOT_FAIL`.  On task 5 the three-seed mean
success changed from 62.50% with fresh single-frame q to 66.67% with the hard
prediction--correction filter, a +4.17-point paired gain with reset-bootstrap
95% CI [-8.33, 14.58].  This misses both the 75% success and +15-point/lower-CI
gate conditions.

The intervention did repair the targeted unstable observer seed: seed-1 task-5
success rose from 43.75% to 68.75%, +25 points with CI [6.25, 43.75].  But seed
0 and seed 2 each fell by 6.25 points.  Planning-q accuracy improved relative
to raw observation for all three task-5 seeds, yet physical success did not.
The hard MAP prior plus irreversible lattice join therefore trades missed
progress for false latched progress.  Do not scale this hard filter.  Any
follow-up must preserve a distribution over q and allow observation evidence
to downweight an incorrect transition prior.
