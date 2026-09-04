# Scene event-posterior support gate

Date locked: 2026-09-04 UTC, after the hard belief-filter FAIL and before
posterior replay.

## Question

When the single-frame observer makes a hard current-q error on the completed
task-5 replication trajectories, does its soft posterior still retain the true
history state?  This is a diagnostic gate between a probabilistic belief update
and a recurrent/history observation model; it is not a new success-rate test.

## Locked replay

- Replay the exact deployed skill sequences of observer seed 1 on all 64
  task-5 resets from the completed replication (84500--84563).
- At every pre-skill frame, recompute the seed-1 frozen LeWM embedding and full
  cube-stage, window-stage, and stable probabilities.  Verify replayed true q
  and final success against the persisted source result.
- Form the factorized joint posterior over 6 x 4 x 2 event states and record
  true-q probability and rank.  Report top-1, top-2, and top-3 coverage on all
  visited frames, hard-error frames, and the catastrophic true state `(1,2)`.

## Locked decision

- `SOFT_POSTERIOR_SUPPORT_PASS`: on hard-MAP errors, true joint q has top-3
  coverage at least 75% and median probability at least 0.05.  This licenses a
  soft belief pilot, but history remains a required comparison.
- `HISTORY_REQUIRED`: hard-error top-3 coverage below 50% or median true-q
  probability below 0.01.  Do not expect a Bayesian wrapper around the current
  single-frame classifier to recover missing history.
- Otherwise the result is `HYBRID_SOFT_HISTORY_REQUIRED` and both mechanisms
  must be tested without privileging either.

The replay uses already evaluated resets only for failure diagnosis.  Any
success-rate comparison of soft/history methods must use a new held-out reset
range.

## Outcome (job 49317, 2026-09-04)

Verdict: **`HISTORY_REQUIRED`**.

64/64 seed-1 task-5 trajectories replayed (`49248`, `49249`), 573 visited
decision frames, every replay matching the source true-q sequence and final
success.

| Subset | n | top-1 | top-2 | top-3 | median true-q prob | median rank |
|---|---:|---:|---:|---:|---:|---:|
| all visits | 573 | 65.62% | 65.97% | 65.97% | 0.9989 | 1 |
| hard MAP errors | 197 | 0.00% | 1.02% | 1.02% | 1.57e-07 | 20 |
| true state `(1,2)` | 179 | 0.00% | 0.00% | 0.00% | 1.53e-07 | 21 |

The classifier is not merely uncertain on its failures: it assigns the true
event state a probability near the numerical floor and a rank around 20 of 48.
A Bayesian/soft wrapper over this single-frame posterior therefore has almost no
true-state mass to reweight, which also explains why the hard prediction
correction filter only helped where the transition prior alone happened to be
right. Under the locked rule this forbids a soft-belief-only pilot; the next
intervention must give the observer history (recurrent or short observation and
action window). A soft belief over a history-conditioned observer remains
allowed as an arm, but must be compared against the history arm, not credited
for it.

All success-rate comparisons of the history observer must use a new held-out
reset range; the resets replayed here are already burned for diagnosis.
