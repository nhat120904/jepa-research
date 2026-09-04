# OGBench-Scene H2 search-width audit

Date locked: 2026-09-03 UTC, after H1b PASS and before H2 execution.

## Question

Does increasing finite-budget Skill-UCT search select Event-SMDP candidates
with systematically worse simulator-grounded error or calibration?  H2 is the
gate for SearchCal; it does not train or test a calibration method.

## Candidate audit

- Use the frozen H1b seed-0 abstract Event-SMDP without retraining.
- Use all 16 H1 test resets and the canonical drawer-first path for each task,
  yielding roots at every task milestone.
- At every root, run nested UCT traces with `K = 7, 14, 28, 56, 112`, horizon
  4, and exploration 0.55.  Search seed is independent of K.
- Execute every unique candidate sequence in the K=112 trace from an exact
  restored simulator snapshot.  Smaller-K metrics use the corresponding
  prefix, so coverage is genuinely nested.
- For the model-best sequence and the best-scored sequence under UCT's robust
  selected root action, record true event reward/success, reward error and
  regret, final-success Brier/log loss, teacher-forced transition NLL, and
  modal step accuracy.
- Report candidate success coverage and successful-candidate recall.  Also run
  the ordinary closed-loop planner at every K to show whether physical success
  improves, plateaus, or declines.

All uncertainty intervals resample reset clusters; milestone roots from the
same reset are never treated as independent.

## Locked SearchCal gate

Compare K=14 with K=112 on the sequence associated with UCT's robust selected
root action.  Verdict `H2_SEARCH_INDUCED_MISCALIBRATION` requires either:

1. paired mean selected reward overestimate increases by at least 0.10 and its
   reset-bootstrap 95% lower bound is above zero; or
2. paired mean selected-success Brier increases by at least 0.05 and its 95%
   lower bound is above zero.

Only that positive verdict licenses implementing SearchCal.  Otherwise, do not
add conformal machinery merely because candidate errors exist.

## Scope

Current automaton state is still simulator-monitored at each real root.  H2
therefore diagnoses search interaction with a learned transition model, not
end-to-end learned event perception.  It also does not establish elapsed-time
utility because duration cost remains zero.

## Result

Jobs `49097` and `49100` completed all 16 reset clusters and 112 canonical
milestone roots.  Verdict: `H2_NO_SEARCH_INDUCED_MISCALIBRATION`.  From K=14
to K=112, selected reward overestimate changed by only +0.00149 (bootstrap 95%
interval [-0.01712, +0.02476]).  Selected Brier changed by +0.05497, but its
interval [-0.01629, +0.13731] crossed zero, so it did not pass the locked
gate.  The selected calibration-gap change was similarly uncertain (+0.05290,
[-0.01972, +0.13472]).

More importantly for utility, structured-model closed-loop success increased
monotonically across K=7,14,28,56,112: 0%, 43.75%, 75%, 87.5%, and 100%.
SearchCal is therefore not licensed in this setup.  A matched wide-budget
terminal-head sweep is required before interpreting the 100% result as a
method advantage rather than a generic search-budget effect.
