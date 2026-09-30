# Bounded stopping refinement, 2026-09-29

Authorized: implement, review and improve the stopping method, with an honest
advantage over tuned simple baselines. Test outcomes remain unopened.

V1 gap gets 68/100 on each task in dev CV but costs more iterations than tuned
fixed K. Two design problems: success-only tuning can buy one success with many
iterations, and adjacent score spacing is not the value of another CEM update.

This iteration replaces the gap decision with **fitted continuation gain**.
Small depth-2 regression trees predict whether continuing search improves
`episode_success - lambda * executed_CEM_iterations`. This uses simulator
success labels from development trees, not a new neural world model. It is
supervised decision calibration, not reward-free learning, not a pairwise
reliability certificate, and not yet a novelty claim.

At each checkpoint fit stop versus the already-fitted downstream continuation
policy, backwards in search time, then backwards from plan 2 to plan 1. Do NOT
train on an oracle maximum over unseen future checkpoints. Inputs are current
and past cost quantiles, carried-mean cost, elite-cost progress, population
spread, proposal std and checkpoint-mean displacement. No physical state,
root identity, future cost or future image is an input. The carried-mean cost
at iteration k evaluates the previous mean; the new mean must not be scored
using iteration k+1 in a stopping decision at k.

Lock before running: depth 2, minimum leaf weight 15%, minimum impurity decrease
0.0005. Root-balanced weights for plan-2 branches. Main lambda=0.002 success
units per iteration; lambda=0 is a declared quality-only sensitivity, not a
search over many penalties. No outcome-based filtering of easy roots.

Controls: fixed K, fixed (K1,K2), convergence, gap and band, all selected on the
same train folds and same utility. Also report quality-only fixed K/pair.
Five folds grouped by source dataset episode; report out-of-fold success,
iteration cost and paired episode-bootstrap differences. CV is development
evidence after iterative method design, not a held-out paper claim. Require
exactly roots 0..99 per task and complete artifacts before analysis.

Save fold policies and all-dev policies. A live runner must actually stop the
upstream solver at the chosen checkpoint, reproduce the tree outcomes for
prespecified development roots, and measure real latency (not infer it from
iterations). No simulator labels enter deployment decisions. Snapshot all
executed sources in unique run directories. CPU fitting/analysis and GPU
simulation both use sbatch. No extra test collection in this iteration.

## One follow-up after 55949, before its own results

V2 loses quality to fixed-pair on Cube and has no non-inferiority evidence on
Reacher. Test one capacity reduction, not a penalty/depth sweep: one policy
stump per plan, choosing its budget from the FIRST population only. Features:
log initial median cost, log carried-mean cost, initial elite band/IQR. Only
25/50/75% quantile split candidates, at least 20% distinct training roots in
each leaf, and at least .02 training-utility improvement to replace a constant
budget. Same two lambdas and same episode-grouped folds as 55949. This is a
contextual budget allocator, not the original R2-CEM calibration. Report both
training and OOF scores to test the overfitting explanation. Preserve every
negative result. No test data, simulator collection or world-model retraining.

### Concrete correction after 55951

Uniform branch fitting returned constant budgets inferior even to the best
fixed pair on Cube: plan 2 was being optimized for branches not visited by
plan 1. Correct this with a best-fixed-pair initialization and exactly two
reached-branch policy updates. Each plan-2 fit sees the branches selected by
the current plan-1 rule; then refit plan 1. Accept an update only if training
utility improves, and report OOF generalization on unchanged folds. Same
features, split candidates, regularization and lambdas. This correction does
not resolve limited-data overfitting by itself; no test result is used.
