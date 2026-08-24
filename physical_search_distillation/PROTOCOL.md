# H0 preregistration: Physical-Elite Refit Distillation (PERD)

Status: locked before model training or held-out evaluation (2026-08-19).

## Question

Can physical counterfactual supervision be used **only during training** to learn a zero-query
CEM objective that improves held-out OGBench-Cube planning?  More specifically, does distilling
the physical elite-to-refit operator add value beyond learning pointwise distances, rankings, or
elite membership?

The H0 experiment is a falsification pilot, not final paper evidence.  Its purpose is to decide
whether the operator-level contribution is real enough to justify a larger DAgger-style study.

## Locked data and split

- Dataset/checkpoint: `ogbench/cube_single_expert.h5`, `quentinll/lewm-cube`.
- 128 fresh episode-disjoint snapshots, excluding all Phase-0d, Phase-1a, and Phase-1av2 episodes.
- Split is a deterministic function of manifest `order`:
  - test: `order % 8 in {0, 4}` (32 states),
  - validation: `order % 8 == 2` (16 states),
  - train: remaining residues (80 states).
- No test physical outcome can enter feature normalization, training, early stopping, or model
  selection.  The evaluation script reads only the test manifest and trained checkpoints; it does
  not read the saved test candidate populations.

## Teacher collection

Frozen LeWM generates two planner-induced populations per state: CEM iteration 0 and iteration
11.  CEM uses 96 candidates, 12 updates, hard top-10 elites, horizon 5, action block 5, and seed
`20260819 + order`.  Every recorded candidate is executed after restoring the exact complete
MuJoCo state.  Physical distance to the goal cube position is the teacher cost.

For population actions `a_i` and hard physical elite set `E*` (the 10 lowest physical costs), the
teacher operator is exactly the upstream CEM refit:

`mu* = mean_{i in E*}(a_i)` and `sigma* = sample_std_{i in E*}(a_i)`.

There is no soft teacher in the primary target.  Soft elite probabilities may only provide an
auxiliary optimization loss.

## Matched arms

All learned arms see identical train/validation populations and use matched seeds/optimization:

1. `pointwise`: regress physical endpoint distance.
2. `listwise`: match the within-population ordering distribution.
3. `elite`: predict exact physical top-10 membership.
4. `operator`: minimize normalized distance between the student's **hard top-10 CEM refit** and
   `(mu*, sigma*)`; a straight-through soft mask supplies gradients, with elite BCE auxiliary.
5. `operator_metric`: the same operator target, but with a trainable residual bottleneck that
   reshapes predicted-endpoint/goal geometry before scoring.  This is a representation-adapter
   arm, not full LeWM encoder fine-tuning.

The student features use only deployable quantities: normalized action sequence, frozen LeWM
predicted endpoint, current/goal embeddings, and the native latent cost/rank of the current
population.  No simulator variable or physical label is an input.

## Zero-query evaluation

Each arm replaces GoalMSE inside otherwise identical CEM.  On each held-out test state it receives
the same compute budget (96 candidates x 12 CEM steps, top-10).  The solver sees no simulator
outcome.  Only the final returned mean plan is executed once post hoc for measurement.  Native
GoalMSE is the B=0 baseline.  Primary outcomes are physical endpoint distance and success.

## Locked decision rule

Use paired state bootstrap confidence intervals (10,000 resamples).

- `GO_OPERATOR` only if `operator` beats both native and `listwise` in physical distance with a
  strictly positive lower 95% CI for the paired improvement, while not reducing success.
- `GO_REPRESENTATION` if `operator_metric` additionally beats `operator` with a positive lower
  95% CI and does not reduce success.
- `STOP_OPERATOR_NOVELTY` if operator does not beat listwise in point estimate, or lowers success.
- Otherwise `HOLD_SCALE`: inconclusive H0; increase held-out states once, without changing losses.

Only `GO_OPERATOR` authorizes a second round of planner-induced population collection under the
learned objective (DAgger-style relabeling).  Full predictor/encoder LoRA is deferred until then;
otherwise it would add capacity while the claimed operator advantage remains unverified.

## Claim boundary

A positive result supports: physical counterfactuals collected during training can teach a
zero-query planner update, and operator supervision is more decision-useful than matched ranking
supervision under this protocol.  It does not establish generality beyond LeWM/OGBench-Cube, and
does not claim that representation reshaping is necessary.
