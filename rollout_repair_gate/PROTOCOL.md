# Deployment-matched predictor gate (locked before execution)

Date: 2026-08-19. Status: implementation in progress; no scientific output has
been inspected.

## Question

The released LeWM checkpoint is trained with one-step teacher forcing but is
queried by a five-step autoregressive CEM rollout.  On the corrected OGBench
Cube audit, the within-snapshot correlation between predicted-endpoint cost and
true-endpoint encoder cost is approximately 0.03.  This experiment asks:

1. Does autoregressive training alone repair that decoupling?
2. Does training on planner-induced, off-policy action sequences add benefit
   beyond deployment-matched rollout training on expert trajectories?
3. Does any fixed-population improvement survive fresh CEM re-optimization?

This is a gate/baseline.  Fast-LeWM, variable-length latent world models, and
autoregressive rollout objectives already occupy the generic multi-horizon
training claim.  A later contribution would require iterative re-mine/repair
behavior against optimizer-induced holes.

## Immutable substrate and split

- checkpoint: `quentinll/lewm-cube`;
- dataset: `ogbench/cube_single_expert.h5`;
- planner: horizon 5, action block 5, history length 1, 96 candidates, 12 CEM
  refits, top-10 elites;
- states and populations: the fresh 128-state PERD H0 manifest and populations;
- split: the existing episode-disjoint 80 train / 16 validation / 32 test split
  from `physical_search_distillation.core.split_for_order`;
- the corrected historical 32-state true-endpoint audit is not a test set for
  this method because it has already been inspected repeatedly.

Snapshot is the unit of uncertainty.  Pair/candidate rows are never treated as
independent replicates.

## Persisted intermediate counterfactual data

For both recorded CEM populations and every candidate, replay from an exact
same-state MuJoCo restore. Persist:

- normalized and raw action sequences;
- rendered RGB after planner horizons `k=1..5` (one image after each block of
  five primitive actions), plus the dataset initial image;
- qpos, qvel, physical cube-to-goal distance, termination step, and a valid
  horizon mask;
- frozen-encoder embeddings of every persisted frame;
- frozen-model predicted embeddings at every `k`;
- dataset-goal and same-renderer-goal embeddings;
- source-population hashes and replay checks.

If termination occurs inside a block, that block's terminal frame is retained
and later horizons are filled for storage but masked from training.  Saving all
25 primitive-step RGB frames is intentionally excluded: the deployed model has
five block-level transitions, so primitive frames add about 5x I/O without
adding a supervised model horizon.

## Matched training arms

All arms start from the identical released checkpoint and update exactly the
same dynamics modules: `action_encoder`, `predictor`, and `pred_proj`.  Encoder
and representation projector remain frozen.  All arms use the same optimizer,
learning rate, batch size, number of optimizer steps, gradient clipping,
sequence count, five prediction calls per sequence, checkpoint schedule, and
final-checkpoint rule.  They also use the exact same loss mask: the natural
off-policy train mask is copied index-for-index onto both expert arms.  Thus
every seed samples the same number of supervised targets at every horizon in
corresponding minibatches, while post-termination targets remain excluded.

1. `one_step_expert`: expert sequences, true-history teacher forcing at all
   five positions.  This is the same-compute ordinary fine-tuning control.
2. `multistep_expert`: the same expert sequence cache, but predictions are fed
   back autoregressively for five steps.  Arm 2 minus arm 1 isolates rollout
   mode.
3. `multistep_offpolicy`: five-step autoregressive training on the persisted CEM
   branches.  It uses the same number of distinct sequences as the expert
   cache.  Arm 3 minus arm 2 tests planner-induced action/state distribution
   while holding training compute and target count fixed.

Three training seeds are used for every arm.  The final fixed optimizer step is
evaluated; there is no arm-specific early stopping.

### Pre-training amendment (2026-08-20)

The first 64 replay shards revealed frequent early task termination (mean valid
target fraction 0.633).  Expert sequences were originally cached with all-five
valid masks, which would have violated the promised target-count control even
though sequence count and forward compute matched.  Before any training job or
scientific result, the protocol was amended to copy the complete off-policy
train-mask multiset index-for-index to both expert arms.  Checkpoint metadata
must contain identical `valid_mask_sha256`, total targets, and per-horizon target
counts across all nine runs; evaluation rejects mismatched same-compute
signatures.  No collected frame, split, arm, hyperparameter, or gate changed.

## Diagnostics and causal interpretation

For `k=1..5`, report within-population Spearman

```text
rho_k = Spearman(
  ||predicted_z_k - dataset_goal_z||^2,
  ||true_endpoint_z_k - dataset_goal_z||^2
)
```

with snapshot-clustered intervals.  Also report latent prediction MSE and
physical-order Spearman.  Interpretation is precommitted as follows:

- low off-policy `rho_1`: one-step distribution shift already exists; recursive
  compounding cannot be the sole explanation;
- healthy `rho_1` followed by a sharp `rho_2` collapse: early recursive
  compounding is supported;
- gradual decline across k: accumulated/off-manifold drift is supported;
- these patterns are diagnostics, not causal proof without the arm contrasts.

## Evaluation ladder

1. **Fixed pool:** score the untouched 32-state held-out populations with every
   checkpoint.  This measures ranking repair without proposal-distribution
   change.
2. **Fresh CEM:** rerun CEM from every held-out state under every checkpoint,
   then physically execute only the returned plan.  Identical CEM seeds and
   budgets are used across arms.  This detects newly exploited pockets.

The fixed-pool held-out set is never used for checkpoint selection.  Fresh-CEM
outcomes are not inspected until all arms/seeds finish.

## Gates

Let arm-level performance average the three training seeds within each snapshot
before snapshot bootstrap.

`PASS_DYNAMICS_REPAIR` requires all of:

1. `multistep_offpolicy` mean held-out `rho_5(predicted,true_endpoint) >= 0.50`;
2. fixed-pool selected physical distance improves over `one_step_expert` with a
   paired 95% bootstrap interval excluding zero;
3. fresh-CEM physical distance improves over `one_step_expert` with a paired
   95% bootstrap interval excluding zero;
4. fresh-CEM success does not decrease by more than 3 percentage points.

If only gates 1--2 pass, verdict is `FIXED_POOL_ONLY_REOPTIMIZATION_FAILURE`.
If arm 2 beats arm 1 but arm 3 does not beat arm 2, verdict is
`ROLLOUT_MODE_ONLY`.  If `rho_1 < 0.20` for the frozen baseline on off-policy
actions, the report explicitly rejects "compounding alone" before considering
later horizons.

No representation fine-tuning or iterative re-mining is licensed unless the
fresh-CEM gate passes.
