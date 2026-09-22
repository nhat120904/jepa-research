# Review of the current pilot and policy-guided planning

2026-09-21. Research review plus a bounded CPU diagnostic. No training or policy
fine-tuning submitted. This review qualifies the earlier implementation/qualification
statements; the current training harness needs correction before method comparisons.

## New evidence: default proposal headroom, job 53568

Replayed the immutable 53529 source, same 48 starts/goals and same 128-candidate banks.
Checked exact bank regeneration against the two archived banks and reproduced all 48
previous image-selected outcomes. Only the fixed A-then-B candidate and the previously
selected candidate were replayed; no data or checkpoints were changed.

- Fixed A-then-B candidate (index 1, no WM): 11/48 = 22.92%.
- Existing best-in-bank success: 19/48 = 39.58%.
- Additional rescuable prefixes: 8/48 = 16.67 percentage points.
- Previous random-bank baseline: 7.44%; it is not the competent default proposal.

This is a development upper bound for full 48-step chunk success in the fixed bank.
It is neither a learned method gain nor an upper bound on eventual receding-horizon
success. No evidence yet that a learned WM beats cheap image-geometry collision
checking or sequential waypoint control. The former headroom conclusion was too broad.

Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/review_default_53568/result.json`.
Job completed in six seconds, exit 0, 2 CPU / 8 GB, no GPU. Both queue/accounting checked.

## Assessment of the supplied review

The recommendation against a large investment in the current protocol is reasonable.
However, the old oracle ladder is explicitly scoped to specific costs, checkpoints,
tasks and planner settings (`diagnosis/docs/CURRENT_STATUS.md`). It does not show that
prediction improvements can never help another controller/objective. The current query
score computed on realized RGB did track success in this toy bank; whether it can be
forecast accurately is an unanswered question.

Hand-written waypoint proposals and progress tracking sharply limit what Wall proves.
They do not invalidate using an external proposal in a WM paper if all models share it,
but Wall alone cannot substantiate realistic manipulation or novel composition.
The historical FSM is a helper/interface; closed-loop FSM/MPC has not been evaluated.

Compression cannot add information to a fully observed sequence. It may improve finite-
capacity/sample-efficient prediction, query distortion per bit/token or planning under
latency constraints. These are testable possibilities, not guaranteed advantages.
The existing frame predictor emits all time tokens in one pass; no H-times speedup or
autoregressive-error advantage follows from this implementation. Direct-query prediction
also has a reusable history/action context and must be cached in fair multi-query timing.

## Implementation issues found before training

1. `pa_wm/data.py` builds positive reach/order queries from the very future being queried.
   For selected ordered frames i<j, the target is essentially one by construction. This
   is valid hindsight pretext construction, but can create shortcuts and is not a
   counterfactual ranking test. Use shared goals across alternative actions, hard
   reversed/order examples, query-only and action-shuffle controls, and episode splits.
2. `pa_wm/train.py` spatially averages the frame target and slices its first 128 channels;
   the generic codec reconstructs only spatial averages. These weaken the controls for
   a question about spatial paths. Preserve an equivalent spatial target/projection and
   match answer supervision; retain both direct multistep and autoregressive references
   if claiming rollout-error benefits. The local frame arm is not a DINO-WM reproduction.
3. Training data use unguided actions in one layout, while the bank uses goal-guided
   chunks across three layouts. Counterfactual support must be measured; keep the old
   dataset but add an explicitly separate, split-before-branching proposal-matched set.
4. The launcher resolves `/ENCODE_RUN/output` to `/53550/output` while the true location
   is `/encode_53550/output`. Fix this path before submission.

Passing 11 unit tests established shape/gradient invariants, not experimental validity.
The frozen feature cache can be reused, but full training under the current protocol is
not recommended until these issues are addressed. No training job has been submitted.

## Recommended direction

Keep Wall as a bounded mechanism test. Correct the comparisons/data first, then test
whether action-conditioned query summaries beat matched summary/frame/direct-query
controls on held-out branches; report policy-relative selection regret, not only MAE.
Cheap geometric control belongs in the Wall controls. There is no reason to interpret a
Wall failure as proof about all predictive abstraction.

For manipulation, use an existing competent stochastic chunk policy as a frozen proposal.
Sample K coherent chunks, evaluate them with each WM under a common task query, execute
the same short prefix and replan from real observations. Sampling-based MPC and a learned
policy prior are compatible. Include native policy cadence, identical candidates, frame
WM reranking and direct-query reranking. Oracle selection must improve over policy-default
under this exact action-execution protocol before training a large reranker.

A Diffusion Policy checkpoint for native PushT is a concrete integration candidate;
release availability is verified, local runtime/checkpoint/task compatibility is not.
Do not assume its standard fixed-goal policy can propose arbitrary A-before-B pushing.
Native-task success can establish integration, but path-query/abstraction evidence still
needs a suitable task or held-out query experiment. Policy data and compute are shared
and disclosed. A self-supervised WM with a demo-trained frozen policy is not an entirely
self-supervised control stack.

Defer actor-critic, unconstrained action gradients and policy distillation until reranking
works. These can amplify an inaccurate scorer or introduce reward supervision and a
second research problem. Policy-guided reranking is the experimental interface; the
method contribution remains the predictive abstraction and its measured tradeoff.

## Primary references checked

- [DINO-WM](https://arxiv.org/html/2411.04983): visual latent planning benchmark reference.
- [Sparse Imagination](https://arxiv.org/html/2506.01392): uses policy-generated candidate
  chunks on complex tasks. Its LIBERO experiment changes action execution cadence and
  reports a lower policy baseline; do not reproduce that degradation as evidence of gain.
- [Diffusion Policy release](https://github.com/real-stanford/diffusion_policy): code,
  data, configs and pretrained checkpoints for its manipulation experiments.
- [TD-MPC2 implementation](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py):
  mixes policy-generated trajectories with optimized samples; it also learns reward/Q,
  so its training recipe is not a self-supervised drop-in.
