# Value learning for GR00T proposal selection: assessment and proposed training design

Date: 2026-09-17. Scope: local code/report inspection and primary-source review. No training, simulator rollout, or remote job verification was performed for this assessment. This is a research recommendation, not a newly locked experiment or evidence that a critic works.

## Decision

The user's concern is valid: a single eventual-success label can be a very noisy signal for ranking short chunks. However, dependence on a frozen continuation policy is **not itself a label error**. It defines a policy-specific action value. Such a value can support repeated policy improvement; it need not equal the value of the improved policy to be useful.

Keep terminal-rollout labels as a Monte Carlo reference. Do not scale the current one-label-per-candidate classifier without qualification. For a scalable baseline, evaluate a time-aware, history-conditioned **chunk critic**, initially estimating the frozen GR00T policy using chunk-level Bellman evaluation, with Monte Carlo calibration on independent branches. Only subsequently consider evaluating an improved selector. Keep JEPA versus direct-Q versus frame prediction as a separate, matched representation comparison.

This recommendation does not reopen the failed learned-composer claim.

## 1. What the label actually estimates

Let h_t be the available causal history, g the task, tau the remaining native time budget, u an action chunk committed before the next decision, and pi_0 the fully specified frozen continuation controller. Its definition includes observation preprocessing, action execution cadence, sampling settings, controller/cache reset semantics, and checkpoint.

For native success before the deadline:

    Q^{pi_0}(h_t, u, g, tau)
      = P(success by the deadline | execute u, then follow pi_0).

A terminal branch label Y is one Bernoulli observation of this quantity. BCE is a proper probabilistic objective: under representative sampling and sufficient model capacity, its population optimum is the conditional success probability, not a deterministic judgment that u is good or bad. One label per training example is not intrinsically invalid; generalization across examples can learn probabilities. Repeating every training branch is not required.

There are three different issues:

| Issue | Meaning | Consequence |
|---|---|---|
| Sampling noise | The same prefix and chunk succeeds under some continuation seeds and fails under others | More independent evidence or a lower-variance estimator is needed |
| Policy dependence | The score measures recoverability under pi_0 | It is neither intrinsic action quality nor Q-star |
| Insufficient inputs/distribution shift | Hidden history, remaining time, controller state, or deployment states are not represented | More labels alone may not fix the model |

Long continuation does not mathematically imply high variance: Bernoulli variance is p(1-p), bounded by 1/4. The practical issue is often a small **between-candidate value difference relative to outcome variance**, plus the expense of observing that outcome.

Different policies can reverse a ranking. A chunk that sets up a maneuver GR00T cannot complete may receive low Q^{pi_0}, even if a better downstream controller could exploit it. No estimator of Q^{pi_0} can reveal that capability simply by being more accurate.

## 2. Replanning does not automatically invalidate Q^{pi_0}

Classical rollout/policy improvement evaluates candidate decisions using a base policy's continuation value, executes an improved decision, and repeats. Under exact evaluation and the relevant improvement condition, the resulting policy can improve on the base policy. This is the relevant foundation, rather than the claim that training and deployed continuation must always be identical. [Bertsekas, rollout and MPC](https://web.mit.edu/dimitrib/www/Bertsekas_NMPC_IFAC.pdf).

For our finite-horizon chunk formulation, a sufficient condition is that the selected-policy expectation of Q^{pi_0} is at least the base-policy expectation at every relevant history/time. Exact maximization over a random candidate bank that includes a genuine baseline draw satisfies this condition in expectation. Approximate critics, incomplete history, unmatched cadence, and uncovered deployment states remove that simple guarantee.

Two limitations remain. First, Q^{pi_0} can be pessimistic about coordinated improvements that only work if later decisions also change. Second, repeated selection visits states the training policy rarely visited. Both motivate later policy iteration and data collection, but neither makes the first policy-evaluation target conceptually wrong.

## 3. What the repository currently implements

The Stage-C scaffold and the later `comp_pilot` experiments are different tracks. The grounded composition pilots tested contact-effect prediction, not a trained terminal-success selector.

Static inspection of Stage C found:

| Location | Current behavior | Implication |
|---|---|---|
| `stage_c/models.py:317-324` | Predicted endpoint/summary and current context feed a scalar head, trained by terminal-label BCE | A supervised outcome critic, not Bellman value learning |
| `stage_c/data.py:165-183` | Returns causal observations, candidate actions, task, and one `eventual_success` label | No explicit remaining-time input or repeated-continuation statistics in the batch |
| `scripts/run_stage_c_train.py:236` | Selects checkpoint by validation value BCE | Useful calibration signal, insufficient on its own for within-prefix ranking |
| `scripts/encode_stage_c_offline.py:179-204` | Constructs future-positive-reward labels from demonstration trajectories; marks these entries trainable | Human continuation labels cannot silently be interpreted as frozen-GR00T values |
| `stage_c/metrics.py` | Requires binary labels and reports `max(label)` as oracle success | That oracle is hindsight availability under sampled outcomes, not necessarily achievable expected-value selection |
| `configs/stage_c_screen.json` | Historical default: three history frames, eight actions | Not the longer-history 32/64-step composition pilot, and not automatically matched to native GR00T execution |

The existing guard correctly rejects all-positive/all-negative value training sets. Nevertheless, a mixed demonstration/GR00T dataset with both classes would pass it while potentially fitting a mixture of continuation policies. Source separation is needed in addition to class balance. Demonstrations remain useful for representation/dynamics learning and, with appropriate assumptions and coverage, off-policy TD; their realized returns are not Q^{pi_0} labels.

The report for job 52573 records 16 new paired continuations from **two selected prefixes**: five wins and four losses, net one extra success. This is consistent with unstable single-seed comparisons, but too narrow to establish a universal noise ceiling or no headroom. See [confirmation report](BASELINE_AUDIT_52597_AND_CONFIRMATION_RESULT.md).

## 4. Why the oracle and the ranking evaluation need care

The quantities below are different:

    E_seed[max_j Y_j(seed)]       hindsight success availability
    max_j E_seed[Y_j(seed)]       best fixed candidate's expected success

The first is at least the second and can be much larger even when all candidates have equal expected value. A selector that cannot observe the future seed should not be judged against the first as if it were an attainable value-ranking ceiling. The existing selected-policy estimate is still a valid noisy estimate when selection is independent of evaluation outcomes; the issue is interpreting the oracle and selecting winners from those same outcomes.

For a reference candidate 0, estimate paired differences:

    Delta_j = mean_m [Y_j(seed_m) - Y_0(seed_m)].

Use the same continuation seed schedule across candidates, with independent schedules across repeats. Common random numbers preserve each branch's marginal distribution if implemented correctly and can reduce difference variance when outcomes covary positively. They do not ensure cancellation or identical downstream actions. Measure the paired variance rather than assuming it is small.

Choose candidates on one seed set, then confirm locked choices on fresh seeds. Split train/validation/test by source episode or scene; branches and overlapping windows from one source are dependent. Distinguish prefix-to-prefix uncertainty from continuation-seed uncertainty. Many repeats on two prefixes are not many independent scenes.

A paired advantage or pairwise ranking loss can emphasize within-prefix distinctions. It does not create information: turning one lucky win into a hard preference merely repackages the noise. Preserve win/loss/tie counts and uncertainty, and retain a calibrated value objective.

## 5. Relevant established methods

| Work and verified venue | What to use from it | What it does not establish here |
|---|---|---|
| [V-GPS, CoRL 2024; proceedings published 2025](https://proceedings.mlr.press/v270/nakamoto25a.html) | Offline-RL critic training followed by generalist-policy proposal reranking | That one GR00T continuation label is a sufficient target |
| [Q-chunking, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html) | Treat the full committed action sequence as the critic's action and use chunk-level TD | A JEPA-specific contribution or a guarantee for this arena |
| [TD-MPC, ICML 2022](https://proceedings.mlr.press/v162/hansen22a.html) | Short latent rollout combined with a learned terminal value | Elimination of critic error or long-horizon credit assignment |
| [Time Limits in RL, ICML 2018](https://proceedings.mlr.press/v80/pardo18a.html) | Include remaining time for a genuinely finite-horizon task; distinguish task termination from training truncation | Permission to label an unfinished, time-capped job as failure |

V-GPS uses Cal-QL, with IQL as an alternative, to learn a value for supported policy improvement rather than merely imitate a behavior policy's terminal outcomes. Its appendix also reports critic exploitation as candidate count increases. This supports a direct critic baseline and a candidate-count audit, not a claim that Cal-QL will solve Scrub. [Method and appendix](https://arxiv.org/html/2410.13816v1).

The specific training design below is an adaptation proposed for this repository, not a recipe demonstrated by any one of these papers.

## 6. Recommended training design

### A. Lock the target and execution semantics

Start with finite-horizon native success under pi_0. Use gamma = 1 for this probability target. Choosing gamma < 1 instead favors earlier success and changes the objective; document that choice if adopted.

Use x_t = (causal history representation, task, remaining time). Preserve the history needed by the native task and any continuation-controller context. Native simulator state may support diagnostic teachers and labels, but is not a deployed input unless explicitly allowed for every compared method.

Define H as the number of actions actually committed before the next policy query. If a 16-action proposal is only executed for eight steps, an eight-step critic and a 16-step committed-action critic answer different questions. Separate lookahead length from executed prefix length. Do not concatenate future action chunks obtained after observing real future frames and present them as proposals available at time t.

### B. Keep Monte Carlo as the reference baseline

Train an initial outcome critic from a separate set of current-runtime pi_0 rollouts, including successes and failures. Chunk-boundary samples from ordinary rollouts provide MC targets without requiring an expensive fork for every training example. Branching is particularly valuable for a smaller ranking/calibration panel and for covering important alternatives.

When repeated continuations are available, retain raw outcomes and success counts; a soft target successes/repeats is compatible with BCE. Weighting every continuation equally and weighting every prefix equally answer different sampling questions, especially with adaptive sampling. Record the sampling scheme. Do not pool human or improved-selector returns into pi_0 MC targets without identifying the policy change.

### C. Add chunk-level Bellman policy evaluation for scale

Let S_u indicate first native success during the committed chunk, and D_u indicate an absorbing end: success, genuine task failure, or expiration of the native task deadline. Let x' contain the observed causal history after the actual executed duration. With no earlier success:

    y_TD = S_u + (1 - D_u) mean_{u' ~ pi_0(. | x')} Q_target(x', u').

Fit Q_theta(x, u) to a stop-gradient y_TD using a proper probability loss, and a slowly updated target critic. This is fixed-policy evaluation: the average over pi_0 proposals is deliberate. Replacing it with a maximum changes the target to policy improvement/control and introduces selection bias in noisy Q estimates.

The transition, success event, and next history initially come from real simulator data. The bootstrap can learn from many contiguous chunks per trajectory, reducing the need to finish an independent terminal branch for every candidate. It trades terminal-rollout variance/cost for bootstrapping bias and approximation error. Sparse success still requires real successful experience; all-zero targets do not become informative just because a Bellman equation is used.

Chunk backups condition on all committed actions. Using ordinary feedback-generated sequences as if they were precommitted open-loop actions needs a dynamics/collection audit, especially with stochastic transitions. The cleanest initial data come from actual GR00T chunk execution boundaries with matched cadence.

Use full-return MC as a baseline/calibration check; consider a controlled MC/TD mixture or longer same-policy returns only after the simplest comparison. Do not silently bootstrap across native deadlines. Conversely, an interrupted compute job before the deadline is censored, not a negative outcome.

### D. Improve the selector only after evaluating the base-policy critic

Define a frozen selector pi_k that samples from GR00T and selects with a fixed critic/version. Collect its behavior and evaluate Q^{pi_k} with an explicitly versioned next-chunk policy. This is approximate policy iteration while leaving GR00T's generator frozen. It may reveal coordinated improvements unavailable to Q^{pi_0}.

A direct max-backup critic or an offline-RL method such as Cal-QL/IQL is another route, but changes the objective and requires support/conservatism checks. It is not a drop-in relabeling fix. Planner-induced distribution shift and value overestimation have also been studied in [TD-M(PC)^2, L4DC 2026](https://proceedings.mlr.press/v331/lin26a.html); that paper does not establish the mechanism of our current null results.

## 7. Preserve an identifiable JEPA comparison

Use the same transitions, rewards, remaining-time inputs, policy target, candidate banks, and supervision budget for:

1. Direct history-and-action chunk critic.
2. Frame-latent prediction plus critic.
3. Trajectory-summary prediction plus critic, without learned composition.
4. The matched composer arm only if its incremental role is explicitly tested.

The JEPA arm can implement Q through predicted endpoint/summary and history. Its critic must see predicted representations in training as it will at deployment; training a head only on observed future summaries creates another distribution shift. Actual-future and privileged-state heads are useful diagnostics, not deployable results.

Predictive accuracy, value calibration, within-prefix ranking, and closed-loop native success are separate measurements. A successful direct critic with no extra JEPA gain would establish useful steering but not the proposed world-model contribution.

Do not replace terminal success with contact count alone. Auxiliary progress prediction is useful, but a task can require release, ordering, or avoiding later damage. If shaping rewards are explored, identify the changed objective or use an appropriate potential-based construction with correct terminal boundaries. [Ng, Harada and Russell, ICML 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

## 8. Next evidence, before a training campaign

First audit the saved paired continuations and data semantics. Then profile the cost of an independent-prefix ranking panel; the prior confirmation already cost about 2.2 GPU-hours for 16 paired continuations. Do not prescribe hundreds of long branches as a cheap preliminary test.

For any new panel, lock candidates and separate seed sets for exploratory ranking and fresh confirmation. Increase repeats based on a predeclared precision/budget rule, reporting unresolved ties rather than forcing a winner. More independent prefixes are essential for population claims. No small fixed repeat count guarantees reliable ranking.

Evaluate calibrated probabilities and, more importantly, the selected candidate's independent-seed success gain against candidate 0. Report within-prefix differences, confidence intervals clustered by independent source, and the fraction of unresolved candidate comparisons. Global AUC alone can reflect scene difficulty rather than action discrimination.

If pi_0 values have no resolvable useful candidate gap, stop scaling that particular frozen-continuation reranking objective. This does not rule out a better continuation policy or a different arena. If there is a gap but no critic recovers it, inspect observation sufficiency, value learning, and coverage. If a direct critic recovers it but JEPA does not improve over that critic, close the JEPA contribution in this setting. A full system result still requires repeated replanning evaluated by native task success.

**Final recommendation:** use MC to establish and audit the target; qualify a chunk-level TD critic as the scalable baseline; only then test whether trajectory JEPA adds decision-relevant information. Do not infer value-learning feasibility from the presence of a BCE head, or infer impossibility from dependence on a frozen continuation policy.
