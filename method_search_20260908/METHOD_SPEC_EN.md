# Candidate specification: Feedback-Equivalent World Model Learning

2026-09-08. Research proposal, not a validated method. This specification deliberately separates a classical finite construction from the unresolved sequential learning contribution.

## Objective and setting

Learn a compact controlled stochastic model whose imagined observation branches support transferable feedback decisions. Target tasks involve acquiring information and then manipulating an object. Do not claim that maximum-likelihood recurrent models inherently violate causality: with sufficient capacity and coverage they can learn the relevant conditional laws.

Start with a finite static hidden variable θ (e.g., which covered location contains the target, or a discretized initial hole-pose hypothesis), a small shared action-primitive library, and short observation horizons. The extension to dynamic hidden states requires an explicit transition model and is not provided by a posterior martingale assumption.

## Finite one-step construction: existing mathematics

For a fixed history/action context, let P[θ,o] denote the reference observation channel and Qφ[θ,y] the model's abstract signal channel. Both matrices are row stochastic. Define:

`d(P → Qφ) = min_K max_θ (1/2) ||P[θ,:] K - Qφ[θ,:]||_1`,

where K[o,y] is row stochastic and cannot depend on θ. Define the reverse direction with another stochastic kernel L[y,o]. This is a classical deficiency construction, not a new distance.

Use `L_channel = d(P → Qφ) + d(Qφ → P)` as an initial finite objective. The forward direction tests whether real observations can support the model's signal; the reverse direction tests information discarded by the model. The reverse adapter is not generally the inverse of the forward adapter. A symmetric information-equivalence objective does not require identical raw observation distributions or an identity mapping.

The LP in channel_example.py computes these quantities exactly for known finite channels. Optimization through the inner optimum can initially use alternating LP solves and a subgradient/envelope update away from degeneracies. Neural scaling, smooth relaxations and estimator bias remain research work. No claim of an efficient new solver is made.

## Reference channel and data privileges

For a controlled toy, P is known. For a robot task, estimate P from repeated sensor observations conditioned on hidden hypotheses, action and valid history; retain sampling uncertainty. A frozen observation tokenizer/filter can define a finite signal alphabet, but its information ceiling must be established independently. Learning both sides freely permits collapse; do not do that.

Privileged simulator state may label θ during training and support a reference filter. All learning baselines receive the same state labels, observations, branch samples and model capacity. An actor never receives θ or the simulator state, either during imagination or real evaluation. A generative dynamics model may sample θ internally, while exposing only predicted observations to the actor.

Resetting the same physical state creates counterfactual actions for that state; it does not sample a belief posterior. To create a channel at a history, sample hidden hypotheses consistent with that history, with a documented prior/posterior or likelihood-weighting scheme. Start with a small explicit hypothesis set so this requirement is auditable.

## Minimum model and training loop

1. Establish a stable low-level controller and physical reward/success definition, shared by all methods.
2. Train a stochastic dynamics/reward model and an observation model on one fixed dataset. Include an explicit learned-likelihood Bayes-filter baseline before proposing a larger architecture.
3. Freeze the reference observation interface; estimate action-conditioned channel rows with held-out calibration samples.
4. Learn an abstract signal generator Qφ and causal real-to-model adapter Kψ. Combine the ordinary dynamics/reward likelihood with channel matching. Train a reverse adapter only for the information-loss constraint.
5. Build the same shallow feedback tree for each model. At a branch, the next action depends on observed signal history. An open-loop action optimizer alone cannot establish the proposed benefit.
6. At evaluation, the real-to-model adapter consumes actual sensor histories, the filter updates its belief, and the planner replans. No simulator access is allowed to the evaluated learned controller; simulator-based oracles are separate reference arms.

## Sequential contribution that remains to be developed

Independent one-step optimal adapters can be incompatible. The proposal requires one consistent recurrent family

`K_t(y_t | o_≤t, y_<t, a_<t)`

and the corresponding reverse family. Neither may use future observations, future actions or hidden-state labels. The adapter must be executed as part of the deployed policy interface.

Training must evaluate conditional channel discrepancies at valid coupled history pairs, across actions the policy can select. It must preserve the coupling between physical outcomes/rewards and information. Matching an observation channel while leaving the associated physical-state law unconstrained is insufficient.

A possible finite implementation enumerates a short interleaved action/observation tree, shares adapter parameters across identical prefixes, and trains under the maximum or a coverage-controlled aggregation of conditional discrepancies. Increasing the number of actions or horizon will make this expensive. A practical neural estimator that avoids losing these constraints is part of the prospective contribution.

Theoretical target: under uniformly controlled conditional discrepancies, bounded rewards, matched physical dynamics (or a separately bounded dynamics error), and a constructible causal policy adapter, derive a finite-horizon return-transfer bound by sequential coupling. Pointwise, data-averaged errors do not imply uniform control; neither does matching only a handful of open-loop trajectory marginals. The one-step Blackwell result does not settle this theorem. A loose generic simulation bound alone is unlikely to be enough novelty.

## Closest competing explanations and methods

- VDB already learns representations through deficiency; this cannot be marketed as the first use of Blackwell sufficiency in deep learning.
- Le Cam Distortion already studies learned directional observation simulation and RL transfer. A fixed observation-noise adapter is not the remaining contribution.
- WBU already learns belief updates with latent models and value-related guarantees. A posterior-matching auxiliary loss is not enough.
- Conditional COT-GAN already learns conditional sequential generators; causal Sinkhorn with actions is not enough.
- Active tactile belief-space control and BayesContact already plan informative robot actions. Adding an information-gain planner is not enough.
- Full conditional likelihood, supervised belief heads, and direct conditional channel KL/TV must be matched in labels, data and planner. Their success can invalidate the proposal's practical motivation.

## Candidate experimental pair

First: occluded target retrieval with shared reliable primitives, measured execution cost, and a small explicit hypothesis space. This is a proposed simulator protocol, not a reproduced benchmark. IMBench supplies a relevant task description but runnable simulator code was not verified in this audit.

Second, only after the first supports the mechanism: pose-ambiguous peg insertion with contact feedback. ManiSkill provides a public insertion environment; the partial-observation and probing protocol would be an explicit modification. Compare learned-channel feedback planning against a BayesContact-style model and a simulator-likelihood oracle. The contact interface and controller still require implementation verification.

## What would make this a method paper?

A usable sequential training estimator and deployed adapter, a rigorous statement of what decisions are preserved, and matched robot experiments showing information-channel correction improves return across more than one sensing/task regime. The baseline must already have memory, stochasticity and the same privileged training labels. A one-step finite illustration, or gains against a deliberately blind JEPA baseline, is not sufficient.
