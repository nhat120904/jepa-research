# Predictive abstraction for action-conditioned trajectory JEPA

Date: 2026-09-21. Research proposal, not a validated method or novelty claim.
Canonical home: `predictive_abstraction_20260921/`. Implementation decisions and current
execution status are recorded in `PILOT_PROTOCOL.md` and `JOB_LEDGER.md` alongside this note.
This note follows the user's selection of predictive abstraction and rejection of
JEPA-WMs as the default experimental foundation. No jobs or training were launched.
It does not reopen the closed optimizer-conditioned-misranking paper direction.

## 1. Recommendation

Study a **query-sufficient representation of an unexecuted action-conditioned trajectory**.
Choose what the summary must preserve before choosing a composer or a memory backbone.

The operational question is: at a controlled representation/compute budget, can a
self-supervised trajectory representation answer held-out visual-temporal queries more
accurately or data-efficiently than endpoint prediction, ordered latent prediction,
generic sequence compression, and direct query prediction, and does that improvement
transfer to decisions made with the same action proposals and scorer?

This is a narrower claim than a universally sufficient world model. It is not inherently
new merely because it is phrased in terms of questions. A successful paper needs a
specific learning/compression mechanism and evidence beyond standard multitask prediction.

## 2. Two meanings of predictive abstraction must remain separate

**History abstraction:** compress observed history h_t to predict observations under
future actions. This is the predictive-state/state-estimation question. A recurrent
network trained on future prediction is a necessary baseline, not a straw man.

**Trajectory abstraction:** given h_t and an as-yet-unexecuted action sequence A, predict
a representation of the resulting future path tau, from which many queries can be answered.
This is the recommended initial subject because it retains the original trajectory idea.

There is an important limitation. For a fully observed Markov system, if every query is
only about times *after* the segment, the exact terminal state suffices. In a POMDP the
terminal belief is the analogous reference. A path summary is not intrinsically better.
Interior path queries, compression efficiency, or imperfect observability must justify
the additional representation. Endpoint *image* need not be a sufficient state, but
beating an insufficient image is not proof that all endpoint representations fail.

Do not mix a query about a path already observed with action-conditioned forecasting:
the former tests the codec; only the latter tests the predictive WM.

## 3. Prior art and honest boundaries

| Line | Relevant idea and boundary |
|---|---|
| [Predictive Representations of State](https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf) | Defines state through predictions of future tests. Query-based sufficiency itself is established, not our contribution. |
| [Predictive-State Decoders, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/61b4a64be663682e8cb037d9719ad8cd-Paper.pdf) | Trains recurrent internal states to predict statistics of future observations. Adding future-query supervision to an RNN is insufficient novelty. |
| [RPSP, ICML 2018](https://proceedings.mlr.press/v80/hefny18a.html) | Combines predictive-state filtering with a reactive policy. A predictive memory that supports control is not a new principle. |
| [Horde, AAMAS 2011](https://sites.ualberta.ca/~amw8/horde.pdf) | Learns many predictive questions from sensorimotor interaction. A bank of learned questions is established. |
| [ICVF, ICML 2023](https://proceedings.mlr.press/v202/ghosh23a.html) | Learns state/outcome/intention representations from passive data for downstream value learning. Do not equate an intention with an experimentally identified arbitrary action intervention. |
| [Self-Predictive RL, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/666c1861d709bd84e20b6e0e02a2c223-Abstract-Conference.html) | Unifies state/history abstractions and analyzes self-predictive learning. Predictability alone is not a new abstraction theory. |
| [TD-JEPA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html) | Policy-conditioned long-term latent prediction connects to successor features. Simple discounted latent sums are especially close prior art. |
| [CompPlan](https://arxiv.org/html/2602.19634v1) | Policy-conditioned jumpy models, successor-measure composition, and cross-horizon consistency. Composition consistency alone is not a new method. |
| [Value-equivalent rate-distortion, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3b18d368150474ac6fc9bb665d3eb3da-Abstract-Conference.html) | Lossy compression defined by downstream decision distortion already has theory. Do not claim the rate-distortion principle or an elementary action-regret bound as novel. |

Additional threats: [first-occupancy representations](https://arxiv.org/abs/2109.13863)
already distinguish first visits from cumulative occupancy; [temporal-logic skill
composition](https://proceedings.iclr.cc/paper_files/paper/2024/file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf)
already addresses ordered specifications. A query-conditioned physical abstraction
[position paper](https://arxiv.org/abs/2605.30542) also makes the general query-first argument.
The literature search is not evidence of global novelty absence or completeness.

The repository's `belief_compression/docs/gateA_novelty_matrix.md` already warns about
value-equivalence claims. This proposal compresses a predicted path, not belief particles,
and does not introduce a new active-sensing/VOI module.

## 4. Formal object: sufficiency relative to a query family

Let tau=(z_1,...,z_H) denote a future trajectory in an observation-derived feature space.
Let q specify a visual-temporal query, and y_q=Psi_q(tau) its observable answer.
An observed-path encoder produces S=E_tau(tau); a reader produces D(S,q).

Define query-equivalence by:

`tau ~_Q tau'  iff  Psi_q(tau)=Psi_q(tau') for every q in Q`.

The practical objective minimizes held-out query distortion under a size budget:

`min_(E,D) E_(tau,q) loss(D(E(tau),q), Psi_q(tau))`, subject to a fixed representation budget.

The borrowed principle is task-/query-weighted rate-distortion: discard differences that
do not change the specified answers. Fixed token count, precision, and channel dimension
are operational budgets, not a proof of an information-theoretic minimal sufficient
statistic. A noisy or quantized bottleneck would be needed for stronger bit-rate claims.

No finite small summary should be promised to answer arbitrary queries losslessly. If
queries can recover every frame, this reduces to trajectory reconstruction. Query family,
query distribution, temporal resolution, and allowed distortion must be explicit.

## 5. Initial self-supervised query family

Use a frozen SSL observation encoder with spatial features, plus permitted proprioception.
Choose visual anchors u,v from *training* observation features. Similarity k(z,u) is
bounded in [0,1]. It may use a predeclared patch-level kernel; it is not a physical contact
detector. Normalize all statistics at the logged cadence.

| Query | Example target | Information tested |
|---|---|---|
| Time-local state | A specified observed feature at offset j | Dynamics/time grounding; reference task |
| Occupancy/duration | mean over j of k(z_j,u) | How much the path resembles anchor u |
| Ever reaches a visual condition | max over j of k(z_j,u) | Interior visit, not endpoint alone |
| Ordered pair | max over i<j of min(k(z_i,u),k(z_j,v)) | Evidence for u followed by v |

The soft scores are similarities, not automatically calibrated probabilities of physical
success. Binary versions require train-defined thresholds and a documented observation
predicate. First-arrival time can be a later query, with a separate never-hit outcome.
Do not encode never-hit as an arbitrary large time and regress it with ordinary MSE.

Train on sampled queries rather than task contact/success annotations. Include negative
anchors, reversed-order queries, and hard examples where endpoint or duration alone is
uninformative. Report prevalence and performance per family. Split episodes before
windowing, selecting prototypes, normalizing, or balancing examples.

Generalization tests distinguish: unseen trajectories; unseen anchor combinations;
unseen query parameters/windows; unseen horizons; and entirely unseen query operators.
Holding out a pair of anchors is not the same as learning a new logical operator zero-shot.
First pilot holds out combinations, not the whole ordered-pair operator.

Self-supervised does not mean assumption-free: the grammar and feature kernel are designed
inductive biases. If they are tuned to task success, disclose that task-specific selection.
Do not claim image-derived targets recover unobservable native Scrub contact state.

## 6. Architecture and losses

Observed-target branch, training only:

`future observations -> frozen spatial features -> temporal encoder -> M summary tokens S_obs`

Forecast branch:

`causal history + ordered proposed actions + horizon -> predictor -> S_pred`

Query branch:

`(S, q) -> shared-capacity query reader -> predicted answer/answer distribution`.

The encoder/predictor summary must be query-independent. Otherwise a new query requires
rerunning the WM, and reuse across queries has not been demonstrated. The decoder may
be query-conditioned. Avoid language models in the first implementation: q is a compact
typed tuple of anchor embeddings, time parameters, and query operator.

Start with a small temporal transformer and learned pooling tokens. These are ordinary
implementation components, not the contribution. Match total channels/precision and
decoder cost against compressed-sequence baselines. Do not reuse the previous fixed
16-dimensional frame bottleneck without checking that the chosen queries survive it.

Phase 1 trains the observed-path codec with `L_codec=loss(D(S_obs,q),Psi_q(tau))`.
It is a representation screen, not a standalone positive WM result. Freeze that codec
for the first forecasting comparison, to avoid a moving teacher target.

Phase 2 trains the action-conditioned predictor with:

`L_forecast=loss(D(S_pred,q),Psi_q(tau)) + lambda*latent_align(S_pred,stopgrad(S_obs))`.

Treat latent alignment as an ablation, not a requirement. Answer-space loss is primary:
matching arbitrary teacher coordinates can punish unpredictable variation or miss the
actual quantity needed. Fixed nonconstant observation-derived answers anchor against
collapse, but trivial prevalence shortcuts remain and need controls.

For stochastic futures, do not assume `D(E[S_obs],q)=E[D(S_obs,q)]`. MSE summary regression
followed by a nonlinear reader can fail even with optimal regression. Either train the
query outputs directly to conditional means/probabilities with suitable losses, or model
a distribution of summaries and marginalize. A deterministic representation can encode
multiple expected answers; it need not be the summary of one physically realized future.
Report calibration for probabilistic queries and avoid calling latent variance uncertainty
unless its calibration is actually tested.

## 7. Composition: conditional extension, not the first proof obligation

First establish that the query-preserving summary is useful. Then test whether adjacent
segments can be combined while preserving answers on their concatenation:

`D(C(S_A,S_B,length_A,length_B),q) ~= Psi_q(tau_A ++ tau_B)`.

For imagined segments, B must be forecast from a predicted boundary context, not from
observed future frames. If that context contains multiple frames, output it explicitly.
The original/current history must be identical across all comparison arms.

Do not require two latent vectors to coincide if they answer the same allowed queries.
Query-space agreement is the relevant criterion. Conversely, low consistency loss alone
can reflect agreement on a wrong prediction and never replaces observed-future targets.

Several query families already have trivial exact sufficient statistics: occupancy uses
sums/counts, visitation uses maxima, and a fixed ordered pair can maintain prefix/suffix
statistics and an exact merge rule. Those are mandatory baselines. Learned composition
only earns a role if it helps beyond these rules at comparable capacity, particularly on
new anchors/combinations. Associativity itself is not a contribution.

## 8. Necessary baselines and safeguards

1. Endpoint context; show a genuine interior-query gap, not just a weak endpoint image.
2. Full ordered observed/predicted latent sequences; the information reference.
3. Matched-budget temporal pooling/autoencoder and recurrent GRU or transformer summary.
4. Finite-horizon and multi-discount successor-style feature summaries.
5. Hand-designed query-statistic banks and exact merges where applicable.
6. Direct query-conditioned prediction `(history, actions, q)->answer`.
7. Frame predictor with the **same query auxiliary supervision**.

Single-discount first-moment features generally do not determine arbitrary ordered or
joint path events. They are not universally order-blind: discounting, time augmentation,
rich features, and distributional extensions can retain temporal information. Compare
these strong variants instead of claiming successor methods cannot represent order.

Readouts use matched architectures, parameter budgets, training examples, and tuning.
Identical weights only make sense for aligned latent spaces. A larger decoder on our
summary than on a baseline is an uncontrolled intervention. Count amortized costs at
one query and many queries, including every reader call.

A compressed summary cannot contain more information than the full trajectory from which
it was computed. Its plausible advantages are finite-data learning, noise suppression,
forecasting difficulty, and reuse/compute. A query-conditioned scalar network may be the
best solution for a single task; summary reuse must justify its additional machinery.

## 9. Experimental sequence and headroom, without another sprawling audit

**A. Offline representation test.** One predeclared dataset and query grammar. Use the
observed-path codec, endpoint, ordered sequence, generic compression, and statistic bank.
Measure query error versus representation budget, including held-out anchor pairs and
endpoint-matched examples. No simulator replay, policy training, or composer is needed.
Full observed features provide target availability, not action-selection headroom.

**B. Forecasting test.** Compare the action-conditioned models on identical inputs/data.
Separate compression/readout error from total forecasting error; these are not generally
additive components. Confirm one-seed development findings across multiple train seeds
and episode-level uncertainty. Include per-horizon and per-query-family results.

**C. Decision test.** Keep a competent action proposal mechanism and query-to-score rule
fixed. Measure candidate quality/selection regret on simulator branches, then closed-loop
outcomes. The method's predicted query answers must explain useful action differences.
This is the minimum control evidence; offline query accuracy alone is not a method win.

Logged actions can leak decisions made after observing the future. Under partial
observability, `P(future | history, logged action sequence)` need not equal the outcome
under intervening to execute that sequence. Counterfactual claims need support/causal
assumptions and preferably open-loop branch data with matched prefixes. One successful
demo per initial condition does not identify arbitrary counterfactual responses.

For bounded query scores, uniform candidate-wise answer accuracy implies a simple
selection bound. If score is a linear combination with ||w||_1<=B and all query errors
are <=epsilon, the selected candidate loses at most 2*B*epsilon in that score relative
to the best in the same bank. This elementary bound is **not a contribution**. Average
test MSE does not establish its uniform assumption, nor does the score guarantee physical
success. It clarifies the interface, not a safety guarantee or full-planning theorem.

## 10. Arena choice and next deliverable

Select arenas by required contrasts: trajectories with similar endpoints but different
interior query answers; diverse logged actions; repeated visual situations; and feasible
control through an established interface. A hard memory task is not required.

Recommended minimal mechanism test: the existing DINO-WM navigation family, with visual
waypoint/ordered-visit queries generated from trajectories. Any ordered-waypoint control
task is a **new diagnostic extension**, not the original benchmark's headline score.
Navigation is not sufficient for a manipulation-paper claim. A preselected visual
manipulation dataset such as OGBench cube can serve as the subsequent transfer test;
its native endpoint task must not be relabeled as evidence for path-order reasoning.
Exact available data, renderer, and query diversity remain to be checked before execution.

Scrub's cached data can test plumbing, but this proposal does not reopen its hidden-contact
claim or use the repeatedly inspected development split as fresh confirmation. MIKASA is
not required, and JEPA-WMs is not the chosen implementation foundation. DINO-WM is a
published frame-planning reference, not a claim that its backbone guarantees success.

The next concrete deliverable should be **one locked query specification and a matched
offline representation/forecasting protocol**, followed by its small pilot. Do not start
with a new composer, a VLA, a learned task-value head, or a broad benchmark sweep.

Continue only if there is both a nontrivial query-compression tradeoff and a forecasting
benefit beyond identical auxiliary supervision. If direct query prediction or a simple
statistic bank ties, narrow/close that summary design. If only the observed-path codec
wins, do not claim WM improvement. If control fails, report that limitation rather than
silently selecting another arena until a positive appears.

All future model/encoding/simulator work and nontrivial dataset scans must be submitted
to Slurm compute nodes under repository policy. No such work was performed in this review.
