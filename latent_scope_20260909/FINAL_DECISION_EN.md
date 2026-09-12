# Research decision: compositional trajectory prediction in a visual JEPA

Date: 9 September 2026. Scope: simulator-only latent world models for robot planning.

**Decision: select compositional trajectory-target JEPA as the next active research direction. Use native RoboCasa tasks, beginning with ScrubCuttingBoard and using RinseSinkBasin as a simpler control. Do not select a signature-specific method, revive the old progress-cost programme, or continue an open-ended search for unrelated directions.**

This selects where to invest the next research effort. It does not certify novelty, predict acceptance, or assert that an unrun method works. The first milestone is a working behaviour baseline plus a discriminating experiment, not a large training campaign. This document supersedes the method recommendation in RESEARCH_VI.md; ARENA_FIT_EN.md remains the source-level arena assessment.

## 1. Why this option wins the comparison

| Option | Reason to consider it | Decision |
|---|---|---|
| Signature-specific JEPA | Mature path representations and a known concatenation law offer a concrete inductive bias. | Do not make it the main method. The selected tasks do not establish a need for signature truncation or noncommutativity. This is an arena–mechanism issue, not a prior-art veto. |
| **Compositional trajectory-target JEPA** | Predict accumulated interaction effects alongside the visual endpoint; test whether a compositional target makes these effects easier to predict and use in planning. | **Selected.** Best current match between the user's scope, native evaluation semantics, and a question not settled by the previous experiments. |
| A new progress head, belief module, or cost mixture on the old stack | Cheapest reuse of the existing code. | Do not restart. The latest local experiments do not support the proposed fixes. |
| Online adaptation or object-factorized JEPA | Reasonable long-term research families; nearby work does not make them invalid. | Reserve, not a parallel programme. We have not established a more specific intervention–task match for them in this assessment. |
| Explicit physical-state hybrid dynamics | Natural for contact mechanics. | Outside the user's chosen long-term scope as the main learned model. |

The selected option has a real risk: it could collapse into ordinary auxiliary prediction plus a progress readout. A better target loss alone would not rescue that outcome. The composition component must earn its place against an equally strong unstructured segment predictor and a direct action-value baseline.

## 2. How to treat prior art and source quality

Separate **mechanistic overlap** from **strength of evidence**. Conference acceptance is useful evidence of review, not proof of correctness; an unreviewed paper can still disclose the same algorithm. Neither a paper title nor a reported score is a sufficient reason to abandon a direction.

| Source | Verified scope/status in this assessment | Consequence |
|---|---|---|
| [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) | ICML 2025 proceedings; visual-feature dynamics for planning. | Core reference and model family. It does not establish that endpoint targets are the only possible JEPA targets. |
| [TD-JEPA, Bagatella et al.](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html) | ICLR 2026 proceedings; TD-based, policy-conditioned long-term latent prediction. | A strong precedent for importing an established principle into latent prediction. Not interchangeable with action-sequence MPC. |
| [Signatures meet dynamic programming](https://proceedings.mlr.press/v242/ohnishi24a.html) | L4DC 2024 proceedings; signature-based control including MPC and simulated robotics. | A legitimate borrowed foundation. It rules out claiming invention of signature control, but is not evidence that a visual JEPA implementation already exists. |
| [CompPlan](https://arxiv.org/html/2602.19634v1), listed in the [ICML 2026 programme](https://icml.cc/Downloads/2026) | Multi-timescale occupancy models, consistency, and composition of policies. | Directly relevant to temporal abstraction. A claim of inventing multi-horizon consistency would be inappropriate. Occupancy prediction and action-conditioned segment-effect prediction still require a mechanism-level comparison. |
| [hint²](https://arxiv.org/html/2608.13678v1) | Inspected August 2026 preprint; future proposition predictions guide a diffusion policy through temporal-logic objectives. Main-conference acceptance and empirical reproduction were not verified here. | Important functional comparator. Do not treat its claimed gains as established, or its existence as automatic rejection of a latent-target method. |
| [JEPA-WAM](https://arxiv.org/html/2608.10780v1) | Inspected August 2026 preprint; observation/instruction-conditioned next-stage latent guidance for a world-action model. Acceptance and empirical reproduction were not verified here. | “Predict stage latents for robot manipulation” is too broad a novelty claim. Its stated predictor does not evaluate alternative supplied action chunks in the way proposed here. |

Names such as Path-JEPA, or speculative claims in lightly verified manuscripts, will not control the decision. Equally, simply adding “visual latent robot planning” to a known algorithm would not by itself establish a substantial contribution. The paper must identify a difficulty introduced by latent prediction and demonstrate that the proposed mechanism addresses it.

This is a targeted audit of relevant mechanisms, not an exhaustive novelty clearance.

## 3. The selected scientific question

> Does learning a compact representation of the interactions accumulated during an action-conditioned future segment, with consistency under temporal composition, improve visual robot planning beyond predicting individual future frames or an unconstrained trajectory embedding?

The borrowed principle is **compositional representation of trajectories**: a representation of a concatenated path should be recoverable from representations of its constituent segments. Path signatures are one established realization; they are not the only realization and are not presumed optimal for native coverage predicates.

The proposed change is to the world model's prediction target and temporal structure. The initial model produces both an endpoint latent and a segment latent, and learns how segment latents combine. It still predicts in learned visual representation space, conditioned on robot actions. No physical-state simulator replaces that predictor.

Claims deliberately excluded from the initial programme:

- All frame-target JEPAs are blind to path effects. Accurate full latent rollouts plus a monitor could suffice.
- Current images necessarily lack progress information. That must be measured on the new data.
- A finite signature represents every relevant trajectory statistic.
- Two fixed native tasks demonstrate arbitrary query reuse or zero-shot temporal-logic generalization.
- Joint deterministic outputs establish a calibrated joint distribution. They do not.

## 4. Concrete first method, not an architecture search

**First implementation choice:** a learned trajectory encoder and learned composition operator. A signature arm is optional later, after the primary comparison; do not start with conditional flows, particle trees, or a suite of competing algebraic architectures.

Let z_t encode the available camera observations and proprioception, and let h_t be the real causal memory updated only from observations already received. Every competitive baseline gets equivalent history access. During imagined subsegment rollout, maintain a separate predicted memory using the same recurrent update with predicted endpoints and supplied actions; never update the real memory from predictions.

For an interval (t, u], form a training-only target:

    c[t,u] = G_target(z[t+1], ..., z[u])
    Y[t,u] = (z[u], c[t,u])
    F(h[t], z[t], actions[t:u], duration) -> (predicted z[u], predicted c[t,u])

The target encoder sees observed future frames during training, but does not receive the candidate action sequence. This avoids the action-copying shortcut identified in the earlier PS-JEPA protocol. Half-open intervals make segment concatenation unambiguous.

Train an operator C with:

    C(c[t,u], c[u,v]) approximately equals c[t,v]

Use endpoint prediction, segment-target prediction, and composition consistency on observed splits. Also compare direct prediction over a supplied action sequence with prediction over its subsegments, conditioning the later prediction on the earlier predicted endpoint and updated latent memory. Keep an anchor to observed visual features; agreement between two predictions is not a substitute for correctness.

This is an approximate learned composition rule. Neither exact associativity nor a sufficient statistic for arbitrary task history follows automatically. Measure errors across different partitions and increasing prefix lengths. Coverage revisits are particularly important: adding two local counts can double-count an already visited region.

Proposed pilot configuration, explicitly **design choices rather than published settings**:

| Component | Initial choice |
|---|---|
| Visual backbone | Frozen DINOv2 ViT-S/14 patch features; same camera inputs across world-model arms. Trainable spatial fusion and causal memory. |
| Causal memory | 256-dimensional recurrent state, updated from actual observations/actions; provide a matched memory baseline. |
| Trajectory target G | Four summary tokens of width 256; four-layer temporal Transformer over spatially fused visual tokens. EMA target copy. |
| Action-conditioned predictor F | Six-layer Transformer, width 384, with duration/action conditioning and endpoint plus segment outputs. |
| Composer C | Two-layer Transformer over two segment-token sets with boundary/duration information. |
| Target protection | Preserve the fixed visual endpoint target; train G with a latent-feature reconstruction anchor and anti-collapse regularization. No pixel reconstruction is required. Verify that G preserves useful interactions rather than only appearance. |
| First temporal lengths | 2, 4 and 8 executed native control steps; compare direct 8-step prediction with 4+4 and 2+2+2+2. Longer lengths are a later, explicitly trained extension. |
| Planning | Sample eight chunks from one frozen behaviour proposal, score their predicted consequences, execute the selected chunk and observe again. No search over the entire 1200-step episode. |

The latent-feature reconstruction anchor is a pilot target-learning mechanism, not a claimed innovation. If target adequacy fails, allow one documented encoder/target correction; do not compensate with indefinite architecture search.

For scoring, use a matched learned continuation-value readout conditioned on predicted endpoint, accumulated latent history and the native task. Its target is eventual native completion under the frozen continuation policy. Train it from the same permitted trajectory/branch data in every arm. Add a direct history-plus-action-chunk value model as a control: if that succeeds equally well, the world-model-specific claim is weak. A hand-tuned progress reward is not the main result.

Any simulator-derived labels used to train readouts must be declared and shared fairly. The initial planning system therefore should not be advertised as fully reward-free. Ground-truth contact state or task counters are not deployment inputs.

## 5. Arena and baseline commitments

Use RoboCasa revision `4f8a2980def75a55dff96b990745b83540425f09`, subject to a documented compatibility check with downloaded assets and data.

**ScrubCuttingBoard is the primary task.** Its native implementation tracks spatially distinct grasped contacts and checks coverage plus release. **RinseSinkBasin is a simpler control:** it remembers whether water has covered three spout orientations. Neither task establishes a general order-sensitive objective. [Scrub source](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/sanitizing_cutting_board/scrub_cutting_board.py), [rinse source](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/cleaning_sink/rinse_sink_basin.py).

An important correction to the review: count plus extrema can evaluate the recorded scrub history at a checkpoint, but are not necessarily sufficient to merge two independently summarized segments. The native distinct-contact filter compares a new position against previously accepted positions. Test overlap explicitly; do not equate approximate count addition with an exact native monitor, and do not build the paper around exploiting this predicate's implementation quirks.

The official release describes 500 human demonstrations per target task and separate target kitchens. Begin with these two target datasets; partition episodes 80/10/10 for pilot training, validation and representation diagnostics. Use native simulator evaluation for completion, with separate validation and final evaluation seeds. This is a task-specific two-task subset, not a full-suite or zero-shot result. [Dataset documentation](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html).

**Shared action proposal:** the [official RoboCasa Diffusion Policy fork](https://github.com/robocasa-benchmark/diffusion_policy), trained or adapted on the declared data. The inspected [configuration](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_transformer_bs192.yaml) has horizon 10, two observation steps and eight executed action steps. Resolve the actual checkpoint configuration before use. A 32/64-step proposal would require an explicit training/configuration extension; it is not already supplied by that setting. Keep native cameras, controller, action meanings, horizons and evaluator. [Official policy integration](https://github.com/robocasa/robocasa/blob/main/docs/benchmarking/policy_learning_algorithms.md).

Minimum decisive comparisons:

1. Behaviour policy without a world-model selector; also a matched history-capable policy.
2. DINO-WM-style frame prediction with full latent rollout and a history-aware monitor/value readout.
3. Direct prediction of simple visual event/coverage summaries, with matched supervision.
4. The same learned trajectory-target model **without composition constraints**, capacity and compute matched.
5. The proposed compositional trajectory-target model.
6. Direct action-chunk value prediction with the same history, data and candidate set.

An adapted baseline must be named as an adaptation, not a reproduction of the original paper. TD-JEPA is an important reference but not a drop-in RoboCasa policy; do not add a costly port before these decisive controls work.

## 6. The next experiment and decision rules

The following thresholds are proposed research operating rules, not effect sizes established by the literature. Keep validation choices separate from confirmatory evaluation.

| Stage | Concrete output | Decision |
|---|---|---|
| A. Execution preflight | Download a small sample; replay five episodes per task; check action alignment, camera visibility and native completion; reproduce a behaviour-policy evaluation. | Resolve integration failures before training the new model. A registry entry is not a successful replay. |
| B. Component headroom | Common behaviour proposal; fixed candidate chunks from replayable prefixes; compare true future outcomes, simple visual monitoring and full-observation trajectory readouts. | Establish whether useful alternatives exist and whether their outcomes can be read from permitted observations. |
| C. First learned comparison | One training seed, fixed candidates, frame rollout vs unstructured segment target vs compositional segment target, with common readout data. | Continue only if a useful candidate-ranking improvement survives selection of the best-scored candidates. Lower MSE alone does not pass. |
| D. Closed-loop confirmation | Three independent training seeds; paired native evaluation starts; strongest simple and unstructured controls included. | Require a meaningful native success gain and evidence that composition contributes. Otherwise retire the method claim. |

**Replay must include history.** Reconstruct task history by replaying from reset, then snapshot task-owned fields, simulator/controller state and relevant random states. Test that repeating the same suffix reproduces the same observations and monitor trajectory. Restoring only qpos/qvel is inadequate. Avoid extra success-check calls when those calls mutate task state. Use the native lifecycle and report any unresolved replay mismatch before oracle results.

**Make Stage B small and interpretable.** Start with one validation prefix from each of 40 independent native episodes per task and eight proposal candidates per prefix. Replay each candidate followed by a frozen common policy to the native horizon, rather than calling short-horizon progress “eventual success.” This is a one-intervention oracle experiment, not a deployable planner and not repeated oracle MPC. Measure candidate diversity, the improvement available from selection, and visual-readout error on the selected candidates. Never feed these validation branches into training. For stochastic continuation policies, use common scoring seeds and separate evaluation seeds: the maximum of noisy single-rollout outcomes is an optimistic availability ceiling, not a calibrated selection gain.

For the oracle selection effect, use a target of at least 10 percentage points on native eventual completion. Forty prefixes are a screening sample and cannot establish a tight null. If uncertain, allow one predeclared expansion to 200 paired prefixes on the primary task. An upper confidence bound below 10 points rejects the intended headroom at this operating point; an interval still spanning useful gains is inconclusive, not a refutation. If eight-step candidates provide no relevant intervention opportunity, permit one explicitly costed 32-step proposal extension before retiring this arena–planner pairing.

For target adequacy, distinguish true visual-sequence readout, compressed-target readout and predicted-target readout. Include positives and negatives from separate training rollouts when success demonstrations provide insufficient outcome variation. Cap an initial extra-data batch at 200 episodes per task, share it across arms, and label the resulting protocol as offline demonstrations plus additional interaction. The cap is a screening choice, not a sufficiency guarantee.

For the eventual method contrast, predeclare a practically meaningful target, initially +5 percentage points native completion against the strongest matched control, with a paired interval excluding zero. Start confirmation with 200 evaluation episodes per task per training seed; report training-seed variation separately from environment-seed uncertainty. Do not treat all episodes from one checkpoint as independent model replications. A smaller gain may be useful, but should not be rescued post hoc by switching the headline to latency.

## 7. Time and resource commitment

Commit the next two working weeks to Stages A–B and, if they pass, the first Stage C comparison. Do not spend those weeks constructing new task semantics or running a broad signature/flow/encoder sweep.

The first execution budget should be capped, not inferred from paper runtimes: a two-hour CPU replay job, a four-hour GPU policy profiling job, then at most 24 additional GPU-hours and 96 CPU-hours for the initial baseline/headroom tranche. These are proposed scheduling caps, not measured requirements. Asset download/storage needs and simulator throughput remain unmeasured. If the baseline cannot become useful inside the tranche, report an infrastructure/proposal bottleneck and revise the estimate before approving a larger campaign; do not call that a negative result for trajectory targets.

After a positive first comparison, expand to additional **unchanged native tasks** with independently checked path semantics, run three-seed confirmation and ablations, and reserve time for analysis. Two handpicked tasks are qualification evidence, not a complete top-conference evaluation. The user-provided ICML/NeurIPS 2027 windows guide scheduling; this document does not verify submission deadlines or guarantee readiness for either venue.

Follow repository compute policy: no simulator, encoding or training on the login node; use Slurm jobs with explicit walltimes, verify both squeue and sacct before submission, and record jobs without a foreground polling loop.

## 8. Final research judgement

The next direction is selected because its prediction target fits an existing manipulation requirement and admits strong causal comparisons. It is not selected because the literature search found an empty name, because a preprint reported a gain, or because the previous hypotheses must somehow have been right.

The publishable hypothesis is that **composition-aware trajectory prediction improves the learned model's ability to choose actions**, beyond memory, a new readout, a stronger proposal and unconstrained multi-frame prediction. If composition contributes nothing, keep the useful engineering result but do not rename it into the intended method paper.

Work performed for this decision: static repository inspection, primary-source literature and source-code checks, and documentation updates. No data download, simulator execution, model training or Slurm submission was performed.
