# Decision on continuing trajectory-target JEPA

Date: 18 September 2026. Evidence: local implementation and completed experiment reports, plus primary-source literature and arena documentation. This assessment does not launch training, establish new experimental results, or change previously locked verdicts.

> Clarification after reading the user's specific source, Ohnishi et al. (L4DC 2024): the negative composer results concern an unconstrained learned Transformer merger, not an implementation of Chen's signature product. Also, composition can have a decision role even with one predicted chunk by combining the observed prefix with the imagined segment. Before abandoning algebraic composition, the specific fixed-algebra target deserves a separate, small representation-level comparison. See [signature-composition addendum](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/docs/SIGNATURE_COMPOSITION_REASSESSMENT_20260918_EN.md). This narrows the recommendation below; it does not reopen the failed Scrub learned-merger campaign.

## Decision

Continue the **action-conditioned trajectory-target question** for one bounded redesign and decision experiment. Retire the current learned-composer claim on Scrub. Do not fund another campaign that simultaneously repairs GR00T, learns a moving trajectory target, learns composition, and learns a long-horizon critic.

The selected next experiment is a **non-compositional trajectory-target JEPA versus endpoint prediction, frame rollout, and direct chunk scoring**, initially on native ManiSkill DrawTriangle-v1. This is a mechanism pilot with an existing demonstration and RGB Diffusion Policy route, not a sufficient paper arena or a claim that endpoint observations lack information. LIBERO-Safety physical avoidance is the next application candidate only if the mechanism pilot earns expansion. Neither arena has been runtime-qualified in this assessment.

This preserves a substantial part of the original idea, but deliberately changes its claim. If retaining a learned composer as the central contribution is non-negotiable, my recommendation is to stop this implementation and formulate a different composition problem before spending more compute. There is currently no verified alternative arena in this assessment that makes the existing composer necessary.

The research question is worthwhile; the present method is not yet a credible top-conference result. An untested full pipeline is not positive evidence. Conversely, a failed component comparison on one development setting does not invalidate every action-conditioned trajectory representation.

## What the experiments actually say

| Question | Evidence | Interpretation |
|---|---|---|
| Does the learned composer beat sequential memory rollout on Scrub? | Grounded shared-teacher experiment: 128-step composed MAE 1.623 versus 1.424 without composition; three-part 1.643 versus 1.419. The three-part paired interval excludes zero in the unfavorable direction. | Invest against this claim. These are several related evaluations, not four independent training-seed replications. |
| Does direct segment prediction contain a useful lead? | Segment versus frame: primary MAE reductions 6% and 7%, with intervals crossing zero; a secondary split improves 8.5%. A 128-step correlation advantage survives its interval. | A small development lead, not an established method advantage. It failed the prespecified rules. |
| Is segment prediction already a major speed win? | Predictor-only 5.9 versus 10.0 ms; including shared context 24.0 versus 28.5 ms, at batch 256 and horizon 128. | About 1.19x for the measured context-plus-predictor path. This excludes the full policy/simulator system and does not establish deployment speedup. |
| Is candidate-selection headroom established? | Locked fresh continuations: 5/16 baseline versus 6/16 selected, five wins and four losses, from two selected prefixes. | Earlier screening gains were not robustly confirmed. Two prefixes do not establish population headroom or its absence. |
| Has the complete learned rerank–execute–replan system failed? | The counterfactual protocol tested one intervention followed by frozen GR00T; the offline composition pilots tested representation/readout metrics. | No. The complete intervention is different and remains untested. This alone does not justify building it at full scale. |
| Is the encoder intrinsically unable to represent the relevant information? | Input probes use one reader and budget, including aggressive spatial compression. Some earlier interpretations were explicitly corrected. | No information-theoretic ceiling was measured. More information being available would still not establish a composer benefit. |

Sources: [grounded result](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/docs/COMP_GROUNDED_RESULT_52655.md), [segment versus frame](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/docs/SEGMENT_VS_FRAME_RESULT_52662.md), [fresh continuation confirmation](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/docs/BASELINE_AUDIT_52597_AND_CONFIRMATION_RESULT.md), and [corrected input-probe interpretation](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/docs/COMP_PILOT_REEVAL_RESULT_52642.md).

The development validation set has been read repeatedly, and a previous re-evaluation accessed the old test set. A new confirmatory claim needs genuinely fresh evaluation episodes/scenes, with overlapping windows and branches grouped by their source. Do not relabel an already inspected split as untouched.

## The surviving motivation

The defensible question is:

> Can a JEPA predict a compact representation of the effects occurring throughout a committed action segment, such that it selects actions more accurately or with less total computation than endpoint prediction, explicit frame rollout, and direct action-value prediction?

This concerns **what to predict and which information the prediction preserves**. It does not require proving that a policy lacks memory. All decisive controls should receive the same permitted causal history and current observation.

An endpoint-only *objective* can overlook what happened during a trajectory, such as a temporary collision. That is not a universal limitation of endpoint *representations*: the final drawing may visibly contain the entire trace, and a recurrent endpoint representation may encode prior events. The relevant comparison therefore includes a history-aware frame model and a strong endpoint-image model, rather than only a terminal DINO distance.

A single task-specific scalar can be learned directly as a chunk score. A trajectory representation needs to earn its extra machinery through a measurable advantage: better candidate selection with the same data, lower total latency at matched decision quality, or reuse across task queries with less additional supervision. The current experiments have not demonstrated these advantages reliably.

Do not present this as an entirely new problem statement. The contribution must be a particular representation-learning mechanism and its demonstrated benefit, not the observation that trajectories matter.

## Why the current implementation is an unfavorable test

### Composition is not yet doing indispensable planning work

The pilot predicts known action sequences of length 64–128. The GR00T selection experiments intervene with chunks of 8–16 actions. Choosing among single short chunks does not automatically use the current future-segment composer. However, a different use—composing the observed prefix with the candidate future segment—can directly affect a whole-path cost, even with one predicted chunk. Concatenating future GR00T outputs obtained from real future observations would leak information unavailable when the first chunk is selected.

In `comp_pilot/models.py`, `SegmentArm.rollout` already advances predicted memory and endpoint sequentially for each part. The composer merges summaries in addition to that rollout; it does not remove the need to generate the later, boundary-conditioned prediction. There is no automatic parallelism or long-horizon planning saving from adding this merge operation.

Composition needs its own use case: for example, a demonstrated reduction in recomputation while evaluating many valid shared subplans, or reliable transfer to unseen segment combinations beyond a matched recurrent model. The current short-chunk reranker does not establish either.

### Compression does not imply compositional sufficiency

Reconstructing a future feature sequence approximately, avoiding collapse, and maintaining an EMA teacher do not guarantee that a summary retains the distinctions a planner needs. EMA specifies how target weights move; it does not specify which task-relevant information survives compression.

For an elementary example, two segments can each cover three locations. The combined coverage can be three or six depending on overlap. Counts alone cannot determine the merge. A learned composer cannot reconstruct information absent from both inputs. Conversely, when the representation explicitly contains a coverage set, union is an exact simple operation and a learned merger must outperform that strong control.

History-conditioned increments introduce another subtlety. The right segment's effect depends on the incoming history produced by the left segment. Sequential memory update already accounts for that dependency. This helps explain why adding an independent merge loss need not help; it is a structural concern, not an experimentally established cause of the observed failure.

### The data, target, and decision horizons are misaligned

Human demonstration windows are useful for learning representations but do not by themselves measure the differences among GR00T proposals at a shared prefix. Native scrub counts are also sensitive to temporal sampling. Coarse pooled features and sparse temporal memory can make precise contact monitoring difficult without proving that the underlying observations are inadequate.

Current `FrameFuser` reduces all camera/patch tokens to one frame token; the earlier reconstruction anchor averages camera and patch features. Those choices justify a targeted reader comparison, not a blanket recommendation to scale the backbone. Spatial grounding was already tried, and the grounded composer still lost. Another larger encoder is therefore not a sufficient rescue argument.

### The value label is legitimate but can be a poor experimental instrument

An eventual-success label after frozen continuation estimates Q under that continuation. It is not incorrect because later actions influence it. However, a small short-chunk effect can be obscured by downstream variation, and a weak continuation may fail to convert useful intermediate states into success.

Chunk-level TD can reduce the expense of collecting an independent full continuation for every candidate. It introduces approximation and bootstrap error and does not manufacture successful experience. Pairwise losses also cannot turn random continuation differences into trustworthy preferences. See the [value-learning assessment](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/docs/VALUE_LEARNING_REASSESSMENT_20260917_EN.md).

These issues justify measuring local effects separately from terminal value. They do not justify replacing native success with an easier proxy and declaring the planning problem solved.

## What to retain and change

Retain RGB/proprioception encoding, explicit current latent z_t, causal history m_t, the frozen proposal policy, and action-conditioned prediction of endpoint plus segment effects:

    (predicted_endpoint, predicted_effect) = F(m_t, z_t, committed_actions, duration)

Remove the learned composer from the first redesigned comparison. Keep ordinary sequential memory update for any legitimate multi-chunk evaluation. This is a narrower trajectory-target method, not evidence for the original composition claim.

Use an independently trained and evaluated trajectory teacher first. Train its online encoder with an explicit feature-preservation objective; if task/event supervision is introduced, declare it and share it with all relevant controls. Freeze the teacher during the first predictor comparison so that target drift and predictor error can be separated. An EMA version can follow if there is evidence to justify it. This diagnostic choice is standard, and freezing a teacher was already tried in the grounded Scrub experiment; freezing alone is not the new hypothesis.

The target should describe effects inside the **committed segment**, rather than folding an entire stochastic continuation into its representation-learning target. Separate effect prediction from the continuation critic. A critic can later consume predicted endpoint/effects, history, task and remaining time; it must train on predicted representations as well as any observed-future diagnostics to address deployment mismatch.

Start from the existing small latent predictor, with a shared spatially preserving input configuration for all model arms. Do not change DINOv2 to a larger backbone before showing that an observed-sequence reader benefits. Do not start a generalist-policy fine-tuning programme to save the method. A task-specific RGB Diffusion Policy is a valid proposal source when shared across arms and evaluated alone.

If uncertainty in outcomes is material, a distribution over effects may eventually be needed: applying a nonlinear value function to a mean latent is generally not the same as averaging value over possible outcomes. Do not add a diffusion/flow predictor until repeated same-prefix/same-action data reveal that this approximation actually causes ranking errors.

## Arena decision

| Arena | Role and verified resources | Main limitation |
|---|---|---|
| RoboCasa ScrubCuttingBoard | Retain existing negative evidence and reusable infrastructure. | Do not reopen the composer campaign. The same task with another target loss is not a fresh independent validation. |
| ManiSkill DrawTriangle-v1 | First mechanism pilot. Native task has a 300-step horizon; official demonstration registry and RGB Diffusion Policy command exist. | Visible trace makes endpoint-image prediction a strong control. Drawing uses proximity-triggered marks, not realistic contact dynamics. No ready task-specific checkpoint was verified here. |
| LIBERO-Safety `obstacle_avoidance` | Conditional application extension. Official simulator, LeRobot data, and a task-suite fine-tuned pi0.5 release are linked publicly. | Collision-free demonstrations alone do not cover failures. Static geometry and frame-wise collision aggregation are strong controls; avoidance can produce deadlock rather than success. Runtime/evaluator integration remains unverified here. |
| OGBench/CALVIN/LIBERO-LONG | Useful broader world-model/control references. | Long horizon alone does not create a segment-interior requirement. Switching to skill-policy composition changes the original problem and introduces a different baseline family. |

DrawTriangle is selected to isolate the mechanism, not because it is guaranteed to favor the proposal. The native evaluator checks accumulated reference coverage and validity of drawn points. The checked source has a batched flattened-mask expression in its validity check; begin with one environment and verify batch independence before using vectorized results. Snapshot/restore must include the marks and evaluator state. Preserve native evaluator cadence and semantics. [Native task source](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/drawing/draw_triangle.py).

The registry exposes raw demonstrations that may require replay to obtain observations. A dataset entry and a training command are not a ready-to-use RGB checkpoint. [Demonstration registry](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/utils/download_demo.py), [official RGB DP command](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/diffusion_policy/baselines.sh).

For LIBERO-Safety, begin with unchanged tabletop avoidance tasks if expansion is earned. Keep task completion and collision metrics together; simply stopping is not a successful intervention. The authors explicitly describe collision-free incompletion. [Official project](https://libero-safety.github.io/), [repository and releases](https://github.com/LIBERO-SAFETY/LIBERO-Safety), [dataset](https://huggingface.co/datasets/LIBERO-Safety/libero_safety), [pi0.5 files](https://huggingface.co/LIBERO-Safety/pi05_libero_safety/tree/main).

Neither drawing coverage nor collision avoidance inherently requires a learned composition operator: union and Boolean OR are strong task-specific aggregation baselines. This is why these arenas support testing trajectory prediction, without rehabilitating the existing composer claim.

## The next bounded experiment

These are proposed operating choices, not published hyperparameters, statistical power guarantees, or authorization to submit jobs automatically.

**Budget and scope:** one arena, one target design, one proposal policy, and one controlled reader correction at most. Allocate at most two working weeks to qualification and the first learned comparison. Use an initial cap of 24 allocated GPU-hours and 96 allocated CPU-hours after profiling, not an assertion that all training fits that cap. If resource qualification consumes the tranche, report an infrastructure outcome; do not treat the calendar as a scientific rejection. Escalation requires a concrete estimate and evidence, not another unbounded architecture search.

1. **Qualify data and native execution.** Pin revisions; inspect the available demo count; use the official RGB DP route with its declared controller/backend. Verify replay, terminal evaluation, and proposal execution. Start with the policy's actual supported horizon and eight candidates. Train and execute the same commitment duration. Do not infer a 128-step open-loop proposal from a 16-step policy output.
2. **Check the measurement chain.** Use initial, valid prefixes from independent native episodes. On a separate diagnostic panel, execute candidates and compare: (a) privileged segment effects, (b) a reader of the observed visual segment, (c) the compressed observed-segment target, and (d) predicted target. Include endpoint-only and simple map/geometry readers. Use local native effects to localize failures, while retaining terminal success for the control result. No diagnostic reader sees future frames at deployment.
3. **Run the minimal learned comparison.** Keep proposal-only as the control reference. Compare a direct history+action chunk scorer; an endpoint-prediction scorer; a frame-latent rollout with temporal readout; and a trajectory-target predictor without learned composition. Share data, allowed labels, history, candidate banks, critic objective, and reasonable tuning budgets. These are adaptations, not claimed reproductions of DINO-WM or Q-chunking. Include both efficient frame sampling and full-cadence variants where event timing matters, rather than choosing an artificially expensive frame baseline.
4. **Test selection and closed-loop behavior early.** Prioritize selected-candidate regret and actual native completion over latent MSE. Use observed-future reads only as diagnostics. A one-intervention frozen-tail experiment is not a mandatory gate for repeated learned replanning: run a small paired closed-loop comparison once the learned selectors function, rather than indefinitely extending the noisy one-intervention audit.
5. **Separate discovery from confirmation.** Suggested screening scale: 40 independent prefixes, eight candidates each, plus 50 paired native evaluation starts for the first closed-loop comparison. These small samples can identify large effects and failure mechanisms; they cannot establish a tight null. Lock the minimum worthwhile improvement and one expansion limit after throughput/pilot variance measurement and before reading confirmation outcomes. Use fresh scenes/episodes and independent training seeds for any paper claim.

The first loop answers a funding question, not a top-conference acceptance question. Continue toward a larger method study only when the trajectory arm has a reproducible decision-level advantage over strong simpler controls, or a clearly measured quality–cost advantage including all shared context and proposal costs. Do not change the winning criterion after seeing the results.

Interpret failures by location:

- If permitted observed-sequence readers cannot recover the effects, permit the one reader correction; then retire that arena/input pairing if it remains uninformative.
- If observed segments add nothing over an endpoint reader, retire the information-preservation motivation for that arena. An efficiency-only question remains possible, but must be selected explicitly rather than used as a post-hoc headline.
- If target compression loses the useful information, fix or abandon the target; more predictor training is not the right response.
- If useful targets exist but action-conditioned prediction fails, investigate supported-action coverage and outcome uncertainty before scaling.
- If a direct chunk scorer or simple geometry/map predictor matches the trajectory model on decision quality and cost, retain the simpler system and stop the trajectory-JEPA method claim for that setting.
- If a better policy accounts for all gains, report a policy improvement rather than a world-model result.

## Literature: what is established and what remains open

The venue distinction matters. I use proceedings for established results and label author-reported acceptance or preprints separately. Search results and title overlap are not a novelty verdict.

| Work | Status checked | Relevance |
|---|---|---|
| [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) | ICML 2025 proceedings | Action-conditioned visual-feature prediction and test-time planning are established. |
| [TD-JEPA, Bagatella et al.](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html) | ICLR 2026 proceedings | Policy-conditioned long-term latent prediction and successor-feature structure are established; this is not simply a one-step endpoint JEPA. |
| [V-GPS](https://proceedings.mlr.press/v270/nakamoto25a.html) | CoRL 2024, proceedings published 2025 | Generalist-policy action reranking with learned offline-RL value is established. |
| [Q-chunking](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html) | NeurIPS 2025 proceedings | Chunk-level TD is a strong established training baseline, not a new JEPA contribution. |
| [Compositional Planning with Jumpy World Models](https://arxiv.org/abs/2602.19634) | Listed in the [official ICML 2026 programme/downloads](https://icml.cc/Downloads/2026) | Policy-induced multi-step occupancy prediction, cross-timescale consistency and sequencing policies. A relevant baseline if switching to skill composition, not identical to learned summaries of explicit action segments. |
| [AnySafe](https://any-safe.github.io/) | Official author page reports ICRA 2026 acceptance | Runtime-conditioned latent safety filtering is already studied. A safety-head pivot is not automatically a novel method. |
| [LIBERO-Safety](https://libero-safety.github.io/) | Official author page/repository report ECCV 2026 acceptance | A possible application arena and released resources, not evidence that our architecture works. |
| [MemoryVLA++](https://arxiv.org/html/2606.09827v1) | June 2026 preprint; acceptance not verified here | Memory plus latent future imagination is already proposed. Its described imagination conditions on current observation and instruction; it is not the same as comparing explicit counterfactual action chunks with a segment composer. |
| [Semigroup-JEPA](https://arxiv.org/abs/2609.10464) | September 2026 preprint; acceptance not verified here | Relevant consistency/physics-generalization overlap. Its name alone does not establish equivalence to a learned interior-effect composer. |
| [Calibrated Predictive Safety for Heterogeneous Robots](https://arxiv.org/abs/2608.17496) | August 2026 preprint; acceptance and empirical reliability not independently verified | Direct conceptual overlap with proposal → JEPA → risk/progress scoring → shield. Do not use it as evidence of feasibility or ignore it when making an originality claim. |

The remaining opportunity is narrower than “memory + JEPA + value + replan”: a **specific action-conditioned segment representation** that delivers demonstrable decision benefits beyond those ingredients. Borrowing a principle from another field is legitimate, but merely attaching a composition loss or renaming an existing critic is insufficient. A new operator formulation would need a well-defined sufficient representation, boundary conditions and a planning use that cannot already be obtained from ordinary recurrent updates.

## Final assessment

There is enough conceptual substance to justify one controlled continuation of trajectory-target learning. There is not enough evidence to keep defending the current learned composer, to promise that a different critic will rescue it, or to begin a large multi-arena training campaign.

Keep the long-term latent-world-model direction. Change the immediate claim and experiment. If the narrower predictor cannot earn a decision-level advantage under the bounded redesign, move to a genuinely new question within latent world models rather than adding another layer to this architecture.

Work completed for this report: static local code/report inspection, official-source literature and arena checks, and this new decision document. No simulator execution, data/model download, training, or Slurm submission was performed. Public resource availability was checked at source level, not by reproducing the released systems.
