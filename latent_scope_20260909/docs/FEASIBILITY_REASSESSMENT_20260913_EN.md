# Feasibility reassessment after the event-aligned Stage-B profile

Date: 13 September 2026. This assessment updates the 9 September recommendation using the implemented code and the completed server artifact, not only the pasted execution summary.

**Decision: retain compositional trajectory-JEPA as the research hypothesis, repair the diagnostic and comparison design, and qualify one smaller native trajectory task. Do not launch the six-arm training campaign or start a broad GR00T fine-tuning campaign yet. The current evidence does not establish that the policy has no useful selection headroom, and it has not tested composition.**

## 1. New evidence: job 52226 has finished

Read-only SSH inspection checked both `squeue` and `sacct`. Job 52226 was absent from the active queue and recorded as COMPLETED, exit 0:0, elapsed 01:40:07. The result was completed on 12 September at 11:52:39 UTC. A copy is preserved in [evidence/stage_b_event_52226_result.json](evidence/stage_b_event_52226_result.json).

| Event-profile observation | ScrubCuttingBoard | RinseSinkBasin |
|---|---|---|
| Prefixes | 1 | 1 |
| Candidate successes | 8/8 | 0/8 |
| Candidate maximum progress scores | All 1.0 | All 0.0 |
| Source event's precondition | Four accepted contacts, sweep milestone reached | Two of three regions already rinsed |
| Actual reconstructed branch history | Four accepted contacts | **Zero regions rinsed** |

Pooled baseline and oracle completion are both 50%; completion and maximum-progress gains are both zero. The projected serial cost for 40 prefixes per task is 66.66 allocated GPU-hours. These are two contexts, not sixteen independent tests of general planning headroom. In this panel, task identity alone explains every binary label, so it provides no within-prefix action-ranking supervision.

The key problem is the Rinse anchor mismatch. `generate_source_prefix` chooses an event from the original successful policy rollout. `reconstruct_canonical` then replays its action prefix, despite known long-replay drift. The branch checks verify that every candidate starts from the same **reconstructed** carrier. They do not check that this carrier still has the intended event precondition. Consequently, all local equality checks can pass while the advertised event alignment has disappeared.

This does not prove that the history-reset code alone is wrong, or identify every cause of trajectory drift. It does establish that this Rinse experiment did not test the intended two-regions-to-three-regions decision. It must not be used to reject that intervention's headroom. The Scrub prefix is a valid but locally saturated example: all sampled candidates succeed there.

Required repair: choose/validate events on the actual carrier that will be branched, or retain and restore a verified live snapshot from that carrier. Compare native history and the relevant physical precondition against the anchor specification before evaluating any candidates. If it fails, classify the sample as an invalid anchor. Do not repair it by manually setting `washed_loc` to the desired value while retaining a different physical state.

## 2. What a weak policy does and does not imply

The running proposal is **GR00T N1.5 post-trained on composite_seen**, not the original standalone Diffusion Policy proposal. Its 3/10 and 6/10 baseline results establish basic usability, not a precise performance estimate or improvement from training.

The quantity currently measured is approximately:

    Q^pi(prefix, chunk) = P(native completion | execute chunk, then follow fixed policy pi)
    selection headroom = best candidate Q^pi - baseline candidate Q^pi

There are several different bottlenecks:

| Bottleneck | Diagnostic | Appropriate response |
|---|---|---|
| Proposal support | No candidate reaches a useful intermediate outcome from a valid prefix | Improve or diversify the proposal, extend its supported horizon, or change the task |
| Continuation | Chunks create useful different outcomes, but the fixed tail policy loses or erases their benefit | Test repeated decisions or one improved continuation; do not call the world model useless |
| Observation/representation | True outcomes differ but permitted visual histories cannot support a reliable readout | Check spatial features and history before training the predictor |
| Prediction | True-target scoring works but predicted-target scoring fails | Study the world model |
| Composition | Unstructured segment prediction works, but composition adds nothing | Reject or narrow the composition claim at that operating point |

An oracle over eight chunks followed by one fixed policy is an upper bound for a selector restricted to that same experiment. It is **not** an upper bound for a planner that changes actions repeatedly. Conversely, “repeated planning might help” is only an untested alternative, not permission for an unlimited training campaign.

A stronger proposal can increase or decrease selection headroom: it may supply useful new behaviours, or make every candidate succeed. Optimizing standalone success is therefore not the same as creating a good scientific setting for studying selection. Track candidate support, baseline completion and the selector's possible improvement separately.

## 3. Stage C is runnable, but not yet a faithful decisive test

The following are static code findings. No model execution or numerical invariance test was run during this assessment.

| Finding | Evidence in this checkout | Consequence and correction |
|---|---|---|
| Memory is a short window, not episode memory | `stage_c/data.py` slices `history_steps=3`; `CausalContext.forward` in `stage_c/models.py:73` calls GRU without an incoming hidden state | All arms can forget earlier accumulated coverage. Use an actual causal prefix encoder or a carried state with a defined reset/burn-in protocol. Do not inject privileged progress at deployment. |
| Composer has no left/right or temporal identity | `stage_c/models.py:148`: Transformer receives query, left, boundary, right tokens without positional/segment embeddings or an asymmetric mask | In evaluation mode, swapping left/right is an input permutation that preserves the query outputs, up to numerical error. A boundary token alone does not label the sides. This could be intentional for set union, but is not a general ordered segment composer. Define the algebra and add boundary/segment identity where needed. |
| Direct-value control ignores action order | `stage_c/models.py:341`: position-free Transformer followed by mean pooling | The control is permutation-invariant over action timesteps at evaluation. Add temporal encoding before using it as a strong action-sequence baseline. |
| Frame control is not a full temporal monitor | `FrameRolloutArm` gives its value head only endpoint and mean of predicted frames | Compare against an ordered sequence readout with matched history/capacity. The current baseline is specifically an endpoint-plus-mean readout. |
| Composition ablation changes additional supervision | `first_endpoint_anchor` is inside the composition-weighted split term | Give both segment arms the same subsegment endpoint targets; vary the intended composition constraints separately. Otherwise a gain could be ordinary multi-horizon supervision. |
| Candidate evaluation accepts multiple windows per branch | `SegmentWindowDataset` creates all sliding windows even for candidate entries; `candidate_metrics` groups by group id and takes the maximum score | Require exactly one designated decision window per `(prefix, candidate)`, with shared pre-action context across candidates. Otherwise a future continuation window can masquerade as an action choice at the original prefix. The current code does not enforce this invariant. |
| Spatial information is discarded early | `encode_stage_c_offline.py:69` keeps only DINOv2's CLS token per camera | This is not the patch-level DINO-WM interface described earlier. Measure contact/coverage readout using CLS versus spatial tokens or a small patch grid before blaming the new target. CLS inadequacy is a risk, not an observed failure here. |

Additional claim boundary: the current scorer uses the directly predicted summary; composed summaries enter consistency losses. That tests a composition **regularizer**, not yet a planner executing composition over multiple imagined segments. A positive result could justify that narrower method, but claims about compositional rollout require an explicit evaluation of the composed inference path.

The target reconstruction anchor also reconstructs only the mean future latent. This does not establish that the teacher retains coverage or distinct events. A target-readout ceiling is needed. The reported forward/backward smoke test checks numerical plumbing, not these scientific properties.

## 4. How to improve the policy without changing the research question

### First: exploit the actual supported horizon

The pinned GR00T `PandaOmronDataConfig` uses 16 action indices. The local `policy_action_chunk` slices the first eight. Therefore **8-to-16 is the first controlled horizon test**, after checking actual returned tensor shapes. It need not begin with retraining. [Pinned upstream configuration](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/gr00t/experiment/data_config.py).

Sample one bank of full 16-step chunks and compare the first eight against all sixteen from the same valid prefix. Keep the continuation policy's execution cadence fixed at eight. The current harness reuses `candidate_chunk_steps` for the wrapper, source rollout and continuation, so merely changing that field would change several variables at once. Separate intervention duration from continuation cadence and prefix selection.

A 32-step intervention is not obtained by copying actions or pretending the checkpoint emits 32 trained steps. It needs either an explicitly extended action head or a well-defined feedback option. Building a second chunk using a simulator's future image is an oracle diagnostic; it is not an action sequence the visual planner already knows at the initial decision.

### Second: distinguish proposal quality from tail-policy quality

For valid prefixes, retain intermediate outcomes at 8/16 steps, local milestone increments and time to the next milestone, alongside eventual completion. Maximum progress over a long continuation can erase useful early differences.

If intermediate quality differs but completion does not, compare the same candidate bank under the original continuation and one independently improved continuation. Use multiple continuation seeds and separate seeds for selection and evaluation. Do not interpret the maximum of noisy single-rollout labels as an achievable expected-success gain.

### Third: one bounded specialist adaptation, only if needed

If the valid-prefix audit shows inadequate proposal support or a repeatable recovery failure, use **task-specific action-head adaptation of the existing GR00T checkpoint**. Keep the visual/language backbone frozen initially; tune the projector/action head, or use the supported action-head LoRA route. These switches exist in the [pinned official fine-tuning script](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/scripts/gr00t_finetune.py). Memory use and runtime still need profiling.

Use declared, disjoint training data with successful current-runtime interactions and successful recovery segments where available. Failure trajectories are valuable world-model/value data, but blindly behavior-cloning failed actions is not a recovery policy. Historical recorded observation/action pairs remain usable as offline data even if they cannot be action-replayed exactly in a different runtime; disclose the runtime shift.

Compare the original proposal, the specialist, and a fixed mixture retaining original candidates. This can preserve useful modes that specialist fine-tuning removes. Freeze the selected proposal for every JEPA/control arm, and evaluate it alone. Improvement from the policy adaptation must not be attributed to composition.

Do not simultaneously change model family, horizon, sampling temperature and continuation. A broad policy improvement project could consume the runway without ever testing the latent-world-model question.

## 5. Recommended task portfolio

**Keep RoboCasa as the application arena. Add ManiSkill DrawTriangle-v1 as the first smaller mechanism pilot; consider PegInsertionSide-v1 next for contact-manipulation transfer.** These are existing tasks; no new evaluation predicates are proposed.

| Task | Resource evidence and scientific role | Limitation |
|---|---|---|
| RoboCasa ScrubCuttingBoard | Existing local execution stack; native interaction coverage. Remains the application candidate. | The latest prefix is saturated. Need valid earlier/middle decision prefixes, not only the final contact milestone. |
| RoboCasa RinseSinkBasin | Existing stack; simple coverage control. | Repair source-to-carrier event alignment first. A three-bit monitor is a serious competing explanation. |
| **ManiSkill DrawTriangle-v1** | Native 300-step task; released demonstrations and an explicit RGB Diffusion Policy baseline command. Best added pilot for accumulated path effects and segmentation. | Drawing is a proximity-triggered dot model, not detailed contact physics. The drawn trace is visible in the current image; do not claim hidden-history necessity. |
| ManiSkill PegInsertionSide-v1 | Released demonstrations, native insertion evaluation, motion-planning data route and a state-DP baseline command; general RGB training support is available. | The inspected baseline script does not provide a task-specific RGB Peg command. Visual policy adaptation is still work. Final insertion success does not by itself establish a path-history objective. |
| ManiSkill StackCube-v1 | Explicit released RGB-DP training recipe; useful backup if a tested RGB integration is the priority. | A general manipulation control, not a replacement for coverage semantics. |

ManiSkill resource checks: [official demonstration registry](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/utils/download_demo.py), [official DP baseline commands](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/diffusion_policy/baselines.sh), [dataset/replay setup](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html). These are source-level availability checks; no new arena was installed or run here. Pin simulator/data revisions and preserve the documented IL evaluation horizon and backend rather than silently changing the task.

DrawTriangle's implementation accumulates reference-point coverage and checks that valid drawn points remain near the target. Its visible trace makes it useful for testing prediction of accumulated effects, while allowing a strong endpoint-image baseline. [Native source](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/drawing/draw_triangle.py). Start its evaluator qualification with a single environment: the inspected batched success code uses a flattened mask over drawn points, so independence between environments must be checked before treating a vectorized batch as independent episodes. Snapshot state includes drawing counters/coverage and marks, not just robot joints.

Do not assume a shorter task guarantees lower walltime. ManiSkill uses a different rendering/simulation stack, and this cluster has already had GPU-render access problems. Its first job must verify rendering and replay. Retain RGB/proprioception as learned-policy/model inputs; privileged state or motion planning can serve as an oracle or data generator, not an undisclosed deployment input.

RinseBowls has relevant native rinse accumulation, but the earlier registry audit found a pretrain-only dataset and a longer horizon. It is not the next low-cost alternative, and the existing target-posttrained checkpoint is not a verified specialist for it. A bare robosuite Wipe environment similarly is not a verified full dataset/policy package in this assessment.

## 6. Concrete next sequence and stopping rules

1. **Repair interpretation and code invariants.** Preserve job 52226 as completed, record the Rinse anchor mismatch, fix temporal identity/history/evaluation issues before treating Stage C as decisive. Add narrow tests for each invariant on the permitted compute node.
2. **Run a valid-prefix 8-versus-16 profile.** Proposed initial panel: six distinct episodes per task, distributed across early/middle/late incomplete stages, eight nested candidates each. This is a screening design, not a powered null test. Validate anchor semantics before spending on continuations. Keep candidate and continuation RNG handling separate.
3. **Measure short outcomes first, then selected full continuations.** Inspect all candidates at short horizons. A predeclared random prefix subset gets full native continuations; targeted full continuations can diagnose mechanisms but cannot estimate an unbiased task-wide success gain. Profile reset/replay, rendering, policy inference and transfer separately before extrapolating cost.
4. **In the same research phase, qualify DrawTriangle's released data and RGB policy.** Only proceed to a new model experiment after render/replay/native evaluator checks and a nontrivial policy result. No general multi-arena training campaign yet.
5. **Permit a small target-adequacy study before a full terminal-success headroom campaign.** On separate current-runtime training/validation windows, compare true-sequence, compressed-target and predicted-target readouts. This may expose an unusable target cheaply. It cannot establish a planning benefit or bypass the later native-success comparison.
6. **Run the corrected learned comparison where headroom exists.** Include the ordered direct-value and full-sequence controls. Test matched multi-horizon supervision separately from composition, and evaluate both direct and composed predictions if claiming compositional inference.

Use one explicitly capped repair tranche; a proposed ceiling is 12 allocated GPU-hours for the RoboCasa diagnostic, excluding any separately approved policy-training or new-arena installation budget. This is a resource limit, not an estimate that the full panel will fit. If it cannot fit, reduce the descriptive panel transparently or revise the budget; do not manufacture a precise no-headroom conclusion from a tiny sample.

Stop expanding this RoboCasa policy–task pairing if a semantically valid horizon audit plus one justified proposal/continuation intervention still yields no useful variation. That is an operational reason to move the mechanism test to the qualified smaller arena, not proof that every compositional latent world model is impossible.

Stop the **composition claim** if, with adequate targets, valid candidate groups and useful oracle headroom, composition fails a sufficiently powered comparison against the matched unstructured model and strong direct-value/frame controls. If a pilot interval remains wide, label it inconclusive. Do not turn a one-seed +5-point threshold into a universal falsification rule.

## 7. Overall judgement

The current evidence supports continued bounded investigation. It does not support optimism about a top-conference result, and does not support the assertion that a weak diffusion policy has already invalidated the idea. The most immediate obstacles are an event-alignment defect and a Stage-C scaffold that does not yet implement the intended scientific comparison.

The next useful result is a valid demonstration that permitted visual information can distinguish useful candidate outcomes, followed by evidence that composition helps predict or use those outcomes. A stronger policy is one possible instrument for obtaining that setting, not the default next research objective.

Assessment actions: read-only cluster status/result retrieval, static code inspection, official-source resource checks and local documentation/evidence updates. No policy/model training, simulator execution, job submission or implementation-code changes were performed.
