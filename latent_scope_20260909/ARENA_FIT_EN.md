# Existing-arena assessment for trajectory-predictive JEPA

Date: 9 September 2026. This assessment supersedes the custom PushCube pilot in the earlier proposal. It responds to the requirement to use released robot tasks and datasets, with JEPA-style latent prediction remaining the scientific core. New research documents will be written in English.

## Decision

**Custom task construction is not necessary to investigate path-dependent latent prediction.** RoboCasa365 already contains relevant tasks with implemented success predicates and registered demonstration datasets. Start the feasibility assessment with **ScrubCuttingBoard** and **RinseSinkBasin**, using their original environments, instructions, action interfaces, horizons and success criteria.

However, these tasks justify a question about contact/coverage over a trajectory, not a claim that noncommutative signatures are necessary. The previous signature-specific method remains unqualified. We should adapt the hypothesis to existing evaluation semantics rather than modify the benchmark to make signatures useful.

This was a static source and documentation audit. No datasets were downloaded, no episodes were replayed, and no simulator or model was run. A registry entry confirms an officially provided dataset route; it does not establish successful download/replay on the user's cluster.

## What the original proposal would have required

The proposed PushCube A→B/avoid/dwell variant required new regions, a new task monitor, an action adapter and additional trajectories. It was not an existing dataset/task package. Those modifications are no longer the recommended starting point.

Distinguish three operations:

| Operation | Needed with a released arena? | Changes the benchmark task? |
|---|---|---|
| Encode existing RGB, sample action windows, compute latent targets | Yes, ordinary model training integration | No |
| Implement the model's observation/action interface and planner | Yes | No, if native semantics are preserved |
| Add forbidden zones, instructions, termination rules or success predicates | No for the recommended tasks | Yes |

Additional training rollouts, if later needed, can be collected in the unchanged environments. They must be disclosed as extra data, not silently counted as part of the original offline dataset.

## Native task evidence

The RoboCasa source revision inspected was `4f8a2980def75a55dff96b990745b83540425f09`. Pin that revision and compatible dataset/assets before any execution. The following are source-level observations, not experimentally verified claims about visual observability or difficulty.

| Native task | Actual implemented criterion | Dataset route | Role |
|---|---|---|---|
| **ScrubCuttingBoard** | While the sponge is grasped and contacting the board, records spatially distinct contact positions. Requires at least five recorded positions, a contact-position bounding-box diagonal of at least 0.1 m, and gripper separation from the sponge. | Human pretrain **and target** entries | Primary contact/coverage candidate |
| **RinseSinkBasin** | Latches whether water has been on while the spout faced left, center and right; succeeds when all three have been visited. | Human pretrain **and target** entries | Simpler history/coverage control |
| RinseBowls | Tracks water-contact runs, latching each bowl as rinsed after 25 qualifying updates; also checks gripper separation. | Human pretrain entry; no target entry in inspected registry | Optional duration-related extension, not part of the same target-task protocol |
| RinseCuttingBoard | Accumulates qualifying hot-water rinsing checks to 100, then requires the faucet to be off. | Human pretrain entry; no target entry in inspected registry | Optional cumulative-exposure extension |

Sources: [ScrubCuttingBoard implementation](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/sanitizing_cutting_board/scrub_cutting_board.py), [RinseSinkBasin implementation](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/cleaning_sink/rinse_sink_basin.py), [RinseBowls implementation](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/washing_dishes/rinse_bowls.py), [RinseCuttingBoard implementation](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/environments/kitchen/composite/sanitizing_cutting_board/rinse_cutting_board.py).

Important boundaries: ScrubCuttingBoard does not establish that circular motion or a particular ordering is required. RinseSinkBasin permits the three regions to be visited in any order. RinseCuttingBoard's implementation accumulates qualifying checks; do not reinterpret it as uninterrupted physical exposure for a fixed number of seconds. Several counters are updated by evaluation/state-update routines, so replay must use the native lifecycle consistently rather than call the success function extra times to generate labels.

The first two tasks are useful precisely because their native outcomes depend on what happened earlier. That does **not** prove an image-only encoder lacks the necessary information in the actual data, or that a recurrent policy plus a simple monitor will fail.

## Dataset and evaluation package

The [official dataset documentation](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html) describes 100 human demonstrations per pretraining task and 500 per target task. Target tasks use a set of 10 kitchen scenes distinct from the pretraining scene collection. Target composite datasets also include per-frame subtask annotations. These are dataset categories, not a promise that any particular pretrained JEPA checkpoint exists.

The [pinned registry](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/robocasa/utils/dataset_registry.py) contains:

| Task | Target human dataset path | Registered evaluation horizon |
|---|---|---:|
| ScrubCuttingBoard | `v1.0/target/composite/ScrubCuttingBoard/20250816` | 1200 |
| RinseSinkBasin | `v1.0/target/composite/RinseSinkBasin/20250816` | 1350 |

The released format includes camera video, proprioception, actions, timestamps and episode metadata. The documented extras include simulator states for replay, not normal visual-model inputs. See [Using Datasets](https://robocasa.ai/docs/build/html/datasets/using_datasets.html).

Use the official target-task training/evaluation regime and report a **two-task subset**, not a full RoboCasa365 leaderboard result. If target demonstrations are used for training, the result is not zero-shot transfer to unseen target kitchens. An independent pretrain-only transfer experiment must be labeled separately. Preserve the official horizon update and native success rules.

The method needs ordinary dataset/agent adapters. It should not inherit the earlier 2D PushCube action restriction, 64-step episode assumptions, or an arbitrary 128×128 single-camera interface and call those native RoboCasa settings. Camera selection and action normalization must be matched across arms; keep official task/controller semantics. The exact data volume, controller configuration, dataset readout, asset availability and baseline runtime still require an execution preflight.

## Why the familiar alternatives are not interchangeable

| Arena | Released resources | Fit for the earlier claim |
|---|---|---|
| CALVIN | Dataset, simulation and official five-subtask evaluation | Useful for language-conditioned execution. The evaluator supplies one subtask at a time and resets model state per subtask; it does not expose the whole chain to the agent for joint planning. Giving future instructions changes the protocol. |
| LIBERO-LONG | Demonstrations, tasks and evaluation | Useful for multi-stage manipulation. The inspected tabletop success function evaluates a conjunction of current predicates; a long task name does not establish trajectory-order evaluation. |
| RoboCasa365 selected native tasks | Dataset registry, instructions, replay assets and native historical criteria | Strongest existing-arena match found for contact/coverage trajectory prediction. |
| LIBERO-Safety physical suites | Released environments/assets, LeRobot demonstrations, model links | Good alternate arena for path safety. Does not independently justify order-sensitive signatures; data coverage and the end-to-end collision reporting path require further verification. |

Sources: [CALVIN evaluator](https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/evaluation/evaluate_policy.py), [LIBERO tabletop success evaluator](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/problems/libero_tabletop_manipulation.py), [LIBERO datasets](https://libero-project.github.io/datasets), [LIBERO-Safety repository](https://github.com/LIBERO-SAFETY/LIBERO-Safety).

Task descriptions also require checking against code within RoboCasa. For example, HeatMug and PreSoakPan have multi-step language descriptions, but the inspected implementations check final configurations rather than independently verifying the described order. They should not replace the two selected tasks merely because they sound procedural.

## LIBERO-Safety: available, but not automatically a complete training solution

The authors' [project page](https://libero-safety.github.io/) and repository identify physical avoidance suites and provide links to data, assets and model weights. The repository still labels the data-generation pipeline as forthcoming. Its generic `libero/lifelong/evaluate.py` reports success; that file alone is not verification of the full collision/violation aggregation used for the paper.

The actual [LeRobot metadata](https://huggingface.co/datasets/LIBERO-Safety/libero_safety/blob/main/meta/info.json) reports 19,664 episodes, 3,443,735 frames, 15 task indices, 20 Hz, two RGB streams, an 8-dimensional state field and 7-dimensional actions. The supplied metadata declares one training split. Those task indices must not be assumed to cover every benchmark suite and difficulty level without checking the mapping.

The demonstrations are described as collision-free. Such data can support behavior and latent-prediction learning but does not establish enough information to learn reliable collision boundaries for arbitrary planner candidates. This is a training-support limitation, not a need to invent new evaluation tasks. Prefer physical avoidance over semantic refusal if studying action-conditioned latent dynamics.

## What this does to the method recommendation

The defensible existing-arena question is now:

> Can an action-conditioned JEPA predict a compact, reusable representation of contact and coverage accumulated over a future segment, improving native task completion over frame-target prediction under matched data and planning budgets?

This keeps the visual encoder and latent predictor central. The future target comes from a sequence of encoded observations, and planning uses predicted segment outcomes. It does not substitute explicit physical-state dynamics for the latent model.

**Do not lock in signatures yet.** Native coverage conditions admit much simpler summaries. A bitset can represent visited spout orientations; contact counting and spatial extrema can represent much of the scrub predicate. These are serious alternative explanations. A new method has to outperform matched visual/recurrent implementations of those alternatives, not a terminal-L2 baseline intentionally denied history.

Required controls:

1. Recurrent imitation policy trained on the same demonstrations.
2. Frame-target JEPA with full latent rollout and a learned contact/coverage monitor.
3. The same encoder and planner with simple pooled/event-count segment targets.
4. An unstructured learned trajectory encoder with matched capacity.
5. Signature-based trajectory targets, only as a competing design until they demonstrate a specific advantage.

Simulator predicates may label training diagnostics, but any such supervision must be disclosed and matched. At evaluation, the visual method cannot read privileged contact lists or washed-region flags. A privileged monitor is an oracle diagnostic, not the main deployable baseline.

The first experiment should use the released demonstrations to compare target/readout adequacy and reproduce a capable behavior baseline. Only then evaluate whether different latent targets improve planning on native episodes. Do not infer model headroom from the mere presence of history in the success function. In particular, the earlier repo's memory/progress nulls require this question to survive strong recurrent and simple-monitor baselines.

## Remaining uncertainty and current status

**Established by this audit:** appropriate native tasks exist; their historical criteria are explicit in source; both primary tasks have pretrain and target dataset registrations; custom PushCube task construction is unnecessary.

**Not established:** successful dataset download/replay, a ready JEPA checkpoint for these tasks, demonstration coverage of failed candidate plans, contact visibility from allowed cameras, planning headroom, training cost, or signature-method novelty.

The main contribution is still intended to be a latent-world-model method, not the creation of a task subset. This assessment identifies an existing arena for testing that ambition; it does not turn a plausible target design into a qualified conference method.
