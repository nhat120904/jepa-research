# ORDER-JEPA: Dataset/Task Selection and Pipeline Qualification

Review date: September 7, 2026. This document supersedes the benchmark order and pilot schedule in the initial proposal. The official PushT data, original simulator, and authors' released DINO-WM checkpoint have now been used for the Stage-A audit in [RUN_STATUS.md](RUN_STATUS.md). The decision is `STOP_OR_FIX_STAGE_A`; no JEPA training has run.

## Current decision

ORDER-JEPA is a mathematically motivated hypothesis whose physical premise passes on PushT, but whose predictor-specific decision premise fails with the original released checkpoint. It is not a qualified learning direction in this setup. Dataset, task, and model must be selected jointly. The unit to audit is `(data distribution, observation/history, encoder, action interface, predictor, goal cost, horizon, planner budget)`. The name “PushT” alone does not specify this system.

The first candidate is **DINO-WM with the official `pusht_noise` data and original PushT goal-reaching protocol**. ORDER training begins only if the audit finds a predictor-induced gap. A rendered unicycle is an additional mechanism diagnostic. Three-dimensional navigation, Rope, and Granular are not yet main-benchmark commitments.

If no practical setting contains an order effect that is observable, decision-relevant, and mispredicted by a strong baseline, confidence in the idea should decrease. Tasks must not be cycled until the method happens to win.

## Verified implementation facts

The DINO-WM source was inspected at commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c` in the [authors' repository](https://github.com/gaoyuezhou/dino_wm/tree/0a9492fa12044b852ae9e001cc74604b79c8bb0c). Its README points to an [OSF data and checkpoint release](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28). Download integrity and checkpoint compatibility remain unverified.

| Component | Inspected recipe | Design consequence |
|---|---|---|
| Dataset | `pusht_noise`; image/video files, `rel_actions.pth`, `states.pth`, `velocities.pth`; train/validation split | Do not substitute another PushT dataset and compare directly with paper results |
| Observation | 224×224 RGB plus pusher position and velocity | The primary reproduction is RGB + agent proprioception, not pixels-only |
| Visual encoder | Frozen DINOv2 ViT-S/14 `x_norm_patchtokens` | Test sensitivity to small pose changes; the DINO name does not certify sufficient information |
| Predictor | Default depth-6 Transformer, history 3, causal modeling | Use the resolved configuration saved with the checkpoint; insufficient history can cause aliasing |
| Time/action | `frameskip=5`; five 2D actions concatenated for one predictor step | Repeating one action five times is a different dynamics interface |
| Environment action | Relative displacement, scale 100, converted to a PD-controller position target | Order effects may arise from servo dynamics and agent momentum; object effects need separate evidence |
| Default planning | `goal_H=5`; CEM horizon 5, 300 samples, 30 iterations, top-k 30; terminal visual + proprio MSE | Five model steps equal 25 control steps when frameskip is 5, not 25 physics substeps |
| Wrapper success | Joint agent/object position error below 20 and object-angle error below pi/9 | An overlap-only metric is a different benchmark |

Sources: [dataset](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/datasets/pusht_dset.py), [training configuration](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/train.yaml), [encoder](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/encoder/dino.yaml), [predictor](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/predictor/vit.yaml), [action grouping](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/datasets/traj_dset.py), [simulator](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/env/pusht/pusht_env.py), [planning configuration](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/plan_pusht.yaml), [cost](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/planning/objectives.py), and [success metric](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/env/pusht/pusht_wrapper.py).

This concern is empirically plausible. DINO-WM v2 reports 90% PushT success with patch features and 44% with CLS features. Its data ablation reports 48%, 72%, and 92% for 1,000, 5,000, and 18,500 trajectories. These are author-reported results from separate ablations, not reproductions in this project, and they do not isolate the encoder as the only cause. See [Tables 2 and 5](https://arxiv.org/html/2411.04983v2).

## Concrete experiment suites

| Suite | Dataset and model | Task | Role and gate |
|---|---|---|---|
| P0 — reproduction | Official `pusht_noise` and released DINO-WM PushT checkpoint | Original goal-reaching evaluator | Validate the setup; record data/checkpoint checksums, resolved configuration, and seed manifest |
| P1 — main candidate | Same base data plus true branches shared by every method; same frozen DINO representation | PushT goal reaching plus object pose and fixed-candidate regret | Train ORDER only if the audit finds order errors that affect physical outcomes |
| U — mechanism | Self-generated unicycle state data, followed by a top-down renderer with heading marker and encoder | Pose reaching through rotate/translate compositions; holonomic counterpart as control | State version isolates dynamics/loss; image version adds perception; this is not an external benchmark |
| P2 — conditional model transfer | Official LeWM PushT release with its own compatible data, loader, and checkpoint | PushT under the LeWM protocol | Test model dependence after P1; it remains the same task family |

P1 retains an original-protocol goal-reaching panel. Its mechanism panel uses anchors with identical warm-up and history, and goals obtained from feasible simulator rollouts. Prespecify no-contact, contact-onset, and multi-contact strata, report their natural distribution and separate results, and never select anchors using ORDER gains.

The main proposed horizon is five model steps. Prespecified boundary tests use two and ten steps, equal to 10 and 50 control steps at frameskip 5. Any horizon change must report goal horizon, prediction horizon, actions executed per replan, and total episode budget.

P2 cannot be pooled directly with P0. The [official LeWM repository](https://github.com/lucas-maes/le-wm) points to the [authors' collection](https://huggingface.co/collections/quentinll/lewm). The README inspected during this review describes HDF5, while the [current PushT configuration](https://github.com/lucas-maes/le-wm/blob/main/config/train/data/pusht.yaml) names `pusht_expert_train.lance`. A compatible release of dataset, loader, and checkpoint must be pinned. The collection procedure has not been verified, so this release cannot support an expert-free claim. Fine-tuning a predictor over a frozen LeWM encoder is not an end-to-end LeWM reproduction.

## Locating the actual ceiling

For each state and goal, first fix a candidate set (A), then replay every candidate in the simulator. Use those same outcomes for the scorers below. Candidates are generated independently of the evaluated method and include ordinary proposals and swapped-order pairs. Candidate sets containing a known witness trajectory are used only for solvable ranking tests and are not mixed with ordinary CEM success.

| Diagnostic | Procedure | What failure still permits |
|---|---|---|
| Coverage/task feasibility | Evaluate true physical cost for every true rollout; verify a good candidate exists and replay a witness trajectory | Candidate distribution, constraints, horizon, or reset may be inadequate; prediction loss is not yet implicated |
| Representation + cost | Encode final **true** images/proprioception with the exact checkpoint and rank candidates with the exact latent objective | The metric may not represent the task, or the representation may lack useful information; better prediction may not help |
| Learned dynamics | Replace encoded true futures with predicted futures while preserving candidates and cost | The gap to the previous row is surrogate error on this candidate set; test whether it is specifically order-related |
| Observation/history | Find similar histories with different velocity/contact states and divergent futures; compare a full-state dynamics diagnostic | Observations or history may be insufficient, or a point predictor may average stochastic/multimodal futures |
| Capacity/optimization/data | Overfit a tiny batch for plumbing, then measure held-out errors across data and predictor size | Tiny-batch fit says nothing about generalization; a larger model winning does not establish the necessity of ORDER |
| Closed-loop search | Compare simulator dynamics + physical cost, simulator dynamics + latent cost, and learned dynamics under the same CEM budget | This tests the real planner; different CEM runs generate different candidates, so it is not a pure ranking isolation |

The physical diagnostic may normalize position and angle errors by the wrapper tolerances and report original success; object-only position and angle are additional metrics. State labels are used for oracle diagnostics, probes, and metrics, not ORDER training. A privileged-state controller is a diagnostic, not an input-matched baseline.

“Oracle” here means an empirical reference, not a mathematical upper bound on episode success. Simulator dynamics with terminal physical cost may still miss necessary detours, and approximate CEM need not find an optimum. Success differences among rows cannot be treated as an additive causal decomposition.

On fixed (A), define

\[
R(q)=J_{\rm true}(a_q)-\min_{a\in A}J_{\rm true}(a),
\]

where (a_q) is the candidate selected by scorer (q). Report regret with predicted futures and with encoded true futures. A reliable gap between them, together with order-dependent physical margins, is the reason to invest in a predictor loss.

## Decodability is insufficient

A probe for `(x,y,theta)` must use held-out episodes, represent angle as `(sin theta, cos theta)`, and measure local error at the scale of swapped-order displacement. High global R-squared can miss the small difference required for action selection. Weak probe performance also does not prove complete absence of information because it depends on probe class, data, and optimization.

Information may be present while Euclidean distance fails to use it. [SCALE](https://arxiv.org/abs/2608.16287) distinguishes state decodability from geometry used by planning. [The Objective Is the Bottleneck](https://arxiv.org/abs/2608.12959) reports an objective failure in a TwoRoom reproduction; it is a preprint and not direct evidence for PushT. These studies justify the diagnostic but do not replace it.

If encoded true outcomes rank the physical task incorrectly, ORDER may predict its latent target faithfully while remaining physically useless. Changing the representation, history, or cost then changes the hypothesis or protocol and requires rerunning all baselines.

For small order effects, measure physical displacement, latent displacement, and cost margin. Repeated branches estimate real noise. In deterministic rendering, additionally test sensitivity to valid observation perturbations and resolution. A small (h) can place the signal below pixel or feature resolution; division by (h^2) amplifies noise. Increasing duration may recover a finite contact effect but no longer validates Lie-bracket asymptotics.

## Required corrections to branch collection

1. **Reset and history.** The original `_set_state` restores poses and agent velocity, but its state vector does not restore block linear/angular velocity. It also advances physics after assignment. Calling `prepare(seed, stored_state)` therefore does not establish an identical recorded mid-contact state. Define anchors through reset plus a replayed warm-up in paired environments and validate determinism, or implement a verified full snapshot. Derive the observation history from this process. Original goal generation also replays actions, so this finding alone does not show that the published benchmark is invalid. See [reset](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/env/pusht/pusht_env.py) and [goal replay](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/plan.py).

2. **Object order versus controller order.** Measure final pusher pose/velocity and block pose separately. Use no-contact controls, then inspect object effects within contact strata. Two internal action blocks may be swapped under a shared prefix and suffix, but a shared suffix does not guarantee identical pusher endpoint or controller memory. An agent-only improvement does not support a contact-manipulation claim.

3. **Frozen metric.** The original code trains the proprio encoder with the action encoder although the visual encoder is frozen. Controlled fine-tuning from one checkpoint freezes both observation encoders across all arms and trains the predictor/action encoder with identical budgets; the first ORDER loss targets visual endpoint differences. This is an added controlled protocol, not the original training recipe. For full cost

\[
C=\|z_v-g_v\|^2/n_v+\alpha\|z_p-g_p\|^2/n_p,
\]

the cost identity applies to the concatenated vector scaled by (1/\sqrt{n_v}) and (\sqrt{\alpha/n_p}). Visual order error alone cannot guarantee full-cost ranking. See the original [optimizer](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/train.py).

## Gated pilot and decision rules

- **Stage A, no method training:** verify data/checkpoint/configuration, reproduce the original panel, and use 200 qualification anchors with two action pairs each. Repeat a subset to test reset and noise. Split by episode and exclude these anchors from final testing. Replay 32 fixed candidates per goal for the initial audit. This screens feasibility and cannot establish a small percentage gain.
- **Stage B, only after a positive audit:** fine-tune from the same checkpoint using ordinary multi-step branch MSE, ORDER, and generic difference/motion-weighted controls over three seeds. Match branch data, horizons, optimizer updates, and tuning budget. The initial cap is 2,000 training anchors with two pairs and five control steps per primitive, yielding 80,000 two-branch suffix transitions before warm-up, repeated resets, validation, and audit candidates.
- **Stage C, robustness:** only after a physical-regret gain, evaluate prespecified 1k/5k/full trajectory levels with randomized nested episode manifests, then state/image unicycle and LeWM where compatible. A full-data checkpoint cannot support a low-data claim. Report fine-tuning and from-scratch curves separately.

Data scaling is a distinct efficiency experiment. Every baseline receives identical information, transition count, and simulator access. Add an equal-acquisition-budget control that collects ordinary trajectories instead of branch pairs, separating acquisition value from loss value. Report simulator, encoder, and training costs rather than update counts alone.

Do not require baseline success to fall in an arbitrary 30–70% range. Near ceiling, retain the full-data result and measure regret, final error, and prespecified data/compute curves. Near floor, if true-dynamics planning also fails, fix the task/search setup before evaluating a predictor loss. Do not manufacture task difficulty after seeing method performance.

Open Stage B only when reset/history is reliable; the object-level order effect exceeds relevant uncertainty; true-outcome latent scoring makes useful selections; learned scoring has reliably higher paired regret; and the gap concentrates on order-sensitive decisions relative to suitable controls. After qualification, freeze the minimum physical effect of interest and sample size before final testing. Confidence intervals resample by episode or anchor and include training-seed variation; branches from one anchor are not independent.

Select task, horizon, and data budgets using baseline/oracle results on the qualification split before observing ORDER gains. Keep the final test fixed, publish all screened settings and exclusion reasons, and treat post-test representation/planner changes as exploratory or use a new test set.

| Observation | Permissible conclusion |
|---|---|
| Search/oracle is poor, or encoded true-future scoring is poor | This task–pipeline pair is unsuitable for testing a predictor-only claim |
| The pipeline passes, but branch MSE or generic reweighting matches ORDER | Specific benefit from ORDER is unsupported; the encoder is not an acceptable fallback explanation |
| Only the toy improves, or PushT gains concern only the pusher | Narrow the claim; current evidence does not support general visual manipulation |
| ORDER reduces physical regret and closed-loop error with matched data/compute and survives controls | Continue the study, while still requiring replication, uncertainty estimates, and broader validation |

Rejecting an incompatible setting prevents a false conclusion. If most realistic settings lack headroom that ORDER can repair, that also weakens the practical value of the idea. No current experiment establishes that ORDER has passed these gates.
