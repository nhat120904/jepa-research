# Research recommendation: event-resolved world models for dynamic manipulation

Date: 8 September 2026. Targets: ICML 2027 or NeurIPS 2027, using the planning windows in the [research brief](../RESEARCH_BRIEF_2027.md). This is a literature-grounded research proposal, not a positive experimental result or a novelty certificate. No simulation, training, or remote job was run for this recommendation.

## Decision

Select **learning contact timing and contact response separately, so that a world model preserves useful action sensitivity through physical interactions** as the primary question for a bounded first study.

Working description: **Event-Resolved World Models (ER-WM)**. This is a provisional label for the proposed programme, not the name of an established method.

The intended contribution is a learning method for action-conditioned predictive dynamics and its use in robot planning. It is not a benchmark, an event detector alone, or a claim that another auxiliary loss will repair the old frozen-DINO pipeline.

The main question is:

> Can event-aligned supervision make learned world models substantially more useful for planning dynamic manipulation, by separately learning how actions change contact timing and post-contact motion, particularly across observation intervals and contact configurations?

I would allocate the next three weeks to establishing this premise. I would not yet allocate the full four-to-eight-month runway. The most serious uncertainty is whether the proposed learning objective contributes anything beyond a properly trained neural hybrid model plus ordinary derivative supervision.

## Why this question, given the repo

The previous programmes establish that lower prediction error, better probing, or smoother local geometry do not necessarily improve the existing planner. They do not establish that prediction is never a bottleneck in a different task with a different cost interface.

The old curvature result is especially relevant: an intervention can remove real and false minima together. The question here is whether a model can **preserve the correct physical sensitivity**, including sharp sensitivity, instead of reducing its magnitude. That is an untested hypothesis; the earlier null is not evidence in its favour.

This proposal deliberately uses an explicit object-state cost shared by all methods and begins with observable state. It later replaces the initial-state input with the same visual estimator for all dynamics models. This isolates dynamics learning before asking whether the advantage survives perception. It does not solve or invalidate the repo's terminal latent-cost failure.

The latest small report read in `scene_progress_wm/outputs/ladder/aggregate/confirm_20260906/DECISION.md` reports mixed results at two goal offsets: +12 points [0,26] and -10 points [-22,2]. This is not a replicated positive foundation for the new programme. `AGENTS.md` also explicitly warns that the single-frame aliasing premise failed in a previous setting. Neither memory nor latent-goal geometry is assumed to be the missing ingredient here.

## Candidate selection

These are three distinct questions considered in this search. Only the first receives an experimental allocation. The other two are not fallback programmes ready to run.

| Candidate | Principle, task, model, and strongest cheap alternative | Assessment |
|---|---|---|
| Event-resolved learning | Hybrid-system sensitivity; dynamic strike/slide/rebound; compact flow–guard–reset model; compare neural event models plus matched secant supervision and an identified physics model | Selected. Specific mechanism, direct physical measurements, controllable task difficulty; novelty remains conditional |
| Passive compositional dynamics | Energy exchange and passivity; pushing/pivoting with multiple objects; objectwise port-Hamiltonian model; compare structured physics/system identification | Not selected. Learning switching port-Hamiltonian systems already preserves physical structure and composition; an energy constraint alone is neither a new method nor protection against wrong contact geometry |
| Refinement of abstract world-model plans | Counterexample-guided abstraction refinement; multi-stage tool use; coarse/fine predictive models with feasibility refinement; compare hierarchical WM and latent collocation | Not selected. Large overlap with hierarchical planning and feasibility projection, plus the unresolved problem of obtaining a trustworthy counterexample without simulator access at deployment |

The passivity prior-art constraint comes from [GP-SPHS, IFAC World Congress 2023](https://www.sciencedirect.com/science/article/pii/S2405896323020293). The planning-refinement competitors include [LatCo, ICML 2021](https://orybkin.github.io/latco/), [Hierarchical Planning with Latent World Models, April 2026 preprint](https://arxiv.org/abs/2604.03208), and [GVP-WM, February 2026 preprint / ICLR World Models Workshop](https://arxiv.org/abs/2602.01960). These are reasons to deprioritize broad proposals, not proofs that all extensions are exhausted.

## The borrowed principle and its limits

A hybrid model has continuous motion within a mode, a guard that determines when an event occurs, and a reset or transition law:

\[
\dot x=f_m(x,u),\qquad g_{m\to n}(x,u)=0,\qquad x^+=R_{m\to n}(x^-,u).
\]

For a time-independent guard/reset and a fixed control during the event, the state sensitivity update is

\[
\Xi=D_xR+\frac{(f^+-D_xR\,f^-)\nabla g^\top}{\nabla g^\top f^-}.
\]

The second term accounts for a perturbed trajectory reaching contact at a different time. Using only the reset Jacobian omits it. This is established hybrid-system mathematics, not our theorem. See [Kong et al., Proceedings of the IEEE 2024](https://arxiv.org/html/2306.06862v2).

For action derivatives, augment the state with held control parameters or propagate the corresponding forced sensitivity equations. Do not apply the displayed state-only formula directly as an action Jacobian. Time-dependent guards, control switches, simultaneous impacts, and grazing contacts require additional treatment.

There is also **no theorem that a black-box discrete-time neural model must omit this term**. A sufficiently accurate model can learn the complete sampled transition map. Our hypothesis concerns learning efficiency and transfer, not universal representational impossibility.

Soft-contact simulators do not implement ideal instantaneous resets. An event model may be a useful approximation when contacts are brief relative to observation intervals; that approximation must be tested. Persistent support contact with the table belongs in the within-mode dynamics. Treating every solver contact report as a new impact would create artificial events.

## What the proposed method actually adds

Use the hybrid model as an existing backbone. The candidate contribution is **event-aligned interventional supervision**, with separate treatment of within-branch responses and changes of contact sequence.

From a reproducible initial state, execute a nominal action sequence and small positive/negative perturbations. All branches share the same initial state and physical parameters. Record ordinary observations and high-rate event metadata during training.

1. Match events across branches by contacting bodies and onset/release identity. A contact sequence is an ordered list of these events, not a language label or task-progress milestone.
2. When the sequence is unchanged, supervise both event time and the state evolution aligned relative to that event. This separates an early collision from a different collision response.
3. Supervise directional changes of event time and of event-aligned motion using finite secants. The method does not require differentiating the simulator.
4. When a perturbation creates, deletes, or reorders an event, train branch/guard prediction and ordinary outcome prediction. Do not pretend that a cross-branch finite difference is a derivative of one smooth branch.
5. Retain prediction at the original clock times. Event alignment must not let a model receive full credit for a correct motion occurring at the wrong time.

A starting objective, with each component normalized by training-set scales, is

\[
L=L_{clock}+\lambda_eL_{event}+\lambda_\tau L_{time}
 +\lambda_aL_{aligned}+\lambda_sL_{secant}.
\]

For a matched event k, action direction v, and perturbation epsilon, one target is

\[
d\tau_k=\frac{\tau_k(a+\epsilon v)-\tau_k(a-\epsilon v)}{2\epsilon}.
\]

An aligned response target uses `x(tau_k + r)` at fixed offsets r, separately from the ordinary fixed-time state targets. Match predicted secants to these observed secants. At finite epsilon they are secants, not exact infinitesimal derivatives.

Initial offsets: -20, 0, +20, +50 ms relative to onset, restricted to windows where matching is unambiguous. Choose endpoint conventions consistently at an ideal jump. For finite-duration contacts, log onset and release separately and evaluate alignment sensitivity to the window definition. Those offsets are proposed engineering defaults, not physically universal constants.

Backpropagation through a learned event time uses implicit event differentiation. [Neural Event ODEs, ICLR 2021](https://iclr.cc/virtual/2021/poster/2743) already provides that machinery; [torchdiffeq](https://github.com/rtqichen/torchdiffeq) supplies an implementation route. Reusing it is preferable to presenting an established derivative rule as a contribution.

### The exact novelty boundary

| Closest work | Established overlap | What must remain different and useful |
|---|---|---|
| [Neural Hybrid Automata, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/5291822d0636dc429e80e953c58b6a76-Paper.pdf) | Learned modes, continuous dynamics, event timing, and transitions | Event-aligned interventional targets must improve learning/control beyond this backbone; architecture alone is insufficient |
| [MB-MIX, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34419) | Supervision of dynamics gradients | Must beat generic endpoint secant/Sobolev supervision on identical data, not just ordinary MSE |
| [TAWM, ICML 2025](https://icml.cc/virtual/2025/poster/44469) | Variable time-step conditioning and training | Event structure must add value beyond time conditioning and matched temporal augmentation |
| [EAWM, ICLR 2026](https://openreview.net/pdf?id=OWkkFaq1IZ) | Event representation and event prediction in world-model learning | Physical action-to-event timing/response, rather than event labels alone, must explain the gain |
| [PIN-WM, RSS 2025](https://pinwm.github.io/) | Learning a visual, physics-informed model for non-prehensile manipulation | Compare against identifying physical parameters, not only an unconstrained neural predictor |
| [ContactGaussian-WM, February 2026 preprint](https://arxiv.org/abs/2602.11021) | Differentiable contact geometry/dynamics, video learning, MPC | Cannot claim to be the first contact-aware or physically grounded visual world model |
| [Contact-Robust Trajectory Planning, ICRA 2026 according to author publication page](https://jameswzhu.github.io/2026-06-02-contact-robust/) | Saltation and parametric sensitivity used for robust control | Learning event response from interaction data is the proposed focus; saltation-based planning itself is established |
| [Saltation-consistent event-aware digital twins, Scientific Reports 2026](https://www.nature.com/articles/s41598-026-53809-5) | Explicit event geometry, sensitivity/uncertainty transport, event-aware inference and reduction | Broad “event geometry unifies learning and control” claims are already occupied. The narrow training contribution and robot evidence must stand independently |

The crucial combined baseline is **the same neural event architecture + ordinary trajectory/event supervision + generic endpoint secant matching**. It receives the same branched transitions and metadata. If that combination matches ER-WM, there is no demonstrated need for the proposed decomposition. Likewise, adding event-aligned losses to a plain predictor tests whether a special architecture is necessary.

An explicit reduction is worth stating: define a transformed target `y(a) = [tau_k(a), x_a(tau_k(a)+r_1), ...]` within a matched branch. Secant matching on y is ordinary Sobolev-style supervision in event coordinates. Thus the proposed loss is not a new general learning principle. The possible contribution lies in the choice and construction of those targets, branch correspondence, their integration into an action-conditioned model, and a demonstrated advantage over the generic combination. This is a substantial novelty risk, not something to hide behind a method name.

This is not yet an equation-level novelty proof. The available evidence supports a research question worth testing; it does not support calling the method a confirmed literature gap. A claim of “first saltation-aware world model” would be particularly unsafe.

## Task–model–data specification

All numbers below are initial design choices. They are not measured runtime, fitted power calculations, or conference acceptance thresholds.

### Arena

Use ManiSkill's robot/table/controller infrastructure. [RollBall-v1 source](https://raw.githubusercontent.com/mani-skill/ManiSkill/main/mani_skill/envs/tasks/tabletop/roll_ball.py) supplies a Panda, a table, object/goal randomization, and a native distance-based success predicate. Its implementation is available; it has not been installed or reproduced in this study.

| Task | Purpose | Construction and success |
|---|---|---|
| Native RollBall-v1 | External anchor and reproduction check | Keep native task and success predicate. Report its result separately from all variants |
| Strike–slide | First method arena | Replace the sphere with a visibly oriented rectangular puck. Target lies beyond the reachable striking region. Robot must transfer momentum and release. Success: puck center within 5 cm of target and speed below 0.1 m/s for 0.2 s, within a 4 s episode |
| Strike–rebound | Event-composition test | Add one fixed sidewall and target positions requiring redirection. Same success rule. Vary wall angle/location and initial object orientation; evaluate held-out configurations |
| Slow push control | Negative control | Same object/observations, target in reachable workspace, low action-speed range and sustained contact. Tests whether gains are merely generic extra supervision |

The rectangular puck makes orientation and rotation observable from geometry. A visually uniform sphere would introduce an avoidable angular-velocity observability issue. Native RollBall remains an anchor rather than the only evidence.

A new variant is justified because the research needs controlled contact timing relative to the observation grid, release followed by unactuated motion, and repeatable event composition. Native success rates do not isolate those properties. This does not assert that no existing environment can express them.

Do not force every trial to have a rebound or use hand-authored contact words as the goal. Task success depends on object placement. Include direct, missed, grazing, and repeated-contact outcomes in data. If a calibrated elementary strike controller already solves the variants robustly, do not keep adding artificial complexity to manufacture a gap.

### Interfaces and model

| Component | Initial choice |
|---|---|
| Robot control | Panda end-effector delta-position PD controller, fixed hand orientation and gripper closure; 3D translation at 20 Hz. Verify exact controller identifier/action scaling from the installed version before collection |
| Physics logging | Target 500 Hz contact/state logging; repeat the mechanism check at 250 and 1,000 Hz where supported. The integrator/substep configuration must actually resolve this rate |
| State pilot | Robot joint position/velocity, puck pose/twist, static wall geometry. No future state or true contact time is provided during planning |
| Visual stage | Two calibrated 128×128 RGB-D views, four observations over 150 ms, plus proprioception. A shared estimator predicts initial object pose/twist and wall geometry; trained using simulator state labels |
| Visual encoder | Four-channel ResNet-18 per view with shared weights, temporal GRU width 256, physical-state heads. Train from scratch initially; no DINO checkpoint dependency |
| Predictive backbone | Mode-conditioned flow network, scalar pairwise guard networks, and contact impulse/reset network; three hidden layers of width 256 each; approximately 1–3M dynamics parameters as an estimate, exact count to be measured |
| Modes | Free/table-supported sliding; hand–object contact; wall–object contact; simultaneous contact flagged separately. Guard/reset networks shared by body-pair type |
| Prediction | Action-conditioned physical state trajectory plus event identities and times. Table friction and motor tracking belong in continuous flow; pose does not jump at ideal impact, velocity may |
| Numerical implementation | Start with standard neural event integration; root tolerance 1e-4 s, bounded integration steps, explicit failure logging. Simultaneous/grazing cases cannot silently disappear from evaluation |
| Context | Fixed material family in the initial pilot. No claim of identifying arbitrary hidden friction from one image. Later use a common interaction-history encoder if materials vary, matched for all arms |
| Training | AdamW, initial learning rate 3e-4, batch 128 windows, 50k updates in the state pilot; validation-selected checkpoint. Sweep loss weight in {0.1, 1, 10} using a shared fixed tuning budget |

This is a supervised structured world model. It is **not** self-supervised JEPA, reward-free learning, or a model that has learned physics from RGB without state/contact labels. Such a scope is legitimate for the brief but must be explicit.

The estimator is a dependency, not an assumed solved problem. Measure its errors against the smallest action/contact perturbations used in the study. If those errors overwhelm the claimed dynamics advantage, the visual version is not established. Ground-truth-state results can support a narrower dynamics-learning claim but cannot be relabeled visual control.

### Data and splits

State pilot: 512 training roots, 128 validation roots, 256 held-out diagnostic roots. Each root supplies one nominal and four perturbed 1 s action sequences (two directions, positive/negative), giving 128k training control transitions at 20 Hz, plus high-rate state/event logs. Roots and all descendants stay in one split.

Choose nominal actions from a fixed mixture: 40% geometrically aimed strikes, 40% noisy aimed trajectories, 20% uniform bounded smooth actions. This uses simulator state during training collection; all methods receive the resulting data. Do not discard failed or no-contact branches. Perturbation scales initially 0.02 and 0.08 of the normalized action range, with feasible perturbations rather than hidden clipping artifacts.

The pairing protocol uses exact simulator restore only during training/diagnosis. It is a simulator-assisted learning assumption, not something a real robot can execute exactly. Count every branch in the data/interaction budget.

Visual stage, only after the state gate: initial cap of 10,000 ordinary 4 s episodes plus 2,000 branched roots across the two new task families. Save images at observation rate, not every physics substep. Record physics metadata separately. A dedicated collector should oversample rare transitions by a declared quota and give the same sampling to all models.

Split along independent axes: held-out initial states/goals; held-out approach speeds/angles; shifted camera sampling phase; held-out wall configurations. Separate interpolation from extrapolation. Begin with fixed material parameters to avoid confounding event learning with hidden system identification. Later material tests must give every method the same available history.

Observation-rate tests keep the underlying physical action signal fixed and resample observations; control-rate tests change the action interface and must be reported separately. Use dt conditioning and temporal augmentation for the strong baselines as well as the method.

### Planner and costs

The first comparison uses one shared planner for every predictive model: CEM with 256 candidates, five iterations, 32 elites, 2 s horizon parameterized by eight piecewise-linear 3D action knots. Execute 100 ms, observe, and replan. No privileged rollout is available at deployment.

Initial objective: squared terminal puck-target distance normalized by 5 cm, terminal speed penalty normalized by 0.1 m/s, bounded action-energy penalty, and a table-exit penalty. Weights: 1, 0.1, 0.01, and 100 respectively. Fix the objective using oracle-dynamics development trials before learned-model comparisons; then freeze it.

Test a second shared planner using gradient refinement from 16 identical initial action sequences, with 20 bounded Adam steps and validation-selected step size. If event changes cause local optimization problems, retain a matched multi-start strategy; do not give ER-WM extra simulator checks. Report planner-by-model interactions. A gain only under an intentionally poor optimizer is not sufficient.

Report actual success, distance, time to success, and observation-to-action latency. Count root finding and perception. Faster forward passes or better derivative alignment alone are not the intended paper result. The goal is primarily control/data efficiency; lower planning latency is optional.

## Baselines that can kill the idea

Run the cheap decisive set first, then the broader paper set.

| Stage | Baseline | What it rules out |
|---|---|---|
| First | Oracle dynamics + the exact shared objective/planner | Task/action/cost incompatibility. A demonstrated successful trajectory alone does not establish planner adequacy |
| First | Identified rigid-body model + shared planner; geometric aimed-strike controller | A small amount of system identification or a simple reactive/analytic policy is enough |
| First | Time-conditioned residual dynamics, matched capacity, multi-step training and all branched data | More transitions, more capacity, or dt conditioning explains the result |
| First | Same predictor + generic endpoint secant matching | Ordinary derivative supervision explains the result |
| First | Dense/high-rate target supervision with matched event-window sampling and loss budget | More high-rate states or event oversampling, rather than alignment, explains the result |
| First | Same hybrid architecture + event/trajectory losses, with and without generic secants | Existing hybrid modeling plus extra event labels explains the result |
| First | Full candidate minus timing targets; minus alignment; shuffled event association | Determines which information, if any, contributes |
| Paper | TAWM/TD-MPC2 and a stochastic recurrent WM under matched information/data | The result survives strong model-based learners; state-only MLPs are not the entire comparison |
| Paper | Event auxiliary prediction inspired by EAWM | Event awareness without the proposed timing/response decomposition suffices |
| Paper | Contact-structured system identification, informed by PIN-WM/ContactGaussian-WM | A physically structured alternative is stronger than the proposed learned event approach |
| Paper | Goal-conditioned BC and SAC/PPO learning curves on the same tasks | Establishes when a predictive model is useful relative to direct policies; disclose extra online interaction separately |

Adaptations are adaptations, not reproductions of the original paper. Keep the exact paired architecture/loss ablations alongside them. Contact labels, true state, branch data, tuning trials, and history must be matched where they explain the claim. Full RGB-only DINO-WM can be a contextual comparator; it cannot be the main baseline for a method trained with physical labels.

## First experiments and decisions

### Days 1–4: establish a valid mechanism measurement

Use an analytic single-impact system only to validate event matching, secants, and the sensitivity implementation. The expected hybrid-calculus result is a software/unit check, not a novel empirical finding.

Then build only strike–slide and reproduce native RollBall plumbing. Establish solvability with the actual action controller, objective, and planner. Check restore determinism, physics-step convergence, and whether apparent event timing changes are contact-logger artifacts.

### Days 5–15: state-based decisive comparison

Train three seeds on the locked small dataset. Evaluate 100 independent task episodes per seed using identical episode draws across methods; use diagnostic roots separately. Plot true cost after one proposed action improvement and full closed-loop success, stratified by contact/no-contact and same/different contact sequence.

Support for further investment requires an actual control advantage over the strongest matched cheap baseline, not only over MSE. A proposed worthwhile effect is 10 percentage points of success or a 20% reduction in physical terminal error with non-inferior success. These are investment criteria, not power claims or universal standards. An imprecise pilot remains inconclusive.

If the screen is promising, confirm with five independently trained seeds and 200 paired episodes per seed. Report training-seed variability separately from episode variability; do not pretend 1,000 episodes make five trained models into 1,000 independent model replications. Predeclare the comparison and uncertainty procedure before this confirmation.

Stop this specific proposal if:

- Ordinary dt-conditioned dynamics or generic secant supervision matches it within the predeclared practically useful range.
- A calibrated simple physics model/control policy already removes meaningful headroom.
- Better event/derivative metrics do not improve physical decisions or control.
- Gains vanish with accurate integration, fair data, or the same event labels supplied to baselines.
- Gains appear only in the artificial variant and fail on a second contact configuration/task.

If the oracle planner fails, fix task/cost/planner compatibility before evaluating the method. If noisy or grazing event correspondence dominates, record a modeling limitation; do not keep only convenient events and claim general contact-rich manipulation.

## What a full method paper would need

The central claim should eventually be something like:

> Separating action-induced changes in event time from event-relative motion improves the sample efficiency and transfer of predictive dynamics used for dynamic manipulation.

That claim would require: a precise training objective; an analysis of what is identified by paired trajectories under isolated transversal events; decisive comparisons to NHA-style models and generic secant learning; control gains on at least two meaningful robot interaction families; an external native task anchor; and robustness to perception noise, sampling phase, and solver/contact softness.

Do not claim a new saltation theorem. A useful theoretical contribution, if obtained, would concern identifiability/error decomposition for these supervision targets under explicit sampling and event-correspondence assumptions. Whether such a result is possible is open. No theory is required to make the first empirical falsifier run.

A state-only result on a puck and a ball would be too narrow to justify the intended broad robot-world-model claim. A visual result solely on a custom arena would also be weak. Expand to a distinct interaction such as box redirection/pivoting only after proving that its dominant limitation is expressed by the same mechanism, rather than adding unrelated task difficulty.

## Resources, schedule, and unresolved dependencies

Assume access to one modern 24 GB GPU for the small models, running sequentially; this is a feasibility planning assumption, not knowledge of the user's allocation. Propose a **120 GPU-hour ceiling for the first three-week study**, including simulation and all screening arms, with runtime measured on the first small run. If this is insufficient, revise the scope visibly rather than silently inflating the experiment. Rendering and physics throughput remain unmeasured.

An initial full-project allowance of 600–1,200 GPU-hours is an order-of-magnitude planning estimate, excluding a large pretrained video model and any real robot. It is not a benchmarked estimate. Profile before committing, and scale the programme to actual hardware.

| Milestone | Required evidence |
|---|---|
| Late September | Reproducible state-based task, strong baselines, decision on the event-alignment hypothesis |
| October | Advantage survives visual estimation and a second interaction condition; data-efficiency curves |
| November | Main comparisons, independent seeds, native task, sampling/solver controls |
| December–January | Write only if a stable method claim exists; official ICML deadline must still be checked |
| February–May if needed | Broader task/shape generalization for NeurIPS, not repeated rescue of a failed first premise |

The largest engineering uncertainties are high-rate contact logging/restoration on the chosen simulator backend, efficient batched event integration, physically meaningful event matching, and visual twist estimation. Public source availability does not establish that any of these has run successfully here.

There is no pretrained checkpoint dependency and no permission requirement to complete this literature/specification work. This document does not authorize or claim that training jobs have been submitted. The remote checkout's Slurm policy still applies to later execution.

## Evidence scope

Sources above were checked on 8 September 2026 using primary papers, proceedings, official documentation, or author project/publication pages. Venue labels are kept distinct from preprints and workshops. This was a targeted search, not an exhaustive census of 2025–2026 acceptances.

One related ICLR contact-modeling result (`NsDwAHwNLB`) was available only as indexed excerpts: its page/API returned an access challenge. Those excerpts describe a soft-contact PINN/UDE comparison. It has not been used as a fully reviewed novelty exclusion. Full text should be obtained during the focused prior-art audit, alongside the combined NHA–Sobolev baseline.

Recommendation status: **SELECT_FOR_BOUNDED_PREMISE_STUDY; NOVELTY_AND_CONTROL_GAIN_UNCONFIRMED**. The research question is selected. A successful method has not been found by literature search alone.
