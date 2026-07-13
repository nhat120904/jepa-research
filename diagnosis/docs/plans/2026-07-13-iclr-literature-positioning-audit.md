# ICLR literature and novelty audit: optimizer--cost exploitation in latent planning

**Audit date:** 2026-07-13
**Scope:** the framing in `paper/main.tex` as of this date, not the superseded
CAI-JEPA objective proposal.  This memo uses primary paper/project pages and
does not treat a 2026 arXiv preprint as peer-reviewed unless the source says so.

## Executive verdict

The broad thesis that a predictive latent can be a poor planning metric is no
longer novel.  TRM directly studies the planner-facing terminal-cost mismatch,
uses same-candidate selection audits, shows that task state is linearly
decodable but underweighted by latent MSE, and contrasts a true-state cost with
the latent cost under fixed CEM.  RC-aux likewise frames predictive-but-not-
plannable geometry and finite-horizon reachability.  Nor is the oracle-dynamics
ablation by itself novel: IMWM replaces the learned predictor with literal
environment rollout while holding encoder, terminal latent-MSE, CEM budget,
and replanning fixed.

The defensible ICLR wedge is narrower and potentially strong:

> In contact-rich Franka manipulation, optimization pressure can turn a
> representation-induced goal cost into a systematically misleading selector
> even when candidate dynamics are simulator-exact.  The evidence is not merely
> low aggregate success: it audits the optimizer's selected/elites in simulator
> state, separates candidate-coverage failure from cost-ranking failure, traces
> proxy-versus-physical behavior with search pressure, and tests whether a
> repaired cost survives a fresh search for new error pockets.

That wedge is a **particular causal design and empirical mechanism**, not first
discovery of action unreliability, latent geometry mismatch, perfect-dynamics
failure, or Goodhart's law.  It becomes convincing only if the second
checkpoint, non-CEM planners, powered search-pressure study, and direct closest-
method comparisons land.  Without those, an ICLR reviewer can reasonably read
the current paper as a MetaWorld case study adjacent to TRM and IMWM rather
than a general mechanism paper.

## The decisive taxonomy

The paper should separate four failure surfaces that concurrent work currently
mixes under “world-model planning failure”:

| Failure surface | Observable signature | Closest work | What the present paper must show |
|---|---|---|---|
| Prediction error | A physically good action is rolled out incorrectly | classic MBRL; gradient-planner train/test-gap work | Failure remains with literal simulator dynamics |
| Proposal/coverage failure | No physically successful candidate enters the sampled population | IMWM | Goal-reaching candidates are present, yet the latent cost ranks/selects bad ones |
| Transition-realizability failure | Predicted intermediate transitions are inconsistent with their conditioning actions | WAV, ACID, ATM | Either control this axis with oracle dynamics or compare its repair directly |
| Cost exploitation / selection failure | Optimizer finds candidates with low proxy cost but poor simulator-state outcome; gap grows or persists under stronger search | TRM is closest on misranking; Goodhart work supplies the general lens | Paired proxy/true curves, selected-candidate regret, simulator-state elites, and replication across optimizer/checkpoint |

This taxonomy is the cleanest answer to the apparent disagreement between
IMWM (“search coverage”), TRM (“terminal metric”), ACID (“trajectory
realizability”), and the present paper (“search exploits error pockets”).  The
present paper should not say those diagnoses are globally wrong; it should
identify which one binds in the tested contact regime.

## Closest papers: exact overlap, difference, and threat

### 1. TRM — critical overlap, mandatory direct comparison

[Beyond Euclidean Proximity: Repairing Latent World Models with Horizon-Matched
Trajectory Reachability Metrics (TRM)](https://arxiv.org/abs/2605.22164)
is the closest paper to the cost-side diagnosis.  It keeps encoder, dynamics,
sampler, CEM optimizer, budget, and evaluation manifest fixed, then replaces or
hybridizes only the terminal cost.  It reports:

- a raw latent cost that misranks candidates despite nearly perfect linear XY
  decodability;
- a true task-state cost reaching 100% under the same CEM settings;
- a solver-stress control where much more search does not rescue the wrong
  objective;
- same-candidate selection audits (SCSA) that compare cost ordering and selected
  endpoints; and
- subspace interventions showing that the useful XY rowspace contributes under
  1% of raw latent MSE but carries most control utility.

This is not merely an “aggregate success” paper.  Any statement that prior work
does not audit candidate selection, does not isolate the terminal selector, or
does not show state-present-but-underweighted is false for TRM.

**Difference that remains.** TRM's decisive rescue is on TwoRoom; on PushT its
task-state metric improves ranking/final distance more cleanly than closed-loop
success.  TRM does deploy the repaired metric inside full CEM, so it is incorrect
to say that TRM never tests a repair under search.  The present work instead
studies pretrained robot-planning checkpoints on MetaWorld Franka push/pick,
uses literal simulator rollout as dynamics while varying representation-induced
costs, verifies converged elites in simulator object state, and—most
distinctively—fits a repair on previously mined failure pockets and then starts
an independent fresh search for new pockets.  That explicitly adversarial
mine--repair--re-mine design is stronger than TRM's fixed-log training plus
ordinary closed-loop evaluation only if mining, training, and final evaluation
episodes are disjoint.

**Threat level: critical.** Cite in the Introduction and Related Work, not only
in an appendix.  A same-protocol TRM baseline is the highest-value additional
comparison.

### 2. IMWM — critical overlap on oracle dynamics; different search diagnosis

[IMWM: Intuition Models Complement World Models for Latent
Planning](https://arxiv.org/abs/2606.01626) explicitly replaces the learned
forward predictor with literal environment dynamics while keeping the encoder,
terminal latent-MSE cost, CEM budget, replanning cadence, and evaluation cells
fixed.  It therefore precludes novelty claims of “first perfect-dynamics
control” or “first demonstration that an idealized world model can still fail.”
IMWM also inspects whether any goal-reaching candidate appeared in the CEM
population.  On its key failing cells, almost all failures contain no successful
candidate, motivating demonstration retrieval, a start--goal--action
compatibility cost, and a reliability gate.

**Difference that remains.** IMWM diagnoses proposal-volume/coverage failure:
when a successful candidate appears, its terminal latent-MSE usually ranks it
correctly.  The present paper's intended mechanism is the complementary case:
physically better candidates are available, but optimization selects low-cost
false minima.  That difference must be measured, not narrated.

**Required discriminator.** For every oracle-dynamics episode report (i) whether
the population contains a true-success or top-quantile physical-progress
candidate, (ii) whether the proxy selects it, (iii) proxy-selected regret against
the best physical candidate in the same pool, and (iv) how those quantities
change with budget.  This produces a coverage-error versus selection-error
decomposition directly comparable to IMWM.

**Threat level: critical.** An IMWM-style demonstration/retrieval initialization
or at least the above mechanism comparison is mandatory.

### 3. ACID — high threat and a natural baseline

[ACID: Action Consistency via Inverse Dynamics for Planning with World
Models](https://arxiv.org/abs/2607.02403) argues that terminal closeness alone
does not ensure that intermediate predicted transitions are realizable.  It
adds per-step inverse-dynamics cycle residuals to the planning cost with an
adaptive scale and reports improvements across four action-conditioned world
models and six manipulation/navigation tasks.

**Overlap.** ACID targets the planner's scalar cost at decision time, uses
action consistency to reject candidates the base terminal cost prefers, and is
evaluated on the same broad family of action-conditioned latent planners.  It
also weakens any claim that inverse/action consistency is merely an offline
diagnostic.

**Difference.** With simulator-exact candidate transitions, dynamics
realizability is controlled by construction; a remaining failure is instead
the representation-induced mapping from physically correct terminal frames to
goal cost.  However, if ACID is run on learned dynamics only, it may fix a
different failure surface and cannot be used as evidence for or against cost
exploitation under oracle dynamics.

**Threat level: high.** Implement ACID cost on the same MetaWorld seeds for
learned and oracle dynamics.  Report whether its gain disappears under oracle
dynamics (different mechanism) or survives (important counterexample to the
paper's mitigation story).

### 4. RC-aux — high conceptual overlap, different repair axis

[Predictive but Not Plannable: RC-aux for Latent World
Models](https://arxiv.org/abs/2605.07278) explicitly states that a candidate can
be close in latent space yet unsupported within a finite horizon.  It adds
multi-horizon open-loop prediction and a budget-conditioned reachability head,
then optionally uses reachability inside the planning cost.  Its labels come
from trajectory offsets and temporal hard negatives, not oracle shortest paths.

**Overlap.** Predictive loss can be satisfactory while the latent geometry
queried by search is wrong; planning must expose finite-horizon reachability,
not raw Euclidean proximity.

**Difference.** RC-aux is a training/planner method centered on temporal
reachability and unsupported shortcuts.  The present paper audits released
pretrained robot checkpoints and emphasizes optimizer-selected simulator-state
false minima in contact, including failures of post-hoc/retrained repairs under
fresh search.

**Threat level: high for broad framing, medium for the exact mechanism.** It is
a must-cite.  TRM is the more urgent direct baseline because it preserves the
fixed world model and isolates terminal ranking more closely.

### 5. Latent Geometry Beyond Search — medium/high threat to “planner is the adversary”

[Latent Geometry Beyond Search: Amortizing Planning in World
Models](https://arxiv.org/abs/2605.08732) replaces online search with a
goal-conditioned inverse dynamics model and reports matching/exceeding CEM in
seven of eight settings at 100--130x lower per-decision cost.  Its sweep includes
CEM, MPPI, iCEM, and gradient-based planning, so it is also a warning against
CEM-only conclusions.

**Overlap.** Both works interrogate the representation--planner interface and
whether online optimization is helpful.  The present paper's search-free
controller null cannot establish that representation information is absent; it
can only establish failure of that controller/training protocol.

**Difference.** Their main result is positive amortization under a structured
LeWorldModel latent.  The present work asks whether optimization amplifies
cost errors in harder contact-rich Franka cells and finds that the tested
amortized controller also fails there.  These can coexist by regime.

**Threat level: medium/high.** The paper should report the non-CEM replication
and treat the amortized null as a boundary condition, not a proof that search is
necessary or that the representation is insufficient.

### 6. Action reliability and causal response: old CAI-JEPA novelty is closed

These works are not the closest to optimizer--cost exploitation, but jointly
preclude novelty claims around action identifiability, counterfactual action
evaluation, scaling nulls, or inverse verification:

- [ATM: Action-Consistency Transfer Matrix](https://arxiv.org/abs/2606.09028)
  trains probes on real encoded versus model-predicted transitions, measures
  within/cross-domain action decodability, screens checkpoints, and adds
  Action-Identifiable Transition Supervision.  Its DINO-WM/LeWM comparisons
  also explicitly warn that score scales need calibration across model families.
  CRA/Boundary Blindness is therefore supporting diagnosis only; ATM is the
  mandatory comparison if those metrics remain in the main paper.
- [UWM-JEPA](https://arxiv.org/abs/2605.25313) reports that action sensitivity
  requires counterfactual rather than teacher-forced targets in its controlled
  partially observed task, alongside a structured belief-space predictor.  A
  counterfactual JEPA objective is not a standalone novelty claim.
- [World Action Verifier (WAV)](https://arxiv.org/abs/2604.01985) decomposes
  prediction verification into state plausibility and action reachability, using
  a sparse inverse model and cycle consistency to improve under-explored action
  regimes.  This is close to both action-grounding repair and ACID's verifier
  family, but not an oracle-dynamics terminal-cost exploitation study.
- [MiraBench](https://arxiv.org/abs/2605.29360) evaluates physics adherence,
  action following, and optimism under failure-inducing actions over 12 model
  configurations.  It reports that visual fidelity is a poor proxy for action
  fidelity and that model scale does not reliably improve action following.
  The present model-scale result must be described only as “no favorable trend
  among these released checkpoints under this diagnostic,” especially until
  physical masks and distractors are shared across models.
- [What-If World](https://arxiv.org/abs/2605.27589) uses 319 paired prompts on
  shared scenes from nuScenes and DROID, changing one physical variable and
  scoring the paired outcomes.  It is causal/counterfactual evaluation of video
  world models, not action-sequence optimization; nonetheless it closes any
  “first paired intervention test” claim.

**Threat level:** high for the old action-grounding paper; low/medium for the
new optimizer-cost paper if these results are clearly demoted to observational
context.

### 7. V-JEPA 2.1 — alternative explanation, not just another baseline

[V-JEPA 2.1](https://arxiv.org/abs/2603.14482) introduces dense predictive loss,
deep self-supervision, multimodal tokenizers, and scaling of capacity/data; it
reports a 20-point real-robot grasping improvement over V-JEPA-2-AC.  This is
direct evidence that dense spatial representation quality can change contact
performance.

The current study of older/released planning checkpoints cannot support “scale
does not fix contact” or “contact failure is intrinsic to JEPA.”  A fair claim is
that the tested costs fail despite high average readout precision, and that
dense feature improvements are a live alternative.  If a compatible 2.1
planning checkpoint/API exists, it is the highest-value model-family extension;
if not, name the absence and keep the conclusion checkpoint-scoped.

### 8. Goodhart, reward hacking, and inference-time proxy exploitation

The general conceptual lens is established:

- [Defining and Characterizing Reward Hacking](https://arxiv.org/abs/2209.13085)
  formalizes when improving a proxy can reduce true reward.
- [Scaling Laws for Reward Model Overoptimization](https://proceedings.mlr.press/v202/gao23h.html)
  measures proxy versus gold reward under RL and best-of-n sampling.
- [Goodhart's Law in Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6ad68a54eaa8f9bf6ac698b02ec05048-Abstract-Conference.html)
  quantifies proxy/true divergence as optimization increases and studies early
  stopping and worst-case reward.
- [Inference-Time Reward Hacking in Large Language
  Models](https://arxiv.org/abs/2506.19248) shows the rise-then-fall pattern for
  broad inference-time selection mechanisms including best-of-n, and proposes
  hedging the proxy.
- [LLMs Gaming Verifiers](https://arxiv.org/abs/2604.15149) finds shortcut
  prevalence increasing with task complexity and inference-time compute when
  the verifier omits task constraints.

These papers justify the analogy but mean “Goodhart under search” is not itself
novel.  The robotics contribution is the controlled instantiation: literal
physics for candidate dynamics, a representation-derived proxy, and physical
simulator truth for the optimizer's selected population.

Use **cost exploitation** as the descriptive result unless the data establish
the stronger Goodhart pattern.  To call a budget curve overoptimization, show
with uncertainty that increasing optimization pressure improves proxy value
while physical outcome degrades (not merely remains poor).  Elite/off-policy
distribution shift alone establishes selection into error pockets, but not a
monotonic Goodhart law.  Current n=8/n=16 trends with overlapping intervals are
suggestive, not a scaling law.

### 9. Other mandatory adjacent work

- [Closing the Train-Test Gap in World Models for Gradient-Based
  Planning](https://arxiv.org/abs/2512.09929) studies a different exploitation
  channel: gradient planners query world-model input gradients unlike next-state
  training, and train-time synthesis improves gradient planning.  The present
  work uses sampling planners and closes dynamics error with oracle rollouts;
  cite it to delimit dynamics/gradient exploitation from cost exploitation.
- [Hidden Failure Modes in Latent World-Model Planning from Offline
  Data](https://openreview.net/pdf/d3544b70ed8ee956f9a6f858ccb9ac1c8ead0be0.pdf)
  (ICML Demo Workshop short paper) shows that terminal-at-horizon scoring can be
  misaligned with prefix execution under receding-horizon MPC, and that running
  costs or local waypoint/actuator interfaces can change conclusions.  It is a
  protocol confound the present paper should rule out by reporting planning
  horizon, executed prefix, terminal/running score, and replanning cadence.

## Claim-by-claim novelty clearance

| Candidate claim | Clearance | Safe replacement |
|---|---|---|
| “Prediction accuracy is insufficient for planning.” | Not novel | Background premise; cite RC-aux/TRM/objective mismatch |
| “Latent Euclidean cost can misrank plans.” | Not novel; TRM is direct | Our contact-rich instances exhibit simulator-verified false minima under the released checkpoints |
| “Perfect/oracle dynamics can still fail.” | Not novel; IMWM already does it | Under fixed simulator dynamics, changing only the cost flips success in our Franka cells |
| “Action identifiability is missing / counterfactual targets help.” | Not novel; ATM/UWM-JEPA/WAV | Supporting observational evidence only; held-out result if retained |
| “Scaling cannot fix action grounding.” | Unsupported and contradicted as a universal claim by V-JEPA 2.1's gains | No favorable trend among tested released checkpoints under a model-native diagnostic |
| “The planner is the adversary.” | Too anthropomorphic/broad | The tested optimizer concentrates probability mass on residual error pockets of these costs |
| “First cost-geometry diagnosis.” | Unsafe | No first claim |
| “First optimizer-conditioned audit.” | Unsafe because TRM has SCSA and Goodhart work audits selected outputs | Distinguish fresh re-search survival and simulator-state contact audit |
| “Grounding is necessary.” | Unsupported | Grounding alone is insufficient in the tested interventions |
| “Frozen costs are irreducibly exploitable.” | Unsupported outside tested repairs/checkpoints | None of the tested repairs eliminated exploitation on these cells |

## Immediate corrections required in the current draft

This audit did not edit `paper/main.tex`, but the following current statements
should be corrected before the next PDF:

1. The draft says RC-aux/TRM/ACID “report only aggregate success gains” and do
   not test the repaired cost under the optimizer.  This is false for TRM: it
   deploys TRM inside CEM and adds same-candidate selection audits, subspace
   interventions, a true-state-cost control, and a high-budget solver stress
   test.  The safe distinction is the explicit held-out mine--repair--re-mine
   protocol on contact-rich Franka manipulation.
2. “Oracle dynamics” must be credited to IMWM.  The possible novelty is the
   **joint factorial isolation**—literal simulator dynamics plus multiple costs
   including true simulator-state cost—not the oracle-dynamics substitution by
   itself.
3. “First optimizer-conditioned audit” is unsafe because TRM audits CEM
   selection and Goodhart work audits selected outputs.  Name the exact new
   instrument: simulator-state truth on contact elites plus independent fresh
   search after repair.
4. Claims that the planner is neither search-limited nor representation-limited
   need the IMWM coverage decomposition.  Current aggregate success and elite
   error do not alone prove that good physical candidates were available in the
   same population.
5. “Overoptimization” should be reserved for a powered proxy-improves/physical-
   worsens curve.  Otherwise use “selection into residual cost-error pockets.”

## Must-cite versus must-compare

### Must cite prominently

1. TRM, IMWM, ACID, RC-aux, Latent Geometry Beyond Search.
2. ATM if CRA/action diagnostics remain in the main text.
3. WAV, UWM-JEPA, MiraBench, and What-If World when discussing action
   reliability/counterfactual evaluation.
4. V-JEPA 2.1 as an explicit alternative explanation.
5. Gao et al., Karwowski et al., Skalse et al., and at least one inference-time
   reward-hacking paper for the Goodhart analogy.
6. Gradient-planning train/test gap and the receding-horizon failure-mode short
   paper as nearby but distinct planner-interface failures.

### Must compare experimentally for a credible ICLR submission

Priority is ordered by reviewer risk, not implementation convenience.

1. **TRM on the same MetaWorld checkpoint, seeds, oracle-dynamics adapter, and
   CEM budget.** Use a trajectory-only horizon-matched head; compare raw latent,
   decoded-state/stateprobe, TRM replacement, and hybrid cost.  Report both
   success and simulator-state candidate-ranking metrics.
2. **ACID on learned and oracle dynamics.** This separates transition
   realizability from terminal-cost exploitation.
3. **IMWM-style proposal control.** At minimum run demonstration/retrieval-
   initialized CEM and report the population coverage/selection decomposition.
   A generic larger population is not an adequate substitute for a data-guided
   proposal.
4. **Optimizer generality.** MPPI and random shooting (ideally iCEM) under the
   same candidate-query budget.  The claim should become “selection pressure
   against an imperfect cost,” not “CEM pathology,” only if it replicates.
5. **Second released checkpoint/model family.** Repeat the oracle cost ladder and
   elite simulator-state audit, not only the observational CRA metric.
6. **V-JEPA 2.1 or a dense-feature control** if technically possible; otherwise
   explicitly scope the paper to the older released planning checkpoints.

### Minimum metrics for every comparison

- episode success with paired seeds and confidence intervals;
- best physical candidate in each population and whether one crosses the success
  or meaningful-progress threshold (coverage);
- physical regret of the proxy-selected candidate versus that best candidate
  (selection);
- rank correlation and top-k overlap between proxy cost and simulator-state
  progress on identical candidates;
- proxy and physical trajectories across optimizer iterations/budgets;
- all metrics separately for reach/free-space and contact manipulation.

These measurements make the paper commensurable with IMWM's coverage audit and
TRM's SCSA instead of presenting a new, incomparable “exploitation gap” scalar.

## Recommended ICLR positioning

### One-sentence claim

> We identify and causally isolate a cost-selection failure in contact-rich
> latent world-model planning: under simulator-exact candidate dynamics, several
> representation-derived costs guide test-time optimizers toward terminal frames
> that score as goals in latent space but remain far from the goal in simulator
> object state.

### Three contributions that survive the literature audit

1. **Contact-rich causal isolation.** A cost-only oracle ladder on released
   robot-planning checkpoints: candidate dynamics, CEM budget, seeds, and task
   stay fixed while the goal cost changes.
2. **Optimizer-conditioned simulator-state audit.** Coverage-versus-selection
   decomposition, elite physical truth, and proxy/true curves across search
   pressure, distinguishing the mechanism from IMWM's finite-query coverage
   failure and TRM's fixed-candidate metric repair.
3. **Adversarial survival test.** Repairs are evaluated under a fresh search on
   held-out episodes, testing whether optimization finds new residual pockets
   after the repair has seen previously mined pockets.

The DROID action metrics, model-scale observations, and Phase-H objective should
be secondary unless their causal/shared-protocol and held-out requirements are
fully met.  They are not needed for the strongest paper and currently invite
direct unfavorable comparison with ATM, MiraBench, UWM-JEPA, and V-JEPA 2.1.

## ICLR go/no-go bar after the closest-work audit

**Go with the optimizer--cost paper** if all of the following hold:

- oracle-dynamics selection failure replicates on a second checkpoint;
- the proxy/physical divergence is statistically clear at powered sample size;
- at least one non-CEM optimizer reproduces the error-pocket selection;
- successful/physically better candidates are present often enough to reject an
  IMWM-style pure coverage explanation;
- TRM and ACID do not trivially eliminate the contact failure, or their behavior
  reveals a sharp, publishable boundary condition; and
- mined repair/re-search evaluation is disjoint and demonstrates genuinely new
  search-induced failure pockets.

**Reframe as a careful benchmark/negative-results paper** if the second
checkpoint replicates but optimizer or method comparisons are mixed.  Center the
oracle protocol and candidate-level audit; remove general Goodhart language.

**Do not submit the current broad mechanism claim** if the effect is CEM-only,
disappears with TRM/ACID, is explained by absence of any successful candidates,
or remains an n=16 point trend with overlapping uncertainty.  In those outcomes,
the strongest contribution would be a benchmark of which concurrent repair
works in contact-rich Franka manipulation, not discovery of a new failure mode.

## Primary-source ledger

| Work | Primary source | Version/date checked | Use in paper |
|---|---|---|---|
| TRM | [arXiv 2605.22164](https://arxiv.org/abs/2605.22164) | v1, 2026-05-21 | closest cost-mismatch mechanism and baseline |
| IMWM | [arXiv 2606.01626](https://arxiv.org/abs/2606.01626) | v1, 2026-06-01 | oracle dynamics and coverage failure |
| ACID | [arXiv 2607.02403](https://arxiv.org/abs/2607.02403) | v1, 2026-07-02 | decision-time action-consistency cost |
| RC-aux | [arXiv 2605.07278](https://arxiv.org/abs/2605.07278) | v1, 2026-05-08 | predictive/plannable geometry and reachability |
| Latent Geometry Beyond Search | [arXiv 2605.08732](https://arxiv.org/abs/2605.08732) | v2, 2026-06-05 | amortized controller and optimizer sweep |
| ATM | [arXiv 2606.09028](https://arxiv.org/abs/2606.09028) | v1, 2026-06-08 | action-transition diagnostic/AITS |
| UWM-JEPA | [arXiv 2605.25313](https://arxiv.org/abs/2605.25313) | v1, 2026-05-25 | counterfactual-target action sensitivity |
| WAV | [arXiv 2604.01985](https://arxiv.org/abs/2604.01985) | v2, 2026-05-29 | state-plausibility/action-reachability verification |
| MiraBench | [arXiv 2605.29360](https://arxiv.org/abs/2605.29360) | v1, 2026-05-28 | action reliability, scale, optimism |
| What-If World | [arXiv 2605.27589](https://arxiv.org/abs/2605.27589) | v1, 2026-05-26 | paired physical interventions |
| V-JEPA 2.1 | [arXiv 2603.14482](https://arxiv.org/abs/2603.14482) | v3, 2026-06-11 | dense-feature/contact alternative |
| Gradient planning train/test gap | [arXiv 2512.09929](https://arxiv.org/abs/2512.09929) | v1, 2025-12-10 | dynamics/gradient exploitation distinction |
| Inference-time reward hacking | [arXiv 2506.19248](https://arxiv.org/abs/2506.19248) | 2025-06-24 | proxy search analogy |
| Reward-model overoptimization | [PMLR ICML 2023](https://proceedings.mlr.press/v202/gao23h.html) | proceedings | proxy/gold curves |
| Goodhart in RL | [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6ad68a54eaa8f9bf6ac698b02ec05048-Abstract-Conference.html) | proceedings | optimization-pressure definition |
| Reward hacking formalization | [arXiv 2209.13085](https://arxiv.org/abs/2209.13085) | NeurIPS 2022 preprint page | terminology |
| Hidden planning failure modes | [OpenReview workshop PDF](https://openreview.net/pdf/d3544b70ed8ee956f9a6f858ccb9ac1c8ead0be0.pdf) | 2026 workshop short paper | receding-horizon protocol confound |
