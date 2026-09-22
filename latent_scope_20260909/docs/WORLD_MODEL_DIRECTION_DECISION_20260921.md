# Decision: a world-model-first, bounded compositional prediction pilot

Date: 2026-09-21. Status: research/design decision, **not an experimental result**.
No training, simulator evaluation, checkpoint download, or Slurm submission was performed
for this decision. Existing dirty files and other sessions' jobs were not modified.

## 1. Recommendation and explicit scope change

Prioritize **self-supervised, action-conditioned multi-timescale prediction under a fixed
MPC controller**, initially on PushT using the released JEPA-WMs implementation.
Do not build a VLA, learned proposal policy, actor-critic, or new value objective in the
first experiment. Do not make LeWM the only base model.

This narrows the research question. It is **not** a test of hidden cumulative contact
effects or memory-dependent manipulation. It retains prediction of unexecuted action
sequences and temporal composition, but drops hidden-memory superiority as the initial
claim. PushT cannot establish that claim. If hidden cumulative effects are non-negotiable,
this recommendation is not a substitute for that research program.

Question: can a predictor jump over an action segment, retain the boundary context needed
for further prediction, and compose such jumps at unseen partitions, improving the
closed-loop success/compute tradeoff over a strong autoregressive latent WM?

There is no guarantee of a positive result or a top-tier paper. The aim is to obtain an
interpretable method test without simultaneously inventing the policy, objective, and
environment infrastructure. Simple jump/split consistency alone is not sufficient novelty.

## 2. Evidence informing this choice

| Evidence | What it supports; what it does not |
|---|---|
| [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | Reward-free latent prediction and goal-image planning are an established method-paper setting. A VLA is not required. |
| [JEPA-WMs study, Table 2](https://arxiv.org/html/2512.24497v3) | PushT success is 70.2 for their model and 66.0 for DINO-WM in their protocol. This is evidence of feasibility, not a promised local result or proof that the remaining failures are dynamics failures. |
| [Official JEPA-WMs release](https://github.com/facebookresearch/jepa-wms) | PushT code, data links, and checkpoints exist. This avoids beginning with an unvalidated policy-training project. |
| [CompPlan](https://arxiv.org/html/2602.19634v1) | Jumpy prediction and cross-timescale consistency overlap with the proposal. Their policy-induced occupancy setting differs from prediction conditioned on an explicit primitive-action sequence, but that difference alone is not enough for a paper. |
| [OGBench, Table 2](https://arxiv.org/html/2410.20092v2) | HIQL obtains 89% on visual-cube-single-play. This supports a possible second manipulation benchmark, not an achievable ceiling for this WM/controller. |

Local evidence:

- `diagnosis/docs/CURRENT_STATUS.md`: in the documented MetaWorld protocol, perfect
  dynamics plus latent-L2 still failed, while the physical reference cost succeeded.
  This blocks the inference that improving prediction necessarily improves control.
- `scene_progress_wm/outputs/ladder/aggregate/confirm_20260906/DECISION.md`: SSL progress
  versus latent-L2 did not show a consistent advantage across the two offsets. Replay
  feasibility is not evidence that search can find those trajectories.
- `docs/TRAJECTORY_SSL_STEP3_RESULT_53275.md`: signature composition amplified prediction
  error; the useful baseline remains ordered latent prediction. Do not discard spatial
  information through another aggressive summary bottleneck at the outset.
- `docs/MIKASA_GATE0_RESULT_53286_53288.md`: the tested ManiSkill/SAPIEN configurations
  are blocked by graphics/device issues. A CPU-render-device alternative remains untested;
  therefore the evidence is not a proof that every possible MIKASA setup is impossible.
  It is sufficient reason not to make it the immediate critical path.

PushT uses a 2D action space and a different rendering stack from SAPIEN. Local execution
and checkpoint compatibility still require a compute-node smoke test. Do not silently
replace the release environment with another PushT implementation.

LeWM's venue status is not a capability measurement. Its published experiments use
different protocols from the JEPA-WMs study; do not compare headline success rates across
those protocols. Keep it as an optional efficiency baseline, not a prerequisite.

## 3. Controller and data contract

Begin with the official config already present at:

`diagnosis/external/jepa-wms/configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml`

Relevant release settings include DINOv2-S patch features, a six-layer AdaLN predictor,
frameskip 5, terminal latent-L2 with proprio weight 0.1, CEM horizon 6, 300 samples,
30 iterations, and 10 elites. These are a reproduction starting point, not newly tuned
settings. The config uses 96 evaluation episodes. Paper aggregates and one released
checkpoint need not have identical success rates.

Before collecting results, record checkpoint hash, environment revision, observations,
normalization, goal sampling, seeds, termination, action bounds, and action grouping.
Trace model steps to native simulator actions and the commit/replan cadence explicitly:
do not infer native horizon merely from `H=6` or `frameskip=5`. The release distinguishes
training context (3) and deployment context (2); preserve and disclose this distinction
in reproduction and use identical context rules for all matched arms.

Disable optional image/state decoding and plotting for timing in every arm. Privileged
object state can be used for evaluation audits, not predictor inputs or training targets.
Proprioception is allowed equally for every arm. Do not use a privileged scripted policy
as though it were an observation-only deployment baseline.

The deployed objective initially stays the released latent goal distance. It is **not
assumed to be correct**; it must pass the objective qualification below. If it does not,
do not simultaneously add a critic and claim the new WM was tested cleanly.

## 4. Architecture to prototype only after qualification

Let z be the frozen image patch grid and p the permitted proprioception. Let X_t contain
the last w latent/proprio observations and the action context required by the base WM.
Actions remain ordered, with their duration/cadence explicit.

`J_theta(X_t, a[t:t+H], H) -> predicted boundary context X_(t+H)`

The predictor also produces a small, fixed set of ordered intermediate latent targets
through timestamp queries. These preserve trajectory supervision without requiring a
sequential frame prediction call at every step. Retain spatial patches initially; do
not replace them with four summary tokens before establishing their sufficiency.

Crucially, the boundary output must include the w-frame latent/proprio context needed to
continue forecasting. Predicting only the last frame and pretending it supplies the
multi-frame baseline's input is an invalid composition interface.

Two paths are trained against observed futures:

1. Direct: `J(X, A ++ B, H_A + H_B)`.
2. Composed: `J(J(X, A, H_A), B, H_B)` with no observed-future teacher forcing at the
   boundary when testing free composition.

Both paths receive observed-future latent/proprio supervision. The method arm additionally
aligns their boundary/anchor outputs with a specified stop-gradient consistency target.
Consistency never replaces grounding against observations: constant predictions must not
be a valid optimum. Freeze the pretrained SSL image encoder in this first pilot so encoder
drift/collapse is not another intervention.

No rewards, contact labels, success labels, simulator state, or privileged teacher enter
these training losses. Evaluation uses simulator success, as usual. The accurate claim
is self-supervised WM training on logged observations/actions, using a pretrained SSL
visual backbone; not learning visual representation from scratch on this dataset.

Initial training horizons: {1, 2, 4} model steps. Evaluate H=6 compositions such as 2+4
and 3+3, and H=8 as 4+4; the unseen duration 3 and unseen total lengths must be reported
separately. Longer H=12 is exploratory until a new protocol is locked. Match the real
action duration and control commitment, not only the number of model calls.

Do not impose diagonal affine operators, Chen products, Gated DeltaNet, or Mamba as the
main innovation. Associativity of a constructed scan is automatic. It neither proves
prediction accuracy nor improves control by itself. Backbone swaps can be later ablations.

## 5. Essential matched comparisons

| Arm | Role |
|---|---|
| Released JEPA-WMs checkpoint | Local capability/reproduction anchor, not the sole causal baseline |
| Strong autoregressive WM, multistep trained | Frame-prediction comparison with matched data and supervision |
| Direct/jumpy WM without consistency | Controls for fewer sequential calls, horizon supervision, and architecture |
| Same jumpy WM plus direct/composed consistency | Actual method intervention |

The two jumpy arms share architecture, initialization protocol, targets, sampling, training
budget, and inference schedule. Ground composed predictions against observed targets in
both arms so the extra consistency term is the clean difference. Also compare a cheaper
autoregressive configuration/frame stride on the compute frontier; do not beat only an
unoptimized serial implementation.

All arms use the same image encoder, goals, action limits, cost, CEM hyperparameter search
budget, and history caching. Report parameter counts, training FLOPs/time, and inference
memory. Evaluate both equal candidate budgets and equal end-to-end decision latency.
Different candidate counts under a fixed latency budget are allowed, but must be disclosed.

## 6. Bounded qualification: four failure modes, not another audit paper

This is an engineering qualification for the method, not a reopening of the audit-paper
direction. Write a small locked development set and an untouched final split first.

| Concern | Qualification | Interpretation |
|---|---|---|
| Actions cannot accomplish useful motion | Restore the same initial states; execute candidate sequences in the simulator. Include candidates from initial and late CEM iterations. Compare with feasible reference trajectories separately. | Feasible demonstrations alone do not prove search support. Report the physical best candidate actually in the bank. |
| Search cannot find good actions | Run the same constrained search with simulator dynamics and a physical reference cost on a bounded subset. | This tests the search/action parameterization, not learned WM quality; it is not a proof of global optimality. |
| Deployed objective picks the wrong outcome | Score the same candidate bank using encoded true simulator futures and the unchanged deployed objective; compare with physical outcome ranking. | If even true futures induce poor choices, a more accurate WM is not justified as the remedy. |
| Learned predictions break useful ranking | Replace true futures with predicted futures on the identical bank; execute selected candidates. Then confirm the relevant gap in closed loop with unchanged controller. | This identifies prediction-associated decision loss; it does not guarantee the proposed consistency loss can remove it. |

Use independent initial states/episodes as the unit of uncertainty. Windows, candidates,
and multiple search iterations from one episode are correlated, not independent samples.
For local bank tests use both success where reachable and a predeclared physical progress
measure; a short prefix's best progress is not a final-success oracle. Avoid a noisy long
uncontrolled policy continuation like the Scrub design.

Closed-loop trajectories change when scores change. Candidate-bank comparisons isolate
scoring on fixed inputs; closed-loop comparisons establish control usefulness. Neither
substitutes for the other. A simulator-dynamics controller is a reference intervention,
not a mathematically guaranteed upper bound, since objective mismatch can make accurate
dynamics worse than a biased predictor.

Proposed sequence: a small profile; official-checkpoint reproduction; a 32-independent-state
bank screen (64 candidates each, sampled across initial/final CEM rounds); expand only a
promising/inconclusive screen to the locked evaluation set. These screen sizes are not
confirmatory sample sizes. Do not label a noisy non-significant screen a proof of no gap.

Qualification stops if the runtime/protocol cannot be made consistent in one bounded
repair cycle, if deployed-cost selection is the evident bottleneck, or if no prediction
or compute bottleneck worth testing is observable. Do not launch a new objective, policy,
and WM repair simultaneously. If the baseline saturates success, only an explicitly
declared efficiency endpoint remains; do not make the benchmark harder after seeing results.

## 7. What counts as progress

The primary claim is a success/compute improvement, not algebraic associativity or lower
latent MSE alone. Predeclare one primary operating point after baseline-only profiling,
before training the method. Suggested practical target: at least 2x lower **full decision
latency**, with a 5 percentage-point success noninferiority margin, compared with the
strongest matched baseline. Measure image encoding, history handling, search, and scoring,
not just predictor kernels. Report common simulator stepping separately.

Check the speed headroom before training. If the accelerated predictor is fraction f of
decision time and its measured speedup is s, the best unchanged-system speedup is
`1 / (1 - f + f/s)`. For example, a 3x predictor speedup with f=0.6 gives only 1.67x overall;
it requires f>=0.75 to reach 2x overall. If profiling rules out the declared compute gain,
do not train on the assumption that fewer model calls must deliver it. This bound assumes
the other costs stay fixed; report any implementation changes in every matched baseline.

Alternatively, if selecting equal latency as the primary endpoint before method runs,
require a prespecified meaningful success increase with an interval excluding zero.
Do not switch primary endpoints after seeing which one wins. Always report the full
accuracy/latency frontier and the jumpy-without-consistency ablation.

One seed is a development pilot. Confirmation requires at least three training seeds,
paired evaluation episodes, and uncertainty accounting for both sources of variation.
Choose evaluation counts from baseline-only paired variance and the declared effect/margin;
96 episodes may be far too few to establish 5-point noninferiority. Insufficient precision
means inconclusive, not pass or proof that the idea is false.

If jumpy prediction wins but consistency does not, the result supports temporal prediction
compression, not compositional learning. If only latent error improves, there is no
demonstrated planning contribution. If all matched arms tie, close this specific design.

## 8. Scope, budget, and follow-on decisions

Recommended initial cap: one qualification tranche of at most 8 GPU-hours and 32 CPU-hours,
then at most 24 GPU-hours for a one-seed matched method pilot, **only if profiling indicates
the compared models can actually be trained adequately within that cap**. These are proposed
spend limits, not runtime estimates or current allocation. Queue delay is not compute.
If the cap cannot support a fair comparison, report that and choose a smaller documented
protocol before running; do not compare a converged baseline with an undertrained method.

No repeated arena search: PushT is the first pilot. A positive, replicated finding earns
an independent qualification on OGBench visual-cube-single-play-v0 as the preselected second
arena, using its native goals and action interface. Its published RL score does not qualify
its latent-L2 objective. Do not start there with Scene or multi-cube tasks.

Actor-critic is deferred. It introduces value learning, out-of-distribution imagined
states/actions, and actor exploitation of model errors. A shared established actor-critic
recipe can later test transfer of the WM improvement, but training a critic now would
prevent clean attribution. Reward-trained control is also not an entirely reward-free
pipeline, even when the WM itself is self-supervised.

Paper-level evidence would require the matched causal comparison, a meaningful control/
compute gain, generalization across held-out horizons/partitions, and independent task
coverage. Applying a known temporal-consistency idea without those findings is a prototype,
not yet a strong method-paper contribution.

All execution must follow repository compute policy: `sbatch` on compute nodes, explicit
time limits, no interactive GPU holdings; check both `squeue` and `sacct` and other-session
work before submitting. This document does not authorize or claim any completed run.
