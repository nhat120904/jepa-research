# A bounded test of selective stability in latent world models

Status: proposed experiment, not implemented or validated. Date: 2026-09-08.
The Vietnamese recommendation is in `REPORT_VI.md`. All sample counts, cutoffs and compute caps below are design choices, not measured facts or a completed power analysis.

## Question and claim boundary

Can a pretrained visual world model be made robust to departures from valid observation histories without suppressing sensitivity to physically meaningful state/action changes, and does this improve the decisions of an unchanged planner?

The starting evidence is the official LeWM Reacher candidate-selection audit: true-endpoint latent selection succeeds on 1.000 of the audited anchors, predicted-endpoint selection on 0.415. This is a witness-conditioned local candidate panel, not a closed-loop planning score. It motivates investigation of dynamics but does not identify normal instability as the cause.

The proposed contribution is not a new stability theorem, manifold projection, global Jacobian penalty, or discovery that prediction accuracy can disagree with control. Those have direct precedents. A potentially defensible contribution would be an estimator and training intervention that distinguish representation-error amplification from required physical sensitivity and demonstrate an advantage over strong matched controls. Novelty remains conditional; a literature search cannot establish priority by absence alone.

## Inherited system

- Environment: `swm/ReacherDMControl-v0`, task `qpos_match`, RGB `(224,224)`, maximum episode steps 1000 as in the existing collector.
- Checkpoint: `quentinll/lewm-reacher`, pinned HF revision `62adae4b71dc474ddf8f794c476ebfe737a743ca`.
- Expected weight SHA256 from the previous completed audit: `eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469`. It was not recomputed this turn.
- Architecture: exact public snapshot in `lewm_reacher_config.json`. ViT-tiny encoder, patch14, 192-dimensional embedding; six-layer predictor, 16 attention heads with dim_head64, hidden192, MLP2048, three-frame context. The attention configuration is copied rather than inferred from hidden width.
- Raw action dimension2; five raw controls per model action, giving action-encoder input10. Normalize by the existing analytic Uniform[-1,1] standard deviation, sqrt(1/3).
- Horizon5 model steps =25 primitive controls. No primitives, learned controller, new reward model or observation tokenizer.
- Encoder and projector frozen throughout the first learned comparison; predictor, action encoder and prediction projection are the trainable modules, identically across fine-tuning arms.
- Goal cost: mean squared distance to the unchanged goal embedding.
- Physical audit cost: maximum wrapped joint error divided by0.05. Success: all wrapped joint errors strictly below0.05 radians.

Existing code to reuse: `vin-research/order-jepa/scripts/run_reacher_stage_a.py`, specifically `build_anchor`, `restore_physics`, `execute_candidates`, `make_transform`, and `encode_and_rollout`; plus `order_jepa/core.py` and `order_jepa/lewm_reacher.py`.

Existing server paths recorded in the scripts:

```
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
/mnt/data/nhatnc129/jepa/lewm_stage0/checkpoints/models--quentinll--lewm-reacher
/mnt/data/nhatnc129/jepa/order_jepa/reacher_stage_a/provenance.json
/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-worldmodel
```

Read and verify the recorded source commit at execution; do not invent a source SHA from the HF checkpoint revision. The source checkout must pass the existing clean-tree/provenance checks. The local pull has the audit code but was not verified to contain the full upstream runtime. Previous job completion is reported from RUN_STATUS, not a fresh queue query.

## Data specification

Use the existing deterministic random-trajectory collector rather than downloading the original large archive or building another arena. These are fresh trajectories from the same collection-policy family, not the exact original training dataset.

| Split | Episodes | Primitive controls/episode | Root seed | Use |
|---|---:|---:|---:|---|
| Train bank |64|200|20260910|PCA charts and later fine-tuning|
| Development |16|200|20260911|Select chart settings, intervention strength and one checkpoint|
| Test |40|200|20260912|One locked evaluation,8 anchors/episode|

Record observations every five primitive controls, using the same image transform. Construct consecutive three-frame history vectors, h=(z[t-2],z[t-1],z[t]) in R^576. Store aligned action histories and next-block targets. Split by source episode before creating overlapping windows. Use explicit disjoint reset-seed ranges; assert no reset-seed overlap across splits or with the old audit, since differently named RNG roots alone are not proof of independence.

For each selected anchor, record simulator state, initial history/action history, candidate raw actions, and encoded true/predicted intermediates at every block. The current JSON-only collector does not persist all these tensors. Adding this cache is required new instrumentation. Do not claim that existing scalar results can be decomposed into tangent/normal error without recollection.

The initial smoke uses only32 dev anchors from four dev episodes. It is a debugging/descriptive screen, not an inferential result. An uninformative screen does not establish a tight null.

## No-training sequence of tests

### A0: localize accumulation

On the same candidates compare:

1. Released autoregressive endpoint prediction.
2. Final-block prediction supplied with the true preceding observation/action history for that candidate.
3. Encoded true endpoint.

Arm2 uses privileged intermediate observations and is diagnostic only. If it improves MSE but not selected physical regret, do not infer useful accumulation headroom. If it does not improve even MSE, look for one-step error or representation insufficiency before proposing a stability intervention. The teacher-forced advantage alone is not evidence for a manifold mechanism.

### A1: test a deployable, nonparametric correction

Estimate local affine charts from train history vectors using Euclidean kNN followed by centered PCA. Primary setting: k=64, rank12. Development-only sensitivity: k in {32,64,128}, rank in {6,12,24}; choose using held-out reconstruction of real histories, never test planning success. If this grid cannot model real histories without erasing their valid variations, fail the estimator gate instead of increasing the search indefinitely.

For a predicted history h, the local retraction is R(h)=mu+U U^T(h-mu). Use h_new=h+alpha*(R(h)-h), alpha in {0.25,0.5,1}, selected on development physical regret. Actual action histories are never altered. Project only the predicted slots of the history window, keeping genuinely observed context slots fixed; fit charts conditional on those fixed coordinates using a regularized least-squares solve. Compare to applying the same correction only at the final endpoint to distinguish a cost effect from rollout propagation.

This conditional projection is an implementation requirement, not a claim that an arbitrary full-history projection is valid. Check that the wrapper with alpha=0 reproduces native rollout exactly. Historical predictions may be revised; measured observations may not.

Matched controls:

- No correction (released model).
- Global PCA reconstruction, same rank and strength-selection budget.
- Nearest train-history reconstruction, keeping measured slots fixed.
- Shrinkage towards the local mean with correction norm matched per query.
- Random subspace with the same rank; scale its correction to the same norm as the chart correction. Report saturation/degenerate cases rather than silently dropping them.

The chart receives no test physics or goal information. Simulator-derived tangent projectors at the true endpoint, if explored separately, are privileged diagnostics and cannot count as deployable gains. In high dimension, removing most coordinates can trivially remove most error; correction-energy controls are essential.

### A2: remove witness-panel dependence

Keep the original ordinary-candidate panel as a reproduction stratum. Add a distinct stratum containing the final population and returned plan of the unchanged baseline CEM. Propose a fixed CEM configuration: horizon5, population96,12 refits,10 elites, action bounds[-1,1], execute one5-control block per MPC step, at most40 replans. This is an experimental configuration, not asserted to be the official Reacher default. Use the existing stable-worldmodel CEM implementation; log all remaining defaults and the exact resolved config before comparing arms.

Do not inject the hidden witness into CEM initialization or population. All methods get only observations, action history and goal image. Replay final candidates from identical reset states to measure true physical outcomes. A gap restricted to witness-neighborhood candidates is insufficient to start method training for planning. CEM itself is inherited on the server; its adapter/configuration in the new experiment has not been implemented or smoke-tested locally.

## Mechanistic measurements and proposed learned intervention

Let F(h,a) predict the next embedding and let G(h,a) be the corresponding shifted three-frame history. Estimate tangent projectors P(h)=UU^T from real train histories and N=I-P. Actions remain fixed in a state-sensitivity measurement.

Decompose the local linearization using projectors at the input and successor histories:

```
A_TT = P_next D_h G P
A_TN = P_next D_h G N
A_NT = N_next D_h G P
A_NN = N_next D_h G N
```

Large A_NN can amplify normal error; A_TN can turn normal error into a lasting tangent error even when later projection removes the normal component. Thus measuring only distance to a manifold is insufficient. A valid physical transition can expand tangent distances. The shifted-history operator also copies old slots, so demanding strict one-step contraction of the entire augmented history is inappropriate.

Measure finite-horizon response of G composed three times (one full context replacement), with identical future actions. Use finite differences in eval mode at eps={0.01,0.03,0.1} times the median train-neighbor distance. Use four random normal probes per history and equally normalized tangent/random probes. Report finite-amplitude behavior as well as sensitivity near zero. Dropout must not masquerade as response noise.

Proposed training loss, conditional on the preceding gates:

```
L = L_AR + lambda_N * E relu(||N_next DeltaG3|| / eps - q)^2
         + lambda_TN * E ||P_next DeltaG3 / eps||^2
```

DeltaG3=G^3(h+eps*v_N,A)-G^3(h,A), with unit train-chart normal direction v_N and identical action continuation A. Projectors and charts are frozen. Primary q=0.9, with development choices q in {0.5,0.9}; lambda_N=lambda_TN in {0.01,0.1,1} after logging loss normalization. L_AR is the same five-block supervised rollout loss in every learned arm. Any term preserving observed pairwise transition differences must also be added to matched baselines; it must not become unaccounted additional supervision.

For this training loss, successor projectors are evaluated at the encoded true successor history from the training trajectory. At deployment, charts can be queried only using the available observed/predicted history. The learned regularizer itself adds no simulator queries at inference. Keep this distinction explicit in the implementation and logs.

This is an empirical finite-difference stability objective, not a certified operator-norm bound. Four random probes do not control worst-case amplification. The construction is a structured Jacobian/noise regularizer in the small-noise limit; do not sell its algebra as a new class of objective. Its prospective contribution depends on the estimator, selective physical preservation and planning advantage.

A plausible local error recurrence has both normal amplification and normal-to-tangent coupling terms. Bounding those can bound accumulated representation error under smoothness, chart accuracy and coverage assumptions. It does not prove planning success or uniformly control a planner-selected cost gap. Do not recycle a data-average bound as a closed-loop guarantee. For contact tasks, do not assume a single smooth manifold across mode boundaries.

## Learned baselines and fixed budgets

First fine-tuning comparison: six arms, three seeds {0,1,2}:

1. Matched five-block AR fine-tuning.
2. AR plus isotropic input noise/consistency.
3. AR plus global Jacobian regularization, estimated by matched random directions.
4. AR plus normal-direction objective above.
5. AR plus same objective with random subspaces.
6. AR plus the best frozen chart correction from A1, to test whether learning adds value.

Released weights are also evaluated as a zero-update reference. Match training windows, targets, optimized modules and minibatch order by seed. Report both update count and measured training FLOPs/GPU time; the plain AR arm gets an additional equal-time run so extra regularizer compute is not hidden. Optimize with AdamW, learning rate1e-5, weight decay1e-4, batch64, gradient clip1.0, at most2000 updates; validation at500-update intervals. These are pilot choices, not tuned optima.

Train only after A0–A2 qualification. Hyperparameter search is development-only and capped by the compute allocation below. If the cap cannot cover these comparisons, report an incomplete comparison and do not claim a winner. Strong recent systems such as Fast-LeWM and a second planner such as GRASP belong in a later paper evaluation; they are not prerequisites for the first mechanism test or substitutes for the cheap controls.

## Decision rules and costs

Primary outcome: paired difference in physical selected regret on the CEM-generated panel. Secondary: selected success, latent prediction error, held-out real-history reconstruction, and physically valid pair-response fidelity. Bootstrap by source episode, not candidate or overlapping anchor. Use one locked test evaluation after development, and adjust simultaneous comparisons against several controls (e.g. Holm); do not select a convenient comparator after seeing the test.

Suggested qualification gate: at least10% relative regret reduction over the strongest development-selected cheap control, with a paired95% interval excluding zero improvement, and success noninferiority margin3 percentage points. Report absolute regret change as well. These practical thresholds are explicit design choices. A wide interval gives INCONCLUSIVE_WITHIN_CAP, not evidence of no effect; do not automatically allocate more compute.

If baseline regret is zero or numerically negligible, the relative gate is undefined or uninformative: report insufficient headroom rather than divide by a small denominator. Do not change the success tolerance to manufacture a gap.

The learned stage must beat the strongest matched training control and frozen projection under the same rules. Before a main programme, require fresh CEM reoptimization and closed-loop improvement; a fixed-pool win alone is insufficient. Test the exact execution success predicate, not only terminal proxy cost.

| Gate | Maximum allocation proposed | Opens what? |
|---|---|---|
| Preflight and32-anchor smoke |1 engineering day;8 GPU-hours;64 CPU core-hours|Larger no-training evaluation only|
| Full no-training evaluation |16 additional GPU-hours;128 CPU core-hours|Matched fine-tuning if causal/decision gates pass|
| Matched fine-tuning screen |36 additional GPU-hours|Closed-loop confirmation if successful|
| Confirmation/transfer |No allocation yet|Requires results and a manipulation-specific gate|

Caps are resource ceilings, not throughput predictions. Persist partial outputs; a timeout is neither GO nor a statistical null. No jobs have been submitted by this document. Follow the repo Slurm policy for model loading, simulation, encoding and analysis. Before any submission, check both squeue and sacct and verify no duplicate work.

## Manipulation transfer and stop conditions

Use existing LeWM–OGBench Cube and `rollout_repair_gate` replay only after an independent gate there. That programme already failed deployment-matched predictor repair. Its published local result is a counterexample to any blanket claim that better rollout training improves planning. Cube's history length, action dimensions and success predicate must remain its native configuration; do not transplant Reacher's numeric setup.

Stop this candidate if chart estimates cannot preserve real held-out variations; if equally sized random corrections work as well; if teacher-forced improvement has no physical decision value; if no benefit exists on CEM-produced candidates; if fine-tuning adds nothing over simple correction; or if CEM reoptimization erases the gain. Do not rescue it by building a bespoke arena, adding state supervision only to the proposed method, or renaming the objective.

Reacher alone supports a control pilot. No contact-rich/long-horizon manipulation paper is qualified until a relevant existing manipulation arena passes. This document specifies the first programme segment concretely and leaves later segments closed rather than pretending to have a validated full-paper recipe.
