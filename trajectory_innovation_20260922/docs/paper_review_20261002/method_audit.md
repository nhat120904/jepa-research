# CTA method/problem audit for the developmental manuscript — 2026-10-02

Scope: read-only manuscript/source audit. No model loading, physics, tests, batch jobs, or manuscript edits were performed. The active paper sections are `paper_cvpr/sec/3_problem.tex`, `4_method.tex`, and `6_conclusion.tex`. Implementation was checked on H100 in `trajectory_innovation_20260922/ti_wm/cta.py`, `cta_parallel.py`, `codec.py`, `cta_geometry.py`, `sibling.py`, `cta_runtime.py`, `pusht_runtime.py`, and `scripts/cta_train_v2.py`. The small JSON config for L15 checkpoint 56504 was also read. Native-hit continuation details below refer to the locally staged continuation source, not a completed experiment.

## 1. Main assessment and defensible contribution scope

CTA's implemented distinction is a future-supervised, context-conditioned segment target, forecast from a proposed action chunk and evaluated through a reader whose only candidate-specific input is that forecast. This is a clear architectural and supervision choice. Compression, factorized prediction, policy reranking, conditional coding, and query reuse are not individually sufficient novelty claims.

Three defensible provisional contributions:

1. A conditional segment-target architecture that learns a compact FSQ target from observed context and actual segment observations, then predicts its continuous coordinate mean from context and a proposed chunk. The deployment reader sees context, predicted target, and goal; it cannot read the proposed action or actual future directly.
2. A decision-aligned training recipe that combines source-code ranking and future-feature reconstruction, followed by categorical target prediction and reader-score consistency/ranking. The native-hit pilot is a bounded investigation of additional native-objective alignment with matched continuation controls; it is not an established new principle or result.
3. A controlled evaluation that separates actual-future reading, source-code reading, predicted-code reading, and learned closed-loop selection, with matched proposals and numerically fixed policy/scorer shapes. Empirical quality/compute gains require completed comparisons, held-out uncertainty, and dedicated deployment timing. The evaluation design itself should not be described as a measured gain.

A factorized direct baseline is particularly informative because direct action-conditioned bottlenecks can also be computed once and reused across queries. Endpoint prediction is a legitimate comparison even when the task is endpoint based. A temporal-task pivot is not a prerequisite for the current contribution.

## 2. Exact code, prediction and information formulation

Use a grid-valued code, rather than integer vocabulary indices inside an expectation. Let

\[
\mathcal Q_l=\{(r-\lfloor l/2\rfloor)/\lfloor l/2\rfloor:r=0,\ldots,l-1\},\qquad
\mathcal Q=\mathcal Q_8\times\mathcal Q_8\times\mathcal Q_4.
\]

The deterministic, evaluation-mode source encoder produces

\[
S=q_\phi(C,\tau)\in\mathcal Q^M\subset\mathbb R^{M\times3}.
\]

The 8-level coordinates are `{-1,-.75,-.5,-.25,0,.25,.5,.75}`; the 4-level coordinate is `{-1,-.5,0,.5}`. Each token has 8×8×4=256 possible values, hence a fixed-length token-index representation requires 8 bits. With M=16, the source alphabet contains at most 2^128 sequences and

\[
I(\tau;S\mid C)=H(S\mid C)\le M\log_2 256=128\text{ bits}.
\]

Equality of mutual information and entropy assumes the frozen deterministic encoder (training uses dropout). This is an upper bound on source-code capacity, not measured entropy, an entropy-coded rate, or a deployment communication bound. There is no learned entropy coder or explicit conditional-entropy minimization in v2. Describe the rate-distortion expression as motivation/finite-alphabet constrained design, not an optimization that measures or attains the information-theoretic optimum.

The parallel predictor has joint contextual hidden computation but conditionally factorized categorical output coordinates:

\[
p_\theta(S\mid C,A)=\prod_{m=1}^{M}\prod_{j=1}^{3}p_{\theta,mj}(S_{mj}\mid C,A),\qquad
\hat S_{mj}=\sum_{r=0}^{l_j-1}p_{\theta,mj}(r\mid C,A)\,\mathcal Q_{l_j}[r].
\]

`ParallelFSQWM.forward` returns a float32 M×3 coordinate mean; at M16 this is 48 continuous numbers. The reader is nonlinear, so generally

\[
D(C,\mathbb E[S],g)\ne\mathbb E[D(C,S,g)].
\]

Different categorical distributions can have the same mean. Replace “exposes the world model's uncertainty” by “provides a differentiable soft prediction”; do not imply calibrated uncertainty, risk integration, full-posterior retention, or discrete hard-code deployment. Any reduction in ties is empirical and requires its own reported comparison.

The source encoder sees `C,tau`; the predictor sees `C,A`; the reader sees `C,S,g`. Geometry and native-hit labels enter training losses, while the actual future is the target used by the source encoder. Therefore “the world model never sees labels” is too broad: labels are not model inputs, but supervised ranking/native-hit losses use them.

## 3. Inputs, reconstruction and exact losses

Context has 256 current-image tokens, 64 pooled previous-image tokens and one 4-D proprioception token. Each image token has 128 PCA dimensions from frozen DINOv2 ViT-S/14 patch features. Native observation pixels are 96×96; DINO input is resized to 224×224. Agent positions are the current and preceding positions divided by 512. Each proposed action is embedded as

\[
e(a_h,p_t)=[a_h/512-1/2,\;(a_h-p_t)/64]\in\mathbb R^4.
\]

The source future input contains 256 endpoint-image tokens, 3×64 pooled intermediate-image tokens, and one endpoint-proprioception token (last two actual agent positions). Intermediate positions are rounded quarter points: `(2,4,6)` for L8 and `(4,8,11)` for L15. Segments stop on native termination or the 300-step episode limit; absent intermediate frames repeat the final observation. The representation summarizes these samples, not every frame of an entire trajectory.

The feature decoder reconstructs endpoint and intermediate *image features only*. It does not reconstruct future proprioception, although the source encoder sees it. Its outputs are residuals added to the current image's full or pooled feature grid.

For a minibatch of sibling banks define

\[
\mathcal P=\{(b,i,j):\ell_{bi}-\ell_{bj}>\epsilon\},\quad
\mathcal L_{rank}=\frac{1}{|\mathcal P|}\sum_{(b,i,j)\in\mathcal P}\operatorname{softplus}[-(s_{bi}-s_{bj})].
\]

Empty eligible-pair sets contribute differentiable zero. The stage-1 code branch uses

\[
\mathcal L_{code}=\mathcal L_{rank}(D(C,S,g),\ell)
+\alpha\mathcal L_{feat}+\lambda_{sat}\mathbb E[(|z_{pre}|-1.5)_+^2],
\]

\[
\mathcal L_{feat}=\tfrac12\left[
\operatorname{MSE}(\hat F_{end},F_{end})/\nu_{end}
+\operatorname{MSE}(\hat F_{mid},F_{mid})/\nu_{mid}\right].
\]

The normalization constants are training estimates of the error of copying the current full/pooled feature grid. Only the *zero-residual copy predictor*, in expectation under that normalization distribution, scores one. A decoder with constant code can still use context to predict changes and can score below one. Reconstruction encourages future information but does not prevent collapse or prove transfer. Do not equate it with EMA or variance regularization as a guarantee.

The parallel predictor's NLL is the mean coordinate cross-entropy:

\[
\mathcal L_{NLL}=\frac{1}{3BM}\sum_{b,m,j}-\log p_{\theta,mj}(d(S_{bmj})\mid C_b,A_b).
\]

The sum across coordinates is nats/token; full segment NLL is 3M times the optimization mean. Keep units explicit when reporting rates or information.

For one bank, center reader scores `s` and source-teacher scores `t` by their own candidate means. Consistency is

\[
\mathcal L_{cons}=\operatorname{SmoothL1}((s-\bar s)/T,(\operatorname{sg}(t)-\bar t)/T).
\]

`T` is a fixed train-only RMS within-bank teacher-score scale (floored at .001). Weighted prediction ranking is

\[
\mathcal L^{weighted}_{rank}=
\frac{1}{|\mathcal P|}\sum_{(b,i,j)\in\mathcal P}
\min(1,(\ell_{bi}-\ell_{bj})/\sigma)\operatorname{softplus}[-(s_{bi}-s_{bj})].
\]

The predictor objective is their unweighted sum, `NLL + consistency + weighted rank`. The frozen reader/source encoder receive no updates, but gradients through the reader shape the predictor. Endpoint prediction uses normalized endpoint-image and future-proprioception MSE instead of categorical NLL, with the same consistency and weighted-rank pattern. Stage-1 FULL and DIRECT are trained by unweighted ranking. v2 stage 1 puts all groups under one optimizer and shared gradient clipping; this should be disclosed if claiming matched optimization. The staged continuation uses independent optimizers/clips.

Geometry training labels are

\[
\ell(x,g)=-\frac1{512}\frac1{8}\sum_{v=1}^{8}\|v(x)-v(g)\|_2,
\]

using corresponding native block vertices/privileged poses. The native terminal task is coverage >95%, which differs from this dense registration label. They are not interchangeable reward definitions. Current v2 can optionally change labels with success bonus/hindsight; the original L15 recipe has both set to zero.

The staged native-hit pilot uses hit flags h from native termination, not thresholded cached geometry. In mixed banks `0<sum(h)<K`,

\[
\mathcal L_{hit}=\mathbb E_{b\in\mathcal B_{mix}}
\left[\log\sum_k e^{s_{bk}/T}-\log\sum_{k:h_{bk}=1}e^{s_{bk}/T}\right].
\]

No mixed bank gives differentiable zero. This rewards probability mass on any successful candidate. The pilot also changes replay sampling (standard-bank mass and mixed-hit replay), so its contrast estimates a combined sampling/objective intervention unless these changes are separately controlled. CTA, ENDPOINT and DIRECT have respective continuation controls and identical selected-update budget/selection rule. No native-hit result is established by this source audit.

## 4. Correct theory and information claims

No proposition appears in the current active problem/method sections. A formal statement can be included as analysis, without claiming novelty or guaranteed task success:

Let `t_k` be source-reader scores and `s_k` predicted-reader scores on a fixed bank and query (or averages over the same goal views). Let their centered discrepancies satisfy

\[
\epsilon_s=\max_k|(s_k-\bar s)-(t_k-\bar t)|.
\]

For `k_t=argmax t` and `k_s=argmax s`,

\[
t_{k_t}-t_{k_s}\le2\epsilon_s.
\]

If the unique source-score winner exceeds every alternative by more than `2 epsilon_s`, prediction selects that winner. Proof: centering does not change argmax; insert predicted centered scores between the two source scores and use `s[k_t] <= s[k_s]` plus the two discrepancy bounds. This motivates within-bank consistency and a diagnostic comparing teacher margins against prediction errors.

This is a reader-score bound. Logistic ranking does not calibrate reader scores to physical utility, so it does not directly bound coverage regret, success differences, or closed-loop regret. A physical-utility bound would require an explicit calibration assumption/error term, and multi-step control would require distribution/control assumptions. Future-feature reconstruction also does not imply a geometry-utility bound: no proved Lipschitz/invertibility relation connects DINO features to native task labels. Do not manufacture such a theorem for the rubric.

For ideal true conditional distributions, the identity

\[
I(S;A\mid C)=H(S\mid C)-H(S\mid C,A)
\]

is correct. But cross-entropy differences of approximate models obey

\[
CE(r_C)-CE(p_{CA})=I(S;A\mid C)+\mathbb E KL(P_C\|r_C)-\mathbb E KL(P_{CA}\|p_{CA}).
\]

Thus the gap is not generally an unbiased MI estimate or a bound. v2 does not train a context-only predictor at all; remove the claim that it estimates conditional MI. Older autoregressive training did train a prior and can be identified as a historical diagnostic. Residual conditional entropy includes partial observability and context compression, not only intrinsically unpredictable dynamics; stochastic predictions remain possible.

## 5. Claims to replace directly

- “Any goal” / “query transfer” → “The predictor is goal independent and its output can be reused by the goal-conditioned reader; transfer to unseen goal distributions is untested.” Sixteen PushT goal views share one target pose. New raw goal images also require feature extraction before reader evaluations.
- “The feature decoder prevents collapse” → “An auxiliary reconstruction loss encourages preservation of future-feature information; its effect is evaluated empirically.”
- “Code spends bits only on what history cannot predict” → “Sharing context permits conditional encoding but does not force a residual-only code.”
- “128-bit deployment code” → “A source alphabet with a 128-bit maximum fixed-length index representation; deployment uses its 48-coordinate predictive mean.”
- “Full distribution exposes uncertainty” → “Factorized categorical prediction provides a differentiable coordinate mean.”
- “Direct and per-step WM are limiting cases of Eq. 1” → “Architectural comparators.” `S=A` is not generally a function of `(C,tau)`, and continuous `S=tau` violates the finite-alphabet constraint; neither is literally a special case of the stated optimization.
- “A direct scorer cannot reuse prediction across goals” → remove. `h(C,A)` followed by `D(h,g)` can also cache a query-independent representation; future supervision must be isolated with a factorized direct comparator.
- “All closed-loop arms see identical proposals” → “They share proposal policy, random-seed mapping, K, execution horizon and numerical sampling protocol; offline comparisons use identical banks.” Closed-loop states and therefore proposals diverge after different choices.
- “Exact simulator copy” → “Validated cloning with exact restored kinematics and measured agreement with live continuation within the reported tolerance.” Historic smoke allows .01; its observed deviation was approximately .002.
- “One-step oracle upper bound on attainable episode success” → avoid. Greedy endpoint/geometry choices do not upper-bound a learned closed-loop selector's long-term success.
- “Native-hit establishes intermediate-event benefits” → avoid. Collection stops on termination and repeats final frames, so endpoint state can often retain that hit. Segment/path superiority still needs an appropriate controlled comparison.
- “The FULL-to-CODE gap measures loss caused solely by compression” → avoid. FULL reads endpoint image/proprioception through its own reader; CODE's source encoder also sees intermediate images and uses a different trained reader. Their difference mixes evidence view, compression and reader optimization. Source-code versus predicted-code reading uses the same frozen code reader and is the cleaner forecast diagnostic. None of these offline gaps is a decomposition of learned closed-loop success.
- “A second path task addresses this limitation” → remove from conclusion until real evidence exists. Keep it as future evaluation, not an obligatory pivot.

## 6. Actual L15 v2 configuration and schedule correction

Checkpoint 56504 config records M16, chunk L15, direct_layers8, lr3e-4, weight decay .05, dropout .1, 16,000 stage-1 and 10,000 stage-2 updates, 16 decisions/update, warmup500, rec1, sat1, rank_margin.001 and rank_scale.01. It uses r4std/r4pert (18,606 banks each) and r4dev selection (1,476 banks). This is from-scratch v2, not a warm-started historical checkpoint.

Best checkpoints are chosen by retained geometry gap, not the network's training objective: codec step8000, FULL16000, DIRECT12000, CTA8000, ENDPOINT8000. v2 training-time selection averages four fixed goal views `(0,5,10,15)`; final ladder/deployment averages all sixteen. This differs from the staged continuation selection, which uses all sixteen and `geometry retained gap + .25*mixed hit capture`.

Current manuscript statements “L8”, “stage2 8000 updates, lr1e-4, 32 decisions/update”, “best own held-out objective”, “all current checkpoints warm-started”, and “context-only model trained alongside v2” must be revised or explicitly labeled historical. Do not use source defaults as proof of what was run; cite the actual config and selected steps.

## 7. Suggested manuscript organization

Problem: fixed proposal bank, finite-horizon native utility/progress, observed context, actual-future supervision, deployable scoring contract, finite source alphabet versus continuous prediction. Replace literal limiting-case arguments with a concise comparison table.

Method: inputs and sampled segment; source encoder/grid; reader/reconstruction and their objectives; factorized parallel predictor and centered consistency; matched endpoint/direct alternatives; optional bounded native-hit continuation; deployment with cached goal features and deterministic tie handling.

Analysis: alphabet capacity and soft prediction distinction; optional teacher-ranking margin lemma; no unsupported MI or generalization theorem.

Conclusion: state what completed experiments establish, numerical/reproducibility constraints, source-vs-deployment capacity distinction, dependence on privileged training labels and pretrained proposal policy, and remaining held-out/query/path/latency evidence. Do not conclude a quality/compute or generalization advantage from architectural token counts alone.
