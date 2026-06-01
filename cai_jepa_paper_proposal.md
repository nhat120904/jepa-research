# Counterfactual Action-Identifiable JEPA World Models for Robot Planning

## Detailed Research Proposal

---

## 1. Background

### 1.1. The JEPA World Model Family

Action-conditioned Joint-Embedding Predictive World Models (JEPA-WMs) have emerged as a dominant architectural family for zero-shot robot planning in learned latent spaces. The core idea is to decouple visual representation learning from dynamics modeling: a powerful self-supervised vision encoder (typically DINOv2 or V-JEPA 2) is frozen, and a lightweight action-conditioned predictor is trained on top to model how latent representations evolve under control actions. Planning is then performed entirely in this latent space using model-predictive control with sampling-based optimizers such as the Cross-Entropy Method (CEM) or Model Predictive Path Integral (MPPI).

The canonical pipeline operates as follows. Given an observation $o_t$, the frozen encoder $E_\phi$ produces a latent state $z_t = E_\phi(o_t)$. An action-conditioned predictor $F_\theta$ is trained to predict the next latent state given the current state and an action: $\hat{z}_{t+1} = F_\theta(z_t, a_t)$. The training objective is a prediction loss against the encoded next observation:

$$\mathcal{L}_{\text{pred}} = \| F_\theta(z_t, a_t) - \text{sg}(E_\phi(o_{t+1})) \|^2$$

where $\text{sg}(\cdot)$ denotes the stop-gradient operator. At test time, given a goal observation $o_g$ with corresponding latent $z_g$, the planner searches for an action sequence $a_{t:t+H-1}^*$ that minimizes the distance between the rolled-out predicted future and the goal:

$$a_{t:t+H-1}^* = \arg\min_{a_{t:t+H-1}} d\left( F_\theta^{H}(z_t, a_{t:t+H-1}), z_g \right)$$

where $F_\theta^H$ denotes $H$-step autoregressive rollout and $d$ is a distance function (typically L2 or cosine).

Representative instantiations include DINO-WM (Zhou et al., 2024), which uses frozen DINOv2 patch features and a causal vision transformer predictor; V-JEPA 2-AC (Assran et al., 2025), which uses frozen V-JEPA 2 features and a 300M-parameter transformer predictor trained on 62 hours of Franka manipulation video from the DROID dataset; PLDM (Sobal et al., 2025), which extends JEPA-WMs to goal-conditioned reinforcement learning settings; and most recently the JEPA-WM of Terver et al. (2026), which systematically ablates architectural design choices and identifies AdaLN-based action conditioning with multistep rollout loss as the empirically optimal configuration.

### 1.2. Why This Family Matters for Robot Planning

JEPA-WMs are appealing for robot planning for three reasons. First, they decouple representation learning (which can leverage large unlabeled video corpora) from dynamics modeling (which requires action-labeled interaction data). This allows the visual backbone to benefit from internet-scale pretraining while the action-conditioned predictor remains compact and trainable on relatively small robot interaction datasets. Second, latent-space planning is computationally efficient compared to pixel-space alternatives such as diffusion video models — V-JEPA 2-AC plans at approximately 16 seconds per action versus 4 minutes per action for Cosmos. Third, the absence of pixel reconstruction objectives means representations are not forced to preserve task-irrelevant visual details, potentially yielding more semantically meaningful latent dynamics.

Empirically, JEPA-WMs achieve strong results on standardized benchmarks. DINO-WM reaches 90% success rate on Push-T and 98% on PointMaze. V-JEPA 2-AC achieves 100% success on Franka reach, 65% on cup grasping, and 65-80% on pick-and-place across two independent labs deployed zero-shot. These results have established JEPA-WMs as a credible foundation for general-purpose robot planning.

### 1.3. The Standard Evaluation Paradigm

Current evaluation of JEPA-WMs focuses on two axes: (i) prediction quality, measured by reconstruction or feature-space MSE on held-out factual transitions, and (ii) planning success rate, measured by deploying the model with CEM/MPC on goal-conditioned manipulation or navigation tasks. Both axes evaluate the model under conditions resembling the training distribution: the prediction loss tests one-step forecasting on factual $(s_t, a_t, s_{t+1})$ tuples, and planning success rates aggregate end-to-end behavior over goal images, conflating prediction quality with planner design choices.

This evaluation paradigm has a fundamental blind spot. It does not directly measure whether the predictor reliably **distinguishes** futures induced by different actions from the same initial state. A predictor that ignores action input entirely could still achieve reasonable prediction loss on factual transitions if the marginal distribution of next states $p(z_{t+1} \mid z_t)$ is sufficiently peaked. Conversely, a predictor that produces visually plausible but action-insensitive rollouts could pass goal-conditioned planning evaluations on tasks where the action effect is large enough that even noisy predictions are sufficient.

---

## 2. The Problem: Action Grounding in JEPA World Models

### 2.1. Precise Problem Statement

The question this paper addresses is the following:

> **Do action-conditioned JEPA world models reliably distinguish the futures induced by different actions across the diverse range of states encountered during robot planning, and if not, how can we systematically diagnose and correct this failure?**

We deliberately avoid the strong claim that existing models "ignore actions." This claim has been refuted: ablation studies in V-JEPA 2-AC show that removing action conditioning causes planning to collapse, and Terver et al. (2026) demonstrate via qualitative counterfactual rollouts that their improved JEPA-WM produces visibly different futures when given contrasting actions (e.g., "open gripper + move up" versus "close gripper + move up" from the same initial state where the robot hand is near a cup).

Instead, we identify a narrower and empirically grounded gap. Existing JEPA-WMs may pass qualitative counterfactual sanity checks on clear-cut scenarios, but the literature lacks:

1. A **systematic quantitative diagnostic protocol** for action grounding across diverse state regimes, particularly contact-rich, fine-precision, and ambiguous-effect regimes where action grounding is most fragile.
2. **Understanding of how counterfactual sensitivity correlates with planning success** — it is currently unknown whether a model that produces well-separated counterfactual rollouts also achieves better planning success, and which counterfactual metrics best predict downstream planning ability.
3. **Training objectives that explicitly enforce action-conditional separability** in the predictor's latent space. Existing objectives — factual prediction loss, multistep rollout loss, inverse dynamics auxiliary loss — provide only indirect pressure for action grounding.

### 2.2. Why the Standard Counterfactual Eval Is Insufficient

Terver et al. (2026) introduce a counterfactual visualization protocol in which two action sequences are unrolled from the same initial state, and the resulting trajectories are decoded for qualitative inspection. While this protocol successfully demonstrates that their model exhibits action sensitivity in clear-cut binary contrasts, it has four methodological limitations that prevent it from serving as a rigorous diagnostic:

**Limitation 1: Binary contrast on highly separable actions.** The standard test case (open versus close gripper, both combined with upward motion) is essentially the most action-discriminable contrast possible in manipulation: the two actions have qualitatively different effects (object lifted versus object remaining), large visual differences (object position change versus no change), and clear physical interpretation. A model that fails this test is essentially broken; passing it does not imply general action grounding.

**Limitation 2: Hand-selected scenarios.** The test is applied to a small number of curated initial states. There is no systematic sampling across the state distribution encountered during planning. State regimes that are quantitatively dominant in planning (pre-grasp approaches, fine gripper adjustments, contact micro-adjustments) are underrepresented or absent.

**Limitation 3: Qualitative judgment.** The evaluation is based on decoded image rollouts inspected by human readers. This precludes statistical analysis, model comparison via confidence intervals, and any form of large-scale benchmarking.

**Limitation 4: No connection to planning.** The counterfactual rollout test is conducted in isolation from planning evaluation. It is not established whether models that perform better on this test also plan better, nor whether the gap between models on this test predicts the gap in planning success rate.

### 2.3. Connecting Action Grounding to Planning Failure

The practical importance of action grounding becomes clear when one examines how CEM-based latent planning operates. At each replanning step, CEM samples $K$ action sequences (typically $K = 512$ to $2048$), rolls each sequence forward through the predictor, computes a cost (distance to goal latent), and uses the top-$M$ low-cost sequences to refit the sampling distribution. This iterative refinement is repeated for several rounds before the first action of the best sequence is executed and the robot observes the resulting state.

If the predictor is poorly action-grounded — that is, if $F_\theta(z_t, a)$ produces similar outputs for different $a$ — then the cost landscape over action sequences becomes nearly flat. The top-$M$ elites will essentially be sampled noise from the prior distribution, the refit distribution will not concentrate around any meaningful action region, and after several iterations the planner will output an action close to the prior mean. This failure mode is silent in the sense that the predictor still produces plausible single-step predictions on factual data; the failure is only visible when one queries counterfactual rollouts or attempts to plan.

The failure modes documented in V-JEPA 2-AC itself are consistent with this account. Meta's own paper reports that grasping a box succeeds only 25% of the time, compared to 65% for a cup, and explicitly attributes this to insufficient gripper precision: "the model requires more precise gripper control to ensure that the fingers are open wide enough to grasp the object." This is precisely the regime where action grounding is most fragile — the action effect (gripper width change) is small in visual terms, the relevant state context (contact configuration) is subtle, and the prediction must distinguish between actions whose visual consequences differ by millimeters of finger position.

### 2.4. Connection to Causal Reasoning in World Models

A growing literature in world models for embodied AI argues that predictive quality is only useful insofar as it supports action selection. A world model serves as an internal simulator for evaluating candidate actions before execution; for this role, what matters is not pixel-level realism but whether the imagined futures preserve the causal consequences of the candidate actions. A photorealistic predictor that produces visually convincing but action-invariant rollouts is useless for planning, while a less photorealistic predictor that correctly captures action-conditional outcome differences is highly useful.

Our framing thus situates the action grounding problem within the broader question of causal fidelity in learned world models. We argue that **action-identifiability** — the property that the predictor's outputs are reliably distinguishable conditional on the action input — is a necessary condition for any latent world model used for planning, and that it deserves to be diagnosed, measured, and explicitly optimized.

---

## 3. Related Work and Positioning

### 3.1. Contrastive World Models

The use of contrastive losses for learning latent dynamics has precedent in the structured world models literature. C-SWM (Kipf et al., 2019) trains an action-conditioned predictor with a contrastive objective in which negatives are sampled from the batch of other states. The Impact of Negative Sampling on Contrastive Structured World Models (Biza, van der Pol, Kipf, 2021) further studies how different negative sampling strategies affect downstream task performance. More recently, TWISTER (Burchi & Timofte, 2025) introduces action-conditioned contrastive predictive coding for transformer world models, again with negatives drawn from other trajectories in the batch.

A common feature of these works is that **negatives are sampled over states rather than over actions**. The contrastive objective trains the predictor to map $(z_t, a_t)$ closer to the true $z_{t+1}$ than to randomly sampled states. While this provides representational structure, it does not explicitly enforce that $F_\theta(z_t, a)$ varies meaningfully as a function of $a$. Our work flips this construction: we sample negatives over actions while keeping the target state fixed, asking the model to distinguish the future induced by the factual action from futures induced by counterfactual actions on the same initial state.

### 3.2. Counterfactual and Causal Approaches to World Modeling

Causal-JEPA (Nam et al., 2026) is the closest prior work in spirit. It introduces object-level latent interventions — masking entire object trajectories during training and requiring the model to infer the missing object's state from the remaining ones — as a form of counterfactual reasoning that prevents shortcut solutions. The empirical results show a 20% absolute gain in counterfactual visual question answering. However, the intervention is performed at the **object level** (masking objects), not at the **action level** (perturbing actions), and the framework is applied primarily to multi-object visual reasoning rather than action-conditioned planning. Actions are mentioned only as optional auxiliary variables.

The World Action Verifier (Liu, Finn, Du et al., 2026) addresses a related concern: that world models must produce reliable predictions over the full range of suboptimal actions encountered during planning, not just over the optimal trajectories in the training data. Their solution is to decompose action-conditioned state prediction into two verifiable components, state plausibility and action reachability, and use a sparse inverse model for post-hoc verification within a self-improvement loop. This is a complementary approach: their mechanism operates **after training** as a verification step that triggers retraining, while ours operates **during training** as an explicit loss term.

Counterfactual data augmentation methods such as CoDA and MoCoDA (Pitis et al., 2020, 2022) construct synthetic counterfactual transitions using local causal factorization and use them as additional training data. These methods generate counterfactual samples as **additional positives** for the standard prediction loss, rather than as **explicit negatives** in a contrastive objective. They are also primarily concerned with sample efficiency in reinforcement learning rather than action grounding in world models for planning.

### 3.3. Architectural Approaches to Action Conditioning

Recent work has improved action grounding through architectural innovations rather than training objectives. Terver et al. (2026) demonstrate that AdaLN-based action conditioning — which modulates each transformer block of the predictor with action information — substantially outperforms feature-based conditioning, and combining AdaLN with multistep rollout loss yields further gains. V-JEPA 2.1 (March 2026) improves grasping success rates by 20% over V-JEPA 2-AC by introducing dense feature losses that improve encoder spatial precision. These works establish that architectural choices substantially affect action grounding, but they treat action grounding implicitly through architecture rather than explicitly through the loss.

Our position is that architectural and objective-level interventions are complementary. AdaLN ensures that action information is available throughout the predictor's layers; our counterfactual objective ensures that the predictor is trained to use this information distinctively. We will use Terver-WM as a primary baseline and demonstrate that adding our objective on top of their architecture yields additional gains.

### 3.4. Inverse Dynamics Models as a Baseline

Inverse dynamics models (IDMs) — which predict $a_t$ from $(z_t, z_{t+1})$ — have been used since Pathak et al. (2017) and Agrawal et al. (2016) as auxiliary objectives for learning action-aware representations. An IDM auxiliary loss applied to predicted latents, $\hat{z}_{t+1} = F_\theta(z_t, a_t)$, indirectly pressures the forward predictor to encode action-distinguishing information in its output.

We treat IDM auxiliary loss as the **strongest competing baseline** and explicitly differentiate. IDM provides action **recoverability**: from a transition $(z_t, z_{t+1})$, the action can be inferred. This is a necessary but not sufficient condition for action grounding in forward prediction. A predictor can satisfy IDM recoverability by encoding $a_t$ as a residual feature in $\hat{z}_{t+1}$ that the IDM head decodes, while the bulk of the prediction remains action-invariant. Our counterfactual contrastive objective enforces a stronger property: action **separability**, meaning that $F_\theta(z_t, a)$ and $F_\theta(z_t, a')$ must be measurably different for different $a$ and $a'$. We will empirically demonstrate that these properties are dissociable — high IDM accuracy does not imply high counterfactual ranking accuracy — on existing baselines.

---

## 4. Contribution 1: CounterfactualBench

### 4.1. Overview

We introduce CounterfactualBench, a systematic quantitative diagnostic protocol for action grounding in action-conditioned latent world models. The protocol consists of four complementary metrics evaluated under a stratified sampling scheme over state regimes. The design goal is to enable statistically meaningful comparison of action grounding across models, identification of regimes where grounding fails, and correlation analysis with downstream planning performance.

### 4.2. Metrics

**Metric 1: Counterfactual Ranking Accuracy (CRA).** Given a factual transition $(z_t, a_t, z_{t+1})$ from a held-out evaluation set, we sample $K = 16$ counterfactual actions $\{a^-_k\}_{k=1}^{K}$ from a defined negative distribution. The ranking accuracy is:

$$\text{CRA} = \mathbb{P}\left[ d(F_\theta(z_t, a_t), z_{t+1}) < \min_k d(F_\theta(z_t, a^-_k), z_{t+1}) \right]$$

We report both top-1 ranking accuracy and Mean Reciprocal Rank (MRR). The metric directly tests whether the factual action's predicted future is closer to the true next state than any counterfactual prediction. A well-grounded model achieves high CRA; a model that ignores action input or produces action-invariant predictions achieves CRA near $1/(K+1) \approx 6\%$ (chance level).

**Metric 2: Action Usage Gap (AUG).** This metric directly quantifies the model's sensitivity to action input by comparing prediction error under factual versus shuffled actions:

$$\text{AUG} = \mathbb{E}\left[ \text{MSE}(F_\theta(z_t, \pi(a_t)), z_{t+1}) - \text{MSE}(F_\theta(z_t, a_t), z_{t+1}) \right]$$

where $\pi(\cdot)$ permutes actions across the batch. A model that ignores actions yields $\text{AUG} \approx 0$. A well-grounded model yields large positive $\text{AUG}$. Unlike CRA, this metric is sensitive to the absolute magnitude of action influence, not just the relative ordering.

**Metric 3: Counterfactual Trajectory Divergence (CTD).** For multi-step rollout, we measure how much the predicted future trajectory diverges under different action sequences from the same initial state:

$$\text{CTD}_H = \mathbb{E}\left[ d\left( F_\theta^H(z_t, a_{1:H}), F_\theta^H(z_t, a^-_{1:H}) \right) \right]$$

evaluated at horizons $H \in \{1, 3, 5, 10\}$. This metric captures whether action-conditional differences accumulate or collapse over rollout. Models that ground actions only at the first step but lose action sensitivity in autoregressive rollout (a common pathology) will show CTD that does not grow with $H$.

**Metric 4: Effect-Conditional Sensitivity (ECS).** AUG can be misleadingly low when measured over the full distribution because many transitions have negligible state change ($z_{t+1} \approx z_t$) where action input genuinely does not matter (e.g., free-space motion that produces little visual change). ECS gates AUG by effect magnitude:

$$\text{ECS} = \text{AUG} \mid \|z_{t+1} - z_t\| > \tau$$

This isolates the regime where action grounding actually matters — transitions where something happened — from the trivial regime where the model's behavior is uninformative.

### 4.3. Negative Action Sampling Strategies

The choice of negative action distribution materially affects what CRA measures. We define three strategies and evaluate all models under each:

**Random negatives.** $a^-$ is sampled uniformly from the action space bounds. This tests the easy case: distinguishing the factual action from arbitrary actions, most of which will be far from the data distribution. We expect high CRA from all models under this strategy.

**Opposite negatives.** $a^- = -a_t + \epsilon$ where $\epsilon$ is a small perturbation. This tests sensitivity to action direction reversal — a setting that mimics the "lift cup versus do not lift cup" qualitative test of Terver et al. but conducted quantitatively at scale.

**Nearest-neighbor hard negatives.** $a^-$ is sampled from $\{a_{t'} : z_{t'} \approx z_t, a_{t'} \neq a_t\}$, the set of actions taken in other transitions from similar initial states. This is the hardest setting: the negative action is one that a competent policy might plausibly take from a similar state, but happens to differ from the factual action. CRA under this setting tests whether the model captures the fine-grained action-conditional distinctions that matter for planning, not just gross action category differences.

### 4.4. State Regime Stratification

The metrics above are reported globally and per-regime under a four-way stratification of evaluation transitions. The regimes are designed to span the action-effect spectrum that arises during manipulation:

**Regime 1: Free-space motion.** The end-effector is not in contact with any object, and the action commands translation through empty space. Action effects are large in joint space but small in visual scene space (only the arm moves; the scene is static).

**Regime 2: Pre-grasp approach.** The end-effector is approaching an object but has not made contact. Small action variations can determine whether the upcoming grasp will succeed (e.g., approach trajectory affecting final gripper alignment).

**Regime 3: Gripper actuation.** The action involves opening or closing the gripper. Visual effects are small (only finger position changes), but the consequence for downstream task success is large.

**Regime 4: Contact + manipulation.** The end-effector is in contact with an object, and the action induces object motion. Action effects propagate to the scene through contact dynamics.

The stratification is performed automatically using proprioceptive signals (gripper state, contact sensors if available) or visual heuristics (optical flow magnitude in object regions, hand-object distance estimates).

### 4.5. Expected Diagnostic Pattern

Our hypothesis is that existing JEPA-WMs achieve high CRA and ECS on Regime 1 (free-space, where action effects on visual scene are small but cleanly directional) and Regime 4 (contact + manipulation, where action effects are large and visible), but exhibit substantial degradation on Regimes 2 and 3 (pre-grasp and gripper actuation), where the action effects are small in visual space but consequential for task success. This pattern, if confirmed, would explain why baselines pass qualitative counterfactual visualizations (which often use Regime 4 examples) while still failing on tasks like fine grasping (Regime 3).

---

## 5. Contribution 2: Correlation Study Between Counterfactual Sensitivity and Planning Success

### 5.1. Motivation

It is conceivable that counterfactual ranking accuracy and planning success are weakly correlated — that a model can fail CounterfactualBench but still plan adequately because the planner is robust to prediction errors, or that a model can pass CounterfactualBench but plan poorly due to other failure modes (representation quality, error accumulation in rollout). The connection between counterfactual metrics and planning is currently assumed but not measured.

We empirically establish this connection by training a population of model variants and measuring both their CounterfactualBench scores and their planning success on standardized tasks.

### 5.2. Experimental Design

We construct a population of approximately 20 model variants by systematically varying: (i) the action conditioning architecture (feature concatenation, sequence prepending, AdaLN, AdaLN-Zero), (ii) the training objective (factual prediction only, factual + IDM, factual + multistep, factual + our counterfactual objective), (iii) the visual encoder (DINOv2-S, DINOv2-L, V-JEPA 2), and (iv) the predictor depth (3, 6, 12 transformer blocks). Each variant is trained on the same DROID subset with matched compute budget.

For each variant, we report the four CounterfactualBench metrics across all four regimes (16 measurements per model) and the planning success rate on Franka grasp, pick-and-place, and reach-with-object tasks. We then compute Pearson and Spearman correlations between each metric and each task's success rate.

### 5.3. Hypotheses

**Hypothesis A:** Multi-step CTD at $H \geq 3$ correlates more strongly with pick-and-place success ($r > 0.7$) than one-step CRA ($r \approx 0.4$). Multi-step counterfactual divergence is a better predictor of planning ability than single-step ranking because planning operates over horizons.

**Hypothesis B:** Regime-specific metrics (ECS in Regimes 2 and 3) correlate more strongly with task-specific failure modes than aggregate metrics. For example, ECS on Regime 3 (gripper actuation) is the strongest predictor of grasp success.

**Hypothesis C:** Standard prediction loss on held-out factual transitions has weak correlation ($r < 0.3$) with planning success, validating the need for counterfactual evaluation as a complementary metric.

These hypotheses, if confirmed, provide concrete guidance to the field: practitioners should evaluate JEPA-WMs not by held-out prediction loss but by counterfactual metrics, with attention to the specific regime relevant to their target task.

---

## 6. Contribution 3: CAI-JEPA Training Objective

### 6.1. Overall Loss Formulation

We introduce CAI-JEPA (Counterfactual Action-Identifiable JEPA), a training objective for action-conditioned latent world models that explicitly enforces action-conditional separability. The total loss is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{cf}} \cdot \mathcal{L}_{\text{cf}} + \lambda_{\text{sep}} \cdot \mathcal{L}_{\text{sep}}$$

where $\mathcal{L}_{\text{pred}}$ is the standard factual prediction loss, $\mathcal{L}_{\text{cf}}$ is the counterfactual margin loss enforcing that factual predictions are closer to the true future than counterfactual predictions, and $\mathcal{L}_{\text{sep}}$ is an optional action separation regularizer that pushes predictions under different actions apart in latent space.

### 6.2. Counterfactual Margin Loss

The core of our method is the counterfactual margin loss:

$$\mathcal{L}_{\text{cf}} = \mathbb{E}_{a^- \sim \mathcal{N}(z_t, a_t)} \left[ \max\left(0, m - d(F_\theta(z_t, a^-), \text{sg}(z_{t+1})) + d(F_\theta(z_t, a_t), \text{sg}(z_{t+1})) \right) \right]$$

where $\mathcal{N}(z_t, a_t)$ is a state- and action-dependent counterfactual distribution, $m > 0$ is a margin hyperparameter, and $d$ is a distance function (we use L2 in the predictor output space). The loss is zero when the predicted future under the factual action is at least $m$ closer to the true future than the predicted future under the counterfactual action, and penalizes the predictor proportionally when this margin is violated.

The stop-gradient on $z_{t+1}$ is essential. Without it, the counterfactual loss could be minimized by collapsing the target representation, defeating the purpose of action grounding. With stop-gradient, the optimization pressure falls entirely on the predictor's response to action input.

We also consider an InfoNCE variant of the loss:

$$\mathcal{L}_{\text{cf-nce}} = -\log \frac{\exp(-d(F_\theta(z_t, a_t), z_{t+1}) / \tau)}{\exp(-d(F_\theta(z_t, a_t), z_{t+1}) / \tau) + \sum_{k=1}^{K} \exp(-d(F_\theta(z_t, a^-_k), z_{t+1}) / \tau)}$$

where $\tau$ is a temperature parameter. The InfoNCE variant naturally extends to multiple negatives per positive and is well-studied in contrastive representation learning. We will ablate both formulations.

### 6.3. Negative Action Sampling

The choice of negative distribution $\mathcal{N}(z_t, a_t)$ is critical. Negatives that are too easy (e.g., uniformly random over the action space) provide weak training signal because the predictor can trivially distinguish them. Negatives that are too hard (e.g., $a^- = a_t + \epsilon$ for very small $\epsilon$) may push the predictor to overfit to spurious differences. We design three sampling strategies and combine them adaptively:

**Strategy 1: Opposite-action negatives.** $a^- = -a_t + \mathcal{N}(0, \sigma^2 I)$. For action spaces centered at zero (e.g., end-effector velocity commands), this samples actions in the opposite direction with small perturbation. Cheap to compute and provides directional contrast.

**Strategy 2: In-distribution random negatives.** $a^-$ is sampled uniformly from the empirical action distribution of the dataset. This ensures negatives remain on the action manifold the predictor has seen during training, avoiding out-of-distribution extrapolation issues.

**Strategy 3: Hard nearest-neighbor negatives.** For each $(z_t, a_t)$, find $K$ other transitions $(z_{t'}, a_{t'})$ in the batch with similar initial states ($\| z_t - z_{t'} \| < \rho$) but different actions, and use $a_{t'}$ as the negative. This is the hardest setting because the negative is a plausible action from a similar state. We use these negatives sparingly (e.g., one hard negative per positive) because they are computationally expensive and can destabilize training.

In practice, we form a negative batch combining all three strategies with adaptive weights: more opposite-action negatives early in training (to establish coarse directional grounding), then transitioning toward hard NN negatives later (to refine fine action distinctions).

### 6.4. Effect-Conditional Gating

A naive application of $\mathcal{L}_{\text{cf}}$ to all transitions has a pathology: in free-space motion regimes, opposite or random counterfactual actions may produce futures that are genuinely similar to the factual future in latent space (e.g., the arm moves slightly differently but the scene is essentially identical). Forcing the predictor to separate these futures introduces noise rather than useful signal.

We address this with effect-conditional gating. The counterfactual loss is weighted by a sigmoid function of the factual transition's effect magnitude:

$$w(z_t, z_{t+1}) = \sigma\left( \alpha \cdot (\|z_{t+1} - z_t\| - \tau_{\text{gate}}) \right)$$

$$\mathcal{L}_{\text{cf-gated}} = w(z_t, z_{t+1}) \cdot \mathcal{L}_{\text{cf}}$$

where $\alpha$ controls the sigmoid sharpness and $\tau_{\text{gate}}$ is calibrated from data so that approximately half of transitions receive substantial weight. This focuses the counterfactual training signal on transitions where action grounding actually matters.

For manipulation domains with available contact information, we additionally incorporate a contact-aware multiplier: $w$ is boosted by a factor (e.g., 2x) on transitions where contact state changes between $t$ and $t+1$, reflecting the fact that contact transitions are precisely where action grounding is most consequential and most fragile.

### 6.5. Action Separation Regularizer

The counterfactual margin loss enforces that factual predictions are closer to the true future than counterfactual predictions, but does not directly enforce that counterfactual predictions are far from each other. In principle, a degenerate solution exists where $F_\theta(z_t, a) = \hat{z}_{t+1}^{\text{factual}}$ for the factual $a$ and $F_\theta(z_t, a^-) = \hat{z}_{t+1}^{\text{factual}} - \delta \cdot m / \|\delta\|$ for any counterfactual $a^-$, where $\delta$ is some fixed direction. This satisfies the margin loss but produces identical predictions for all counterfactual actions, which is not desired.

We introduce an optional separation regularizer:

$$\mathcal{L}_{\text{sep}} = -\log\left( \frac{d(F_\theta(z_t, a_t), F_\theta(z_t, a^-))}{d_{\max}} \right)$$

This encourages diversity among predictions under different actions, complementing the margin loss. We ablate whether this term is necessary in practice; it may be redundant in the presence of sufficiently diverse negative sampling.

### 6.6. Anti-Collapse Considerations

JEPA architectures are known to be susceptible to representation collapse, in which the encoder learns to produce trivially predictable representations. Standard mitigations include EMA target encoders, VICReg-style variance/covariance regularization, and SIGReg. Our counterfactual losses do not directly address collapse — in fact, they could in principle interact with collapse-mitigating mechanisms.

Specifically, $\mathcal{L}_{\text{cf}}$ pushes the predictor's output to vary with action input, but does not prevent the encoder from producing representations where this variation is concentrated in a low-dimensional subspace while the bulk of representation collapses. To guard against this, we keep the standard collapse mitigation (we use frozen pretrained encoders, which sidestep the issue entirely for the encoder, and EMA target for the predictor output's target).

We also monitor representation rank and effective dimensionality throughout training as a diagnostic. If we observe that the predictor's output rank decreases sharply when $\mathcal{L}_{\text{cf}}$ is added, we will additionally apply VICReg-style regularization to predictor outputs.

### 6.7. Multi-step Counterfactual Training

The losses above are defined for one-step prediction. Extending them to multi-step rollout is conceptually straightforward but computationally expensive: at each step of the rollout, $K$ counterfactual actions must be sampled and rolled forward. For horizon $H$ with $K$ negatives, the total number of counterfactual rollouts is $K \cdot H$ per training sample.

We adopt a pragmatic compromise. The primary $\mathcal{L}_{\text{cf}}$ is applied only at one-step prediction (the highest-leverage location for action grounding). For multi-step rollout, we apply the standard multistep rollout loss without counterfactuals, relying on the one-step counterfactual loss to instill action grounding that propagates through rollout. We verify this propagation empirically via the CTD metric in CounterfactualBench: if one-step counterfactual training induces multi-step counterfactual divergence (rising CTD with $H$), the design choice is justified.

### 6.8. Compatibility with Existing Architectures

CAI-JEPA is a loss-level intervention and is architecturally agnostic. It composes naturally with any existing JEPA-WM architecture: DINO-WM, V-JEPA 2-AC, Terver-WM, or future variants. We will demonstrate that adding $\mathcal{L}_{\text{cf}}$ to each of these architectures yields consistent improvements on CounterfactualBench, supporting the claim that the contribution is orthogonal to architectural choices.

---

## 7. Contribution 4 (Optional): Action Information Flow Probing

### 7.1. Motivation

Beyond the methodological contributions above, a deeper question remains: **where in the predictor does action information reside, and how does it flow through the network?** Existing analyses are largely behavioral, examining model outputs without inspecting internal representations.

We propose to probe action information flow through the predictor using lightweight diagnostic classifiers attached to each transformer block. This analysis serves both a scientific purpose (understanding how JEPA-WMs internally represent actions) and a methodological purpose (locating where existing models fail and how our intervention changes the flow).

### 7.2. Probe Design

For each transformer block $\ell$ in the predictor, we train three lightweight MLP probes (2 layers, ~10K parameters) on the block's output activations $h_\ell$, with the predictor frozen:

**Action recovery probe:** $h_\ell \rightarrow \hat{a}_t$. Measures how much of the input action is linearly decodable from the block's intermediate representation. A high score at layer $\ell$ indicates that action information is still present at depth $\ell$.

**Action effect probe:** $h_\ell \rightarrow \widehat{\|z_{t+1} - z_t\|}$. Measures whether the block's representation has integrated action information with state context to predict the magnitude of the upcoming state change.

**Contact probe (manipulation tasks):** $h_\ell \rightarrow \widehat{\text{contact}_{t+1}}$. Measures whether the block represents the task-relevant prediction (will contact be made next step).

### 7.3. Hypotheses

**Hypothesis 1:** In baseline DINO-WM (feature-concatenation action conditioning), action recovery probe accuracy peaks at layers 1-2 and decays sharply toward later layers — action information "vanishes" through depth.

**Hypothesis 2:** In Terver-WM (AdaLN conditioning), action recovery probe accuracy remains high across all layers, consistent with the architectural design of re-injecting action at every block.

**Hypothesis 3:** In CAI-JEPA, the action effect probe peaks at intermediate layers (where state context and action information have been integrated), and this peak is higher than in baselines, indicating that action grounding has been pushed deeper than mere action presence.

These probing results would provide mechanistic understanding of where current JEPA-WMs succeed and fail, and how our objective changes the internal computation.

---

## 8. Experimental Design

### 8.1. Datasets and Tasks

**Primary dataset:** DROID (Khazatsky et al., 2024), a large-scale Franka manipulation dataset with diverse environments, objects, and tasks. We use the same subset as V-JEPA 2-AC (approximately 62 hours of interaction data) for direct comparability.

**Primary evaluation tasks:** Franka grasp (cup and box), pick-and-place (cup and box), and reach-with-object, following the V-JEPA 2-AC evaluation protocol. We additionally evaluate on the harder cluttered-scene variants if available.

**Secondary evaluation tasks:** LIBERO subset (LIBERO-Goal and LIBERO-Long), which contains multimodal demonstrations — multiple valid action sequences from similar initial states — providing natural test conditions for action grounding.

**Sanity check tasks:** Push-T (Zhou et al., 2024) and PointMaze. These are saturated benchmarks; we report them only to verify that our method does not regress on tasks where baselines already perform well.

### 8.2. Baselines

We compare against four baselines representing the state of the art:

1. **DINO-WM** (Zhou et al., 2024): DINOv2 encoder + causal transformer predictor with feature concatenation action conditioning.

2. **V-JEPA 2-AC** (Assran et al., 2025): V-JEPA 2 encoder + 300M transformer predictor with sequence-prepending action conditioning.

3. **Terver-WM** (Terver et al., 2026): The optimized configuration from "What Drives Success in Physical Planning" — DINOv3 encoder + AdaLN-conditioned transformer predictor + multistep rollout loss.

4. **DINO-WM + IDM auxiliary loss:** DINO-WM trained with an additional inverse dynamics auxiliary loss head, representing the strongest competing approach to action grounding via auxiliary supervision.

### 8.3. Ablation Plan

To isolate the contributions of each component of CAI-JEPA, we run the following ablations:

**A1: Removing counterfactual loss entirely** (recovers the underlying baseline). Establishes that counterfactual loss is the source of improvements.

**A2: Removing effect-conditional gating** (uniform weighting on all transitions). Tests whether the gating is necessary or whether undifferentiated counterfactual loss suffices.

**A3: Removing separation regularizer.** Tests whether $\mathcal{L}_{\text{sep}}$ provides marginal benefit over $\mathcal{L}_{\text{cf}}$ alone.

**A4: Varying negative sampling strategy** (random only, opposite only, hard NN only, all combined). Identifies which negative type contributes most.

**A5: Varying number of negatives** $K \in \{1, 4, 8, 16, 32\}$. Identifies the cost-benefit tradeoff of negative sample count.

**A6: Margin loss versus InfoNCE.** Compares the two formulations.

**A7: Applying CAI-JEPA on different base architectures** (DINO-WM base, V-JEPA 2-AC base, Terver-WM base). Tests architectural orthogonality.

### 8.4. Evaluation Protocol

Each trained model is evaluated along three axes:

**CounterfactualBench scores:** All four metrics across all four regimes, averaged over 5 random seeds for the held-out evaluation set.

**Planning success rate:** Standard goal-conditioned planning evaluation on Franka manipulation tasks, averaged over 10 trials per task with varied initial conditions, replicated across 3 random seeds for the planner.

**Correlation analysis:** All trained models contribute to the correlation study between CounterfactualBench metrics and planning success.

---

## 9. Scope and Limitations

### 9.1. What We Claim

We claim that (a) existing JEPA-WMs exhibit measurable action grounding failures in contact-rich and fine-precision regimes, even when they pass standard evaluations and qualitative counterfactual visualizations; (b) these failures correlate with planning failures on relevant tasks; and (c) an explicit counterfactual training objective with effect-conditional gating reliably improves both diagnostic metrics and planning success on these tasks.

### 9.2. What We Do Not Claim

We do not claim that our objective makes JEPA-WMs solve planning in general, or that it addresses other failure modes such as long-horizon error accumulation, distribution shift between training and test environments, or representation quality issues. We do not claim that counterfactual contrastive training is the unique or best way to enforce action grounding — alternative approaches (architectural innovations, IDM auxiliary losses, multistep training) are also valuable and may be complementary.

We do not claim novelty of the broad idea of using counterfactual signals in world models. Causal-JEPA, WAV, CoDA, and others have explored related ideas. Our specific contribution is the formulation of action-level counterfactual contrastive training with effect-conditional gating for JEPA-WM action grounding, combined with a systematic diagnostic protocol that did not previously exist.

### 9.3. Known Limitations

**Limitation 1: Counterfactual actions are inherently out-of-distribution for the predictor.** The predictor is trained on factual transitions where actions are sampled from the data distribution. Asking the predictor to predict under counterfactual actions queries it outside this distribution. Our method partially mitigates this by sampling counterfactuals from the empirical action distribution (Strategy 2) rather than uniformly, but the issue cannot be eliminated entirely.

**Limitation 2: Effect-conditional gating depends on heuristics.** The gating function $w(z_t, z_{t+1})$ depends on encoder-derived effect magnitude, which is an imperfect proxy for true action consequence. In settings where the encoder fails to capture relevant scene changes (e.g., subtle contact state changes invisible to the visual encoder), the gating may misweight transitions.

**Limitation 3: Single-step counterfactual training may not propagate to long horizons.** Our hypothesis that one-step counterfactual training induces multi-step counterfactual divergence is empirical and may not hold for very long horizons. Models intended for long-horizon planning may require multi-step counterfactual training, with its associated computational cost.

**Limitation 4: Improvements may be regime-specific.** We expect our method to improve action grounding most strongly in regimes where baselines fail, and to have negligible effect on regimes where baselines already succeed. Aggregate metrics may therefore show modest improvements that mask large regime-specific gains.

---

## 10. Summary

This proposal addresses a specific methodological gap in the JEPA world model literature: the absence of systematic diagnostics and explicit training objectives for action grounding in latent dynamics predictors. We argue that existing evaluations — held-out prediction loss, goal-conditioned planning success rate, qualitative counterfactual visualization — fail to detect quantitative action grounding failures in regimes where these failures most consequentially affect planning (contact-rich, fine-precision, ambiguous-effect transitions).

We propose four contributions. **CounterfactualBench** is a systematic diagnostic protocol with four metrics (Counterfactual Ranking Accuracy, Action Usage Gap, Counterfactual Trajectory Divergence, Effect-Conditional Sensitivity) evaluated under stratified state regimes (free-space, pre-grasp, gripper actuation, contact + manipulation) with three negative action sampling strategies (random, opposite, hard nearest-neighbor). A **correlation study** between CounterfactualBench metrics and planning success establishes which metrics best predict downstream task performance. **CAI-JEPA**, our training objective, combines a counterfactual margin loss with effect-conditional gating and an optional action separation regularizer, designed to be architecturally orthogonal and composable with existing JEPA-WMs. Finally, an optional **probing analysis** of action information flow through the predictor provides mechanistic understanding of where and how our objective changes internal computation.

The overarching thesis is that **action-identifiability** — the property that a predictor's outputs are reliably distinguishable conditional on the action input — is a necessary condition for latent world models used in robot planning, and deserves to be diagnosed, measured, and explicitly optimized as a first-class design objective alongside prediction quality and architectural choices.
