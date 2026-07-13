# Novel-methods survey for the action-grounding gap (2026-07-05)

Literature sweep of 2025–2026 work relevant to our gap: JEPA world models are boundary-blind at
contact; frozen post-hoc costs and encoder-LoRA are proven nulls; CEM reward-hacks any post-hoc
readout; Phase-H counterfactual predictor objective is our one positive (DROID Action-Score
0.432→0.580). This doc records (1) concurrent-work threats, (2) candidate method families for the
contact wall, ranked by fit, (3) a recommended repositioning of the paper.

## 1. Concurrent work — threats and anchors

### UWM-JEPA (arXiv 2605.25313, May 2026) — SAME core insight, toy setting
States our Phase-H thesis almost verbatim: *"Under a teacher-forced JEPA objective the target
encoder observes the trajectory that already contains the action's effect, which admits an
action-invariant solution in which the predictor matches the target without using the explicit
action channel"* — and shows *counterfactual targets* restore action sensitivity. **But**: it is a
physics/toy study (density-matrix latents, unitary predictor, hidden-velocity indicator task, blind
rollout probes). No pretrained robot world models, no planning, no manipulation.
- **Threat level: medium.** The *idea* of counterfactual targets fixing teacher-forced JEPA is now
  published. We can no longer claim to have discovered the mechanism.
- **Anchor value: high.** Independent confirmation of the mechanism. Our contribution becomes the
  *first demonstration at scale on real pretrained robot world models* (DINO-WM/DROID, 22M–1B),
  with planning-level consequences (CEM reward-hacking) they never touch. Cite it, don't fear it.

### ATM: Action-Consistency Transfer Matrix (arXiv 2606.09028, June 2026) — concurrent diagnostic
Post-hoc analysis on a frozen encoder + dynamics predictor: builds "real encoded transition" vs
"model-predicted transition" domains, a transfer matrix diagnosing whether transitions preserve
*action-identifiable semantics*, a screening score across checkpoints, plus an
"Action-Identifiable Transition Supervision" training objective.
- **Threat level: high for the diagnostic half.** This is a CRA/boundary-blindness sibling. Must be
  cited as concurrent work and differentiated. Our differentiators: contact-regime stratification,
  the 22M→1B scaling null, the oracle ladder (localizes the wall to the cost, not the predictor),
  and the planner-level reward-hacking evidence (mined-elite decode 91.5%→24%). ATM has none of
  those, as far as the abstract shows. Read the full paper on the server (arxiv blocked from this
  sandbox).

### Other latent-action-grounding papers (same problem, different lever)
- **DiLA: Disentangled Latent Action World Models** (2605.15725), **CLAW** (2606.04130, adversarial
  latent regularization), **Olaf-World** (2602.10104, orienting latent actions) — all attack action
  controllability in *latent-action* (action-free) WMs. Adjacent, cite in related work.
- **ACT-Bench**-style action-following benchmarks exist for video WMs; our effect-conditioned CRA is
  the robot-manipulation analogue.

## 2. Candidate methods for the contact wall, ranked by fit

### (a) Amortized planning: replace CEM with a goal-conditioned inverse-dynamics controller — BEST FIT
**"Latent Geometry Beyond Search: Amortizing Planning in World Models"** (arXiv 2605.08732):
replaces CEM with a lightweight GC-IDM mapping (z_t, z_goal, horizon) → action; matches or beats
CEM in 7/8 settings at 100–130× lower cost. Their framing — *much of what test-time search recovers
is already locally encoded in the latent* — plus IMWM's observation that *even with a perfect world
model a finite-budget sample-based planner still fails* is exactly our oracle-ladder story.

**Why it fits us:** our entire reward-hacking chain (Phase 3b/4/G) shows the *adversary is the
search*. Every fix so far hardened the cost and left the adversary in place. Amortized inference
removes the adversary instead. And we already have `inverse_proposal_dino_wm_metaworld.pt` +
Phase-H CF-predictor — the novel pairing **CF-grounded predictor + amortized inverse-dynamics
control** appears in none of these papers.
- Experiment: GC-IDM (or our inverse proposal head, goal-conditioned, horizon-aware) driving
  MetaWorld push with the frozen and CF-fine-tuned predictors. Success = first contact-wall
  crossing; failure = strengthens the "representation is insufficient" claim with a search-free
  control.

### (b) Hybrid cost + reliability gate (IMWM-style) — second-best, cheap
**IMWM** (arXiv 2606.01626): frozen "intuition model" = inverse-side encoder + bilinear scorer over
(start, goal, action) trained with InfoNCE on demos; planner cost = z-scored blend of intuition
score and rollout error; reliability gate modulates trust; retrieval initializes CEM.
- Direct counter to CEM exploitation: the intuition score is an *action-space* prior the planner
  cannot hack by finding weird latents, and retrieval-init keeps CEM near the data manifold.
- We have most pieces (inverse proposal ckpt, InfoNCE machinery from scripts/40). Cheap ablation:
  CF-cost + λ·intuition-score + demo-retrieval init on push-held.

### (c) Observability: vision-only latents may lack contact state — analysis, not pivot
**ContactWorld** (arXiv 2606.13877) benchmark finding: tactile/force helps *most* in contact-rich,
long-horizon settings; force-field + point-cloud reps lift planning 20.7%→36.1%. Plus a wave of
visuo-tactile WMs (VT-WAM 2607.02503, Dream-Tac 2606.08737, FAWAM 2606.08555, TacForeSight
2606.11184, OmniVTA 2603.19201).
- Supports a *fundamental-limit* reading of our null: object pose under occlusion/contact may be
  weakly observable in DINOv2 patch tokens, so no objective on the same inputs crosses the wall.
- For this paper: a discussion-section argument + (optional) an oracle experiment adding GT
  contact/force channel to the cost to upper-bound what any vision-only method could do. A full
  tactile pivot is a different paper.

### (d) Plannable-cost geometry / decision-aware latents
- **"Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models"**
  (2605.06388) and **"Latent State Design under Sufficiency Constraints"** (2605.01694): what
  latent structure makes L2-in-latent a valid planning cost — directly our "the cost is the wall"
  finding from the oracle ladder.
- **Value-equivalent / decision-aware WMs** (SPOWL 2506.04828; "Value-guided action planning with
  JEPA world models", 2601.00844, World-Modeling Workshop 2026): learn the latent/value so that
  planning objective = decision objective, the classic objective-mismatch fix (Lambert et al.).
  2601.00844 trains distance-as-negative-value on JEPA latents — closest existing attempt; it's a
  workshop paper, so the arena is open.

### (e) Belief-space / distributional latents
UWM-JEPA's other half: point-estimate latents dissipate uncertainty under blind rollout; contact
dynamics are exactly where multimodality lives, and an L2 cost on a mean prediction is biased
there. A cheap probe: measure predicted-latent variance collapse across our contact regimes —
would give a *mechanistic* explanation for why contact is the wall. Full distributional predictor =
future work section.

## 3. Recommended paper repositioning

Story: **"Grounding is necessary but not sufficient: diagnosing and repairing action grounding in
pretrained JEPA world models."**
1. Diagnostic + scaling null (22M→1B) with contact stratification — ours alone; position against
   ATM (concurrent) and UWM-JEPA (mechanism confirmed in toy setting).
2. Oracle ladder + reward-hacking evidence: the failure is the *cost/search interface*, not missing
   information in the predictor — no other paper has the mined-elite decode analysis.
3. Constructive fix on the predictor axis: CF objective (Phase H), hardened (seeds, 2nd model,
   RoboCasa) and compared against published DROID Action-Score numbers.
4. NEW experiment to close the loop: amortized GC-IDM control (a) ± hybrid cost (b) on push-held.
   Either outcome is a headline: crossing = first positive on contact; null = completes the proof
   that vision-only latent geometry, not optimization, is the wall (supported by (c)/(e) analyses).

Priority order: E1 = (a) amortized controller on push; E2 = Phase-H hardening (already planned);
E3 = (b) hybrid-cost ablation; E4 = (e) variance-collapse probe for the mechanism section.

## Sources
- UWM-JEPA: https://arxiv.org/abs/2605.25313
- ATM: https://arxiv.org/pdf/2606.09028
- Latent Geometry Beyond Search: https://arxiv.org/pdf/2605.08732
- IMWM: https://arxiv.org/abs/2606.01626
- ContactWorld: https://arxiv.org/abs/2606.13877
- Reconstruction or Semantics?: https://arxiv.org/pdf/2605.06388
- Latent State Design under Sufficiency Constraints: https://arxiv.org/pdf/2605.01694
- Value-guided action planning with JEPA WMs: https://arxiv.org/abs/2601.00844
- SPOWL: https://arxiv.org/html/2506.04828
- DiLA: https://arxiv.org/pdf/2605.15725 ; CLAW: https://arxiv.org/pdf/2606.04130 ;
  Olaf-World: https://arxiv.org/pdf/2602.10104
- Visuo-tactile WMs: VT-WAM https://arxiv.org/html/2607.02503 ; Dream-Tac
  https://arxiv.org/html/2606.08737 ; FAWAM https://arxiv.org/pdf/2606.08555 ; TacForeSight
  https://arxiv.org/abs/2606.11184 ; OmniVTA https://arxiv.org/html/2603.19201
- HWM hierarchical planning: https://arxiv.org/abs/2604.03208 ; PRISM: https://arxiv.org/pdf/2606.07974
- V-JEPA 2 / 2-AC baseline: https://arxiv.org/abs/2506.09985
