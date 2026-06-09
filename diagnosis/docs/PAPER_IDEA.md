# Paper idea — Boundary-Blind JEPA World Models (contact-boundary reframing)

**Status:** 2026-06-09. Current idea-of-record. Supersedes the *fix* framing of
`cai_jepa_paper_proposal.md` §6 (the one-step counterfactual margin loss); the
diagnostic framing (§4–§5 of the proposal) stands and is extended here with the
**Boundary Blindness** metric.

---

## 1. One-paragraph thesis

Action-conditioned JEPA world models (DINO-WM, V-JEPA-2-AC, JEPA-WM) are not
merely weak at amplifying action effects — they **fail to model high-sensitivity
action regimes, where small action perturbations near a contact boundary produce
qualitatively different futures** (gripper centred → object lifts; 2–3° off → no
lift). Unimodal latent prediction (a point predictor trained with L2) *provably
averages* across such outcome bifurcations, and a vision-only latent may not even
resolve the boundary-relevant state. We (1) give a diagnostic that **isolates and
measures** this failure on frozen baselines, (2) tie it to **planning failure**,
and (3) fix it with a **distributional, boundary-supervised** latent predictor
plus an optional state-grounded latent.

## 2. Why the obvious story is wrong (the critique that motivates the paper)

The intuitive fix — push `F(z,a)` and `F(z,a')` apart (classifier-free action
guidance, or a one-step counterfactual margin loss) — **cannot work in the regime
that matters**. If the model has already collapsed `F(z,a) ≈ F(z,a')`, the guidance
vector is ~0 and the margin loss has ∂/∂a ≈ 0: there is nothing to amplify. The
boundary is sharp in *outcome* space, but the one-step `z_{t+1}` after a small
perturbation is near-continuous — the divergence appears **over the rollout**
(sensitive dependence). One-step methods are structurally blind to it. This is the
deeper reason to move to a **distributional, boundary-aware** formulation.

## 3. Contributions

**C1 — CounterfactualBench + Boundary Blindness (diagnostic).**
On frozen, pretrained checkpoints (nothing trained):
- CRA / AUG / ECS measure "does the model use actions at all" (existing; Metaworld
  result: CONDITIONAL_GO — effect-conditioned CRA collapses in contact regimes).
- **Boundary Blindness (BB)** — the new number — measures "does the model *resolve
  the sharp boundary*", which CRA/ECS cannot see. For each boundary-regime
  transition (a similar-state neighbourhood where a small action change fans the
  true outcome out), `BB = relu(S_true_norm − S_model_norm)`: the rectified gap
  between the world's local action-sensitivity and the model's. **Predicted result:
  BB concentrates in pre-grasp / gripper-actuation boundary transitions even where
  aggregate CRA looks acceptable.** (Implemented: `metrics/boundary_blindness.py`,
  `stratification/boundary_regime.py`, `scripts/12_boundary_diagnostic.py`.)

**C2 — Counterfactual sensitivity ⇄ planning failure (correlation study).**
A faithful CEM planner (paper's DROID Action Error) correlated with per-transition
action-grounding. The one-step CRA_eff correlation is null/underpowered (severe
class imbalance: ~4% positives) — *which is itself evidence for the reframing*:
the relevant failure is the boundary bifurcation, not one-step action-ignoring. BB,
not CRA_eff, is the right planning-linked signal; the planning leg is recast around
it.

**C3 — The fix: distributional + boundary-supervised latent prediction.**
- **C1-fix (hero, force-free):** replace the point head with a **mixture-density
  head** over `z_{t+1}` (K≈2–4, NLL loss) so the predictor can represent "lift OR
  not-lift" instead of their L2 mean; add a **boundary-supervision head** (grasp /
  object-moves event from Metaworld state) that forces capacity onto the boundary.
- **D (supporting, Metaworld only):** augment the latent with the boundary-relevant
  state slice `z̃ = [z^vis ‖ φ(ee–object geometry, gripper)]` so the latent can
  *resolve* the boundary. Best result expected from C1+D.
- Planning integration: CEM scores candidates by mixture mode / NLL-of-goal, so it
  no longer optimises against an averaged-out, action-insensitive cost surface.

## 4. What makes it publishable (claims, falsifiable)

- **Novel, sharp claim:** "unimodal latent prediction is structurally incapable of
  representing contact bifurcations; distributional, boundary-supervised latent
  prediction restores it." Clean and falsifiable.
- **Falsification gates built in:** if BB is *not* elevated in contact-rich regimes
  on frozen baselines, the reframing is wrong (we stop). If C1 does not drop BB in
  boundary regimes and improve planning vs. the one-step-margin baseline, the fix is
  wrong.
- **Force-free hero:** C1 runs on cached latents and is unaffected by the DROID data
  limitation (§5) — only a small predictor head trains.

## 5. Scope & honesty (pre-empt reviewer objections)

- **Metaworld is the boundary *proof*** — the only dataset with object state, so the
  only place we can *define*, *supervise*, and *measure* the boundary against ground
  truth. The boundary label is an **object-displacement proxy** (HF dataset has no
  MuJoCo contact GT) — stated explicitly; it is nonetheless a clean *outcome* label.
- **DROID is a transfer check, not the proof.** Verified at the loader: DROID
  proprio is 7-dim pose+gripper with **no force/torque, no joint** channel. So
  direction D's force-grounded form is impossible there and BB uses the weaker
  ‖Δz‖ proxy. We do **not** claim tactile/force grounding on DROID.
- Push-T / PointMaze are sanity checks only — never thesis evidence.

## 6. Headline figures (target)

1. **BB-per-regime** bar chart (baselines): BB high in pre-grasp / gripper /
   contact, low in free-space — the gap.
2. **BB before/after C1** in boundary regimes: sharp drop for the mixture head.
3. **Planning Action Error vs. BB**: the link that CRA_eff could not establish.
4. Ablation: C1-only / D-only / C1+D, mixture-K, boundary-head on/off.

## 7. Working titles

- "Boundary-Blind World Models: Why JEPA Predictors Smear Contact Bifurcations, and
  How to Fix Them"
- "Action Identifiability Is Not Enough: Resolving Contact Boundaries in JEPA World
  Models for Planning"
