# Paper idea — Boundary-Blind JEPA World Models (contact-boundary reframing)

**Status:** 2026-06-10. Current idea-of-record. Supersedes the *fix* framing of
`cai_jepa_paper_proposal.md` §6 (the one-step counterfactual margin loss); the
diagnostic framing (§4–§5 of the proposal) stands and is extended here with the
**Boundary Blindness** metric. **The BB gate has been RUN and PASSED**
(2026-06-10, frozen baselines, Metaworld + DROID transfer): BB concentrates at the
pre-grasp boundary, CI-aware, on both datasets — see §3/C1 below and
`results/boundary_gate_report.md`. **The fix leg is now complete on Metaworld**
(same day): head-level mixture C3 → quantified null; metric-only re-weighting →
null (both kept as ablations); **the grounded object-dynamics channel
`h(z,a)→Δobject` works** — counterfactual tracking corr +0.035 → **+0.682**, BB
at the pre-grasp boundary **−50%** (1.323 → 0.660), the pre_grasp-vs-free_space
BB gap collapses 1.04 → 0.32. See the revised C3 below and
`docs/FIX_C1_EXPLAINER.md` §6–§7. Planning A/B (CEM with the grounded cost) in
progress.

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
  between the world's local action-sensitivity and the model's. **Measured result
  (2026-06-10, frozen baselines — gate PASSED):** BB concentrates at the
  **pre-grasp boundary** exactly as predicted. Metaworld (object-Δ outcome, pooled
  n_b-weighted, excl. the `mw-door-close` proxy anomaly): `bb_boundary` pre_grasp
  **1.323 / 1.280** (dino_wm / jepa_wm) vs free_space 0.282 / 0.299 (≈4.5×), contact
  0.481 / 0.441; per-task CI-aware pairing: pre_grasp confidently elevated in 4/6
  (dino) and 5/6 (jepa) tasks, zero confident reversals. DROID transfer (‖Δz‖
  proxy, dino_wm): pre_grasp **1.975 [1.601, 2.350]** vs free_space 0.721
  [0.613, 0.834] — CI-confident. Source: `results/metaworld_boundary.csv`,
  `results/droid_boundary.csv`; analysis in `results/boundary_gate_report.md`.
  (Implemented: `metrics/boundary_blindness.py`, `stratification/boundary_regime.py`,
  `scripts/12_boundary_diagnostic.py`.)

**C2 — Counterfactual sensitivity ⇄ planning failure (correlation study).**
A faithful CEM planner (paper's DROID Action Error) correlated with per-transition
action-grounding. The one-step CRA_eff correlation is null/underpowered (severe
class imbalance: ~4% positives) — *which is itself evidence for the reframing*:
the relevant failure is the boundary bifurcation, not one-step action-ignoring. BB,
not CRA_eff, is the right planning-linked signal; the planning leg is recast around
it.

**C3 — The fix. MEASURED UPDATE (2026-06-10): head-level C1 (incl. C1+D at the
head) is a quantified structural null; the fix moves down the stack.**
- What was run (all on frozen trunks, cached latents; `docs/FIX_C1_EXPLAINER.md` §6):
  mixture-density head with soft-NLL, WTA/hard-EM, head-level D (state-slice
  context), and **supervised mode assignment** (K=2 components labeled by the
  object-moves event). Every variant beat its K=1 control on NLL; **none moved BB**
  (`results/metaworld_boundary_fix.csv`).
- Two measured causes: **(a)** the supervised variant's conditional "moves"/"doesn't"
  future-means differ by only **9.9 L2 vs a 106 median residual** — the bifurcation
  is ~9% of the residual in the latent's L2 geometry (the planning metric itself
  underweights the boundary subspace); **(b)** the boundary event's
  *action*-dependence is unlearnable from expert-only data (π calibrates to the
  24.9% base rate but action-flip rate ≈ 0; CE 0.487 vs 0.562 baseline) — experts
  almost never execute the failing counterfactual.
- **Both follow-up directions were then run to ground (2026-06-10 evening,
  `docs/FIX_C1_EXPLAINER.md` §7):**
  - (i) metric-level re-weighting (probe `g(z)→obj` + φ-metric): **null** —
    redistributes BB without fixing it; you cannot expose a signal the predictor
    does not produce (`results/metaworld_boundary_metric.csv`). Kept as the
    ablation killing the "just fix the metric" alternative.
  - The probe chain localized the true bottleneck: V1 ✓ object decodable from the
    latent (err 0.064 vs sd 0.094); V2 ✓ propagated for factual actions (0.059);
    V3 ✗ counterfactual object response is noise (spread corr +0.035).
  - **(iii) — the fix that works: the grounded object-dynamics channel**
    `h(z_t, a) → Δobject` (0.5M params, frozen everything, cache-only).
    Counterfactual tracking corr **+0.682** (20× the frozen predictor); **BB at
    pre_grasp −50%** (1.323 → 0.660 pooled, excl. the door-close proxy anomaly);
    the pre_grasp-vs-free_space gap collapses 1.04 → 0.32
    (`results/metaworld_boundary_dynamics.csv`). Notably this revises the earlier
    "no counterfactual data" reading: cross-sample neighbourhood variation IS
    sufficient — the bottleneck was the **training target** (full-latent L2), not
    the data.
- Planning integration, measured: the grounded CEM cost is **no-harm / no-gain on
  open-loop Action Error** (paired Δ(hdyn−l2): pre_grasp −0.15 [−1.07, +0.79] n=6,
  contact −0.07 [−0.51, +0.34] n=21 — `results/metaworld_planning_metric.csv`;
  a first run with a per-dim-MSE scale bug is preserved as `_buggy_scale.csv`).
  Expected: Action Error rewards full-arm expert mimicry, which the L2 term
  already optimises; the boundary fix matters closed-loop. **Closed-loop success
  rate with the grounded cost is the declared next experiment** (needs env
  rollouts — server-side).

## 4. What makes it publishable (claims, falsifiable)

- **Novel, sharp claim:** "unimodal latent prediction is structurally incapable of
  representing contact bifurcations; distributional, boundary-supervised latent
  prediction restores it." Clean and falsifiable.
- **Falsification gates built in:** the first gate — BB elevated in contact-rich
  regimes on frozen baselines — **passed** (2026-06-10; pre-grasp locus, CI-aware,
  both datasets). The second gate — C1 drops BB in boundary regimes — **fired
  negative the same day** for every head-level variant; per the gate's design this
  falsifies the *head-level* form of the fix, and the measured causes (C3 above)
  redirect it to the latent/metric/data level. The claim sharpens rather than dies:
  "boundary blindness" includes the latent metric, not just the predictor.
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

1. **BB-per-regime** bar chart (baselines): **now real** —
   `results/figures/figure_bb_per_regime.pdf` (Metaworld both baselines + DROID
   transfer; data from `results/{metaworld,droid}_boundary.csv`). BB high at the
   pre-grasp boundary, low in free-space — the gap, measured. Caveats carried on
   the figure: Metaworld object-Δ proxy (mw-door-close excluded), DROID ‖Δz‖
   proxy/transfer-only.
2. **BB before/after the fix ladder** — now fully real: frozen base 1.323 →
   mixture heads 1.32 (null) → φ-metric 1.17 (null) → **grounded dynamics
   channel 0.660** at pre_grasp (pooled bb_boundary, excl. door-close; CSVs:
   `metaworld_boundary{,_fix,_metric,_dynamics}.csv`). Support panel: the
   V1/V2/V3 localization chain + counterfactual-tracking corr 0.035 → 0.682.
3. **Planning Action Error vs. BB**: the link that CRA_eff could not establish.
4. Ablation: C1-only / D-only / C1+D, mixture-K, boundary-head on/off.

## 7. Working titles

- "Boundary-Blind World Models: Why JEPA Predictors Smear Contact Bifurcations, and
  How to Fix Them"
- "Action Identifiability Is Not Enough: Resolving Contact Boundaries in JEPA World
  Models for Planning"
