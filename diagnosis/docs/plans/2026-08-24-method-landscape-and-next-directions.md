# Method landscape + next directions (2026-08-24)

> **SUPERSEDED 2026-09-08.** The ICLR 2027 target and its Sep 2026 schedule
> (abstract 09-19 / paper 09-24 / experiment freeze 09-05) were **abandoned**.
> Do not plan against any date in this file. Current target: a new direction
> aimed at ICML 2027 (~Jan) / NeurIPS 2027 (~May). Kept as provenance only.

Literature sweep triggered by: every method axis tried so far (frozen post-hoc cost,
encoder-LoRA, ensemble/disagreement, CF predictor, HyS gating, tactile input,
amortized control) is a null or a scooped positive. This doc records (1) where the
latent-world-model field actually is in Aug 2026, (2) which arena/dataset to compete in,
(3) ranked candidate paper ideas with the baseline number each must beat.

Sources are web-search snippets only (arxiv.org is blocked from the dev sandbox);
re-verify every abstract on the server before committing.

---

## 1. The arena has moved — and it is not MetaWorld

Every 2026 JEPA-planning paper reports on the **stable-worldmodel / LeWM suite**:
Two-Room (2D nav), Reacher, **Push-T** (contact-rich 2D manipulation), **OGBench-Cube**
(3D pick-and-place). `stable-worldmodel` (arXiv 2602.08968 / 2605.21800,
github.com/galilai-group/stable-worldmodel) is the de-facto platform: one interface for
collect → train → MPC-eval.

Published numbers to anchor against:

| model | Push-T | OGBench-Cube | note |
|---|---:|---:|---|
| DINO-WM | 92 | 86 | foundation-encoder baseline, ~50x slower planning |
| LeWM (~15M, end-to-end from pixels, 2 loss terms) | 94 | 72 | single GPU, a few hours |
| PhyLatent (LeWM + physical grounding) | — | 70.0 → **78.1** | arXiv 2608.05720, Aug 6 |
| ProWorld (hyperbolic progress cost) | — | — | **+9.67 avg** over LeWM on 4 tasks, 2608.01926 |

Implication: MetaWorld + DINO-WM/JEPA-WM released checkpoints is a *legacy* arena. No
reviewer in this line has a reference number there. We already have
`stable_worldmodel` training plumbing in `crod_h0/` (prejepa.py reproduction) and
OGBench-Cube plumbing in `counterfactual_flow/` — migration is half-done.

**Recommendation: make Push-T + OGBench-Cube (+ Two-Room/Reacher for breadth) the primary
arena. Keep RoboCasa/DROID for a real-robot generality section. MetaWorld only as legacy
evidence in the audit paper.**

## 2. The publishable template in this line

The ICML-2026-accepted "Temporal Straightening for Latent Planning" (2603.12231,
ICML poster 64904) is the shape the user noticed: take the LeWM/DINO-WM base,
add **one** geometry-shaping loss (a curvature regularizer on a light projector),
show MPC success gains on the standard suite. Same template repeated all year:

- **RC-aux** — "Predictive but Not Plannable" (2605.07278): predictive ≠ plannable.
- **Temporal-Distance-JEPA** (2607.25337): mine directed temporal distance from logs as
  the planner's terminal cost + rollout-consistency loss. Code on GitHub.
- **PhyLatent** (2608.05720): 3 collapse modes (physical invariance / identifiability /
  counterfactual dynamics), 3 auxiliary pathways.
- **ProWorld** (2608.01926): goal-conditioned progress order + hyperbolic entailment.
- **Slot-MPC** (2605.14937): object-centric latents for GC-MPC.
- **Latent State Design under Sufficiency Constraints** (2605.01694).

So: one loss + the standard suite + a beat-the-baseline table = a paper. Our audit-style
rigor (locked protocols, matched nulls, CIs) is a *bonus*, not the entry ticket.

## 3. Theory papers that hand us a mandate

- **A Control Theory of Predictability in Latent World Models** (2607.10362, Jul 2026).
  Planner queries the model *off* the data manifold; suboptimality splits into a small
  on-manifold residual and a binding **off-manifold divergence** that no data-averaged
  loss bounds. Empirically: single-step validation error is essentially **uncorrelated**
  with control success across seeds, while a fidelity score on the *planner-reachable
  measure* tracks it. Linear/Koopman analysis, small experiments.
  → This is our whole null corpus, formalized by someone else. It also leaves the
  practical half wide open: **no training recipe and no cheap model-selection score on
  the reachable measure exists.**
- **What Can Latent World Models Know? Physical Parameter Identifiability** (2607.27017).
  POKEWORLD + the X-JEPA family factorially varies *which modality is an input* vs
  *which is a prediction target*. Headline: **contact stiffness enters the latent only
  when touch is a prediction target (R²=0.50) — not when touch is merely fused into the
  input (R²=-0.02)**; a supervised sys-id head lifts drag 0.12 → 0.45.
  → Directly re-opens our ContactWorld NO-GO: that pilot tested tactile as an **input**,
  which is precisely the axis this paper says is inert. The **target** axis is untested
  by us and untested on any real planning benchmark.
- **How Should World Models Be Evaluated for Embodied Decision-Making?** (2606.15032):
  position paper, L0–L7 evidential ladder, explicitly names "claim/evidence mismatch" as
  the field's recurring problem. A natural home for our diagnostic corpus.
- **On the Identifiability of Controlled World Models** (2607.22430): joint identifiability
  (representation + transition) under which LeJEPA-style objectives recover controlled
  dynamics up to orthogonal transform.

## 4. Ranked candidate directions

### A. "Forecast the contact, not the pixels" — contact as a *prediction target* (RECOMMENDED)
One auxiliary head on the LeWM base predicting **contact events / contact forces /
contact-normal** at t+k from vision-only inputs (sim gives GT free in Push-T,
OGBench-Cube, RoboCasa, MetaWorld). Factorial ablation replicating X-JEPA's design on a
*real planning benchmark*: contact as {input, target, both, neither}.
- Beat: LeWM 72 and PhyLatent 78.1 on OGBench-Cube; LeWM 94 on Push-T.
- Novel vs PhyLatent: PhyLatent grounds *object state*; contact events are a different,
  regime-specific target, and the input-vs-target factorial + contact-stratified eval is
  ours alone (we already have the stratifier and boundary-blindness metric).
- Cost: LeWM is ~15M params, single GPU, hours. Cheapest real shot we have.
- Risk: PhyLatent-adjacent; must show the delta concentrates in contact regimes, which is
  exactly what our stratification measures.
- Null value: high. "Contact-as-target does not help MPC on real benchmarks even though it
  helps identifiability in POKEWORLD" is itself a reportable contradiction of 2607.27017.

### B. Planner-reachable model selection / fidelity score
Ship the practical half of 2607.10362: a cheap score, computed without running MPC, that
ranks checkpoints by control success. Validate across the LeWM suite × many checkpoints ×
seeds, plus our released DINO-WM/JEPA-WM checkpoints. Our oracle-dynamics harness gives
the ground-truth on/off-manifold decomposition nobody else has.
- Deliverable even on failure: "validation loss ⊥ control success at scale on real
  pretrained WMs" is a measurement result the position paper (2606.15032) explicitly asks for.
- Cost: lowest of all options — mostly re-runs of existing scripts on new envs.
- Risk: 2607.10362 owns the framing; we own the practice and the scale.

### C. Contact-stratified extension of the stable-worldmodel benchmark (D&B / TMLR resource)
Fold the whole null corpus (scaling null 22M→1B, oracle ladder, CEM reward-hacking,
mined-elite decode 91.5%→24%, encoder-LoRA null, ensemble null) into a
contact-stratified evaluation suite on top of stable-worldmodel, framed against the
L0–L7 ladder. Turns sunk cost into a resource contribution.

### D. Off-manifold / reachable-measure fine-tuning (higher risk)
Train predictor + cost jointly on CEM-elite and off-manifold candidates (the states the
planner actually queries), not on the demo distribution. Distinct from our Phase-A
adversarial mining, which hardened a **cost only, under a frozen encoder**. Under
2607.10362 the fix must touch the *predictor* on the *reachable measure*.
- Risk: the closest thing to what already failed; but the theory says the previous
  attempt was applied at the wrong place.

### E. Hierarchical / subgoal planning to remove the adversary (FF-JEPA line)
FF-JEPA (2606.09311) is action-free-subgoal + short-horizon CEM, "preliminary experiments
on PushT" only — an open arena. Pairs with our finding that flat CEM *is* the adversary.
Our E1 amortized-control null used a different mechanism (single-shot GC-IDM), so the
subgoal decomposition is untested by us.

## 5. Do NOT re-enter these (crowded or scooped)

- Counterfactual/action-conditioned objective — UWM-JEPA 2605.25313 + our Phase H.
- Temporal-distance / progress-ordered cost — TD-JEPA 2607.25337, ProWorld 2608.01926.
- Curvature / straightening — ICML 2026, 2603.12231.
- Ensembles + disagreement penalties — our Phase G null, and standard MBRL.
- Stochastic / distributional predictors — MoP-JEPA + Branch-JEPA (2607.05238),
  VJEPA (2601.14354, ICML 2026 poster), Var-JEPA (2603.20111), UWM-JEPA belief space.
- Tactile/force as an **input** modality — our ContactWorld NO-GO and 2607.27017 agree
  this axis is inert. (The *target* axis is a different claim; see A.)
- Diagnostic-only action-consistency analysis — ATM 2606.09028.

## 6. Suggested near-term plan (deadline: ICLR 2027, recorded as Sep 19 abstract / Sep 24 full)

1. **Week 1** — entry ticket: reproduce LeWM baseline MPC numbers on Push-T +
   OGBench-Cube inside `stable-worldmodel`. Nothing publishes here without that table.
2. **Week 1–2** — port the contact stratifier + boundary-blindness metric onto the
   stable-worldmodel envs (contact GT is free in MuJoCo/Push-T).
3. **Week 2–3** — direction A factorial: {input, target, both, neither} × 3 seeds,
   Push-T + OGBench-Cube, contact-stratified MPC success.
4. **In parallel, cheap** — direction B: correlate validation loss vs MPC success across
   every checkpoint produced along the way. Costs almost nothing and is a section either way.
5. Fall back to C (resource paper) if A and B both come back null.

