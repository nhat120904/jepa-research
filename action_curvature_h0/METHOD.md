# Action-Space Curvature Mismatch in Latent World Models

Full statement of the idea, the theory behind it, the measurement, the
intervention, and the ways it can fail. `PROTOCOL.md` is the locked
preregistration; this document is the reasoning that produced it.

Status: Stage 1 (measurement) specified and implemented, not yet executed.
Stage 2 (intervention) specified, gated on Stage 1.

---

## 1. The question

> Do latent world models distort the local action geometry of the environment,
> particularly in regions visited by planning, and does correcting that
> distortion improve control?

The first half is a measurement and stands on its own. The second half is an
intervention used to test whether the measured distortion is causal. A null on
the second half does not invalidate the first.

## 2. Where this comes from

A latent world model is trained to predict. It is then deployed inside a
planner that optimizes over action sequences. Those are different objects, and
a long series of interventions on this project failed to close the gap between
them: post-hoc costs on a frozen encoder, encoder-LoRA fine-tuning, ensemble
disagreement penalties, contact-aligned gating, amortized control. Each was a
clean null. The one positive (a counterfactual predictor objective) was
concurrently published elsewhere.

What survived every time was the *diagnosis*. That asymmetry is the reason this
program is built measurement-first.

Two 2026 results sharpen the target:

- **A Control Theory of Predictability in Latent World Models**
  (arXiv:2607.10362) shows the planner queries the model off the data manifold,
  splits planner suboptimality into a small on-manifold residual and a binding
  off-manifold divergence that no data-averaged loss bounds, and reports that
  single-step validation error is essentially uncorrelated with control success.
  It is a theory paper; it leaves the practical half open.
- **What Can Latent World Models Know?** (arXiv:2607.27017) shows that inputs
  bound what *can* be represented while prediction targets decide what *is*
  retained. It is an identifiability study in a synthetic environment.

Neither measures the object a sampling-based planner actually consumes.

## 3. The core observation

Temporal Straightening (ICML 2026) regularizes the curvature of latent
trajectories in time:

    d^2 z / dt^2  ~  0

But a planner does not optimize over `t`. It optimizes over action sequences,
and consumes a single scalar per candidate: the terminal cost. The object it
actually queries is the **composed H-step action-to-outcome map**

    Phi_H : a = (a_0, ..., a_{H-1})  |->  F^H(E(o_t), a)

The corresponding geometric property is

    d^2 z_H / d a^2  ~  0

These are **not equivalent**. A trajectory can be perfectly straight in time
while the map from the action sequence to the terminal latent is strongly
curved. LeWM has been observed to develop temporal straightening on its own
without an explicit loss for it, which makes the interesting version of the
opening claim:

> The state trajectories are already straight. The control map is not.

## 4. Why curvature should matter to a planner

For a goal `z_g`, the planner minimizes

    C_g(a) = || Phi_H(a) - z_g ||^2

With `J = d Phi / d a` and residual `r = Phi_H(a) - z_g`, the Hessian splits:

    grad^2 C_g  =  2 J^T J  +  2 sum_k r_k grad^2 Phi_k
                   \_______/     \___________________/
                   Gauss-Newton   residual-weighted curvature

The first term is positive semi-definite and well behaved. The second comes
from the curvature of the action-to-outcome map and is what can make the
landscape indefinite. Note it is weighted by the residual: curvature matters
most far from the goal, which is where planning starts and where long horizons
live.

Along a direction `d`, the curvature term is estimable by finite differences
without ever forming a Hessian:

    Phi(a+d) - 2 Phi(a) + Phi(a-d)  ~  d^T grad^2 Phi d      (error O(d^4))
    Phi(a+d) - Phi(a-d)             ~  2 J d

**An honest caveat that shapes the whole design.** This Hessian argument is an
argument about *gradient-based* planners, and Temporal Straightening's gains
are reported in exactly that setting. LeWM plans with CEM, which is
derivative-free and largely indifferent to Hessian conditioning. The CEM-side
argument is weaker and is stated as a hypothesis, not a theorem:

> CEM refits a diagonal Gaussian to its elites. A fixed rotation of the
> cost's sublevel sets is partly absorbed by that refit, because the mean
> tracks it. Curvature is different: it makes the local principal axes vary
> *with* `a`, so each CEM iteration lands in a region whose geometry differs
> from the one the previous refit modeled.

Low curvature does **not** imply diagonal-CEM-friendliness — a purely linear
`Phi` still yields an ellipsoidal sublevel set that can be rotated 45 degrees
away from the coordinate axes. So the CEM claim must be earned empirically, by
measuring elite retention and recall across CEM iterations and by checking
whether the off-diagonal structure of the elite covariance *changes across
iterations* (curvature) rather than merely being nonzero (rotation).

## 5. The decomposition that defines the intervention

Write the two displacements of a symmetric triplet:

    v- = Phi(a) - Phi(a-d)        v+ = Phi(a+d) - Phi(a)

Then the second difference is `D2 = v+ - v-`, and exactly:

    ||D2||^2 = ( ||v+|| - ||v-|| )^2  +  2 ||v+|| ||v-|| ( 1 - cos(v-, v+) )
               \________radial_______/    \_____________angular___________/

The angular term is `2 ||v+|| ||v-||` times the cosine straightening loss
`L_AS = 1 - cos(v-, v+)`. The radial term is `L_norm = (||v+|| - ||v-||)^2`.

Three consequences:

1. The two candidate losses are not treatment and placebo. They are the
   **angular and radial halves of one exhaustive decomposition of curvature**,
   so the ablation answers *which component of curvature matters*, not merely
   *does the loss help*.
2. A cosine loss is blind to radial curvature: `v+ = 100 v-` scores as
   perfectly straight. This is why the reported metric is the normalized
   finite difference, not the cosine.
3. The split is measurable **before any training**. If the mismatch is almost
   entirely radial, a cosine regularizer targets the wrong component, and that
   is known for the price of the diagnostic rather than the price of a sweep.

## 6. Stage 1 — the measurement

The novel ingredient is a **reference**. Curvature alone cannot distinguish "the
model invented this geometry" from "the world really is this nonlinear". With an
exact full-state simulator reset, both are available in the same latent chart:

    model map     Phi_H(a) = F^H(E(o_t), a)
    realized map  Psi_H(a) = E(o_H^sim(s_t, a))

Same encoder, same action triplet, same horizon. Their difference is
type-correct, and the primary quantity is the **curvature mismatch**

    E_K = || D2 Phi - D2 Psi || / ( || Psi(a+d) - Psi(a-d) || + eps )

Three properties make this the right primary metric:

- Because `D2` annihilates affine terms, `D2 Phi - D2 Psi = D2 eps` where
  `eps(a) = Phi_H(a) - Psi_H(a)`. So `E_K` measures the **nonlinear
  action-dependence of the model's error**, not rollout MSE: constant and
  locally affine prediction bias along the probed direction cancels exactly.
- A signed magnitude gap `K_model - K_true` would miss the case where both maps
  curve equally hard in opposite directions. `E_K` catches it; the alignment
  `cos(D2 Phi, D2 Psi)` reports it directly.
- The denominator is the *realized* sensitivity, never the model's own, so an
  action-deaf model is not flattered by a small denominator.

The taxonomy this yields:

| phenomenon                          | K_model | K_true | E_K  |
|-------------------------------------|---------|--------|------|
| model genuinely smooth              | low     | low    | low  |
| genuine physical nonlinear boundary | high    | high   | low  |
| **spurious model curvature**        | high    | low    | high |
| model erases real nonlinearity      | low     | high   | high |
| right magnitude, wrong direction    | similar | similar| high |

Row 4 is the failure the intervention itself risks creating. Row 3 is the
target.

### Boundaries, defined counterfactually

"Is this a contact transition?" is the wrong question, because the property that
matters belongs to the *perturbation*, not to the logged state: the logged
transition may be contact-free while `a+d` touches the object and `a-d` misses
it. So the mode label comes from the contact patterns of the three **true**
rollouts, unioned over the rollout rather than sampled at the terminal instant.

A second, label-free detector cross-checks it. Fitting the **raw** second
difference against the perturbation scale,

    log || D2 Psi ||  =  alpha log ||d|| + c

gives `alpha ~ 2` in a smooth region, `alpha ~ 1` at a kink, `alpha ~ 0` at a
jump. This must be fit on the raw quantity: at a symmetric kink the central
span vanishes while `D2` stays large, so the normalized curvature diverges and
its slope is not `alpha`. Where the two detectors disagree, the disagreement
localizes mode changes the contact proxy misses — slips, rolls, joint limits.

### Regions

Measuring only around logged actions would probe a narrow tube along the
behavior manifold, which is *not* the planner-reachable measure. Cached CEM
populations supply the planner-visited region, and the comparison is run two
ways because a single fixed perturbation scale confounds two different things:

- `K_fixed` — same scale as the offline measurement, isolating **location** shift.
- `K_local` — perturbation drawn from the recorded CEM proposal covariance,
  giving the geometry at the scale the planner **actually queries**.

Claiming "the planner enters highly curved regions" requires `K_local` to move,
not merely `K_fixed`.

### Pipeline gates

Second differences are fragile, so the measurement carries its own controls:

- **Effector curvature** on the directly actuated end-effector position is
  expected to be low. High values indict the measurement pipeline — reset,
  action scaling, triplet construction — rather than the physics, and block the
  run.
- **Repeat floor** at `d = 0`. With deterministic reset, physics, renderer and
  encoder this is exactly zero, and that is a *false all-clear*: the binding
  small-`d` limit is float cancellation, since `D2` subtracts `O(1)` quantities
  to recover an `O(d^2)` result. The real test is the existence of a clean
  `d^2` regime, and all differences are taken in float64.
- **Clipped triplets are discarded, not corrected**: asymmetric spacing injects
  a first-order term `f'(a)(h+ - h-)`, exactly the affine component the
  diagnostic relies on cancelling. Since near-bound actions are not uniformly
  distributed across contact regimes, the discard rate is logged per stratum.

## 7. Stage 2 — the intervention

Only if Stage 1 finds spurious curvature. Four arms on the LeWM backbone:

1. `LeWM`
2. `+ open-loop multi-step prediction`
3. `+ open-loop multi-step prediction + AS`     (angular half, `1 - cos`)
4. `+ open-loop multi-step prediction + norm-symmetry`  (radial half)

Arm 2 is not optional. The AS loss backpropagates through `H` unrolled steps
while LeWM trains on teacher-forced next-latent prediction, so without a
matched multi-step arm any gain is confounded with "multi-step training beats
one-step training". The arm must match the computational graph, not merely the
step count; setting `lambda = 0` while still rolling out is **not** a control,
because no gradient flows through those rollouts.

Arms 3 and 4 have gradient-norm-matched `lambda`.

A secondary stress test uses independent perturbations `(a+d1, a, a-d2)`. It is
explicitly **not** a null: forcing `cos(v+, v-) -> 1` across independent
directions pressures `J` toward a rank-1 response, so it is an actively harmful
treatment. It answers "does the symmetry matter", not "does anything help".

Reading rule:

| outcome                                   | interpretation                        |
|-------------------------------------------|---------------------------------------|
| AS > baseline, symmetry-broken <= baseline| mechanism is curvature                |
| AS > baseline, symmetry-broken ~ AS       | mechanism is any alignment pressure   |
| AS ~ baseline, symmetry-broken < baseline | loss acts but in the wrong direction  |

## 8. Failure modes

**Action-linear collapse.** The degeneracy is not global collapse — the cosine
loss penalizes that. It is that `Phi` can satisfy the loss by depending on the
action sequence *linearly*, e.g. only through the sum of actions. Straightness
becomes perfect, sensitivity stays high, and the model becomes exactly the
action-blind model this project spent a year documenting. Straightening could
therefore improve plannability by making the model less physically correct at
contact — a Goodhart of the program's own thesis. Sensitivity metrics do not
detect this; contact-stratified CRA and boundary-blindness do, and they are
guard metrics on every arm. If plannability rises while contact fidelity falls,
that trade-off is a stronger result than a success, and only this project is
instrumented to see it.

**Chart dependence.** Curvature is not coordinate-invariant: under a smooth
reparameterization `z = g(s)`, `d^2 (g o h) = g'' (h')^2 + g' h''`. Two
consequences are binding. Latent and simulator-state curvature are never
compared in absolute terms — the state chart is one arbitrary chart, and a
heterogeneous one. And since Stage 2 arms have *different encoders*, a raw
cross-arm `E_K` comparison is not by itself meaningful; the primary cross-arm
metric must be chart-invariant, i.e. rank agreement between the model's
ordering over candidates and the true ordering. `E_K` is safe within Stage 1
because it uses a single frozen checkpoint.

An encoder that straightens the latent manifold relative to raw state
coordinates is doing its job, not cheating. The state anchor therefore only
*localizes mechanism* — predictor repair versus representation reshaping — and
reshaping is called harmful only when task-relevant fidelity falls with it.

**Genuine discontinuity.** In manipulation the map is not globally smooth: a
millimetre can separate "touch" from "miss". Straightening across such a
boundary erases real physics. The measurement answers this before the method
commits: at a genuine boundary the realized map is also strongly curved, so the
mismatch is small there and the loss should not be pushing. If instead the
mismatch concentrates *inside* physical modes, a gated V2 —
`w(s,a,d) * L_straight` with `w ~ 0` at real boundaries — follows from the data
rather than from intuition. No gating is built in up front; a previous program
on this project died because a random control tied with contact-aligned gating.

## 9. Relation to prior work

The nearest prior art is **not** the 2026 LeWM line but **PCC — Prediction,
Consistency, Curvature** (ICLR 2020, arXiv:1909.01506), whose third principle is
literally low curvature of latent transition dynamics, with an ablation showing
the curvature term matters. Related: E2C/RCE locally-linear latent dynamics,
Predictive Coding for Locally-Linear Control, and the deep-Koopman line that
linearizes dynamics to make control easy.

Novelty cannot be "nobody has regularized curvature for control". It has to be
specific:

> PCC regularizes local one-step latent transition dynamics so that a
> linearization-based controller (iLQR) is valid. We study the curvature of the
> **composed H-step action-sequence-to-terminal-latent map** that modern JEPA
> MPC actually queries, for **sampling-based** planners, and we measure it
> against a **realized reference** obtained by replaying the same action triplet
> through the simulator, rather than assuming low curvature is good.

The 2026 line is adjacent but attacks the cost rather than the action geometry:
Temporal Straightening (temporal curvature), TD-JEPA and ProWorld (progress
ordering), PhyLatent (physical grounding), RC-aux (predictive-but-not-plannable),
Slot-MPC (object-centric latents).

## 10. What is claimed under each outcome

- **Spurious curvature found, intervention helps.** A measurement, a mechanism,
  and a one-loss method with a component-level ablation.
- **Spurious curvature found, intervention does not help.** The strongest form
  of the project's recurring finding, now with a reference map: the model's
  action geometry provably differs from the environment's, and correcting the
  measured difference is not sufficient for control. This is publishable and is
  the outcome the program is designed to survive.
- **Curvature is genuine everywhere.** The model reproduces the environment's
  local action geometry; there is nothing to repair on this axis, and the search
  for the contact wall moves elsewhere. Cheap, and it closes an axis honestly.
- **Model is action-deaf.** A different defect, out of scope here, and the
  diagnostic says so rather than producing a flattering curvature number.

## 11. Deferred

- Push-T and the rest of the stable-worldmodel suite, pending a verified
  exact-reset harness. Stage 1 is OGBench-Cube only.
- On-policy AS sampling from CEM populations during *training* (as opposed to
  measurement), which would make the loss genuinely act on the
  planner-reachable measure. Justified only if the Stage-1 region comparison
  shows the planner-visited geometry differs.
- Gripper aperture in the state anchor, excluded because it is not a position
  and would reintroduce unit mixing.
