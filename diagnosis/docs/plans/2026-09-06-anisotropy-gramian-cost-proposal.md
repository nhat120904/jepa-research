# Proposal: the planner's cost is unwhitened — action-Gramian metrics for latent world models

Date: 2026-09-06. Status: **idea + preregistration sketch**, nothing executed.
Author-facing document; the locked protocol must be written separately before any job.

---

## 0. One sentence

> A latent terminal cost penalises a candidate for the **total** predicted change it
> causes, not for the change **in the direction the task needs**; because a JEPA encoder
> spends its variance on what moves most in the data (the arm) rather than on what the
> success predicate scores (the object, the latched buttons), Euclidean latent-L2 makes
> the planner's optimum "move as little as possible while pointing vaguely goalwards" —
> and whitening the goal residual by the **covariance of outcomes the current action set
> can actually achieve** removes that pathology at zero extra model calls.

Property name: **directional attainability** of the goal residual.
Measured quantities: the condition number `kappa(J)` of the action->outcome map and the
alignment angle `theta` between the goal residual and the top singular subspace of `J`.

---

## 1. Why this idea and not another one

Every intervention in this repo so far changed **what the cost measures** (grounding,
decoding, progress, temporal distance, curvature, tactile, ensembles) or **who computes
it** (LoRA, adapters, amortisation). None changed **the metric the residual is measured
in**. The whole corpus is consistent with the metric being the defect:

| repo result | reading under the anisotropy hypothesis |
|---|---|
| oracle ladder: exact dynamics, latent-L2 still 0/64 push | exact `Phi` does not change `kappa(J)` or `theta`; the metric is untouched |
| reference (physical) cost 64/64 | it is *hand-whitened*: object-relative coordinates, arm term shaped in |
| reach 16/16, push 0-2/16, drawer-close solved, button/window fail | exactly the `theta` ordering: reach's task direction **is** the encoder's high-variance direction |
| grounding MSE 0.0043, push still 1/16 | accuracy of coordinates does not fix their **weighting** |
| off-policy-robust probe still re-gated to 1/16 | any *readout* has a residual to exploit; a second moment of the model's own candidate set has none |
| scene-aliasing probe 98-99% on latched vars | the information is in the latent; only the metric discards it |
| angular curvature -> false valleys, but continuation is a planning null | curvature is the residual-weighted Hessian term; `J^T J` — the Gauss-Newton term — was never used |
| elite-optimism `EXPLOITATION_ABSENT` on `prog_ssl` | failure is not adaptive over-fitting; it is a **static** mis-weighting, present at iteration 0 |
| 31% of action components scored out of bounds, flat across generations | CEM pushes along directions the cost keeps rewarding but the box cannot deliver |
| elites change any goal component only 7.7-18%; `random_init` net component change is **negative** | the "prefer small predicted motion" bias, measured |
| "propose more diverse actions" beat CROD/GFPR/PSD | diversity widens the achievable ellipsoid; it is a crude version of the same fix |
| pure progress cost -13.0 pts, CI [-22, -4] | changing *what* is measured without changing *how* it is weighted made it worse |

Nothing in the list is refuted by the hypothesis, and four of them (reach-vs-push,
drawer-vs-button, the negative `component_delta`, the flat clipping rate) are
**retrodictions** — data already on disk that the theory predicts before it is fitted.

---

## 2. The imported principle

Two mature fields say the same thing, and neither has been transplanted into JEPA planning:

1. **Control theory.** The natural metric on a control system's state space is not
   Euclidean; it is the **inverse controllability Gramian**. `r^T W^{-1} r` is the
   minimum control energy to move the state by `r`. Balanced truncation and Hankel
   singular values are built on exactly this. Empirical Gramians for nonlinear systems:
   Lall, Marsden & Glavaski (1999).
2. **Detection / estimation theory.** To compare a signal against a background you
   **whiten by the background covariance** — Mahalanobis distance, the matched filter,
   Fisher's discriminant. An unwhitened inner product is the textbook wrong detector when
   the noise is anisotropic.

Both reduce to one statement: **Euclidean assumes isotropy. The outcome space of a
learned world model is violently anisotropic, and the anisotropy is task-irrelevant.**

Why JEPA specifically must violate it: the encoder's objective allocates representation
variance to what varies most in the training data. In manipulation play data that is the
arm and the gripper. The success predicate scores the object pose and a handful of latched
button/drawer/window pixels. So the encoder's principal directions and the task's
directions are *systematically misaligned by construction*, and the terminal cost sums
over both with weight 1.

---

## 3. Formalisation (linearised; one page of real theory)

At a replan state, let `Phi_H: a -> z` be the model's H-step action-to-outcome map, `abar`
the current CEM proposal mean, `delta = a - abar ~ N(0, sigma^2 I)`, and

    Phi_H(a) = mu + J delta + O(||delta||^2),   J = D Phi_H(abar),
    Sigma    = sigma^2 J J^T                       (achievable-outcome covariance)
    c        = mu - z_g                            (goal residual at the proposal mean)

**Euclidean cost.**

    C_E(a) = ||c||^2 + 2 c^T J delta + ||J delta||^2

The `a`-dependent part is `2 c^T J delta + ||J delta||^2`. The quadratic term penalises
displacement **weighted by the singular values of J** — i.e. a candidate is punished for
moving the high-variance (arm) directions even when the goal needs no such motion. Its
constrained minimiser over `||delta|| <= rho` is

    delta*_E  =  -(J^T J + lambda I)^{-1} J^T c

whose alignment with the min-energy solution degrades as `kappa(J)` grows and as `c`
concentrates on the small singular directions of `J`.

**Whitened cost.**

    C_W(a) = (Phi_H(a) - z_g)^T (Sigma + eps I)^{-1} (Phi_H(a) - z_g)

The displacement penalty becomes isotropic in `delta`, and (eps -> 0, `c` in range(J))

    delta*_W  =  -J^+ c / sigma^2   (the minimum-energy action achieving the residual)

**Proposition (informal).** The ranking induced by `C_E` and by `C_W` coincide iff
`J J^T` is isotropic on the subspace spanned by `c` and the candidate spread. The regret
of `C_E` relative to `C_W`, measured as the angle between the selected displacement and
`J^+ c`, is monotone in `kappa(J)` and in `sin(theta)` where `theta` is the angle between
`c` and the leading singular subspace of `J`.

**Two testable consequences, before any experiment:**

* P1 (task ordering): planning success under latent-L2 should order tasks by
  `cos(theta) / kappa(J)`. Reach and drawer-close high, push / button-press /
  window-close low. *Already observable in existing oracle-ladder data.*
* P2 (null prediction): where `kappa(J) ~ 1` (isotropic controllability — Reacher,
  TwoRoom, short goal offsets), the whitened cost must give **no** gain. A gain there
  falsifies the mechanism even though it would look like a win.

---

## 4. Method: one metric change, three levels

Everything below leaves the encoder, the predictor, the sampler, the budget and the
evaluation manifest **fixed**. Only the scalar CEM descends changes.

**L0 — baseline.** `C = ||Phi_H(a) - z_g||^2` (identity metric).

**L1 — population whitening (free).** `Sigma_hat` = shrinkage covariance (Ledoit-Wolf) of
the *predicted terminal latents of the current CEM population*. The planner already
computed all of them. Cost `= r^T (Sigma_hat + eps I)^{-1} r`, low-rank via the N x N
Gram of the centred population — **zero extra model calls**, one `N x N` solve per CEM
iteration.

**L2 — analytic action-Gramian.** `W = J J^T` from `m+1` forward rollouts (`m = H * |A|`,
about 25-75 columns), once per replan, not per candidate. More faithful than L1 when the
proposal has collapsed.

**L3 — anticipatory Gramian (the contact claim).** Evaluate the metric at the *predicted
end state*: `C(a) = r(a)^T (Sigma(zhat_end(a)) + eps I)^{-1} r(a)`. This prefers
candidates that end where the goal direction is **cheap to move** — i.e. it rewards
*acquiring* controllability (getting the gripper onto the object) without any hand-written
hand-to-object term and without privileged state. Made tractable by distilling a low-rank
`Sigma` head `g_psi(z)` self-supervised on the frozen model's own Jacobians (no labels).

**Oracle ceiling (internal, not deployable).** `W_sim` from simulator finite differences
via snapshot/restore. Gives a 2x2 ladder `{Phi_model, Phi_sim} x {W_model, W_sim}` that
separates "the idea is wrong" from "the model's Jacobians are wrong" — the same
methodology as the oracle ladder, on the new axis.

### Ablation ladder (each rung has a scientific reading, not a module name)

| arm | question it answers |
|---|---|
| L0 identity | baseline |
| diag(`Sigma`) only | is it merely per-coordinate scaling? |
| `logdet W` bonus (undirected empowerment) | is it merely "be in controllable states"? |
| L1 root whitening | does attainability *now* carry the ranking? |
| L3 end-state whitening | does *acquiring* controllability carry the contact gain? |
| oracle `W_sim` | how much of the gap is the model's Jacobian error? |

---

## 5. Evidence chain (locked shape, mirrors the E1-E6 template)

**E0 — kill gate, cheapest, no training.** On Scene and Cube replan states, estimate `J`
(finite differences, frozen LeWM) and report the spectrum, `kappa(J)`, and
`alpha = ||P_J c||^2 / ||c||^2` measured **inside the data-latent PCA subspace** (not the
ambient dimension — see risks). Ground truth cross-check: in OGBench-Scene the buttons
*lock* the drawer and the window, so the simulator's own Jacobian has a hard structural
zero there. Compare the model's Gramian energy in the drawer direction before vs after the
unlock, against the simulator's.
* **STOP** if `alpha` is already high at the states where planning fails (residual is
  attainable -> thesis dead), or if `kappa(J) ~ 1` on the failing tasks (no anisotropy to
  exploit).
* **Branch** if model-J and sim-J rank-disagree: the zero-training version dies, only the
  oracle ceiling plus a Jacobian-matching training fix survive — and that is a reportable
  result either way.

**E1 — the property exists and is missing.** `kappa`, `theta`, `alpha` across tasks and
goal offsets. Must reproduce P1 on data already collected.

**E2 — the property predicts the failure.** Per-state regression of `cos(theta)/kappa`
against (a) CEM ranking Spearman vs `c_true`, (b) closed-loop success, (c) the
out-of-bounds clipping rate. Episode-clustered bootstrap. Reuse the elite-optimism
`c_true` oracle-repair harness verbatim.

**E3 — the intervention changes the property (manipulation check).** The induced ordering
must actually move (Kendall tau vs the L0 ordering strictly below 1) and the whitened cost
mass must shift onto attainable directions. *This is the check the cosine-straightening
loss failed; it is a hard gate here.*

**E4 — the predicted intermediate mechanism moves.** Ranking accuracy and top-k recall vs
`c_true` up; clipping rate down; `component_delta` (already instrumented) turns positive.

**E5 — closed loop.** Scene offsets {25, 50, 100, 200} against the replay ceiling;
OGBench-Cube; PushT. Paired episodes, paired plan seeds, matched budget, reported next to
published baselines.

**E6 — boundary tests, preregistered to fail.**
1. Reacher / TwoRoom / short offsets: predicted gain **zero** (P2).
2. **Unlocked-Scene ablation:** start the buttons unlocked so the structural zero in the
   Gramian disappears. The theory predicts the gain shrinks. This is an intervention on
   the environment, not on the method, and it is cheap.
3. `eps` sweep: the effect must be a plateau, not a knife edge.

---

## 6. Arenas, each with a role

| arena | role | why |
|---|---|---|
| TwoRoom / Reacher | **predicted-null** control | isotropic controllability; a gain here falsifies |
| OGBench-Scene | **diagnostic + stress** | buttons lock the drawer/window: ground-truth structural zeros in the Gramian; full plumbing already exists here |
| OGBench-Cube, PushT | **external validity + published baselines** | LeWM 72 / 94, DINO-WM 86 / 92, PhyLatent 78.1, ProWorld +9.67 |
| RoboCasa / DROID | optional generality section | legacy assets |

Scene is the environment this idea was born for and the only public benchmark with
**known, state-dependent, ground-truth controllability structure**.

---

## 7. Baselines that must be beaten at matched budget

LeWM latent-L2; DINO-WM; TRM (2605.22164); RC-aux (2605.07278); ACID (2607.02403);
Temporal-Distance-JEPA (2607.25337); ProWorld (2608.01926); plus the internal
diag / logdet / oracle-W rungs.

---

## 8. Novelty wedge (web-snippet audit 2026-09-06; arxiv is unfetchable from the login
node, so every abstract below must be re-verified on a compute node before locking)

The 2026 cost-geometry cluster is crowded, and the crowding is on *what* the cost
measures:

* **TRM** — a *learned* pairwise reachability head fitted to **logged trajectory
  structure**; horizon-matched supervision; TwoRoom 7.0 -> 97.0.
* **RC-aux** — a learned reachability head asking "is the goal reachable in the remaining
  budget".
* **ACID** — inverse-dynamics cycle-consistency residual folded in as an adaptive weight.
* **TD-JEPA / ProWorld** — temporal distance / progress order as the cost.
* **Temporal Straightening** — curvature of the latent trajectory in *time*.
* **Decision-Metric Alignment (2608.18746)** and **The Objective Is the Bottleneck
  (2608.12959)** — diagnostics of latent-vs-real rank agreement.

None of them is the same object as this proposal:

1. **It is not a learned metric.** L1/L2 are the frozen model's own second moment /
   differential. TRM and RC-aux are fitted to logged trajectories, so they are
   data-averaged and defined on the data manifold; the candidates CEM invents are exactly
   outside that support, which is the failure 2607.10362 formalises.
2. **It is anisotropic and state-dependent.** Every neighbour rescales or replaces the
   scalar. This changes **which directions count**, per state and per horizon.
3. **It is defined at query time, off-manifold.** Wherever the planner asks, the metric
   exists.
4. **The mechanism is new**: the isotropic displacement penalty and its
   `kappa(J)`/`theta` law, together with the prerequisite-lock structure as ground truth.
5. **The oracle-Gramian ladder** — separating "the metric idea is wrong" from "the
   model's Jacobians are wrong" — cannot be run without this repo's snapshot/restore
   harness.

**Standing scoop risk, unrelated to this proposal but urgent:** 2608.18746 and 2608.12959
appear to overlap the audit paper's diagnostic claims (rank agreement between latent cost
and real cost, "encoders encode what planners cannot use"). Read both before the next
revision of `paper/main.tex`.

---

## 9. Risks, named in advance

1. **Rank deficiency.** `rank(Sigma) <= N` (population) or `<= m` (Jacobian), against a
   latent of ~10^4 dims, so a naive `alpha` is trivially near zero and a naive projector
   destroys the signal. *Mitigations, preregistered:* measure `alpha` inside the
   data-latent PCA subspace; use the ridge/shrinkage form `(Sigma + eps I)^{-1}`, which is
   well-defined at any rank and reduces to L0 as `eps -> inf`; truncate at a fixed energy
   fraction. The ridge form also makes L0 a **special case** of the method, so the
   comparison is a one-parameter path, not two unrelated costs.
2. **Whitening amplifies junk directions** where `Sigma` is numerically tiny. Same
   mitigation; `eps` chosen once on validation, frozen, sensitivity reported.
3. **Non-smoothness at contact** makes a pointwise Jacobian meaningless. The population
   estimator is the *smoothed* (proposal-averaged) sensitivity and is the correct object
   for a CEM that ranks under a Gaussian proposal — this is a feature, not a patch.
4. **The gain may be a re-derivation of shaped costs.** Answer: the shaping here is
   derived from the model, uses no privileged state, and the unlocked-Scene ablation
   tests whether it tracks real controllability structure.
5. **It might just be better-conditioned optimisation, not better ranking.** Separated by
   E3/E4: ranking metrics are measured against `c_true` at fixed populations, before any
   closed-loop number is read.
6. **Compute.** L1 is free; L2 is `m+1` rollouts per replan; L3 needs the distilled head.
   Report wall-clock per decision against DINO-WM and LeWM.

---

## 10. First three jobs (nothing else is submitted until E0 reads)

1. `sbatch` Gramian probe on Scene replan states: `J` by finite differences on the frozen
   LeWM, spectrum + `kappa` + `theta` + `alpha`, plus the simulator Jacobian on the same
   snapshots.
2. Same probe on OGBench-Cube and on Reacher/TwoRoom (the predicted-null cells).
3. Retrodiction pass, CPU-only, on existing MetaWorld oracle-ladder artifacts: does
   `cos(theta)/kappa` order {reach, drawer-close} above {push, pick, button-press,
   window-close}?

E0 must be locked in its own protocol document, with the STOP thresholds written before
job 1 is submitted.
