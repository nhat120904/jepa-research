# Inverse action proposal — fix the SAMPLE side end-to-end (planner-side, WAV-style)

**Date:** 2026-06-24 · **Status:** design
**Builds on:** `docs/plans/2026-06-23-contact-aware-action-proposal-design.md`
(lever #2 — BC-seeded CEM, **designed but not yet run**), `results/closed_loop_report.md`
(hdyn +0.089 but 0/16; arm reaches ee 2–4 cm, object never moves),
`results/grounded_explore_report.md` (hexp null), `scripts/17` (dyn-head),
`scripts/22` (spatial object probe, 2.1 cm), `scripts/19` (ee probe),
`scripts/18` (closed-loop CEM), `planning/cem_planner.py`. **External anchor
(verified, read directly):** `world_model/world-action-verifier.pdf` — Liu, Feng,
Kong, … Murphy, Finn, Du (forward-inverse asymmetry; sparse inverse over
action-relevant features; Eq. 5 mask `M`; Prop. 3.1).

## 1. Why this leg exists — the bottleneck is sampling, not scoring

Every model-side, scoring-only, frozen lever is exhausted and the diagnosis is
precise: the dyn-head *scores* a contact-creating rollout better (cf-corr +0.68;
closed-loop cost +0.089 [+0.022, +0.162]), yet success stays **0/16**. The signature
is decisive — CEM finds a low-cost arm-at-goal basin **without ever making contact**;
zero-mean Gaussian shooting essentially never samples the reach→grasp→lift sequence
that would let the grounded term grade a contact. The residual is no longer "can the
model tell good contact from bad" — it is **"does the planner ever propose a contact
at all."** This is the planner-side half V-JEPA-2-AC papers over by hand-feeding three
sub-goal images.

## 2. The idea (WAV, adapted): propose contacts with a *sparse inverse model*

WAV's key result: a **sparse inverse dynamics model** over a *subset of
action-relevant state features* (end-effector pose, manipulated-object motion) is far
more reliable than the forward model, because (i) action-free data over-covers
plausible futures and (ii) the action is identifiable from a low-dimensional,
agent-centric subspace (Sec. 2.2, Eq. 5: `â = h_ψ(M⊙sᵗ, M⊙sᵗ⁺¹)`; Prop. 3.1: the
inverse generalises to compositional out-of-support transitions where the *forward*
model fails). **This is exactly our situation:** the forward predictor's
counterfactual object channel is dead, but the object/ee subspace is provably
decodable (spatial probe 2.1 cm, ee probe). So an inverse over that subspace can
*propose* the contact-creating action the forward rollout could never search to.

### The mechanism
Train an inverse proposal `h_inv(z_t, sub-goal_obj) → â` over the **object/ee-relevant
latent subspace** (mask `M` = the features the spatial/ee probes read), supervised on
cached `(z_t, obj_t, obj_{t+1}, a_t)` triples — i.e. "what action moved the object
from here to there." At plan time:

1. **Sub-goal generation (object-relative waypoints).** From the spatial-probe object
   position, decompose pick-place into `reach → grasp → lift`: ee→object, then
   object→goal. Each is a *low-BB* segment (the regimes where the model is reliable).
   This automates the hand-crafted sub-goals V-JEPA-2-AC supplies manually, derived
   from the probed object instead of human-picked images.
2. **Inverse proposal → CEM seed.** For the current sub-goal, `h_inv` proposes an
   action chunk that drives ee/object toward it; use it to **initialise the CEM mean**
   (and seed a fraction `p_seed` of first-iteration samples). CEM then *refines* a
   proposal that already reaches toward and closes on the object, instead of searching
   from a zero-mean Gaussian that never proposes contact.
3. **Scoring stays = lever #1 / the object-aware predictor.** Cost = upstream latent
   L2 + grounded object term (dyn-head, or the corrected rollout from
   `2026-06-24-object-aware-predictor-design.md`). The inverse fixes **sampling**; the
   model fixes **scoring**.

New script `28_train_inverse_proposal.py` (`h_inv` over the probe subspace,
cache-only). New CEM arm `l2inv` / `hdyninv` in `scripts/18` — same cost as `l2`/`hdyn`,
only the CEM init changes (`cem_plan(..., init_mean=h_inv_rollout, seed_frac=p_seed)`).
Paired against `l2`/`hdyn` on the same env/seeds.

### Rungs (cheapest-first, from lever #2)
- **A (primary): inverse / BC-seeded CEM** — `h_inv` (WAV-style, over the probe
  subspace) is the principled upgrade of lever #2's `π_BC`; if a plain BC policy is
  faster to stand up, run it first as the cheap variant, then swap in `h_inv`.
- **B: explicit sub-goal decomposition** — plan each `reach/grasp/lift` segment to its
  object-relative intermediate goal (the §2.1 waypoints) instead of one end-goal.
- **C: scripted contact primitive (control/upper-bound)** — seed a fixed "move ee to
  probed object, close gripper" chunk. If even C is 0/16, the predictor rollout (not
  proposal) is the wall → the object-aware-predictor leg owns it.

## 3. What stays fixed (reuse, do not reinvent)

- Cost surface = lever #1: spatial probe (`spatial_object_probe_dino_wm_metaworld.pt`,
  2 cm) for goal/init object + dyn-head object term (β object-dominant); or the
  corrected rollout from the object-aware-predictor leg.
- Protocol = `scripts/18` closed-loop: 16 (→ ≥24, Part 3) paired episodes/task, horizon
  6, 300 samples, 15 iters, 3 stepped, ≤100 env steps, paired env+seed across arms.
- Encoder + base predictor frozen. Only `h_inv` (and optionally `π_BC`) trains,
  cache-only.

## 4. Evaluation & decision

- **Primary:** task success on `mw-push` + `mw-pick-place`, inverse-seeded arm vs `l2`
  and vs `hdyn`, paired; **no-harm** on `mw-reach` (must stay ≥ baseline, 37.5%/50.0%).
- **Secondary (the direct readout of whether proposal fixed sampling):** fraction of
  episodes that **make contact at all** (object displaced > τ). If this jumps but
  success does not, scoring (the model leg) is the remaining wall.
- **GO (success moves):** BB (model) + proposal (planner) jointly clear the contact
  wall → the complete diagnosis+solution story.
- **NULL (contact now proposed but still 0/16):** the predictor mis-scores even
  proposed contacts → compose with the object-aware-predictor leg (run both arms:
  inverse-seed **and** corrected rollout).

## 5. Risks

- **Prior wash-out:** 15 CEM iterations may erode the seed back to the arm-at-goal
  basin → keep `p_seed` on later iterations / elite injection, or fewer iterations.
- **Inverse OOS:** `h_inv` trained on expert object motion may mispredict on
  off-distribution CEM states → use it as a *seed*, not a hard constraint; CEM owns
  refinement (WAV Prop. 3.1 argues the *agent-side* subspace stays on-support even when
  the full scene transition is novel — the favourable case).
- **Predictor still wrong on contact rollouts** (the hexp failure mode): a correctly
  *proposed* contact may be mis-imagined by the frozen forward predictor → this is
  precisely why the two legs are run **together**; the object-aware predictor supplies
  the corrected rollout the proposal is scored against.

## 6. Scope

Planner-side, frozen encoder + base predictor. Run in parallel with the
object-aware-predictor leg; the composed `{l2, +predictor, +proposal, +both}` matrix
on the contact tasks (no-harm reach) is the paper's end-to-end result either way
(success flip = solution; clean isolation of which half remains = honest diagnosis).
