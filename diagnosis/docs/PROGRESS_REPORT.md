# Progress Report — CAI-JEPA: Diagnosing Action-Grounding Failures in JEPA World Models

**Author:** Nhat
**Date:** 2026-06-10
**Status:** Diagnostic complete (CONDITIONAL_GO) · Mechanism identified · Working fix demonstrated on Metaworld

---

## 1. Research question

Action-conditioned JEPA world models (DINO-WM, V-JEPA-2-AC, JEPA-WM) are increasingly
used for robot planning. The concern motivating this project is that these models may be
**action-grounding-blind**: they produce nearly identical latent predictions for different
actions taken from the same state, especially in contact-rich manipulation. If true, this
would silently break planning exactly where precision matters most (grasping, contact).

This phase was a **go/no-go diagnostic**: measure whether the failure is real and where it
concentrates, before committing to a full paper. The study runs entirely on **pretrained,
frozen checkpoints** — nothing about the baselines is trained or fine-tuned, so any failure
we find is a property of the published models, not our setup.

## 2. Method (one paragraph)

We encode every frame once into the model's own latent space and cache it, then run all
metrics on the cache. Each transition is stratified into a **regime** (`free_space`,
`pre_grasp`, `gripper_actuation`, `contact_manipulation`) so we can ask *where* failures
live. We test action-grounding with **Counterfactual Ranking Accuracy (CRA)**: can the model
rank the true action above alternatives, given the same state? The strict version pairs the
counterfactual action with a **similar latent state** (`hard_nn`) — this is the honest test.
All confidence intervals are trajectory-clustered bootstraps, and the pipeline is validated
with synthetic models and a 68-test offline suite plus published-number sanity checks.

Datasets: **Metaworld** (primary, simulated Franka manipulation) and **DROID** (secondary,
real-robot transfer check).

## 3. Initial result — the failure is real (Decision: CONDITIONAL_GO)

**The headline finding:** the models react fine to *gross* action changes but break down on
*subtle, similar-state* counterfactuals — precisely in the pre-grasp and contact regimes.

On Metaworld (CRA on transitions whose latent actually changed, `CRA_eff`):

| Negative type | free_space | **pre_grasp** | contact_manipulation |
|---|---|---|---|
| `opposite` (easy) | 0.99 | 0.97 | 0.98 |
| `hard_nn` (strict) | 0.60 | **0.49** | 0.53 |

The gap between the easy and strict columns is the pathology: a model that truly grounded
actions would not collapse from 0.97 to 0.49 just because the alternative action is paired
with a nearby state. `jepa_wm` is consistently stronger than `dino_wm`, but both lose margin.

On **DROID** (real robot), the strict test falls to the **16-way chance floor** in the
gripper and contact regimes (`CRA_eff` ≈ 0.04–0.07) — the sharpest action-grounding failure
we observed.

**Decision logic:** Metaworld is clearly below the abandon threshold but not strong enough
for a full GO on its own; DROID is severely pathological. → **CONDITIONAL_GO**: the
phenomenon is real and worth pursuing, with the caveats below.

## 4. Sharpening the thesis — "Boundary Blindness"

CRA tells us the models stop *distinguishing* similar-state counterfactual actions, but not
*why it matters*. We reframed the thesis around **contact bifurcations**: at a grasp
boundary, a tiny action difference selects between two qualitatively different futures
("lift" vs "slip"). We defined **Boundary Blindness (BB)** = how much the spread of the
model's predicted futures falls short of the spread of the *true* futures over nearby
actions, restricted to bifurcation-like transitions.

**BB gate result (PASS, both datasets):** the blindness concentrates exactly at the
**grasp/approach (pre_grasp) boundary**, not post-contact manipulation:

- **Metaworld** `bb_boundary`: pre_grasp **1.32 / 1.28** (dino/jepa) vs free_space 0.28 / 0.30 — ~4.5× elevated, CI-confirmed per task.
- **DROID** (transfer proxy): pre_grasp **1.98** vs free_space 0.72 — ~2.7× elevated, CI-confirmed.

This gives the project a precise, falsifiable target: *the models smear the grasp boundary.*

## 5. Why it happens — a measured mechanism

We then asked whether this is fixable with the obvious knobs, and the answer was a clean
**no**, which turned out to be the most informative result:

1. **It is structural, not a training shortfall.** Every baseline predictor is a point
   estimator trained with L2, whose optimum is the *conditional mean*. At a bifurcation the
   true future is bimodal, so the mean is an average future that never happens and barely
   moves as the action crosses the boundary — exactly the high-BB signature.
2. **Predictor-side multimodality alone does not help.** We trained mixture-density heads
   (K=2–4, several objectives) on the frozen latents: BB did not move. The probes told us
   why — the object bifurcation that is large in physical space occupies a latent subspace
   ~10× smaller than the prediction residual (9.9 vs 106 L2 units), so the latent's own
   metric barely registers it.
3. **Re-weighting the metric alone does not help either** — it just moves BB from one regime
   to another. You cannot expose a signal the model does not produce.

A probe chain localized the bottleneck precisely: the latent *contains* object position
(V1 ✓) and *propagates* it under the factual action (V2 ✓), but does **not** respond to
**counterfactual** actions (V3 ✗, spread correlation +0.035 ≈ 0).

## 6. A fix that works (on Metaworld)

The diagnosis pointed at the training objective, not capacity or the metric: full-latent L2
buries the object's action-dependence. So we added a **0.5M-parameter grounded dynamics
channel** `h(z, a) → Δobject` — trained on the *same cached data* with the object
displacement as the explicit target, encoder/predictor/data all frozen, ~25 min on a 12 GB GPU.

Results:

- **Counterfactual tracking restored:** spread correlation with the true outcome jumps from
  **+0.035 → +0.682** (a ~20× improvement in boundary tracking).
- **Boundary Blindness halved at the boundary:** pre_grasp `bb_boundary` **1.32 → 0.66 (−50%)**;
  the pre_grasp-vs-free_space gap collapses from **1.04 → 0.32**.
- **Plugs into the planner** (CEM) as a grounded cost term with a clean hook. Open-loop
  planning A/B shows **no harm** (and a slight lean the right way in contact regimes);
  closed-loop success-rate validation is the declared next experiment.

This is the cleanest possible story: a tightly isolated, frozen-trunk intervention moves the
exact metric the diagnosis predicted, and nothing else changed.

## 7. Honest limitations / caveats

- **Metaworld boundary labels are an object-displacement proxy** (the public dataset has no
  MuJoCo contact ground truth); DROID outcomes use a weaker latent-change proxy, so **DROID
  is a transfer check, not the proof** — the proof lives on Metaworld.
- One articulated task (`mw-door-close`) confounds the displacement proxy and is excluded
  from pooled numbers (disclosed).
- **`vjepa2_ac_droid` not yet run** — it needs ~24 GB VRAM (server A5000); the rest runs on
  a local 12 GB GPU. This is the main piece of remaining baseline coverage.
- The fix's *principle* is model-agnostic, but its *training recipe* uses Metaworld sim
  state; real-robot transfer of the recipe is future work.

## 8. Status summary & next steps

**Done:**
- ✅ Diagnostic pipeline (frozen checkpoints, 68 offline tests, synthetic validation).
- ✅ Metaworld + DROID CRA diagnostic → **CONDITIONAL_GO**, failure localized to pre-grasp/contact.
- ✅ Boundary Blindness gate → **PASS**, locus = grasp boundary, both datasets.
- ✅ Mechanism measured (latent metric compression + missing counterfactual signal).
- ✅ Working fix on Metaworld (grounded dynamics channel: BB −50% at the boundary).

**Next:**
1. **Closed-loop planning evaluation** with the grounded cost (environment rollouts → success rate) — the decisive planning endpoint.
2. **Run `vjepa2_ac_droid`** on the server to complete baseline coverage.
3. Write up: diagnosis (CRA + BB) + mechanism + the grounded-channel fix as the paper's contribution.

**Artifacts:** `diagnosis/results/decision_report.md` (full tables/CIs),
`diagnosis/docs/FIX_C1_EXPLAINER.md` (mechanism + fix narrative),
`diagnosis/docs/METHODOLOGY.md` (concepts + code map).
