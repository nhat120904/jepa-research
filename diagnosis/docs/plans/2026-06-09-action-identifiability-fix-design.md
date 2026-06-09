# Design — Action-identifiability "fix" leg, reframed around contact boundaries

**Date:** 2026-06-09
**Status:** proposal / design (no code beyond `scripts/inspect_droid_observation_keys.py` yet)
**Supersedes (for the "fix" contribution):** the framing of Contribution 3 in
`cai_jepa_paper_proposal.md §6` as a *one-step counterfactual/contrastive loss*.

## 0. TL;DR

The diagnostic (Contributions 1+2) stands. The **fix** leg is reframed after two
findings:

1. **A critique that kills the easy fixes.** Classifier-free action guidance (CFAG)
   and the one-step counterfactual margin loss both operate on `F(z,a) − F(z,a')`.
   They *amplify* sensitivity the model already has; they cannot *create* the
   resolution of a sharp action→outcome boundary (gripper centred → lift vs. lifted
   2–3° off → no lift). The real gap is **high-sensitivity / bifurcation regimes**,
   not "weak effect."
2. **A data reality that kills the strongest grounding fix.** The contact-force
   signal that would most cleanly resolve such boundaries **does not exist in our
   DROID pipeline** (verified — §2).

Reframed gap statement for the paper:

> Existing JEPA world models are not merely weak at amplifying action effects; they
> **fail to model high-sensitivity action regimes, where small action perturbations
> near contact boundaries produce qualitatively different futures.** Unimodal latent
> prediction (point prediction + L2) provably averages across such outcome
> bifurcations, and vision-only latents may not even resolve the boundary-relevant
> state.

Two distinct failure points, each needing its own fix:

| | Failure point | Fix | Needs force? | Feasible now? |
|---|---|---|---|---|
| (a) | Latent doesn't *resolve* boundary state | **D**: state-grounded latent | force = no; pose+gripper = yes | Metaworld: yes; DROID: weak |
| (b) | Predictor too *smooth* to represent a step / unimodal averaging | **C1**: distributional/multimodal prediction + boundary supervision | no | yes |

**Decision:** hero contribution = **C1** (force-free, attacks (b), works on cached
latents). Supporting = **D restricted to Metaworld** (attacks (a) where we have the
state to define and supervise the boundary). New **boundary diagnostic** (§3) ships
first because it runs on frozen baselines and de-risks everything.

---

## 1. Why CFAG and one-step contrastive loss are insufficient (the critique, recorded)

For action `a` (centred grasp) vs `a'` (2–3° off) with opposite outcomes:

- If `F(z,a) ≈ F(z,a')` the model has already collapsed the distinction. CFAG's
  guidance vector `γ·(F(z,a) − F(z,∅))` is then ~0 or mis-oriented → amplifying it is
  useless or harmful.
- The counterfactual margin loss `max(0, m − d(F(z,a'),z_{t+1}) + d(F(z,a),z_{t+1}))`
  has ∂/∂a ≈ 0 in exactly this regime — there is nothing to push on if `z`/`F` carry
  no boundary information.

So the critique applies to **both** the test-time and the originally-proposed
loss-level fix. The boundary is sharp in *outcome* space but the immediate `z_{t+1}`
after a small perturbation can be near-continuous; the divergence often only appears
**over the rollout** (the cup slips over several steps — sensitive dependence). This
is why one-step methods miss it and is the deeper reason to move to a
distributional, boundary-aware, multi-step-aware formulation.

---

## 2. Data reality — verified (this constrains everything)

Verified directly in the upstream loader that produced our latent caches:

- `droid_dset.py:259-265` (HF) and `:316-322` (decord): DROID `state` == `proprio` is
  built as `concat(cartesian_position[6], gripper_position[1]) → [T, 7]`. `obs["proprio"]
  = states` (line 186).
- `grep -i 'force|torque|wrench|joint'` over `droid_dset.py` → **no matches**. The
  loader never reads joint state and there is no force/torque channel.
- Config agrees: `diagnostic_droid.yaml` `action_dim: 7`, `GRIPPER_IDX["droid"]=6`.
- Caches (`droid__*.h5`) are produced through this loader → **cache proprio is 7-dim,
  no force, no joint.**

**Conclusion:** direction D's strongest form (contact-force-grounded latent) is **not
possible on DROID**. Open question handed to `scripts/inspect_droid_observation_keys.py`:
do raw episodes expose `joint_position/velocity` (richer proprio, still not force)?
That script's output decides whether DROID-D gets pose+gripper only, or pose+gripper+joints.

**What we DO have, per dataset:**

| Dataset | Grounding available | Boundary LABEL available? | Status |
|---|---|---|---|
| **Metaworld** (primary, **done**) | full 39-dim state: ee + object positions | **yes** — object displacement = grasp success | cached, diagnostic complete |
| **DROID** (secondary) | ee-pose(6) + continuous gripper-width(1) [+ joints?] | **no** — real-world, no object GT | cached |

Implication: **Metaworld is the correct primary testbed for the boundary work** — it
is the only dataset where we can *define* the boundary (object-z threshold), *supervise*
it, and *measure* sensitivity against ground truth. DROID becomes a "does the
mechanism transfer to real, vision-grounded data" check with weaker grounding and no
boundary label.

---

## 3. Diagnostic extension: a boundary regime + sensitivity-mismatch metric (ships first)

Runs on **frozen baselines**, no training — extends CounterfactualBench, reuses
existing machinery, de-risks the whole leg by *proving the boundary gap is real and
measurable* before we build any fix.

### 3.1 Boundary regime selection

Reuse the existing `hard_effect` negative machinery (`diagnostic_droid.yaml` already
defines it: similar state, most-different *true* `Δz`, action-penalty weighting). A
transition `(z_t, a_t)` is in the **boundary regime** if, among its similar-state
neighbours (the `hard_effect` pool: `‖z_t − z_{t'}‖ < ρ`), the *true* outcome is
**bimodal under small action differences** — i.e. some near-by actions produce large
effect and some near-zero. Operationally on Metaworld, "effect/outcome" = object
displacement from the 39-dim state (not just `‖Δz‖`), which is the contact proxy the
stratifier already uses.

Concretely: boundary score = (spread of true outcome over the small-action-difference
neighbourhood) normalised by (spread of the action differences). High = a small action
change flips the outcome = a bifurcation.

### 3.2 Sensitivity-mismatch metric (the new number)

For each boundary transition, over the same neighbourhood of near-by actions
`{a_t'}`:

- `S_true`  = spread (var / range) of the **true** outcome (object Δ on Metaworld;
  `‖Δz‖` proxy on DROID).
- `S_model` = spread of the **model's** predicted `F(z_t, a_t')` over the same actions.
- **Boundary Blindness** `BB = relu(S_true_norm − S_model_norm)` (both standardised).

`BB ≈ 0` → model tracks the true local sensitivity. `BB` large → model smooths over
the bifurcation. This is the quantitative form of the reframed gap statement and is a
*contribution on its own* (it is what CRA/ECS cannot see — they test "distinguishes
actions at all," not "resolves the sharp boundary").

**Predicted result (the thesis):** baselines show high `BB` concentrated in
pre-grasp/gripper-actuation boundary transitions, even where aggregate CRA looks ok.

---

## 4. Hero fix — C1: distributional / multimodal latent prediction + boundary supervision

Attacks failure point (b): a unimodal `F` + L2 is *forced* to predict the mean of a
bimodal future near the boundary → the smear. Replace the point head with a small
distributional head so the predictor can represent "lift OR not-lift."

### 4.1 Mechanism

- Keep the frozen encoder and the base predictor trunk (DINO-WM scale on Metaworld is
  cheap on the A5000). Replace/augment the output head with one of:
  - **Mixture-density head (default):** predict `K` components `{π_k, μ_k, σ_k}` over
    `z_{t+1}` (small `K`, e.g. 2–4). NLL training. Near a boundary the head can go
    bimodal; away from it, it collapses to one component.
  - (Ablation) a lightweight conditional flow / a small diffusion head on the latent.
- **Boundary-supervision auxiliary head:** predict a sharp boundary event
  `g_{t+1}` (grasp-success / object-moves indicator from Metaworld state, or a
  continuous "graspability margin"). This forces capacity onto the boundary instead of
  letting it be smoothed. Loss `+ λ·L_boundary` (BCE/regression).
- (Optional ablation) **sensitivity supervision**: penalise mismatch between the head's
  local `∂μ/∂a` and `S_true` estimated from data (directly fights spectral bias).

### 4.2 Total objective

```
L = L_pred(NLL of z_{t+1} under the mixture)            # replaces L2 point loss
  + λ_b · L_boundary(g_{t+1})                            # sharp-event head
  + λ_s · L_sensitivity   (optional ablation)
```

### 4.3 Planning / diagnostic integration

- **Diagnostic:** `BB` (§3.2) should drop sharply for C1 vs baseline in boundary
  regimes — the headline figure.
- **Planning (CEM):** score candidates by mixture mode, or by NLL of the goal under
  the predicted distribution. The multimodal future means CEM no longer optimises
  against an averaged-out (action-insensitive) cost surface in boundary regimes.

### 4.4 Why this is the right hero (vs the original contrastive loss)

- **Force-free** → unaffected by the §2 data limitation.
- Runs on **cached latents** (Metaworld done; DROID cached) — only the small predictor
  head trains.
- Novel angle for JEPA-WM: "**unimodal latent prediction is structurally incapable of
  representing contact bifurcations; distributional latent prediction restores it.**"
  Clean, falsifiable, top-venue-shaped.

---

## 5. Supporting fix — D restricted to Metaworld: state-grounded latent

Attacks failure point (a): give the latent the state needed to *resolve* the boundary.

- **Augmented latent** `z̃_t = [z_t^vis ‖ φ(p_t)]`, `φ` = small MLP. On Metaworld
  `p_t` = the informative slice of the 39-dim state (ee pose, **ee–object relative
  geometry**, gripper). The cache already stores `state`/`proprio`, so this is a
  re-wire, **no re-encoding of visual latents**.
- Predictor predicts `z̃_{t+1}` incl. the proprio/geometry channel, where the boundary
  is sharp.
- Composes with C1's boundary head (boundary label from object displacement).
- **DROID transfer:** only `[ee-pose ‖ gripper-width (‖ joints if §2 script finds them)]`
  — no object state, no boundary label. So DROID-D is a *partial* transfer check, not
  where the boundary claim is proven. Be explicit about this in the paper to avoid
  over-claiming tactile/force grounding (a reviewer who checks the DROID schema will
  catch it).

D alone is **not** sufficient (a resolving latent still gets smoothed by a unimodal
predictor) — it is the (a)-side complement to C1's (b)-side fix. Best result is
expected from **C1 + D on Metaworld**.

---

## 6. Experimental plan (de-risked order)

1. **Boundary diagnostic on frozen baselines (Metaworld first).** Implement §3, show
   high `BB` in boundary regimes. *No training.* This alone strengthens Contributions
   1+2 and validates the reframed gap. **Gate:** if `BB` is not elevated in boundary
   regimes, the whole reframing is wrong — stop and reconsider before building fixes.
2. **C1 on Metaworld.** Train mixture-density predictor + boundary head on cached
   latents. Measure `BB` drop and planning improvement vs baseline + vs the original
   one-step contrastive loss (now a *baseline* for the fix, not the hero).
3. **C1 + D on Metaworld.** Add state-grounded latent; ablate D's contribution.
4. **Transfer to DROID.** C1 on DROID latents (force-free, so unaffected by §2);
   D-on-DROID limited to pose+gripper(+joints). Report as transfer, not as the
   boundary proof.
5. Ablations: mixture vs flow vs diffusion head; `K`; with/without boundary head;
   with/without sensitivity supervision; C1-only vs D-only vs C1+D.

## 7. Risks / open questions

- **R1 (data, §2):** force unavailable; boundary label unavailable on DROID. *Mitigation:*
  Metaworld is primary for the boundary claim; DROID is transfer only. **Pending:**
  `inspect_droid_observation_keys.py` output decides if joints enrich DROID-D.
- **R2:** Metaworld is a sim HF dataset (CLAUDE.md: no MuJoCo contact GT, object-
  displacement proxy). The boundary label is a proxy, not true contact. *Mitigation:*
  it is a clean *outcome* label (object moved or not), which is exactly the bifurcation
  we care about; state the proxy explicitly.
- **R3:** mixture head may collapse to one component (mode-averaging returns).
  *Mitigation:* boundary-supervision head + monitor component usage / entropy; this is
  itself an ablation result.
- **R4:** distributional prediction complicates CEM. *Mitigation:* start with mode-based
  scoring (drop-in), add NLL-of-goal scoring as an extension.

## 8. Concrete next code steps (when approved)

- `scripts/inspect_droid_observation_keys.py` — **done**, run on server (§2 open Q).
- `stratification/boundary_regime.py` + extend `04_classify_regimes.py` — boundary
  regime selection (§3.1), reusing `hard_effect` pooling.
- `metrics/boundary_blindness.py` + wire into `05_run_diagnostic.py` — `BB` metric (§3.2).
- `models/heads/mixture_predictor.py` + a `train_predictor_head.py` — C1 (§4), Metaworld
  latents first.
- latent-augmentation path in the predictor input — D (§5), Metaworld state slice.
