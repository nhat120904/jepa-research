# The C1 fix, explained — why a mixture head un-blinds a boundary-blind world model

**Audience:** anyone who read `PAPER_IDEA.md` / `PROJECT_OVERVIEW_VI.md` and wants to
understand *why* this particular fix follows from the diagnosis, not just what the
code does. Code: `models/heads/mixture_predictor.py`, `scripts/train_predictor_head.py`,
`scripts/13_eval_fix_boundary.py`. Tests: `tests/test_mixture_predictor.py`.

---

## 1. The problem we measured (recap of the diagnosis)

Our diagnostic asked two questions of frozen JEPA world models and got two answers:

1. **"Do you use the action at all?"** (CRA/CRA_eff) — *Mostly, but it degrades*:
   with similar-state hard negatives, CRA collapses from ~0.97–0.99 (`opposite`) to
   ~0.46–0.57 in pre-grasp/contact regimes on Metaworld, and to the chance floor on
   DROID.
2. **"Do you resolve sharp action→outcome boundaries?"** (Boundary Blindness, the
   2026-06-10 gate — **PASS**) — *No*: at bifurcation-like transitions, the spread
   of the model's predicted futures collapses relative to the spread of the true
   outcomes. `bb_boundary` pooled: pre_grasp **1.32/1.28** (dino/jepa) vs free_space
   0.28/0.30 on Metaworld; pre_grasp **1.98** vs 0.72 on DROID (transfer, ‖Δz‖ proxy).

Concretely, the failure lives at the **grasp boundary**: gripper centred on the
handle → the cup lifts; 2–3° off → it slips. Two *qualitatively different futures*,
selected by a *tiny* action difference. The diagnostic says: at exactly these
transitions, the frozen models predict ~the same future for every nearby action.

## 2. Why this failure is structural, not a training shortfall

This is the key insight that dictates the fix.

Every baseline predictor is a **point estimator trained with L2**: it outputs one
latent `ẑ` and minimizes `‖ẑ − z_{t+1}‖²`. A classical fact: the minimizer of
expected L2 error is the **conditional mean** `E[z_{t+1} | z_t, a]`.

Away from boundaries that's fine — the future is unimodal, the mean is the future.
But *at* a bifurcation the conditional distribution is **bimodal**: given the
state and a near-boundary action, the future is "lift" with some probability and
"no-lift" otherwise (the latent does not perfectly resolve which side you're on, and
the dynamics are sensitive). The L2-optimal prediction is then the **average of
"lift" and "no-lift"** — a future that never happens, sitting between the modes. And
because that average moves only slightly as the action crosses the boundary, the
prediction is nearly **action-insensitive** right where action sensitivity matters
most. That is precisely the high-BB signature we measured.

Two corollaries the design records (design doc §1):

- **More training cannot fix it.** A *perfectly* trained point+L2 predictor still
  outputs the conditional mean. The failure is in the function class, not the fit.
- **The "obvious" fixes cannot work either.** Classifier-free action guidance and
  the one-step counterfactual margin loss both amplify `F(z,a) − F(z,a′)`. At the
  boundary that difference is ≈ 0 (both predictions are the same smear), so there
  is nothing to amplify and no gradient to push on. You cannot sharpen a boundary
  the representation has already averaged away.

## 3. The fix: let the predictor say "lift OR no-lift" instead of their average

If the disease is "one point forced to summarize two futures", the cure is to let
the predictor output **a small distribution over futures**:

```
F(z_t, a)  →  { (π_k(z_t,a),  μ_k(z_t,a),  σ_k(z_t,a)) }  k = 1..K     (K ≈ 2–4)
```

a **mixture-density head** trained with negative log-likelihood (NLL) instead of L2.
Why this changes everything at the boundary:

- NLL, unlike L2, **punishes the smear**: a Gaussian centred between two modes
  assigns low likelihood to both real futures; the head earns strictly better NLL
  by placing one component on "lift", one on "no-lift" (our synthetic test shows a
  >1-nat gap; unimodal K=1 must instead inflate σ to cover both modes).
- The mixture weights `π_k(z_t, a)` are a function of the **action**: as `a` crosses
  the boundary, probability mass flows from the "no-lift" component to the "lift"
  component. The most-likely-component ("mode") prediction therefore **switches
  discontinuously** with the action.
- That switching is *exactly* the quantity BB measures (`S_model` = spread of
  predictions over a neighbourhood of nearby actions). A point predictor's spread is
  ~0 at the boundary; the mixture's mode jumps between futures → `S_model` rises to
  match `S_true` → **BB falls**. The fix and the metric are two sides of the same
  definition, which is what makes the claim falsifiable.

### 3.1 The boundary-supervision head (the second ingredient)

A mixture can still cheat: collapse to one component everywhere (risk R3 — mode
averaging returns by the back door, since most transitions are *not* boundaries and
a single component is cheapest for them). We counter it with an **auxiliary boundary
head**: from the same shared features the mixture parameters use, predict the sharp
event `g_{t+1}` = "the object moves this step" (from Metaworld's state — an
object-displacement proxy, no contact GT, stated everywhere). The BCE loss forces
the *shared* representation to encode "how close to the boundary am I", which is
precisely the feature `π_k` needs in order to know *when* to split. Total loss
(design §4.2):

```
L = NLL(z_{t+1} | mixture)  +  λ_b · BCE(g_{t+1})
```

### 3.2 What stays frozen, and why that is the point

The encoder and the entire base predictor stay **frozen**; the trained part is a
~10M-parameter residual head: `μ_k = ẑ_base + Δ_k(z_t, ẑ_base, a)`. Three reasons:

1. **Attribution.** If BB drops, the only changed ingredient is the output
   *representation* (point → mixture). Nothing else moved — no new data, no
   encoder finetune. That isolates the paper's claim: *unimodal latent prediction is
   structurally incapable of representing contact bifurcations; distributional
   prediction restores it*.
2. **The K=1 control.** We train an identical head with K=1 in the same pass on the
   same batches. K=1 *with* the boundary head and the same capacity is the ablation
   that rules out "it was just extra parameters / the auxiliary loss": only K≥2 can
   represent two futures.
3. **Cost / data reality.** It trains on the already-cached latents in ~minutes on a
   12 GB GPU ("force-free hero") — unaffected by DROID's missing force channels, and
   no visual re-encoding.

### 3.3 How a planner consumes it

CEM currently scores action candidates by L2 distance of the *point* prediction to
the goal — at boundaries that cost surface is flat (averaged futures ≈ equally far
from the goal), so the planner picks noise; this is the planning failure the probe
associated with contact regimes. With the mixture head, CEM scores by the **mode**
(drop-in, what our adapter exposes) or by **NLL of the goal under the mixture**: a
candidate action whose "lift" component matches the goal scores sharply better than
one 2° off. The cost surface regains exactly the action sensitivity the boundary
demands.

## 4. How we verify it (pre-registered, falsifiable)

1. **Synthetic mechanism proof** (`tests/test_mixture_predictor.py`, offline, part of
   the 62-test suite): in a constructed bifurcating world with an action-ignoring
   frozen trunk — the hardest case the §2 critique allows — the K=2 head (a) beats
   K=1 NLL decisively, (b) mode-switches across the boundary, and (c) **drops BB
   through the production BB code path** while the frozen base stays blind. The
   boundary head separates boundary from smooth states.
2. **The real gate** (`scripts/13_eval_fix_boundary.py`): re-run the *identical*
   boundary diagnostic (same cache, same seed-0 neighbour pool, same cells, same
   per-model standardisation) on three variants: frozen base, base+K1 head,
   base+K≥2 head. **Success =** the K≥2 row shows materially lower `bb_boundary`
   than both base and K1 in `pre_grasp` (the measured locus) — and not by
   degrading elsewhere. **Failure =** K≥2 ≈ K1 ≈ base (the head collapsed → the
   distributional claim dies, report as such).
3. (Next, C2 recast) the planning probe re-run with mode/NLL scoring: Action Error
   in boundary regimes should drop relative to the frozen baseline planner.

## 5. Honest limitations

- **The boundary label is a proxy** (object displacement from sim state; no MuJoCo
  contact GT). The `mw-door-close` anomaly in the gate run shows what this proxy
  does on articulated objects; it is excluded from pooled numbers and disclosed.
- **One-step formulation.** The critique notes the divergence often grows over the
  rollout; C1 attacks the one-step conditional. The mixture's mode-switching does
  propagate through autoregressive rollouts (each step re-selects a mode), but we
  make no multi-step distributional claim yet.
- **Frozen trunk ceiling.** If the *encoder's latent* does not contain the
  boundary-relevant state at all (failure point (a)), no head can recover it — that
  is direction D (state-grounded latent, Metaworld-only), deliberately separate.
- **DROID is transfer-only** for any of this (pose+gripper proprio, no force, no
  object GT, ‖Δz‖ outcome proxy); the boundary *proof* lives on Metaworld.
- Mode-based evaluation is a choice; reporting could also use expected-NLL. Mode is
  what the planner consumes, so it is the planning-relevant read.

## 6. Results (measured, 2026-06-10, `dino_wm_metaworld`, local RTX 5070)

Four head variants were trained on the cached latents (frozen trunk, 12,312 train /
1,368 val transitions, trajectory split, 3 epochs each) and probed for the two
quantities the mechanism needs: **component-mean separation** (can the mixture
represent two futures?) and **π action-flip rate** (does the action select between
them?). Summary — **the fix did not move BB**, and the probes say *why*, precisely:

| variant | val NLL (K≥2 / K1) | μ separation (median L2) | π flips w/ action |
|---|---|---|---|
| C1, soft NLL (K=3) | 31,531 / 33,177 | 6.8–16.7 | 0.000 |
| C1, WTA hard-EM (K=3) | 31,536 / 33,169 | 6.1–12.2 | 0.000 |
| C1+D, WTA + state slice (K=3) | 31,431 / 33,143 | 6.4–16.4 | 0.006 |
| C1+D, **supervised assignment** (K=2) | 33,181 / 33,305 | **9.9** | 0.000 |

Reference scales: median true step ‖z_{t+1}−z_t‖ = **170.4**; median base residual
‖ẑ_base−z_{t+1}‖ = **106.1**. BB before/after through the production pipeline
(identical pool/cells): **unchanged to ~3 decimals** for the variants run through it —
soft-NLL K1/K3 (`results/metaworld_boundary_fix_nll.csv`, incl. the frozen-base
reproduction) and supervised K1/K2+state (`results/metaworld_boundary_fix.csv`).
The two WTA variants were probe-screened instead (μ separation and π flip rate
indistinguishable from the NLL variant — the quantities BB's S_model is a function
of) and the ~25-min BB pass was skipped for them.

### 6.1 Why — two measured causes, and what they mean

**(1) The bifurcation is nearly invisible in latent L2 geometry.** The supervised
variant is the decisive probe: its two component means are *by construction* the
conditional means of the "object moves" vs "object doesn't move" futures — no EM,
no winner noise, no optimization excuse. They differ by **9.9 L2 units, i.e. ~9% of
the typical prediction residual (106)**. The outcome bifurcation that is large in
object space occupies so few latent dimensions that the latent's L2 metric — the
same metric the CEM planner optimises and BB's spread uses — barely registers it.
This *deepens* the paper's claim: the boundary blindness is not only a predictor
pathology; **the latent metric itself underweights the boundary-relevant subspace**.
A perfectly mode-switching head could add at most ~10 units of prediction spread —
standardised against the population, BB cannot move. The earlier EM-based failures
(mean collapse; winner-label noise: the winning component is decided by the other
~91% of the residual) are downstream symptoms of this same geometry.

**(2) Expert-only data carries no counterfactual action signal at the boundary.**
π trained as the boundary-event classifier becomes well *calibrated* (mean π =
[0.746, 0.254] vs the true 24.9% positive rate) but never flips under action
perturbation, and its CE (0.487) only modestly beats the always-no baseline
(0.562). In expert demonstrations, at boundary states the executed action almost
always *succeeds* — the dataset contains essentially no "2–3° off, object did not
move" counterfactuals — so P(move | state, action) is learnable from the state but
not its **action**-dependence. The model cannot learn the boundary's action side
from data that never crosses it.

### 6.2 What this means for the paper (the C3 contribution, revised)

The honest claim is now sharper and better supported than a cheap win would be:

- **Diagnosis (C1, C2)** stands and gains a mechanism: BB is high at pre-grasp
  boundaries (gate PASS) *because* (a) the latent metric compresses the
  boundary-relevant subspace ~10× below the residual scale, and (b) training data
  cannot teach the action-dependence. Both are now **measured numbers**, not
  hypotheses (9.9 vs 106; CE 0.487 vs 0.562 base, flip rate 0).
- **The fix must therefore act on the latent or the data, not the head:** (i)
  re-weight / learn a metric that amplifies the boundary subspace (e.g. supervise a
  projection with the object-displacement label — direction D at the *encoder*
  level rather than the head level), and/or (ii) collect or synthesize
  counterfactual boundary actions. Predictor-side multimodality alone — any K, any
  objective — is **structurally insufficient**, which is itself the cleanest
  falsification result this leg could produce.
  **Status: (i) is now implemented as the *state-grounded latent metric*** —
  `models/probes/object_probe.py` (probe `g(z)→object xyz` + the φ-metric adapter +
  the boundary-aware CEM cost), evaluated by `scripts/14_train_object_probe.py`
  (V1–V3 validations), `scripts/15_eval_metric_boundary.py` (BB under φ) and
  `scripts/16_planning_metric_compare.py` (paired Action-Error A/B, L2 vs φ cost).
  Results land in `results/metaworld_boundary_metric.csv` and
  `results/metaworld_planning_metric.csv`.
- The synthetic suite confirms every mechanism works when the latent
  *does* resolve the boundary — isolating the failure to the representation, not
  the method.

## 7. The fix that works: the grounded object-dynamics channel (measured 2026-06-10)

§6.2's two directions were then executed the same evening, with the probe chain
(`scripts/14`) localizing the failure one level deeper first:

| validation | question | result |
|---|---|---|
| V1 | object position decodable from the latent? | **✓** median err 0.064 vs per-dim sd 0.094 |
| V2 | does the predictor propagate it (factual action)? | **✓** 0.059 through F(z,a) — no loss |
| V3 | does it respond to **counterfactual** actions? | **✗** spread corr with true outcome = **+0.035** |

So the bottleneck is precisely the predictor's *counterfactual action → object
motion* channel. Two consequences, both verified through the production BB path:

1. **Metric re-weighting alone cannot fix it** (`scripts/15`,
   `results/metaworld_boundary_metric.csv`): amplifying the object subspace in the
   distance (φ-metric) just redistributes BB (pre_grasp 1.32→1.17 but free_space
   0.28→0.47) — you cannot expose a signal the model does not produce. The blended
   metric is indistinguishable from base. This is the ablation that kills the
   "just change the metric" alternative.
2. **A directly-supervised dynamics channel CAN learn it** (`scripts/17`):
   `h(z_t, a) → Δobject` (0.5M params, frozen everything else, cache-only,
   3 epochs ≈ 25 min). The cross-sample neighbourhood variation — similar state,
   different action, different outcome, which the hard_nn pools prove exists —
   is sufficient supervision; it never needed counterfactual *pairs*, only a
   training target that isn't buried in 98k-dim L2. Measured:
   - counterfactual gate: corr(spread_h, spread_true) = **+0.682** (vs +0.035
     for the frozen predictor — a 20× jump in boundary tracking);
   - val MSE 2.4e-4 (≈1.6 cm RMSE on the displacement vector);
   - **BB with h as the object-prediction channel**
     (`results/metaworld_boundary_dynamics.csv`, pooled n_b-weighted, excl.
     `mw-door-close`):

| regime | base bb_boundary | **+h** | base bb | **+h** |
|---|---|---|---|---|
| free_space | 0.282 | 0.345 | 0.069 | 0.178 |
| **pre_grasp** | **1.323** | **0.660 (−50%)** | 0.541 | 0.328 |
| contact_manipulation | 0.481 | 0.464 | 0.212 | 0.298 |

   The headline read: **the pre_grasp-vs-free_space BB gap collapses from 1.04 to
   0.32** — boundary blindness no longer concentrates at the boundary. (Honest
   note: free_space rises slightly because in object space the model's spread is
   near-zero where the object never moves, a standardisation effect; the paper
   reports the gap, not just the level.)

**Planning integration — measured (and one disclosed bug):**
`grounded_dynamics_cost` integrates h along the CEM rollout
(`cost = ‖Δz‖²/s_z² + β²·‖(g(z₀)+Σₜh(ẑₜ,aₜ)) − g(goal)‖²/s_g²`;
`planning/cem_planner.py` gained a `traj_cost_fn` hook). The paired Action-Error
A/B (`scripts/16`, arms l2 / probe-metric / obj-only / hdyn, identical CEM noise
per transition, H=3, 41 plannable transitions):

- *First run was invalid*: per-dim MSE normalisation made the latent term ~1e-5 of
  the object term (98,304 vs 3 dims), silently turning every "blended" arm into
  object-only — which *hurts* Action Error badly (flat cost surface in free space
  → CEM optimises noise; +5.2 paired). Preserved as
  `results/metaworld_planning_metric_buggy_scale.csv`; the `aware`≡`obj` identity
  in that CSV is the fingerprint. Costs are now squared-norms over scale norms.
- *Corrected run* (`results/metaworld_planning_metric.csv`): **no harm, no
  measurable gain** — paired Δ(hdyn−l2): pre_grasp −0.15 [−1.07, +0.79] (n=6),
  contact −0.07 [−0.51, +0.34] (n=21), free +0.15 [−0.15, +0.52] (n=14); aware
  arm similar (contact −0.26 [−0.59, +0.06]). Point estimates lean the right way
  in contact regimes; nothing survives the CI.
- *Why this was predictable in hindsight*: open-loop **Action Error rewards
  reproducing the expert's full arm trajectory**, which the plain L2 term already
  optimises; the boundary fix changes *which future the model can distinguish*,
  a distinction that mostly matters closed-loop (does the grasp succeed?), and
  pre_grasp had n=6. The right planning endpoint is **closed-loop success rate**
  with the grounded cost.

**§7b. Closed-loop run (2026-06-12) — done.** Full report:
`results/closed_loop_report.md`; data `results/metaworld_closed_loop.csv`
(96 paired arm-episodes, upstream-parity protocol, local 12 GB box). Headlines:

- **Reach reproduces above the paper**: L2 15/16 (94%) vs paper DWM 44.8 ±8.9
  (grounded 16/16) — the harness and protocol are right (three env-side
  reproduction bugs had to be found first: default-camera 480px renderer,
  the training data being flipud(corner2+tweak), goal = expert's FINAL frame).
- **Push and pick-place are 0% for BOTH arms** — the closed-loop face of
  Boundary Blindness: the arm reaches the goal pose (final ee 2–4 cm) while
  the object never moves (state-dist ~0.5–0.6). The upstream paper's appendix
  describes exactly this qualitatively ("hallucinates grasping the object");
  its closed-loop Metaworld tables stop at Reach/Reach-Wall.
- **The grounded cost is no-harm (reach 100%) and measurably better on
  contact**: paired final state-dist improvement pooled over push+pick-place
  +0.089 [bootstrap CI +0.022, +0.162] — but it flips no successes. CEM
  rarely samples a contact-creating sequence for the grounded term to score,
  so the residual bottleneck is exploration/imagination at the boundary
  (planner-side), consistent with the ladder's localization.

The paper's planning claim therefore upgrades from "conservative drop-in" to:
the grounded cost restores model-side boundary resolution (BB −50%), keeps
free-space planning intact closed-loop, and yields a CI-supported paired
improvement on contact tasks; converting that into task success additionally
requires contact-aware action proposal — measured future work, with the 0%/0%
baseline table as its motivation.

**The method, in one paragraph (for the paper):** *Frozen JEPA world models are
boundary-blind: at grasp boundaries their predictions neither separate (BB gate)
nor respond to counterfactual actions in the object subspace (V3). The failure is
not capacity (mixture heads: null), not the metric (φ-reweighting: null), and not
missing information (V1–V2 pass): it is the training objective — full-latent L2
buries the object's action-dependence. A 0.5M-parameter grounded dynamics channel
h(z,a)→Δobject, trained on the same cached data with the object as the explicit
target, restores counterfactual tracking (corr 0.03→0.68), halves BB at the
pre-grasp boundary, and plugs into CEM as a grounded cost term — with the encoder,
predictor and data untouched.* Scope caveat as always: the supervision label is
Metaworld sim state (object displacement proxy); DROID has no such label, so the
real-robot transfer of the *training recipe* (not the principle) is future work.
