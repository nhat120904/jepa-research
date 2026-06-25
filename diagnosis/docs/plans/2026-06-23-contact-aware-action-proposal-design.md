# Contact-aware action proposal — flipping contact success (lever #2)

**Date:** 2026-06-23 · **Status:** design
**Builds on:** `results/closed_loop_report.md` (hdyn +0.089 but 0/16),
`results/grounded_explore_report.md` (hexp null), `docs/PAPER_IDEA.md` §C2/2.7
("necessary-but-not-sufficient"; the residual bottleneck is *contact-creating
action proposal*), `scripts/17` (dyn-head), `scripts/18` (closed-loop CEM),
`scripts/22` (spatial object probe). Companion to the spatial-h leg
(`run_spatial_h_sweep.ps1` / `slurm_spatial_h.sh`).

## 1. Why this leg exists — the bottleneck has moved to proposal

Every **model-side, scoring-only, frozen** lever is exhausted, and the diagnosis
is unusually precise:

- The model can now *score* a contact-creating rollout better: the dyn-head
  `h(z,a)→Δobject` is action-discriminative (cf-corr **+0.68**), and as a planning
  cost it moves the closed-loop surface a CI-supported **+0.089 [+0.022, +0.162]**
  on pooled contact.
- Yet success stays **0/16**. The closed-loop signature is decisive: the arm
  reaches the goal *pose* (ee 2–4 cm) while the **object never moves**. CEM finds a
  low-cost basin — arm-at-goal — that satisfies the latent cost *without ever
  making contact*, and zero-mean Gaussian shooting essentially never samples the
  precise reach→grasp→lift sequence that would let the grounded term grade a
  contact.

So the residual is no longer "can the model tell good contact from bad" — it is
**"does the planner ever propose a contact at all."** This is the planner-side
piece `PAPER_IDEA.md` §2.7 deliberately scoped out of the model-centric paper, and
the precise thing V-JEPA-2-AC papers over by hand-feeding three sub-goal images.

**Falsifiable framing.** If we *give* the planner contact-making proposals and the
grounded cost to grade them and success still does not move, the failure is back
*inside the predictor's rollout* (→ option D: train the counterfactual channel).
If success *does* move, BB was the model-side half and proposal the planner-side
half of the same wall — a clean, complete story.

## 2. The intervention — three rungs, cheapest-first

All three keep the encoder + predictor **frozen** and reuse lever #1's scoring
(spatial probe object readout + dyn-head object term). They differ only in **how
CEM is initialised / proposed from**, i.e. they attack the *sampling* failure, not
the *scoring* failure.

### Rung A (primary): BC-seeded CEM — an amortized contact prior
Train a small behaviour-cloning policy `π_BC(a | z_t, z_goal)` on the **expert
Metaworld data**, which *does* contain successful grasps (the data the WM was
trained on). Use it to **initialise the CEM mean** of each replan window
(optionally seed a fraction `p_seed` of the first-iteration samples from
`π_BC` rollouts) instead of the zero-mean Gaussian. CEM then *refines* a
proposal that already reaches toward and closes on the object, rather than
searching from scratch.

- **Why primary:** directly addresses "CEM never samples contact"; the contact
  sequences already exist in data; minimal new infra (one MLP/GRU policy + a seed
  hook in `cem_plan`); amortized (no per-episode scripting).
- **Architecture:** `π_BC` = small GRU or causal MLP over `(z_t, proprio_t,
  z_goal)` → action mean (+ optional log-std), horizon-`H` open-loop chunk to seed
  the CEM window. ~0.5–2M params. Trained teacher-forced on cached
  `(z, proprio, a, goal)` tuples; goal = the episode's final-frame latent (same
  goal def as scripts/18).
- **Loss:** action MSE (or NLL if heteroscedastic); optionally a small
  object-reaching auxiliary (predicted-ee→object via the ee-probe) to bias toward
  contact-approaching chunks.
- **Integration:** new arm `l2bc` / `hdynbc` in `scripts/18` — same cost as `l2`
  / `hdyn`, only the CEM init changes (`cem_plan(..., init_mean=pi_bc_rollout,
  seed_frac=p_seed)`). Paired against `l2` and `hdyn` on the same env/seeds.

### Rung B: sub-goal decomposition (object-relative waypoints)
Use the **spatial probe** object position to decompose pick-place into
`reach → grasp → lift` and plan each segment to an **object-relative intermediate
goal** (ee→object, then object→goal), each of which is a *low-BB* segment. This
automates the hand-crafted sub-goals V-JEPA-2-AC supplies manually — but derives
them from the probed object instead of human-picked images.

- Cheaper to prototype than training `π_BC`, but more bespoke per task; best as a
  **second rung** / ablation if Rung A under-delivers.

### Rung C: scripted contact-primitive injection (control / ablation)
Seed a fixed fraction of CEM samples with a scripted "move ee to probed object,
then close gripper" primitive. Crudest, most brittle — kept only as an upper-bound
sanity check that *if* a contact sequence is proposed and the grounded cost grades
it, success can move at all. If even Rung C is 0/16, the predictor rollout (not
proposal) is the wall.

## 3. What stays fixed (reuse, do not reinvent)

- Cost surface = lever #1: spatial probe (`scripts/22`,
  `spatial_object_probe_dino_wm_metaworld.pt`, 2 cm aim) for goal/init object +
  dyn-head object term (`object_dynamics_dino_wm_metaworld.pt`, β object-dominant).
- Protocol = scripts/18 closed-loop, 16 paired episodes/task, horizon 6, 300
  samples, 15 iters, 3 stepped, ≤100 env steps, paired env+seed across arms.
- Encoder + predictor frozen. Only `π_BC` (Rung A) trains, cache-only.

## 4. Evaluation & decision

- **Primary:** task success rate on `mw-push` + `mw-pick-place`, BC-seeded arm vs
  `l2` and vs `hdyn`, paired. No-harm check on `mw-reach` (must stay ≥ baseline).
- **Secondary:** paired final object–goal distance (the +0.089-style metric);
  fraction of episodes that *make contact at all* (object displaced > τ) — the
  direct readout of whether proposal fixed sampling.
- **GO (success moves):** BB (model) + proposal (planner) jointly clear the
  contact wall → strong paper result / paper #2.
- **NULL (still 0/16 despite contact being proposed):** the predictor's rollout
  mis-scores even proposed contacts → escalate to option D (train the predictor's
  counterfactual object channel), now with proposal ruled in/out as the cause.

## 5. Risks

- **Prior wash-out:** 15 CEM iterations may erode the BC seed back toward the
  arm-at-goal basin. Mitigate with `p_seed` on later iterations / elite injection,
  or reduce iterations.
- **Distribution shift / BC overfit:** `π_BC` trained on expert states may
  mispredict off-distribution CEM states. Keep it as a *seed*, not a hard
  constraint; CEM still owns refinement.
- **Predictor still wrong on contact rollouts:** the deepest risk — even a
  correctly *proposed* contact may be mis-imagined by the frozen predictor (this
  is exactly what hexp hit). If so, Rung A converts into the motivation for option
  D and the two compose (BC-seed + corrected rollout).

## 6. Scope

Planner-side. Likely a **future-work section** of the current paper or the seed of
a follow-up; not folded into the frozen-model diagnostic story. Order of work:
Rung A first (run the spatial-h leg now as the scoring half it depends on), then
decide B/C or escalate to option D from the result.
