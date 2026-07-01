# Oracle-ladder cost/readout localization — the contact-manipulation wall

Solution-track investigation (branch `exp/oracle-ladder-cost-localization`). Goal: find
*where* action-conditioned planning fails on contact-rich MetaWorld (`dino_wm_metaworld`)
and whether any cheap fix on the **frozen encoder** crosses task success. Everything below
holds the encoder + predictor frozen; only small probes / cost heads are trained.

All numbers: MetaWorld, strict env-flag success, CEM budget 100 samples / 6 iters / H=6 /
num-act-stepped=3, push radius 5 cm, pick radius 7 cm. 16 episodes unless noted.

## The ladder (each rung swaps exactly one thing)

| rung | dynamics | cost | push | pick | script |
|---|---|---|---|---|---|
| **state-oracle** | perfect (true sim) | **true sim state** (`‖obj−goal‖ + 0.5·‖hand−obj‖`) | **16/16** | **11/16** | `scripts/29` |
| latent-oracle `l2` | perfect (sim→render→encode) | `‖z_fin − z_goal‖²` | 0/16 | 0/16 | `scripts/30 --cost l2` |
| latent-oracle `gobj` | perfect | `‖g(z_fin) − g(z_goal)‖²` (γ=1/β=5) | 0/8 | 0/8 | `scripts/30 --cost gobj` |
| latent-oracle `metric` | perfect | learned `d_θ(z_fin, z_goal)` | 0/8 | 0/8 | `scripts/30 --cost metric` |
| latent-oracle `stateprobe` | perfect | **exact state-oracle cost, probe readouts** | 2/16 | 0/16 | `scripts/30 --cost stateprobe` |
| latent-oracle `stateprobe` **(robust probe)** | perfect | same, off-policy-robust probes | **1/16** | 0/16 | 3b, job 23553 |

The state-oracle (16/16) and every latent-oracle rung share the **same CEM budget, same
success radius, same perfect dynamics**. The only thing that changes down the ladder is the
**cost / readout**. So the wall is isolated to the cost, not the predictor, planner, or budget.

## What each rung rules out

- **Predictor `F` is not the wall.** Latent-oracle gives the predictor perfect dynamics
  (step the real sim, render, encode the *true* next latent) and it still fails 0/16 under
  L2 → fixing `F` alone cannot beat baseline.
- **Planner / CEM budget / success radius are adequate.** Same planner solves 16/16 with the
  true-state cost.
- **Cost *formula* and hand-approach term are not the wall.** `stateprobe` reproduces the
  exact state-oracle cost (object + 0.5·approach) but reads object and hand from probes →
  still collapses 16→2. Matching the successful cost's functional form does not rescue it.
- **Static readout precision is not the wall.** Test-1b (`scripts/21`, spatial probe) decodes
  the static object to **92% <5 cm** on expert held-out frames (contact regime 90.7%). The
  encoder carries the object position within the push radius.

## Phase-3: the off-policy-readout hypothesis (confirmed, then REFUTED as the wall)

Reconciliation attempt: the planner scores **off-policy** frames (latents of arbitrary CEM
action rollouts), where the expert-trained probe was never fit.

- **3a diagnose** (`scripts/34`, job 23553): the expert probe on off-policy random-action
  frames drops to **obj 78% <5 cm (push 69%)**, **ee 45.8% (push 30%)** — real degradation
  vs Test-1b's 92%. So the readout *does* degrade exactly where CEM searches.
- **3b fix** (`scripts/22/19 --offpolicy-frac 0.5`, job 23553): retrain both probes with
  off-policy frames mixed in. The object readout is **fixed**:

  | probe | off-policy <5 cm | median |
  |---|---|---|
  | expert (3a) | 78% (push 69%) | — |
  | **robust (3b)** | **91.5%** (push 92.9%, pick 90.0%) | **2.0 cm** |

  (ee robust probe: V1 median 6.65 cm, ee per-dim sd 9.2 cm — hand still the weak dim.)

- **3b re-gate** (`scripts/30 --cost stateprobe` with the robust probes): **push 1/16, pick
  0/16** — no improvement over the expert-probe baseline (2/16). Object still ends 13–30 cm
  from goal; `final_state_dist` 0.5–2.5.

**A real, measured readout fix (obj 78→92% <5 cm off-policy) transferred ZERO planning
success.** Off-policy readout precision was never the binding constraint.

## Verdict: a frozen-encoder post-hoc readout is not a *plannable* cost

Even at 2 cm average off-policy accuracy, CEM — which searches for the cost **minimum** over
100×6 candidates — **exploits the probe's residual error structure**: it drives the object to
where `probe(z_fin)` reads "at goal" while the true object is 13–30 cm away. This is a
**reward-hacking** failure (the optimizer finds the cost model's blind spots), *not* a
missing-information failure — the object position IS recoverable off-policy at 2 cm.

Ruled out, in order: **predictor → planner → cost formula → hand-approach term → static
precision → off-policy readout precision.** Every lever that operates on the frozen encoder
(better dynamics, better cost formula, better readout) is now exhausted at 0–2/16 on contact.

## Consequences

- The cheap frozen-encoder program (grounded corrector Track A, learned-metric Track B as a
  post-hoc readout, off-policy-robust probes) is closed. No frozen-encoder cost crosses
  contact success under perfect dynamics — so no closed-loop H100 run on them can win.
- The fix must make the cost **un-exploitable by the planner**: either
  (a) a cost trained **end-to-end as a planning objective** with the planner's own
  off-policy / adversarial rollouts as negatives, or
  (b) an **encoder-level action-conditioned objective** that reshapes the representation so
  that L2 / a readout is plannable (the original proposal's training objectives).
- This is a *stronger* paper claim than "L2 is contact-blind": **any post-hoc cost on a
  frozen action-conditioned JEPA encoder is planner-exploitable on contact manipulation.**

Design + plan for direction (a)/(b) is the next deliverable (heavier; needs H100).
