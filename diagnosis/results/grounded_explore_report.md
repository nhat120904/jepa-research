# Grounded exploration (option B) — closed-loop result

**Date:** 2026-06-18 · **Model:** `dino_wm_metaworld` (frozen public checkpoint)
· **Harness:** `scripts/18_closed_loop_eval.py` (arm `hexp`) via
`scripts/run_explore_sweep.ps1` · **Data:** `results/metaworld_grounded_explore.csv`
(64 episodes = 16 paired seeds × 2 arms × 2 contact tasks, seeds 10000–10015,
same env rand_vec + CEM noise per pair) · **Design:**
`docs/plans/2026-06-18-grounded-exploration-design.md`.

## What was tested

The 2026-06-12 sweep showed the grounded object term used as a **score** (`hdyn`)
improves contact-task final state-distance (+0.089) but flips zero successes,
because CEM rarely *samples* a contact-creating plan. Option B uses the same
frozen grounded channel to shape the **search** instead: arm `hexp` adds, along
the imagined rollout, a dense **APPROACH** term (predicted ee → object, via a new
ee-probe; drives the gripper into contact range) and a dense **MANIPULATE** term
(object → goal, via the dyn-head integration). Weights `lambda_app = lambda_obj =
1.0`, set so the grounded terms sit at parity with the visual term (smoke:
visual ≈2.0, app ≈1.0, man ≈1.0) — a decisive exploration push, vs `hdyn`'s
`beta=0.1` (~5% of visual).

## Results

| task | arm | success | mean final state-dist | mean final ee-dist |
|---|---|---|---|---|
| mw-push | l2 | **0/16** | 0.619 | 0.036 |
| mw-push | hexp | **0/16** | 0.644 | 0.115 |
| mw-pick-place | l2 | **0/16** | 0.504 | 0.036 |
| mw-pick-place | hexp | **0/16** | 0.641 | 0.170 |

Paired per-episode delta (l2 − hexp on final state-dist; **positive = hexp ends
closer**; 10k-resample bootstrap over the 16 pairs):

| task | mean Δ | 95% CI | hexp wins | worst hexp loss |
|---|---|---|---|---|
| mw-push | −0.025 | [−0.387, +0.237] | 7/16 | −2.298 |
| mw-pick-place | −0.137 | [−0.332, −0.001] | 2/16 | −1.396 |
| **pooled (n=32)** | **−0.081** | **[−0.278, +0.080]** | 9/32 | — |

## Reading

1. **Zero successes, both arms, both tasks** — option B does not move the metric
   that has never moved off 0%.
2. **hexp is worse, not better, on placement.** Pooled Δ = −0.081 (hexp ends
   *farther* on average), and pick-place is significantly worse (CI excludes 0).
   This *inverts* `hdyn`'s +0.089: pushing the grounded channel hard enough to
   actually steer the search makes the result worse, not better.
3. **The intervention mechanically worked** — hexp's final ee-distance is 3–5×
   larger than l2's (push 0.115 vs 0.036, pick-place 0.170 vs 0.036). The arm
   *did* abandon goal-pose mimicry and go to the object, exactly as the approach
   term intends. It simply did not convert to a grasp/push.
4. **Two catastrophic divergences** (push seed 10001: object ends 2.6 away;
   pick-place 10013: 1.77) — the planner reaches the object, acts on the frozen
   predictor's *imagined* contact dynamics, and shoves the object the wrong way.
   ee-distance there is small (0.02–0.20) — the gripper went where planned; the
   **object went where the model wrongly imagined**.

## Conclusion — the bottleneck is the predictor's rollout, not the planner

The fix ladder is now complete and the negative is decisive across the planner:

- **Scoring** (`hdyn`): can't help — nothing contact-creating to score.
- **Exploration** (`hexp`): forces contact, still 0 success, slightly worse — the
  planner now acts on imagined contact dynamics that are **wrong**.

Both planner-side levers are exhausted with the model frozen. The failure lives
**inside the frozen predictor's imagined rollout**: its counterfactual
action→object→latent dynamics at the grasp boundary do not match the simulator,
so no cost function and no search strategy over those rollouts can plan a
successful contact. This is sharper than the prior "exploration is the gap"
framing — it eliminates the planner entirely and pins the residual squarely on
the predictor, which a frozen-checkpoint study cannot fix from the outside.

→ The next real lever (breaks the frozen-everything constraint): a **latent-space
residual corrective predictor** `ẑ_{t+1} = F_frozen(z,a) + Δ(z,a)`, trained on
cached transitions with an object-grounded loss, so the corrected object channel
lives *inside* the unroll rather than in a side-cost. The dyn-head already proved
the signal is learnable (cf-corr +0.682); this puts it where CEM can use it.

Note: on the task the **paper actually reports** (Reach), the harness
*reproduces* the published number — episode-end L2 37.5% [18.5–61.4%] /
grounded 50.0% vs the paper's 44.8 ± 8.9 %, inside the CI (D.2 strict re-score;
the earlier "beats it / 94%" was an any-step latch artifact, retracted — see
`closed_loop_report.md`). Contact-task success is where both we and the paper
sit at 0, and reaching it requires the predictor-side fix above.
