# Object-grounded residual corrector inside the CEM unroll (2026-06-25)

## Why

The oracle ladder (`docs/plans/2026-06-25-oracle-ladder-results.md`) localised the
contact-task wall:

- **State oracle** (`scripts/29`, perfect sim dynamics + object-state cost): push 16/16,
  pick 11/16 → the planner/CEM budget/success radius are adequate.
- **Latent oracle** (`scripts/30`, *perfect* latent dynamics through the real encoder +
  the upstream **L2-in-DINO-latent cost**): push 0/16, pick 0/16 → fixing the predictor `F`
  alone cannot beat baseline. The only thing that changed between the two is the
  **cost/representation**, so the wall is the **L2-latent cost geometry**.

The corollary is sharp: **any fix scored purely by L2-to-`z_goal` is doomed** (the latent
oracle has perfect dynamics and still fails). The earlier residual corrector (`scripts/20`)
was trained single-step and only ever evaluated under the L2 cost (`l2c`/`hdync` arms) —
exactly the cost the latent oracle rules out. And `hdyn` added an object term but integrated
the dyn-head `h` over a *frozen* rollout for **scoring only** — it improved final state-dist
(+0.089) but flipped 0 successes, because the object signal never enters the latent the CEM
searches over.

What we already know works: a 0.5M-param head `h(z,a)→Δobject` on frozen latents recovers the
counterfactual object channel (cf-corr +0.68, ~20× the frozen predictor). **The object
information is in the latent.** We need it (a) *inside* the searched rollout and (b) *in* the
cost.

## What

Two coupled changes, both inside the planner loop:

1. **`scripts/31_train_grounded_corrector.py`** — train `Δ_θ(z,a)` (reuse
   `ResidualPredictorHead`, zero-init → identity start) with an objective that grounds the
   corrected object channel *after recursive application*:
   - `recon` `‖ẑ_{t+1}−z_{t+1}‖²` — keep the latent sane.
   - `λ_obj ‖g(ẑ_{t+1})−obj_{t+1}‖²` — 1-step object grounding (frozen probe `g`).
   - `λ_cf ‖g(ẑ(z,a'))−(g(z)+h(z,a'))‖²` — distil the cf-corr-0.68 dyn-head `h` into the
     corrected object channel (`a'` = in-batch-shuffled actions; `h(z,a')` is the teacher).
   - `λ_ms Σ_{k=1..K} ‖g(ẑ_k)−obj_{t+k}‖²` — **K-step recursive grounding** so the channel
     survives the unroll ("readout-fixed ≠ rollout-fixed",
     `docs/plans/2026-06-24-object-aware-predictor-design.md`). Sequences built with the same
     `build_sequences` contiguous-slice logic as `scripts/23`.

   Everything frozen except `Δ`. **Gates** (printed; do not spend closed-loop budget unless
   both pass): corrected obj-MSE < frozen baseline (1-step *and* K-step), and cf-corr ≥ 0.5
   (the corrected object readout's spread across neighbour actions tracks the true outcome
   spread — same gate as `scripts/17`, applied to `g(F+Δ)`).

2. **`gobjc` arm in `scripts/18`** — new `gobj` branch in `make_traj_cost`: read the object
   straight off the rollout latent, `g(ẑ_t)` driven to `g(z_goal)` dense along the horizon,
   per-dim normalised on `s_g`; L2 demoted to a `--gamma-l2` regulariser. The `c` suffix
   routes the rollout through `plan_adapter_c` (the corrected `F+Δ`), so `g(ẑ_t)` reads the
   *corrected* latent. This is the one ingredient `hdyn` lacked: the object signal lives in
   the searched latent, not just the score.

   Suffix parsing already supports this: `gobjc` → corrected adapter + cost arm `gobj`.
   `gobj` (no `c`) is the frozen-rollout ablation.

## Comparison matrix (identical protocol to 18/29/30)

H=6, nas=3, 300 samples, 15 iters, 16 eps, `--strict-success`, expert-final-frame goal.

| Task | l2 | hdyn | **gobjc** | latent-oracle | state-oracle |
|---|---|---|---|---|---|
| mw-push | 0/16 | 0/16 | **target >0** | 0/16 | 16/16 |
| mw-pick-place | 0/16 | 0/16 | **target >0** | 0/16 | 11/16 |
| mw-reach | 13–16/16 | — | **≥13/16 (no regress)** | 16/16 | n/a |

Headline = `success_end` flips 0→>0 on push and/or pick-place with reach intact.

**`gamma_l2` tradeoff:** reach has no object motion, so its only signal is the L2 term —
`--gamma-l2 1.0` keeps reach drivable. Contact tasks may need the object term dominant; the
knob lets the sweep trade reach-safety against contact-commitment without retraining.

## DROID regression check

The corrector needs object GT, so it is Metaworld-only. The `gobjc` arm is only selected
when requested, so baseline DROID Action-Score is untouched; confirm with
`scripts/terver_gripper_test.py` and the existing `dino_wm_droid` contact action_score
(~0.60, memory: `compare-against-paper-numbers`). A `--label state` proxy corrector on DROID
(via `scripts/17`-style state diff) is possible future work but out of scope here.

## Validation order (cheap → expensive)

1. `pytest tests/` + `scripts/07_validate_synthetic.py` (metric wiring).
2. `scripts/31` against the synthetic adapter (loss numerically sane offline).
3. `scripts/check_normalization.py` (≤ ~2× eval loss) — action norm is the #1 silent bug;
   `scripts/31` and `_corrected_rollout_chunk` both go through `adapter.normalize_action`.
4. Train `scripts/31` on the real cache; require the obj + cf gates to pass.
5. `scripts/23_rollout_fidelity.py --residual-head` (recursive correction doesn't blow up).
6. `scripts/24_counterfactual_spatial.py` (cf spread ≥ ~2cm, corr ≥ 0.5).
7. Only then: `scripts/18` `gobjc` closed-loop (the H100 spend), via
   `scripts/slurm_grounded_corrector.sh`.

## Files

- New: `scripts/31_train_grounded_corrector.py`, `scripts/slurm_grounded_corrector.sh`, this doc.
- Modified: `scripts/18_closed_loop_eval.py` (`gobj` cost branch + `--gamma-l2`; `gobjc` routes
  to the corrected adapter via existing suffix logic).
- Reused unchanged: `models/heads/residual_predictor.py`, `models/probes/object_probe.py`
  (`ObjectProbe`/`SpatialObjectProbe`, `ObjectDynamicsHead`), `scripts/23` `build_sequences`
  pattern, `scripts/17` cf-gate pattern.
