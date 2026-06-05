# Design — Planning Action-Score probe + CRA_eff correlation (DROID)

**Date:** 2026-06-05
**Goal:** Close the causal link "action-grounding failure → planning failure" that the
diagnostic currently leaves *inferential*. On the **same cached DROID transitions**, measure
the paper's **planning Action Score** per regime and correlate it with **CRA_eff**. If the
regimes where CRA_eff ≈ chance are also the regimes where the planner's Action Score is worst,
the link stops being a mechanistic argument and becomes evidence.

Scope (confirmed with user): **DROID only**, Robocasa hooked later. Metric = **Action Score**
(paper-aligned; DROID is offline, there is no simulator → no task-success). Horizons: **both
1-step and the paper's H=3.**

---

## 1. Fidelity to the paper (this is the whole point)

Everything below is copied from the upstream repo, not invented:

- Planner: `evals/.../planning/planning/planner.py::CEMPlanner` (the `L2_cem` the configs use).
- Objective: `objectives.ReprTargetDistMPCObjective` with `sum_all_diffs=False` →
  **MSE between the LAST unrolled latent and the goal latent** (mean over feature dims),
  `alpha=0` (no proprio term).
- DROID dino-wm config
  `configs/evals/simu_env_planning/droid/dino-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0_ep64_decode.yaml`:

  | param | value |
  |---|---|
  | `iterations` | 15 |
  | `num_samples` | 300 |
  | `num_elites` | 10 |
  | `horizon` | 3 |
  | `var_scale` (init std) | 0.1 |
  | `momentum_mean`, `momentum_std` | 0, 0 |
  | `max_norms` | `[0.1, 0.75]` |
  | `max_norm_dims` | `[[0,1,2,3,4,5], [6]]` |
  | `num_act_stepped` | 3 |
  | `goal_source` | `dset` |
  | `goal_H` | 3 |
  | objective | `L2`, `sum_all_diffs=False`, `alpha=0` |

  **Box clip, not L1-ball:** pose dims 0–5 clipped to ±0.1, gripper dim 6 to ±0.75. (The
  diagnostic's `random_negative` L1-ball projection is for *negatives*, not the planner — do
  not reuse it here.)

- CEM loop (exact):
  1. `mean = zeros(H, A)`, `std = 0.1 * ones(H, A)`.
  2. per iter: `actions = mean + std * randn(H, num_samples, A)`; `actions[:,0]=mean`
     (mean-inclusion trick); clip per `max_norm_dims`.
  3. `cost = objective(unroll(z_init, actions))` → MSE(last latent, goal) per candidate.
  4. `elite_idxs = topk(-cost, num_elites)`; `mean = elites.mean(dim=1)`,
     `std = elites.std(dim=1)` (momentum 0 → straight replace).
  5. after `iterations`, return `mean[:num_act_stepped]`.

- Action Error (exact, `plan_evaluator.py:543-550`, DROID branch):
  ```
  d = |Σ_t planned[:,:3] − Σ_t expert[:,:3]|.sum()        # xyz cumulative
    + |Σ_t planned[:,3:6] − Σ_t expert[:,3:6]|.sum()      # orientation cumulative
    + |Σ_t planned[:,6:] − Σ_t expert[:,6:]|.sum()         # gripper closure
  ```
  i.e. sum each action stream over the executed horizon (pose deltas are additive → net
  displacement), then L1 between planned and expert net deltas, grouped xyz / orient / grip.
  **Action Score** = rescaled to maximize: `score = 1 − d / d_ref`, `d_ref` = p95 of `d` over
  the whole eval set (so it is cross-regime comparable; report raw `d` too).

---

## 2. Components / files

```
planning/__init__.py
planning/cem_planner.py          # CEMPlanner ported to run on WorldModelAdapter primitives
metrics/action_score.py          # action_error (grouped summed-delta L1) + rescale to score
scripts/08_planning_probe.py     # per transition: CRA_eff(hard_nn) + plan → action_error
scripts/09_correlate_planning.py # join the two → Spearman/Pearson + per-regime table + scatter
tests/test_cem_planner.py        # recovers a known action on a linear toy WM
tests/test_action_score.py       # grouped-L1 + rescale monotonicity
```

### `planning/cem_planner.py`
`cem_plan(adapter, z_init, z_goal, *, horizon, action_dim, num_samples=300, iterations=15,
num_elites=10, var_scale=0.1, max_norms=[0.1,0.75], max_norm_dims=[[0,1,2,3,4,5],[6]],
proprio_t=None, generator=None) -> planned_actions (horizon, action_dim)`.

- Pure function; the only model touch-points are `adapter.predict_rollout(z_init, actions,
  proprio_t)` (unroll) and the MSE objective on the last latent vs `z_goal`.
- Batched: one `z_init` planned with `num_samples` candidate sequences per iter.
- Deterministic via an explicit `torch.Generator` seed.
- Works at the adapter's **model-step granularity**: `action_dim = adapter._model_action_dim`,
  `horizon` counted in model steps. (Resolve frameskip folding by reusing the same action
  layout as `scripts/05` builds; verify shapes against `predict_rollout` at runtime.)

### `metrics/action_score.py`
- `action_error(planned, expert) -> dict(xyz, orient, grip, total)` — grouped summed-delta L1
  exactly as §1. `planned`,`expert`: `(T, A)` raw (un-normalized) actions.
- `rescale_action_score(errors, d_ref) -> scores` — `1 − d/d_ref`, clipped to be sane.
- Per-transition function so synthetic validation hits the production path (mirrors `metrics/`
  convention).

### `scripts/08_planning_probe.py`
Reuses `build_transition_records`, `materialize_records`, `action_spec`, `read_regimes`,
`bootstrap_ci` from the existing runner/data layer.

For each horizon in `{1, goal_H}`:
1. Build per-transition `(z_t, z_goal, expert_actions, regime, tid)`. For horizon `H`,
   `z_goal = z[idx0 + H*step]`, `expert_actions` = the `H` stacked model-step actions; skip
   transitions whose trajectory is too short for `+H*step`.
2. Optionally restrict to **effectful** transitions (`‖Δz‖>median`, same mask as ECS) so the
   pairing with CRA_eff is apples-to-apples — config flag, default effectful-only.
3. For each transition: (a) `cra_per_transition` with `hard_nn` negatives (reuse pool build
   from `05`), giving `cra_eff_correct ∈ [0,1]`; (b) `cem_plan` → `action_error`.
4. Subsample per regime to `max_planning_transitions` (CEM is `num_samples*iterations` unrolls
   per transition — bounded by `max_planning_transitions`; on the A5000 it is set to cover
   every transition in each regime).
5. Emit:
   - `results/droid_planning_pertrans.npz` — arrays `[tid, idx0, regime, horizon,
     cra_eff_correct, action_error_total, action_error_xyz/orient/grip]`.
   - `results/droid_planning.csv` — per (regime, horizon): mean action_error, action_score,
     mean CRA_eff, n, with trajectory-clustered bootstrap CIs.

### `scripts/09_correlate_planning.py`
- Per-transition correlation (thousands of points): Spearman + Pearson between
  `action_error_total` and `cra_eff_correct`. Expectation: **negative** (high error ↔ low CRA).
- Per-regime table: regime × {CRA_eff, Action Score, Action Error} + the regime-level
  correlation (4 points, reported as secondary — the per-transition corr is the headline).
- Scatter figure `results/figures/figure_c_planning_vs_cra.pdf`.
- Append a "Planning correlation" section to `decision_report.md` (or a sidecar md the report
  links), with the honest caveats from §4.

---

## 3. Testing (TDD, offline — no GPU/data)

- `test_cem_planner.py`: linear toy WM `z' = z + W a` exposed through a tiny fake adapter.
  With `z_goal = z + W a*`, CEM must recover `a*` within tolerance (cost → ~0). Also: clipping
  respected; determinism under fixed generator.
- `test_action_score.py`: grouped summed-delta L1 matches a hand computation; `planned==expert
  → error 0 → score 1`; monotonic in error; rescale clipping.
- Extend `scripts/07_validate_synthetic.py` (or a new synthetic test): a **grounded** synthetic
  model → planner recovers near-GT action (low error / high score) AND high CRA_eff; an
  **action-ignoring** model → planner cannot recover the action (high error / low score) AND
  CRA_eff ≈ chance. This is the end-to-end sign check that the correlation script will later
  confirm on real data.

All of the above run with `pytest` offline; only `08` needs the GPU/cache on the server.

---

## 4. Honest caveats (to record in the report + handoff)

1. **DROID Action Score is a proxy, not task success** — DROID is offline, no simulator. This
   is exactly the paper's DROID metric, so it is defensible, but it is not a grasp/lift success
   rate.
2. **Multimodality** — Action Error compares to a *single* expert trajectory; several actions
   can be valid, so error has a positive floor. The **per-transition correlation** (not the
   absolute score) is the evidence; a strong negative correlation survives the floor.
3. **CEM hyperparameters are the paper's** (iter 15 / 300 samples / 10 elites / H3) — but the
   subsample size (`max_planning_transitions`) is ours; record it so the number is reproducible.
4. **Causal, not just correlational, only if** the per-regime ordering matches: contact/gripper
   regimes should show *both* low CRA_eff *and* high Action Error. A null result (no
   correlation) would itself be informative — it would mean CRA_eff over-states the planning
   impact, and the report must say so.

---

## 5. Run order (server)

```bash
# after 03 (latents) + 04 (regimes) already cached:
python scripts/08_planning_probe.py    --config configs/diagnostic_droid.yaml
python scripts/09_correlate_planning.py --planning_csv results/droid_planning.csv \
                                        --pertrans results/droid_planning_pertrans.npz \
                                        --diagnostic_csv results/droid_diagnostic.csv
```
`05` (CRA/AUG/ECS) is still the prerequisite for the per-regime CRA_eff column used by `09`.
