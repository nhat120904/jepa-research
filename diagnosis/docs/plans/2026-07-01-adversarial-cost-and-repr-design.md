# Beating the contact wall: adversarial planning cost + action-conditioned representation (2026-07-01)

## Why (what Phase-0/3 pinned)

The oracle ladder is closed (`results/oracle_ladder_cost_report.md`, memory
`phase0-gate-readout-is-the-wall`). Under PERFECT latent dynamics (latent-oracle,
`scripts/30`), same CEM budget as the state-oracle:

| cost (perfect dynamics) | push | pick | note |
|---|---|---|---|
| `l2` | 0/16 | 0/16 | null |
| `gobj` | 0/8 | 0/8 | no approach term |
| `metric` d_θ | 0/8 | 0/8 | dead (gate failed) |
| `stateprobe` (expert probes) | 2/16 | 0/16 | exact state-oracle cost, probe-read |
| `stateprobe` (**off-policy-ROBUST** probes, Phase-3 3b) | **1/16** | 0/16 | obj 78→92% <5cm off-policy — no transfer |
| *state-oracle* (TRUE readouts) | **16/16** | 11/16 | the ceiling |

Phase-3 3b is the clincher: a REAL readout fix (measured on the exact off-policy
distribution CEM samples) transferred **zero** planning success. The mechanism:
CEM searches for the cost **minimum** over 100×6 candidates every planning step —
it doesn't sample uniformly, it adversarially **finds** whatever residual-error
pocket a post-hoc frozen-encoder cost has and converges its population there. This
is reward-hacking, not a missing-information problem (the object IS recoverable
off-policy at ~2cm median).

Every lever that operates on the frozen encoder — dynamics (predictor), cost
formula, hand-approach term, readout precision (even off-policy-robust) — is now
exhausted at 0–2/16 on contact.

## Two tracks, run in parallel (user-chosen)

- **Track 1 — adversarial cost hardening.** Cheap, frozen encoder. Directly attack
  the reward-hacking mechanism: mine the frames CEM actually exploits, retrain the
  cost against them, repeat (DAgger).
- **Track 2 — action-conditioned representation adapter.** Heavier. Reshape the
  representation with a small learned head trained with an action-grounding /
  temporal / adversarial objective, so L2 in the new space is harder to exploit.

Both plug into the same latent-oracle gate (`scripts/30 --cost`, perfect dynamics)
so each is priced for ~1 GPU-hr per rung before any closed-loop spend — the same
cheap-first discipline as Phase 0/3.

## Shared foundation: mining the exploited distribution (`scripts/_cem_mining.py`)

`cem_plan_latent` (scripts/30) already renders + encodes the TRUE latent for every
CEM candidate every iteration; it just never kept the elites around. `on_elites`
is a small hook (`(z_elite, raw_elite, cost_elite, iteration) -> None`, default
`None`, fully backward-compatible with every existing Phase-0/3 gate run) called
once per CEM iteration with the surviving elite population — the frames CEM
**trusts** and converges its search around, as opposed to a random off-policy
sample (`scripts/_offpolicy_frames.py`).

`mine_cem_frames` re-runs full latent-oracle episodes with a given cost and
records every elite's true latent, true object/ee, and the episode's goal latent +
goal object (broadcast per row) — everything both tracks need.

## Track 1 — Phase A (diagnose) → Phase B (harden)

**Phase A** (`scripts/35_cem_exploit_precision.py`, `slurm_phaseA_exploit.sh`,
~1 GPU-hr): mine CEM-elite frames using the `stateprobe` cost with the off-policy-
robust probes already trained in Phase-3 3b (no retraining needed). Measure probe
decode error **on those elites** vs the 5/7cm radii, next to 3b's 91.5% random-
off-policy number, plus the "exploitation gap" (the cost's own value vs the true
object→goal distance on converged elites). If elite accuracy collapses vs 91.5%,
the reward-hacking-pocket mechanism is confirmed and Phase B is licensed.

**Phase B** (`slurm_phaseB_adversarial.sh`, a K-round DAgger loop, no new
orchestrator script — driven entirely by existing/extended scripts): each round
(1) mines exploited elites with the current probes (scripts/35 `--save-buffer`),
(2) retrains both probes mixing that buffer in via the new `--extra-buffer
--extra-frac` flags on `scripts/22`/`scripts/19` (mirrors the existing
`--offpolicy-frac` mixing pattern exactly), (3) re-gates `--cost stateprobe`. Push
climbing toward 16/16 over rounds means Track 1 wins; a plateau near 1-2/16 means
the pockets are inherent to the frozen `z` geometry, not fixable by re-supervising
a readout — escalate to Track 2.

`build_oracle_cost` also gained an `advmetric` arm — numerically identical to
`metric` (locked by `tests/test_latent_oracle_costs.py`), a distinctly-named alias
meant to pair with a hardened `LatentMetric` checkpoint for CSV/log traceability.
Lower priority than the `stateprobe` DAgger loop since the `metric`/0c branch was
already a dead candidate (Spearman gate failed, job 23359).

## Track 2 — action-conditioned representation adapter `φ`

`models/heads/action_repr_adapter.py`'s `ActionReprAdapter`: a CLS-attention
transformer over the frozen patch tokens (`SpatialObjectProbe`'s architecture,
since Test-1b showed mean/max pooling destroys the spatial signal a plannable cost
needs) producing `φ = A(z)`, `phi_dim` wide. The first `obj_dim` (3) components are
the grounded object estimate; the rest ("extra") are free.

Four losses, trained by `scripts/37_train_repr_adapter.py`
(`slurm_phaseC_repr.sh`, ~1 GPU-hr for the gate):

1. **grounding** — MSE: `φ[:, :obj_dim]` → true object xyz. Keeps the ~2cm
   precision the frozen z already carries.
2. **cf-contrastive** — margin: for an IN-BATCH hard-effect negative
   (`metrics.negative_samplers.hard_effect_negative`, extended with an optional
   `return_indices` so the negative's true next-latent can be gathered — a small,
   backward-compatible addition, default `False`), the object estimates of the
   factual and negative transitions must be separated by **at least their true
   object gap**. This is the mechanistic improvement over Phase-3: a margin
   constraint on hard cases specifically, not just average MSE — MSE alone is
   exactly what 3b already tried and failed to make exploitation-robust.
3. **temporal ranking** — a scale-free pairwise hinge (near-to-goal frame must
   have smaller φ_extra-distance to the goal than a far one), NOT a literal
   step-gap regression: regressing raw step counts would force φ_extra's
   magnitude onto a ~1-12 scale that fights term 1's metres-scale grounding once
   both live in the same φ vector. Lives entirely in the "extra" subspace so it
   cannot dilute the object signal. CEM only needs the cost to *descend* toward
   the goal, not match a literal step count, so ranking is the right primitive.
4. **adversarial** — margin: the SAME mechanism as term 2 (`margin_loss`, shared
   helper), applied to CEM-EXPLOITED elites mined live via `scripts/_cem_mining`
   (`--mine-adv --adv-cost l2`, or a pre-mined `--adv-buffer` from `scripts/35`)
   instead of an in-batch source — this is what directly closes the pockets that
   broke Phase-3 3b.

`scripts/30`'s new `phi` cost arm scores
`‖φ_obj(z_fin)−φ_obj(z_goal)‖²/s_g² + β·‖φ_extra(z_fin)−φ_extra(z_goal)‖²/s_extra²`
— the same "squared norms over scale norms" pattern already used by `gobj` /
`boundary_aware_cost` (`models/probes/object_probe.py`), reusing the existing
`--s-g`/`--beta` flags. `s_extra` (`extra_scale`) is calibrated at train time (std
of `φ_extra` over held-out data) and saved in the checkpoint — mixing an
unscaled metres-scale object term with an unscaled extra-subspace term would let
whichever happens to have larger raw magnitude dominate the sum, silently
defeating term 1's grounding (the exact pitfall `grounded_dynamics_cost`'s
docstring already documents for this codebase).

## Gate

`scripts/30 --cost phi`, 16 episodes, strict success, same CEM budget as every
other rung. Push/pick 0 → >0 under perfect dynamics licenses carrying `φ` into the
`scripts/18` closed loop (a later step, not built here — needs a `φ`-space rollout
cost, analogous to the existing `gobj`/`lmet` arms). Staying at 0-2/16 means even an
adversarially-hardened, spatially-aware, action-grounded readout on the frozen
latent cannot escape the pockets — the next escalation is an encoder-level
objective (fine-tuning, out of scope here).

## Files

- **New:** `scripts/_cem_mining.py`, `scripts/35_cem_exploit_precision.py`,
  `models/heads/action_repr_adapter.py`, `scripts/37_train_repr_adapter.py`,
  `scripts/slurm_phaseA_exploit.sh`, `scripts/slurm_phaseB_adversarial.sh`,
  `scripts/slurm_phaseC_repr.sh`, `tests/test_cem_mining.py`,
  `tests/test_latent_oracle_costs.py`, `tests/test_action_repr_adapter.py`.
- **Modified:** `scripts/30_latent_oracle.py` (`on_elites` hook in
  `cem_plan_latent`; `advmetric`/`phi` arms in `build_oracle_cost`; `--repr-adapter`
  CLI flag). `scripts/22_train_spatial_probe.py` / `scripts/19_train_ee_probe.py`
  (`--extra-buffer`/`--extra-frac`). `metrics/negative_samplers.py`
  (`hard_effect_negative(..., return_indices=False)`, backward-compatible).
  `models/heads/__init__.py` (export `ActionReprAdapter`).
- **Reused unchanged:** `cem_plan_latent`/`roll_final_frame`/`encode_batch`/
  `build_oracle_cost` (scripts/30), `models/heads/latent_metric.py`,
  `models/probes/object_probe.py` (`SpatialObjectProbe` architecture),
  `scripts/_offpolicy_frames.py`, `data/latent_cache.py`,
  `scripts/_boundary_diagnostic.py` helpers.

## Verification

1. Offline: `tests/test_cem_mining.py` (the `on_elites` hook, monkeypatched
   MuJoCo-free — population size, cost-ascending order, backward-compat default),
   `tests/test_latent_oracle_costs.py` (`advmetric` numerically equals `metric`;
   `phi` is zero at the goal, monotone, and correctly scale-weighted),
   `tests/test_action_repr_adapter.py` (each of the 4 losses converges on planted
   synthetic data, mirroring `tests/test_latent_metric.py`'s approach) — all pure
   CPU tensors, no MuJoCo/GPU/cache.
2. `scripts/check_normalization.py` (unaffected — no new rollout path; both tracks
   route through the existing `cem_plan_latent`/`adapter.normalize_action`).
3. Phase A (~1 GPU-hr, decisive): CEM-elite decode vs 91.5%.
4. Phase B: per-round push/pick via the `stateprobe` re-gate.
5. Phase C: `scripts/37`'s held-out gates (grounding decode, cf-margin satisfied
   fraction, temporal ranking accuracy + monotone-to-goal Spearman) then
   `scripts/30 --cost phi`.
6. Closed loop (only past a gate): whichever track crosses success carries into
   `scripts/18` vs the state/latent oracles; headline = push/pick 0 → >0 with
   reach ≥13/16 intact.
