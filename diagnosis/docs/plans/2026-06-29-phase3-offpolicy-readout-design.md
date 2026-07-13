# Phase 3 — the wall is the off-policy readout, not static precision (2026-06-29)

## Why (what the Phase-0 ladder + Test-1b pinned)

Under PERFECT latent dynamics (latent-oracle, `scripts/30`), same CEM budget as the
state-oracle, swapping only the cost:

| cost (perfect dynamics) | push | pick | note |
|---|---|---|---|
| `l2` | 0/16 | 0/16 | null |
| `gobj` (object→goal, no approach) | 0/8 | 0/8 | confounded: no hand-approach term |
| `metric` d_θ | 0/8 | 0/8 | d_θ gate failed (Spearman 0.62) |
| **`stateprobe`** (obj→goal + 0.5·hand→obj, **probe** readouts) | **1/16** | 0/… | exact state-oracle cost, probe-read |
| *state-oracle* (same cost, **TRUE** readouts) | **16/16** | 11/16 | the ceiling |

`stateprobe` is the clincher: the cost is now structurally identical to the state-oracle
(object term **and** hand-approach term), the dynamics are perfect, the planner budget is
identical — the **only** remaining difference is the readout source (sim truth vs probe
`g(z)`), and success collapses 16/16 → 1/16.

Yet **Test-1b** (`scripts/21`, held-out EXPERT frames) shows the spatial probe localises
the STATIC object to median 2.1 cm, **<5 cm 92%** (contact 90.7%). So the encoder is NOT
the static-precision ceiling. The reconciliation: the probe is accurate on the expert
distribution it was trained on, but the planner scores **off-policy** frames — the latents
of arbitrary CEM action rollouts (odd arm/object configurations) — where the probe is
untrained and degrades. The cost surface is therefore noisy exactly where CEM searches,
so it leads the planner astray (object ends 8–30 cm away; the lone push success ep07 had
obj_goal 0.014, i.e. the rare case the probe happened to read accurately).

**Localised wall:** the **off-policy robustness of the object readout from the frozen
latent**. Ruled out, in order: predictor, planner, cost formula, hand-approach term,
static readout precision.

## What (deconfound first, then fix)

### 3a — confirm the root cause (cheap diagnostic), `scripts/34_offpolicy_precision.py`
Measure the probe's decode error on **off-policy** frames, vs the expert-frame error from
Test-1b. From many sim start states, roll RANDOM action sequences (the same distribution
CEM samples from — Gaussian in raw action space, clipped), render + encode each resulting
frame, read TRUE object/hand from sim state, and tabulate `‖g(z) − true‖` vs the 5/7 cm
radii — exactly Test-1b's table, but on off-policy frames.
- Reuse: `scripts/29` `snapshot`/`restore` + `env.step`, `scripts/30` `encode_frame`/
  `roll_final_frame`, `scripts/21`'s error-vs-radius reporting, `OBJECT_SLICE` + the ee
  slice (`state[:3]`).
- **Decision:** off-policy `<5 cm` ≪ the 92% expert number (e.g. drops below ~50%) →
  root cause CONFIRMED, proceed to 3b. If off-policy precision stays high → the wall is
  elsewhere (planner-noise interaction); re-open the planner budget question.

### 3b — fix: off-policy-robust readout, `scripts/22` extended (`--offpolicy-frac`)
Augment the spatial-probe training set with off-policy frames (random/perturbed-action
sim rollouts → render → encode → label with the TRUE object/ee from sim state), mixed at
`--offpolicy-frac` with the existing expert cache records. Same `SpatialObjectProbe`,
encoder frozen. Retrain object + ee probes.
- Re-gate with the robust probes: `scripts/30 --cost stateprobe --probe <robust_obj>
  --ee-probe <robust_ee>` (and re-run Test-1b 3a on them).
- **Decision:**
  - push ≫ 1/16 (approaches state-oracle) → **the fix is an off-policy-robust readout** —
    cheap, frozen-encoder, paper-worthy. Then carry it into the closed-loop arms
    (`gobjc`/`stateprobe`-style cost in `scripts/18`) as the method.
  - still ~1/16 → the frozen latent itself loses object info on off-policy frames (no
    readout can recover what is not there) → escalate to an **encoder-level objective**
    (action-conditioned / contrastive fine-tune that keeps the object linearly-decodable
    off-policy) — heavier, separate design.

## Validation order (cheap → expensive)
1. Offline: unit-test the off-policy frame sampler shape/labels on the synthetic adapter.
2. `scripts/34` 3a diagnostic (one modest GPU job; encode-only, no planning) — gate.
3. Only if 3a confirms: `scripts/22 --offpolicy-frac` retrain (GPU), then `scripts/30
   --cost stateprobe` re-gate (the decisive spend).

## Files
- New: `scripts/34_offpolicy_precision.py`, `scripts/slurm_phase3_offpolicy.sh`, this doc.
- Modified: `scripts/22_train_spatial_probe.py` (`--offpolicy-frac` augmentation).
- Reused: `scripts/29` sim rollout, `scripts/30` encode/cost (`--cost stateprobe` already
  added), `scripts/21` reporting, `models/probes` (`SpatialObjectProbe`, ee probe).
