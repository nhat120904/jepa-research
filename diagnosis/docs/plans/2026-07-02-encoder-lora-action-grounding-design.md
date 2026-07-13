# Phase D — encoder-level LoRA: reshape the latent geometry itself (2026-07-02)

## Why (what Phase 4 pinned)

Phase 4 closed the frozen-encoder program **empirically**:

- **Phase A** (`results/cem_exploit_precision.csv`): the off-policy-ROBUST object
  probe decodes 91.5% <5cm on random frames but only **24.0%** on ALL CEM-mined
  elites, **19.2%** on final-iteration elites, **15.3%** on push elites. CEM
  adversarially converges its population into the readout's residual-error pockets.
- **Phase C v2** (`results/latent_oracle_phi_v2.csv`): the relearned representation
  `φ = A(z)` (2.6M-param head, grounding + cf-margin + temporal + adversarial
  losses) **fixed grounding** (MSE 0.0143→0.0043; v1 had collapsed to 7.1%) and
  still gates at **push 1/16, pick 0/16 — identical to v1**, object ending ~0.216m
  from goal (v1: 0.205m). Accurate average grounding transferred zero planning
  success, a same-model repeat of Phase-3 3b (probe 78→92% <5cm → 0 transfer).

Every lever that is a *function of the frozen* `z` is now exhausted at 0–2/16 on
contact: predictor → planner → cost formula → hand-approach term → static readout →
off-policy readout → learned metric → adversarially-hardened relearned
representation. The pockets are **inherent to the frozen-`z` geometry**: for any
`φ(z)`, latents the encoder has entangled stay entangled (`z_X ≈ z_Y ⇒ φ(z_X) ≈
φ(z_Y)` for every `φ`). The only untried lever is the encoder itself.

**Phase B (Track-1 probe DAgger) is skipped**: it re-supervises a readout, a
strictly weaker frozen lever than φ v2 which just failed with zero improvement —
its own design doc licensed escalation at a 1–2/16 plateau. (It stays available as
an honest-negative row if a reviewer asks for iterated adversarial rounds.)

## What: LoRA-finetune the encoder with the Phase-C objective

Inject zero-init LoRA adapters (`models/heads/lora_predictor.LoRALinear`, the same
toggleable mechanism scripts/26 already uses on the *predictor*) into the
**encoder's** transformer blocks (`dino_wm_metaworld` = DINOv2 ViT-S/14, `blocks.*`
Linears: qkv/proj/fc1/fc2; r=16, α=32 ≈ 1.2M trainable params). Train them
**jointly with a fresh φ head** using the Phase-C losses — unchanged in form, but
now their gradients flow *into the encoder*, so the geometry that creates the
pockets can move instead of being read around:

1. **grounding** — MSE `φ_obj(z̃_{t+1}) → true object xyz` (z̃ = LoRA-encode).
2. **cf-margin** — in-batch hard-effect negatives (`hard_effect_negative`,
   `return_indices=True`): φ_obj separation ≥ true object gap (capped). Same
   effect-conditioned margin as Phase C — contrast on *outcomes*, never on raw
   action identity (many-to-one dynamics).
3. **temporal ranking** — scale-free pairwise hinge on φ_extra distance-to-goal
   (near-goal frame < far frame), plus the cross-trajectory floor. Unchanged.
4. **adversarial** *(off in v0, on in round ≥1)* — margin on CEM-mined elites.
   **Stale-buffer caveat:** Phase-A buffers store frozen-`z`; once the encoder
   moves those latents are meaningless. Mining must keep the **frames**
   (`mine_cem_frames(keep_frames=True)`, new) so elites can be re-encoded live
   through the current LoRA encoder each round.
5. **preservation** *(new, the encoder-specific term)* — `‖z̃ − z_frozen‖²`
   normalized, z_frozen computed with LoRA toggled off (free via
   `set_lora_enabled`). Small λ (default 0.05): keeps the new geometry in the
   predictor's neighborhood (for the eventual closed loop) and guards against
   collapse to a degenerate feature. λ=0 is the "unconstrained reshape" ablation.

### Why the losses can win here when they lost in Phase C

Phase C optimized `φ` subject to a **fixed** geometry: where the encoder had
already merged distinct physical states, no head could separate them, so the
margin/adversarial losses bottomed out with residual violations — the pockets.
The same losses through LoRA can *move the merged latents apart* at the source.
The frozen-vs-LoRA comparison is automatic and already priced: Phase C v2's
1/16 **is** the frozen arm of the ablation.

## Data: raw frames, not the latent cache

The latent cache stores frozen-encoder `z` — invalid the moment the encoder
trains. `scripts/38_train_encoder_lora.py` therefore consumes **frames**:

- **Expert transitions** — `data.loaders.iterate_metaworld_trajectories` (the
  scripts/03 source): frames kept in RAM as uint8, states give object labels
  (`OBJECT_SLICE`) and transition structure for terms 1–3.
- **Off-policy frames** — `scripts/_offpolicy_frames.collect_offpolicy_frames`
  gains `return_frames=True` (it already renders; today it throws frames away
  after encoding). Mixed at `--offpolicy-frac` (default 0.5), the 3b recipe.
  These contribute grounding rows (term 1) on the distribution CEM scores.
- **Adversarial elites** *(round ≥1)* — `mine_cem_frames(keep_frames=True)`
  buffers, re-encoded live.

Grad-enabled encoding goes through `adapter.encpred.encode` — the same upstream
API `adapter.encode` wraps (that wrapper is `@torch.no_grad()`), so the
training-time pipeline (÷255 + transform + encoder) is bit-identical to eval.

## Gate (unchanged, that's the point)

`scripts/30_latent_oracle.py` gains `--encoder-lora <ckpt>`: inject + load after
`build_adapter`, before any encoding. Because the gate encodes every CEM candidate
live through `adapter.encode`, the whole ladder — goal encoding, candidate scoring
— runs in the new representation with **zero further changes**. Same CEM budget
(100×6, H=6, nas=3), strict success, 16 episodes, tasks incl. `mw-reach`:

```bash
python scripts/30_latent_oracle.py --config configs/diagnostic_metaworld.yaml \
    --model dino_wm_metaworld --cost phi --repr-adapter checkpoints/phi_enclora_v0.pt \
    --encoder-lora checkpoints/encoder_lora_v0.pt --strict-success \
    --out results/latent_oracle_phi_enclora.csv
```

Read-out:
- **push/pick > 2/16** (above every frozen arm) → the geometry lever works; carry
  into the closed loop (needs a predictor-compat step, below).
- **reach ≥ 13/16 must hold** — the generality check that the reshape did not
  trade the hand subspace for the object subspace.
- Also run `--cost l2 --encoder-lora` (cheap): if even *plain L2* becomes
  plannable in the reshaped space, that is the cleanest possible headline.
- Still 0–2/16 → even the encoder-level reshape at LoRA rank is insufficient;
  the honest escalation is full encoder finetuning / retraining with the
  proposal's objectives (a different paper scale), and the negative-result paper
  shape stands with one more (strong) rung.

## Predictor compatibility (explicitly out of scope for the gate)

The latent-oracle gate never calls the predictor — perfect dynamics via
sim→render→encode. If the gate passes, the closed loop needs `F` to be consistent
with the new `z̃`: either the preservation term already suffices (measure: the
`check_normalization`-style one-step MSE with LoRA on), or refit the existing
predictor-LoRA (`scripts/26`) on top. Decided *after* a gate pass; no H100 spent
on it before.

## Files

- **New:** `models/heads/lora_encoder.py` (inject/save/load LoRA rooted at
  `adapter.wm.encoder`, reusing `LoRALinear`),
  `scripts/38_train_encoder_lora.py`, `tests/test_lora_encoder.py`,
  `scripts/slurm_phaseD_encoder.sh`.
- **Modified:** `scripts/30_latent_oracle.py` (`--encoder-lora`),
  `scripts/_offpolicy_frames.py` (`return_frames=`, backward-compatible),
  `scripts/_cem_mining.py` + `cem_plan_latent` (`keep_frames=` /
  `frames_elite=` kwarg, signature-detected so every existing hook and
  `tests/test_cem_mining.py` stay valid), `scripts/35` (`--keep-frames`).
- **Reused unchanged:** `LoRALinear`/`set_lora_enabled` (lora_predictor),
  `ActionReprAdapter`/`margin_loss` + the scripts/37 φ-checkpoint format (so
  `load_repr_adapter` and the `--cost phi` arm work as-is),
  `hard_effect_negative`, `collect_offpolicy_frames`, the whole scripts/30 gate.

## Verification

1. Offline: `tests/test_lora_encoder.py` — zero-init identity (encode with LoRA
   disabled == injected-but-zero == never-injected), toggle round-trip, state-dict
   save/load by module name, injection targets found on a synthetic ViT-like
   encoder. Pure CPU, no upstream.
2. `pytest tests/` green (hook backward-compat in `test_cem_mining.py` included).
3. Server, before training: `scripts/check_normalization.py` with LoRA injected
   but **disabled** must reproduce the baseline number exactly (identity check on
   the real checkpoint).
4. Train (~2–4 GPU-hr): held-out prints — grounding decode (median cm, <5/<7cm),
   cf-margin satisfaction, ranking acc + monotone Spearman, preservation drift.
   Grounding must at least match Phase C v2 before spending the gate hour.
5. Gate: `scripts/30 --cost phi --encoder-lora` (+ the `--cost l2` variant),
   push/pick vs the frozen ladder, reach intact. Report next to the state-oracle
   ceiling (16/16, 11/16) per compare-against-paper-numbers.
