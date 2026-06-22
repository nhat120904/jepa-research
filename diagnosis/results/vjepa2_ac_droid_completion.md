# DROID completion: `vjepa2_ac_droid` (V-JEPA-2-AC ViT-G/16, ~1B) — 2026-06-22

The last hardware-blocked baseline is now measured. Run on a SLURM H100 80GB node
(8×H100/node cluster); the 24 GB-VRAM requirement that previously gated this leg is gone.
Both DROID baselines (`dino_wm_droid` DINOv2 ViT-S/14 22M and `vjepa2_ac_droid`
V-JEPA-2 ViT-G/16 1.01B) were extracted + scored on the **same** 333-episode /
2331-transition subset, fresh, on one machine.

> Numbers below are the **gripper-primary stratifier** (post origin/main merge): the
> 04-regime assignment now prioritises the gripper channel, shifting per-regime counts
> (free_space 998→409, contact 415→1000 for dino) but leaving effect-conditioned CRA
> essentially invariant. Latent caches were reused; only 04/05/12 were re-run
> (`scripts/slurm_reconcile.sh`, job 21780).

## Headline: scaling the encoder 45× (22M → 1.01B) does NOT fix action-grounding

**Effect-conditioned CRA (top-1, 16-way; primary decision signal), trajectory-clustered CI:**

| regime | strategy | dino_wm_droid CRA_eff | vjepa2_ac_droid CRA_eff [95% CI] |
| --- | --- | ---: | --- |
| pre_grasp | random | 0.374 | 0.140 [0.114, 0.165] |
| pre_grasp | hard_nn | 0.068 | **0.061 [0.044, 0.079]** |
| pre_grasp | opposite | 0.050 | 0.065 [0.046, 0.086] |
| contact_manipulation | random | 0.448 | 0.110 [0.075, 0.145] |
| contact_manipulation | hard_nn | 0.055 | **0.035 [0.015, 0.058]** |
| gripper_actuation | hard_nn | 0.031 | 0.049 [0.028, 0.072] |

16-way chance ≈ 0.0625. V-JEPA-2-AC sits **at or below the chance floor** for the hard
negatives (hard_nn / hard_effect) across pre_grasp, gripper_actuation and contact —
i.e. it cannot tell its factual action from a similar-state alternative. AUG ≈ 0 in
every regime (0.000–0.004; slightly negative in contact), vs DINO-WM's small-but-positive
0.005–0.055. The 1B model is, if anything, *more* action-blind on this metric.

## Boundary Blindness reproduces in the 1B model

`12_boundary_diagnostic.py` (`results/droid_boundary.csv`), BB_boundary per regime:

| regime | BB_boundary | n_b |
| --- | ---: | ---: |
| **pre_grasp** | **+1.933** | 133 |
| free_space | +1.010 | 186 |
| gripper_actuation | +0.643 | 30 |
| contact_manipulation | +0.390 | 234 |

The pre-grasp boundary is the blindness locus (≈1.9× free_space) — the same signature
DINO-WM / JEPA-WM showed on Metaworld (pre_grasp 1.32/1.28 vs free_space 0.28). DROID
uses the `‖Δz‖` latent proxy for the true outcome (no object GT), so this is a transfer
check, not the boundary *proof*; it agrees with the Metaworld proof.

## Sanity gate — the null is REAL, not a plumbing artifact

`terver_gripper_test.py` (512 transitions, open vs close gripper), **PASS** for both. The
point is the *magnitudes* for `vjepa2_ac_droid`:

```
fact_vs_zero  = 187.5   # prediction DOES move with the action (not collapsed)
open_vs_close =  39.8   # responds to the max gripper action difference
fact_to_next  = 533.9   # factual-action prediction error vs true next latent
zero_to_next  = 525.3   # ZERO-action prediction error vs true next latent
```

The model is not ignoring actions (response ≫ 0, plumbing consistent: gripper-delta err
6e-8, cache/loader err 0). But **`fact_to_next` ≈ `zero_to_next`** — using the *true*
action gets the prediction no closer to the true future than feeding a *zero* action
(actually 8.6 units farther). The action response exists but is mis-directed: it does not
track the real action→outcome map. That is precisely the action-grounding-failure thesis,
now confirmed for V-JEPA-2-AC. (DINO-WM: 578.6 vs 590.1 — factual a hair *better* than
zero, consistent with its marginally higher CRA; still negligible against the ~580 error
scale.) Action norm for DROID is identity (mean 0/std 1), confirmed in the model load log.

## Reproduce

```bash
# env on the H100 cluster (see results note / memory for the fbaipublicfiles workaround):
#   TORCH_HOME, HF_HOME -> /mnt/data/nhatnc129/jepa/cache; JEPAWM_OSSCKPT -> .../ossckpt
#   backbones+decoders pre-staged from HF (cluster blocks dl.fbaipublicfiles.com)
sbatch scripts/slurm_droid_pipeline.sh    # 03 -> 04 -> 05 -> 12, both models, ~33 min
sbatch scripts/slurm_sanity.sh            # terver gripper sensitivity, both models
```

Artifacts: `results/droid_diagnostic.csv` (32 rows, both models), `results/droid_boundary.csv`
(8 rows), latent caches `data/precomputed_latents/droid__{dino_wm,vjepa2_ac}_droid.h5`.

## Still open (not hardware-blocked)
- `jepa_wm_droid` — gated DINOv3 ViT-L `.pth` weights (Meta request form), a licensing
  blocker, not hardware (HANDOFF_DROID §1).
- Planning probe for `vjepa2_ac_droid` (`08`/`09`) — optional, the BB framing is the
  headline now.
