# DROID action-grounding does not scale — 22M → 300M → 1B (2026-06-22)

Four DROID world-model baselines, same 333-episode / 2331-transition subset, same
gripper-primary regimes, scored fresh on one SLURM H100 80GB node. Spans a **45×
encoder scale-up** and **two encoder families** (DINOv2/v3 vs V-JEPA-2):

| model | encoder | enc. params | source |
| --- | --- | ---: | --- |
| `dino_wm_droid` | DINOv2 ViT-S/14 | ~22M | upstream |
| `jepa_wm_droid` | DINOv3 ViT-L/16 | ~300M | DINOv3 .pth reconstructed from HF safetensors mirror (`scripts/convert_dinov3_hf_to_orig.py`; fbaipublicfiles firewalled) |
| `vjepa2_ac_droid` | V-JEPA-2 ViT-G/16 | ~1.01B | upstream |
| `vjepa2_ac_oss` | V-JEPA-2 ViT-G/16 | ~1.01B | public OSS checkpoint (independent training, buggy-loss variant per JEPA-WMs appendix) |

## Headline: every model is action-blind, at every scale

**Effect-conditioned CRA (top-1), `hard_nn` negatives (similar state, must pick the
factual action) — the deterministic, discriminative metric. 16-way chance = 0.0625.**

| regime | DINO-WM 22M | JEPA-WM 300M | V-JEPA2-AC 1B | V-JEPA2-AC-OSS 1B |
| --- | ---: | ---: | ---: | ---: |
| pre_grasp | 0.068 | 0.077 | 0.061 | 0.069 |
| gripper_actuation | 0.031 | 0.059 | 0.049 | 0.031 |
| contact_manipulation | 0.055 | 0.045 | 0.035 | 0.033 |

All four sit **at the 0.0625 chance floor** in the three action-critical regimes. The
300M DINOv3 model is no better than the 22M DINOv2 one; the 1B V-JEPA-2 models are, if
anything, slightly *lower* on contact. The curve is **flat across 45× scale**. (free_space
has n_effect=0 — no above-median-Δz transitions — so it carries no effect-CRA cell.)

## Boundary Blindness reproduces at every scale

**`bb_boundary` per regime (`12_boundary_diagnostic.py`); BB > 0 = boundary-blind.**

| regime | DINO-WM 22M | JEPA-WM 300M | V-JEPA2-AC 1B | V-JEPA2-AC-OSS 1B |
| --- | ---: | ---: | ---: | ---: |
| **pre_grasp** | **+1.916** | **+1.205** | **+1.933** | **+1.560** |
| free_space | +0.963 | +1.103 | +1.010 | +0.819 |
| gripper_actuation | +1.088 | +1.057 | +0.643 | +0.828 |
| contact_manipulation | +0.421 | +0.897 | +0.390 | +0.595 |

BB is **positive in every cell of every model** — all four are boundary-blind. The
pre-grasp boundary is the blindness locus for 3/4 (and second-highest for JEPA-WM), the
same signature seen on Metaworld (pre_grasp 1.32/1.28 vs free_space 0.28). Scaling and
switching encoder family does not close the contact/pre-grasp boundary gap.

## Caveat on the easier negatives

`random` and `opposite` negative sampling is **not seeded** (only the `hard_nn` pool and
the bootstrap CIs are), so those columns drift run-to-run by sampling noise; `hard_nn` /
`hard_effect` are deterministic (NN by latent distance) and reproduce bit-identically —
which is why this writeup leads with `hard_nn`. On `opposite`, JEPA-WM 300M is modestly
higher (pre_grasp 0.122, gripper 0.228, contact 0.214) than the others, but `opposite`
(literally the negated action) is the *easiest* negative and does not probe discriminative
action-grounding; on the hard negatives the 300M model collapses to chance like the rest.

## Takeaway

The action-grounding failure is **not a capacity artifact**. A 45× encoder scale-up
(22M → 300M → 1B) and a switch of self-supervised family (DINO → V-JEPA-2), plus an
independently-trained public checkpoint, all land in the same place: chance-floor
effect-CRA against hard negatives and positive boundary-blindness with the pre-grasp
locus. This turns the prior two-point result into a 4-point curve and rules out "just
scale the encoder" as a fix — consistent with the Boundary-Blindness thesis.

## Reproduce
```bash
sbatch scripts/slurm_droid_pipeline.sh   # dino_wm + vjepa2_ac (03/04/05/12)
sbatch scripts/slurm_droid_oss.sh        # + vjepa2_ac_oss (extract oss, 04/05/12 all)
# jepa_wm_droid (DINOv3 ViT-L): first reconstruct the backbone .pth, then run:
.venv/bin/python scripts/convert_dinov3_hf_to_orig.py   # HF safetensors -> orig .pth
sbatch scripts/slurm_droid_jepawm.sh     # + jepa_wm_droid (extract it, 04/05/12 all)
```
Artifacts: `results/droid_diagnostic.csv` (64 rows = 4 models × 4 strategies × 4 regimes),
`results/droid_boundary.csv` (16 rows). DINOv3 backbone reconstruction is validated by a
strict-load (only buffers missing) + finite forward; end-to-end by a clean 03 extract
(predictor loads "All keys matched", features non-degenerate) and sane regime counts.
