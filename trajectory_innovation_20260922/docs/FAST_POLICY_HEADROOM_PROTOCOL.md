# B: selection headroom of a few-step DDIM policy (PushT, pre-registered screen)

Pinned 2026-09-29, before any code or data for it. It runs in parallel with `LABEL_REGIME_PROTOCOL.md`.

## Why

The closed-loop gain of any selector is capped by the bank's headroom, and that headroom cannot exceed 1 − P0.
With the released 100-step policy, P0 ≈ 62%.

Raising K does not lift the cap:
- Gate A–C measured the per-decision oracle gain rising ×1.42 from K = 8 to K = 32 (0.0074 → 0.0105).
- The expected maximum of K Gaussian draws predicts ×1.45.
- On the same curve, K = 16 gives ×1.24 and K = 64 gives ×1.65.

At Δretention ≈ 0.2, CTA − DIRECT therefore stays at or below 0.2 × 38 ≈ 7.6 pp, whatever K is.

Lowering P0 is the only lever that raises the cap. Few-step sampling does that without training anything, and it
also makes the policy fast, which is when the scorer's latency starts to matter.

The paper story this would support: a few-step policy plus selection reaches the success of the 100-step policy
at lower total latency.

## Sampler change

`PolicyRunner(checkpoint, device, sampler="ddpm" | "ddim", steps=None)`.
- `ddim` builds `DDIMScheduler.from_config(policy.diffusion.noise_scheduler.config)` with eta = 0 and `steps` inference steps.
- Candidates still differ only through the seeded initial noise (`candidate_seed`).
- The full scheduler config is written to every output.
- **Regression test:** `sampler="ddpm"` with default steps reproduces the current banks bitwise on the smoke state.

## Roots

Fresh roots **4000–4049** (n = 50). This range has not been used before. It is used only for this screen and
never again for confirmation.

## Arms

The runtime is the same as W2: `deepcopy_fixed` clones, a nested seeded bank with `bank_size = 8`, and the chosen branch is adopted.

| Arm | Sampler | Selection |
|---|---|---|
| P0@DDPM100 | current policy | candidate 0 (reference) |
| P0@DDIM4 | DDIM, 4 steps | candidate 0; all 8 siblings simulated for the per-decision oracle gain |
| PHYS8@DDIM4 | DDIM, 4 steps | best 8-step coverage |
| P0@DDIM2 | DDIM, 2 steps | candidate 0; all 8 siblings simulated |
| PHYS8@DDIM2 | DDIM, 2 steps | best 8-step coverage |

Latency: wall time of one K = 8 bank per sampler, as the median over decisions on the same MIG slice.

## Pre-registered rules

For each k ∈ {2, 4}:
- H_k = PHYS8_k − P0_k, with a root-paired bootstrap (10,000);
- R_k = PHYS8_k − P0@DDPM100.

**k qualifies** if all of these hold:
- H_k ≥ 0.30;
- the CI lower bound of H_k is ≥ 0.15;
- R_k ≥ −0.05.

The last condition matters because, without it, the recovery story is impossible even with a perfect selector.

Among qualifying k, choose the one with the largest H_k. Ties go to the smaller k.

If no k qualifies, **STOP B**.

Why 30 pp: at Δretention ≈ 0.2 it gives CTA − DIRECT ≈ 6 pp, which ≈ 400 sealed roots × 3 seeds can resolve.

With 50 roots the paired CI half-width is about ±17 pp. This is a screen. Nothing here is a confirmation claim.

Reported, not gating:
- McNemar tests;
- P0 success per sampler;
- the per-decision oracle coverage gain at K = 8;
- latency.

## If B qualifies (not approved here; decided with the user)

- Collect training and development roots with DDIM-k banks on new root ranges.
- Encode, then train CTA and DIRECT, using the label regime chosen from A.
- Run the dev closed loop, then sealed roots 3000–3399 with 3 seeds.
- Estimated at 8–10 GPU-h before the sealed run.

## Compute

- A CPU smoke: 2 roots, 3 decisions, every arm.
- Then an array of 2 × 25 roots: 1× MIG 3g.40gb, 4 CPU, 32 GB, limit 1:45 each.
- No H_rep and no OFFICIAL arm.
- About 1.5 GPU-h.
