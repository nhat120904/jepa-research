# Fully self-supervised trajectory-target pilot — result (job 53275)

Date: 2026-09-20  
Protocol: `docs/TRAJECTORY_SSL_STEP3_PROTOCOL.md`  
Artifact: `/mnt/data/nhatnc129/jepa/latent_scope_trajectory_ssl/step3_full_53275/result.json`

## Verdict

`NO_SIGNATURE_MECHANISM_LEAD_IN_STEP3_PILOT`

The fixed degree-2 trajectory signature did not satisfy any of the four locked lead
conditions. The ordered future-frame target was the strongest and most stable arm. A
learned compact summary also failed to improve over it.

This pilot used no success, reward, contact, coverage, value, or language-goal label. It
did not load the held-out test episodes or the earlier grounded checkpoints.

## Setup

- 345 train and 76 development-validation episodes; 191,900 cached 20 Hz frames.
- A shared self-supervised frame autoencoder compressed DINOv2 patch features and
  proprioception to 16 dimensions. Its train/validation reconstruction losses were
  0.192/0.217.
- Every arm received the same causal history and proposed actions and used a matched
  predictor budget (1.09--1.11 M parameters).
- Training lengths were 32 and 64 steps. Evaluation included 128 steps and the unseen
  split 64 = 24 + 40.
- `ordered_frames`: predict 20 ordered future frame latents.
- `learned_summary`: predict four learned summary tokens trained to reconstruct the same
  future latents.
- `signature_d2`: predict a fixed degree-2 signature of integrated future latents; combine
  adjacent segments with the exact Chen product.

## Main results

All numbers below are common future-latent MSE, so the three targets are compared through
the same decoded quantity.

### Ceiling with the true target

| Horizon | Ordered frames | Learned summary | Signature d2 |
|---:|---:|---:|---:|
| 32 | 0.000 | 0.0137 | 0.0361 |
| 64 | 0.000 | 0.0252 | 0.0583 |
| 128 | 0.000 | 0.2724 | 2.6658 |

At the unseen 128-step horizon, the true signature target itself decoded 9.79 times worse
than the learned-summary target. Thus the signature representation/readout did not
generalize in length even before action-conditioned prediction error was introduced.

### Direct action-conditioned prediction

| Horizon | Ordered frames | Learned summary | Signature d2 |
|---:|---:|---:|---:|
| 32 | **0.1087** | 0.1090 | 0.1323 |
| 64 | **0.1423** | 0.1478 | 0.1606 |
| 128 | **0.2641** | 0.4875 | 0.2944 |

The compact targets gave no accuracy advantage over directly predicting an ordered future
sequence.

### Split rollout and composition

| Evaluation | Ordered frames | Learned summary | Signature sequential | Signature fixed composition |
|---|---:|---:|---:|---:|
| 64 = 32 + 32 | **0.1401** | 0.1404 | 0.1638 | 0.1628 |
| 64 = 24 + 40 | **0.1399** | 0.2830 | 0.2297 | 0.2685 |
| 128 = 64 + 64 | **0.1788** | 0.1860 | 0.1905 | 2.5112 |

On the seen 32 + 32 structure, exact composition was stable but remained 16% worse than
ordered frames. On the unseen 24 + 40 split it was 92% worse than the best baseline and
17% worse than its own sequential rollout. At 64 + 64 it was about 14 times the best
baseline and 13 times its own sequential rollout.

## Locked decision gates

| Gate | Result |
|---|---|
| Signature 128-step target ceiling <= 1.2x summary ceiling | Fail: 9.79x |
| Fixed composition no worse than best baseline on both unseen splits | Fail |
| At least 10% better on one unseen split | Fail |
| Fixed composition within 5% of its own sequential decode | Fail on both unseen splits |

## Interpretation

The algebra implementation is not the failure: Step 2 verified Chen composition to
numerical precision. The problem is that the bilinear cross term in exact composition
amplifies small errors in two predicted signatures. The composed value also moves outside
the target/readout distribution at an unseen total length. A mathematically exact
composition law therefore does not imply robust composition of noisy learned predictions.

This result rejects only the tested formulation: degree-2 signature of an integrated
16-dimensional latent, its current readout, and this Scrub development split. It does not
prove that every trajectory JEPA or every self-supervised world model is impossible.
However, the central proposed advantage did not appear: neither a learned summary nor a
fixed compositional summary beat the ordered-sequence target. The branch has not earned a
planning experiment or a broad hyperparameter sweep.

## Recommended decision

Pause the compositional-summary branch in its current form. Keep the ordered-sequence
self-supervised predictor as the world-model baseline. Separately evaluate a
history-conditioned VLA if the practical goal is task success: memory and world-model
planning answer different questions, and this pilot did not test policy success.

