# Decision report: Moment-Regularized Context World Model H0

Date: 2026-09-02 UTC

## Decision

**STOP this method branch after Gate 4.**  Hidden episode drag creates a large
oracle planning advantage and is recoverable from frozen visual anchors, but
the tested kernel conditional-moment regularizer does not improve true planning
over an ordinary MSE context world model.  The preregistered Gate-4 conditions
both fail, so extending this branch to a larger robot dataset is not justified
by the H0 evidence.

This is a result about the implemented H0 and objective, not evidence that all
conditional-moment objectives or all hidden-physics world models must fail.

## Gate results

| Gate | Primary result | Decision |
|---|---|---|
| 1. Decision room | Oracle drag lowers true cost from 0.14230 to 0.01420; mean relative reduction 0.793, 95% CI [0.745, 0.837] | GO |
| 2. Anchor recoverability | Held-out drag R2: raw pixels 0.988, DINOv2 0.902, RAFT 0.822, privileged state 1.000 | GO |
| 3. Strong baselines | Best arm is plain MSE: drag R2 0.773, true cost 0.12325, oracle-gap recovery 0.149, 95% CI [-0.010, 0.276] | CONTINUE_TO_MMR |
| 4. Kernel MMR | Validation selects lambda 1.0. True cost 0.12402 versus MSE 0.12325; paired improvement -0.00077, 95% CI [-0.01819, 0.01649] | STOP |

Gate 4 also yields incremental oracle-gap recovery -0.0060 with 95% CI
[-0.1420, 0.1309], below the required 0.10.  Thus neither the positive paired
cost criterion nor the effect-size criterion passes.

## Mechanistic reading

The negative result is not an encoder-ceiling result.  Frozen DINOv2 features
retain enough information to decode episode drag with held-out R2 0.902, and the
learned MSE context representation retains it with R2 0.773.  Nevertheless,
CEM finds action sequences whose predicted cost is very low (0.01276 for MSE)
while their true cost remains high (0.12325).  The dominant failure in this H0
is therefore downstream use/calibration under optimizer selection, not failure
to acquire the hidden scalar.

The kernel MMR term leaves drag decoding essentially unchanged (R2 0.77285)
and does not close that planner-facing proxy--truth gap.  Its slightly better
decoded endpoint MAE (0.10783 versus 0.10908) also does not transfer to lower
true planned cost.  This is exactly why selection used locked true-simulator
planning rather than prediction metrics alone.

## Protocol boundaries

- The simulator is an independent, specification-faithful preliminary
  implementation based on the POKEWORLD paper, not the unreleased official
  code.
- Mass and stiffness are fixed; only episode drag varies.
- The anchor is frozen DINOv2 with train-only PCA and a frozen train-only
  position decoder.
- All methods use the same data, forward capacity, CEM budget, and common
  standardized CEM noise; lambda is selected on validation episodes before the
  locked test evaluation.
- The MMR implementation uses fixed instruments, covariance-whitened residuals,
  an RBF kernel, and the exact off-diagonal U-statistic on top of ordinary MSE.
- All simulation, feature extraction, training, and evaluation ran through
  Slurm compute jobs; the complete record is in `JOB_LEDGER.md`.

## Research consequence

Do not sell this result as "MMR fails" in general, and do not scale this exact
branch yet.  The useful finding is sharper: in this controlled setting,
recovering hidden physics is easy enough, but low average prediction error and
conditional residual moments still do not control errors selected by a
powerful planner.  A subsequent direction should target optimizer-aware
uncertainty, pessimism, or adversarial on-policy coverage directly, with a gate
on paired true planning cost from the start.

## Artifacts

- Gate 1: `../outputs/gate1_full/summary.json`
- Gate 2: `../outputs/gate2_full/certificates/summary.json`
- Gate 3: `../outputs/gate3_full/evaluation/summary.json`
- Gate 4 selection: `../outputs/gate4_full/selection.json`
- Gate 4 locked test: `../outputs/gate4_full/evaluation/summary.json`
