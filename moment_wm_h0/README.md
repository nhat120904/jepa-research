# Moment-Regularized Context World Model: H0 gates

This directory contains the staged falsification protocol for the proposed
moment-regularized visual world model.  The first gate deliberately contains no
learned model: it asks whether episode-specific drag creates enough decision
room to justify any representation/objective work.

## Scope and provenance

POKEWORLD's official implementation was not public when this gate was written
(the paper says that code will be released).  `scripts/gate1_decision_room.py`
is therefore a minimal, independent implementation of the dynamics specified
in Appendix A of arXiv:2607.27017:

- semi-implicit Euler, control `dt=0.05`, 20 substeps;
- force-controlled circular finger, mass 1, radius 0.06;
- circular object, radius 0.09, drag force `-gamma * mass * velocity`;
- Hertzian normal contact `k * overlap**1.5`, with damping ratio 0.25.

The gate fixes mass and stiffness and varies only `gamma ~ U[0.5, 4.0]`.
It is a specification-faithful preliminary test, not an official POKEWORLD
reproduction.  Once the official code is available, this gate must be rerun in
that implementation before it is paper evidence.

## Gate 1: decision room

Oracle-drag and population-median-drag planners receive identical full states,
goals, CEM budget, standardized CEM noise, and all non-drag physics.  Both use
best-candidate CEM over smooth two-knot force programs.  Their chosen programs
are evaluated in the true simulator.

Primary estimand:

`E[(cost_median - cost_oracle) / max(cost_median, 0.05)]`.

The gate passes only if:

1. the paired 95% bootstrap CI for `cost_median - cost_oracle` excludes zero;
2. mean relative cost reduction is at least 10%.

If either condition fails, work stops before anchor/model training.  Goal-region
reach, terminal distance, and drag-bin results are secondary diagnostics.

## Cluster-only execution

Do not execute the Python script on the login node.  Submit the wrappers:

```bash
sbatch moment_wm_h0/scripts/slurm_gate1_smoke.sh
sbatch moment_wm_h0/scripts/slurm_gate1_full.sh
```

Job IDs, exact commands, output paths, dependencies, and terminal states are
recorded in `docs/JOB_LEDGER.md`.

## Gate 2: anchor certificate

The second gate generates episode-split, 17-frame free-glide windows, then
compares matched recurrent recoverability probes over raw pixels, frozen
DINOv2 spatial features, frozen RAFT flow, pixel centroids, and privileged
state.  The best raw-visual estimator (raw CNN-GRU, centroid GRU, or analytic
centroid glide-decay estimator) and at least one frozen candidate anchor must
reach held-out `R^2 >= 0.40`; otherwise model training stops or the observation design is
repaired.  A passing probe certifies available information only, not functional
use by a world model.

## Gate 3: strong baselines before MMR

The control corpus pairs a 17-frame identification glide with twelve
independent smooth two-knot finger-force queries under the same hidden drag.
Frozen DINOv2 spatial targets are compressed by a train-only PCA and predicted
at direct horizons 7/14/27/28.  A train-only frozen ridge readout maps those
targets to object position for planning.

Three arms share the forward architecture, data, histories, seeds, and CEM:

- `mse`: ordinary deterministic frozen-anchor prediction;
- `cov_mse`: PCA-diagonal covariance-whitened residual MSE;
- `cadm`: the same context/forward model plus the CaDM forward + 0.5 backward
  multi-horizon objective.  As in CaDM, temporal feature differences feed the
  context encoder.

If the best strong baseline recovers at least 80% of the Gate-1
oracle-vs-median true-cost gap, the MMR branch stops for lack of causal room.

## Gate 4: kernel MMR

The MMR arm adds an RBF-kernel conditional-moment penalty to ordinary anchor
MSE.  Instruments are fixed (context-difference summaries, current frozen
anchor, and action knots); residuals are PCA-covariance-whitened.  The moment
term is the off-diagonal U-statistic, so its diagonal does not silently become
another MSE term.  Lambdas `{0.1, 1, 10}` are selected by mean decoded endpoint
error across three seeds on validation episodes only.

MMR passes only if its paired true-cost improvement over the best Gate-3 arm
has a 95% CI strictly above zero and adds at least 10% oracle-gap recovery.
