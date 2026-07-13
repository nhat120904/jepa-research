# ACID-style inverse-consistency baseline (2026-07-13)

## Purpose

This is a concurrent-method baseline for the paper's cost-exploitation result. It asks whether
penalizing action-inconsistent predicted transitions prevents the released MetaWorld planners
from selecting bad trajectories. It does **not** claim to reproduce the authors' IDM architecture.

Primary source: G. Seo, D. Kim, and S. Kwak, ["ACID: Action Consistency via Inverse Dynamics for
Planning with World Models"](https://arxiv.org/abs/2607.02403), arXiv:2607.02403v1 (2 July 2026).
The implementation was checked against Eqs. 3--5, Algorithm 1, and Appendix A on 13 July 2026.

## What is faithful

For every candidate latent trajectory, the implementation computes

\[
c_a=\frac{1}{H}\sum_{t=0}^{H-1}\lVert a_t-G(z_t,z_{t+1})\rVert_2^2,
\qquad
c=c_g+\lambda\frac{\sigma_g}{\sigma_a}c_a.
\]

- `c_g` is the existing upstream terminal latent MSE, unchanged.
- `sigma_g` and `sigma_a` are recomputed over the candidate pool at every CEM iteration.
- The residual is evaluated at every predicted model step and uses the raw stacked action chunk
  sampled by CEM (five 4-D MetaWorld actions, so 20 dimensions).
- The terminal and ACID arms use identical environment seeds and CEM noise inside one process.
- `learned` dynamics uses the released predictor. `oracle` dynamics snapshot/restores MuJoCo,
  renders every true model-step boundary, and passes each true frame through the same frozen
  encoder. Thus only trajectory dynamics changes; the terminal latent cost remains identical.
- If `sigma_a` is zero, the consistency term cannot rerank the pool and its weight is set to zero.
  This is the finite, ranking-equivalent limit of the paper's formula.

## Deliberate approximation

The paper's IDM is a four-layer, three-head, width-192 prefix/suffix transformer trained by flow
matching, with a one-step Euler sampler. As of the implementation date, the paper/project page did
not link official code or a released verifier checkpoint. The local `ACIDInverseDynamics` therefore
uses deterministic mean/max pooling of each frozen latent followed by a small MLP trained with
action regression. It has the same `G(z_t,z_t1) -> a_t` contract, but it is an **ACID-style cost
baseline**, not an exact architectural reproduction. Results must carry this qualifier. If official
code/checkpoints appear, replace only the verifier module and keep the cost/evaluation protocol.

The manifest is a deterministic 70/15/15 trajectory split. Training and model selection use only
train/validation trajectories. Test metrics are computed once after checkpoint selection. Online
planning uses new environment seeds `22000:22031`, disjoint from all development seed blocks.

## Files and outputs

- `models/heads/acid_idm.py`: verifier, Eq. 3 residual, Eqs. 4--5 adaptive cost.
- `scripts/51_train_acid_idm.py`: split-safe cache training and one-pass held-out report.
- `scripts/52_eval_acid_baseline.py`: paired learned/oracle planning runner.
- `scripts/slurm_acid_idm_train.sh`: two-model training array.
- `scripts/slurm_acid_baseline_eval.sh`: 2 models x 2 dynamics x 2 contact tasks.
- Checkpoints: `checkpoints/acid_idm_<model>_split0.pt`.
- Results: `results/acid_<model>_<dynamics>_<task>_seed22000_n32.csv`.

Do not execute either Python runner on the login node. Suggested submission (not submitted by the
implementing agent):

```bash
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
TRAIN=$(sbatch --parsable scripts/slurm_acid_idm_train.sh)
sbatch --dependency=afterok:${TRAIN} scripts/slurm_acid_baseline_eval.sh
```

Record the returned job IDs and terminal states in the job ledger.

## Interpretation gates

1. The verifier is usable only if held-out action MSE beats the checkpoint's constant-mean baseline
   and `mean_sigma_acid` is nonzero in planning pools.
2. Compare terminal versus ACID with paired seed-level success differences and an exact McNemar or
   paired bootstrap interval; do not compare only aggregate percentages.
3. Improvement under learned but not oracle dynamics supports ACID's unrealizable-transition
   mechanism. No improvement under learned dynamics is informative: the current paper's oracle
   ladder instead localizes the contact wall to terminal cost geometry/selection.
4. A large change under oracle dynamics is a warning that the approximate verifier penalizes valid
   off-policy transitions or is distribution-shifted; it is not evidence that perfect dynamics are
   inconsistent.
5. Because inverse dynamics may be non-identifiable under partial observability, report the held-out
   verifier error and the approximation caveat next to every planning result.
