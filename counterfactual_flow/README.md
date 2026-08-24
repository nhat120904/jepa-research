# Counterfactual Anti-Proxy Policy Distillation

This independent project contains the first gate for a new real-time policy
paper.  It does **not** alter the diagnostic or the current TMLR paper.

## Phase 0: OGBench-Cube counterfactual mining

For each locked OGBench-Cube snapshot, `scripts/mine_ogb_counterfactuals.py`
replays two persisted LeWM CEM populations from a complete MuJoCo state:

- `cem_initial`: the original standard-Gaussian proposal population (the
  random-proposal control);
- `cem_final`: the population after proxy-guided CEM refitting.

It records the frozen planner's predicted terminal latent cost together with
the exact physical endpoint distance and success label.  A mined group contains
a physical-best positive, a low-proxy/high-physical-regret candidate, and a
physical-regret-matched proposal control.  Policy training is deliberately out
of scope until this gate establishes that the desired supervision exists.

The executor imports the repository's corrected OGBench reset routine from
`diagnosis/scripts/76_ogb_true_endpoint_corrected.py`.  It never calls the
retracted Stage-0 reset path.  Every task replays one action sequence twice as
an integrity gate.

## Run sequence

All model/MuJoCo work belongs on a Slurm GPU node.

```bash
cd /home/nhatnc129/nhat.nc/jepa-research
sbatch counterfactual_flow/scripts/slurm_phase0_smoke.sh
sbatch --dependency=afterok:<SMOKE_JOB> counterfactual_flow/scripts/slurm_phase0_array.sh
sbatch --dependency=afterok:<ARRAY_JOB> counterfactual_flow/scripts/slurm_phase0_aggregate.sh
```

Outputs are written under `counterfactual_flow/outputs/ogbench_cube_phase0/`.
The command lines, dependencies, output paths, and terminal states are kept in
`JOB_LEDGER.md`.

## Phase 0b: pairwise ordinal-inversion verification

Phase 0's low-proxy/high-regret gate is necessary but does not, by itself,
prove that the proxy reverses the ordering of a specific pair.  Phase 0b reads
the persisted candidate tables without re-running simulation and verifies
within-population pairs where the proxy prefers one action while exact physics
prefers another by at least 2 cm.  It also constructs the 2x2
proxy-good/proxy-bad × physical-good/physical-hard table and regret-matched,
proxy-rejected hard controls from the same population. A proxy-rejected control
is not asserted to be globally inversion-free; it is simply not among the
actions that the proxy itself rates highly.

```bash
sbatch counterfactual_flow/scripts/slurm_phase0b_inversions.sh
```

Outputs are written to `outputs/ogbench_cube_phase0/phase0b/`.  They verify
that inversions exist in the saved pools; they are not policy-improvement or
physical-query-efficiency evidence.

## Phase 0c: selection-enrichment audit

Phase 0c compares the proxy argmin and proxy-top-decile in each saved initial
and final population. It reports snapshot-paired final-minus-initial physical
regret, rank, false-elite rate, and verified-inversion rate. The original CEM
solver's final proposal mean was not persisted, so this audit explicitly does
not call the final-pool argmin the deployed action.

```bash
sbatch counterfactual_flow/scripts/slurm_phase0c_selection_enrichment.sh
```

Outputs are written to `outputs/ogbench_cube_phase0/phase0c/`.

## Phase 0d: exact deployed CEM plan

Phase 0d re-runs frozen LeWM CEM and records `CEMSolver.solve(...)["actions"]`,
which is the solver's final elite-refitted proposal mean. It snapshot/restores
the physical world to replay that exact plan and the final sampled population.
The aggregate tests selection regret of the returned plan relative to the
physical-best candidate and whether a proxy-rejected candidate is at least 2 cm
better physically. A clean pass opens only the small, matched-budget BC pilot;
it is still not policy-improvement evidence.

```bash
sbatch counterfactual_flow/scripts/slurm_phase0d_smoke.sh
sbatch --dependency=afterok:<SMOKE_JOB> counterfactual_flow/scripts/slurm_phase0d_array.sh
sbatch --dependency=afterok:<ARRAY_JOB> counterfactual_flow/scripts/slurm_phase0d_aggregate.sh
```

## Phase 1a: locked matched-budget acquisition gate

Phase 0d established that a physically better, proxy-rejected corrective often
exists, but it found that corrective by replaying all 300 final candidates.
Phase 1a tests an attainable acquisition rule on 128 fresh episodes, disjoint
from the Phase-0d audit episodes. The rule first runs frozen CEM, takes its
returned mean as the anchor, partitions the proxy-rejected final candidates
into eight proxy-rank strata, and chooses the most action-novel candidate from
each stratum. All choices are fixed before MuJoCo is queried.

Every arm is charged the same nine-branch physical budget: the returned CEM
anchor plus eight alternatives. The preregistered controls are a uniform final
population sample, the eight most proxy-rejected candidates, and action-space
farthest-point coverage. The primary metric is the paired gain over random in
the rate of finding an at-least-2-cm proxy-rejected corrective and its best
physical advantage.

```bash
sbatch counterfactual_flow/scripts/slurm_phase1a_manifest.sh
sbatch --dependency=afterok:<MANIFEST_JOB> counterfactual_flow/scripts/slurm_phase1a_smoke.sh
sbatch --dependency=afterok:<SMOKE_JOB> counterfactual_flow/scripts/slurm_phase1a_array.sh
sbatch --dependency=afterok:<ARRAY_JOB> counterfactual_flow/scripts/slurm_phase1a_aggregate.sh
```

Only `GO_MATCHED_BUDGET_POLICY_PILOT` authorizes the subsequent policy-training
experiment. A Phase-1a acquisition result is still not a learned-policy claim.

## Phase 1a-v2: proxy-instability replication

The first Phase-1a selector did not beat random with a CI-clean margin and pure
action diversity was strongest in mean. Phase 1a-v2 is one fresh, locked
replication on another 128 episodes disjoint from both earlier cohorts. Its
primary selector estimates local proxy instability by perturbing final-CEM
actions at one quarter of the final proposal scale, then applies action coverage
inside the most unstable quartile. A secondary selector tests disagreement
between proxy rank and final-proposal likelihood.

The primary must beat both random and pure action diversity with CI-clean gains
in inversion hit rate and physical corrective advantage. Failure yields
`STOP_NO_ROBUST_MODEL_SIGNAL_BEYOND_DIVERSITY`, which blocks policy training for
this acquisition formulation.

## Decision boundary

The aggregate reports an exploratory GO only when at least 8/32 snapshots have
a final-population low-proxy candidate with at least 2 cm physical regret and a
proposal-population control can be matched within 1 cm physical regret.  This
is a dataset-construction gate, not a paper claim or a substitute for held-out
policy evaluation.
