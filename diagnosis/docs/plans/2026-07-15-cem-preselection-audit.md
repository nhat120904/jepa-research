# CEM same-population error-pocket audit

**Date locked:** 2026-07-15, before submission of the jobs below.

## Question

The existing comparison (about 2 cm object decode error on random off-policy
frames versus 6--8 cm on CEM elites) is not paired: the frame distributions,
horizons, and action proposals differ.  It therefore does not by itself prove
that CEM top-k selection enriches readout error.  The 6-iteration instrumented
run only measures a smaller first-elite-to-final-elite shift (6.01 to 6.84 cm).

This audit asks, on the **same simulator-rolled candidate population before
selection**, whether candidates selected by the proxy have larger object-readout
error and larger proxy optimism than the population and than candidates selected
by simulator truth.

The tested object is the composite encoder--probe--cost interface.  A positive
result must not be described as encoder-only exploitation.

## Locked Stage-A protocol

- Models: `dino_wm_metaworld`, `jepa_wm_metaworld`.
- Tasks: `mw-push`, `mw-pick-place`.
- Costs: probe-based state cost (primary) and latent L2 (negative/mechanism
  control).
- Seeds: `41000..41015`, fixed before result inspection.
- Planner: CEM, 100 candidates, 6 iterations, top 10%, horizon 6, strict
  episode-end success.
- Dynamics: literal MetaWorld rollout from a saved state; each final frame is
  encoded by the released frozen encoder.  The learned predictor is not called.
- For all 100 candidates before each top-k selection, store action hash, proxy
  cost, simulator-state costs, object/EE probe predictions and errors, exact
  success flags, and decoded-versus-true cost components.
- Inference: average replans within episode seed, then bootstrap episode seeds.
  Replans and candidates are never treated as independent observations.

## Primary estimands and decision rule

At CEM iteration 0, on each identical candidate population:

1. `median object error(proxy top-10%) - median object error(all candidates)`;
2. `mean optimism(proxy top-10%) - mean optimism(all candidates)`, where
   optimism is `true shaped cost - decoded state-probe cost` in metres.

Call the result **immediate selection of error pockets** only when both
seed-clustered 95% CIs are entirely above zero for the primary DINO push cell.
Require directionally consistent replication on JEPA push and at least one
pick-place cell for a cross-checkpoint/contact claim.  If the primary test is
null, the random-reference-to-elite gap remains a distribution/horizon shift,
not evidence of top-k exploitation.

Secondary estimands are proxy-top-k versus true-state-top-k error on the same
population, proxy/true Spearman and top-k overlap, proxy-selected physical
regret, and final-minus-first enrichment.  Call the latter **adaptive
amplification** only when enrichment increases after iteration 0 with a
seed-clustered CI above zero; a large iteration-0 effect with a flat later curve
means selection is immediate but not progressively deepened.

## Stage-B discriminator

A synchronized shared-noise branch starts from the same snapshot, mean,
variance, and iteration-0 candidates.  A state-probe branch and a true-state
branch refit on their own top-k candidates while using paired Gaussian noise at
subsequent iterations.  Dual-scoring both branches separates within-pool
selection from cost-guided proposal collapse.  Stage B is confirmatory for the
dynamic mechanism; Stage A is sufficient to answer the immediate-selection
question.

The locked Stage-B pilot uses both checkpoints and both contact tasks, seeds
`42000..42007`, 100 candidates, 6 CEM iterations, and a `true_state` carrier.
The carrier is executed only to visit informative approach/contact snapshots;
all branch contrasts remain paired within each snapshot.  This is a mechanism
comparison, not a comparison of two independently executed closed-loop agents.

## Files and expected outputs

- GPU runner: `scripts/slurm_cem_preselection_audit.sh`.
- CPU analysis: `scripts/slurm_cem_preselection_analysis.sh`.
- Shared branch runner/analysis: `scripts/slurm_shared_population_branch.sh`
  and `scripts/slurm_shared_population_branch_analysis.sh`.
- Candidate outputs: `results/cem_preselection_<tag>_candidates.csv.gz`.
- Report: `results/cem_preselection_audit.md` plus population, summary, and
  paired first/final CSVs.

Job IDs, dependencies, final state, and exact submission commands are recorded
in `../JOB_LEDGER_2026-07-13.md` after submission.

## Submission record

- `27990`: `sbatch scripts/slurm_cem_preselection_audit.sh` (8-cell GPU array).
- `27991`: `sbatch --dependency=afterok:27990 scripts/slurm_cem_preselection_analysis.sh`.
- `27994`: `sbatch scripts/slurm_shared_population_branch.sh` (4-cell GPU array).
- `27995`: `sbatch --dependency=afterok:27994 scripts/slurm_shared_population_branch_analysis.sh`.
- `28009`: releases Stage B only after successful Stage A.
- `28010`: restores temporarily held unrelated queued work after Stage B terminates.
