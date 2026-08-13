# OGBench-Cube Stage 0: cross-substrate mechanism gate

Date: 2026-08-10. Status: **historical artifact retracted; scoped conclusion
replicated by the corrected 2026-08-11 audit**.

> Candidate-level results recorded below are provenance only because replay did
> not restore full simulator/controller state. Do not cite their numbers. The
> fresh two-world audit independently passes exact renderer/physics gates and
> finds 21/32 true-endpoint successes with 3.29 cm [1.49, 5.67] residual regret.
> Use `2026-08-11-ogb-corrected-true-endpoint-design.md` and
> `../../results/ogb_true_endpoint_corrected/TRUE_ENDPOINT_DECISION.md`.

## Decision being purchased

This stage does not assume a new method and does not use MetaWorld as a primary
benchmark. It asks whether the paper's cost-side mechanism transfers to the
official LeWM OGBench-Cube stack:

> On identical start snapshots and identical candidate action sequences, does
> latent goal distance misrank the candidates' true simulator outcomes, and is
> there enough candidate-set headroom for a true physical cost to select a
> materially better outcome?

If the answer is no, planner-aware representation/cost training is not justified
by the current evidence outside MetaWorld. If learned rollout error explains the
gap instead, the relevant comparison is AWM-style dynamics adaptation. If good
outcomes are absent from the candidate set, the relevant comparison is proposal
methods such as IMWM/SAGE rather than a representation method.

Stage 0 is deliberately method-agnostic because simulator-assisted iterative
training changes the problem setting from offline zero-shot planning and does not
yet have a sufficiently clear novelty claim.

## Upstream lock

- LeWM: `lucas-maes/le-wm`, commit
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- stable-worldmodel: `galilai-group/stable-worldmodel`, commit
  `9a66d7d020043c8efb507f45373e808714f0842d`.
- stable-pretraining: `galilai-group/stable-pretraining`, commit
  `9aa93f8b6153eebb73f57d4853ccf8a13d848310`.
- Model: `quentinll/lewm-cube`, HF revision
  `b0747c5002e86d2ce8f3cd8178004b97524c587d`.
- Dataset: `quentinll/lewm-cube`, HF dataset revision
  `02a19a67a0dc8c9d6215f89c19e0a597691e152a`;
  archive SHA256 is computed after download and recorded in the job log; the
  immutable HF revision pins the source object.

External repositories and generated environments/data live under ignored
`diagnosis/external/` and `/mnt/data/nhatnc129/jepa/lewm_stage0/`. No model,
simulator, HDF5 scan, or sustained analysis runs on the login node.

## Locked protocol

### Gate 0A: official baseline reproduction

Run the official stable-worldmodel LeWM evaluator on `swm/OGBCube-v0` with:

- 50 evaluation episodes, seed 42;
- goal offset 25 and evaluation budget 50;
- horizon 5, action block 5, receding horizon 5;
- CEM 300 candidates, 30 iterations, top-k 30, variance 1.0;
- official `quentinll/lewm-cube` checkpoint and ImageNet preprocessing.

The baseline must run without API or normalization changes. A difference from
the paper number is reported, not tuned away. The reference shown for LeWM on
OGBench-Cube in Figure 6 is 74% success; that paper value averages three
training seeds, while the released HF checkpoint is one concrete checkpoint,
so this is a sanity target rather than an equality assertion.

### Gate 0B: matched-candidate audit

Use precommitted evaluation start rows selected deterministically from the
official dataset by a persisted manifest. These rows are not claimed to be
held out from LeWM training. At each selected snapshot:

1. load the start simulator state and goal exactly as the official evaluator;
2. obtain a candidate action population from the released LeWM/CEM stack;
3. restore the same simulator snapshot for every candidate;
4. execute the same action block sequence in the true simulator;
5. render and encode the true endpoint with the released LeWM encoder;
6. attach both costs to every candidate:
   - `latent_true_endpoint_cost`: latent MSE between the encoded true endpoint
     and encoded goal;
   - `physical_cost`: cube-to-target Euclidean distance in metres;
7. retain the learned rollout cost produced for that exact action sequence.

The smoke and locked audit record both initial and final CEM populations without
merging different closed-loop states. The unit of uncertainty is the evaluation
start snapshot, not the candidate row.

### Gate 0C: interpretation

Primary candidate-set quantities, all computed per snapshot before aggregation:

- Spearman correlation of learned-latent and true-endpoint-latent costs with
  physical cost;
- top-10% physical recall;
- physical selection regret of each cost;
- false-elite rate: fraction of proxy top-10% outside physical top-50%;
- physical headroom: median candidate physical cost minus best available cost.

Bootstrap confidence intervals cluster by snapshot. No cross-replan comparison
is used to infer a cost mechanism.

Interpretation:

- a **strong cost/representation pass** requires, on the final population,
  lower 95% cluster-bootstrap bounds above zero for both true-endpoint-latent
  physical selection regret and the paired physical-oracle success advantage;
  the lower bound on oracle candidate success must also be above zero;
- positive physical selection regret without a success gap is recorded only as
  distance-level support, not as sufficient motivation for a top-tier method;
- learned cost bad but true-endpoint latent cost good localizes the issue mainly
  to learned dynamics;
- both latent costs rank well but good candidates are unavailable localizes the
  issue mainly to proposal coverage;
- no mechanism is declared from correlation alone.

## Staging and stop rules

1. Environment/data/model smoke: import, checkpoint, dataset, and one-env reset.
2. Official baseline reproduction.
3. Matched-candidate smoke: two snapshots, 32 candidates, three CEM iterations,
   with both initial and final populations audited.
4. Locked audit: at least 32 precommitted evaluation snapshots after the smoke
   passes.

Stop before the locked audit if snapshot restoration is not deterministic to
`1e-8` in qpos/qvel and pixel equality, candidate action normalization cannot be
matched to the official evaluator, or the official checkpoint/data baseline does
not run. Diagnose rather than silently change the protocol.

## Execution ledger

Every submission must be appended here with exact command, dependency, output,
and terminal state.

| Job | Command | Dependency | Output | State |
|---|---|---|---|---|
| `37956` | `sbatch diagnosis/scripts/slurm_ogb_stage0_setup.sh` | none | `/mnt/data/nhatnc129/jepa/lewm_stage0/{.venv,datasets/}`; log `ogb_stage0_setup_37956.out` | **failed** 2026-08-10: upstream `env` extra pulled `gymnasium[all]`; `box2d-py` required unavailable `swig` |
| `37957` | `sbatch --dependency=afterok:37956 diagnosis/scripts/slurm_ogb_stage0_smoke.sh` | setup | `results/ogb_stage0_smoke.json`; log `ogb_stage0_smoke_37957.out` | dependency never satisfied; cancelled before rerun |
| `37958` | `sbatch --dependency=afterok:37956 diagnosis/scripts/slurm_ogb_stage0_baseline.sh` | setup | official evaluator output `ogb_stage0_baseline.txt`; log `ogb_stage0_baseline_37958.out` | dependency never satisfied; cancelled before rerun |
| `37961` | `sbatch diagnosis/scripts/slurm_ogb_stage0_setup.sh` | none | same setup paths; log `ogb_stage0_setup_37961.out` | submitted 2026-08-10 with Cube-only dependencies |
| `37962` | `sbatch --dependency=afterok:37961 diagnosis/scripts/slurm_ogb_stage0_smoke.sh` | setup rerun | `results/ogb_stage0_smoke.json`; log `ogb_stage0_smoke_37962.out` | submitted 2026-08-10 |
| `37963` | `sbatch --dependency=afterok:37961 diagnosis/scripts/slurm_ogb_stage0_baseline.sh` | setup rerun | official evaluator output `ogb_stage0_baseline.txt`; log `ogb_stage0_baseline_37963.out` | submitted 2026-08-10 |
| `37969` | `sbatch --dependency=afterok:37962 diagnosis/scripts/slurm_ogb_stage0_audit_smoke.sh` | stack smoke | `results/ogb_stage0/audit_smoke/`; artifacts under `$STABLEWM_HOME/artifacts/audit_smoke`; log `ogb_stage0_audit_smoke_37969.out` | submitted 2026-08-10 |
| `37970` | `sbatch --dependency=afterok:37963:37969 diagnosis/scripts/slurm_ogb_stage0_audit_locked.sh` | official baseline and matched-candidate smoke | `results/ogb_stage0/audit_locked/`; artifacts under `$STABLEWM_HOME/artifacts/audit_locked`; log `ogb_stage0_audit_locked_37970.out` | submitted 2026-08-10 |
| `37962,37963,37969,37970` | prior chain above | setup | prior outputs | cancelled before execution: upstream loader resolves HF `main`; inserted an immutable model-revision pin |
| `37972` | `sbatch --dependency=afterok:37961 diagnosis/scripts/slurm_ogb_stage0_pin_model.sh` | setup rerun | exact-revision checkpoint under `$STABLEWM_HOME/checkpoints/models--quentinll--lewm-cube`; log `ogb_stage0_pin_model_37972.out` | submitted 2026-08-10 |
| `37973` | `sbatch --dependency=afterok:37972 diagnosis/scripts/slurm_ogb_stage0_smoke.sh` | exact model pin | `results/ogb_stage0_smoke.json`; log `ogb_stage0_smoke_37973.out` | submitted 2026-08-10 |
| `37974` | `sbatch --dependency=afterok:37972 diagnosis/scripts/slurm_ogb_stage0_baseline.sh` | exact model pin | official evaluator output `ogb_stage0_baseline.txt`; log `ogb_stage0_baseline_37974.out` | submitted 2026-08-10 |
| `37975` | `sbatch --dependency=afterok:37973 diagnosis/scripts/slurm_ogb_stage0_audit_smoke.sh` | exact-revision stack smoke | smoke outputs/artifacts as above; log `ogb_stage0_audit_smoke_37975.out` | submitted 2026-08-10 |
| `37976` | `sbatch --dependency=afterok:37974:37975 diagnosis/scripts/slurm_ogb_stage0_audit_locked.sh` | exact-revision official baseline and audit smoke | locked outputs/artifacts as above; log `ogb_stage0_audit_locked_37976.out` | submitted 2026-08-10 |
| `37961` | setup retry listed above | none | partial archive retained; log `ogb_stage0_setup_37961.out` | **failed** after 25m53s at 92% download: `curl` HTTP/2 stream reset, exit 92 |
| `37972,37973,37974,37975,37976` | prior exact-revision chain | failed setup | no experiment outputs | cancelled before execution for resumable setup retry |
| `37995` | `sbatch diagnosis/scripts/slurm_ogb_stage0_setup.sh` | none | resume same partial archive; log `ogb_stage0_setup_37995.out` | submitted 2026-08-10 with HTTP/1.1 and retry-all-errors |
| `37997` | `sbatch --dependency=afterok:37995 diagnosis/scripts/slurm_ogb_stage0_pin_model.sh` | setup retry | exact checkpoint and environment freeze; log `ogb_stage0_pin_model_37997.out` | submitted 2026-08-10 |
| `37998` | `sbatch --dependency=afterok:37995 diagnosis/scripts/slurm_ogb_stage0_smoke.sh` | setup retry | none | cancelled immediately: missing model-pin dependency |
| `37999` | `sbatch --dependency=afterok:37997 diagnosis/scripts/slurm_ogb_stage0_smoke.sh` | exact model pin | stack smoke output; log `ogb_stage0_smoke_37999.out` | submitted 2026-08-10 |
| `38000` | `sbatch --dependency=afterok:37997 diagnosis/scripts/slurm_ogb_stage0_baseline.sh` | exact model pin | official baseline output; log `ogb_stage0_baseline_38000.out` | submitted 2026-08-10 |
| `38001` | `sbatch --dependency=afterok:37999 diagnosis/scripts/slurm_ogb_stage0_audit_smoke.sh` | stack smoke | matched audit smoke output; log `ogb_stage0_audit_smoke_38001.out` | submitted 2026-08-10 |
| `38002` | `sbatch --dependency=afterok:38000:38001 diagnosis/scripts/slurm_ogb_stage0_audit_locked.sh` | official baseline and audit smoke | locked n=32 output; log `ogb_stage0_audit_locked_38002.out` | submitted 2026-08-10 |
| `37995` | setup retry listed above | none | dataset plus archive SHA256 `3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600` | **completed** in 12m08s |
| `37997` | model pin listed above | setup | exact-revision checkpoint | **completed** in 4s |
| `37999` | exact stack smoke listed above | model pin | smoke log | **failed** in 7s: unconstrained resolver installed CUDA-13 torch on CUDA-12.8 cluster driver |
| `38000` | official baseline listed above | model pin | baseline log | **failed** in 26s: missing `pygame` after narrowing upstream `env` extra; no evaluation ran |
| `38001,38002` | prior audit smoke/locked audit | failed stack smoke | no outputs | cancelled before execution |
| `38003` | `sbatch diagnosis/scripts/slurm_ogb_stage0_fix_runtime.sh` | completed setup/model pin | cu126 torch pair, Cube runtime dependencies, environment freeze; log `ogb_stage0_fix_runtime_38003.out` | submitted 2026-08-10 |
| `38004` | `sbatch --dependency=afterok:38003 diagnosis/scripts/slurm_ogb_stage0_smoke.sh` | runtime fix | stack smoke output; log `ogb_stage0_smoke_38004.out` | submitted 2026-08-10 |
| `38005` | `sbatch --dependency=afterok:38003 diagnosis/scripts/slurm_ogb_stage0_baseline.sh` | runtime fix | official baseline output; log `ogb_stage0_baseline_38005.out` | submitted 2026-08-10 |
| `38006` | `sbatch --dependency=afterok:38004 diagnosis/scripts/slurm_ogb_stage0_audit_smoke.sh` | stack smoke | matched audit smoke output; log `ogb_stage0_audit_smoke_38006.out` | submitted 2026-08-10 |
| `38007` | `sbatch --dependency=afterok:38005:38006 diagnosis/scripts/slurm_ogb_stage0_audit_locked.sh` | official baseline and audit smoke | locked n=32 output; log `ogb_stage0_audit_locked_38007.out` | submitted 2026-08-10 |
| `38003` | runtime fix listed above | completed setup/model pin | environment freeze | **completed** in 2m39s; CUDA-12.6 torch and missing Cube imports fixed |
| `38004` | stack smoke listed above | runtime fix | smoke log | **failed** in 39s: unconstrained `transformers 5.15` changed ViT state-dict names relative to released checkpoint |
| `38005` | official baseline listed above | runtime fix | no result | cancelled after the common checkpoint incompatibility was identified |
| `38006,38007` | prior audit smoke/locked audit | failed stack smoke | no outputs | cancelled before execution |
| `38008` | `sbatch diagnosis/scripts/slurm_ogb_stage0_fix_runtime.sh` | completed setup/model pin | additionally pin `transformers==4.57.1` and CPU checkpoint-load check; log `ogb_stage0_fix_runtime_38008.out` | submitted 2026-08-10 |
| `38009` | `sbatch --dependency=afterok:38008 diagnosis/scripts/slurm_ogb_stage0_smoke.sh` | checkpoint compatibility check | stack smoke output; log `ogb_stage0_smoke_38009.out` | submitted 2026-08-10 |
| `38010` | `sbatch --dependency=afterok:38008 diagnosis/scripts/slurm_ogb_stage0_baseline.sh` | checkpoint compatibility check | official baseline output; log `ogb_stage0_baseline_38010.out` | submitted 2026-08-10 |
| `38011` | `sbatch --dependency=afterok:38009 diagnosis/scripts/slurm_ogb_stage0_audit_smoke.sh` | stack smoke | matched audit smoke output; log `ogb_stage0_audit_smoke_38011.out` | submitted 2026-08-10 |
| `38012` | `sbatch --dependency=afterok:38010:38011 diagnosis/scripts/slurm_ogb_stage0_audit_locked.sh` | official baseline and audit smoke | locked n=32 output; log `ogb_stage0_audit_locked_38012.out` | submitted 2026-08-10 |
| `38008` | checkpoint compatibility fix listed above | completed setup/model pin | pinned environment and CPU checkpoint-load check | **completed** in 49s |
| `38009` | stack smoke listed above | compatibility fix | `results/ogb_stage0_smoke.json` | **completed** in 18s: dataset/model/GPU/env/render pass |
| `38010` | official baseline listed above | compatibility fix | `checkpoints/quentinll/ogb_stage0_baseline.txt` | **completed** in 4m19s: 68% success (34/50), evaluation 205.3s |
| `38011` | matched audit smoke listed above | stack smoke | smoke summary and artifacts | **completed** in 1m44s: exact restoration and normalization gates pass |
| `38012` | original monolithic locked audit | baseline and audit smoke | one partial population artifact only | cancelled after 4m46s to shard independent snapshots; no partial result used |
| `38013_[0-31]` | `sbatch diagnosis/scripts/slurm_ogb_stage0_audit_array.sh` | completed baseline and audit smoke | per-snapshot shards under `results/ogb_stage0/audit_locked_shards/`; artifacts under `$STABLEWM_HOME/artifacts/audit_locked_array`; per-task logs `ogb_stage0_audit_array_38013_%a.out` | **completed** 2026-08-10: all 32 tasks exit 0; effective concurrency was 2 under `QOSMaxGRESPerUser` |
| `38016` | `sbatch --dependency=afterok:38013 diagnosis/scripts/slurm_ogb_stage0_aggregate.sh` | all 32 array tasks | combined locked summary/CSVs under `results/ogb_stage0/audit_locked/`; log `ogb_stage0_aggregate_38016.out` | **completed** in 2s: 10,000 snapshot-bootstrap draws; locked verdict `strong_pass` |

## Locked result and decision

The official evaluator obtained 34/50 successes (68%) for the released
checkpoint. Figure 6 of the LeWM paper reports 74%, averaged over three training
seeds. The 6-point difference is treated as a finite-sample/checkpoint sanity
difference; no hyperparameter or seed was tuned to close it.

All 32 precommitted matched-candidate snapshots completed. Snapshot restoration
errors were exactly zero for qpos, qvel, target pose, dataset-state agreement,
and rendered pixels. The action-normalizer round-trip maximum absolute error was
`5.96e-08`.

On the final CEM population, with 95% snapshot-bootstrap intervals:

| Quantity | Mean | 95% CI |
|---|---:|---:|
| true-endpoint-latent physical selection regret | 0.0313 m | [0.0143, 0.0543] |
| representation success gap | 0.125 (4/32) | [0.03125, 0.25] |
| oracle candidate success | 0.781 (25/32) | [0.625, 0.906] |
| learned-cost selected success | 0.500 (16/32) | [0.344, 0.656] |
| true-endpoint-latent selected success | 0.656 (21/32) | [0.500, 0.812] |
| end-to-end physical selection regret | 0.0761 m | [0.0475, 0.1075] |
| learned-cost/physical Spearman | 0.0016 | [-0.0243, 0.0268] |
| true-endpoint-latent/physical Spearman | 0.345 | [0.198, 0.481] |

All three precommitted lower-bound criteria are above zero, so Stage 0 is a
**strong cost/representation pass**. The decomposition is material: replacing
the learned rollout endpoint with the encoded true endpoint improves mean
selected physical distance from 0.1209 m to 0.0761 m, but the physical oracle
over the same 300 candidates still reaches 0.0448 m. Thus learned dynamics
error is important, but it does not explain away the residual representation/
cost selection error. Candidate coverage is adequate for this diagnostic.

This result authorizes replication and method ideation; it does not validate a
new method. It is one released checkpoint, one OGBench task, 32 snapshots, and
dataset rows are not claimed held out from model training. The simulator was an
offline measurement instrument after candidate generation and was never
available to the planner. Stage 1 remains **HOLD** until there is a method whose
training and test-time inputs are available in the intended offline setting and
whose novelty survives comparison with dynamics adaptation, proposal coverage,
and simple cost calibration baselines. Simulator-in-the-loop training would be
a different setting and must not be presented as the default follow-up.

The success-level criterion passes narrowly: four of 32 snapshots contribute a
paired representation success gap, and the lower bootstrap bound is 1/32. This
is enough for the locked Stage 0 rule, but not enough to treat generalization as
settled without a larger preregistered replication.

The concise decision record and immutable output hashes are in
`diagnosis/results/ogb_stage0/STAGE0_DECISION.md`.
