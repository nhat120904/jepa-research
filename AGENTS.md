# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Repository Status (2026-07-13)

This repository contains an implemented diagnostic and an active ICLR paper project. The
research has moved materially beyond the original CAI-JEPA proposal:

- `paper/main.tex` — the current paper-of-record, framed around test-time search exploiting
  representation-induced costs under oracle-perfect dynamics. Do not derive current claims
  from the original proposal without checking this file and `diagnosis/docs/CLAIMS_EVIDENCE.md`.
- `cai_jepa_paper_proposal.md` — historical/original CAI-JEPA proposal. It must clearly mark
  which action-identifiability and counterfactual-objective claims were superseded by the
  July 2026 cost-exploitation results and concurrent literature.
- `diagnostic_implementation_plan_v2.md` — historical implementation plan plus later phase
  record; keep provenance, but label completed, superseded, and confirmatory work explicitly.
- `diagnosis/` — the implemented diagnostic, oracle/planning experiments, and post-hoc
  intervention code. It targets the real `facebookresearch/jepa-wms` API and has an offline
  unit-test suite under `diagnosis/tests/`. Model/data execution follows `diagnosis/RUNBOOK.md`.

The most important orientation doc is `diagnosis/docs/plans/2026-06-01-real-api-rewrite-design.md` — it records what the upstream API actually is and the key design decisions.

## Current Research Goal

Prepare an ICLR-quality mechanistic paper establishing when test-time planners exploit
representation-induced costs in contact-rich latent world-model planning. The strongest current
evidence is the MetaWorld oracle ladder and simulator-state verification of CEM elites. DROID
action metrics are observational transfer evidence, not causal same-state counterfactuals.

Claims must respect these current limits:

- say **"grounding alone is insufficient"**, not "grounding is necessary";
- say **"no scale trend among the released checkpoints under this diagnostic"**, not
  "scaling cannot fix the problem";
- `hard_nn` CRA/Boundary Blindness uses actions from nearby states and is observational until
  validated by snapshot/restore same-state interventions;
- Phase-H predictor-LoRA DROID numbers require a persisted trajectory split and held-out-only
  rerun before they can be used as paper evidence;
- cost-exploitation generality requires the second-checkpoint and non-CEM planner replications.

The original go/no-go deliverable remains in `diagnosis/results/decision_report.md`; current
claim status lives in `diagnosis/docs/CLAIMS_EVIDENCE.md`.

## Cluster Compute Policy (mandatory)

This checkout is on a Slurm **login node**. Never run model loading, MuJoCo rollouts, latent
extraction, training, large CSV/NPZ analysis, full pytest, or other sustained CPU/GPU/RAM work
directly on the login node.

- Submit every heavy experiment with `sbatch` to a compute node. GPU/model/MuJoCo jobs must
  request a GPU and appropriate CPU/RAM in the Slurm script.
- Login-node work is limited to lightweight inspection and editing: `rg`, `sed`, `git status`,
  small metadata reads, `squeue`/`sacct`, syntax checks, and narrowly targeted unit tests that do
  not load models or large data. When uncertain, use Slurm.
- Do not use interactive Python to scan multi-GB result files on the login node. Add an analysis
  script and submit it through Slurm, even when the analysis is CPU-only.
- Record every submitted job ID, exact command/config, output paths, dependency chain, and final
  state in the relevant plan/handoff document.
- Never overwrite another agent's dirty files. As of 2026-07-13, Agent PROBE owns the in-progress
  edits to `diagnosis/scripts/48_encoder_info_upperbound.py` and
  `diagnosis/scripts/slurm_encoder_infoUB.sh` (with the old wrapper deletion). Coordinate before
  editing those paths.
- Before assuming a user-reported job is still running, verify with both `squeue` and `sacct`.
  Original jobs `26166` (second checkpoint) and `26400` (planner generality) were cancelled on
  2026-07-12. They were resumed as `26481` and `26482`; Agent PROBE submitted encoder
  information upper-bound job `26485`. Held-out Phase-H arrays `26493` (DINO-WM) and `26494`
  (JEPA-WM) depend on those three jobs; aggregation job `26495` depends on both arrays. Locked
  n=64 confirmatory array `26491` then depends on the held-out arrays, and paired-analysis job
  `26492` depends on successful completion of `26491`. Exact same-state intervention job
  `26497` completed; shared physical-scaling `26498` OOMed and was retried as `26610`; compute-node
  paper-build jobs `26499/26504` are recorded in the ledger. Instrumented exploitation array
  `26502` and component-analysis job `26503` separate readout shift, true outcome/regret, and
  proxy--truth corruption. Coverage-vs-selection array `26505` and analysis `26506` test the
  IMWM proposal-coverage alternative. TRM-style jobs are `26507` train, `26508` eval, `26509`
  analysis; ACID-style jobs are `26510` train, `26511` eval, and `26512` paired analysis.
  Re-check live state before acting on these IDs.

## Architecture of the Diagnostic (`diagnosis/`)

The diagnostic itself operates on **pretrained, frozen checkpoints**. Later paper phases train
explicitly named probes, LoRA adapters, and small auxiliary heads; never describe those phases as
fully frozen. Core data flow:

1. **Adapters** (`models/adapters/`) — `WorldModelAdapter` ABC + one unified `EncPredWMAdapter` for all three baselines. They all load via `torch.hub.load('facebookresearch/jepa-wms', hub_id, trust_repo=True)` returning `(EncPredWM, preprocessor)`. The adapter drives the model through its own `EncPredWM.encode` (raw `[0,255]` visual in → `(B,T,V,H,W,D)` latent) and `EncPredWM.unroll` (the planner's prediction primitive) — **never** `.encoder`/`.predictor` directly.
2. **Latent extraction** (`scripts/03_extract_latents.py`) — encode every frame once, cache to HDF5 under `data/precomputed_latents/` (z + proprio + raw state + gripper). All metrics run on the cache.
3. **Regime stratification** (`stratification/`) — `free_space`, `pre_grasp`, `gripper_actuation`, `contact_manipulation`. Metaworld uses a **proxy** from the 39-dim `state` vector (ee/object positions; object displacement = contact proxy) — the HF dataset has no MuJoCo contact GT. DROID/RoboCasa use proprioception + latent-change heuristics. Metrics are per-regime; the thesis is failures concentrate in contact-rich regimes.
4. **Metrics** (`metrics/`) — CRA, AUG, ECS, CTD (optional). Each exposes a *per-transition* function the runner calls directly (so synthetic validation tests the production path). The primary decision signal is **effect-conditioned CRA** (CRA on transitions with `‖Δz‖>τ`). CIs are **trajectory-clustered** bootstrap.
5. **Analysis** (`scripts/06_analyze_results.py`) — CSVs, figures, and the decision report. Decision logic is CI-aware (ABANDON needs the upper CI bound confidently high).

## Key Implementation Pitfalls

- **Action normalization is the #1 bug.** The real method is `preprocessor.normalize_actions` (plural); the adapter calls it. Validate on the server with `scripts/check_normalization.py` (predict a real transition; MSE within ~2× the model's eval loss). DROID = identity (mean 0/std 1); Metaworld = real shift+scale.
- **Always sanity-check against a published number** and run `scripts/terver_gripper_test.py` (open vs close gripper on DROID; expect 2-way CRA > 0.90).
- **Validate metrics with synthetic models** first: `python scripts/07_validate_synthetic.py`.
- All planning configs are `L2_cem` → CRA uses **L2** for every baseline.
- Push-T / PointMaze are sanity checks only — never report them as thesis evidence.

## Environment & Commands

`uv` is used for dependency management. The commands below describe the compute-node workflow;
do not execute the full pipeline or full test suite directly on the login node. Submit the
corresponding `scripts/slurm_*.sh` wrapper, or create one when none exists (see
`diagnosis/RUNBOOK.md`).

```bash
cd diagnosis
# Login node: lightweight only
git status --short
rg -n "pattern" scripts tests docs
python -m py_compile path/to/changed_file.py

# Compute node via Slurm: tests/validation and all data/model work
sbatch scripts/<appropriate_slurm_wrapper>.sh

# Server: clone upstream + run the pipeline (see RUNBOOK.md for the full sequence)
bash scripts/01_setup_environment.sh          # clones external/jepa-wms + uv sync
python scripts/smoke_test.py                  # real checkpoints load + encode + predict
python scripts/check_normalization.py --config configs/diagnostic_metaworld.yaml --model jepa_wm_metaworld --ref-eval-loss <L>
python scripts/03_extract_latents.py  --config configs/diagnostic_metaworld.yaml
python scripts/04_classify_regimes.py --config configs/diagnostic_metaworld.yaml
python scripts/05_run_diagnostic.py   --config configs/diagnostic_metaworld.yaml
python scripts/06_analyze_results.py  --metaworld_csv results/metaworld_diagnostic.csv --droid_csv results/droid_diagnostic.csv
```

If `torch.hub.load` returns 503s, delete `external/jepa-wms/uv.lock` and re-run `uv sync`.
