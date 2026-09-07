# ORDER-JEPA implementation

The latest executed qualification result and current blocker are recorded in
[`RUN_STATUS.md`](RUN_STATUS.md).

This directory contains a falsification-first implementation of the PushT
Stage-A audit and the proposed ORDER loss. It uses only the authors' original
[`gaoyuezhou/dino_wm`](https://github.com/gaoyuezhou/dino_wm) implementation at
commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c` and the official PushT
checkpoint published in the authors' [OSF release](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28).
The consolidated `facebookresearch/jepa-wms` checkpoint is deliberately not a
supported backend.

It also contains an independent Stage-A port to the official
[`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)
checkpoint. That port uses LeWM's three-frame history, five-control action
blocks and five-block horizon, exact MuJoCo state restore, and the upstream
Reacher qpos success tolerance. See [`RUN_STATUS.md`](RUN_STATUS.md) for the
executed negative result.

## What is implemented

- Deterministic manifest construction with episode/anchor IDs and dataset hashes.
- Reset-from-start plus warm-up replay using the original PushT environment.
- True swapped-order branches, ordinary candidates, and a separately labeled
  feasible witness in each fixed candidate set.
- Exact original preprocessing, five-control action grouping, latent rollout,
  and terminal visual + proprio MSE.
- Fixed-candidate physical-oracle, true-latent, and predicted-latent regret.
- Signed order-vector error, object-level effects, contacts, reset/replay noise,
  episode-clustered confidence intervals, and pre-registered decision gates.
- A standalone PyTorch ORDER loss and algebraic/unit sanity checks.

Training is intentionally not launched by this pipeline. `GO_STAGE_B_SCREEN` is
required first; otherwise predictor fine-tuning would not test the stated
hypothesis.

## Required release layout

Download the PushT model from the official OSF `checkpoints` folder without
renaming its internal files. The default expected layout is:

```text
/mnt/data/nhatnc129/jepa/dino_wm_official/outputs/pusht/
├── hydra.yaml
└── checkpoints/
    └── model_latest.pth
```

If the extracted location differs, set `DINO_WM_MODEL_DIR`. The loader rejects
a non-PushT config, a non-DINOv2 patch encoder, the wrong frameskip/history, a
non-author repository origin, or a source commit other than the pinned commit.
Every score shard records SHA-256 hashes for the checkpoint and resolved config.

The Python environment must satisfy the original repository's
`environment.yaml`. Set `ORDER_PYTHON=/path/to/that/environment/bin/python` if
the default diagnosis environment does not contain those dependencies.

## Slurm workflow

All simulator, encoding, model loading, and aggregation jobs run on compute
nodes. From the repository root:

```bash
sbatch order-jepa/scripts/slurm_prepare_source.sh
sbatch order-jepa/scripts/slurm_sanity.sh
sbatch order-jepa/scripts/slurm_manifest.sh
```

After those jobs pass, submit collection. The default full run uses eight
workers, each processing a deterministic stride of the 200 anchors (this stays
within the cluster's array-submit limit):

```bash
sbatch order-jepa/scripts/slurm_collect_sharded.sh
```

`slurm_collect.sh` remains useful for small smoke arrays where one Slurm task
maps to one anchor.

Verify collection completion with both `squeue` and `sacct`, then submit four
GPU scoring shards and, after they finish, aggregation:

```bash
sbatch order-jepa/scripts/slurm_score.sh
sbatch order-jepa/scripts/slurm_analyze.sh
```

For a smoke run, isolate its output and override both array sizes explicitly:

```bash
ORDER_RUN_ROOT=/mnt/data/nhatnc129/jepa/order_jepa/smoke \
ORDER_ANCHORS=4 sbatch order-jepa/scripts/slurm_manifest.sh

ORDER_RUN_ROOT=/mnt/data/nhatnc129/jepa/order_jepa/smoke \
sbatch --array=0-3 order-jepa/scripts/slurm_collect.sh

ORDER_RUN_ROOT=/mnt/data/nhatnc129/jepa/order_jepa/smoke \
ORDER_ANCHORS=4 ORDER_SCORE_SHARDS=1 \
sbatch --array=0 order-jepa/scripts/slurm_score.sh
```

The final machine-readable result is `analysis/summary.json`; the concise audit
decision is `analysis/DECISION.md`. A witness-containing ranking panel certifies
feasibility but must not be reported as ordinary CEM success.

## Code map

- `order_jepa/original_dino_wm.py`: strict original-checkpoint adapter.
- `scripts/make_manifest.py`: immutable Stage-A design.
- `scripts/collect_pusht_anchor.py`: true branch replay and reset audit.
- `scripts/score_pusht_anchor.py`: original-model fixed-candidate scorer.
- `scripts/analyze_stage_a.py`: clustered estimates and go/stop gates.
- `scripts/run_reacher_stage_a.py`: exact-reset LeWM-Reacher branch collection
  and scoring.
- `scripts/analyze_reacher_stage_a.py`: trajectory-clustered Reacher gates.
- `order_jepa/lewm_reacher.py`: strict official checkpoint/config provenance.
- `order_jepa/loss.py`: signed ORDER objective and paired-error identity.
- `proposal_en.md` and `compatibility_plan_en.md`: claims and experiment design.
