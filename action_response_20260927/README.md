# Planning-scale action response on LeWM

**Status: three-seed endpoint-response result complete; published baseline
not beaten.** See [JOB_LEDGER.md](JOB_LEDGER.md) for exact runs and
[OVERLAP_REVIEW.md](OVERLAP_REVIEW.md) for the distinction from prior work.

Independent PushT experiment based on the official LeWM checkpoint and the
`stable-worldmodel` evaluation code pinned in `../diagnosis/external/`.

LeWM is listed as NeurIPS 2026 by co-author Quentin Le Lidec. The public
arXiv/repository citation files have not yet been updated. The paper reports
96.0 ± 2.83% PushT success over three training seeds and 50 common tasks, so
this is a high-ceiling baseline.

## Experiment

`pipeline.py` runs all work on a Slurm compute node:

1. Load the official `quentinll/lewm-pusht` checkpoint and expert dataset.
2. Restore a PushT state from an expert trajectory, replay ten real expert
   actions to create a physically consistent current-frame context, and simulate
   two clipped action perturbations from that same context. Check replay
   determinism before recording the pair.
3. Freeze the image encoder/projector and encode the context and all five
   future block endpoints. Fine-tune two models from identical initial weights,
   batches, and branch trajectories: endpoint prediction only, and prediction
   plus matched-action response loss. Prediction uses block horizons 1, 3,
   and 5. The reported endpoint variant compares the difference in predicted
   and observed embeddings at block horizon 5, matching the planner's goal
   scoring point.
4. Evaluate prediction-only and response models on common held-out
   episode/start/goal tasks using the upstream `World.evaluate` and CEM. Save
   each root immediately for crash recovery; aggregate paired success only
   after every intended root completes. Record position and symmetry-aware
   angle error alongside success.
   The published 96.0% LeWM PushT result is the external reference; this run
   does not spend time reproducing it.

The simulator state and goal state are used only to restore/measure rollouts,
never fed into LeWM. This is simulator-assisted training, not offline-only.
The official evaluation action scaler is fit on the full expert dataset and
applied to every candidate. The paper's settings are horizon 5, action block
5, receding horizon 5, history length 1 (the upstream evaluation config's
default), image 224, ImageNet normalization, CEM 300 candidates,
30 rounds, 30 elites, and evaluation budget 50 environment steps.

`sbatch slurm_pipeline.sh` starts a bounded end-to-end development run.
Environment overrides: `RUN_ROOT`, `PAIRS`, `UPDATES`, `EVAL_ROOTS`, `SEED`,
`LR`, `RESPONSE_TARGET`, and optional `BRANCH_CACHE`.
All outputs and source hashes go into a unique timestamped directory under
`RUN_ROOT`. The corrected experiment uses `RESPONSE_TARGET=endpoint` and
three independent fine-tuning seeds on 50 common roots. It achieved 94.0%
mean success versus 92.7% for matched prediction-only fine-tuning, with a
two-level paired bootstrap 95% interval of [-6, 8] percentage points for
the gain. This is below the paper's 96% point estimate and does not establish
a stable improvement.

Sources: [LeWM paper](https://arxiv.org/abs/2603.19312),
[author venue listing](https://quentinll.github.io/),
[official code](https://github.com/lucas-maes/le-wm).
