# Predictive abstraction for latent world models

An independent, self-supervised WM method pilot. Status: implementation/preflight,
**no demonstrated method improvement**. Not a continuation of the closed Scrub composer
or optimizer-conditioned-misranking paper.

## Start here

- [Research question and literature](docs/PREDICTIVE_ABSTRACTION_RESEARCH_20260921.md)
- [Locked first implementation protocol](docs/PILOT_PROTOCOL.md)
- [Job ledger and current status](docs/JOB_LEDGER.md)
- [Corrected prospective-branch training protocol (active)](docs/TRAIN_PROTOCOL_V2.md)
- [Ready-to-present professor demo and one-minute script](docs/PROFESSOR_DEMO_20260922.md)

2026-09-22: frozen encoding 53550 finished. CPU job 53629 generated a standalone
HTML/GIF oracle demo and 672 fresh prospective RGB/action branches. The active training
entry point is `pa_wm.train_branches`, with complete spatial-frame baselines, shared
queries across candidates, no-action/query-only controls and prefix-level ranking.
Encoding/profile job 53630 completed in 1m48s, with 14/14 tests passing. The next run
is a bounded full one-seed offline pilot using these cached features, not MPC/control.
See [profile result and small-bank headroom warning](docs/PROFILE_RESULT_53630.md).
The old `train.py` protocol is superseded.

Full pilot 53642 completed but did not improve A-before-B selection over default.
The current bounded follow-up is [observed-future codec rescue](docs/CODEC_RESCUE_PROTOCOL.md):
feature recoverability, tiny-set fitting, and matched temporal-pooling × candidate-loss
ablation. It is not a new control result or a relaunch of the old audit paper.

53659 completed: the visual-affinity probe succeeds while generic sequence/summary
readers remain weak. Next bounded pilot: [metric-space bridge](docs/METRIC_BRIDGE_PROTOCOL.md),
with the frozen RGB-trained metric as target, common temporal query readout, fixed
pooling controls and gated summary forecasting versus a matched metric-frame WM.

Learn a query-independent summary of an unexecuted action-conditioned trajectory;
read visual-temporal queries from it. Compare with frame prediction, generic compression,
and direct query prediction, then use identical action proposals for control.

## Implementation

`pa_wm/` contains a RGB-only adapter to the local **original DINO-WM Wall source**, smooth
action proposals, observation-derived queries, a compute-node preflight/data collector,
and summary codec/predictor/query-reader modules. `tests/` exercises semantics,
causality interfaces, action bounds, and exact branch replay.

No simulation/model work runs on the login node. Submit `scripts/slurm_preflight.sh`
through `sbatch`. Each run snapshots source/config/protocol, records hashes and versions,
has a time limit, and writes outside the source tree. Do not launch duplicates.

The first job is CPU-only. It checks the environment, query semantics, proposal support,
and creates a small RGB/action dataset. It does **not** train a world model or demonstrate
learned planning. Feature encoding and matched training follow after inspecting it.

Only the current predictive-abstraction research note was moved out of `latent_scope`;
a redirect preserves old links. Earlier conflicting architecture proposals remain there
as history, not active specifications.
