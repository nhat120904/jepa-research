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

53674 completed: neither learned codec passed every retention gate; no summary
students trained. Fixed 16-bin pooling retains queries, while action forecasting
is still weak. Current bounded follow-up: [fixed-summary forecast feasibility](docs/FIXED_SUMMARY_FORECAST_PROTOCOL.md),
including a tiny-fit gate and matched full-frame/coarse-target control. No new control claim.

53698 completed: action conditioning improves latent forecast error, but not query
ranking. Next: [frozen-predictor readout diagnostic](docs/READOUT_DIAGNOSTIC_PROTOCOL.md).
See [conditional architecture changes](docs/ARCHITECTURE_DECISION_AFTER_FORECAST_FAILURE.md)
for the decision tree; no automatic architecture sweep or MPC job is authorized by it.

53700 completed: [readout result](docs/READOUT_RESULT_53700.md) shows no consistent
rescue. Next authorized bounded experiment: [local patch transitions](docs/LOCAL_TRANSITION_PROTOCOL.md),
separating one-step capability from open-loop error accumulation. Not a new summary
method win or control result.

53741 completed: [short dynamics passed, long queries failed](docs/LOCAL_TRANSITION_RESULT_53741.md).
Next authorized diagnostic is [controlled rollout extension](docs/ROLLOUT_EXTENSION_PROTOCOL.md):
same checkpoint/architecture, longer unroll versus extra short-unroll training.

53748 completed: [longer rollout failed the query gate](docs/ROLLOUT_EXTENSION_RESULT_53748.md).
The [last query-native pilot and stop rule](docs/QUERY_NATIVE_STOP_PROTOCOL.md) closes
this implementation repair cycle on a negative result; no automatic architecture sweep.

53750 completed with the predeclared verdict
`STOP_CURRENT_WALL_METHOD_IMPLEMENTATION`. See the
[final result and scope](docs/QUERY_NATIVE_RESULT_53750.md). No summary/frame/direct
arm generalized candidate ranking at both main horizons; the current Wall repair
cycle is closed and no follow-on job is queued.

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
