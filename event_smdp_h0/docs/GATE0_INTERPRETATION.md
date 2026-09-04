# Gate 0 interpretation

Date: 2026-09-03. Locked verdict: **STOP_NO_MATERIAL_CAUSAL_ROOM** for this
OGBench-Cube/future-demonstration proposal protocol.

## What was tested

The paired intervention changed only the information backed up by deterministic
skill-level UCT. `terminal_only` received exact stable terminal success;
`event_state` received the oracle cyclic event state, event timing, and drop
branch. Both received identical skill support and transition-call budgets. The
64 evaluation windows came from fresh episodes disjoint from all earlier Stage-0
and Phase-1 manifests and had at least 8 cm start-to-goal cube displacement.

## Result

| Budget per replan | Terminal-only | Event-state | Paired gain |
|---:|---:|---:|---:|
| 16 | 64/64 | 64/64 | 0/64 |
| 32 | 64/64 | 64/64 | 0/64 |
| 64 | 64/64 | 64/64 | 0/64 |

At the primary budget 64, the paired gain is 0.000 with a bootstrap interval
[0.000, 0.000], no discordant pairs, and one-sided exact McNemar p=1. Final
physical distance differs by -0.87 mm in favor of event feedback, but its 95%
interval [-2.79, 1.03] mm includes zero and it does not change success.

## Scope of the negative result

This gate does not falsify action-conditioned multi-state event-time world
models. It identifies the wrong substrate for testing their claimed advantage:
the privileged lattice contains each evaluation trajectory's exact future
demonstration chunks, so a sparse terminal oracle already solves every locked
case with the smallest paper-relevant search budget. There is no success-rate
headroom for richer event feedback.

The correct next step is not to train a hazard head here and hope learned-model
error creates a gap. That would confound richer supervision with model quality
and reintroduce optimizer exploitation. A new test must use an offline learned
or retrieval skill prior that does not include the evaluation trajectory, and a
task with genuine failure branches and longer compositional depth. Its mandatory
baseline remains a terminal event head with the identical planner and proposal
prior.
