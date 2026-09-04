# SearchCal-EventWM / OGBench-Scene handoff

Date: 2026-09-03 UTC

This handoff records the completed MIG compute-node chain and its locked H0
decision.  All scheduler states below were checked with both `squeue` and
`sacct` on 2026-09-03 UTC.

## Research decision

The paper direction is no longer “event world model + temporal automaton.”
That combination is directly overlapped by hint^2 (arXiv:2608.13678).  The
revised candidate is **SearchCal-EventWM**: grouped conformal calibration over
the exact candidate set exposed to search, followed by robust planning on the
event-SMDP/task-automaton product.  See `SCENE_RESEARCH_POSITIONING.md`.

## Submitted jobs

| Job | Exact submission | Output | State at record time |
|---|---|---|---|
| `48877` | `sbatch --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 event_smdp_h0/scripts/slurm_scene_gate0_plumbing.sh` | `outputs/scene_gate0/plumbing/48877/` | **completed**, 22s, exit 0; exact restore and fixed known solutions passed tasks 4 and 5 |
| `48879` | `sbatch --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency=afterok:48877 event_smdp_h0/scripts/slurm_scene_gate0_smoke.sh` | `outputs/scene_gate0/smoke/48879/` | **completed**, 9m07s, exit 0; event state wins 3/4 task-budget pairs, terminal-only wins 0/4 |
| `48944_[0-31]` | `sbatch --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 event_smdp_h0/scripts/slurm_scene_gate0_pilot.sh` | `outputs/scene_gate0/pilot/shards/<index>/result.json` | **completed**, all 32 shards exit 0; 16 task-4 and 16 task-5 fresh resets, each evaluated at budgets 14 and 28 |
| `48988` | `sbatch --partition=mig --dependency=afterok:48944 --export=ALL,PILOT_JOB_ID=48944 event_smdp_h0/scripts/slurm_scene_gate0_analysis.sh` | `outputs/scene_gate0/pilot/aggregate/48944/` | **completed**, 2s, exit 0; verdict `PILOT_GO_LEARNED_EVENT_WM` |

Compute logs are under `/mnt/data/nhatnc129/jepa_runs/logs/` with the job names
defined in the Slurm wrappers.  A duplicate completed chain (`48873`, `48874`,
`48880_[0-31]`, `48881`) produced a byte-identical aggregate summary.  The
authoritative aggregate above was generated only after all `48944` shards had
overwritten the shared shard paths.  Superseded main-partition jobs `48751`,
`48752`, `48875`, and `48876` were cancelled before node allocation and
produced no experimental output.

## Locked H0 result

There are 64 paired task-reset-budget comparisons.  At budget 14, pooled
physical success is `19/32` (59.4%) for event-state planning versus `1/32`
(3.1%) for terminal-only planning, a paired difference of `+56.25` points
(bootstrap 95% CI `[37.5, 75.0]`).  At budget 28 it is `31/32` (96.9%) versus
`1/32` (3.1%), a paired difference of `+93.75` points (95% CI
`[84.375, 100]`).  The exact McNemar p-values are `4.01e-5` and `1.86e-9`.

This clears H0: the Scene substrate contains substantial causal room for an
event-aware planner under matched proposal support and search budgets.  It
does **not** show that a learned event world model or SearchCal calibration can
recover the oracle advantage; those are H1--H3.

## Interpretation lock

This Scene experiment is only H0 causal-room evidence with oracle MuJoCo skill
transitions.  It cannot validate conformal calibration or a learned world
model.  Continue to H1 only if the event arm improves paired physical success
and the fixed support check remains successful.
