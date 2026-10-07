
## 2026-10-06 puzzle root-cause investigation (user request)

- 57669 ew_puzcause: mig/main,1GPU,2CPU,12GB,15min. Frozen57636 checkpoint; saved dev roots; diagnostic known-press trajectories for5tasks and task2 search controls separating learned/oracle WM and h plus generic nearest-rest-prototype projection. No training, no simulator, no core source changes. Oracle controls use the simulator's puzzle rules only for attribution and are not method results. Source docs/diag_puzzle_20261006; output /mnt/data/nhatnc129/jepa/event_wm/diag_puzzle_cause_20261006/job_57669.
- Before submission squeue empty and sacct checked. Month usage GPU88h/108h ceiling, CPU680h/1094.4h, memory6,195,495MBh/7,969,481.1MBh. Planned15min adds0.25GPUh,0.5CPUh,3072MBh.
- 57636 saved dev traces: all746 ended events in failed tasks2-5 reached their acted-entity target and toggled that entity. Full-effect-set and state-cycle audit saved skill_trace_check.json; simulator information only diagnostic.
