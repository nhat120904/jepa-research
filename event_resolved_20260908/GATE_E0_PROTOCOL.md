# Gate E0: strike--slide premise and plumbing screen

Status: implementation screen only. This gate trains no learned model and does not
qualify ER-WM by itself.

## Question

Before implementing a neural hybrid model, determine whether the proposed arena has
all of the following properties:

1. exact simulator rollouts searched with the deployment action parameterization can
   solve the task;
2. a fixed scripted strike does not already consume essentially all available
   headroom;
3. small action perturbations measurably change contact time while retaining enough
   matched, isolated contact sequences for event-coordinate supervision;
4. event timestamps and outcomes are stable enough across physics frequencies to be
   meaningful targets.

The screen uses simulator state only. It does not render and it does not train ER-WM.

## Locked first-screen interface

- ManiSkill `3.0.1`, CPU simulation, Panda, state observations.
- Custom `ERStrikeSlide-v0`: rectangular puck, 5 cm terminal goal radius, terminal
  speed below 0.1 m/s for 0.2 s, four-second horizon.
- Panda `pd_ee_target_delta_pos` controller. Target changes occur at 20 Hz while the
  target is held over physics substeps.
- A low-dimensional strike family parameterized by lateral impact offset, approach
  standoff, and strike gain. The same family is used by the fixed scripted baseline
  and exact-dynamics random/CEM-style search.
- Hand--puck onset and release are detected from pairwise finger contact force. Table
  support contact is not treated as an event.

The low-dimensional strike family is an E0 task-compatibility interface, not the final
paper planner. If it passes, E1 must either retain it for every learned-model arm or
separately qualify the proposed eight-knot action interface with exact dynamics.

## Outputs

The runner writes one JSON artifact containing:

- package/config provenance;
- fixed-script and oracle-search success and terminal errors on paired reset seeds;
- all nominal and perturbed onset/release sequences;
- matched-sequence and transversal-event coverage;
- contact-time shifts under paired action perturbations;
- cross-frequency onset and endpoint differences;
- restore/repeat determinism checks;
- a machine-readable decision and reasons.

## Decision rules

These are screening rules, not universal statistical thresholds.

`E0_PASS_OPEN_E1` requires all of:

- exact-dynamics search success at least 70% on the smoke cohort;
- at least 10 percentage points of success headroom over the fixed script, unless the
  fixed script is below 70% and exact search still clears 70%;
- at least 60% of nominal roots have one matched onset/release sequence across all
  four perturbations;
- at least 50% of matched onsets are transversal by the declared approach-speed
  threshold;
- median non-zero paired onset shift is at least two 500 Hz physics steps (4 ms);
- median matched onset disagreement between 500 Hz and each of 250/1000 Hz is at
  most 8 ms, with terminal puck-position disagreement at most 2 cm;
- exact replay has no success mismatch and maximum endpoint error at most 1 mm.

`E0_STOP_TASK_OR_PLANNER` is returned when exact search cannot solve the task or the
fixed script removes useful headroom. `E0_STOP_EVENT_SIGNAL` is returned when contact
pair coverage/timing sensitivity is insufficient. `E0_STOP_NUMERICS` is returned when
frequency or replay checks fail. A run that lacks required cells is `E0_INCOMPLETE`,
not a pass.

The first smoke uses few seeds and cannot establish a publication claim. Thresholds
must not be loosened after seeing the output; a revised task requires a new output tag
and an explicit protocol amendment.

## Compute and execution

All simulation runs through Slurm. The setup job creates a dedicated environment at
`/mnt/data/nhatnc129/jepa/erwm_e0/.venv`. The smoke job is CPU-only, requests a bounded
wall time, and performs no rendering. No interactive allocation is used.

