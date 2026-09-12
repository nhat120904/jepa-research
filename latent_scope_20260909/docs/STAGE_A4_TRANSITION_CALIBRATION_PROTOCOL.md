# Stage A4: current-runtime transition calibration

Launched on 11 September 2026 as revised collection job `52073` followed by CPU
calibration job `52074` with dependency `afterok:52073`. Initial harness job `52059`
failed before collection because vector auto-reset erased its action counter; dependent
`52060` was cancelled without running. Stage B remains blocked until this gate finishes
and passes.

## Purpose

Measure and, if necessary, train against one-step dynamics from the actual RoboCasa
`1.0.1` evaluation runtime without requiring unstable full open-loop demonstration replay.
This is a bridge between historical offline trajectories and current native evaluation;
it is not a replacement task or a scientific result.

## Locked inputs

- The official GR00T N1.5 target-posttrained checkpoint passed the baseline gate with
  3/10 successes on `ScrubCuttingBoard` and 6/10 on `RinseSinkBasin`.
- A4 therefore recollects successful trajectories directly in the current RoboCasa
  `1.0.1` runtime; it does not reuse the incompatible released `0.5.1` action rollouts.
- Progress comes from the native task-owned histories: board contact positions/timer for
  Scrub and the latched three-location wash mask for Rinse. These are runtime predicates,
  not externally released per-frame semantic annotations.
- Action schema, three native cameras, controller, task success predicates, checkpoint,
  and source revisions are locked in `configs/stage_a4.json`.

## Pilot

The GPU collection job retains five successful current-runtime episodes per task. The
CPU calibration job deterministically replays them and chooses eight history-labelled
anchors per task across progress phases. At each anchor:

1. load the current-runtime collected XML and initial simulator state directly;
2. render the three native camera observations;
3. branch the recorded action plus four bounded perturbations in the current runtime;
4. repeat each branch twice from a fresh simulator carrier and require deterministic state and image
   equality;
5. save the before/after observations, action, simulator state, source episode/step, and
   progress annotation.

All future world-model arms must receive exactly the same offline and calibration data.
Task-owned history is not reconstructed from qpos/qvel. The use of privileged native
history to construct progress targets is a limitation and must be reported; the learned
model may consume observation history, but not privileged simulator state at evaluation.

## Gate

Proceed to Stage B only if all anchors load, camera keys are valid, repeated branches are
deterministic, action perturbations produce measurable outcome diversity, and progress
labels cover at least two distinct milestones per task. Otherwise retire this
RoboCasa/data pairing for the proposed experiment rather than adding more compatibility
patches.

Exact terminal-state equality after replaying an entire 1,200--1,350-step trajectory is a
reported diagnostic, not a condition of this local-transition gate. Failure of that
diagnostic forbids using full open-loop replay as calibration truth; it does not override
exact fresh-carrier equality at the selected one-step anchors.

Calibration runs CPU-only with OSMesa and an explicit eight-hour walltime. The collection
and calibration jobs and their artifact paths are recorded in `docs/JOB_LEDGER.md`.
