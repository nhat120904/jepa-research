# Stage A: native RoboCasa execution preflight

Locked: 10 September 2026, before any Stage-A simulator job was submitted.

## Purpose

Stage A establishes whether the two selected native tasks can be reproduced on this
cluster with their released data, actions, cameras, success lifecycle, and history-bearing
task state. It does not train or evaluate the proposed world model.

Primary tasks:

- `ScrubCuttingBoard` on target kitchens;
- `RinseSinkBasin` on target kitchens.

The canonical research decision remains `FINAL_DECISION_EN.md`. This file only locks the
first execution gate.

## Pinned sources and isolated runtime

| Component | Revision |
|---|---|
| RoboCasa | `4f8a2980def75a55dff96b990745b83540425f09` |
| robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| RoboCasa Diffusion Policy fork | `41212698b6f481ed92a55f0d5f1778ec56bea417` |

Large artifacts use HTTPS-accessible, revision-pinned Hugging Face mirrors because this
cluster cannot connect to `utexas.box.com`:

| Artifact | Repository | Revision |
|---|---|---|
| Official textures and Objaverse assets | `robocasa/robocasa-assets` | `1b92c3d02ca4354984fec961357db0bff7b32166` |
| Byte-identical Lightwheel fixture mirror | `IIFAN/robocasa365-lightwheel-fixtures` | `0cf51d45ab07fce13c4733ee1fe20bb01399768c`; upstream archive SHA-256 `62c1c87776edda9b7c42dda7b00781a1e5a7f90acc3e49d06c39c9545cbd6c3e` |
| Byte-identical Lightwheel object mirror | `IIFAN/robocasa365-lightwheel-objects` | `3f7f7aee8ef4367dfdc85a86778467cf8527f8fb`; upstream archive SHA-256 `a1db739b0e80fc4d56d4be27d89dba3aa60000e9648c5607874700e0f14b7955` |
| Released demonstrations | `nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos` | `522e4ffa3bb9d9729f79b241c65fc9e589f2659c` |

The runtime lives below `/mnt/data/nhatnc129/jepa/latent_scope_stage_a`. It does not
modify the historical `diagnosis/external/robocasa` fork or the old custom PnP dataset.
Source revisions, package versions, dataset paths, and Slurm metadata are written into
every result.

## A0: setup and data gate

The setup job must:

1. verify all three source revisions;
2. create a Python 3.11 environment;
3. install the pinned robosuite and RoboCasa checkouts;
4. configure an isolated dataset root;
5. download the task-minimal assets (`textures`, `fixtures`, `objaverse`, and the
   checksum-verified Lightwheel object pack required by native kitchen fixtures) and
   only metadata, actions, states, XML, and episode metadata for the first five released
   target-human demonstrations of each task; videos are not needed because Stage A
   renders observations from restored simulator state;
6. verify that each dataset has metadata, simulator extras, and at least five episodes.

The setup is idempotent and may be resumed after timeout. It never uses `--overwrite` on
an existing complete dataset. A partial or missing artifact is an infrastructure outcome,
not evidence against the research hypothesis.

## A1: replay and history-state gate

Use the first five registered episodes of each dataset. For every episode:

1. create the environment from released dataset metadata;
2. restore its released XML, episode metadata, and initial MuJoCo state;
3. replay the released relative actions through `env.step`;
4. read success only from the reward returned by that native step; never call
   `_check_success` an extra time;
5. compare each replayed simulator state against the next released state;
6. record task-history changes and save camera frames at the first such changes;
7. at a deterministic mid-episode action boundary, snapshot simulator state, task-owned
   history, episode counters, controller/interpolator state, robot buffers, MuJoCo
   warm-start/control state, and RNG state;
8. execute an eight-action suffix, restore the snapshot, repeat the same suffix, and
   compare simulator states, native rewards, task history, and camera observations.

Task-owned state explicitly covered by the first implementation:

- `ScrubCuttingBoard`: `board_contact_positions`, `board_contact_timer`,
  `sponge_contact_height`;
- `RinseSinkBasin`: `washed_loc`.

The harness also records previously unseen mutable scalar/list/array attributes so that a
successful numerical replay is not treated as proof that the whitelist is complete.

## Verdicts

- `STAGE_A_PASS`: both tasks load; all ten episodes reach native success under released
  actions; both tasks expose the expected history changes; suffix replay is exact within
  the locked tolerances; and every required camera key is present and non-constant.
- `REPLAY_DIVERGED`: environment loads, but released actions do not reproduce the released
  states or native completion. Report the first divergence and do not proceed to Stage B.
- `HISTORY_RESTORE_BROKEN`: uninterrupted replay works but the repeated suffix differs
  after restoration. Extend the snapshot carrier and repeat Stage A; do not run oracle
  branches yet.
- `VISUAL_SIGNAL_UNRESOLVED`: replay works, but required camera streams are missing,
  constant, or no event frame can be captured. Inspect saved frames before changing the
  observation interface.
- `SETUP_INCOMPLETE`: source, package, asset, or dataset setup did not complete inside the
  cap. Resume or repair setup without interpreting it as a scientific result.

Locked numerical tolerances:

- repeated-suffix MuJoCo state maximum absolute difference: `1e-9`;
- repeated-suffix image maximum absolute difference: `0` for uint8 observations;
- released-action replay state error is reported at every step; the gate requires maximum
  absolute error at most `1e-5` through the evaluated suffix and native success at the end.

The strict released-state tolerance is intentionally diagnostic. If deterministic action
replay accumulates larger float error but still reproduces the complete monitor trajectory
and success, amend the gate before Stage B using the observed error distribution rather
than silently weakening it.

## Resource cap

- setup/download: CPU Slurm job, `02:00:00`, 8 CPUs, 32 GB;
- replay/render: CPU OSMesa job on the `main` partition, `02:00:00`, 8 CPUs,
  48 GB. Neither the MIG nor full-GPU cgroup exposes `/dev/dri` render devices
  on this cluster; its installed OSMesa backend is therefore the valid
  headless renderer and avoids allocating an unusable GPU;
- no policy loading, policy training, encoding, or world-model work in Stage A1.

The replay job depends on successful setup. Before submission, inspect both `squeue` and
`sacct` for duplicate work. Record job IDs in `docs/JOB_LEDGER.md` and do not poll in a
foreground loop.
