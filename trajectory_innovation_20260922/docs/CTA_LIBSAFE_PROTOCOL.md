# CTA on LIBERO-Safety (trajectory arena)

Decision 2026-09-28 (user): the trajectory claim moves to LIBERO-Safety (ECCV 2026) with the released pi0.5; OGBench
cube stays the endpoint arena (cube-single only; cube-double cancelled). Code: `ti_wm/libsafe_runtime.py`,
`ti_wm/pi05_policy.py`, `scripts/libsafe/`. Pinned: LIBERO-Safety `19ec8df`, openpi `215abfb`, checkpoint
`LIBERO-Safety/pi05_libero_safety@e66d84b`.

## Why this arena tests the trajectory part

The native cost is evaluated at every control step (`info['cost']` from the BDDL `(:constraints ...)`: robot–obstacle
contact, held object–obstacle contact, gripper force > 100 N). Obstacles in `(:dynamics ...)` are mocap bodies: the robot
cannot push them, so a contact in the middle of a chunk leaves no mark in the end frame. Labels are the benchmark's own.

Checked in the BDDL files (not taken from summaries): of the 45 tasks with contact constraints, **13 have obstacles
that actually move**:
- obstacle_avoidance (TSA) L1, 5 tasks: linear, 1–2 mm per control step, or circular (periods 300 and 60 steps);
- obstacle_avoidance_human (FSHOA) L1, 5 tasks: linear, 2–2.5 mm per step;
- human_safety (HRI) "banana on the plate in my hand", L0–L2: linear, 1 mm per step, 10 cm back and forth.

The other `(:dynamics)` obstacles have `motion_travel_dist 0` (static mocap: contact still leaves no mark, but an
endpoint reader can infer it from geometry). TSA L0/L2 obstacles are free bodies (contact can displace them).
Correction to `CTA_ARENA_AND_PUSHT_STRENGTH_20260928_VI.md`: "35/45 moving" is wrong (13/45), and the motion generators
are plain iterator classes (no `yield`), so `copy.deepcopy` clones them; no rewrite is needed.

## Published pi0.5 numbers (project page tables, 3 seeds)

| Suite | SR L0 / L1 / L2 (%) | violation rate L0 / L1 / L2 (%) |
|---|---|---|
| HRI | 84.7 / 88.7 / 83.3 | 6.7 / 6.3 / 8.0 |
| TSA | 58.0 / 62.7 / 56.7 | 14.0 / 12.3 / 16.0 |
| FSHOA | 55.3 / 58.7 / 51.3 | ~6–10 (figure labels ambiguous; data-scaling table: 10.0 at SR 51.3) |

Most pi0.5 failures are collision-free incompletion (deadlock, timeout; the authors say so). So the trajectory-specific
headroom is bounded by the violation rate (about 6–16% of episodes); success headroom is mostly progress-type, where
an endpoint prediction suffices. Both are reported; the trajectory claim rests on violations and safe success.

## Protocol

- Episode: openpi LIBERO client procedure (10 dummy steps; images rotated 180 deg, resize-with-pad 224; state = eef
  pos + axis-angle + gripper qpos; chunk of 10, first 5 executed). Episode limit = longest training demo of the task
  (LeRobot `meta/episodes.jsonl`) rounded up to 50; the paper's limit is unpublished, so SR is compared with the paper
  only as a sanity check. Each episode re-seeds LIBERO's global NumPy RNG with its root id before reset.
- Roots: `700000 + 100000*suite + 10000*level + 1000*level_id + init`.
- Bank: K = 8 chunks per decision from one batched pi0.5 call; row k integrates from noise seeded by
  `candidate_seed(root, d, k)`; P0 = candidate 0.
- Metrics per episode: success (native `_check_success` within the limit), violation (any native cost > 0), safe
  success (both). Paired by root across arms.
- ORACLE8 (privileged): each candidate's 5 steps simulated in a camera-free twin loaded with the exact state; score =
  goal progress (1 on success) − 2·violation. Fidelity of the twin is logged on every executed chosen chunk.

## Steps

1. `setup` (CPU): environment, checkpoint, assets, tokenizer; probe = task build, init counts, twin fidelity, step
   timing. **55603**.
2. `headroom` (one MIG slice): P0 and ORACLE8 on TSA L1 and FSHOA L1, inits 0–4 (50 roots per arm). Gives P0 against
   the paper, the oracle's success/violation, how often a bank mixes violating and violation-free chunks, and cost per
   episode for budgeting the branched collection.
3. Branched collection → CTA and matched arms (DIRECT, ENDPOINT, per-step FRAME WM) → closed loop, as on PushT/OGBench.
