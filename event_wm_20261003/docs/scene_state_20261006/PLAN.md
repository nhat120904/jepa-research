# Scene STATE experiment, 2026-10-06

User authorized implementing, training and evaluating scene with the learned unified backend.
Keep peer/core dirty files unchanged; source is frozen under this directory and remote scene_state_20261006.

Inputs: the official 40-dimensional state observation and play actions. Tokens: cube, two buttons,
drawer, window; same six fields and learned WM/h/BC skill architectures as cube/puzzle.
Scene observation has slider joint positions, not handle site positions: calibrate an affine contact
location map from TRAIN public effector/joint trajectories. Button anchors also come from TRAIN contacts.
No lock relation, prescribed action sequence, solver or scripted executor in training/planning/control.
The adapter is benchmark-specific preprocessing and must be disclosed as such.

General finite-attribute canonicalization is inferred from TRAIN event before/after values (at most8
values after rounding1e-5). Continuous positions/heights remain continuous. No finite-state successor
cache, since scene contains continuous attributes. No LHBL in this first run.

Prepare all available official train episodes (request up to3000),100validation episodes. Use the
cube/puzzle state configuration: WM30000steps, h150000steps, width2048, absdiff, imagined walks up to30,
pool300000, event-pos-only; skill40000steps, final checkpoint initially. A*20k expansions, same execution
chunk/contact-free rest logic. Development:1episode/task seed0; held-out evaluation:20/task seed3.
Save WM-stage/partial h checkpoints so time limits do not discard completed work.

Preparation includes semantic tests, per-identity event counts and diagnostic simulator alignment.
Training and loop use only public observations; diagnostic simulator fields remain output-only.
Run end to end first, then attribute any observed failure to extraction/WM/candidates/h/skill before
making a bounded correction. Report data count, actual source/config/checkpoints/jobs and limitations.

Quota check09:20UTC: account GPU89h vs90% fifth129h=116.1h; CPU682h vs1167.3h;
memory6200677MBh vs9482886.9MBh. Fetch57767 (CPU2/8GB/25min) COMPLETED18s.
Before every subsequent GPU job, repeat the ranking and include its full requested time limit.

Actual data:1000train/100validation episodes;1,001,000/100,100frames. Preparation57771 completed39s.
Training+pilot57772 has4h limit; dependent100-episode evaluation57774 has1h limit and afterok:57772.
Both completed successfully. Initial scene result34/100; subsequent support repair87/100 on a different evaluation seed. See STATUS.md for current results and paired comparison.
The limit is a resource cap, not a completion estimate. Per-episode JSON and partial h checkpoints are
saved to preserve completed work if a time limit is reached. Do not interpret partial evaluation as100episodes.
