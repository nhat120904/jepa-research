# Scene STATE status, 2026-10-06

Latest learned closed-loop result: **92/100 on fresh environment seed5**, versus **87/100 with the frozen parent skill on the same seed/hardware/planner**. Tasks1..5:20/20,20/20,20/20,18/20,14/20. Initial scene pipeline:34/100(seed3GPU),32/100(seed4CPU). Learned event-support repair:85/100(seed3CPU),87/100(seed4CPU). The92result uses a subsequently continued skill; these are distinct variants.

All scene jobs are completed and verified by bothsqueue andsacct, exit0:0:
-57772WM/h/BC training+pilot:2:35:29.
-57774initial100episode evaluation:47s.
-57808support training+105episode loop:2:31.
-57810matched controls+boundary diagnostic:4:29.
-57812continued skill+240episode loop:2:19.

The support model learns public-state/event/target compatibility from17,858TRAIN events with random context corruptions, calibrated on1851play-validation events. It corrected locked-handle/closed-drawer unsupported transitions; no lock relations, task names or prescribed order are encoded. WM/h/skill parent weights stayed frozen for the support study. CPU seed4 matched control32→87has55fixed,0regressions. Approximate density support is not a guaranteed physical feasibility classifier.

The latest skill continues57772for10000updates atlr3e-5, adding948genuine multi-object TRAIN events that the original BC filter excluded. WM/h/support/candidates/original termination stay frozen. Fresh seed5 parent/newskill comparison87→92has9fixed,4regressed. Extra training and inclusion change together; their individual contributions are not isolated. A one-training-seed100episode result remains preliminary (episode Wilson95 interval85.0–95.9%, paired +5point gain p≈0.267). No matched published-baseline win is established.

Acted-rest-only termination diagnostic did not help:task4/5=26/40 vsoriginal27/40; not adopted. Remaining8latest failures are2task4 and6task5, exhausting750steps. Cube/drawer coupled WM predictions and skill execution still produce repeated correction cycles. No claim of complete scene robustness.

Raw data:1000TRAIN/100play-validation episodes,1,001,000/100,100frames. Public40Dstate maps to5object tokens through a benchmark-specific observation adapter; handle geometry is calibrated from TRAIN public effector/joint trajectories. Separate scene model training and preprocessing must be disclosed. Simulator fields are diagnostic/scoring outputs only. NoLHBL or pixelreader is used in these STATE results; zero-shot transfer and pixel quality remain unmeasured.

Remote root:`/mnt/data/nhatnc129/jepa/event_wm/scene_state_20261006`.
Current source:source_skill_v4; model:support_57808/u_model.pt; skill:skill_side_effects_57812/skill/u_skill.pt; evaluation:skill_side_effects_57812/new_skill_seed5/u_closed_loop.json. Reproduction paths inLATEST_RUN.json; detailed results/configuration/uncertainty inREPORT.md; all episode logs/checkpoints/source SHA manifests and job ledgers are retained.
