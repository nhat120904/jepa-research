# One-version STATE rerun: puzzle, cube, scene

> Execute inline in this chat. User/AGENTS experiment instructions govern execution; preserve existing dirty files and earlier runs.

**Goal:** Train and evaluate all three families with the same frozen method implementation and recipe.

**Architecture:** Public observation adapter → generic rest-to-rest extraction → object/event transformer WM → learned h and weighted A* → event-conditioned BC skill. Add the same learned contrastive event-support prior and the same two-stage BC recipe to all families. No task solver, simulator transition labels, scripted executor or family-specific planner.

**Tech stack:** Existing NumPy/PyTorch, OGBench/MuJoCo, Slurm; frozen experiment source.

**Spec:** User request “chạy lại cả 3 bằng phiên bản thống nhất cho tôi” and supplied AGENTS.md.

## Shared configuration

- Data: first1000TRAIN episodes, first100play-validation episodes, for every family. Old cube/puzzle used3000; disclose this reduction. State track only.
- Train from scratch, seed0: WM30000updates; h150000updates, width2048, absdiff, imagined walk pool300000/max30; event-pos-only; TRAIN-derived finite-attribute canonicalization; prototypes8.
- Skill:40000updates at3e-4 on single-moving-object segments; then10000updates at3e-5 including genuine multi-object segments. Final checkpoint, chunk8/history2/release10, same architecture and normalization. Clip release supervision at the episode boundary to prevent assigning an old event to a new reset.
- Support:6000updates, width256, lr3e-4, seed61006; TRAIN contexts corrupted as density-estimation negatives; actor threshold at2ndpercentile of true play-validation scores. Added after h training, matching the scene recipe. No eval-state/goal/score training input.
- Evaluation: five benchmark tasks,20episodes/task on each predeclared fresh environment seed6 and7; max20000expansions, maxdepth30, weightedA*lambda0.6, event timeout250, execution4actions, rest5frames, recovery8steps. Native benchmark horizons differ (cube/puzzle1000, scene750); report them. No finite-only successor cache or LHBL.
- Different object counts, observation dimensions, adapter geometry, learned scales/prototypes/thresholds and model parameter counts are necessary data/interface differences. Checkpoints are trained separately; no zero-shot transfer claim.
- Login node only source/metadata/syntax/scheduler. Preparation/physics/model work/bulk analysis all sbatch. Explicit resources/time, no salloc, no idleGPU. Check bothsqueue/sacct and monthly90%-of-fifth cap before submissions. Distinct outputs, source hashes and job ledger.

## Tasks and verification

- [x] Freeze shared source/config; test BC episode-boundary supervision RED→GREEN. Generalize support trainer to explicit parent/events/out arguments, without family-dependent training branches. Run syntax and source checks.
- [x] One CPU preparation job builds all three datasets/events and integrates model interface, public-state alignment and supervision checks. No separate sequence of preliminary experiment gates.
- [x] Three identical training jobs after preparation: same command/recipe, different family dataset and output paths. Record source/config/checkpoints/actual data and job IDs.
- [x] Six evaluation jobs (seed6/7 per family), after their training job. Save every episode and distinguish partial from complete evaluation. A small dependent CPU aggregation job validates200complete episodes/family and writes per-task/seed results and uncertainty.
- [ ] Verify scheduler state and observed results; diagnose concrete failure or configuration error and make bounded fixes under the quota. Do not report historical scores as these rerun results.

## Resource reservation

CPU preparation4CPU/24GB/25min; each of3training jobs1GPU/6CPU/24GB/4h; each of6evaluation jobs1GPU/12CPU/24GB/2h; aggregation2CPU/4GB/5min.
Full planned maximum:24GPUh,217.834CPUh,600405.3MBh. Current month snapshot:91GPUh,697CPUh,6264884MBh; ceilings120.6GPUh,1170.9CPUh,9780289.2MBh. Repeat ranking before each GPU submission and include all pending commitments. No speculative duplicates or array.

## Review focus

Episode resets must not acquire previous-event BC labels; inherited checkpoints must preserve normalization; finite canonicalization must leave continuous attributes continuous; support negatives must be described as density negatives; source/config/seeds/training stage and full episode counts must match the declared protocol before comparisons are trusted.
