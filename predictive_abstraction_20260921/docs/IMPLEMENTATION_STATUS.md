# Implementation handoff — 2026-09-21

## Implemented

- Original DINO-WM Wall RGB adapter; no state/proprio in returned observations.
- Explicit native-step temporal indexing and strict A-before-B semantics.
- Smooth action proposals independent of hidden state and goal, bounded by norm.
- Fixed image-only query targets for a CPU qualification screen.
- Counterfactual branch replay checks, per-prefix outcome records and sample banks.
- Independent episode-level train/val/test data collection; no privileged training arrays.
- Observed trajectory codec, action-conditioned summary predictor, query reader, and
  forecasting loss that preserves gradients through the frozen reader.
- Unit tests, bounded CPU Slurm launcher, source/upstream snapshots and provenance.

## Not implemented/completed yet

- Frozen DINO spatial feature cache and train-only anchor bank.
- Complete training/evaluation runners for all matched baselines.
- Trained codec/predictor or learned action ranking.
- Closed-loop MPC comparison, actual-observation threshold calibration, waypoint MPC.
- Held-out query-combination/horizon protocol and multi-seed confirmation.

These are subsequent implementation steps, not completed results. No pretrained VLA,
GR00T, JEPA-WMs predictor, learned composer or actor-critic is required by this pilot.

The development training runner and all five declared arms are now implemented; they
remain unexecuted pending completion/inspection of frozen feature job 53550. Thus the
item above means empirically completed/validated, not merely source code present.

## Latest qualification

Jobs 53521 and 53529 completed. Proposal/query/data plumbing is qualified for the learned
pilot: ordered-success candidates occur in 19/48 v2 prefixes and are non-saturated.
See `HEADROOM_RESULT_53521_53529.md`. No learned predictor has run yet.

## Checks before submission

Python source compiled and Slurm script passed `bash -n`. Unit/model/environment tests
run on the compute node at the start of the CPU job; compilation is not a passing test.
See `JOB_LEDGER.md` for submitted job state and artifact paths.
