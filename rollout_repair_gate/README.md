# Rollout Repair Gate

This directory contains the bounded Stage-1 experiment for the released
`quentinll/lewm-cube` checkpoint.  It tests whether matching predictor training
to the deployed five-step autoregressive rollout, and then matching the
planner-induced action distribution, repairs candidate ordering.

The experiment is a diagnostic/baseline, not a paper contribution.  Heavy
collection, encoding, training, CEM, and analysis run only through the Slurm
wrappers in `scripts/`.

See `PROTOCOL.md` for locked comparisons and `JOB_LEDGER.md` for execution.

