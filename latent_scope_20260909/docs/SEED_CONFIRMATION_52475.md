# Fixed-choice continuation confirmation

Submitted 2026-09-14 as Slurm job 52475, lscope_seed_confirm, one GPU,
12 CPUs, 64 GB, hard limit 04:00:00. Soft stop is 03:45:00 including startup.
This is a newly authorized bounded follow-up to 52419, not part of its original
eight-hour tranche. No duplicate jobs were present in squeue/sacct at submission.

Lock before new outcomes: reuse attempt 4 candidate 7 and attempt 5 candidate 3
from 52419, each paired with candidate 0. Eight new common continuation seeds per
prefix: 153400–153407 and 153500–153507. No candidate reselection or new proposals.
Both prefixes were selected from earlier apparent recoveries; this is conditional
confirmation, not a deployment-wide or independent-prefix headroom estimate.

Use the frozen simulator/policy helper scripts and config from 52419. Copy the
new confirmation runner into the job snapshot and record code hashes. Recheck
each saved source with the original replay gates before expensive branches.
Retain per-branch output and partial pairs; exclude incomplete pairs from gain.
Report per-prefix and pooled wins, losses and success counts for these new seeds
separately from the earlier screening and confirmation results.

Interpretation: a positive new-seed gain supports further independent-prefix
evaluation, but does not by itself establish the composition hypothesis. Zero
or negative gain gives no support for these two fixed choices and argues against
expanding Stage C on the strength of the original +25-point screening result.
Sixteen seed pairs on two prefixes cannot certify a tight population null.

Validation before submission: Python compilation, shell syntax, existing frozen
helper interfaces, and diff whitespace checks passed. End-to-end execution and
replay tests run on the allocated compute node. No Stage-C job is chained.

Result: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/seed_confirm_52475/seed_confirmation_result.json`.
Log: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/seed_confirm_52475.out`.
Submission is not completion. Verify squeue and sacct before reporting final state.
