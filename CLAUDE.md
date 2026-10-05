# CLAUDE.md

Read and follow the root [AGENTS.md](AGENTS.md), the shared repository instructions.
It defines the current research, execution workflow, evidence rules and mandatory Slurm
policy. Keep those rules there rather than maintaining a conflicting second research brief.

## Execution reminders

- All substantial compute, including CPU analysis, goes through time-limited `sbatch`.
  No training, model loading, physics or rendering on the login node; no interactive or
  idle GPU allocations. Record submitted jobs and return rather than blocking to poll.
- Check both `squeue` and `sacct`, avoid duplicate work, cancel invalid/unneeded jobs,
  and preserve peer edits and existing run artifacts.
- Never let `nhatnc129` enter the cluster's top 5 users of the month for GPU, CPU or
  memory hours; stay at or below 90% of the 5th-ranked user, counting a planned job at its
  time limit. Check with `sreport` before each GPU job or CPU array (command in `AGENTS.md`).
- Keep offline, privileged-oracle and learned closed-loop results distinct. Use matched
  controls and held-out evidence for paper claims; the deadline does not change that standard.
