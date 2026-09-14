# Confirmation result and independent-prefix replication

Verified 2026-09-14 using squeue and sacct. Job 52475 completed, exit 0:0,
elapsed 02:11:50; artifact verdict SEED_CONFIRMATION_COMPLETE.

New-seed paired outcomes: selected 8/16 versus baseline 5/16 (+18.75 percentage
points), four wins and one loss. Attempt 4: selected 5/8 versus baseline 2/8,
three wins, zero losses. Attempt 5: both 3/8, one win and one loss.
This is conditional evidence for a useful fixed action on one saved prefix.
Sixteen seed pairs are not sixteen independent prefixes. The earlier four
confirmation pairs remain a separate zero-gain result; neither set is discarded.
No population headroom, learned-selector improvement or composition benefit is
established by these conditional results.

## Locked follow-up

The user authorized continuation after completion. Run one independent-prefix
replication, at most six allocated GPU-hours including startup. Four hours for
screening, total soft stop 05:45, Slurm hard stop 06:00. This is a new bounded
tranche, not a retroactive extension of the original pilot budget.

Target eight new Scrub prefixes, at most twelve source attempts, environment
seeds 74000 onward. Preserve the original first-partial-contact anchor rule
without inspecting source future success. Use N8/H8 and cadence 8. Candidate 0
remains baseline; select lowest-index successful candidate before confirmation.
Use source/proposal/scoring/confirmation seed bases 163000/173000/183000/193000.
Two new paired confirmation seeds per apparent recovery, same rule as 52419.
If none recover, preserve the original predeclared null-repeat diagnostic.

Do not reuse the old verified source as attempt zero. The standalone replication
runner differs from the existing qualification runner only by a configuration
switch disabling this reuse. Old helper files, results and dirty code are preserved.
The existing CPU proof checks restore implementation, not the new scenes: every
new prefix must independently pass its original live-versus-restored suffix gate.
No replay tolerance is relaxed. The batch snapshots scripts/config and their hashes.

Report all attempts, complete and partial prefixes, apparent recoveries, and
independent-seed confirmation separately. A positive screening result alone does
not open Stage C. Replicated confirmation gains across multiple new prefixes
would justify designing the larger gate. Null/negative confirmation would argue
against scaling on the strength of the selected old prefix. No automatic retry,
candidate-count expansion, or Stage-C job is chained.

Static Python/shell/JSON checks passed. The runner diff was checked to contain
only the source-reuse switch. Runtime and fresh-scene validation occur in Slurm.

Submitted 2026-09-14: job **52524**, `lscope_q_replicate`, main partition,
one GPU, 12 CPUs, 64 GB, six-hour hard limit. No duplicate work appeared in
squeue/sacct before submission. Submission is not completion.
Result: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/qualification_52524/qualification_result.json`.
Log: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/qualification_52524.out`.
