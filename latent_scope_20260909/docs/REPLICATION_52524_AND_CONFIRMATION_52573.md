# Independent-prefix replication and locked confirmation

Verified 2026-09-14 using squeue and sacct. Calibration 52284 and follow-ups
52399, 52418, 52419, 52475, 52524 have completed. No duplicate active work
was present before submitting this follow-up. STATUS.json reflects older work;
this report records the newer artifacts without overwriting peer-session edits.

52524 completed with exit 0:0 in 04:32:43, but its scientific verdict is
QUALIFICATION_PILOT_INCOMPLETE: seven complete prefixes of eight intended,
nine attempted sources, one partial prefix. Report the missing eighth prefix.
Two of seven complete prefixes had apparent oracle recoveries (+28.57 percentage
points in screening). Four independent confirmation pairs had baseline 2/4 and
selected 4/4: two wins, no losses. Both wins came from attempt 7; attempt 4 tied.
These are conditional, exploratory results, not a population estimate or evidence
for learned composition. Retain earlier null confirmation results as well.

The restore problem was subsequently localized to the XML round trip changing
the compiled model, not proven renderer randomness. Compiled-model restoration
passed the live-versus-restored suffix proof in 52418. See RESTORE_REPAIR_20260913.md.

## Follow-up locked before execution

User requested continued work. Job 52573, lscope_rep_confirm, uses the frozen
52524 helper scripts/config and the new replication-confirmation entry point.
Keep attempt 4 candidate 4 and attempt 7 candidate 1 fixed against candidate 0.
Eight new common continuation seeds each: 203400–203407 and 203700–203707.
No reselection, new candidate proposals or reuse of previous confirmation seeds.
Recheck the original replay gate for each saved source. Save every branch and
exclude incomplete pairs from the paired gain. Preserve partial outcomes.

Budget: one GPU, 12 CPUs, 64 GB, four-hour hard cap and 3h45 soft stop including
startup. Static Python/shell checks passed and the runner diff changes only source
artifact, fixed candidate identities and seed base. Runtime validation is in Slurm.

Result: /mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/seed_confirm_52573/seed_confirmation_result.json

Positive gain across fresh seeds would strengthen the candidate-availability lead.
It would still not establish a learned selector or composition advantage. Stage C
comparison repairs and observation/readout qualification remain outstanding.
No automatic training or further GPU run is chained.
