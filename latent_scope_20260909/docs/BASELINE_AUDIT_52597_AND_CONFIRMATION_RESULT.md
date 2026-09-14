# Confirmation result and next bounded audit — 2026-09-14

Verified using both squeue and sacct: 52475, 52524 and 52573 are COMPLETED,
exit 0:0; no active user jobs before the new submission.

52573 completed in 02:10:47. The 16 fresh paired continuations yield baseline
5/16 versus fixed selected candidate 6/16, gain 6.25 percentage points, five
wins and four losses. Attempt 4: 2/8 versus 3/8; attempt 7: 3/8 versus 3/8.
Restore gates pass with zero state/image differences. This does not robustly
confirm the earlier screening gain. These are two selected prefixes, NOT 16
independent scenes or full-episode population headroom. No learned composition
advantage has been established. Do not launch Stage C training on this evidence.

Source: /mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/seed_confirm_52573/seed_confirmation_result.json

## Submitted continuation

Job 52597 (`lscope_base_audit`): one GPU, 12 CPUs, 64 GB, hard limit 01:30:00.
No dependent or duplicate array. Python compilation and shell syntax passed.

One initial episode per task, environment seeds 204000 and 204001; policy seeds
304000 and 304001. Compare SyncVectorEnv baseline execution with direct single-env
execution, both using the released MultiStepWrapper and cadence 16. Recreate each
scene with the same seeds. Record initial observation hashes, kitchen layout/style,
per-native-step action/observation hashes and success, plus first success time.
Check trajectory equivalence and chunk-sampled versus per-step-any success.

This is a differential harness smoke test, NOT the full upstream evaluator
comparison or a paper replication. Exact differences can also arise from runtime
nondeterminism and must be diagnosed, not automatically attributed to a wrapper bug.
The preserved source config describes the original 10-episode gate; this audit
explicitly overrides episode count to one per task per execution path and does
not use its minimum-success gate. No automatic larger evaluation is chained.

Expected artifact:
/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/base_audit_52597/execution_audit_result.json

Next: inspect this audit before scaling; complete upstream evaluator parity and
freeze a logged evaluation seed pool before a 50-episode/task current-protocol
baseline. Keep paper, current runtime baseline, and conditional oracle results
separate. Do not overwrite the peer-modified STATUS or historical ledger.
