# Qualification implementation and bounded execution

Date: 2026-09-13. Implements the bounded tranche recommended in
`RESEARCH_INVESTMENT_REVIEW_20260913.md`; it does not reopen full Stage-C training.

Update: 52410 failed replay before candidate evaluation. The corrected v2 compiled-
model snapshot passed CPU verification 52418; see `RESTORE_REPAIR_20260913.md`.
The current runner requires that proof, preserves full native dynamics, and uses
an untouched live reference. The historical v1 implementation description below
must not be read as claiming 52410 passed.

## Implementation changes

- Candidate datasets require explicit `source_episode_uid`, `prefix_uid`,
  `decision_step`, `candidate_count` and index. Exactly one causal window per
  candidate; missing/duplicate candidates, different pre-action observations,
  and source/file overlap across splits raise errors before training.
- Ranking groups include task identity. Non-finite scores and incomplete groups
  are rejected. Equal scores retain the smallest candidate index, including 0.
- Direct-value action tokens and left/right composer tokens have temporal/segment
  positions. The frame baseline reads the ordered predicted sequence with a GRU.
- Both segment arms receive identical full/subsegment endpoint, summary and
  reconstruction targets. Only composition/partition-consistency terms differ.
- Target reconstruction now predicts the ordered frozen visual/proprioceptive
  sequence, not the mean of a moving latent target. A small variance penalty is
  included in both arms. Observed composition has a gradient into the online
  target. This is an anti-shortcut design, not a proof against collapse.
- EMA teachers stay in evaluation mode during training. Observed and imagined
  context updates share GRU parameters. History remains a **bounded three-frame
  window**, and an imagined endpoint is a macro-step approximation, not a claim
  of exact recurrent full-history simulation.
- Direct and explicitly composed inference are separate interfaces. The trainer
  reports held-out partitions 1/3/5/7 for an eight-step segment; midpoint 4 is used
  in training. These evaluations do not select the checkpoint. All arms now
  select checkpoints on the same validation continuation BCE.
- Checkpoints record implementation version and manifest hash. Old checkpoints
  are not compatible with these changed architectures.
- Offline encoding supports explicitly selected spatial patch-grid features;
  the existing config remains an explicitly labeled CLS baseline. Selecting a
  patch grid requires matching feature dimensions in encoder and model configs;
  no re-encoding campaign was launched.
- A bounded CPU ridge readout script compares true ordered sequences,
  history+endpoint, optional checkpoint targets, and a privileged native-monitor
  reference. It enforces manifest source splits and refuses missing/one-class
  training data. It is a diagnostic, not a method-comparison or authorization gate.

## New simulator runner; old Stage-B files preserved

`scripts/run_stage_b_qualification.py` uses the existing frozen GR00T runtime,
native success predicate and OSMesa renderer. Previously dirty Stage-B profile,
protocol, ledger and STATUS files were not overwritten. This document and the new
qualification result are the status sources for this tranche.

The source rule is locked before labels: first observed eight-step chunk boundary
with 1–4 Scrub contacts and incomplete progress, after at least 16 native steps,
before 70% of the native horizon. Source future success is never inspected.
At most 12 source attempts target eight qualifying prefixes. Non-reached attempts
are retained. This is a **conditional partial-contact panel**, not an unbiased
estimate over all deployment states; no population weighting is inferred.

Snapshots include controller goals and simple part/composite controller state,
interpolators, robot buffers/gripper state, simulator warm-start inputs, clocks,
task history, environment/NumPy/Python RNG, observable caches and wrapper history.
Source snapshots, live 16-step suffixes and exact candidate banks are persisted
before branch evaluation in exclusive files. Pickles are trusted local artifacts,
not an interface for untrusted uploads.

Every selected prefix must pass:

1. Initial snapshot alignment.
2. Live-source versus fresh restored carrier on identical 8- and 16-step suffixes.
3. Two fresh restores on the identical 16-step suffix.
4. Success-check calls must not mutate native task history.

Physical state tolerance is 1e-9 with equal history, labels, rewards and clocks.
Restored repeats require exact images. Source-versus-restore suffix images use
the preregistered normalized MAE <= 0.004 and p99 absolute pixel error <= 6;
maximum error is also recorded. Passing this tolerance does not establish a
causal explanation for rendering differences. Failed replay stops the job,
without relaxing thresholds or running the expensive candidate bank.

H=8, N=8, continuation cadence=8. Candidate 0 stays the locked baseline. Every
intervention retains ordered native-step images, proprioception and progress
including native completion, plus post-state diversity. Source history images
have cadence 8 while intervention images have cadence 1; artifacts record this
explicitly and must not be blindly converted into a uniform-cadence manifest.

One scoring continuation seed is shared across candidates. The lowest-index
successful candidate is selected before confirmation. Observed recoveries are
evaluated against candidate 0 using two new common continuation seeds. If no
recovery is observed, candidate 1 versus 0 on the first completed prefix is the
predeclared repeat diagnostic. Confirmation is conditional on this screening
panel, not an independent deployment estimate. Incomplete pairs are excluded
from the reported paired confirmation gain.

## Resource and stopping contract

- GPU job: one GPU, Slurm hard limit **07:45:00**, including model startup.
- Scoring soft limit: five hours from batch startup.
- Confirmation/all-work soft limit: 07:15:00, leaving cleanup/write margin.
- Save results after every completed branch; partial prefixes/pairs remain
  explicit and are not counted as complete.
- No automatic Stage-C job, no N=32 expansion, no Rinse GPU sweep.
- Policy-server cleanup runs when the runner exits, including exceptions.
- The batch job copies scripts/config into its output and records SHA-256 hashes
  at startup, so later workspace edits cannot alter an already-started run.

## Verification and job ledger

- Login-node verification: `py_compile`, `bash -n`, `git diff --check` only.
- CPU job 52405: failed a test comparing grad and no-grad inference kernels
  bit-for-bit. Test repaired to compare the same no-grad kernel path, without
  weakening the no-future-input requirement.
- CPU jobs 52407 and 52408: completed; invariants passed. 52408 additionally
  checked ordered frame readout and rejected non-finite candidate scores.
- 52408 target probe: `TARGET_PROBE_NOT_READY`, missing
  `/mnt/data/nhatnc129/jepa/latent_scope_stage_c/features/manifest.json`.
- CPU job 52409: **COMPLETED**, eight invariant groups passed after teacher-mode
  and common-checkpoint-criterion changes. Target probe still reports the missing
  feature manifest; no readout evidence or training pass was fabricated.
- GPU job **52410** submitted with `afterok:52409`, one GPU, hard limit 07:45:00.
  Result path:
  `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/qualification_52410/qualification_result.json`.
  Log path:
  `/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/qualification_52410.out`.
  Submission is not completion. Verify both `squeue -j 52410` and `sacct -j 52410`
  when checking its state. No foreground monitoring loop is left running.

No measured headroom, target sufficiency, composition benefit or learned control
improvement is claimed by these implementation tests. Full model comparison
remains gated on a reproducible useful candidate problem, separate labeled
training data, target adequacy and a fresh held-out evaluation panel.
