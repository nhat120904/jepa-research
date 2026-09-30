# Round 2 offline result and locked follow-up — 2026-09-26

Verified with both squeue (no active user jobs) and sacct: checks 54932,
training 54933_0 / 54933_1, and comparison 54934 all COMPLETED, exit 0:0.
Training took 53m13s and 52m57s, respectively. This is seed 0 development data.

Source: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_r2_compare_54934/comparison.json`.
These numbers measure offline coverage-gap retention, **not task success**.

| Tier / diagnostic | Matched continuation, lambda=0 | Co-design, lambda=.1 |
|---|---:|---:|
| FULL retained gap | .785 | .786 |
| Actual source CODE retained gap | .706 | .458 |
| Greedy predicted CODE retained gap | .293 | .243 |
| Expected predicted CODE retained gap | .501 | .463 |
| Four-sample predicted CODE retained gap | .398 | .345 |
| DIRECT retained gap | .388 | .388 |
| Source token perplexity | 131.20 | 1.37 |
| Source bank distinctness | .765 | .206 |
| Predicted bank distinctness | .234 | .078 |
| WM token CE (nats) | .558 | .086 |
| Prior token CE (nats) | .596 | .087 |
| Privileged true-prefix retained gap, first 600 decisions | .695 | .410 |
| Predicted-code collisions on label-distinct pairs, first 600 decisions | .719 | .917 |

Method minus control, paired root-bootstrap 95% intervals:

- CODE: -.248 [-.297, -.200]. The predeclared retention concern is triggered:
  source retention is .251 below its Round-1 parent, versus a .05 threshold.
- Greedy prediction: -.050 [-.108, +.007].
- Expected-code prediction: -.039 [-.092, +.013].
- FULL: +.001 [-.006, +.008]; DIRECT: approximately 0 [-.033, +.033].

The tested co-design loses source information while lowering prediction loss.
Prediction loss alone therefore gives a misleading impression of improvement.
The strong perplexity decrease and increased collisions support code collapse
as the immediate failure; they do not establish that every co-design objective
or weight must fail. No lambda sweep or new training is launched in response.
Greedy/expected prediction differences are not significant wins or losses here.

The lambda=0 expected reader remains a secondary variant: its point estimate
is promising but does not replace greedy as the predeclared primary method.
The privileged true-prefix diagnostic suggests exposure mismatch deserves
attention, but is neither deployable nor an upper bound. Reader adaptation alone
cannot distinguish identical codes given identical context.

FULL/DIRECT were independently retrained on matched data/update budgets and
are not bitwise identical. Maximum absolute score differences are 6.01 / 17.05;
do not assume these are harmless floating-point offsets. The CPU follow-up
reports centered score differences and argmax agreement. This is still a
single-seed comparison, not a replicated causal claim about arbitrary lambdas.

## Follow-up fixed before new closed-loop results

Follow the Round-2 protocol: both final checkpoints, roots 2100–2199, arms
P0, CODE8, CTA8, DIRECT8, CTA8E. Greedy CTA8 remains primary. No sealed roots,
no new reward/value targets, no checkpoint selection or training changes.

1. CPU audit and all unit suites: 4 CPU, 16 GB, 20 minutes. Read only train/dev
   metadata and saved score arrays; partition all-zero, nonzero exact-flat,
   near-tie and informative banks at the existing 1e-3 margin. Report time bins,
   decision- and root-weighted frequencies, and overrides by score variant.
   Overrides on flat banks do not imply causal harm or benefit.
2. GPU preflight: two array tasks, concurrency 1, 1x 3g.40gb MIG each, 4 CPU,
   32 GB, 20 minutes. Run the existing closed-loop preflight with count=0,
   once per checkpoint. This checks pipeline compatibility, not control success.
   It gates the rollout array to avoid repeatedly launching an invalid checkpoint.
3. GPU closed loop: 20 tasks, concurrency 2, 10 roots each, 90-minute hard limit
   per task. Tasks 0–9 use control; 10–19 use method, with matching root shards.
   Candidate coverage is logged to audit each selector's own visited states.
   Estimated total runtime is roughly 12–15 MIG GPU-hours based on prior runs;
   the allocation ceiling is 30 MIG GPU-hours plus 40 minutes of preflight.
4. CPU aggregation: 4 CPU, 16 GB, 20 minutes. Require exact complete root sets,
   one checkpoint per arm group and identical P0 outcomes across groups.
   Report per-model contrasts and paired method-minus-control, root-bootstrap
   intervals and discordant outcomes; reused dev roots are not confirmation.

Each stage has afterok dependencies and invalid dependencies are cancelled.
No automatic Round 3, new arena, or terminal-continuation campaign follows.
Source is snapshotted before submission and again inside each job.

Submitted: CPU audit/tests **54975** -> GPU preflight **54976** -> closed-loop
array **54977** -> CPU aggregation **54978**, all afterok-linked. Runtime
results of this chain are pending; source compilation/shell checks passed.
Immutable release: `cta_r2_followup_release_20260926T022231Z` under the shared
trajectory_innovation results directory. See JOB_LEDGER.md for output paths.
