# Conditional trajectory abstraction — research design

## Current status (2026-09-28)

Newer entries in [JOB_LEDGER.md](JOB_LEDGER.md) supersede this section. The dated blocks below
it are history, newest first; their "current"/"latest" labels refer to their own dates.

- **PushT, on-policy continuation (55345-55351, [protocol](docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md)):**
  200 paired development roots 2200-2399, one training seed. Successes P0 122, CTA4 142
  (+10 pp [2, 18], McNemar p=.024), CTA8O 138, ENDPOINT8O 138, DIRECT8O 124, oracle GEOM8 160.
  Native-score contrasts are null and CTA8O ties ENDPOINT8O, so this is a development signal,
  not a confirmed claim. Rounds 5/6 did not improve on Round 4 (Round 6 cancelled after the
  offline read).
- **OGBench visual-cube single/double ([protocol](docs/CTA_OGBENCH_PROTOCOL.md)):** branched
  collection, encode and train chain in progress; a GCIVL value teacher (55589) is training
  for a label-free reader. Evaluation jobs are submitted only after the offline ladder is read.
- **Paper:** CVPR 2027 (paper 2026-11-16 AoE, supplementary 2026-11-23). Draft in `paper_cvpr/`;
  required experiments in [the paper review](docs/CTA_PAPER_REVIEW_20260927_VI.md).

## History (newest first)

Paper, 2026-09-27: CVPR draft (`paper_cvpr/`) reviewed against the code and rewritten; findings and the list of
experiments still required are in [the paper review](docs/CTA_PAPER_REVIEW_20260927_VI.md). No new result is claimed.

Current work, 2026-09-27: Round 3 finished (55077–55079). Offline, the parallel WM with decision
losses (CTA3 .555) matches the source code (CODE .549), but in 100-root closed loop no arm beats
P0 (P0 62, CTA3 62, NLL8 71, DIRECT3 72, FRAME8 60; CTA3−DIRECT3 score −.022 [−.044, −.002]).
Dev NLL rises during training (overfitting on 800 episodes), and offline order inverts in closed
loop. Submitted chain 55137–55144: (1) on-policy diagnostic with all candidates simulated and
cross-scored, including PHYS8/GEOM8/FULL; (2) 1,200 new episodes with a perturbed training bank
and oracle-mixed execution, round-4 WM training with dev early stopping, closed loop. See
[protocol](docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md) and the job ledger.

Current Round 3 work, 2026-09-26: **55077 train / 55078 control / 55079 aggregate
submitted**, replacing 55072–55074 cancelled before starting. Budget is now 6000
updates x 32 effective banks, with dev curves every 1000 updates and measured
per-update/per-network timing. Reader array 55067 is throttled to 1 without killing
active shards. Parallel 16-token coordinate-FSQ WM, NLL-only versus NLL + frozen-reader
consistency/ranking; matched weighted DIRECT continuation and predicted endpoint
frame/proprio baseline. Source/reader remain fixed. Primary is normalized native
score excluding reset over 100 paired dev roots, with success also reported. See
[Round-3 protocol](docs/CTA_PARALLEL_ROUND3_PROTOCOL.md). No new result is claimed.
Reader-only 55066 completed in 16m10s; its 55067 closed loop is running and is preserved.

Current work, 2026-09-26: user authorized continued refinement. **55066 training,
55067 closed-loop array, 55068 aggregation submitted** as one end-to-end chain.
Frozen source/WM; matched reader-only continuation versus adaptation on source,
greedy and expected codes. Eight learned/policy arms, 100 paired dev roots, no
oracle selectors. Primary outcome is adapted greedy CTA versus P0 success.
See [reader refinement protocol](docs/CTA_READER_REFINEMENT_PROTOCOL.md) and
[job ledger](JOB_LEDGER.md). No new training/control result is claimed yet.

Latest job result, 2026-09-26: **55052 COMPLETED** in 36m30s, recovering geometry
training 55018 without retraining. All ten dev roots finished: P0 7/10, PHYS8 4/10,
GEOM8 8/10, FULL8 5/10, CODE8 5/10, CTA8 6/10, CTA8E 4/10, DIRECT8 4/10.
No confirmed improvement; expected-code's offline advantage did not translate to
higher success here. Reader adaptation on deployed predictions is now submitted
with frozen source/WM and a matched continuation control, as recorded above.
See [closed-loop result and next step](docs/CTA_GEOMETRY_CLOSED_RESULT_55052_VI.md),
[offline result and recovery](docs/CTA_GEOMETRY_RESULT_55018_VI.md), and the job ledger.

Latest execution decision, 2026-09-26: user reiterated **end-to-end first, then debug**.
The [PushT geometry iteration](docs/CTA_GEOMETRY_E2E_20260926_VI.md) connects existing-data
relabeling, codec/WM training, offline diagnostics and ten-root closed loop in one
bounded job. Scientific diagnostics do not gate implementation; technical integrity
checks remain. The job ledger records submission state. This supersedes the sequential
research-gate workflow proposed in the recent target/task qualification notes.

Status, 2026-09-26: CTA rounds 0 and 1 are complete on development data.
Round 1 greedy CTA succeeds on 66/100 closed-loop dev roots vs P0 62/100;
the paired +4 pp interval [-5,+13] includes zero. Round 2 training is complete:
lambda=.1 co-design loses source-code retention (.458 vs matched control .706)
and collapses code usage; lower prediction loss does not indicate improvement.
Round-2 closed-loop results are not yet available. See the
[offline result and follow-up](docs/CTA_ROUND2_OFFLINE_RESULT_54933.md).
The first Round-2 runtime preflight failed. A DINO/PCA autocast mismatch has
been repaired and regression tests pass, but replacement GPU preflights timed
out; closed-loop jobs were cancelled before execution. WM audit 54982 completed.
Those jobs are no longer active. See the
[endpoint-scope correction and status](docs/CTA_ENDPOINT_SCOPE_AND_STATUS_20260926_VI.md).
The endpoint-versus-trajectory codec comparison was stopped after the user
challenged its relevance to the central trajectory claim. Tests 54988 passed;
54989/54990 were cancelled without final comparison results.
See the [research reset and claim/evidence map](docs/CTA_RESEARCH_RESET_20260926_VI.md).
Next: [native task qualification](docs/CTA_TASK_QUALIFICATION_20260926_VI.md).
RinseBowls is the priority for premise checks, not a qualified training arena.
A bounded CPU audit checks native semantics and reuses saved intervention labels;
no new model training or policy rollout is chained. See the ledger for submission state.
After the user's GPC comparison request, the immediate PushT priority is a
[registration-target audit](docs/CTA_PUSHT_GPC_TARGET_REVIEW_20260926_VI.md).
CTA can support endpoint and temporal objectives; RinseBowls qualification is not
a prerequisite for continuing a properly controlled PushT test.
See [research review (Vietnamese)](docs/CTA_RESEARCH_REVIEW_20260925_VI.md),
[Round-2 protocol](docs/CTA_ROUND2_PROTOCOL.md),
[debug log and results](docs/CTA_DEBUG_LOG.md), and [job ledger](JOB_LEDGER.md).
The proposal below records the original design; later protocol amendments
allow task labels for the reader. Closed Wall and Scrub runs remain closed.

**Question:** can a compact, action-predictable code for a future trajectory preserve
new information relative to observed history, answer independent temporal queries,
and improve the accuracy/compute trade-off of policy-guided planning?

Borrowed principle: conditional coding with information already available to the
decoder, from learned video compression. The proposed adaptation is a query-trained
code for the *whole future segment*, plus action-conditioned prediction of that code.
The potential contribution must be demonstrated against conditional frame compression
and a cached direct-query model, not merely against frame-by-frame reconstruction.

Flow:

    history -> shared cached context C ---------------------------> query reader
       |                                                             ^
       + frozen proposal policy -> K candidate chunks                |
       + C + each action chunk -> WM -> predicted innovation code ---+
                                                          queries -> scores
                                                select chunk -> execute prefix

The future-observation encoder is used only for training the code. The planner never
observes future frames. Decoder receives C, the predicted code and queries; it never
receives the candidate actions directly. Direct-query baselines may receive actions.

Documents:

- [Original execution plan (2026-09-22) and CompPlan interpretation](IMPLEMENTATION_PLAN.md)
- [Job ledger](JOB_LEDGER.md)
- [Method, equations and novelty boundaries](RESEARCH_DESIGN.md)
- [Arena, action proposals, qualification and stopping rules](EXPERIMENT_CONTRACT.md)

Original entry-point recommendation (superseded by the research reset above):
qualify frozen Diffusion Policy on its native gym-pusht setup first.
Its published model card is a feasibility reference, not our achieved baseline or
proof of selection headroom. An original DINO-WM reproduction is a separate control;
its action/runtime interface must not be silently mixed with gym-pusht.

All compute follows the root AGENTS.md: sbatch only, explicit limits, unique snapshots,
queue/accounting checks, no duplicate jobs or idle GPU allocations.
