# Job ledger: cem_stopping_20260929

## 2026-09-29 continuation refinement (this chat)

- Verified with squeue+sacct: 55928, all eight 55934 shards, 55935 completed.
  Both tasks have exactly dev roots 0..99. V1 gap CV success 68/100 on both,
  versus tuned fixed 67 Cube / 66 Reacher, but more search (19.36 vs 4.32,
  42.24 vs 30.45 iterations). Test remains unopened.
- **55949**: CPU, main, 2 CPU, 8 GB, 20 min cap. Protocol
  `docs/REFINEMENT_V2.md`; causality tests + episode-grouped CV of fitted
  continuation gain, with fixed-pair and all original controls. Lambda .002
  primary / 0 quality sensitivity locked before results; small depth-2 trees.
  Snapshot/output: `/mnt/data/nhatnc129/jepa/cem_stopping/refine_v2_55949/`.
  Syntax checks passed; runtime tests run in job. No extra simulation, test
  outcomes or model checkpoint updates. Source snapshots are copied before use.
  Account before work: GPU 207/243 h, CPU 1778/1945 h, memory 14.3M/25.1M;
  queue empty. This single CPU job reserves at most 0.67 core-h.
- 55949 COMPLETED 46 s; causality/serialization/path tests pass. Grouped-CV
  fixed-pair (1,10) improves Cube to .71 at 6 iterations, versus fixed .67 at
  4.32. Continuation lambda .002: Cube .65 at 2.92 (vs pair -.06 CI
  [-.11,-.02]); Reacher .67 at 13.15 (vs fixed .72 at 25.5, -.05 [-.14,.03]).
  Quality-only continuation .68/.62. Thus reduced compute is real in replay,
  but non-inferiority and a method advantage are NOT established.
- **55950** live stopping validation: MIG 1x3g.40gb, 4 CPU, 32 GB, 45 min.
  All 100 dev roots/task, continuation .002, fixed-pair .002, quality-fixed,
  original quality-gap; each uses its held-out-fold policy. Callback interrupts
  upstream CEM after chosen update; compare exact action/cost prefix/outcome
  to recorded tree, then measure actual planning latency. Source snapshot and
  output `/mnt/data/nhatnc129/jepa/cem_stopping/live_v2_55950/`.
  Before GPU submission: empty peer queue, sreport still 207 GPU / 1778 CPU /
  14.3M memory. Planned +.75 GPU-h, +3 core-h fits all caps.
- **55951**: CPU, 2 CPU, 8 GB, 10 min cap. One predeclared capacity reduction:
  first-population contextual budget stump, same grouped folds/lambdas as v2;
  at most one split/plan, three inputs, no depth/penalty sweep. Also measure v2
  training versus OOF to diagnose overfitting. Output/source snapshot
  `/mnt/data/nhatnc129/jepa/cem_stopping/budget_v3_55951/`. No new rollout data.
- 55951 COMPLETED 7 s. Stump lambda .002: Cube .67 at 3.94 iterations,
  Reacher .62 at 21.57; quality-only .68/.69. No robust method advantage.
  Diagnosis: v2 all-dev continuation Reacher .89 train vs .67 OOF (.002),
  overfitting. Uniform branch weighting also fits plan 2 for many unvisited
  states; constant stump budgets can underperform the fixed-pair baseline.
- **55952**: CPU 2 CPU/8GB/10min, same script after the documented correction:
  best fixed-pair anchor, exactly two reached-branch updates; unchanged features,
  lambdas and grouped folds. Output `/mnt/data/nhatnc129/jepa/cem_stopping/budget_v3_55952/`.
- 55952 COMPLETED 57 s. Reached-branch correction recovers Cube .71 at 6
  iterations for lambda .002, exactly the fixed-pair baseline (no adaptive
  gain); Reacher .61 at 19.65 versus fixed .72 at 25.5. Quality-only Cube .71,
  Reacher .63. Keep all negative results; no further threshold sweep justified.
- **55953**: live integrity/timing check of corrected budget policies on roots
  0..15/task (128 episodes, four arms), with fixed baselines running plain
  upstream CEM at their selected n_steps WITHOUT recorder overhead. Also exact
  action/outcome checks versus the full tree. MIG 1 slice, 4 CPU, 32 GB, 20 min.
  `/mnt/data/nhatnc129/jepa/cem_stopping/live_v2_55953/`. Before submission,
  squeue/sacct show only 55950 active; sreport 207 GPU/1778 CPU/14.3M memory.
  Combined limits 55950+55953 reserve 1.084 GPU-h, 4.334 CPU-h, below caps.
  This small check validates deployment, not another outcome-tuning experiment.

- Outcomes recorded afterwards (checked with sacct: 55950 COMPLETED 9m52s,
  55953 COMPLETED 1m56s, both `LIVE_V2_OK`, every episode `LIVE_MATCH_OK`).
  55950 (100 dev roots/task, held-out-fold policies, real early stop): Cube
  continuation .65 / 2.92 it / 0.08 s planning per episode, fixed pair .71 / 6.0
  / 0.14 s, fixed K .67 / 4.32 / 0.10 s, gap v1 .68 / 19.4 / 0.39 s; Reacher
  continuation .67 / 13.2 / 0.28 s, fixed pair .64 / 22.3 / 0.45 s, fixed K .72 /
  25.5 / 0.51 s, gap v1 .68 / 42.2 / 0.83 s. 55953 (16 roots/task) confirms exact
  deployment of the corrected budget policies; too small for comparisons.
- **55955** (main CPU, 2 CPU, 8 GB, 15 min): `scripts/diag_tree.py` on dev_v1 --
  success stability across K (flip rates, K-to-K correlation), whether the
  planner's own cost ranks its checkpoint means against their plan-1 distance,
  random-checkpoint baseline, mean-shift rule (CV). Output `diag_tree_55955/`.
- **55956** (mig 1 slice, 4 CPU, 32 GB, 20 min): `scripts/diag_render.py` --
  dataset versus rendered start/goal frames in pixels and LeWM latent for 20 dev
  roots/task, against the recorded cost scale. Motivation: our Reacher fixed-30
  success (.69 dev) is below the reproduced LeWM evaluator (.80) and the smoke
  showed a global Reacher shading difference. Output `diag_render_55956/`.
  Quota before both: GPU 208/243 h, CPU 1781/1945 h; queue empty.

- 55955 COMPLETED 2 s. Outcome stability: Cube success is nearly a function of
  the root (K10-K30 success correlation .91, 19 diagonal-mixed roots, adjacent
  flip rate .16); Reacher is much less stable (corr .51, 72 mixed roots, flip
  .29), so its large tree oracle (.96) is mostly selection among near-random
  outcomes. Random checkpoint per plan: Cube .65, Reacher .63. The planner's own
  cost of its checkpoint means barely tracks their plan-1 distance (Spearman .03
  Cube, .14 Reacher) and its argmin is almost always a late iterate. Mean-shift
  rule (CV) Cube .65 / 36 it, Reacher .69 / 49 it: no gain.
- 55956 COMPLETED 49 s. Renderer domain: Cube dataset-vs-render latent gap .017
  (final CEM elite cost 1.66): negligible. **Reacher gap 41 at both start and
  goal** (pixels 2.58/255 mean, global shading; final elite cost 3.7). Current
  frames (rendered) and goals (dataset) are in different renderer domains in our
  Reacher harness; likely part of the .69 vs .80 baseline gap. Cube unaffected.
- **55957_[4-7%2]** (mig, 4 CPU, 32 GB, 20 min): Reacher dev trees 0-99 with
  `--goal-source render` (goal state restored and rendered by the same renderer),
  `RUN_NAME=dev_v1_rgoal`; **55958** analysis afterok (reacher only).

- 55957/55958 COMPLETED. Same-renderer goal does NOT raise the Reacher baseline:
  fixed-30 .69 again (fixed 1 .52, 5 .66, 15-30 .69); success rises then
  plateaus, no degradation with more iterations. Margin-reliability again
  monotone (.47-.56 below gap/IQR .03, .75 at .3-1, .85 above 1). CV: fixed .61
  (chosen K unstable 5-30 across folds), converge .69 at 12.2 it (+8 [-1,+18] vs
  fixed CV), band .67, gap .62. In dev_v1 (dataset goals) converge CV was .62, so
  the convergence result is not stable across harness variants. The renderer gap
  is not the explanation for .69 vs the reproduced .80; that gap stays open.

Outputs under `/mnt/data/nhatnc129/jepa/cem_stopping/` (logs in `logs/`).
Every run directory stores `SOURCE_SHA256SUMS` and `STABLE_WORLDMODEL_COMMIT`.

## 2026-09-29

Quota before submission (`sreport`, month to date): GPU 206 h (5th-ranked
486 h, cap 243 h), CPU 1770 h (cap 1945 h), memory 14.2M (cap 25.1M). Peer CTA
array 55912 (2 MIG tasks, 2.5 h limit) running.

| Job | Work | Resources | State |
|---|---|---|---|
| 55928 | Smoke: unit tests; upstream-equivalence + prefix check; exact restore/replay; tree on dev roots 0-1 per task (root 0 labelled); tree == live for 7 rules; runtime per root | mig 3g.40gb, 8 CPU, 64 GB, 1 h | COMPLETED 4m07s, SMOKE_OK |
| 55934_[0-7%2] | Dev pilot trees, roots 0-99 per task (0-19 labelled); idx 0-3 Cube shards, 4-7 Reacher shards; `RUN_NAME=dev_v1` | mig 3g.40gb, 4 CPU, 32 GB, 1h15 | submitted |
| 55935 | Dev analysis (`analysis_55935.json`), afterany 55934 | main, 2 CPU, 16 GB, 30 min | dependency |

Smoke 55928 results (`smoke_55928/checks.json`): all 7 unit tests pass. Upstream
`CEMSolver` with the same seed and n = 30/10/3 equals the recorded checkpoint
means exactly (max abs 0), so the prefix property holds; repeated plans are
bit-identical. Branch replay max abs 0. Tree path equals the live continuous
episode for 7 rules x 2 roots x 2 tasks (k, success, final distance, and plan
costs identical). Start frame rendered vs dataset frame: Cube mean abs 0.44/255
(max 106, local), Reacher mean abs 2.57/255 (max 49, global shading); goals stay
dataset frames as upstream. Runtime: Cube ~65 s per root (100 s labelled; mostly
MuJoCo restore/replay on CPU), Reacher ~7 s (15 s labelled); one 30-iteration CEM
plan ~0.58 s. Cube dev root 0 starts inside the goal tolerance (trivial success
for every leaf); such roots are kept, as in the upstream protocol, and reported.
