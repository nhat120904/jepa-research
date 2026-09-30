# CTA on OGBench visual-cube (2026-09-27)

Arena: OGBench (Park et al., ICLR 2025) `visual-cube-single-play-v0` first (64x64 RGB, goal images, 5 evaluation
goals); cube-double later; scene later. Published success (paper table): GCBC 5%, GCIVL 60%, HIQL 89% (cube-single);
HIQL 39% (cube-double). Datasets: the official Berkeley server is blocked from the cluster; files come from the HF
mirror `ryanhoangt/ogbench_data` (unofficial; sha256 and array shapes recorded in `ogbench/data_manifest.json`).

## Proposal design (user decision 09-28, supersedes option (a) of 09-27)

The method design is kept literally: a frozen, goal-conditioned policy proposes K action chunks from K seeded noise
draws (candidate 0 = what the policy would execute); CTA's world model p(S | C, A) predicts one code per chunk in one
batched pass; the reader D(C, S, goal image) scores them; the best chunk is executed; replan.
- HIQL is not a proposal policy (per-step actions, no chunks) and is reported only as published context (cube-single
  89%, cube-double 39%). Subgoal candidates (option a) were dropped: they change what the world model predicts.
- No released goal-conditioned chunk policy exists for visual OGBench, so one is trained with published recipes only
  (`scripts/ogbench/gcfbc.py`: OGBench visual GCBC recipe + Q-chunking/FQL flow chunk head, H = 5). Its own success is
  P0; it is not tuned weaker or stronger.
- Fairness: every reranker (CTA, DIRECT, value, per-frame WM, endpoint WM) uses the same policy, the same seeded bank,
  the same training data and label type; compute per decision is reported. HIQL/GCBC numbers are context only.

## First measurement (decides the arena)
P0 and ORACLE8/ORACLE16 (privileged: exact restore, simulate each seeded chunk, execute the best by cube-target
progress) on the 5 evaluation goals, 20 episodes each. ORACLE >> P0: collect branched data and train CTA and the
matched rerankers. ORACLE ≈ P0: the design's premise (the bank contains better options) fails here; fallback arena is
LIBERO-Long with a released chunk policy (sampling-then-ranking gains are published there, e.g. RoboMonkey).
Jobs: cube-single 55415 → 55416, cube-double 55417 → 55418 (JOB_LEDGER.md).
