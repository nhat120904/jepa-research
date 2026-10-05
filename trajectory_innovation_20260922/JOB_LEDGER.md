# Jobs

CTA v2 (2026-09-28, user: "debug and improve CTA on PushT and OGBench"). Diagnosis of 55565 (OGBench) and the PushT
chain:
- Readers memorized small data. OGBench has 4.7k informative banks; FULL train rank loss .003 while held-out
  retained gap is .80, below DIRECT's .94.
- Stage 1 had no model selection.
- The single-goal ranking target let the code collapse.
- PushT's codec/readers (55018) were trained on Round-0 data only and then frozen.

v2 changes:
- Scorers train on several goals per bank: the official goals and hindsight goals from later decision frames. Labels
  are recomputed from the logged cube positions and exactly match the collected ones (max error 3e-8).
- Every module group is selected on its held-out split.
- Dropout 0.1 everywhere; optional dropout added to the ti_wm modules (default 0, old checkpoints unchanged).
- A leave-goals-out mode.
- PushT: every module retrained from scratch on Round-0 + Round-4 (re-encoded with full tokens) + on-policy banks.

Jobs:
- OGBench smoke **55609** COMPLETED in 4:22 (code path only).
- **55610_0** CANCELLED at 7 min: the code collapsed to 2 distinct values per code by step 500. The FSQ saturation
  penalty was 0 in v1; it is now 1.0.
- Resubmitted as **55614_0** (in-distribution; mig, 8 h;
  out `/mnt/data/nhatnc129/jepa/ogbench/cta_v2/iid_s0_55614`).
- **55637_1**: goals 4 and 5 held out (never trained on or selected with). HELD (`scontrol hold`) behind the
  code-capacity test.
- v1 closed loop 55601 final (100 dev episodes): P0 4, CTA 14, FRAME 18, ENDPOINT 21, DIRECT 23.
  CTA is the weakest learned arm; scorer ms/decision: DIRECT 3.1, CTA 5.5, ENDPOINT 5.7, FRAME 15.5.
- 55614 stage 1 (held-out, mean over the 5 official goals): CODE .83, FULL .88, DIRECT .94. The gap between CTA and
  ENDPOINT follows from CODE < FULL.
- **55645_2**: code capacity 8,8,8,5,5 (~16 bits/token), otherwise v2; stage 2 3000 steps.
- **55614 COMPLETED** (3:31). Held-out ladder, own goal (5-goal mean in parentheses):
  - FULL .81 (.88); CODE .89 (.83; v1 own .75); DIRECT .94 (.94).
  - CTA .82 [.64, .93] (.88); ENDPOINT .92 (.89); FRAME .82 (.91).
  - Code: 2.5 distinct values per 16-token code, perplexity per position 133, siblings' codes differ 97%.
  - v2 fixed the lossy code but not CTA ≤ ENDPOINT/DIRECT in distribution.
- **55658_3**: nested dropout on code tokens (`--nested 0.5`; `ti_wm.cta.nested_drop`, masks in Scorer and
  FutureDecoder, default off) against token redundancy.
- **55656** (v1) and **55657** (v2), CPU: `ogb_extra_tiers.py`, reading S through the stage-1 decoder with the FULL
  reader (600 held-out banks).
- **55615 COMPLETED**: play features `ogbench/enc_play/visual-cube-single-play-v0`; 1,001,000 frames, cube z median .02.
- Play smoke **55617** COMPLETED; **55618_0** play-trained, 500k frames, all goals, running.
- **55645_2** and **55658_3** requeued and HELD, to give the PushT closed loop a slot.
- PushT **55616 COMPLETED** (2:17). Held-out ladder, geometry retained gap, 16 goals, 4,260 selection banks:
  - FULL .76 [.73, .79]; **CTA .67 [.62, .71]**; ENDPOINT .64; CODE .63; DIRECT .49 [.43, .54].
  - Code: 1.7 distinct values per code; siblings' codes differ 72%.
- DINO-WM smoke:
  - 55638 FAILED (checkpoint unpickling needs accelerate).
  - 55663 FAILED (`VWorldModel.eval()` returns None).
  - After both fixes, **55665** SMOKE_OK: adapter tests pass (the PD path equals gym-pusht exactly); DINO-WM scores
    8 candidates in ~56 ms per decision.
- **55666_0-7%2**: PushT closed loop, roots 2200-2399 (the 55349 roots). Arms P0, GEOM8, CTA4, CTAV2, ENDV2, DIRV2,
  DINOWM; also logs FULLV2 / CODEV2.
- 55656 / 55657 (CPU, 600 held-out banks): reading S through the stage-1 decoder does not help.
  - v1: code_dec .76 / .67 (own / 5-goal) vs code .85 / .84.
  - v2: code_dec .79 / .86 vs code .87 / .84.
  - The reader is not the bottleneck; this direction is dropped.
- **55675**: OGBench v2 closed loop, arms CTA / DIRECT / ENDPOINT / FRAME, dev episodes 0-19 × 5 tasks; P0 comes from
  55601.
- v1 closed loop 55601 (peer session), 100 dev episodes: P0 4%, CTA 14%, DIRECT 23%.
  CTA − DIRECT = −9 pp [−18, 0], p = .11.
- **55615**: encode the official play dataset (1M transitions; `ogb_encode_play.py`) for the play-trained variant
  (`ogb_cta_train_play.py`: CTA vs DIRECT when every state has a single logged action).
- PushT smoke **55611** COMPLETED in 2:10.
- Round-4 full-token encode **55612** COMPLETED in 36 min: 32,590 decisions; informative banks 13,977 standard and
  26,856 perturbed. Output `cta_v2_encode_55612/{r4std,r4pert}`.
- **55616** v2 train (`--with-perturbed`), waiting for a GPU slot.
- **55625** (CPU) builds auxiliary labels for v2.1 (`scripts/cta_v2_aux.py`, opt-in flags
  `--aux/--success-bonus/--hindsight`): end block poses, native success, hindsight goals.
- Closed-loop wrapper `scripts/slurm_cta_v2_closed.sh`: arms P0 / GEOM8 / CTA4 / CTAV2 / ENDV2 / DIRV2 / DINOWM.
  DINOWM is the official DINO-WM PushT checkpoint scoring the same candidates through `ti_wm/dinowm_scorer.py`.
  - The PushT agent is a kinematic PD-driven body, so absolute targets convert exactly to DINO-WM's relative actions.
  - DINO-WM's pusht_noise rel = target − agent position, max error 3e-5 (checked in 55613).

No result is claimed at submission.

PushT policy-bank CTA on-policy continuation (2026-09-27, authorized after Round-5/6 read):
source release `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_onpolicy8_20260927_OLBunz`
(`SOURCE_SHA256SUMS`); **55345** smoke COMPLETED 0:0 in 58 s (unit tests; collection, encode,
training, closed-loop and aggregation on tiny live inputs). It exercises
CTA4-driven K=8 collection, encoding with the frozen original PCA, warm-started CTA/direct/endpoint
training, and paired control/aggregation on 2 roots. Full afterok chain submitted:
**55346** collection array 0-8%1 (50 roots/shard, MIG 4 CPU 64 GB 2 h);
**55347** encode (MIG 8 CPU 96 GB 3 h);
**55348** train (MIG 8 CPU 96 GB 3 h);
**55349** paired closed-loop array 0-7%1 (25 roots/shard, MIG 4 CPU 48 GB 2 h);
**55351** aggregate (main CPU 2 CPU 8 GB 15 min), from immutable CPU wrapper release
`/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_onpolicy8_aggregate_20260927_i6e7hq`.
Train roots 32250-32649, checkpoint-selection roots 32650-32699, and held-out control roots
2200-2399; arms P0/GEOM8/CTA4/CTA8O/DIRECT8O/ENDPOINT8O.
No full-run result is claimed at submission; source and old runs are preserved. The endpoint WM starts
from Round-3 while CTA/direct start from Round-4, so its historical training budget differs;
this continuation is a diagnostic, not yet the matched paper baseline.
**Result (recorded 2026-09-28 from `cta_onpolicy8_aggregate_55351/summary.json`; sacct: 55349_0-7 and 55351
COMPLETED):** 200 paired roots 2200-2399 (scope field: development roots, single training seed; pairing exact).
Successes P0 122, GEOM8 160, CTA4 142, CTA8O 138, ENDPOINT8O 138, DIRECT8O 124. Success contrasts: CTA4-P0 +.10
[.02, .18] (McNemar p=.024), CTA8O-P0 +.08 [-.005, .165], ENDPOINT8O-P0 +.08 [-.005, .17], DIRECT8O-P0 +.01,
CTA8O-DIRECT8O +.07 [-.025, .16], CTA8O-ENDPOINT8O 0 [-.08, .08], CTA8O-GEOM8 -.11 [-.18, -.04]. Native-score
contrasts are all null (CTA4-P0 +.0005 [-.017, .018]). CTA4 within-bank Spearman .48, retained gap .65.

Paper figure (2026-09-27): **55314** `ti_cta_fig1` (mig 3g.40gb, 8 CPU, 48 GB, 45 min), `scripts/slurm_cta_fig1.sh` →
`scripts/cta_fig1_data.py`: policy-only closed loop (K = 8) on dev roots 2100–2107, ≤ 20 decisions each; decisions for
paper Fig. 1 picked by geometry-label spread across the proposals (no learned score in the rule); native 512 px renders,
CTA4 expected codes, source codes and scores saved to `cta_fig1_55314/`. Figure assets only; no result is claimed.
**55314 COMPLETED** in 59 s: 146 decisions; 6 rendered (roots 2102/2103/2105/2106/2104/2100); every clone reproduces the
model's observation exactly, labels match the log exactly, recomputed CTA4 scores match the log within bf16 noise.
Fig. 1 uses r2102_d16. Side finding (48 source codes): 1.8 distinct token values per 16-token code on average, 35/256
indices used; to be measured on all dev decisions (docs/CTA_PAPER_REVIEW_20260927_VI.md §6).
Paper builds 55309–55313 (CPU, seconds) compiled the revised draft.

Latest action (2026-09-28 16:15): **55589** GCIVL CANCELLED during its final 250-episode evaluation (training reached
500k; checkpoints params_100000-400000.pkl kept, 500k not saved because main.py saves after evaluating). The label-free
OGBench reader is deprioritised now that OGBench is the endpoint arena; the MIG slot goes to the LIBERO-Safety headroom run.

Latest action (2026-09-28, arena decision by user: pi0.5 on LIBERO-Safety for the trajectory claim; cube-double
cancelled): 55567 (had run 5 min) and 55568 CANCELLED. 55565 cube-single train COMPLETED 3:45; offline ladder (heldout
Spearman / retained gap): FULL .57/.80, CODE .52/.75, CTA .65/.91, DIRECT .70/.94, ENDPOINT .71/.94, FRAME .71/~.94 —
learned arms beat the privileged readers offline, CTA below DIRECT/ENDPOINT. **55601** cube-single closed loop
(P0,CTA,DIRECT,ENDPOINT,FRAME; dev episodes 0-19 x 5 tasks; mig, 6 h). LIBERO-Safety: protocol
[docs/CTA_LIBSAFE_PROTOCOL.md](docs/CTA_LIBSAFE_PROTOCOL.md); **55603** setup (main CPU 16, 4 h: openpi venv, pi0.5
checkpoint, assets, CPU probe). Headroom job is submitted only after 55603's probe is read.

Latest action (2026-09-28, label-free reader teacher, user-approved): **55589** GCIVL official visual-cube-single-play
command seed 0 (mig, 10 h; `slurm_ogb.sh gcivl`, W&B offline, ckpt every 100k, one final 250-episode eval). Its V(o, g)
labels the label-free CTA reader. Collection at submit: cube-single 271/300, cube-double 131/200; 55564/55565 and
55567/55568 (encode -> oracle-label train = ceiling) still waiting on collection.

Latest action (2026-09-28, queue trimmed on user request): cancelled the not-yet-needed eval/distill jobs 55566,
55569-55579 (none had started). Kept: collections 55512/55513, encode 55564/55567, train 55565/55568. Evaluation and
distill are submitted only after the offline ladder is read.

Latest action (2026-09-28, OGBench chain): pipeline smoke passed (55557 encode after the 4-future-frame fix; 55558
train; 55559 eval with all 7 arms; 55540 distill). Scorer ms/decision (8 candidates, smoke): DIRECT 2.9, ENDPOINT 6.7,
CTA 7.7, FRAME 15.4 (tokenization 6.6 shared). Full chain, s0 = training seed 0, dev episodes 0-19:
cube-single: encode 55564 (after collection 55512) → train 55565 → eval P0+CTA 55570, DIRECT 55571, ENDPOINT 55572,
FRAME 55573; distill 55566 → eval 55574. cube-double: encode 55567 (after 55513) → train 55568 → eval 55575-55578;
distill 55569 → eval 55579. Offline ladder (offline_ladder.json) is read before trusting the closed loop.

Latest action (2026-09-28, 08:30): cube-double eval 55418 cancelled during ORACLE16 (P0 .22 and ORACLE8 .52 complete;
ORACLE16 partial, 62/100 episodes) to free CPU quota for the pipeline smoke 55537 (encode) → 55538 (train) → 55539
(eval, all arms, 1 episode) and 55540 (distill). Collection: cube-single 56/300, cube-double 12/200 episodes.

Latest action (2026-09-28, OGBench branched collection): [protocol](docs/CTA_OGBENCH_PROTOCOL.md) fixed before any CTA
training. Headroom (dev eps 0-19): cube-single P0 7%, ORACLE8 37%, ORACLE16 34%; cube-double P0 22%, ORACLE8 52%.
Collection (scripts/ogbench/ogb_collect.py, CPU/OSMesa, replay of every executed chunk checked against its branch):
smokes 55506 (GPU/EGL, 295 s/episode) and 55509 (CPU, 275 s/episode) OK. Full: cube-single **55512** (array 0-9%8,
5 tasks x 2 parts x 30 train episodes 1000-1059), cube-double **55513** (array 0-9%7, 5 x 2 x 20, episodes 1000-1039).
Next: ogb_encode.py → ogb_cta_train.py (CTA, ENDPOINT, FRAME, DIRECT, FULL/CODE) → ogb_cta_eval.py; DISTILL policy.

Latest action (2026-09-28, OGBench design-faithful setup): user decision — proposal policy = a goal-conditioned
action-chunk policy (no released one exists: OGBench baselines are per-step and unreleased; the only third-party chunk
policies are state-based single-task). HIQL cube-double 55379 CANCELLED (not needed). `scripts/ogbench/gcfbc.py`:
OGBench visual GCBC recipe (IMPALA-small early fusion, crop aug, trajgoal relabeling, batch 256, 500k) + flow-matching
chunk head (H 5, 10 flow steps, MLP 4x512, as Q-chunking/FQL). `ti_wm/ogb_runtime.py`: exact MuJoCo restore (checked),
physics-only branch labels (cube-target distance). Smoke 55409/55411 OK (2k-step policy: P0 0/5, ORACLE8 1/5).
Full runs: cube-single **55415** train → **55416** eval (P0, ORACLE8, ORACLE16; 20 episodes x 5 tasks);
cube-double **55417** train → **55418** eval.

Latest action (2026-09-28, GPU saving): no released visual HIQL checkpoints exist (HF search; OGBench README), so
training is required. Timing of 55328: 100k training steps ≈ 1.1 h, each 250-episode evaluation ≈ 1.3 h. 55328
cancelled at ~300k (checkpoints 100k/200k kept; eval .78/.848, enough to validate the setup and to serve as the
cube-single proposal policy). 55378 cancelled during its step-1 eval; resubmitted as **55379** with one final evaluation
(eval_interval 500k), 10 h limit.

Latest action (2026-09-28, cube-double): user asked why arenas with strong baselines; added OGBench
visual-cube-double-play (HIQL paper 39%, more headroom). **55377** setup (dataset from HF mirror) → **55378** HIQL
official command (same hyperparameters as cube-single), mig, 30 h. Perturbed deployment banks marked out-of-design.

Latest action (2026-09-28, penalty test read): 55330/55331 COMPLETED, 50 dev roots 2100-2149: P0 31,
CTA4~0.04 25, CTA4~0.08 21, CTA4~0.16 28 (all below P0; CTA4~0.08-P0 -0.20 [-0.38,-0.02]). Override fraction
.90-.94 at every λ. The offline trade-off did not transfer. No PushT learned selector with a 16-bank beats P0 so far;
CTA4@8 (R5) 32 ≈ P0. HIQL 55328: eval success .78 at 100k, .848 at 200k (paper final .89).

Previous action (2026-09-27, Round 6 read + penalty test): 55304 r6train COMPLETED; offline on dev 16-banks Round 6
≈ Round 4 (retained .78 vs .80; worse-than-default 2.7% vs 2.4%), so 55305/55306/55307 were CANCELLED before results
(55305_0 ran 1 min). Round-5 logs show large losses when choosing perturbed copies (p90 .023, max .08). Submitted the
deviation-penalized selection test ([protocol](docs/CTA_ROUND6_PROTOCOL.md)), release
`releases/cta_r5pen_20260927_499b75f4`: **55330** r5pen (array 0-1, roots 2100-2149, arms P0, CTA4~0.04/0.08/0.16)
→ **55331** r5pen_agg. OGBench: smokes 55316/55320 FAILED (jax CUDA path; wandb Settings), 55324 COMPLETED;
HIQL 55325 cancelled after 1 min (was syncing to W&B online), **55328** HIQL running with W&B forced offline
(`scripts/ogbench/run_offline.py`).

Previous action (2026-09-27, OGBench start): [plan](docs/CTA_OGBENCH_PLAN.md), proposal option (a) = HIQL subgoals
as candidates (user decision). **55315** setup (main CPU, 2 h: venv, ogbench@1d41409, visual-cube-single-play from the
HF mirror) → smoke 55316 FAILED (jax CUDA path; fixed with CUDA_ROOT), resubmitted **55320** smoke → **55321** HIQL official command seed 0 (mig, 24 h). DINO-WM repro 55279
TIMEOUT at 3 h after MPC iteration 1: success 0.92 on 50 pairs (paper 0.90); timing is dominated by the config's
per-CEM-step simulator evaluation, not usable as a planning-cost number.

Previous action (2026-09-27, Round 6 submitted): [protocol](docs/CTA_ROUND6_PROTOCOL.md). Paired 16-bank training
(rank16 + anchor-to-default losses); stage A code reader, stage B CTA WM + direct; dev 16-banks = collection roots
32200-32249. Release `releases/cta_r6_20260927_90223949`: **55303** r6smoke → **55304** r6train (3 h, 96 GB) →
**55305** r6closed (array 0-7%2, 200 dev roots) → **55306** r6_agg (CPU) and **55307** r6demo (videos, P0 vs CTA6).

Previous action (2026-09-27, Round 5 read): 55280 smoke OK; 55281_0 COMPLETED (roots 2100-2124), 55281_1 cancelled at
52 min with P0/GEOM8/GEOM16/CTA4/CTA4@8/DIRECT4 logged (2125-2149); shards 2-7, 55282, 55283 CANCELLED because the
pattern was already clear. 50 roots: P0 31, GEOM8 36, **GEOM16 43**, CTA4 19, CTA4@8 32, DIRECT4 7; 25 roots: NLL4 8,
FRAME8 3. The 16-bank raises oracle headroom a lot, but every learned scorer choosing among 16 collapses. Offline on
the logged states, even the reader on the ACTUAL future (FULL) picks a candidate >0.005 worse than candidate 0 in
5.9% of decisions (CTA4 8.8%, DIRECT4 8.3%, FRAME8 15.8%; GEOM16 0%). CTA4 ≈ FULL again, so the bottleneck is the
frozen reader (55018, trained on policy banks only), not the WM. DINO-WM repro 55279 reaches 0.88-0.90 (paper 0.90).

Previous action (2026-09-27, Round 5 PushT): deployment bank = 8 policy samples + their 8 perturbed copies
(`ti_wm.cta_batch.mixed_bank`, same construction as the Round-4 perturbed training banks); same Round-4 nets (55149),
no retraining. Arms P0, GEOM8 (oracle over the policy half), GEOM16 (oracle over all 16), CTA4, CTA4@8 (policy half),
DIRECT4, NLL4, FRAME8 (Round-3 endpoint-latent WM read by the FULL reader); 200 dev roots 2100-2299. Demo videos and
stills via `scripts/cta_demo.py`. Release `releases/cta_r5_20260927_39bfb181`: **55280** smoke (unit tests, tiny closed
loop, aggregate, demo) → **55281** r5closed (array 0-7%2, 2.5 h each) → **55282** r5_agg (CPU) and **55283** demo.

Previous action (2026-09-27, results): **PushT Round 4 closed loop (55169/55170, 100 dev roots)**: P0 64, PHYS8 64,
GEOM8 73 (+9 [-1,+19]), CTA4 68 (+4 [-7,+15], p=.61), CTA4S 68, NLL4 64, DIRECT4 60; CTA4-DIRECT4 +8 [-3,+20]
(p=.24); score contrast CTA4-P0 +.018 vs GEOM8-P0 +.016. Offline geometry retention on visited states CTA4 .62-.67
≈ FULL .58-.64 > DIRECT4 .41-.52. No contrast significant at n=100. **LIBERO L4 aggregate 55276** (55232 failed:
int64 JSON): held-out continuation-select gain -.013 [-.05,+.025], progress-select -.013; not expanded.
DINO-WM repro attempts 55259/55273/55274/55275/55277 failed on environment setup (time, gym, Hydra launcher,
accelerate, skimage); full import scan done, **55278** submitted.

Previous action (2026-09-27, newest-1): main end-to-end test = CTA inside the DINO-WM PushT planner
([protocol](docs/CTA_DINOWM_PLANNING_PROTOCOL.md)). **55259** (mig 3g.40gb, 8 CPU, 64 GB, 3 h): reproduce the
official DINO-WM PushT planning with timing. LIBERO L4 not expanded (interim reading in the continuation protocol).
PushT Round-4 closed loop 55169_0 RUNNING (was blocked by the per-user CPU limit while L4 ran).

Previous action (2026-09-27, latest-1): benchmark decision replaced by
[docs/CTA_LIBERO_CONTINUATION_PROTOCOL.md](docs/CTA_LIBERO_CONTINUATION_PROTOCOL.md): LIBERO-Goal with a
continuation-success label is the main task for the trajectory claim; PushT stays as development task.
RECON/NWM/CompACT and the LeWM suite are dropped (their scripts and the old benchmark plan were deleted; 55171/55172/
55212/55224 fetched data only and are not used). Submitted **55231** (L4 collect, main, array 0–19%10, 4 CPU, 24 GB,
12 h) and **55232** (aggregate, afterany). PushT Round-4 chain 55149 → 55169 → 55170 continues.

Previous action (2026-09-27, later): user asked to stop circling PushT and add benchmarks.
Diagnostic 55138: shard 0 COMPLETED (25 roots: P0 17, PHYS8 19, GEOM8 18, FULL 16, CTA3 18, NLL8 20,
FRAME8 17, DIRECT3 18); shard 1 FAILED (consistency check too strict on near-tied banks: CTA8E median
rel .002 but rho .66 — fixed to use clear banks only); shards 2–3 and 55139 CANCELLED to free GPU.
PushT collection 55147 throttled to %1. Round-4 closed loop 55150/55151 cancelled before start and
replaced by **55169 (%1, adds PHYS8/GEOM8 arms) / 55170** from release `cta_plus_20260927_av0Ksz`.
New LeWM benchmark suite (dropped later the same day; plan deleted): **55171** fetch (main CPU,
4 CPU, 16 GB, 4 h) → **55172_0–3** official LeWM eval tworoom/pusht/reacher/cube, seeds 42–44
(mig, 8 CPU, 64 GB, 3 h, %1). Network probe 55164: compute nodes reach HF/GitHub, not Google Drive
or the CALVIN server.

Earlier action (2026-09-27): on-policy diagnostic (1) + round-4 data/training (2)
**submitted as chain 55137–55144**; see the section below. Round 3 (55077–55079) and
reader-only (55066–55068) are COMPLETED (both queue and accounting checked).

## On-policy diagnostic (1) and round-4 data (2) — 2026-09-27

Protocol: [docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md](docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md).
Immutable release `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_plus_20260927_Gc0So3`
(hashes in `cta_plus_20260927_Gc0So3.SHA256SUMS`, job list in `.JOBS`). Queue was empty before
submission. GPU use before this chain: 51.4 MIG GPU-h since 09-22; estimate for the chain ~11–12.

| Job | Work | Resources | State at submission |
|---|---|---|---|
| 55137 | Smoke: unit tests + every new path on 2 roots x 3 decisions (diag, collect, encode, train 20 updates, r4 closed, both aggregates) | mig 3g.40gb, 8 CPU, 64 GB, 1 h | RUNNING |
| 55138_0–3 | (1) lockstep closed loop, roots 2100+25*task, arms P0/PHYS8/GEOM8/FULL/CTA3/NLL8/FRAME8/DIRECT3, all candidates simulated and cross-scored; batched-vs-sequential scorer check | mig, 8 CPU, 96 GB, 2 h, %1 | afterok 55137 |
| 55139 | (1) aggregate: success/score contrasts, on-policy retention matrix (geometry + coverage) | main, 4 CPU, 16 GB, 1 h | afterok 55138 |
| 55140_0–23 | (2) collection, roots 31050–32249 (50/task): standard + perturbed bank, odd roots oracle-mixed execution | mig, 4 CPU, 64 GB, 1 h, %2 | afterok 55137 |
| 55141 | (2) compact features (original PCA) + frozen source codes, old + new | mig, 8 CPU, 96 GB, 2 h | afterok 55140 |
| 55142 | (2) round-4 training NLL4/CTA4/CTA4S/DIRECT4, 8000 updates, per-net dev-objective selection, offline ladder | mig, 8 CPU, 96 GB, 3 h | afterok 55141 |
| 55143_0–3 | (2) closed loop P0/CTA4/NLL4/DIRECT4/CTA4S with logging, roots 2100–2199 | mig, 8 CPU, 96 GB, 2 h, %2 | afterok 55142 |
| 55144 | (2) aggregate merged with 55138 when P0 trajectories match | main, 4 CPU, 16 GB, 1 h | afterok 55143, 55138 |

Outputs: `cta_plus_<mode>_<jobid>/` under the trajectory_innovation data root; logs
`logs/cta_plus_ti_cta_plus_<mode>_*.out`. No result is claimed at submission.

Update (same day): smoke 55137 **COMPLETED, SMOKE_OK** in 2m17s — 30 + 41 unit tests pass;
batched scorers reproduce the sequential Planner/R3Planner paths on live states (median
|Δ|/std ≤ .003, within-bank ρ ≥ .997); the P0 trajectories of two separate lockstep runs are
identical (cross-run pairing exact). Round-4 smoke timing: ~0.89 s/update.
Before 55137 finished, 55140–55144 (all PENDING, zero elapsed, verified with sacct) were
**cancelled** to add per-step physical states + contact counts of every candidate to the new
data (`run_segment_states`, tested against `run_segment`), so path-dependent query labels can be
computed later without re-collection. 55138/55139 (diagnostic) kept. Replacement chain from
release `cta_plus_20260927_OOAdzO`:

| Job | Work | Resources | Dependency |
|---|---|---|---|
| 55146 | Smoke of the new release (same modes as 55137) | mig, 8 CPU, 64 GB, 45 min | — |
| 55147_0–23 | Collection roots 31050–32249, now also `step_states` (8 x 10), `step_contacts`, `steps_executed` | mig, 4 CPU, 64 GB, 1 h, %2 | afterok 55146 |
| 55148 | Encode | mig, 8 CPU, 96 GB, 2 h | afterok 55147 |
| 55149 | Round-4 training | mig, 8 CPU, 96 GB, **4 h** (0.89 s/update measured) | afterok 55148 |
| 55150_0–3 | Round-4 closed loop | mig, 8 CPU, 96 GB, 2 h, %2 | afterok 55149 |
| 55151 | Aggregate merged with diagnostic 55138 | main, 4 CPU, 16 GB, 1 h | afterok 55150, 55138 |

Previous action (2026-09-26): Round-3 budget replacement **55077 / 55078 / 55079
submitted**. Old 55072/55073/55074 verified CANCELLED at zero elapsed time.
55067 array throttle is now 1; completed shards and active shards are preserved.
Before replacement submission, squeue/sacct showed reader shards 0/1 completed,
2/3 running, remaining pending; 55068 waits for aggregation. Older rows
retain their submission-time states and are not a live queue snapshot.
Both squeue and sacct must be checked before acting on job state.

## Round-3 budget and scheduling amendment — 2026-09-26

User explicitly requested cancellation/resubmission before training, 6000 steps x
32 banks, dev curves and timing, and reduced reader-only concurrency. Verified
old 55072/55073/55074 PENDING with both queue/accounting, then cancelled all three;
sacct records zero elapsed time for all. `scontrol` reported completed array members
on the throttle command but did apply it: `scontrol show job` confirms
**ArrayTaskThrottle=1**, and squeue displays `55067_[4-9%1]`. No active shard killed.

New budget: 6000 optimizer updates x 32 effective banks = 192,000 bank exposures
per network. Four microbatches of 8 preserve GPU memory. Ranking uses a global
eligible-pair denominator; clipping and optimizer update occur once per full batch.
Added a gradient-equivalence regression test. Same four models/targets/arms as
the original Round-3 protocol; training is not claimed converged by budget alone.

Every 1000 updates: full dev ladder, coordinate NLL and normalized frame/proprio
errors; `dev_curve.json` and intermediate `dev_step_*/` reports. No metric gate or
best-checkpoint selection. Log per-update/per-network wall time and estimated
remaining training time; the proposed 0.5 s/update estimate remains unverified.
Checkpoint networks, optimizers and RNG every 250 updates, with early Slurm warning
forwarded to Python for a final safe checkpoint. Resume into a fresh run if needed;
partial training cannot silently proceed to closed-loop evaluation.

| Job | Work | Resources / cap | Submission |
|---|---|---|---|
| 55077 | Tests, larger matched training, six dev evaluations, runtime checks | mig, 1 H100 3g.40gb, 8 CPU, 96 GB, 3 h | Submitted |
| 55078_0–9 | Same eight arms on 100 dev roots | mig, 1 H100 3g.40gb, 4 CPU, 48 GB, 75 min each, concurrency 2 | afterok:55077 |
| 55079 | Paired score/success/latency aggregation | main, 2 CPU, 8 GB, 10 min | afterok:55078 |

Allocation ceiling remains 15.5 MIG GPU-hours. No duplicate Round-3 work was queued.
Syntax/shell/whitespace checks passed; runtime regression tests run inside 55077,
not claimed passed at submission. The reader-only 55067/55068 chain remains intact.

Release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_parallel_budget_20260926_b2JpqH`.
Sibling `.SHA256SUMS` and `SUBMITTED_JOBS` preserve provenance. Expected outputs:
`cta_parallel_train_55077/`, `cta_parallel_closed_55078/shard_*/`, and
`cta_parallel_aggregate_55079/comparison.json` under the shared results root.
Logs: `logs/cta_parallel_r3_55077_*.out`, `logs/cta_parallel_r3_55078_*.out`,
`logs/cta_parallel_aggregate_55079.out`. The original cancelled release is preserved.

## Parallel WM Round 3 submitted — 2026-09-26

**Superseded before execution:** 55072–55074 cancelled at zero elapsed time;
the budget-amended replacement is 55077–55079 above. Historical submission follows.

User's revised plan targets WM structure and learning objective. Protocol:
`docs/CTA_PARALLEL_ROUND3_PROTOCOL.md`; historical round-cap amendment recorded
in `docs/CTA_E2E_PROTOCOL.md`, motivation in `docs/CTA_DEBUG_LOG.md` before submission.
Source encoder and reader frozen; NLL-only parallel FSQ predictor versus identically
initialized task-trained predictor. Matched weighted DIRECT continuation and endpoint
frame/proprio WM included. Four networks, same 4000 batch updates each. No M/K sweep.

| Job | Work | Resources / cap | Submission |
|---|---|---|---|
| 55072 | Unit suites, four-network training, offline ladder with center/vertex strata, runtime checks | mig, 1 H100 3g.40gb, 8 CPU, 96 GB, 3 h | Submitted |
| 55073_0–9 | P0/CTA3/NLL8/FRAME8/DIRECT3/old AR greedy/old AR expected/old DIRECT; 100 roots 2100–2199 | mig, 1 H100 3g.40gb, 4 CPU, 48 GB, 75 min each, concurrency 2 | afterok:55072 |
| 55074 | Paired normalized score, success, baselines, latency | main, 2 CPU, 8 GB, 10 min, no GPU | afterok:55073 |

Primary: CTA3-P0 mean maximum native reward, excluding reset. Success remains
secondary. Reader adaptation keeps its original success/greedy primary. New controls
re-evaluate old AR/P0 under the new no-reset score contract; this is a new WM experiment,
not a duplicate reader-adaptation array. All selectors simulate only the chosen branch.
Old queued jobs use immutable snapshots and were not modified/cancelled.

Allocation ceiling 15.5 MIG GPU-hours, no foreground polling. Invalid dependencies
auto-cancel. Syntax/shell/whitespace checks passed before submission; numerical tests
(including frozen-reader gradient flow and reset exclusion) run in 55072 and are
not yet claimed passed. No offline scientific-score threshold gates closed loop.

Release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_parallel_round3_20260926_GSBjQt`.
Source hashes in sibling `.SHA256SUMS`, submission IDs in `SUBMITTED_JOBS`.
Expected artifacts under shared trajectory_innovation results root:

- `cta_parallel_train_55072/{round3.pt,train_report.json,metrics.jsonl,offline.json,dev_scores.npz}`.
- `cta_parallel_closed_55073/shard_*/roots_*.jsonl`.
- `cta_parallel_aggregate_55074/comparison.json`.

Logs: `logs/cta_parallel_r3_55072_*.out`, `logs/cta_parallel_r3_55073_*.out`,
`logs/cta_parallel_aggregate_55074.out`. Parent geometry checkpoint unchanged.
No result/advantage claim at submission; sealed roots remain unopened.

## Reader refinement submitted — 2026-09-26

User authorized continued refinement to beat the learned policy baseline.
Protocol: `docs/CTA_READER_REFINEMENT_PROTOCOL.md`. One matched intervention:
freeze geometry source/WM; train source-only control and source/greedy/expected
mixture reader with identical batches, goals, labels and 3000 updates each.
Primary comparison ADAPT_G-P0; expected-code secondary. No oracle selector in
the eight-arm, 100-root dev evaluation; only selected branches are simulated.

| Job | Work | Resource/time limit | Submission |
|---|---|---|---|
| 55066 | Unit suites, frozen code cache, matched reader training, offline scores, runtime checks including expected-code | mig, 1 H100 3g.40gb, 8 CPU, 96 GB, 2 h | Submitted |
| 55067_0–9 | P0/DIRECT/base/control/method, greedy and expected, roots 2100–2199, 10 per shard | mig, 1 H100 3g.40gb, 4 CPU, 48 GB, 1 h each; concurrency 2 | afterok:55066 |
| 55068 | Strict root/checkpoint aggregation, paired success/coverage CIs and McNemar | main, 2 CPU, 8 GB, 10 min, no GPU | afterok:55067 |

Dependencies cancel automatically on failure; scientific metric values do not
gate evaluation. No foreground polling or idle GPU allocation. Allocation ceiling
12 MIG GPU-hours, actual expected lower. Queue/accounting checked before submission,
no duplicate work present. Syntax and whitespace checks passed; new numerical tests
are delegated to 55066 and are not claimed passed at submission.

Immutable release:
`/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_reader_refine_20260926_YmuOTQ`.
Sibling `.SHA256SUMS` and `SUBMITTED_JOBS` record source and submission provenance.
Each batch snapshots source again. Expected artifacts under the shared results root:

- `cta_reader_train_55066/train_report.json`, `train_codes.npz`, `metrics.jsonl`,
  `control/` and `method/` checkpoints (every 1000 updates, final at 3000), dev scores.
- `cta_reader_closed_55067/shard_*/roots_*.jsonl`, eight-arm complete root records.
- `cta_reader_aggregate_55068/comparison.json`.

Logs: `logs/cta_reader_refine_55066_*.out`, `logs/cta_reader_refine_55067_*.out`,
`logs/cta_reader_aggregate_55068.out`. Original geometry checkpoint is preserved.
These are submitted work/output destinations, not improvement or completion claims.

## Geometry closed-loop result 55052 — 2026-09-26

Tests passed (30 + 25, including reporting regression); existing four-tier runtime
preflight passed. No training was repeated. Ten dev roots 2100–2109, eight arms:
P0 7/10, PHYS8 4/10, GEOM8 8/10, FULL8 5/10, CODE8 5/10, CTA8 6/10,
CTA8E 4/10, DIRECT8 4/10. All artifacts and aggregation completed.
GEOM8 adds one P0 failure and loses none; CTA8 adds one and loses two.
No comparison is statistically confirmed. Expected-code's offline advantage
did not translate into a higher success count here. Learned future readers also
underperform P0; do not attribute all closed-loop losses to the WM.

Full report and next intervention rationale: `docs/CTA_GEOMETRY_CLOSED_RESULT_55052_VI.md`.
Recommended next: freeze source/WM, adapt the reader to deployed greedy/expected
codes with a matched source-code-only continuation control, then paired closed-loop
evaluation. This is a recommendation, not a submitted training job. No new job
was submitted during this result-reporting turn.

## Geometry recovery — 2026-09-26

Both squeue (empty) and sacct verified **55018 FAILED, exit 1:0, 40m27s**.
All 3000 source/baseline and 6000 WM updates finished; `cta.pt`, `dev_scores.npz`
and the final step-9000 metrics survived. The new native-coverage diagnostic kept
the NumPy `chosen` array from `ranking_metrics`; `json.dumps(default=float)` then
failed in training-report finalization. No preflight or closed-loop episode ran.
The job passed its two unit suites (30 + 24 tests) before training.

Fix: omit per-row `chosen` from the JSON summary (as the existing ladder does),
always finalize the logger, and add recovery from completed saved scores/checkpoint.
A regression test covers serialization with multiple decisions. Recovery verifies
the final update marker and score/feature identity, reconstructs summaries and CIs,
then runs the original preflight, eight-arm ten-root closed loop and aggregation.
Original artifacts stay unchanged; no optimizer/model updates or relabeling repeat.

**55052 submitted**: `ti_cta_geometry_resume`, one MIG H100 3g.40gb, 8 CPU,
128 GB, **2-hour hard cap**, no array/dependencies. No duplicate job was queued
before submission; accounting for 55018 was checked again. Syntax and diff checks
passed locally; the new runtime regression test is delegated to this batch job.

Release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_geometry_resume_20260926_LVSxNq`.
Output: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_geometry_e2e_55052`.
Log: `/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_geometry_e2e_55052.out`.
Report: `docs/CTA_GEOMETRY_RESULT_55018_VI.md`. Offline point estimates do not
establish a closed-loop improvement; this evaluation still uses only ten dev roots.

## PushT geometry end-to-end — 2026-09-26

**55018 submitted**, `ti_cta_geometry_e2e`: one MIG H100 3g.40gb, 8 CPU, 128 GB,
hard cap **3 hours**, no array and no dependent research gates. User reiterated
end-to-end implementation/debugging. Protocol: `docs/CTA_GEOMETRY_E2E_20260926_VI.md`.
Technical tests -> existing-data registration labels -> Round-1 warm-start codec,
reader and matched FULL/DIRECT updates (3000) -> frozen-source WM/prior (6000) ->
offline ladder -> runtime consistency -> eight-arm full-episode closed loop on
ten development roots 2100–2109 -> aggregation. No sealed roots, new collection,
co-design, or new arena. A low scientific metric does not cancel later stages.

squeue was empty and sacct checked before submission; no duplicate work. Syntax
and diff checks passed. Numerical/native-geometry tests run inside the job and
are not claimed passed at submission. The preflight now reads its 32-bank NPZ
prefix once instead of repeatedly decompressing whole image arrays; same thresholds.

Release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_geometry_e2e_20260926_rnIPxk`.
Output: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_geometry_e2e_55018`.
Log: `/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_geometry_e2e_55018.out`.
`e2e_report.json` records current stage; partial root records and checkpoints are
preserved. Ten-root outcomes are development diagnostics, not confirmation.

## Native task premise audit — 2026-09-26

Result update: **55016 COMPLETED, 1 s, exit 0:0**, verified with both squeue and
sacct. No active jobs remained at that check. Native semantics examples passed;
this is not visual/policy evidence. Scrub contact-count spread is nonzero in 5/8
and 2/7 complete banks; immediate released-success ties in all 15. Details in
`docs/CTA_GEOMETRY_E2E_20260926_VI.md`; original submission record follows.

Job **55016**, `ti_cta_task_premise`: submitted to main, **1 CPU, 2 GB, 5 min,
no GPU**, no dependencies or automatic follow-ups. Before submission squeue was
empty and sacct confirmed previous jobs completed/cancelled/timed out; no duplicate
audit was active. Python compilation, shell syntax and diff whitespace checks passed.
Runtime assertions are delegated to the job; submission is not a test result.

Protocol: `docs/CTA_TASK_QUALIFICATION_20260926_VI.md`. Executes native RinseBowls
methods on fixed mock predicate traces (semantics only) and audits exactly the H8
intervention labels from existing compiled-restore Scrub banks 52419/52524.
No simulation, rendering, encoding, model loading/training, or new continuation.

Immutable release:
`/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_task_premise_20260926_TQo1fp`.
Expected report: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_task_premise_55016/report.json`.
Log: `/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_task_premise_55016.out`.
The release snapshots the audit, config, protocol, Slurm script and native task source
with SHA256SUMS. RinseBowls is a qualification priority, not a training-gate pass.

## Scope comparison stopped for research relevance — 2026-09-26

User challenged further PushT spending because the endpoint-coverage target
does not test the central trajectory claim. Stopped 54989 and 54990, preserving
all artifacts. Verified with both squeue and sacct: no active jobs; 54989_0/_1
CANCELLED after 9m50s each, _2/_3 cancelled before execution, 54990 cancelled
before execution. CPU check 54988 completed in 37s. No final codec comparison
exists; this is not evidence of endpoint/trajectory equivalence or failure.

No new jobs submitted. See docs/CTA_RESEARCH_RESET_20260926_VI.md for the
corrected claim/evidence map and premise checks required before more training.

## Endpoint–trajectory scope comparison submitted — 2026-09-26

User approved this offline codec comparison. Protocol locked in
docs/CTA_ENDPOINT_TRAJECTORY_PROTOCOL.md before submission. squeue was empty;
sacct confirmed the previous chain's completed/timed-out jobs, with no duplicate
scope training. Existing peer/session dirty files were preserved.

| Job | Purpose | Resources / hard limit | Submission state |
|---|---|---|---|
| 54988 | Unit suites plus four tiny real-cache codec runs and paired aggregation | main, 4 CPU, 24 GB, 30 min, no GPU | Submitted |
| 54989_0 | Endpoint, seed 0, 10k updates | mig, 1x 3g.40gb, 8 CPU, 64 GB, 90 min | afterok:54988 |
| 54989_1 | Trajectory, seed 0, same updates/data/bit budget | same | afterok:54988 |
| 54989_2 | Endpoint, seed 1 | same | afterok:54988 |
| 54989_3 | Trajectory, seed 1 | same | afterok:54988 |
| 54990 | Paired root-bootstrap comparison with initialization/batch/config checks | main, 4 CPU, 24 GB, 15 min, no GPU | afterok:54989 |

Training concurrency 2, total allocation ceiling **6 MIG GPU-hours**. Both
arms start fresh, share common initial weights and sampled batches/goals;
both reconstruct endpoint only and see all training phases. Trajectory has
more input tokens/path-specific parameters, so compute is measured rather
than claimed exactly equal. No WM, closed loop, new collection or sealed roots.

Release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_scope_release_20260926T040509Z/`.
SOURCE_SHA256SUMS and SUBMITTED_JOBS preserve provenance. Python compilation,
shell syntax and whitespace checks passed; numerical tests/smokes are delegated
to 54988, not yet claimed passed. Invalid dependencies cancel automatically.
No foreground polling after submission.

Expected outputs under shared trajectory_innovation results:
`cta_scope_check_54988_0/smoke_comparison.json`,
`cta_scope_train_54989_{0,1,2,3}/{codec.pt,train_report.json,dev_scores.npz}`,
`cta_scope_compare_54990_0/comparison.json`. A null interval is not interpreted
as equivalence; the result concerns endpoint ranking, not temporal-query sufficiency.

## Latest verified outcome — 2026-09-26

squeue is empty; sacct confirms 54980 COMPLETED 0:0 (7s), 54982 COMPLETED
0:0 (1m06s), both 54981 preflight tasks TIMEOUT (20m29s each), and 54983/54984
CANCELLED before execution due to dependencies. No Round-2 closed-loop results.
Precision regression passed, but GPU checkpoint preflight has no completed
verdict. Do not interpret timeouts as numerical validation failure or success.

WM audit on 600 decisions / 50 roots: actual code gap .694, free-running WM
.301, free-running prior .000; true-prefix WM .695, true-prefix prior .596,
true-prefix WM with wrong sibling actions .522. Correct-action advantage
over wrong actions +.173 [.081,.294]. True prefixes carry useful future
information themselves, so this does not isolate autoregression as sole cause.

New scope clarification: FULL uses endpoint features only, and cov8 is an
endpoint label. CODE/FULL retention does not demonstrate whole-trajectory
query sufficiency or advantage over a matched endpoint codec. See
docs/CTA_ENDPOINT_SCOPE_AND_STATUS_20260926_VI.md. No new jobs submitted in
this status/claim assessment.

## Runtime precision repair and replacement chain — 2026-09-26

Verified with squeue+sacct: 54975 COMPLETED 0:0 in 7s; 54976_0 FAILED
1:0 at CODE8 preflight (.75 clear-bank argmax agreement vs required .85).
Cancelled 54977/54978 before any rollout, and cancelled old preflight
54976_1 at 11m48s after identifying the obsolete runtime precision path.
Queue was empty before this replacement submission; no duplicate work.

Static bug: Planner.future called DINO/PCA inside scorer bf16 autocast,
whereas cached features are computed float32 then stored fp16. Fixed
Planner.tokens to disable outer autocast during frozen feature extraction.
Regression test added. Same preflight thresholds, model weights and root sets.
See docs/CTA_RUNTIME_PRECISION_FIX_20260926.md. GPU verification pending;
do not claim the fix has fully resolved the mismatch until preflight passes.

| Job | Purpose | Resources / hard limit | Submission state |
|---|---|---|---|
| 54980 | All unit suites including precision regression; CPU metadata audit | main, 4 CPU, 16 GB, 20 min | Submitted |
| 54981_0–1 | Corrected runtime preflight, both checkpoints | mig, 1x 3g.40gb, 4 CPU, 32 GB, 20 min each, concurrency 1 | afterok:54980 |
| 54982 | Frozen WM vs prior vs wrong-action prefix audit, fixed 600-decision dev panel | mig, 1x 3g.40gb, 4 CPU, 32 GB, 45 min | afterok:54980 |
| 54983_0–19 | Original Round-2 closed-loop design, 100 roots per checkpoint, 10 roots/shard, five arms | mig, 1x 3g.40gb, 4 CPU, 32 GB, 90 min each, concurrency 2 | afterok:54981:54982 |
| 54984 | Strict paired aggregation and visited-state audit | main, 4 CPU, 16 GB, 20 min | afterok:54983 |

Release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_precision_release_20260926T024651Z/`;
SOURCE_SHA256SUMS and SUBMITTED_JOBS preserve provenance. Python compilation,
shell syntax and whitespace checks passed; compute-node regression suite is
delegated to 54980. All invalid dependencies cancel automatically. At most
two MIG jobs from this chain run simultaneously; no foreground waiting.

Expected outputs under the shared results root: `cta_r2_audit_54980/audit.json`,
`cta_r2_preflight_54981/`, `cta_r2_wm_audit_54982/report.json`,
`cta_r2_closed_54983/{control,method}/`, `cta_r2_aggregate_54984/comparison.json`.
No Round-3 training or new progress target is submitted.

## Round 2 results and follow-up submission — 2026-09-26

Before submission, squeue showed no active user jobs and sacct confirmed
54932, 54933_0, 54933_1 and 54934 COMPLETED, exit 0:0. Training elapsed
53m13s / 52m57s. No duplicate follow-up was present. Offline co-design
lambda=.1 loses source retention (.458 vs control .706) and collapses code
usage (source perplexity 1.37 vs 131.20). See
docs/CTA_ROUND2_OFFLINE_RESULT_54933.md for intervals and interpretation.

| Job | Purpose | Resources / limit | Submission state |
|---|---|---|---|
| 54975 | Unit suites including partition/pairing tests; train/dev coverage-flat audit and baseline score drift | main, 4 CPU, 16 GB, 20 min, no GPU | Submitted |
| 54976_0–1 | Existing inference preflight for both final checkpoints, no episodes | mig, 1x 3g.40gb each, 4 CPU, 32 GB, 20 min, concurrency 1 | afterok:54975 |
| 54977_0–19 | Both fixed checkpoints, 100 dev roots each (2100–2199), P0/CODE8/CTA8/DIRECT8/CTA8E, 10 roots/shard | mig, 1x 3g.40gb each, 4 CPU, 32 GB, 90 min, concurrency 2 | afterok:54976 |
| 54978 | Strict completeness/P0 checks, within-model and method-control paired CIs; visited-state coverage audit | main, 4 CPU, 16 GB, 20 min, no GPU | afterok:54977 |

Invalid dependencies cancel automatically. Greedy remains primary; CTA8E is
secondary. No sealed roots or new training. Allocation ceiling is 30 MIG
GPU-hours for closed loop plus 40 minutes for preflight; estimated actual
12–15 MIG GPU-hours based on earlier execution. No foreground waiting.

Immutable release:
`/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_r2_followup_release_20260926T022231Z/`.
Submission IDs also persisted in its SUBMITTED_JOBS. Python compilation,
shell syntax and git diff whitespace checks passed before submission;
runtime tests are delegated to 54975 and are not yet claimed passed.

Expected outputs: `cta_r2_audit_54975/audit.json`,
`cta_r2_preflight_54976/{control,method}/`,
`cta_r2_closed_54977/{control,method}/roots_*.jsonl`, and
`cta_r2_aggregate_54978/comparison.json`, under the shared trajectory_innovation
results directory. These are destinations, not completed-result claims.

## Round 2 submission — 2026-09-25

Protocol: docs/CTA_ROUND2_PROTOCOL.md. Both queue and accounting were empty
for recent active work immediately before submission; no duplicate CTA job found.
Python compile, shell syntax and git diff whitespace checks passed on login.
Runtime tests are delegated to job 54932; their success is not yet claimed.

| Job | Purpose | Resources | Submission state |
|---|---|---|---|
| 54932 | Existing unit suites + new NLL/gradient/paired-statistics tests; tiny real-checkpoint continuation at lambda 0 and .1 | main, 4 CPU, 48 GB, 00:30:00, no GPU | Submitted |
| 54933_0 | Matched Round-1 continuation, lambda=0 | mig, 1x 3g.40gb, 8 CPU, 128 GB, 03:00:00 | afterok:54932 |
| 54933_1 | CTA co-design, lambda=.1 | same; array concurrency 1 | afterok:54932 |
| 54934 | Root-paired offline comparison of the two final checkpoints | main, 2 CPU, 8 GB, 00:15:00, no GPU | afterok:54933 |

Invalid dependencies are automatically cancelled. No closed-loop array or
sealed-root evaluation is submitted. Training ceiling: 6 MIG GPU-h total.

Immutable source release:
`/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_r2_release_20260925T174903Z/`
(manifest `SOURCE_SHA256SUMS`). Each job also snapshots this release and hashes
its own source copy. Parent checkpoint: `cta_train_54717/cta.pt`.

Expected outputs under `/mnt/data/nhatnc129/jepa/trajectory_innovation/`:
`cta_r2_54933_0/`, `cta_r2_54933_1/` (checkpoints, train_report.json,
dev_scores.npz, metrics.jsonl, wandb), and
`cta_r2_compare_54934/comparison.json`. These are output destinations, not
claims that the jobs have completed. Wandb training group: `r2_codesign`.

## Historical jobs

| Job | Purpose | Resources | State |
|---|---|---|---|
| 53776 | Pinned-source/checkpoint preparation, five contract tests, native environment replay checks | main, 4 CPU, 16 GB, 00:45:00, no GPU | COMPLETED 0:0 in 2m24s: `READY_FOR_GPU_SMOKE_NOT_QUALIFIED`, ckpt `84a7c23`, reset replay exact 4/4 (reset determinism only, not branching) |
| 53787 | Gate A-C smoke: strict load, bitwise select_action equivalence, nested banks, clone exactness, timing, dev goal map (roots 5000-5031) | main, 1 GPU, 4 CPU, 32 GB, 00:40:00 | FAILED 1:0 at 20m51s (by design: assertion). Strict load OK; official equivalence bitwise 0.0 on 3 seeds; nested K8/K32 <= 0.008 px. BUG: plain deepcopy clone deviated up to 112 units from live continuation, so dev P0 went 0/32 (clone artifact). Led to protocol Amendment 1 |
| 53803 | Gate A-C smoke rerun under Amendment 1 (clone must track the live continuation) | main, 1 GPU, 4 CPU, 32 GB, 00:40:00 | COMPLETED 0:0 in 16m21s, SMOKE_PASS. clone=`deepcopy_fixed` (repeat exact, live deviation 0.002); dev P0 20/32; goal map from 16 dev successes; ~1 s per decision |
| 53806 | Gate A-C main: 4 shards x 25 qualification roots 1000-1099, arms OFFICIAL/P0/PHYS8/VIS8/MEDOID8 + H_rep | main, array 0-3, 1 GPU each, 02:00:00 | COMPLETED 0:0, all 4 shards (~44 min each), 100/100 roots |
| 53811 | Gate A-C aggregation (CPU) | main, 2 CPU, 8 GB, 00:15:00 | COMPLETED. **A PASS, B PASS (+15 pp [5,25]), C FAIL (VIS8 +3 pp, retention 0.20)**. See docs/GATE_ABC_RESULT_53806.md |
| 53938 | Gate C2 stage 1: collect P0 rollouts (train roots 20000-20799, val 20800-20999), DINOv2 encode, train hindsight progress reader + goal-only control | main, 1 GPU, 4 CPU, 64 GB, 01:45:00 | COMPLETED 00:42:39; TRAINED, val Spearman full 0.854 vs goal-only 0.169 |
| 53992 | Gate C2 stage 2: closed-loop P0/PHYS8/PROG8 on roots 1200-1299 (array 0-1, 50 roots each), reader from c2_train_53938 | main, 1 GPU, 4 CPU, 32 GB, 02:00:00 per task | COMPLETED 00:41:50 / 00:43:02, 100 roots |
| 54139 | Gate C2 aggregation (CPU): B-replication + C2 retention verdict, diagnostics | main, CPU, 2 CPU, 8 GB, 00:15:00 | COMPLETED. B EXTEND (PHYS8-P0 +10 pp [-1,22]) so C2 NOT_INTERPRETED; PROG8-P0 +14 pp [3,25], McNemar p=0.020. See docs/GATE_C2_RESULT_53992.md |
| 54144 | Gate C2 stage 3: CPU aggregation of c2_eval_53992 (pre-registered B-replication + C2 retention verdict) | main, CPU, 2 CPU, 8 GB, 00:15:00 | COMPLETED 6 s (peer-session duplicate of 54139, output c2_agg_53992; summary identical to 54139) |
| 54268 | Gate C2-confirm: train reader r1 (--seed 1 --no-control), same data/hyperparameters as 53938 | main, 1 GPU, 4 CPU, 64 GB, 01:45:00 | COMPLETED 00:29:08; val Spearman 0.856 |
| 54269 | Gate C2-confirm closed loop: P0 / PROG8_r0 / PROG8_r1 on fresh roots 1300-1499 (array 0-3 x 50), afterok:54268 | main, 1 GPU each, 4 CPU, 32 GB, 01:00:00 | COMPLETED, 4 shards ~38-40 min each |
| 54270 | Gate C2-confirm aggregation (CPU), afterok:54269. Protocol docs/GATE_C2_CONFIRM_PROTOCOL.md | main, 2 CPU, 8 GB, 00:15:00 | COMPLETED. **NOT_CONFIRMED**: PROG8_r0-P0 +4.5 pp [-4,13]; r1 +2.5 [-5.5,10.5]. See docs/GATE_C2_CONFIRM_RESULT_54269.md |
| 54413 | S1 collection smoke (2 roots) on main | main, 1 GPU | CANCELLED while pending: main GPU node fully allocated; moved to MIG |
| 54414 | LIBERO L0 (CPU): pinned venv lerobot 0.6.1 [libero,smolvla] + mujoco 3.3.7, SmolVLA LIBERO ckpt download, OSMesa render probe. Protocol docs/LIBERO_QUALIFICATION_PROTOCOL.md | main, 16 CPU, 48 GB, 01:00:00, no GPU | FAILED at probe: install OK (lerobot 0.6.1, hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.3.7, torch 2.11.0); ckpts pinned smolvla_libero 6721902, SmolVLM2 7b375e1; robosuite hardcoded /tmp/robosuite.log owned by another user -> PermissionError |
| 54423 | LIBERO L0 probe rerun on 54414 venv with robosuite macros_private.py (file logging off, CPU rendering) | main, 4 CPU, 16 GB, 00:30:00, no GPU | COMPLETED 3m16s: 10/10 tasks render 2x256x256, 50 init states each, 160-166 ms/step rendered. PASS after Amendment 1 (speed line replaced by budget check) |
| 54415 | S1 collection smoke via --wrap | mig | FAILED 1 s: /bin/sh has no pipefail (wrapper bug, no work done) |
| 54416 | S1 collection smoke (roots 39990-39991). Protocol docs/SIBLING_READER_OFFLINE_PROTOCOL.md | mig, 1x 3g.40gb, 4 CPU, 16 GB, 00:20:00 | COMPLETED 2m: 76 decisions, all fields present, 42/76 decisions with cov8 spread, median 8 distinct sibling states |
| 54418 | S1 collection: 7 x 50 roots (train 30000-30249, held-out 1500-1599), 8 siblings + 2-chunk continuations per decision | mig, array 0-6, 1x 3g.40gb each, 4 CPU, 48 GB, 01:30:00 | COMPLETED, 7/7 shards, 12-13 min each; 7,049 train + 2,778 held-out decisions |
| 54419 | S1 train + offline eval of readers S1-a / S1-b vs C2 r0 and gate-C L2, afterok:54418 | mig, 1x 3g.40gb, 4 CPU, 64 GB, 02:00:00 | COMPLETED 24m. **WEEK1_KILL** (S1-a did not train: loss = ln 8 throughout, held-out top-1 at chance). Diagnostic S1-b: within-bank rho 0.258, retained gap 0.549 (C2 r0 0.14, L2 0.29). See docs/SIBLING_READER_RESULT_54419.md |
| 54417 | Consistency check C2 vs confirm (CPU) | main, 2 CPU, 8 GB, 00:10:00 | FAILED 2:0 at 0 s: script path was node-local /tmp; resubmitted as 54422 |
| 54422 | Consistency check C2 vs confirm (CPU), script in trajectory_innovation/checks/ | main, 2 CPU, 8 GB, 00:10:00 | COMPLETED. Same reader hash, score distribution and selection behaviour; C2 vs confirm diff +9.5 pp CI [-4,+23]. No implementation difference found |
| 54429 | LIBERO L1 seed 0: P0 reproduction via unmodified lerobot-eval (libero_goal 10 tasks x init 0-9, smolvla_libero 6721902, n_action_steps=10). Seeds 1-2 submitted after this one is read | mig, 1x 3g.40gb, 12 CPU, 64 GB, 01:30:00 | CANCELLED at ~40 min: ~9 min/task (4/10 tasks done), would hit the 1:30 limit and lerobot-eval saves only at the end; time limit cannot be raised by user |
| 54432 | LIBERO L1: same unmodified lerobot-eval, split into array 0-5 = 3 seeds x 2 task halves (tasks 0-4 / 5-9), init 0-9 | mig, 1x 3g.40gb, 16 CPU, 64 GB, 01:30:00 each | COMPLETED 6/6 (47-57 min): P0 74/76/64% by seed (mean 71.3%) vs published 91%; 45 always-success, 50 stochastic, 5 systematic roots; 71/86 failures stochastic. See docs/LIBERO_L1_RESULT_54432.md |
| 54431 | S1 signal check (CPU): pixel-NN sibling identification from goal16, frame distinctness, cov8-cov24 rank agreement | main, 2 CPU, 32 GB, 00:20:00 | COMPLETED 12 s: pixel-NN top-1 0.197 (chance 0.125), 8 distinct frames median, rho(cov8,cov24) 0.26 |
| 54455 | W2 closed loop: P0 / PHYS8 / RANK8 (frozen S1-b reader, preflight vs S1 held-out scores) on fresh roots 1600-1999, 8 x 50. Protocol docs/W2_CLOSED_LOOP_PROTOCOL.md; label rule relaxed per RESEARCH_DESIGN 5a (user decision) | mig, array 0-7, 1x 3g.40gb, 4 CPU, 32 GB, 01:15:00 (pending tasks raised to 02:00:00) | Tasks 0-3 COMPLETED (200 roots, ~1 h each). Tasks 4-7 FAILED at 5 s: their start-time code snapshot included the new gate-D test tests/test_codec.py, which had a bug (.grad on a non-leaf tensor), so the unit-test step aborted before any W2 work. W2 code files verified byte-identical to the task-0 snapshot |
| 54456 | W2 aggregation (CPU), afterok:54455 | main, 2 CPU, 8 GB, 00:15:00 | CANCELLED (DependencyNeverSatisfied after tasks 4-7 failed) |
| 54464 | Gate D collection smoke (roots 39992-39993): segments + chunks. Protocol docs/GATE_D_CODEC_PROTOCOL.md (gate D itself runs only if W2 passes) | mig, 1x 3g.40gb, 4 CPU, 16 GB, 00:20:00 | CANCELLED before start (user: wait for running jobs first) |
| 54465 | Gate D train/eval pipeline check (--pipeline-check: smoke shard as both splits, 30 steps; code path only, not a result), afterok:54464 | mig, 1x 3g.40gb, 4 CPU, 96 GB, 02:30:00 | CANCELLED before start (user: wait for running jobs first) |
| 54474 | W2 rerun of tasks 4-7 (roots 1800-1999) after the test fix; same W2 code (verified identical) | mig, array 4-7, 1x 3g.40gb, 4 CPU, 32 GB, 01:15:00 | COMPLETED, 56-59 min each |
| 54475 | W2 aggregation over w2_eval_54455 (tasks 0-3) + w2_eval_54474 (tasks 4-7), afterok:54474 | main, 2 CPU, 8 GB, 00:15:00 | COMPLETED. **W2 PASS**: RANK8-P0 +8.5 pp [2.75,14.25] (n=400), PHYS8-P0 +14.0; retention 0.61 (not STRONG). See docs/W2_CLOSED_LOOP_RESULT_54455_54474.md |
| 54488 | CTA end-to-end CPU smoke (collect 2 roots x 3 decisions -> encode -> train tiny -> closed loop 1 root -> aggregate). Protocol docs/CTA_E2E_PROTOCOL.md (replaces the sequential gate D/E plan, user decision 2026-09-24) | main, 16 CPU, 64 GB, 02:00:00, no GPU | FAILED 8 s at unit tests: `self.type` shadows nn.Module.type in ti_wm/cta.py; renamed to type_embed |
| 54494 | CTA CPU smoke rerun after the fix | main, 16 CPU, 64 GB, 02:00:00, no GPU | FAILED in training: the 3 smoke decisions had no cov8 spread, so the ranking pool was empty. Added a fallback pool and raised the smoke to 8 decisions |
| 54496 | CTA CPU smoke rerun | main, 16 CPU, 64 GB, 02:00:00, no GPU | FAILED at the final ladder: a tier that ties a whole bank left the Spearman set empty. Tied banks now count as rho = 0; they were not excluded |
| 54497 | CTA CPU smoke rerun | main, 16 CPU, 64 GB, 02:00:00, no GPU | COMPLETED, SMOKE_OK. Collect, encode, train (stages 1-3 + adapt) and ladder all ran, then the closed loop with 6 arms and the aggregation. Preflight rel diff ~1e-7; wandb offline runs written. Code path only, not a result. The preflight argmax check now ignores banks tied within 1e-3 sd |
| 54489 | CTA collection: P0 trajectory, all 8 seeded candidates per decision with segments + chunks. Tasks 0-15 train roots 30250-31049, 16-17 dev roots 2000-2099 | mig, array 0-17, 1x 3g.40gb, 4 CPU, 32 GB, 01:00:00 | Submitted 2026-09-24 (queued behind W2 on the 2-GPU cap) |
| 54490 | CTA feature cache: DINOv2 + PCA-128 (fit on train frames), afterok:54489 | mig, 1x 3g.40gb, 8 CPU, 64 GB, 01:00:00 | Submitted |
| 54491 | CTA round 0 training (lambda=0, seed 0) + offline ladder on dev 2000-2099, wandb group r0, afterok:54490 | mig, 1x 3g.40gb, 8 CPU, 128 GB, 03:00:00 | Submitted |
| 54492 | CTA round 0 closed loop, dev roots 2100-2299 (array 0-3 x 50): P0 / FULL8 / CODE8 / CTA8 / DIRECT8, afterok:54491 | mig, array 0-3, 1x 3g.40gb, 4 CPU, 32 GB, 01:45:00 | Submitted |
| 54493 | CTA round 0 aggregation (CPU), afterok:54492 | main, 2 CPU, 8 GB, 00:15:00 | Submitted |
| 54525, 54526 | Network probe from main / mig compute nodes (curl api.wandb.ai) | 1 CPU, 2 min | Both reach api.wandb.ai: wandb can log online from jobs |
| 54541 | LIBERO L2 task 0 (tasks 0-1, init 0-1): branching fidelity + candidate diversity, SmolVLA on CPU. Code ti_wm/libero_runtime.py, scripts/libero/l2_branch_check.py | main, 4 CPU, 24 GB, 01:30:00, no GPU | CANCELLED after the first root. That root: task 0 success in 119 steps; clone EXACT (live and repeat max dqpos 0.0 over 4 checks); chunk relative spread 0.17-0.40. Object spread was 0, because the task-0 target is a drawer fixture that objects_dict does not include. Resubmitted as 54553 with a non-robot qpos spread and an end-effector spread |
| 54553 | LIBERO L2, all 5 array tasks (tasks 0-9 x init 0-1 = 20 roots), with the scene-qpos and end-effector diversity metrics | main, array 0-4, 4 CPU, 24 GB, 01:30:00, no GPU | Submitted 2026-09-24 |
| 54553 (result) | LIBERO L2, 20 roots | — | COMPLETED 5/5. Clone EXACT on every root and check (live and repeat max dqpos 0.0). CPU P0 17/20. Scene-qpos spread > 0.01 in 0-75% of checked decisions depending on the task; median end-effector spread 2-4 cm |
| 54670 | LIBERO L3: closed-loop P0 vs ORACLE8 (BDDL goal progress, ti_wm/goal_progress.py + libero_runtime.goal_progress), 100 roots = tasks 0-9 x init 0-9, CPU | main, array 0-9%5, 4 CPU, 24 GB, 04:00:00 | Running. Partial at 48/100 roots: P0 37/48, ORACLE8 39/48 (not a result) |
| 54668 | CTA round 0 closed loop (resubmission of 54492 with the noise-relative preflight), roots 2100-2299, 5 arms | mig, array 0-3, 1x 3g.40gb, 4 CPU, 32 GB, 01:45:00 | Preflight passed (within-bank rho now vs training: FULL 0.90, CODE 0.86). Measured 3.6 min per root (estimate was 85 s), so tasks 0-1 run into the 1:45 limit at ~29 of 50 roots. Tasks 2-3 CANCELLED: the dev closed loop is cut to roots 2100-2199 on GPU cost, before any result was read |
| 54678 | CPU diagnostic of round-0 FSQ usage (scripts/cta_diag_codes.py) | main, 4 CPU, 24 GB, 00:20:00 | COMPLETED. Saturation confirmed: dims 1-2 sit at one level 99% of the time (median pre-tanh abs 5.5 and 3.6) |
| 54679 | Round-1 CPU smoke | main, 8 CPU | CANCELLED while pending on the per-user CPU cap; resubmitted as 54692 with 4 CPU, 12 GB |
| 54692 | Round-1 CPU smoke: new code paths (--sat, --wm-contrast) on the smoke features | main, 4 CPU, 12 GB, 00:40:00 | Submitted |
| 54680 | CTA round 1 training: --sat 1.0 --wm-contrast 1.0 --contrast-banks 16, seed 0, wandb group r1, afterok:54692 | mig, 1x 3g.40gb, 8 CPU, 128 GB, 03:00:00 | Pending |
| 54681, 54682 | Round-1 closed loop + aggregation in 4 x 50-root shards | — | CANCELLED before start: 50 roots per shard would exceed 1:45. Replaced by 54690/54691 |
| 54688 | Round-0 closed-loop fill: roots 2100-2199 in 10 x 10 shards, skipping roots already in cta_cl_54668 (--skip-done), afterany:54668 | mig, array 0-9, 1x 3g.40gb, 4 CPU, 32 GB, 01:45:00 | Pending |
| 54689 | Round-0 aggregation over cta_cl_54668 + cta_cl_54688, roots 2100-2199 | main, 2 CPU | Pending |
| 54690 | Round-1 closed loop: roots 2100-2199, 10 x 10 shards, arms P0/CODE8/CTA8 (FULL8/DIRECT8 unchanged from round 0), afterok:54680 | mig, array 0-9, 1x 3g.40gb, 4 CPU, 32 GB, 01:45:00 | Pending |
| 54691 | Round-1 aggregation | main, 2 CPU | Pending |

All jobs snapshot source/protocol files and use a job-specific output directory.

53776 output: `/mnt/data/nhatnc129/jepa/trajectory_innovation/prepare_53776/`.
Log: `/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/prepare_53776.out`.
Expected result: `preparation.json`, including resolved HF commit, file hashes,
config compatibility edits, package versions, and exact reset/action replay checks.
This is preparation only: no model loaded, no policy rollout, no measured headroom.
Before submission, py_compile and shell syntax checks passed. Runtime unit tests
are delegated to the CPU job, not claimed passed at submission.

- 2026-09-29 PushT v2 closed loop 55666 + aggregate 55676 (200 dev roots 2200-2399): success P0 122, GEOM8 160, CTA4 142, CTAV2 140, ENDV2 138, DIRV2 135, DINOWM 136. CTA4-P0 +0.10 [.02,.18] p=.024; CTAV2-P0 +0.09 [.01,.17] p=.044; CTAV2-CTA4 -0.01 p=.90; CTAV2-DIRV2 +0.03 p=.63; CTAV2-DINOWM +0.02 p=.72; DINOWM-P0 +0.07 p=.14. Offline geom retention on visited states: FULLV2 .74-.76 > CTAV2 .64-.69 > DIRV2 ~.48 > DINOWM ~.35.
- 2026-09-29 OGBench play-trained 55618 (offline, own/5-goal): CTA .81/.88, DIRECT .79/.90, FRAME .81/.89, ENDPOINT .80/.88, FULL .77/.82, CODE .88/.82; codes 6.3 distinct values/token (no collapse).
- 2026-09-29 cancelled held OGBench v2 variants 55637 (LTO 4,5), 55645 (levels 8,8,8,5,5), 55658 (nested). Submitted **55762** (CPU, `slurm_cta_budget.sh ladder`: paired root-bootstrap CIs of scorer differences on the 55666 logged banks, P0 states and all states) and **55763** (MIG, `slurm_cta_budget.sh budget`: params / FLOPs / ms per decision for CTAV2, ENDV2, DIRV2, CTA4, DINOWM at K 8/16/64, G 1/16). GPU usage before submit 193 h (5th-ranked 486).
- 2026-09-29 results. 55762 paired ladder (P0 states, geometry label, root bootstrap): retained gap CTAV2 .680, ENDV2 .635, DIRV2 .483, DINOWM .334, FULLV2 .756, CODEV2 .628; CTAV2-ENDV2 +.045 [+.011,+.079], CTAV2-DIRV2 +.197 [+.153,+.241], CTAV2-DINOWM +.346 [+.294,+.398]; same signs on all states and on the coverage label. 55763 budget: params CTAV2 10.95 M, ENDV2 11.14 M, DIRV2 6.57 M, DINO-WM 20.12 M (+ own DINOv2); K=8 ms/decision G=1: CTAV2 4.1, ENDV2 4.5, DIRV2 2.5, DINOWM 29.0 (+ shared context 6.5); G=16: 16.9 / 23.4 / 26.8 / 29.1; policy sampling ~770 ms dominates the decision. OGBench v2 closed loop 55675 (100 dev episodes): CTA 15, DIRECT 13, ENDPOINT 12, FRAME 16.
- 2026-09-29 parameter-matched DIRECT: **55840** (MIG, 3 h cap; `slurm_cta_budget.sh direct`: smoke, then `cta_direct_matched.py --layers 14`, same v2 banks / batch stream / loss / selection as 55616 stage 1) -> **55841** (array 0-7%2, P0-only rerun of roots 2200-2399 logging CTAV2, ENDV2, DIRV2, DIRV2L, FULLV2) -> **55842** (CPU `ladder_rerun`: paired CIs, with a check that P0 states and shared scorer scores reproduce 55666). GPU usage before submit 193 h (5th-ranked 486; planned ≤ 9 h at limits).
- 2026-09-29 matched-DIRECT results: 55840 DIRV2L (14 layers, 11.31 M params) selection-bank retained gap .504 vs DIRV2 .52 (55616). 55842 on P0 states of roots 2200-2399 (states and shared scores reproduce 55666): geometry retained gap CTAV2 .680, ENDV2 .635, DIRV2 .483, DIRV2L .502; CTAV2-DIRV2L +.178 [+.138, +.220]; DIRV2L-DIRV2 +.019 [-.021, +.058]; coverage label: CTAV2-DIRV2L +.197 [+.155, +.241]. Parameter count does not explain the CTA-DIRECT ranking gap.
- 2026-09-29 55840 COMPLETED (train split result, before 55842): parameter-matched DIRV2L (14 layers, 11.31 M params) held-out pooled retained gap .504 [.451, .554], selected step 14000; same pooled split as 55616 where DIRECT .49, ENDPOINT .64, CTA .67, FULL .76. Matching parameters does not close the CTA-DIRECT gap. 55841 (P0-only rerun) and 55842 (P0-state ladder) still running/pending.
- 2026-09-29 CPU analyses of the 55666 logs (main, 1-2 CPU, <1 min each; scripts in `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_crossing_analysis/`, outside the repo): **55877** success-critical decisions, **55878** time-to-success, **55879** recovery after a missed success. A "crossing" is a decision where some sibling reaches coverage > .95 in the chunk and candidate 0 does not.
  - P0 states: 58 crossings in 5,558 decisions (1%), in 45/200 roots; they carry 4.6% of the geometry-label headroom (59% sits in decisions whose best sibling has coverage < .5).
  - Crossing capture, P0 states: GEOM .98, FULLV2 .97, CODEV2 .90, CTAV2 .79, CTA4 .74, ENDV2 .74, DIRV2 .64, DINOWM .60. Net success flips vs candidate 0: CTAV2 +35, DIRV2 +18, DINOWM +17; CTAV2-DIRV2 +17 [+6, +29], CTAV2-DINOWM +18 [+8, +29], CTAV2-ENDV2 +6 [-1, +14] (root bootstrap).
  - Missed crossings are mostly recovered: P0 misses one in 45 roots and 33 of those still succeed (DIRV2 27/37, DINOWM 27/37, CTAV2 18/29), typically 2-9 decisions later. This, not a linear retained-gap x headroom law, explains why the decision-level CTA advantage gives only +2-5 closed-loop successes: DINOWM (retained .33) 136 vs DIRV2 (.48) 135 vs CTAV2 (.68) 140.
  - Episode length (failures = 300) does not separate arms: CTAV2-DIRV2 -11.5 steps [-23.4, +0.5], paired SD ~85 steps.
- 2026-09-29 55841 (8/8) and **55842** COMPLETED: P0-state ladder with the parameter-matched direct scorer. Shared scorers reproduce 55666 exactly (max abs diff 0). Retained gap DIRV2L .502 [.445, .553] vs DIRV2 .483; CTAV2-DIRV2L +.178 [+.138, +.220]; DIRV2L-DIRV2 +.019 [-.021, +.058].
- 2026-09-29 (user: "deploy both") **PushT in-support K-scaling**. Same frozen diffusion policy; 64 samples per decision instead of 8. Candidate k is seeded by (root, decision, k), so candidates 0-7 are exactly the 55666 bank (no perturbed copies, unlike Round 5).
  - Code: `run_arm(draw=)`, oracle `GEOM64`, and scorer kind `parallel_rs` (CTAV2S = reader answers averaged over 8 codes sampled from the WM's categorical, RESEARCH_DESIGN §4; CTAV2SD = their spread) in `ti_wm/cta_batch.py`; `--draw` in `scripts/cta_diag_onpolicy.py`; DRAW and a `kscale` mode in `scripts/slurm_cta_v2_closed.sh`; analysis `scripts/cta_kscale_analysis.py`. Tests: `cta_tests/test_cta_kscale.py` plus the existing `test_cta_batch.py`, 11/11 OK (55884). Dry run of the analysis on 55666 (55886) reproduces the ladder and crossing numbers, and the reference check is exact.
  - Jobs: **55887** smoke (MIG, 40 min) -> **55888** array 0-7%4 (roots 2200-2399, arms P0 and GEOM64, logging CTA4, CTAV2, CTAV2S, CTAV2SD, ENDV2, DIRV2, DIRV2L, DINOWM, FULLV2, CODEV2; MIG 4 CPU 48 GB, 2.5 h cap) -> **55889** `kscale` analysis (CPU).
  - Decision rule, fixed before any result: closed loop of learned arms at K=64 only if, on P0 states, (a) CTAV2's realized geometry gain at K=64 exceeds its K=8 gain (root-bootstrap CI of the difference above 0), and (b) CTAV2-DIRV2L at K=64 has a CI above 0. CTAV2S and the uncertainty-penalized variant are exploratory; a penalty weight chosen on these roots needs a fresh closed loop. GEOM64 success against GEOM8 160/200 (55666) gives the success headroom.
- 2026-09-29 **LIBERO-Safety headroom** (docs/CTA_LIBSAFE_PROTOCOL.md step 2), resubmitting 55634.
  - Fixes to `scripts/libsafe/slurm_libsafe.sh`: 55634 died on `ModuleNotFoundError: ti_wm` because the snapshot copied `scripts/libsafe` to `code/libsafe`, while headroom.py imports from parents[2]; the project layout is now kept. Before this fix, `wait` returned 0 even when a worker failed, so it could not gate a chain; each worker's exit status now fails the job.
  - Jobs: **55890** smoke (WORKERS=2, TSA L1 task 0 init 0, arms P0 and ORACLE8; MIG 8 CPU 64 GB, 40 min) -> **55891** headroom (TSA L1 and FSHOA L1, inits 0-4, P0 and ORACLE8, 100 episodes, WORKERS=3; MIG 12 CPU 96 GB, 3 h).
  - Usage before submit: GPU 193 h (5th-ranked 486), CPU 1715 h (5th 3890), mem 13.6M (5th 50.1M). Planned at limits: about 24 GPU h and 125 CPU h, inside 50% on all three.
- 2026-09-29 smokes COMPLETED.
  - **55887** (K=64, 58 s): all 10 scorers log (64 columns). Candidates 0-7 equal the 55666 bank to 3e-6 in geometry (0.0015 px), which is batch-size numerics in the policy, not seeding; the analysis tolerance is 1e-4. Scoring cost 3.3 s per root-decision, more than estimated, because CTAV2S and CTAV2SD each drew the 8 code samples separately.
  - **55890** (LIBERO, 3:27): P0 success in 214 steps, ORACLE8 in 180; twin fidelity exact (max dqpos 0, no violation mismatch); no mixed-violation bank on this root.
- 2026-09-29 55888 (had run 5 min; 2 shards) and 55889 CANCELLED. The per-user GPU QOS (2 slices) would have queued LIBERO 55891 behind the whole array, and the sampling was done twice. The sampled-code answers are now computed once per scores() step and shared by CTAV2S and CTAV2SD (`BatchScorer._rs_cache`; tests 12/12 OK, 55894). Scope cut to 100 roots (2200-2299). Resubmitted **55895** smoke -> **55896** array 0-3%2 (2.5 h cap) -> **55897** kscale analysis; decision rule unchanged. **55891** LIBERO headroom RUNNING.
- 2026-09-29 **55891 FAILED** at 47 min, with partial results kept. Worker 1 hit JAX out-of-memory at start: three pi0.5 copies do not fit one 3g.40gb slice, so use WORKERS<=2. Workers 0 and 2 finished 67/100 episodes: P0 34, ORACLE8 33, mixed tasks and inits.
  - P0: success 21/34, violation 1/34, safe success 20/34. ORACLE8: success 22/33, violation 0/33.
  - In all 2,038 ORACLE8 decisions, no candidate chunk violated any native constraint: 0% of banks contain a violating chunk, so the share mixing violating and safe chunks is also 0%.
  - Candidate goal-progress spread per bank: median 0, p90 .0097. A success-reaching candidate exists in 1.1% of decisions. Twin fidelity exact.
  - With frozen pi0.5, K=8 and 5-step chunks, this arena gives no safety headroom and little success headroom. The missing shard was not rerun.
- 2026-09-29 **55897 K-scaling result** (100 roots 2200-2299, P0 states, 2,810 decisions). Both predeclared gates PASS.
  - Oracle geometry gain per decision (x1e3): .97 at K=8, 1.61 at K=64.
  - CTAV2 gain .61 at K=8, 1.00 at K=64: +.39 [+.33, +.46], retained ~.62 at every K, worse-than-P0 choices .6%.
  - CTAV2-DIRV2L +.17 [+.11, +.23] at K=8, +.26 [+.17, +.35] at K=64. CTAV2-DINOWM +.35 at K=8, +.50 at K=64. CTAV2-ENDV2 +.09 at K=8, +.11 at K=64.
  - Sampled-code averaging (CTAV2S) is worse than the expected code (-.31 at K=64), so it is dropped.
  - Closed loop on these roots: P0 61, GEOM64 86. In 55666 (K=8, same roots): P0 59, GEOM8 81, CTAV2 72, DIRV2 71, DINOWM 63. Success headroom grows only +5 from K=8 to K=64.
  - Submitted **55912**: closed loop at K=64, roots 2200-2299, arms CTAV2, DIRV2L, ENDV2, DINOWM, cross-scored (array 0-3%2, MIG 2.5 h). Usage before submit: GPU 200 h, CPU 1748 h.
- 2026-09-29 **55912** K=64 closed loop: shards 0-1 COMPLETED, shards 2-3 CANCELLED at 1:27 elapsed on user request (their partial episodes are not used), because the partial result cannot change the conclusion.
  - Roots 2200-2249 at K=64: CTAV2 32, DIRV2L 30, ENDV2 28, DINOWM 33 /50. Reference at K=64 (55896): P0 30, GEOM64 41.
  - Same roots at K=8 (55666): P0 28, GEOM8 40, CTAV2 34, DIRV2 35, ENDV2 33, DINOWM 33.
  - More in-support candidates raise offline gain (x1.66) but not success, for the oracle or for any learned arm. The K-scaling route is closed for PushT closed-loop claims.

## 2026-10-01 replan interval (docs/CTA_REPLAN_INTERVAL_PROTOCOL.md; user: wrap up CTA for CVPR with this test)

Code: executed chunk length is now a parameter, 8 stays the default (old checkpoints load unchanged):
`PolicyRunner(n_exec=)` + `executed_slice` (ti_wm/pusht_runtime.py); `chunk=` in `Scorer`, `ParallelBackbone`
(ti_wm/cta.py, cta_parallel.py); `keep_steps` and `run_segment(keep=)` (ti_wm/cta_runtime.py; (4, 8, 11) at 15);
`macro=` in ti_wm/dinowm_scorer.py (3 macro steps at 15); `--n-exec` in scripts/cta_collect_plus.py and
scripts/cta_diag_onpolicy.py (8-step networks refused at other lengths; v2 checkpoint chunk must match);
`--only-r4 --select-r4` in scripts/cta_train_v2.py (chunk length read from the banks into the config);
read-out scripts/cta_replan_analysis.py; wrapper scripts/slurm_cta_replan.sh; tests cta_tests/test_cta_replan.py.
Outputs `$TI/cta_replan15_<mode>_<job>/`.

| Job | Work | Resources | State |
|---|---|---|---|
| 56348 | smoke: unit tests; collect 2+2 roots x 6 decisions at 15 actions; encode; v2 train 20+20 steps; closed loop 2 roots x 2 decisions, 6 arms; aggregate; read-out (code path only) | mig, 4 CPU, 48 GB, 1 h | COMPLETED 1:42 (quota before: GPU 0 h, 5th-ranked 4 h). REPLAN_SMOKE_OK: 30 + 72 unit tests pass (incl. test_cta_replan); collection n_exec 15, keep (4, 8, 11), chunk (.., 15, 2), steps_executed 15; v2 trains with chunk 15; closed loop n_exec 15 ran P0/GEOM8/CTAV2/ENDV2/DIRV2/DINOWM; aggregate and read-out ran. Code path only |
| 56349_[0-1] | collect_dev: selection banks, roots 2000-2099, 15 actions | mig, 4 CPU, 32 GB, 20 min | COMPLETED 6:27 / 6:47: 1,476 decisions; even (P0) roots 31/50 success; standard-bank siblings with block spread >= 1 px in 33% of banks (same roots at 8 actions: 9.3%, memory 09-26) |
| 56350_[0-1%1] | collect: train banks, roots 34000-34099 (first 2 of 24 tasks) | mig, 4 CPU, 32 GB, 20 min | COMPLETED 7:07 / 6:57: 1,569 decisions; even roots 24/50, odd (mixed) 36/50; bank0 block spread >= 1 px 32-36% |
Quota before 56349/56350: GPU 0 h used, 5th-ranked 4 h (cap 2 h); planned 4 x 20 min = 1.3 GPU-h.
| 56359_[2-7%2] | collect: train roots 34100-34399 | mig, 4 CPU, 32 GB, 12 min | submitted 2026-10-01 06:34 UTC (own use ~0.5 GPU-h, 5th-ranked 4 h, planned 6 x 0.2 h = 1.2 h) |
| 56359_[2-7] | (result) | — | COMPLETED 6:42-7:08 each; train roots 34000-34399 now collected (8 of 24 tasks) |
| 56374_[0-1] | headroom: closed loop at 15 actions, arms P0 + GEOM8 only (no trained scorer), roots 2200-2399 in 2 x 100; later merged with the learned-arm run (identical P0/GEOM8 trajectories are checked by merge_runs) | mig, 4 CPU, 32 GB, 30 min | submitted 07:33 UTC (sreport: own GPU 1 h, 5th-ranked 5 h, cap 2.5 h; planned 1.0 h) |
| 56374_[0-1] | (result) | — | COMPLETED 13:45 / 13:54. Roots 2200-2399 at 15 actions: P0 100/200, GEOM8 152/200 (at 8 actions, 55666: P0 122, GEOM8 160). Oracle headroom 52 vs 38 episodes |
| 56388_[8-23%2] | collect: train roots 34400-35199 (remaining 16 tasks) | mig, 4 CPU, 32 GB, 12 min | submitted 08:10 UTC. User relaxed the 50%-of-5th margin for the first days of October (still never top 5); GPU 5th-ranked 6 h, own ~2 h |
| 56388 | (result) | — | COMPLETED 08:16-09:06 UTC; all 24 train shards (roots 34000-35199, 18,606 decisions) collected. `$TI/cta_replan15_collect_merged/` symlinks the shards of 56350, 56359 and 56388 |
| 56503 | encode: full-token features of the merged train collection + dev 56349, original PCA | mig, 8 CPU, 64 GB, 1:30 | submitted 14:31 UTC (GPU: own 4 h, 5th-ranked 10 h; relaxed margin, never top 5) |
| 56504 | train: v2 recipe at 15 actions (--only-r4 --with-perturbed --select-r4 dev), seed 0, afterok:56503 | mig, 4 CPU, 64 GB, 4 h | submitted, waits for encode |
| 56503 | (result) | — | COMPLETED 18:20 |
| 56504 | (result) | — | COMPLETED 2:01:31. Offline ladder on selection roots 2000-2099 (1,476 banks), geometry retained gap: FULL .776 [.735,.811], CODE .667 [.608,.719], ENDPOINT .440 [.313,.548], CTA .374 [.265,.478], DIRECT .219 [.071,.349] (8-step v2 55616: FULL .76, CODE .63, CTA .67, ENDPOINT .64, DIRECT .49). Codec keeps its quality at 15 actions; the WM tier (CODE -> CTA) loses .29; code 1.8 distinct values per code, siblings distinct 76%. Stage-2 selected step 8000/10000 |
| 56597 | closed: 15 actions, roots 2200-2399 (8 x 25), arms P0/GEOM8/CTAV2/ENDV2/DIRV2/DINOWM, FULLV2/CODEV2 logged | mig, 4 CPU, 48 GB, 40 min, %2 | submitted 2026-10-02 00:53 UTC (GPU own 6 h, 5th-ranked 21 h; relaxed margin, never top 5) |
| 56600 | aggregate + replan read-out vs 55666 (CPU), afterok:56597 | main, 2 CPU, 8 GB, 15 min | submitted, waits for closed loop |

## 2026-10-02 improvement review (user: investigate fixes and a stronger CVPR paper)

- **56686** `ti_cta_review`: read-only audit of archived L=15 train/selection metadata, training-selection curves, and P0/GEOM8 episode reproduction between 56374 and 56597. Main CPU, 1 CPU, 2 GB, 5-minute cap. No models, physics, rendering, images or sealed roots. Source snapshot: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_improvement_audit_20261002/audit.py` and `SHA256SUMS`; output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_improvement_audit_56686/`.
- Before submission: no peer/duplicate CTA review job in squeue/sacct. Month-to-date sreport: own GPU 9 h / 5th 21 h; CPU 39 h / 5th 191 h; memory 493,873 / 5th 1,561,029 MB-hours. This CPU-only job adds at most .083 CPU-hours and 171 MB-hours, below the 50% limits. No GPU submitted for the review.
- Latest end-to-end **56597_0-7 and 56600 COMPLETED**, verified using both squeue and sacct. Full 15-action run (200 development roots): P0 103, GEOM8 149, CTAV2 121, ENDV2 108, DIRV2 103, DINOWM 112. CTAV2-DIRV2 +9 pp [.5,17.5], exact McNemar p=.05045; CTAV2-P0 +9 pp [1,17], p=.04437. Difference-in-differences of CTAV2-DIRV2 between 15 and 8 actions +6.5 pp [-4,17]. Source `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_replan15_aggregate_56600/replan/replan_summary.json`. Earlier 56374 P0/GEOM8 100/152 does not exactly reproduce the full run; the audit measures the mismatch without assuming its cause. Diagnostic runner seconds include simulator branching and cross-scoring and are not deployment latency.
- **56686 COMPLETED 0:0**, verified with squeue and sacct: P0 has **43/200 success flips** (76 step-count mismatches) between 56374 and 56597, GEOM8 has **29/200 flips** (93 step-count mismatches). The runtime, batched loop and evaluation source files compared byte-identically; batch size/scorer setup differ, cause not yet established. This is materially larger than the ±3 aggregate-success difference suggests.
- Audit: L=15 standard train banks have 513/18,606 mixed-success banks, including 45 ignored by the geometry bank margin; perturbed banks 1,233/18,606, including 4 ignored. Geometry oracle misses available native success in only 6 standard / 10 perturbed banks, so this does not establish that the true geometry oracle is generally misaligned. The two-pool sampler allocates 57.23% of exposures to perturbed banks; 5.56% of exposures are mixed native-success banks. Source `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_improvement_audit_56686/audit.json`.
- **56690** `ti_cta_repro_review`: bounded proposal reproducibility investigation, no training or sealed roots. MIG 1 slice, 4 CPU, 32 GB, **5-minute hard cap**. Draws initial K=8 banks at batch sizes corresponding to 1/4/25/100 roots, then P0 for development root 2205 alone / with 2206 using the old sampler and an isolated per-root canonical-shape adapter. Source/config snapshot and partial results in `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_repro_review_56690/`. Existing dirty runtime files unchanged.
- Before 56690: squeue/sacct show no duplicate CTA audit; 56657 is the unrelated peer `cg_p1` job. Required hour-ranking query checked; exact seconds: own GPU 31,555, fifth 76,570 (50% = 38,285). Conservatively adding the entire 6,000-second limit of live 56657 plus 300 seconds of 56690 gives 37,855, below the cap. CPU/memory caps also hold counting the peer's full limit (6 CPU / 160 GB / 100 min) and this job. No training campaign is queued.
- **56690 COMPLETED 0:0 in 2:00**, squeue/sacct verified. At root 2205, observation and candidate seeds fixed, initial proposal action max differences are .0563 / .0371 / .0356 world units when old batch size changes from 1 root to 4 / 25 / 100. Old P0 root-2205 max coverage changes .5075 -> .9382 when evaluated alone versus alongside root 2206 (both fail); the isolated canonical adapter exactly reproduces success at step 233 and coverage .952236 in both settings. This demonstrates batch-size dependence and a bounded repair; it is not a full paper evaluation or a complete causal accounting of all 43 flips.
- Added opt-in `--proposal-mode canonical` to collection and closed-loop diagnostics, backed by new `ti_wm/pusht_canonical.py`. It computes policy conditioning once per history and denoises fixed 8-candidate microbatches, padding the last microbatch, so candidate prefixes are invariant to bank size and active-root count. Mode/microbatch are recorded in run reports. Existing dirty code edits preserved; historical batched mode remains default. No model weights changed.
- **56693** (same MIG / 4 CPU / 32 GB / 5-min cap): qualify the production canonical sampler on real inputs, K=1/8/16 prefix equality, grouping equality, and complete P0 trajectory equality for root 2205 alone / with 2206. Source snapshot and partial results `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_repro_review_56693/`. Before submission, squeue/sacct showed only peer 56691 (`cg_probe`, 35-min limit); 56657 and 56690 COMPLETED. Required hour-ranking checked. Conservative stale-accounting GPU upper bound = 31,555 reported + 3,738 actual completed peer 56657 + 120 actual 56690 + 2,100 full-limit live 56691 + 300 full-limit qualification = 37,813 s < half fifth-ranked 38,285 s; memory upper bound also below 50% counting both 160-GB peer jobs.

- **2026-10-02 research review correction:** `56693 FAILED 1:0` after 2:05, verified with both squeue/sacct. Initial production canonical proposal/prefix checks K=1/8/16 are bitwise equal, and target root's episode result matches alone/paired. The strict full-log bitwise assertion failed; do not describe this GPU job as completed successfully.
- **56705 — ti_cta_logcheck**, CPU main, 1 requested CPU / 2G / 5 minutes, `COMPLETED 0:0`, both scheduler views checked. Immutable script SHA256 `ad27a8c7122a228f9df7bb705552d64981fa5316787f16c04653dcca604d3dd2`; output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_repro_logcheck_56705/logcheck.json`. Per-field comparison of archived 56693 NPZ logs: canonical root/decision/t/chosen/geom identical; 4/128 coverage labels differ by at most `3.33e-16`, with no episode change. Old sampler coverage max discrepancy `.92608`, geometry `.30250`. Coverage ULP differences do not explain the large old-sampler discrepancy. Future audit requires exact proposal prefixes and discrete/geometry log fields, coverage `atol=1e-12, rtol=0`; raw failed job remains failed. This is a bounded two-root P0 check, not full population, scorer, backend or hardware qualification.

- **56709 — ti_cta_logcontract**, CPU main, 1 requested CPU / 2G / 5 minutes, `COMPLETED 0:0`, verified with squeue/sacct. Immutable script SHA256 `5a3657f92b29b9b2226e3295822bf03facfee059a01f04c004924246e603010d`; output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_repro_logcontract_56709/logcheck.json`, `bounded_contract_pass=true`. Reuses archived 56693 outputs; no additional model loading, physics, GPU use or evaluation roots. Stores the initial bank/prefix exactness and per-field logged-decision checks explicitly; does not certify other roots, learned scorers or hardware. Review and recommendations saved in `docs/CTA_IMPROVEMENT_REVIEW_20261002_VI.md`; sampler is opt-in in collection/diagnostic scripts. No training campaign submitted by this review. Existing peer cg jobs left untouched.


## 2026-10-02 H100 continuation during DNS failure

- User asked to continue the interrupted native-hit implementation on H100, then confirmed their terminal also cannot use Slurm/DNS and authorized source preparation first. SSH/filesystem work; `/etc/resolv.conf` on `login-restricted-1` is 0 bytes. Both `squeue` and `sacct` fail to reach/resolve controller/accounting. Current peer/job state and month-to-date quota are UNKNOWN; no jobs were submitted or cancelled, and no model, physics, encoding, training or bulk analysis was run on the login node.
- Finished `scripts/cta_hit_train.py` with six matched CTA/ENDPOINT/DIRECT control/hit models. Each pair warm-starts identical weights; equal continuation update budgets, independent optimizers/clipping, common full selection split and 16-goal averaging. Source codec/code reader/FULL reader frozen. Controls use the old sampler/loss; HIT uses rebalanced sampling plus native-hit loss. Actual exposure logs cover both groups. This is a combined intervention, not an isolated hit-loss ablation. Existing training banks remain historical batched-policy data; deployment proposals use canonical mode.
- Hardened training CLI and K8/L15 action, label and feature shapes via pure `ti_wm/cta_hit_contracts.py`. Closed loop uses 11 acting arms (P0, GEOM8, CTA/END/DIR x BASE/CTRL/HIT), scores C,A before candidate simulation, fixed per-root scoring, and logs clone/smoke/preparation/dependency/actual visual weight/source identity. Integrated bounded qualification adds exact native-hit and discrete episode outcome checks; boolean log comparisons cast before max-absolute-difference. Aggregation rejects invalid candidate indices/nonbinary labels/nonboolean outcomes, partial/truncated shards and identity mismatch; undefined ratios serialize as strict JSON null and render n/a.
- Added `scripts/slurm_cta_hit.sh` stages tests/smoke/train/closed/aggregate. CPU defaults 2 CPU/8G/10 min; GPU modes need explicit allocation. Immutable `CTA_SOURCE_ROOT` must be passed via sbatch --export (Slurm script spool path is not source path). Planned initial closed pilot is 20 dev roots (array 0-1, COUNT=10), with all baseline/treatment arms; no sealed claims or deployment latency claims.
- Immutable release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_hit_resume_20261002_dns_01a0fcfe/code/`; 241 source/document files, read-only, hashes in sibling `SOURCE_SHA256.json`. Original unfinished three scripts backed up in sibling `prior_work/`; installation checked their SHA256 before replacement, preserved all unrelated dirty files, and recorded `VERIFICATION.json`. Run protocol: `docs/CTA_H100_RESUME_20261002_VI.md`.
- Verification: 8 pure contract tests failed against stubs then passed after implementation; an added feature-shape regression brings total to **9 PASS locally** (no model/simulator imports). Remote changed-source AST, new launcher `bash -n`, `git diff --check`, and all 241 source hashes PASS. Six compute-only aggregate/qualification regressions are written but NOT RUN; full existing tests, GPU smoke, actual checkpoint forward/backward and closed-loop improvement remain UNVERIFIED. No new checkpoint or success result exists.
- Next when DNS/scheduler recover: check both squeue/sacct, peers/duplicates and current sreport ranking; retain mandatory <=50%-of-fifth-ranked GPU/CPU/memory budget counting planned limits and live/pending work. Submit CPU tests, then a bounded GPU smoke, then quota-fitting continuation and matched closed loop/CPU aggregation from this release. Do not queue a campaign from stale quota or infer success from the source checks above.

### 2026-10-02 — CTA paper review/rewrite (no cluster job)

- Rewrote `paper_cvpr/main.tex`, sections, bibliography and vector figures from archived evidence. Added `supplement.tex`, Vietnamese reviewer report, archived evidence/config hash manifest and revision source/PDF manifest.
- Main: 8 pages including references (start p7); supplement: 3 pages. Local MacTeX builds and visual QA pass; no overfull/underfull or unresolved citation/reference warnings.
- Preserved active prior paper in `paper_cvpr/draft_v4_before_20261002_review/`. Immutable H100 release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_paper_20261002_review_01a0fcfe/`. Guarded source sync preserves peer files and does not modify training/checkpoints/results.
- Claims are development ranking findings and explicitly historical single-seed control. Canonical learned-arm/native-hit results and sealed multi-seed validation remain unrun. No Slurm submission or new bulk result analysis; DNS outage does not justify using stale quota/job state.


## 2026-10-03 — CTA paper world-model revision after user feedback (no cluster job)
- Revision: `cta_paper_20261003_world_model_repair_01a0fcfe`.
- Restored world-model title and CVPR review mode; neutral framing, two-stage method and four detailed figures. Main performance includes only deployable learned methods; actual-future diagnostics moved to supplement.
- Added existing100-root candidate-expansion evidence; no new experiment. Main9pages total, body ends on8; supplement4pages. Local MacTeX build and visual QA; no unresolved layout/reference/font warnings.
- Guarded38 unchanged remote files; previous source/PDF backed up at `paper_cvpr/draft_v5_before_20261002_user_feedback`. Immutable release: `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_paper_20261003_world_model_repair_01a0fcfe`.
- squeue timed out after12s; sacct accounting/address-family error. Jobs submitted0; no training/evaluation or compute-budget use.
- Remaining research: canonical learned control on independent roots/seeds, focused attribution controls, full-planner timing. Existing continuation remains unrun. See `docs/CTA_PAPER_REVISION_20261003_VI.md` and paper manifests.

## 2026-10-03 — reviewed native-hit implementation and end-to-end pilot (user: review then run)

- Scheduler/DNS recovered: both squeue and sacct reachable, no active account jobs at first review. No native-hit training/evaluation job had been submitted during the interrupted/DNS-outage work. Existing paper and unrelated dirty source preserved.
- Independent read-only review of six paired CTA/END/DIRECT trainers and deployable evaluator found no blocking leakage/alignment/autograd issue. Root retained DIRECT's original stage-1 `rank_loss` in both continuation arms (WM pairs retain weighted ranking), pinned backend flags, and added Torch/CUDA RNG + scheduler states to recovery checkpoint. No resume CLI claim. Native labels remain authoritative `done8`; source/readers frozen.
- Immutable source release `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_hit_review_20261003_v1/code`, hashes in sibling `SHA256SUMS`; Python AST, launcher bash syntax, git diff check pass.
- **56912 — ti_cta_hit tests**: CPU main, 2 CPU / 8G / 10-min cap. Full tests and cta_tests from immutable release; no models/physics on login. Output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_tests_56912`; log `/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_hit_ti_cta_hit_56912_4294967294.out` (confirm actual glob). Runtime tests pending at submission, not claimed passed.
- Current required month ranking: own GPU12h / fifth49h, CPU56h / fifth395h, memory949394 / fifth4283924 MB-hours. Planned bounded smoke15min + train90min + 2×45min closed at1GPU/4CPU/64G, plus CPU tests/aggregate: <=3.25GPUh, ~13.7CPUh, ~215722MB-hours added, inside even the original50% margins. User's earlier permission to ease early-month compute is not needed for this pilot; still check ranking/peers before each GPU submission. Full training only after real GPU smoke succeeds. No speculative sweep.

- **56912 COMPLETED 0:0, 23 seconds**, verified using squeue+sacct. All 30 core tests and 97 CTA tests pass (127 total), including native-hit gradients/sampling/contracts/aggregate tests. These are correctness tests, not evidence of improved control.
- **56913 — GPU smoke**: MIG 1×3g.40gb, 4 CPU / 64G / **15-min cap**. 20 continuation updates on a bounded feature slice, real checkpoint strict-load and gradients, frozen-source checks, then all11 learned/oracle arms for2decisions on2development roots with numerical qualification. Immutable review v1 release. Source/results `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_smoke_56913`; output glob `logs/cta_hit_*56913*`. Required ranking and both scheduler queries checked immediately before submission; no live peer/duplicate jobs. No full training queued yet.

- **56913 COMPLETED 0:0, 1:50**, both scheduler views verified. Real checkpoint load, all six continuation networks' forwards/backwards, frozen source/readers and 11-arm bounded runtime pass. Qualification PASS: K1/8/16 proposal prefixes and grouping exact; all9 learned scorers' initial scores exact alone/paired; bounded P0/GEOM decision/geometry/native-hit/outcome logs match. Smoke's first64train rows contain0mixed-hit banks, so native-hit loss follows its zero branch here; synthetic gradient tests pass, and nonzero real native-hit gradients must be checked in full train's first logs. Truncated2decision outcomes are not a success comparison. Diagnostic ~1.25s/root-bank includes policy+all9scorers+physics; no deployment speed claim.
- **56914 — bounded full continuation**, MIG1slice/4CPU/64G/**90min cap**, 2400updates for each of six paired networks, full1476-bank selection/all16goal images at0/1200/2400, frozen source/readers. Output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_train_56914/train` with control/hit checkpoints, last states, losses/exposure/selection. RUNNING confirmed with squeue+sacct; first selection currently in progress, improvement not claimed.
- **56915 — matched closed pilot**, afterok56914, array0-1%1, MIG1slice/4CPU/64G/**60min per shard**.20development roots2200–2219,11actingarms and9crossscorers, canonicalK8/L15/per-root scorer. Cap increased from proposed45min after observedsmoke cost to avoid partial long episodes; total reservedGPU <=3.75h includingsmoke, still below strict50% GPU/CPU/memory margins counting full limits. Source/results `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_closed_56915/shard_<task>/results`.
- CPU aggregate submitted afterok56915,2CPU/8G/10min; jobID recorded in next scheduler update. Expected exact rootpopulation2200–2219. No training/eval/analysis on login. Required ranking and both scheduler views checked before GPU submissions; own12GPUh/fifth49, CPU56/fifth395, memory949394/fifth4283924MBh. No peer/duplicate job active; all stages use immutablev1release. Pending dependencies allocate no GPUs.

- **56916 — CPU aggregate**, afterok56915,2CPU/8G/10min. Source/result `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_aggregate_56916`; bothschedulerqueries confirm dependency pending. Reads completed11-arm pilot only; truncatedsmokeexcluded.
- **56914 runtime confirmation:** initial full1476-bank/all16goal selection exactly reproduces expected base retainedgaps CTA.373872/END.439646/DIRECT.219485, with eachcontrol/hit pair initiallyidentical. First real trainupdate has8mixed-hit banks of16 inintervention vs1incontrol; native-hit parts CTA.501566/END.531281/DIRECT.444937, finite nonzerogradient norms, noautograderror. Full source frozen check remains forcompletion. Initialselection finished at~213sec. This validates realnative-hit trainingpath missing fromboundedfirst64rowsmoke; noimprovementclaim yet.
- **56917 — training artifact readout**, CPU main 2 CPU / 4G / 5-min cap, afterok56914. Immutable readout source `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_hit_readout_20261003_v1/code`; runs five artifact tests then reconstructs selected scores/exposure and paired selection-root uncertainty without model loading. Output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_training_readout_56917`, log `logs/cta_hit_readout_56917.out`. Both squeue and sacct confirm dependency pending. Selection uncertainty is development evidence only because these labels select the checkpoints.
- **56917 cancelled while dependency pending**, replaced by **56918**, main 2 CPU / 8G / 5-min cap afterok56914. Added source-agreement and score-scale diagnostics for all saved states so a selected-step0 fallback cannot conceal trained-state deterioration. Immutable source `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_hit_readout_20261003_v2/code`, hashes in sibling SHA256SUMS. Runs five fresh Factorized DIRECT architecture/leakage/gradient/checkpoint tests plus five readout artifact tests, then real training readout to `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_training_readout_56918`. No additional GPU job and no in-flight training/eval source changed. New Factorized DIRECT trainer is prepared, reviewed read-only and AST-checked but NOT TRAINED; geometry/done8 are future-derived task labels, while future features/code-target supervision are absent. Its deployment architecture/data match CTA; training compute, initialization, dense loss and hit temperature differ and are explicitly disclosed.
- **56919 — fallback numerical check**, main 2 CPU / 8G / 5-min cap afterok56915. Immutable `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_hit_fallback_20261003_v1/code`; eight compute-only tests then completed pilot readout to `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_fallback_56919/fallback_check.json`. Checks selected-step0 aliases against BASE: bitwise cross-scores on every acting state, exact decision/native-hit/geometry trajectories and episode flags; coverage/score tolerance1e-12. No model/checkpoint tensor loading, no extra GPU evaluation. Does not attest tensor-weight equality or universal reproducibility. Both scheduler views confirm dependency pending.
- **56919 cancelled while pending**, replaced by **56920** from immutable `cta_hit_fallback_20261003_v2/code`, same CPU resources/dependency. Added ninth test for all CTRL step0 aliases as well as HIT; tests plus actual fallback check run once, with no allocated compute wasted by replacement. Result `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_fallback_56920/fallback_check.json`. No GPU pipeline changes.
- **56914 COMPLETED 0:0, 40:22**, verified by squeue+sacct. All2400 updates completed, source encoder/code reader/FULL reader unchanged by exact tensor check. Selected steps CTA_CTRL2400, CTA_HIT0, END_CTRL2400, END_HIT2400, DIR_CTRL0, DIR_HIT1200. CTA HIT did not improve and falls back to BASE; END HIT final selection retained gap.498263 vsBASE.439646 and CTRL.454543, mixedcapture24/49 vs19/49 inBASE/CTRL. Selection-only gain relativeBASE.0586 (descriptive paired95%CI[.007,.127]); HIT-CTRL.0437 CI[-.020,.123]. No held-out/closed-loop claim.
- **56918 COMPLETED 0:0, 4seconds**: all10 new FD/readout tests pass, real artifact readout completed, so127+10 tests verified across jobs. Source CODE mixedcapture44/49 vsCTA22/49, but new predicted-source centered normalizedRMSE~.95 remains large. Source agreement uses historical batch4 source scores vsnewbatch1 prediction and lacks embedded historicalrowIDs, disclosed. Actual intervention exposure80.14%standard/28.32%mixed vsCONTROL42.74%/5.57%, validating intended sampling. Output `cta_hit_training_readout_56918/training_readout_vi.md` and JSON. **56915_0 RUNNING** aftertrain, shard1pendingarraylimit; remaining CPUjobsdependentpending.
- Focused next iteration source preparation: paired CODE-reader calibration on actual hard source codes and frozen WM expected codes, shared rebalanceddata; CTRL denseonly vsHITdense+native objective. Addresses a concrete observed source-to-deployed score gap and possible reader hard/continuous input shift; this is a hypothesis, not a proven cause. Encoder/WM remainfrozen, reader receivesC,S,g only. No new training/evaluation job submitted for calibration yet; currentpilotunchanged.
- Reader calibration implemented in NEW `scripts/cta_reader_calibrate.py` and five compute-only tests; independent read-only review found no blocker. No-grad frozen codes support trainable-reader backward; CPU/CUDA dropout states are paired; independent optimizer/clipping. Selection opens observed features only and reads frozen-WM predicted codes. Immutable `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_reader_calibrate_20261003_v1/code`, siblingSHA256SUMS.
- **56922 — reader CPU tests COMPLETED 0:0 in6seconds**, both scheduler views verified; all5tests pass (142 total verified across current test jobs). Source/expected native gradients, no-mixed behavior, frozen model/gradient checks, paired dropout and future-free selection verified on CPU. GPU runtime remains for training.
- **56923 — bounded paired reader calibration**, MIG1slice /4CPU /64G /**20min cap**,800updates, full1476/all16predicted-onlyselection0/400/800, shared.8standard/.25mixeddata. CTRL/HIT each train source+predicted dense loss; HIT alone adds native-hit term, fixed BASE-reader temperature. Source/WM frozen; stage1.reader+wms.cta minimal deployment bundles. Output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_reader_train_56923/train`, immutable per-jobcode/source manifest. This is a focused iteration after observed CTA intervention failure, not a speculative sweep; original56915pilot stays unchanged. GPU submission preceded both squeue+sacct, peer/duplicate check and required sreport: ownGPU12h/fifth49h, CPU58/fifth395h, memory991043/fifth4382228MBh. Conservatively add both existing60minclosed limits +new20minreaderlimit, CPU downstream jobs: GPU<=14.34h, CPU<68h, memory<1.15millionMBh, all below50% margins. No reader closed job queued yet.
- **56923 COMPLETED 0:0,5:35**, both views verified. Initialselection reproducesBASE.373872/22of49 forbothreaders; firstCUDAstep dense/source-hit/pred-hit parts exactly equal acrossreaders beforeoptimization, finite gradients, real8mixedbanks. Final exactenc/WM parameter+buffer+no-gradient checks pass. Laststep800 CTRLretention.307467/HIT.296366; bothmixed22/49, so bothselectedstep0. This calibration recipe did not help; no duplicate closed evaluation submitted forunchangedselectedreaders. Source-retention/expected-code diagnostics are next, not a proven attribution of failure.
- NEW `scripts/slurm_cta_reader_closed.sh` prepared and syntaxreviewed forexplicitminimalreaderCTA specs,8arms/6scorers,newdevroots2220–2239 andpairedaggregation. Not submitted becausebothreadersretainedBASE. Currentnative20rootpilotunchanged.
- **56915_0 COMPLETED0:0,35:02**, numericalqualificationPASS andall11armscompleted onroots2200–2209; shard1RUNNING withqualificationPASS, bothschedulerqueriesverified. Compute-generated firstshard counts P0/GEOM8 7of10 each; CTA_BASE/CTRL/HIT6each; END_BASE5,END_CTRL7,END_HIT6; DIR_BASE/CTRL6,DIR_HIT5. Incomplete20rootpilot: no winclaim orpooledCI yet. CTA_HIT andDIR_CTRL step0 aliases have samefirstshardcounts/decisioncounts asBASE, fullbitwisecheckswaitCPU56920.
- NEW code-prediction audit (`scripts/cta_code_prediction_audit.py`, three tests) reviewed for exactMAPgrid, future-free mean/MAP paths, last-reader vsselected-step0 state and matchedlabels/rootbootstrap. Immutable `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_code_prediction_audit_20261003_v1/code`; source/base/reader/goal/metadata identities saved; scores embedrowIDs. **56930 CPU tests COMPLETED0:0,6seconds**, all3pass (145 totalverified). No bulk/model workonlogin.
- **56932 — focused code/reader audit**, MIG1slice /4CPU /64G /**10min cap**, full1476selectionbanks/all16goals/fixedbankbatch1. Sevenconditions: BASEreader(sourceoracle/MEAN/MAP), finaltrainedCTRL/HITreaders(sourceoracle/MEAN). Measures source-semantic retention afterfailedreader calibration, codepredictionvariance/NLL/entropy andMAPinput-domain effect. Actualfuture only privilegedsourceoracles/diagnosticlabels; mean/MAP fromWM(C,A), readerC,code,g. No newtraining; this check resolves the nextmethodchoice afterobservedfailures. Output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_code_audit_56932/audit`. BeforeGPUsubmit bothschedulerqueries/peers/duplicate andrequiredsreport verified: ownGPU12/fifth49,CPU58/fifth395,memory991043/fifth4382228MBh. Addingbothexistingclosedfulllimits +new10mincap +pendingCPUjobs remainsstrictlyunder50% margins. Nativepilotcontinuesunchanged.
- Audit job ID correction: actual `sbatch --parsable` returned **56932**; **56931** is peer `ew_code2`, not CTA. Both scheduler views show CTA audit pending on QOS while native pilot shard1 runs. Shared-account peer jobs 56931 (6 CPU/48G/50min) and 56933 (6 CPU/48G/40min) are preserved. Rechecked sreport: GPU12/fifth49, CPU58/fifth395, memory991043/fifth4382228 MBh. Adding full limits of both native60min shards, audit10min, both peers90min total and downstreamCPU remains below GPU15.67h<24.5h, CPU<77h<197.5h, memory<1.22million<2.191millionMBh. No extra GPU submission.
- **56934 — completed-pilot success figure**, main1CPU/2G/**5mincap**, afterok56916:56920. Immutable minimal source `/mnt/data/nhatnc129/jepa/trajectory_innovation/releases/cta_hit_plot_20261003_v1` (script+compute contract, SHA256SUMS), output `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_plot_56934/plot`. Reads CPU-produced summary/fallback JSON only; no model/physics/NPZ work. PNG/SVG use saved per-arm Wilson95 intervals, distinguish marginal intervals from paired comparisons, annotate step0 BASE aliases and privilegedGEOM8; all20roots/11arms and successful fallback check required. This reporting job adds at most.083CPUh/171MBh, below previously verified shared-account margins. ASTreviewpassed; no tests added for low-impact reporting.
- **56932 COMPLETED0:0,2:48**, both scheduler views verified; frozenencoder/WM exact checksPASS. Full1476/all16/batch1 internal comparisons: SRC_BASE retainedgap.668 /44of49 mixed, MEAN_BASE.374 /22of49, MAP_BASE.098 /24of49. MAP-MEAN geometry-.276 descriptivepaired95CI[-.399,-.155], mixed+4.1pp CI[-12.5,+19.6]; this does not justify closedMAP evaluation. TrainedHITreader preserves source quality(.669) while deployedMEAN falls(.296), so semanticforgetting alone is not an explanation. CONTROLsource.639 andMEAN.307. Expected within-bank code variance is4.23% ofsource, candidate-centeredcodeMSE/sourcevariance.9516; these indicate severely attenuated predictedactioncontrasts, not proof oftheonlycause orallfuturepredictability. Audit `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_code_audit_56932/audit/audit_vi.md`. Nativepilotshard1 stillRUNNING; noadditionalGPUtraining submitted.
- **Native 20-root pilot complete**, verified with both squeue and sacct: 56915_0 COMPLETED 0:0 / 35:02; 56915_1 COMPLETED 0:0 / 34:36; aggregate 56916 COMPLETED 0:0 / 14 sec; fallback 56920 COMPLETED 0:0 / 5 sec (9 tests PASS); plot 56934 COMPLETED 0:0 / 3 sec. **154 tests verified**: 127 + 10 + 5 + 3 + 9. Both qualifications PASS; matching immutable source/backend/checkpoints; all 11 arms and 20 roots complete. No CTA job remains pending or running; other account `ew_*` jobs are peer-owned and preserved.
- Success: P0 10/20, privileged GEOM8 17/20; CTA BASE 14 / CTRL 11 / HIT 14 (HIT = BASE step 0); END BASE 11 / CTRL 13 / HIT 13; DIR BASE 13 / CTRL 13 / HIT 10 (CTRL = BASE step 0). Fallback checks PASS for CTA_HIT and DIR_CTRL: bitwise cross-scores on all 11 acting-state distributions, matching discrete/native/geometry/episode logs, coverage tolerance 1e-12. CTA_HIT-P0 +20 pp, paired 95% CI [-10,+50], McNemar p=.3438; no reliable paper win established. END_HIT-END_CTRL 0 pp, CI [-30,+25.1]; DIR_HIT-DIR_CTRL -15 pp, CI [-30,0]. The native intervention did not improve CTA and has no closed-loop gain over END control.
- Offline-to-control mismatch: CTA_CTRL selection .3844 vs BASE .3739, but closed success 11 vs 14 and mean score .858 vs .931; END_HIT selection .4983 vs CTRL .4545, but closed success 13 each and mean score .908 vs .932. On shared P0 states (325 decisions), only 4 native crossing opportunities; all CTA/END scorers capture 3/4, DIRECT 1/4. This pilot does not justify strong phase or causal claims. Roots 2200–2219 are reused development roots, not new or sealed held-out data.
- Completed artifacts: `/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_hit_aggregate_56916/aggregate/summary.json` and `.md`; fallback `cta_hit_fallback_56920/fallback_check.json`; PNG/SVG/provenance `cta_hit_plot_56934/plot/`. Scientific plot rendered on CPU batch node; visual inspection PASS. All model, physics and analysis work ran through Slurm, with no idle GPU holders or speculative duplicate arrays.
