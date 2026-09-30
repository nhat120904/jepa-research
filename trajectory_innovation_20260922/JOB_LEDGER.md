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
