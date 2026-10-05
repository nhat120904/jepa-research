# CTA paper evidence audit — 2026-10-02

Read-only audit of saved H100 summary JSON/Markdown and current ledger/review. No model, physics, rendering, statistics regeneration, tests, sealed roots, or new jobs. Remote repository: `/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922`. Artifact prefix below: `/mnt/data/nhatnc129/jepa/trajectory_innovation/`.

## L8 ranking evidence (all same saved P0 decision banks)

Source: `cta_ladder_55762/paired_ladder.json` and `.md`, saved from `cta_v2cl_closed_55666`; parameter matched control: `cta_ladder_rerun_55842/paired_ladder.json` and `.md`, from `cta_v2cl_closed_55841`. 200 development roots 2200–2399, single training seed, K8/L8. Geometry label; 3,433 spread decisions among 5,558 P0 decisions. Paired root-bootstrap 95% intervals.

| Arm | Geometry retained gap [95% CI] |
|---|---|
| FULLV2, true final future reader | .756 [.731,.778] |
| CODEV2, true source code reader | .628 [.596,.657] |
| CTAV2, predicted code | .680 [.629,.726] |
| ENDV2 | .635 [.588,.678] |
| DIRV2, original direct | .483 [.425,.535] |
| DIRV2L, parameter matched direct | .502 [.445,.553] |
| DINOWM, official checkpoint adapter | .334 [.272,.393] |

Paired CTA−END: +.045 [.011,.079]; CTA−DIR: +.197 [.153,.241]; CTA−DIRL: +.178 [.138,.220]; CTA−DINO: +.346 [.294,.398]. DIRL−DIR: +.019 [−.021,.058]. Shared scorer scores reproduce reference exactly in saved reference check (max absolute difference zero).

Supported: substantial ranking advantage in this development distribution; larger parameter count alone does not close direct ranking gap. Unsupported: attribution specifically to conditional future supervision (factorized direct still missing), superiority in closed-loop success, semantic goal reuse, sealed evaluation. FULL/CODE are privileged logged scorer tiers, not acting closed-loop arms or strict upper bounds. At L8 source CODE is below predicted CTA: do not force a monotonic loss ladder.

## Closed-loop development results

Sources: `cta_v2cl_aggregate_55676/summary.json` (L8); `cta_replan15_aggregate_56600/aggregate/summary.json` (L15); `cta_replan15_aggregate_56600/replan/replan_summary.json` (same exact counts plus matched contrasts). Both: 200 reused development roots 2200–2399, one training seed, historical batched proposal mode, internal pairing only.

| Arm | L8 success /200 | L8 mean score | L15 success /200 | L15 mean score |
|---|---:|---:|---:|---:|
| P0 | 122 | .960874 | 103 | .911431 |
| GEOM8 oracle | 160 | .970499 | 149 | .948252 |
| CTA4 parent | 142 | .961408 | — | — |
| CTAV2 | 140 | .959566 | 121 | .929717 |
| ENDV2 | 138 | .956092 | 108 | .936284 |
| DIRV2 | 135 | .972704 | 103 | .920229 |
| DINOWM | 136 | .974226 | 112 | .932264 |

| Paired success difference | L8 pp [95% CI], exact McNemar p | L15 pp [95% CI], exact McNemar p |
|---|---|---|
| CTA−P0 | +9 [1,17.5], .044371 | +9 [1,17], .044371 |
| CTA−DIR | +2.5 [−5.5,10.5], .625407 | +9 [.5,17.5], .050452 |
| CTA−END | +1 [−7.5,9.5], .907561 | +6.5 [−2.5,15.5], .182077 |
| CTA−DINO | +2 [−6,10], .716301 | +4.5 [−4.5,13.5], .385669 |

L15 CTA−DIR p is slightly above .05, although paired bootstrap CI excludes zero: state both, no unqualified significance claim. Mean-score contrast intervals all cross zero for these CTA comparisons; L8 CTA mean score is below P0/DIR/DINO. L15−L8 difference-in-differences CTA−DIR +6.5pp [−4,17], CTA−END +5.5pp [−6.5,17.5], CTA−DINO +2.5pp [−10,14.5]. Cannot claim stronger CTA advantage at longer horizon; L8 and L15 training data also differ.

## Oracle/source/predicted ladder on selection data

Sources: `cta_v2_train_55616/train_s0/offline_ladder.json`, `cta_replan15_train_56504/train/offline_ladder.json`.

| Reader tier | L8 selection RG [95% CI] | L15 selection RG [95% CI] |
|---|---|---|
| FULL | .761 [.731,.787] | .776 [.735,.811] |
| CODE | .626 [.588,.662] | .667 [.608,.719] |
| CTA | .670 [.623,.715] | .374 [.265,.478] |
| ENDPOINT | .642 [.592,.688] | .440 [.313,.548] |
| DIRECT | .490 [.432,.544] | .219 [.071,.349] |

L8: pooled 4,260 selection banks (3,011 r0dev, 1,249 onpolicydev). L15: 1,476 selection banks, 100 roots 2000–2099. L15 CODE−CTA point gap .2936 supports prediction/readout as a specific bottleneck at L15; no paired CI for this subtraction was regenerated. On P0-state native crossing banks, saved replan summary: L8 CTA capture .7931, CODE .8966, FULL .9655, geometry .9828 (58 crossings/5,558 decisions); L15 CTA .3714, CODE .9143, FULL .9429, geometry 1.0 (35 crossings/3,249 decisions). Rare crossings, recovery and distribution changes limit translation into success. Selection curves CTA L15 .0846/.2075/.2604/.3245/.3243 at 2k/4k/6k/8k/10k: plateau under current criterion, not global convergence.

## Component timing and fairness

Source: `cta_budget_55763/budget.json`, `.md`, and immutable `code/scripts/cta_budget.py`. Archived launcher/ledger: H100 3g.40gb MIG. Ten initial states roots 2200–2209, ten scoring repeats/state, three warmups, CUDA synchronization. Components below are median ms; p90 saved, no p95/whole-planner benchmark/memory saved.

| Component, K8 | G1 | G16 |
|---|---:|---:|
| Shared two-frame DINOv2/PCA context (ours only) | 6.4865 | 6.4808 |
| CTA predictor+reader | 4.1096 | 16.8681 |
| Endpoint predictor+reader | 4.5169 | 23.4048 |
| Direct scorer | 2.5358 | 26.8131 |
| Official DINO-WM own encode+rollout+latent cost | 28.9570 | 29.0724 |
| Policy sampling, common | 765.3326 | 765.3326 |

Params: CTA 10,950,933; endpoint 11,136,901; direct 6,571,777; matched direct 11.31M (14 layers); official DINO-WM predictor/action/proprio encoders 20,122,760; shared/own DINOv2 22,056,576; common policy 262,709,044. Exclude frozen encoder/policy when describing learned scorer params; report exclusions.

CTA/END/DIR timings exclude shared encoding, DINO already includes its own encoding. Our scorer path uses planner AMP; equal precision with official DINO is not established. DINO observes current agent velocity and renders current 224px frame; CTA observes agent-position history and 96px frames. Same candidate action banks do not imply matched observations/training data/model selection. G16 averages images of one PushT target; it measures query-computation scaling, not distinct semantic goal transfer. Sampling dominates; 4.1 vs29 ms is not a corresponding whole-planner speedup. Diagnostic runner seconds include simulator branching and cross-scoring, so they are not deployment latency.

## K scaling and Round-2 co-design failures

K-scale source: `cta_v2cl_kscale_55897/kscale.json` and `.md`; ledger closed-loop partial result `55912`, complete shards0–1 only, roots2200–2249. On 100 P0 roots/2,810 decisions, CTA geometry gain x1e3 is .61 at K8 and1.00 at K64; +.39 [.33,.46]. Oracle .97→1.61; CTA retained gap ~.62. CTA−DIRL gain .17 [.11,.23] atK8 and .26 [.17,.35] atK64. Eight-sampled-code reader vs expected-code CTA: −.31 [−.35,−.26] atK64.

K64 learned closed loop /50: CTA32, DIRL30, END28, DINO33; reference P030, GEOM6441. Same roots historical K8: CTA34, DIRV2(original, not DIRL)35, END33, DINO33, P028, GEOM840. Remaining shards cancelled; no population generalization or complete100-root result. K64 and K8 old trajectories are not exact matches: saved kscale reference has 2,698 common decisions and only .4318 of full K8 banks identical, although decision0 is identical. Do not describe cross-run candidate prefixes as universally exact.

R2 source: `cta_r2_compare_54934/comparison.json`, summary `docs/CTA_ROUND2_OFFLINE_RESULT_54933.md`; seed0 offline development continuation lambda0 vs co-design lambda.1. Source CODE retention .706 [.668,.739]→.458 [.406,.508], paired Δ−.248 [−.297,−.200]. Source token perplexity131.197→1.371, sibling distinctness.7645→.2062, WM CE.5577→.0860 nats. Greedy predicted RG.293→.243, expected.501→.463 (both difference intervals cross0); direct.388 in both. Supports tested code collapse/lower prediction loss failing to preserve decision information, not a universal failure of co-design.

## Reproducibility evidence and repair scope

Sources: `cta_improvement_audit_56686/audit.json`; `cta_repro_review_56690/results/reproduction.json`; `cta_repro_review_56693/results/reproduction.json`; `cta_repro_logcontract_56709/logcheck.json`; ledger.

Same 200 L15 roots/weights: earlier56374 P0=100, GEOM8=152 vs56597 P0=103, GEOM8=149; P0 43/200 success flips and76 step mismatches, GEOM8 29/200 flips and93 step mismatches. Runtime/batched/eval sources compared byte-identically; grouping/scorer setup differed. Proposal batch shape is one demonstrated cause, not full causal accounting or simulator-restore failure.

Root2205 initial action maxima when grouping1→4/25/100: .056335/.037109/.035614 world units. Old alone/paired max coverage .507541/.938176 (both fail). Canonical alone/paired root2205 succeeds step233, coverage.952236. Production K1/8/16 prefixes exact; logged root/decision/t/chosen/geometry exact;4/128 coverage entries differ max3.33e−16 and pass atol1e−12/rtol0.56709 bounded_contract_pass=true.56693 remains FAILED1:0 because strict bitwise coverage assertion failed; CPU log reanalysis does not relabel it. Bounded P0/two-root logged-decision check does not certify full simulator state, all roots, learned scorers, hardware or backends. Canonical differs from old samples by .01733398 at initial bank: historical success cannot be silently attached to new proposal mode.

## Missing validation / claims to remove or label pending

No sealed400-root/three-training-seed result; no implemented/evaluated factorized direct control; no matched compact per-frame predictor or conditionality/end-only/equal-bits path ablations; no distinct-goal transfer. Current native-hit continuation is source-prepared only during DNS failure, with no checkpoint, GPU smoke, backward or closed-loop improvement result. Existing hit training is combined sampler+loss intervention, not isolated hit loss. Before strong claims: broad canonical/scorer invariance qualification, matched continuation/control evaluation and held-out independent roots/seeds; fair component accounting, precision/input comparison; leakage and restore checks retained. Any future compute requires both scheduler views and fresh month rankings, currently unavailable.

Source data differences: L8 banks r0=22,680/onpolicy=10,708/r4std=32,590/r4pert=32,590; L15 r4std=r4pert=18,606. L15 old sampler exposure57.225% perturbed, only5.557% mixed native-success; native-success bonus and hindsight both zero. Stage1 original jointly clips independent model groups, a fairness confound; separately trained matched direct helps but does not isolate all architecture/loss effects.
