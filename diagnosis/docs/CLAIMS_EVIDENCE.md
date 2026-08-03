# Claim → evidence map (paper assembly checklist)

> **Current validity note (2026-07-30):** use together with
> `CURRENT_STATUS.md`. Rows record artifacts, but an artifact is not automatically
> a defensible causal or held-out claim. In particular, C1 CRA/BB is
> observational (`hard_nn` negatives are cross-state), model-scaling cells are
> not yet evaluated on a shared physical effect mask/fixed distractor set, and
> the historical C7 Phase-H planning results may overlap predictor-LoRA training
> trajectories. Strict held-out replacements are now complete and recorded in
> C7, but remain offline supporting evidence. The TMLR submission uses the
> terminal-cost comparison and C5/C8 mechanism artifacts as its empirical spine.
> The TRM-style adaptation is excluded from the paper because its null result
> lacks sufficient adaptation-specific ranking and positive-control validation.

**Purpose:** every sentence the paper claims, with the exact artifact that backs
it. If a claim has no row here, it does not go in the paper. Companion to
`CURRENT_STATUS.md` (status/claim discipline) and `../../paper/main.tex` (draft).

Status legend: ✅ measured & in-repo · 🟡 measured, caveat carried · ⛔ blocked
from paper use pending a validity fix.

Sections C1--C3, C6, and C7 are historical/supporting ledgers and are excluded
from the current TMLR narrative unless `../../paper/main.tex` cites them
explicitly. In particular, old “in paper” annotations from the
Boundary-Blindness draft are superseded.

## C1 — The observational boundary diagnostic is localized and transfers

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1.1 | Frozen JEPA WMs show elevated cross-neighbourhood BB proxy at the pre-grasp boundary; this is not yet an exact-same-state bifurcation test | `results/metaworld_boundary.csv`: pooled bb_boundary pre_grasp **1.323 (dino) / 1.280 (jepa)** vs free_space 0.282/0.299 (~4.5×), contact 0.481/0.441; per-task CI-aware: elevated in 4/6 (dino), 5/6 (jepa), zero confident reversals | 🟡 observational proxy |
| 1.2 | The locus replicates across two model families | same CSV, both `dino_wm_metaworld` and `jepa_wm_metaworld` columns | ✅ |
| 1.3 | The phenomenon transfers to real-robot data | `results/droid_boundary.csv` (current): pre_grasp **1.916 [1.560, 2.279]** vs free_space 0.963 [0.793, 1.153] — CI-confident (the 1.975/0.721 figures were from a superseded run) | 🟡 ‖Δz‖ proxy, transfer-only (no object GT on DROID) |
| 1.4 | CounterfactualBench precursor: models rank factual vs *opposite* actions near-perfectly everywhere (CRA 0.96–0.99) but score much lower vs cross-state *nearest-neighbour* distractors, worst at pre-grasp (CRA 0.47–0.57) → motivates an observational follow-up | `results/metaworld_diagnostic.csv` pooled CRA_top1_eff: DINO opp free/pre/contact 0.990/0.961/0.975 vs nn 0.566/**0.467**/0.481; JEPA opp 0.993/0.972/0.984 vs nn 0.635/**0.541**/0.571; `droid_diagnostic.csv` hard_nn near `1/17` | 🟡 observational; nominal chance is not a calibrated causal null |
| 1.5 | Boundary label is an object-displacement proxy, not contact GT | stated in `boundary_gate_report.md`; mw-door-close excluded as proxy anomaly | 🟡 carried on figure |
| 1.6 | No favorable scale trend is observed under the current model-native DROID diagnostic across a 45× encoder scale-up and two families, including V-JEPA2-AC. This does **not** rule out scale because each model currently defines its own latent effect mask and neighbour set | `results/droid_scaling_curve.md`, `results/vjepa2_ac_droid_completion.md` (+ `droid_diagnostic.csv`, `droid_boundary.csv`): hard_nn effect-CRA near 0.0625 for all 4 models (22M/300M/1B/1B-OSS) in pre_grasp/gripper/contact; `bb_boundary` >0 every cell, pre_grasp locus 3/4 (+1.2 to +1.9) | 🟡 DROID transfer only; requires shared physical mask and fixed distractors for an apples-to-apples scaling claim |
| 1.7 | The metric is not saturated-low by construction (positive control) | `results/{pusht,point_maze,wall}_diagnostic.csv`: toy fully-actuated datasets score eff-CRA **0.66–1.0** → near-chance contact scores are real failures | ✅ |
| 1.8 | The pre-grasp locus is robust to the regime-threshold cut (not an artifact of 5mm/10cm/0.10): swept object-move∈{2.5,10}mm, pre-grasp-dist∈{8,12}cm, gripper-delta∈{0.05,0.20} × baseline (7 configs); pre-grasp effect-CRA moves ≤0.037 (DINO [0.450,0.488], JEPA [0.529,0.558]) and stays the lowest-scoring regime in all 7/7 configs, both models | `results/regime_robust_{base,objmove2p5,objmove10,pregrasp8,pregrasp12,gripdelta05,gripdelta20}.csv` + `results/regime_robustness_summary.csv` (scripts/04 env-override + scripts/46, job 26110, completed 3h33m); baseline config reproduces the historical draft table exactly (DINO 0.466≈0.467, JEPA 0.541=0.541) | ✅ |

## C2 — BB is associated with planning failure (regime-level, not causal)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 2.1 | Where BB is low, planning works — and our harness reproduces the published baseline | **D.2 strict re-score** (`results/metaworld_reach_strict.csv`, episode-end judging, same seeds): mw-reach L2 **6/16 = 37.5%** [Wilson 18.5–61.4] vs paper Table 1 DWM CEM-L2 **44.8 ± 8.9 → [35.9, 53.7]** — inside the CI; grounded hdyn 8/16 = 50.0% [28.0–72.0] | ✅ harness reproduces (does not beat) the published number. The prior "94%" used an any-step success latch (still in `metaworld_closed_loop.csv` as the `success` col); retracted as the headline. Paper averages 3 seeds × 96 ep; we use the released ckpt × 16 ep — same ballpark, not seed-matched. |
| 2.2 | Where BB is high, planning collapses with the predicted signature | same CSV: mw-push & mw-pick-place **0/16 both arms**; final ee 2–4 cm (arm arrives) vs state-dist ~0.5–0.6 (object unmoved) | ✅ |
| 2.3 | The contact collapse is unlikely to be explained by a globally broken harness | reach reproduces the paper (episode-end 37.5%, inside 44.8±8.9 — harness healthy) + render fidelity verified: one-step pred err **1.4×** dataset, latent NN ratio **0.97** (`results/closed_loop_report.md`, probes `_baseline_probe/_camera_calib/_replay_check`) | ✅ scoped diagnostic control |
| 2.4 | The field shows the same wall | jepa-success appendix: "hallucinates grasping the object", closed-loop MW tables stop at Reach/Reach-Wall; V-JEPA-2-AC Table 2: pick-&-place only with **3 hand-crafted sub-goal images** (65–80%), grasp-with-goal-image 25–65%; Octo grasp 0–20% | ✅ (citations) |
| 2.5 | Per-transition CRA_eff ↔ Action Error correlation is null — the failure is the bifurcation, not one-step action-ignoring | planning probe run (HANDOFF_DROID §8; class imbalance ~4% positives) | ✅ (negative result, kept) |
| 2.6 | The model-side fix improves the closed-loop cost surface | paired Δ(l2−hdyn) final state-dist, pooled contact (n=32): **+0.089 [bootstrap +0.022, +0.162]**; per-task: pick-place +0.081 [+0.007, +0.160], push +0.097 [−0.014, +0.221]; no-harm on reach | ✅ |
| 2.7 | …but flips no successes → reducing this grounding/BB proxy is insufficient for contact planning; the residual cause was not isolated by this intervention | 0/16 hdyn on both contact tasks despite 2.6 | ✅ null result; no necessity claim |

## C3 — The fix ladder: what fails, what works, why

| # | Claim | Evidence | Status |
|---|---|---|---|
| 3.1 | Head-level mixture predictors (NLL/WTA/hard-EM/supervised-mode, K∈{2,3}) do NOT reduce BB | `results/metaworld_boundary_fix.csv`, `_fix_nll.csv`: all ≈ frozen base | ✅ quantified null |
| 3.2 | Cause (a): the bifurcation is ~9% of the residual in latent L2 geometry | supervised-variant conditional means differ 9.9 L2 vs 106 median residual | ✅ |
| 3.3 | Cause (b): boundary *action*-dependence unlearnable from expert-only data at head level | π action-flip rate ≈ 0; CE 0.487 vs 0.562 base-rate | ✅ |
| 3.4 | Metric-only re-weighting (φ over probe subspace) does NOT fix BB | `results/metaworld_boundary_metric.csv`: redistributes, doesn't reduce | ✅ kills "just fix the metric" |
| 3.5 | The information IS in the latent (V1) and propagates for factual actions (V2); the broken piece is the counterfactual action→object channel (V3) | probe chain: V1 err 0.064 vs sd 0.094 ✓; V2 0.059 ✓; V3 counterfactual spread corr **+0.035** ✗ | ✅ |
| 3.6 | A 0.5M-param grounded dynamics channel h(z,a)→Δobject (frozen everything, cache-only) restores counterfactual tracking | corr **+0.682** (dino), **+0.702** (jepa) — ~20× the frozen predictor | ✅ |
| 3.7 | …and halves BB at the boundary, on both models | `metaworld_boundary_dynamics.csv` (dino): pre_grasp 1.323→**0.660** (−50%), gap 1.04→0.32; `_dynamics_jepa.csv`: 1.280→**0.620** (−52%), gap 0.98→0.27 | ✅ |
| 3.8 | The comparison supports the interpretation that explicit object-displacement supervision exposes signal hidden by full-latent L2; it does not prove that data coverage is generally sufficient | 3.6 succeeded on the same cached expert data that 3.1/3.3 failed on | 🟡 interpretation, not unique causal mechanism |
| 3.9 | The recipe does NOT transfer to DROID's noisy whole-state proxy label at 2.1k transitions | `droid_boundary_dynamics.csv`: val MSE 0.865, rank-corr +0.708 but spread magnitude collapsed (0.0038 vs 115), BB not reduced (pre_grasp 1.85 vs 1.98 base, free_space worse) | ✅ honest negative; scope: the *label*, not the principle |
| 3.10 | Open-loop Action Error: grounded cost no-harm/no-gain (metric rewards arm mimicry) | `metaworld_planning_metric.csv`; scale-bug run preserved `_buggy_scale.csv` | ✅ historical artifact |

## Reproduction integrity (the credibility section)

| # | Claim | Evidence | Status |
|---|---|---|---|
| R.1 | Closed-loop protocol is upstream-parity | `scripts/18_closed_loop_eval.py` header: shipped config (H=6, 300×15 CEM, nas=3, α=0, warmup, horizon shrink, expert-final goal) | ✅ |
| R.2 | Three env-side reproduction bugs found+fixed (default-camera 480px renderer; training data = flipud(corner2+tweak), MSE 71.6 vs ≥3000 unflipped; goal = expert FINAL frame) | `results/closed_loop_report.md` §pitfalls (narrative + numbers). ⚠️ 2026-07-12 audit: the previously-cited `results/logs/camera_calib*` artifacts and `_baseline_probe/_camera_calib/_replay_check` probe scripts are NOT on disk — evidence rests on the report; regenerate probes before claiming committed artifacts | 🟡 report-backed only |
| R.3 | Physics/world identical to data-gen (only visuals differed) | `_replay_check.py`: action replay ee err median 1.5 mm | ✅ |

## C4 — Terminal-cost comparison: learned-dynamics error is not necessary for failure

| # | Claim | Evidence | Status |
|---|---|---|---|
| 4.1 | Fresh paired 64-seed terminal-cost comparison: same CEM budget + exact simulator candidate dynamics + privileged state-reference cost solves push **64/64** and pick-place **49/64** | `results/trm_heldout_oracle_*_seed30000_n64.csv`; `../../paper/main.tex`, Table 1 | ✅ headline; strict `success_end` |
| 4.2 | With exact simulator candidate rollouts and the same CEM budget, latent L2 solves **0/64** in all four checkpoint×task cells; stateprobe solves DINO push/pick **5/64, 0/64** and JEPA **1/64, 1/64** | `results/trm_heldout_{dino,jepa}_wm_metaworld_{l2,stateprobe}_*_seed30000_n64.csv`; raw endpoint counts cross-checked against Table 1 | ✅ headline; detailed mechanism below applies only to stateprobe |
| 4.3 | Object-only decoding is stronger than other cost components, not a certificate of full cost accuracy: off-policy object median **2.01cm, 91.5% <5cm**; end-effector median **5.14cm, 45.8% <5cm**; hand−object relative median **11.18cm, 3.7% <5cm** | `results/encoder_info_upperbound.md`, job `26485` | ✅ scoped; no encoder-only attribution |
| 4.4 | Off-policy hardening improves the object-only metric (78.3→**91.5%** <5cm) but the historical n=16 planning gate remains **1/16**; use only as an insufficiency result, superseded for headline success by row 4.2 | `results/oracle_ladder_cost_report.md` §Phase-3 | 🟡 supporting |

## C5 — Optimizer-conditioned cost misranking

| # | Claim | Evidence | Status |
|---|---|---|---|
| 5.1 | Random off-policy→elite comparison shows distribution shift: elite decode **24.0% <5cm** (final 19.2%; median 7.3cm) vs random off-policy **91.5%** (median 2.0cm). Because populations are unmatched, this is supporting evidence, not the primary selection test | `results/cem_exploit_precision.csv` | 🟡 supporting |
| 5.2 | Probe-minimized plans read "at goal" while the true object is 13–30cm away | `oracle_ladder_cost_report.md` §Verdict | ✅ |
| 5.3 | Historical search-budget sensitivity (n=16, 3 tasks): reference-aligned reach cost converts search→success **5/16→16/16→16/16→16/16** at iterations {2,6,12,24}; contact success stays at most 2/16. The separate population axis {50,100,300}@12it is n=8 and stays at most 1/8. Within-search point-estimate curves are weaker: push×stateprobe proxy falls 18% while the reference outcome falls 2% and decode error rises **6.6 [5.7,7.4]→8.3 [7.1,9.3] cm**, but the decode CIs overlap | `results/overopt_{curves,episodes}_mwpush_mwreach_n16.csv`, `_mwpickplace_n16.csv`, `_mwpush_nsamp.csv`, and `results/overopt_analysis.md` | 🟡 endpoint sensitivity included in paper Appendix C; within-search trend remains excluded from the causal headline |
| 5.4 | Probe-independent simulator-state cross-check: final-iteration stateprobe-selected elites' true object-to-goal median is **24.9cm** (push 19.9, pick 28.0), with **1.7% <5cm** | `results/exploit_simstate_crosscheck.csv` (scripts/45, offline re-analysis of `cem_exploit_lite.npz`; no GPU) | 🟡 supporting; excluded from current paper |
| 5.5 | On the identical first CEM population, proxy elites have **1.24–1.40cm** more object error than population, **0.91–1.06cm** more than true-cost elites, optimism enrichment **2.16–2.33cm**, and selected physical regret **2.05–2.47cm**; all relevant seed-clustered CIs exclude zero | `results/cem_preselection_audit.md`, candidate dumps, script 53 | ✅ primary same-population evidence |
| 5.6 | Matched-residual permutation null preserves the exact within-population residual multiset but breaks candidate association: actual argmin regret **2.05–2.48cm** vs null **1.07–1.15cm**; excess **0.91–1.36cm**, all four CIs exclude zero | `results/cem_residual_permutation_null.md`, script 56, job `28322` | ✅ structured alignment beyond generic noisy-score winner's curse |
| 5.7 | Shared-population branch changes only the refitting cost after identical initial candidates; after five refits, proxy branch best physical candidate is **1.51–2.01cm** worse in all four checkpoint×task cells, all CIs exclude zero; the seed-level mean contrast is positive in **8/8** seeds in every cell (two-sided exact sign test **p=0.0078** per cell) | shared-branch candidate/metadata files and script 55 analysis | ✅ adaptive intervention |
| 5.8 | Exact stateprobe checkpoints have held-out expert object coordinate RMSE **1.64–1.71cm** and hand RMSE **3.75–4.31cm**. On optimizer-induced candidates, initial stateprobe/reference cost Spearman is **0.43–0.55** with reference top-10 recall **0.19–0.26**; final-population Spearman is **0.11–0.16** with recall **0.07–0.10**. Each cell/stage uses 112 populations, 11,200 candidates, and 16 seed clusters | `results/stateprobe_cem_validation{,_expert_validation,_summary}.csv`, report, script 63, job `33076` | ✅ fixed-probe validation; no architecture/training-seed sensitivity claim |

## C6 — Mitigation nulls: tested post-hoc repairs do not cross contact

| # | Claim | Evidence | Status |
|---|---|---|---|
| 6.1 | Relearned representation adapter φ (v2, rebalanced): held-out decode median 6.9cm, re-gate push **1/16**, pick 0/16 | phaseC_repr_v2 log (job 24018), `metaworld_latent_oracle_phi*.csv` | ✅ |
| 6.2 | Encoder-LoRA + φ, 5 seeds: push-held **{5,0,2,1,1}/16, mean 1.8, CI [0, 3.7]** = inside frozen baseline 0–2/16; grounding 94–100% <5cm across seeds → grounding↑↛planning at the encoder level | phaseD/phaseF logs (jobs 24128, 24270/24276/24299 + r1), seed-sweep summary | ✅ |
| 6.3 | Ensemble disagreement penalty (5 LoRA seeds, cost = mean_k d_k² + λ·Var_k): λ∈{0,0.5,1,2} → push-held **{2,1,2,1}/16** = frozen noise (the {2,2,2,1} variant was the non-held eval — do not reintroduce); the standard MBRL pessimism fix fails because seeds share the frozen-base blind spot | phaseG logs (jobs 24340–24357), `results/ladder_mitigation_cis.csv` | ✅ historical artifact |
| 6.4 | phil2 hybrid (β=0.15) = minor generality win only (one cost: reach 16/16 + push 2/16), not a crossing | phaseD_phil2_gate log | ✅ |
| 6.5 | Per-arm Wilson 95% CIs (mw-push, success_end): every frozen-base arm upper bound ≤3.1/16 for n=16 rungs (l2/phi/stateprobe_robust [0,19.4%]); the 5/16 LoRA seed's CI [14,56%] overlaps every other arm → "inside the band" stated with intervals, not point estimates | `results/ladder_mitigation_cis.csv` (scripts/45, offline; no GPU) | ✅ |

## C7 — The predictor axis: action-counterfactual objective (Phase H, DROID)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 7.1 | Historical CF InfoNCE-over-actions predictor-LoRA training metric on dino_wm_droid; not a held-out planning claim | old scripts/40 log / ckpt meta (`predictor_cf_dino_wm_droid.pt`) | 🟡 historical only; replacement protocol now persists immutable 70/15/15 manifest |
| 7.2 | Strict held-out DINO-WM four-seed result: AE **1.492→1.085**, AS **0.471→0.545**, CRA **0.036→0.233**; every seed improves all three pooled metrics | `results/cf_heldout_dino_wm_droid_summary.md`, jobs `26493/26495` | 🟡 offline supporting only; no task-success claim |
| 7.3 | Strict held-out JEPA-WM result is mixed: CRA **0.030→0.085**, AE **1.337→1.344**, AS **0.501→0.501** | `results/cf_heldout_jepa_wm_droid_summary.md`, jobs `26494/26495` | 🟡 blocks a general method contribution |

## C8 — Elite-conditioned audit components

The old single "exploitation gap" in `scripts/47_exploitation_gap_ladder.py`
conflated probe distribution shift, absolute task outcome, and opportunity
regret. It is retired for headline use. The replacement protocol reports three
separate estimands and refuses to infer candidate regret from legacy files that
did not persist the best true candidate.

| # | Claim | Evidence | Status |
|---|---|---|---|
| 8.1 | Elite readout shift is measured separately from task truth; $91.5\%\to24\%$ within 5cm remains evidence of selection-induced probe degradation, not task regret | `results/cem_exploit_precision.csv`, script 35 | 🟡 measured on existing elite dumps |
| 8.2 | Simulator outcome and opportunity regret are measured as true selected cost minus best true cost among candidates actually present | preselection row 5.5 and `results/exploitation_components.md`; jobs `26502/26746` | ✅ |
| 8.3 | Within-search corruption and candidate-order inversion are measured, but same-population preselection regret and the branch intervention are the cleaner headline estimands | same artifacts plus rows 5.5–5.7 | ✅ supporting |
| 8.4 | Exact-success candidate availability in the final proxy-guided population: DINO stateprobe push **8.0%**, pick **3.6%**; JEPA stateprobe contact cells **0%**. Positive physical regret persists. This does **not** independently identify proposal failure because earlier proxy refits, the six-step horizon, and encountered snapshots jointly determine the population | `results/oracle_coverage_selection.md`, candidates from job `26505`; corrected analysis artifact dated 2026-07-14 | ✅ scoped availability result |
| 8.5 | TRM-style replacement/hybrid comparison was executed on held-out contact seeds, two checkpoints, and three head seeds. Analyzer summaries use strict endpoint field `success_end`. The empirical result is excluded from the TMLR paper because adaptation-specific ranking/training and positive-control checks are insufficient to distinguish a method failure from an implementation/adaptation failure | `scripts/52_analyze_trm.py`, `results/trm_heldout_*`, jobs `26507--26509`; pooling approximation means this is not an exact reproduction | 🟡 artifact only; excluded from paper |
| 8.6 | ACID-style adaptive inverse-consistency cost under learned versus oracle dynamics | ACID protocol and jobs `26510--26512`; deterministic pooled-IDM approximation because no official verifier is released | ⛔ pending |

## Optional strengtheners

| # | Item | Status |
|---|---|---|
| D.3 | Imagined-rollout object-error table (baseline vs +h, cache-only, ~2 h) | optional C3 strengthener: quantifies "hallucinated grasping" along full imagined rollouts |
| E.1 | Amortized GC-IDM control (removes test-time search; `inverse_proposal` ckpt exists) | historical optional experiment |
