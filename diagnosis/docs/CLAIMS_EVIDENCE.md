# Claim → evidence map (paper assembly checklist)

> **Current validity note (2026-07-13):** use together with
> `CURRENT_STATUS.md`. Rows record artifacts, but an artifact is not automatically
> a defensible causal or held-out claim. In particular, C1 CRA/BB is
> observational (`hard_nn` negatives are cross-state), model-scaling cells are
> not yet evaluated on a shared physical effect mask/fixed distractor set, and
> the historical C7 Phase-H planning results may overlap predictor-LoRA training
> trajectories. The split leak is fixed in code, but C7 remains blocked until
> held-out jobs `26493`--`26495` finish. Resumed generality jobs `26481/26482`,
> encoder probe `26485`, same-state job `26497`, and shared-scaling job `26498`
> must not be recorded as successful before their final artifacts land.

**Purpose:** every sentence the paper claims, with the exact artifact that backs
it. If a claim has no row here, it does not go in the paper. Companion to
`CURRENT_STATUS.md` (status/claim discipline) and `../../paper/main.tex` (draft).

Status legend: ✅ measured & in-repo · 🟡 measured, caveat carried · ⛔ blocked
from paper use pending a validity fix.

## C1 — The observational boundary diagnostic is localized and transfers

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1.1 | Frozen JEPA WMs show elevated cross-neighbourhood BB proxy at the pre-grasp boundary; this is not yet an exact-same-state bifurcation test | `results/metaworld_boundary.csv`: pooled bb_boundary pre_grasp **1.323 (dino) / 1.280 (jepa)** vs free_space 0.282/0.299 (~4.5×), contact 0.481/0.441; per-task CI-aware: elevated in 4/6 (dino), 5/6 (jepa), zero confident reversals | 🟡 observational proxy |
| 1.2 | The locus replicates across two model families | same CSV, both `dino_wm_metaworld` and `jepa_wm_metaworld` columns | ✅ |
| 1.3 | The phenomenon transfers to real-robot data | `results/droid_boundary.csv` (current): pre_grasp **1.916 [1.560, 2.279]** vs free_space 0.963 [0.793, 1.153] — CI-confident (the 1.975/0.721 figures were from a superseded run; paper uses current CSV) | 🟡 ‖Δz‖ proxy, transfer-only (no object GT on DROID) |
| 1.4 | CounterfactualBench precursor: models rank factual vs *opposite* actions near-perfectly everywhere (CRA 0.96–0.99) but score much lower vs cross-state *nearest-neighbour* distractors, worst at pre-grasp (CRA 0.47–0.57) → motivates an observational follow-up. **In paper Table 2 (Sec 3.1).** | `results/metaworld_diagnostic.csv` pooled CRA_top1_eff: DINO opp free/pre/contact 0.990/0.961/0.975 vs nn 0.566/**0.467**/0.481; JEPA opp 0.993/0.972/0.984 vs nn 0.635/**0.541**/0.571; `droid_diagnostic.csv` hard_nn near `1/17` | 🟡 observational; nominal chance is not a calibrated causal null |
| 1.5 | Boundary label is an object-displacement proxy, not contact GT | stated in `boundary_gate_report.md`; mw-door-close excluded as proxy anomaly | 🟡 carried on figure |
| 1.6 | No favorable scale trend is observed under the current model-native DROID diagnostic across a 45× encoder scale-up and two families, including V-JEPA2-AC. This does **not** rule out scale because each model currently defines its own latent effect mask and neighbour set. **In paper Sec 3.6 (Table `tab:scaling`).** | `results/droid_scaling_curve.md`, `results/vjepa2_ac_droid_completion.md` (+ `droid_diagnostic.csv`, `droid_boundary.csv`): hard_nn effect-CRA near 0.0625 for all 4 models (22M/300M/1B/1B-OSS) in pre_grasp/gripper/contact; `bb_boundary` >0 every cell, pre_grasp locus 3/4 (+1.2 to +1.9) | 🟡 DROID transfer only; requires shared physical mask and fixed distractors for an apples-to-apples scaling claim |
| 1.7 | The metric is not saturated-low by construction (positive control) | `results/{pusht,point_maze,wall}_diagnostic.csv`: toy fully-actuated datasets score eff-CRA **0.66–1.0** → near-chance contact scores are real failures | ✅ |
| 1.8 | The pre-grasp locus is robust to the regime-threshold cut (not an artifact of 5mm/10cm/0.10): swept object-move∈{2.5,10}mm, pre-grasp-dist∈{8,12}cm, gripper-delta∈{0.05,0.20} × baseline (7 configs); pre-grasp effect-CRA moves ≤0.037 (DINO [0.450,0.488], JEPA [0.529,0.558]) and stays the lowest-scoring regime in all 7/7 configs, both models | `results/regime_robust_{base,objmove2p5,objmove10,pregrasp8,pregrasp12,gripdelta05,gripdelta20}.csv` + `results/regime_robustness_summary.csv` (scripts/04 env-override + scripts/46, job 26110, completed 3h33m); baseline config reproduces the paper table exactly (DINO 0.466≈0.467, JEPA 0.541=0.541) | ✅ |

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
| 3.10 | Open-loop Action Error: grounded cost no-harm/no-gain (metric rewards arm mimicry) | `metaworld_planning_metric.csv`; scale-bug run preserved `_buggy_scale.csv` | ✅ disclosed |

## Reproduction integrity (the credibility section)

| # | Claim | Evidence | Status |
|---|---|---|---|
| R.1 | Closed-loop protocol is upstream-parity | `scripts/18_closed_loop_eval.py` header: shipped config (H=6, 300×15 CEM, nas=3, α=0, warmup, horizon shrink, expert-final goal) | ✅ |
| R.2 | Three env-side reproduction bugs found+fixed (default-camera 480px renderer; training data = flipud(corner2+tweak), MSE 71.6 vs ≥3000 unflipped; goal = expert FINAL frame) | `results/closed_loop_report.md` §pitfalls (narrative + numbers). ⚠️ 2026-07-12 audit: the previously-cited `results/logs/camera_calib*` artifacts and `_baseline_probe/_camera_calib/_replay_check` probe scripts are NOT on disk — evidence rests on the report; regenerate probes before claiming committed artifacts | 🟡 report-backed only |
| R.3 | Physics/world identical to data-gen (only visuals differed) | `_replay_check.py`: action replay ee err median 1.5 mm | ✅ |

## C4 — Oracle ladder: the wall is the cost, not the predictor/planner/budget

| # | Claim | Evidence | Status |
|---|---|---|---|
| 4.1 | Same CEM budget + perfect dynamics + true-state cost solves contact: push **16/16**, pick 11/16 | `results/oracle_ladder_cost_report.md` (scripts/29 state-oracle) | ✅ |
| 4.2 | Every cost computed through the frozen encoder collapses under the SAME perfect dynamics: l2 0/16, gobj 0/8, learned metric 0/8, stateprobe 2/16 | same report (scripts/30 rungs) | ✅ |
| 4.3 | Static readout precision is NOT the wall: spatial probe decodes object 92% <5cm (contact 90.7%) | same report, Test-1b (scripts/21) | ✅ |
| 4.4 | Off-policy readout precision is NOT the wall: probes hardened on off-policy frames (obj 78→**91.5%** <5cm, median 2.0cm on the frames CEM scores) re-gate at push **1/16** — a real, measured readout repair transfers ZERO planning success | same report §Phase-3 (3a/3b, job 23553) | ✅ |

## C5 — The planner is the adversary (cost overoptimization / Goodhart)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 5.1 | CEM converges its elite population onto the cost's residual-error pockets: elite decode **24.0% <5cm (final-iter 19.2%, push 15.3%; median 7.3cm)** vs **91.5% (median 2.0cm)** on random off-policy frames, same probes | `results/cem_exploit_precision.csv` (scripts/35, Phase A) | ✅ |
| 5.2 | The exploited plans read "at goal" while the true object is 13–30cm away | `oracle_ladder_cost_report.md` §Verdict | ✅ |
| 5.3 | Overoptimization curves (n=16, 3 tasks): honest cost (reach×l2) converts search→success **5/16→16/16→16/16→16/16** at iters {2,6,12,24}; exploitable (push×stateprobe) proxy −18% / true −2% / decode **6.6 [5.7,7.4]→8.3 [7.1,9.3] cm (+25%)** with pick corroborating (+21%); no-signal (push×l2) proxy −35%, obj_med **exactly flat 0.239m** every budget. **Decode growth is a point-estimate trend — first/last CIs OVERLAP at n=16** (all 6 arms), disclosed as such in the paper. Population axis {50,100,300}@12it is a separate **n=8** run (0.233/0.207/0.252m; l2 flat 0.252) | `results/overopt_{curves,episodes}_mwpush_mwreach_n16.csv`, `_mwpickplace_n16.csv`, `_mwpush_nsamp.csv` + `results/overopt_analysis.md` (scripts/41+42); figure `paper/figures/figure_overopt.pdf` regenerated from these | ✅ n=16 final; CI-overlap caveat carried in text+caption (v0.3) |
| 5.4 | Exploitation confirmed on SIM STATE (probe-independent): final-iter elites' TRUE object→goal median **24.9cm** (push 19.9, pick 28.0), only **1.7% <5cm** — worse than the probe's 19.2%, so the probe number understates the wall. Closes the "probe-OOD vs object-really-far" confound | `results/exploit_simstate_crosscheck.csv` (scripts/45, offline re-analysis of `cem_exploit_lite.npz`; no GPU) | ✅ |

## C6 — Mitigation nulls: exploitation is irreducible for frozen-base costs

| # | Claim | Evidence | Status |
|---|---|---|---|
| 6.1 | Relearned representation adapter φ (v2, rebalanced): held-out decode median 6.9cm, re-gate push **1/16**, pick 0/16 | phaseC_repr_v2 log (job 24018), `metaworld_latent_oracle_phi*.csv` | ✅ |
| 6.2 | Encoder-LoRA + φ, 5 seeds: push-held **{5,0,2,1,1}/16, mean 1.8, CI [0, 3.7]** = inside frozen baseline 0–2/16; grounding 94–100% <5cm across seeds → grounding↑↛planning at the encoder level | phaseD/phaseF logs (jobs 24128, 24270/24276/24299 + r1), seed-sweep summary | ✅ |
| 6.3 | Ensemble disagreement penalty (5 LoRA seeds, cost = mean_k d_k² + λ·Var_k): λ∈{0,0.5,1,2} → push-held **{2,1,2,1}/16** = frozen noise (this is the number in `ladder_mitigation_cis.csv` and in paper Table tab:mitigations since v0.3; the {2,2,2,1} variant was the non-held eval — do not reintroduce); the standard MBRL pessimism fix fails because seeds share the frozen-base blind spot | phaseG logs (jobs 24340–24357), `results/ladder_mitigation_cis.csv` | ✅ |
| 6.4 | phil2 hybrid (β=0.15) = minor generality win only (one cost: reach 16/16 + push 2/16), not a crossing | phaseD_phil2_gate log | ✅ |
| 6.5 | Per-arm Wilson 95% CIs (mw-push, success_end): every frozen-base arm upper bound ≤3.1/16 for n=16 rungs (l2/phi/stateprobe_robust [0,19.4%]); the 5/16 LoRA seed's CI [14,56%] overlaps every other arm → "inside the band" stated with intervals, not point estimates | `results/ladder_mitigation_cis.csv` (scripts/45, offline; no GPU) | ✅ |

## C7 — The predictor axis: action-counterfactual objective (Phase H, DROID)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 7.1 | Historical CF InfoNCE-over-actions predictor-LoRA training metric on dino_wm_droid; not a held-out planning claim | old scripts/40 log / ckpt meta (`predictor_cf_dino_wm_droid.pt`) | 🟡 historical only; replacement protocol now persists immutable 70/15/15 manifest |
| 7.2 | Exploratory A/B planning probe (n=157): pooled Action-Error **1.468→1.086** (−26%), Action-Score 0.466→0.525, effect-CRA **0.061→0.605**; AE better in all 6 regime×horizon cells, AS in 5/6 | `results/droid_planning_cf_dino_wm_droid_{frozen,lora}.csv`; current probe can read the full cache, including possible train trajectories | ⛔ blocked from paper until test-only rerun |
| 7.3 | Exploratory seed/model hardening: 4 CF seeds and a second DROID model. Repetition did not remove the shared train/eval overlap. | historical `results/cf_seed_summary.md`, `droid_planning_cf_*_s{1,2,3}.csv`, `droid_planning_cf_jepa_wm_droid_*.csv` | ⛔ blocked; replacement jobs `26493/26494`, aggregation `26495` |

## C8 — Elite-conditioned audit components

The old single "exploitation gap" in `scripts/47_exploitation_gap_ladder.py`
conflated probe distribution shift, absolute task outcome, and opportunity
regret. It is retired for headline use. The replacement protocol reports three
separate estimands and refuses to infer candidate regret from legacy files that
did not persist the best true candidate.

| # | Claim | Evidence | Status |
|---|---|---|---|
| 8.1 | Elite readout shift is measured separately from task truth; $91.5\%\to24\%$ within 5cm remains evidence of selection-induced probe degradation, not task regret | `results/cem_exploit_precision.csv`, script 35 | 🟡 measured on existing elite dumps |
| 8.2 | Simulator outcome and opportunity regret: true selected cost minus best true cost among candidates actually present | new instrumentation in scripts 41/43; analyzer 50; protocol `plans/2026-07-13-exploitation-components-protocol.md` | ⛔ pending jobs `26502/26503` |
| 8.3 | Within-search corruption: proxy improves while simulator truth worsens, plus candidate-order inversion | same instrumented jobs and seed-clustered analyzer | ⛔ pending jobs `26502/26503` |
| 8.4 | IMWM discriminator: successful/physically better candidate coverage versus present-but-misranked selection error on identical oracle populations | scripts 51/52 and protocol `plans/2026-07-13-coverage-selection-protocol.md` | ⛔ pending jobs `26505/26506` |
| 8.5 | Direct TRM-style replacement/hybrid comparison on held-out contact seeds, two checkpoints and three head seeds | TRM protocol and jobs `26507--26509`; documented pooling approximation means this is not an exact reproduction | ⛔ pending |
| 8.6 | ACID-style adaptive inverse-consistency cost under learned versus oracle dynamics | ACID protocol and jobs `26510--26512`; deterministic pooled-IDM approximation because no official verifier is released | ⛔ pending |

## Optional strengtheners

| # | Item | Status |
|---|---|---|
| D.3 | Imagined-rollout object-error table (baseline vs +h, cache-only, ~2 h) | optional C3 strengthener: quantifies "hallucinated grasping" along full imagined rollouts |
| E.1 | Amortized GC-IDM control (removes the search adversary; `inverse_proposal` ckpt exists) | queued behind E0 (C5.3) |
