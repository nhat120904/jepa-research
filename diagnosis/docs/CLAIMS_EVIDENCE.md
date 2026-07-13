# Claim → evidence map (paper assembly checklist)

**Purpose:** every sentence the paper claims, with the exact artifact that backs
it. If a claim has no row here, it does not go in the paper. Companion to
`docs/PAPER_IDEA.md` (idea-of-record) and `paper/main.tex` (draft).

Status legend: ✅ measured & in-repo · 🟡 measured, caveat carried.

## C1 — Boundary Blindness is real, localized, and transfers

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1.1 | Frozen JEPA WMs fail to resolve contact bifurcations: BB concentrates at the pre-grasp boundary | `results/metaworld_boundary.csv`: pooled bb_boundary pre_grasp **1.323 (dino) / 1.280 (jepa)** vs free_space 0.282/0.299 (~4.5×), contact 0.481/0.441; per-task CI-aware: elevated in 4/6 (dino), 5/6 (jepa), zero confident reversals | ✅ |
| 1.2 | The locus replicates across two model families | same CSV, both `dino_wm_metaworld` and `jepa_wm_metaworld` columns | ✅ |
| 1.3 | The phenomenon transfers to real-robot data | `results/droid_boundary.csv` (current): pre_grasp **1.916 [1.560, 2.279]** vs free_space 0.963 [0.793, 1.153] — CI-confident (the 1.975/0.721 figures were from a superseded run; paper uses current CSV) | 🟡 ‖Δz‖ proxy, transfer-only (no object GT on DROID) |
| 1.4 | CounterfactualBench precursor: models rank factual vs *opposite* actions near-perfectly everywhere (CRA 0.96–0.99) but collapse vs *nearest-neighbour* distractors, worst at pre-grasp (CRA 0.47–0.57) → motivates BB. **In paper Table 2 (Sec 3.1).** | `results/metaworld_diagnostic.csv` pooled CRA_top1_eff: DINO opp free/pre/contact 0.990/0.961/0.975 vs nn 0.566/**0.467**/0.481; JEPA opp 0.993/0.972/0.984 vs nn 0.635/**0.541**/0.571; `droid_diagnostic.csv` hard_nn at 16-way chance floor | ✅ |
| 1.5 | Boundary label is an object-displacement proxy, not contact GT | stated in `boundary_gate_report.md`; mw-door-close excluded as proxy anomaly | 🟡 carried on figure |
| 1.6 | The failure does not scale on DROID: action-grounding + BB persist across a 45× encoder scale-up and two families, including V-JEPA2-AC → rules out "just scale the encoder". **In paper Sec 3.6 (Table `tab:scaling`).** | `results/droid_scaling_curve.md`, `results/vjepa2_ac_droid_completion.md` (+ `droid_diagnostic.csv`, `droid_boundary.csv`): hard_nn effect-CRA at the 0.0625 chance floor for all 4 models (22M/300M/1B/1B-OSS) in pre_grasp/gripper/contact; `bb_boundary` >0 every cell, pre_grasp locus 3/4 (+1.2 to +1.9) | ✅ DROID transfer diagnostic only: ‖Δz‖ proxy, no object GT, no grounded-fix transfer, no closed-loop V-JEPA2 claim; deterministic hard_nn + CIs; random/opposite unseeded (drift run-to-run) — not reported |
| 1.7 | The metric is not saturated-low by construction (positive control) | `results/{pusht,point_maze,wall}_diagnostic.csv`: toy fully-actuated datasets score eff-CRA **0.66–1.0** → near-chance contact scores are real failures | ✅ |
| 1.8 | The pre-grasp locus is robust to the regime-threshold cut (not an artifact of 5mm/10cm/0.10): swept object-move∈{2.5,10}mm, pre-grasp-dist∈{8,12}cm, gripper-delta∈{0.05,0.20} × baseline (7 configs); pre-grasp effect-CRA moves ≤0.037 (DINO [0.450,0.488], JEPA [0.529,0.558]) and stays the lowest-scoring regime in all 7/7 configs, both models | `results/regime_robust_{base,objmove2p5,objmove10,pregrasp8,pregrasp12,gripdelta05,gripdelta20}.csv` + `results/regime_robustness_summary.csv` (scripts/04 env-override + scripts/46, job 26110, completed 3h33m); baseline config reproduces the paper table exactly (DINO 0.466≈0.467, JEPA 0.541=0.541) | ✅ |

## C2 — BB ⇄ planning failure (regime-level, closed-loop)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 2.1 | Where BB is low, planning works — and our harness reproduces the published baseline | **D.2 strict re-score** (`results/metaworld_reach_strict.csv`, episode-end judging, same seeds): mw-reach L2 **6/16 = 37.5%** [Wilson 18.5–61.4] vs paper Table 1 DWM CEM-L2 **44.8 ± 8.9 → [35.9, 53.7]** — inside the CI; grounded hdyn 8/16 = 50.0% [28.0–72.0] | ✅ harness reproduces (does not beat) the published number. The prior "94%" used an any-step success latch (still in `metaworld_closed_loop.csv` as the `success` col); retracted as the headline. Paper averages 3 seeds × 96 ep; we use the released ckpt × 16 ep — same ballpark, not seed-matched. |
| 2.2 | Where BB is high, planning collapses with the predicted signature | same CSV: mw-push & mw-pick-place **0/16 both arms**; final ee 2–4 cm (arm arrives) vs state-dist ~0.5–0.6 (object unmoved) | ✅ |
| 2.3 | The collapse is a *model* failure, not a harness failure | reach reproduces the paper (episode-end 37.5%, inside 44.8±8.9 — harness healthy) + render fidelity verified: one-step pred err **1.4×** dataset, latent NN ratio **0.97** (`results/closed_loop_report.md`, probes `_baseline_probe/_camera_calib/_replay_check`) | ✅ |
| 2.4 | The field shows the same wall | jepa-success appendix: "hallucinates grasping the object", closed-loop MW tables stop at Reach/Reach-Wall; V-JEPA-2-AC Table 2: pick-&-place only with **3 hand-crafted sub-goal images** (65–80%), grasp-with-goal-image 25–65%; Octo grasp 0–20% | ✅ (citations) |
| 2.5 | Per-transition CRA_eff ↔ Action Error correlation is null — the failure is the bifurcation, not one-step action-ignoring | planning probe run (HANDOFF_DROID §8; class imbalance ~4% positives) | ✅ (negative result, kept) |
| 2.6 | The model-side fix improves the closed-loop cost surface | paired Δ(l2−hdyn) final state-dist, pooled contact (n=32): **+0.089 [bootstrap +0.022, +0.162]**; per-task: pick-place +0.081 [+0.007, +0.160], push +0.097 [−0.014, +0.221]; no-harm on reach | ✅ |
| 2.7 | …but flips no successes → BB is necessary-not-sufficient; the residual bottleneck is contact-creating action proposal (planner-side, future work) | 0/16 hdyn on both contact tasks despite 2.6 | ✅ (one-sentence future work) |

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
| 3.8 | The bottleneck was the training target (full-latent L2), not the data — cross-sample neighbourhood variation suffices | 3.6 succeeded on the same cached expert data that 3.1/3.3 failed on | ✅ |
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
| 7.1 | CF InfoNCE-over-actions predictor-LoRA (0.469M params, no object GT) on dino_wm_droid: cf_rank_acc 0.195→0.736, recon 0.695× (improved) | scripts/40 training log / ckpt meta (`predictor_cf_dino_wm_droid.pt`) | ✅ |
| 7.2 | A/B planning probe (n=157): pooled Action-Error **1.468→1.086** (−26%), Action-Score 0.466→0.525, effect-CRA **0.061→0.605**; AE better in all 6 regime×horizon cells, AS in 5/6 | `results/droid_planning_cf_dino_wm_droid_{frozen,lora}.csv` (pooling n_planned-weighted, verified 2026-07-06) | ✅ |
| 7.3 | Hardening: 4 CF seeds all beat frozen on all 3 metrics (AE 1.076±0.012, AS 0.544±0.019, CRA 0.610±0.034); 2nd model jepa_wm_droid r16_l3p0 all-better (AE 1.386→1.269, AS 0.476→0.514, CRA 0.076→0.315). Paper notes single-seed column is min-on-AS, not seed mean (fixed v0.3). RoboCasa = weak-positive at DROID recipe, overfits when pushed; kept out of main.tex per 2026-07-07 decision (`robocasa_summary.md`) | `results/cf_seed_summary.md`, `droid_planning_cf_*_s{1,2,3}.csv`, `droid_planning_cf_jepa_wm_droid_*.csv` | ✅ (RoboCasa optional robustness add) |

## C8 — Evaluation axis: the exploitation gap (methodology, no new experiment)

Paper contribution bullet **C7** and the Discussion "exploitation gap" protocol.
Not a new empirical claim — a named packaging of existing C5 evidence into a
reusable evaluation protocol $G(c,P) = \mathbb{E}_{\mathcal{E}}[\mathrm{err}_v] - \mathbb{E}_{\mathcal{D}_\mathrm{off}}[\mathrm{err}_v]$.

| # | Claim | Evidence | Status |
|---|---|---|---|
| 8.1 | Exploitation gap separates honest ($G\approx0$) from exploitable ($G\gg0$) costs where average-case metrics (readout/grounding/disagreement) rate them identically | worked instance = C5.1 (91.5%→24% <5cm on elites) + C5.3 honest-cost control (reach 2/8→8/8); average-case parity = C6.2/6.3 (94–100% grounding, high readout, same 0–2/16) | ✅ derived, no new run |
| 8.2 | The protocol is cheap: one decode pass over already-produced elites | `results/cem_exploit_precision.csv` produced from committed CEM elite dumps (scripts/35) | ✅ |
| 8.3 | G computed across all offline-recoverable arms reproduces the honest/no-signal/exploitable trichotomy (honest +1.4 flat; no-signal +1.7..+2.5 with true obj 22–27cm away; exploitable +5.9..+6.5) AND exposes the inversion: best off-policy readout (91.5%) = largest G = fails, while the working cost has no readout at all. Scoped honestly in-text: exploitation *diagnostic*, not a graded success predictor (failing arms too flat at 0–2/16 for rank-correlation validation; remaining arms need elite-dump re-runs, tooling shipped) | `results/exploitation_gap_ladder.{csv,md}` (scripts/47, offline; in paper §Discussion since v0.3) | ✅ |

## Optional strengtheners

| # | Item | Status |
|---|---|---|
| D.3 | Imagined-rollout object-error table (baseline vs +h, cache-only, ~2 h) | optional C3 strengthener: quantifies "hallucinated grasping" along full imagined rollouts |
| E.1 | Amortized GC-IDM control (removes the search adversary; `inverse_proposal` ckpt exists) | queued behind E0 (C5.3) |
