# Claim → evidence map (paper assembly checklist)

**Purpose:** every sentence the paper claims, with the exact artifact that backs
it. If a claim has no row here, it does not go in the paper. Companion to
`docs/PAPER_IDEA.md` (idea-of-record) and `paper/main.tex` (draft).

Status legend: ✅ measured & in-repo · 🟡 measured, caveat carried · ⬜ decision pending.

## C1 — Boundary Blindness is real, localized, and transfers

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1.1 | Frozen JEPA WMs fail to resolve contact bifurcations: BB concentrates at the pre-grasp boundary | `results/metaworld_boundary.csv`: pooled bb_boundary pre_grasp **1.323 (dino) / 1.280 (jepa)** vs free_space 0.282/0.299 (~4.5×), contact 0.481/0.441; per-task CI-aware: elevated in 4/6 (dino), 5/6 (jepa), zero confident reversals | ✅ |
| 1.2 | The locus replicates across two model families | same CSV, both `dino_wm_metaworld` and `jepa_wm_metaworld` columns | ✅ |
| 1.3 | The phenomenon transfers to real-robot data | `results/droid_boundary.csv`: pre_grasp **1.975 [1.601, 2.350]** vs free_space 0.721 [0.613, 0.834] — CI-confident | 🟡 ‖Δz‖ proxy, transfer-only (no object GT on DROID) |
| 1.4 | Effect-conditioned CRA collapses in contact regimes (the precursor metric) | `results/metaworld_diagnostic.csv`: opposite CRA ~0.97–0.99 but hard_nn ~0.46–0.57 in pre-grasp/contact; `droid_diagnostic.csv`: hard_nn at 16-way chance floor | ✅ |
| 1.5 | Boundary label is an object-displacement proxy, not contact GT | stated in `boundary_gate_report.md`; mw-door-close excluded as proxy anomaly | 🟡 carried on figure |

## C2 — BB ⇄ planning failure (regime-level, closed-loop)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 2.1 | Where BB is low, planning works — and our harness reproduces the published baseline | `results/metaworld_closed_loop.csv`: mw-reach L2 **15/16 (94%)** [Wilson 72–99] vs paper Table 1 DWM CEM-L2 **44.8 ± 8.9**; grounded 16/16 | 🟡 success = any-step flag (TD-MPC2 convention); upstream judges at episode end; paper averages 3 training seeds. Optional strict re-score declared. |
| 2.2 | Where BB is high, planning collapses with the predicted signature | same CSV: mw-push & mw-pick-place **0/16 both arms**; final ee 2–4 cm (arm arrives) vs state-dist ~0.5–0.6 (object unmoved) | ✅ |
| 2.3 | The collapse is a *model* failure, not a harness failure | reach 94% (harness healthy) + render fidelity verified: one-step pred err **1.4×** dataset, latent NN ratio **0.97** (`results/closed_loop_report.md`, probes `_baseline_probe/_camera_calib/_replay_check`) | ✅ |
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
| R.2 | Three env-side reproduction bugs found+fixed (default-camera 480px renderer; training data = flipud(corner2+tweak), MSE 71.6 vs ≥3000 unflipped; goal = expert FINAL frame) | `results/closed_loop_report.md` §pitfalls; calibration artifacts `results/logs/camera_calib*`, probes committed | ✅ |
| R.3 | Physics/world identical to data-gen (only visuals differed) | `_replay_check.py`: action replay ee err median 1.5 mm | ✅ |

## Decisions still open

| # | Item | Options |
|---|---|---|
| D.1 | V-JEPA-2-AC leg (named in the thesis sentence) | (a) run BB gate on A5000 server (`03`→`05`+`12` for vjepa2_ac_droid); (b) scope claim to DINO-WM/JEPA-WM families + cite V-JEPA-2 Table 2 sub-goal dependence as external corroboration |
| D.2 | Reach success-criterion strict re-score (end-of-episode judging, ~3 h local) | optional; pre-empts the "94 vs 44.8" reviewer question — currently carried as caveat 2.1 |
| D.3 | Imagined-rollout object-error table (baseline vs +h, cache-only, ~2 h) | optional C3 strengthener: quantifies "hallucinated grasping" along full imagined rollouts |
