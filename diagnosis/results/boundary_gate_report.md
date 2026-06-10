# Boundary Blindness gate — run & analysis report

**Date:** 2026-06-10. **Machine:** Windows 11, RTX 5070 (12 GB VRAM), 15.7 GB RAM,
`diagnosis/.venv` (torch 2.11+cu128), frozen `facebookresearch/jepa-wms` checkpoints from the
local torch.hub cache. Nothing was trained; all numbers come from frozen baselines on the
pre-existing latent caches (steps 03+04).

**Bottom line: the gate PASSES.** `bb_boundary` is confidently (CI-aware) elevated at the
**pre-grasp boundary** on both Metaworld baselines and on the DROID transfer check, exactly the
locus the Boundary-Blind reframing predicts. `contact_manipulation` is moderately elevated on
Metaworld and not elevated on DROID. Recommendation: **green-light fix C1**
(`models/heads/mixture_predictor.py` + `train_predictor_head.py`).

---

## 1. Run log

All GPU runs after the incident (see §1.1) were executed via `scripts/run_with_watchdog.ps1`
(BelowNormal priority + kill-switch at <1.2 GB free RAM; peak GPU sampled at 10 s intervals).
Exit codes were verified from the scripts' own terminal `Wrote …` lines and output artifacts
(the watchdog's `EXIT=` field is blank due to a `Start-Process` handle quirk; `KILLED=False`
confirms no watchdog intervention).

| # | command | outcome | wall | peak GPU |
|---|---|---|---|---|
| 0 | `pytest tests/test_boundary_diagnostic.py -q` | **11 passed** | 2 s | — |
| 0b | `pytest tests/ -q` | **56 passed** | 3 s | — |
| 1 | `12_boundary_diagnostic.py --config configs/diagnostic_metaworld.yaml` (first attempt) | **machine froze; hard restart** (§1.1) | — | — |
| 2 | same, after chunking fix + watchdog | exit 0 but **degenerate output** (BB≡0, n_b≡0; §1.2) | 19.6 min | 7,597 MiB |
| 3 | same, after neighbour-fallback fix | **OK → `results/metaworld_boundary.csv` (64 rows)** | 19.6 min | 7,606 MiB |
| 4 | `terver_gripper_test.py --config configs/diagnostic_droid.yaml --model dino_wm_droid` | **PASS** (§4) | 0.8 min | 2,826 MiB |
| 5 | `05_run_diagnostic.py --config configs/diagnostic_droid.yaml` (`CAI_JEPA_ONLY_MODEL=dino_wm_droid`) | **OK → `results/droid_diagnostic.csv` (16 rows)** | 3.7 min | 6,238 MiB |
| 6 | `12_boundary_diagnostic.py --config configs/diagnostic_droid.yaml` (same env) | **OK → `results/droid_boundary.csv` (4 rows)** | 1.2 min | 5,155 MiB |
| 7 | `06_analyze_results.py --metaworld_csv results/metaworld_diagnostic.csv --droid_csv results/droid_diagnostic.csv` | **OK → `results/decision_report.md` + figures, CONDITIONAL_GO** | <1 min | — |
| 8 | `scripts/_make_bb_figure.py` | **OK → `results/figures/figure_bb_per_regime.pdf`** | <1 min | — |

Warnings common to all runs (benign): upstream `JEPAWM_LOGS`/`JEPAWM_CKPT` env-var placeholders;
unauthenticated-HF-Hub rate-limit notice; cp1252-console Unicode logging errors from emoji in
upstream log strings (cosmetic; silenced with `PYTHONIOENCODING=utf-8` in later runs).

### 1.1 Incident: machine freeze on the first attempt, and the fix

`metrics/boundary_blindness.py::boundary_sensitivities_per_transition` launched **one predictor
forward for the whole cell × neighbourhood** (`B×M` ≈ up to 16,000 unrolls in a single batch,
~6 GB of input tensors before activations). On Windows the NVIDIA driver's sysmem fallback turns
the resulting VRAM overflow into system-RAM thrash instead of a clean CUDA OOM; with 15.7 GB
total RAM the machine froze and had to be hard-restarted. Two changes:

- **Chunked the forward** (max `CAI_JEPA_BB_PREDICT_ROWS`=256 predictions per call; per-anchor
  spreads computed chunk-wise, numerically identical — 11/11 unit tests pass, also under an odd
  chunk size). Peak GPU thereafter stayed ≤7.6 GB on the 12 GB card.
- **Added `scripts/run_with_watchdog.ps1`**: BelowNormal priority + automatic kill below 1.2 GB
  free RAM, so any future blow-up aborts the process instead of the machine.

### 1.2 Incident: degenerate first result (BB≡0), and the fix

Run 2 completed but returned `BB=0.000`, `n_boundary=0` for **every** cell. Cause:
`stratification/boundary_regime.py::state_neighbours` applied `hard_nn.similarity_radius=0.5`
**strictly**, in raw latent L2 units — but real Metaworld frame-latent distances are O(100)
(the dataset's median ‖Δz‖ alone is ≈175), so zero neighbours were ever in radius and every
spread/score degenerated to 0/nan. The production `hard_nn` sampler
(`metrics/negative_samplers.py`), whose neighbourhood the boundary diagnostic is documented to
reuse, has always had a **relax-to-K-nearest fallback** for exactly this case — and the
published CRA numbers were produced under that fallback. The same fallback was ported to
`state_neighbours` (faithfulness fix, not a tuning choice; full `pytest tests/` still 56/56).
Consequence to disclose: on these caches the neighbourhoods are effectively
**16-nearest-by-latent from the shared 256/512-transition pool**, not strictly in-radius sets.

## 2. Pre-flight

- `pytest tests/test_boundary_diagnostic.py -q` → **11 passed** (required gate).
- `pytest tests/ -q` → **56 passed** (before and after both fixes above).
- Caches present with `.regimes.json` sidecars: `metaworld__dino_wm_metaworld.h5` (24.9 GB),
  `metaworld__jepa_wm_metaworld.h5` (24.9 GB), `droid__dino_wm_droid.h5` (0.94 GB).

## 3. Task 1 — Boundary Blindness on Metaworld (the gate)

Source: `results/metaworld_boundary.csv` (64 rows; per-(task, regime) `bb`/`bb_boundary` with
trajectory-clustered bootstrap CIs; `boundary.quantile=0.75`, `max_neighbours=16`, L2 throughout).

**Pooled per regime** (n_boundary-weighted mean of `bb_boundary` over task cells; excluding
`mw-door-close` — see caveat below):

| regime | dino_wm bb_boundary | jepa_wm bb_boundary | dino_wm bb (all) | jepa_wm bb (all) |
|---|---|---|---|---|
| free_space | 0.282 | 0.299 | 0.069 | 0.070 |
| pre_grasp | **1.323** | **1.280** | 0.541 | 0.581 |
| contact_manipulation | 0.481 | 0.441 | 0.212 | 0.194 |
| gripper_actuation | — (no populated cells) | — | — | — |

**CI-aware per-task pairing** (a regime counts as confidently elevated in a task iff
`bb_boundary_lo(regime) > bb_boundary_hi(free_space)` in that task):

| comparison | dino_wm | jepa_wm |
|---|---|---|
| pre_grasp vs free_space | 6/6 tasks higher; **4/6 CI-confident**; 0 reversals | 5/6 higher; **5/6 CI-confident**; 0 reversals |
| contact_manipulation vs free_space | 5/7 higher; 2/7 CI-confident; 0 CI-reversals | 3/7 higher; 2/7 CI-confident; 1 CI-reversal (`mw-assembly`) |

Representative cells (point [95% CI]): `mw-door-open` pre_grasp dino 2.157 [1.800, 2.508] vs
free_space 0.399 [0.261, 0.581]; `mw-window-close` pre_grasp jepa 1.151 [0.937, 1.320] vs
free_space 0.190 [0.056, 0.335]; `mw-button-press` pre_grasp jepa 1.102 [0.882, 1.299] vs
free_space 0.089 [0.053, 0.127].

**GATE VERDICT: PASS.** `bb_boundary` is CI-confidently elevated relative to `free_space` in
the contact-rich regimes for both frozen baselines, with the elevation concentrated in
**pre_grasp** (≈4.5× free_space pooled, both models) — precisely the predicted locus of the
grasp/approach bifurcation. `contact_manipulation` is moderately elevated (≈1.6×, point), but
is not the locus; `gripper_actuation` is empty on Metaworld as anticipated (no gripper signal
in this 60-traj/task subset). Interpretation: at bifurcation-like transitions the models'
predicted-future spread collapses relative to the true outcome spread — they smear the
boundary — and this happens overwhelmingly where the gripper is about to make-or-miss contact,
not in free space and not (post-grasp) in continuous contact, where dynamics are smooth again.
This is the headline pattern the Boundary-Blind framing predicted.

**Outlier disclosed:** `mw-door-close` shows BB ≈ 1.7–2.9 in *every* regime including
free_space (n_b=448–466 there). Articulated door motion produces large object-displacement
spread regardless of the proxy regime label, inflating S_true globally for that task.
Including it, pooled free_space `bb_boundary` rises to ≈1.37–1.45 and the pooled ordering
blurs; the per-task paired comparison above (which never pairs door-close against itself
across regimes — it has no pre_grasp cell) is the robust read. This is a known weakness of
the object-displacement proxy for articulated objects, to be stated in the paper.

## 4. Task 2 — DROID (`dino_wm_droid` only)

Sanity gate first — `terver_gripper_test.py`: **PASS** (518 transitions; gripper-delta
alignment max error 5.96e-08; cache↔loader max error 0.0; model action-path sensitivity
non-collapsed: fact-vs-zero 178.3, open-vs-close 52.9). Note this script replaced the old
2-way-CRA>0.90 form; it validates plumbing (alignment/normalization/frameskip), not grounding.

**CRA (refreshed `results/droid_diagnostic.csv`, 16 rows, single flat `droid` pool):**

| regime | n | random CRA | opposite CRA | hard_nn CRA | hard_nn CRA_eff [95% CI] |
|---|---|---|---|---|---|
| free_space | 998 | 0.153 | 0.396 | 0.084 | 0.000 [0.000, 0.000]* |
| pre_grasp | 518 | 0.367 | 0.044 | 0.070 | 0.070 [0.048, 0.094] |
| gripper_actuation | 400 | 0.212 | 0.085 | 0.022 | 0.035 [0.015, 0.061] |
| contact_manipulation | 415 | 0.443 | 0.166 | 0.055 | 0.056 [0.036, 0.078] |

*free_space has only n_effect=4 effectful transitions — its CRA_eff is not meaningful.
16-way chance ≈ 0.059: `hard_nn`/`hard_effect` are at or below the chance floor in every
contact-rich regime, reproducing the 2026-06-05 server run (differences ≤0.01–0.02 from
pool-sampling randomness).

**BB (`results/droid_boundary.csv`, ‖Δz‖ proxy outcome):**

| regime | bb [95% CI] | bb_boundary [95% CI] | n / n_boundary |
|---|---|---|---|
| free_space | 0.422 [0.365, 0.476] | 0.721 [0.613, 0.834] | 998 / 415 |
| pre_grasp | 0.887 [0.753, 1.034] | **1.975 [1.601, 2.350]** | 518 / 97 |
| gripper_actuation | 0.395 [0.330, 0.463] | 1.093 [0.791, 1.393] | 400 / 30 |
| contact_manipulation | 0.311 [0.243, 0.375] | 0.463 [0.266, 0.686] | 415 / 41 |

`pre_grasp` is CI-confidently elevated over free_space (lo 1.601 > hi 0.834, ≈2.7×);
`gripper_actuation` is point-elevated (1.093 vs 0.721) with marginal CI overlap (n_b=30);
`contact_manipulation` is not elevated. **Sample-size caveats:** one flat `droid` pool (no
per-task split), proxy regime labels from proprio + latent-change heuristics, small boundary
subsets in gripper/contact cells, and 333 episodes total. Transfer check only.

## 5. Cross-check: BB vs the existing CRA finding

The two signals agree and are complementary; no contradiction found.

- Metaworld CRA: `opposite` ≈ 0.97–0.99 vs `hard_nn` ≈ 0.46–0.57 collapse, worst in pre_grasp
  (hard_nn CRA_eff 0.452–0.530) — the models stop distinguishing similar-state counterfactual
  actions exactly where BB now shows they smear the bifurcation (pre_grasp pooled bb_boundary
  ≈ 1.3 vs ≈ 0.29 free_space).
- DROID CRA: chance floor in gripper/contact/pre-grasp regimes; BB adds resolution *within*
  that floor — the boundary-smearing concentrates pre-grasp (1.975) rather than uniformly.
- One nuance, consistent with the reframing rather than against it: CRA_eff is also mediocre in
  `contact_manipulation`, but BB is only moderately elevated there. Post-grasp contact dynamics
  are comparatively smooth (few bifurcations), so low CRA there reflects weak action
  amplification, while the *bifurcation-resolution* failure (BB) is specific to the pre-grasp
  boundary. This is exactly the distinction the two-metric design was built to expose.

## 6. Honesty caveats (binding for any use of these numbers)

1. **Metaworld boundary/contact labels are an object-displacement proxy** — the HF dataset has
   no MuJoCo contact ground truth. The `mw-door-close` anomaly (§3) is a concrete failure mode
   of this proxy for articulated objects.
2. **DROID is transfer-only**: proprio is pose(6)+gripper(1), no force/torque, no joint state;
   regimes and the BB outcome (‖Δz‖) are proxies. No tactile/force claims.
3. **`vjepa2_ac_droid` was not run** — it needs ~24 GB VRAM (server A5000); this machine has a
   12 GB RTX 5070. Server-only follow-up.
4. **Neighbourhood semantics**: the similarity radius always relaxes to 16-nearest-by-latent on
   these caches (§1.2) — the same behaviour as the production hard_nn sampler used for CRA.
5. Distance is **L2 everywhere** (matching the upstream `L2_cem` planning configs). AUG is not
   cross-model comparable; CRA/BB are.
6. Push-T / PointMaze remain sanity checks only — none of their numbers appear here.
7. The first (degenerate) Metaworld boundary run and the two code fixes are disclosed in §1.1–1.2;
   both fixes were validated against the full offline test suite before re-running.

## 7. Recommendation

**PASS → green-light fix C1**: build `models/heads/mixture_predictor.py` (mixture-density head,
K≈2–4, NLL over cached `z_{t+1}`) + `train_predictor_head.py` with the boundary-supervision head
(grasp/object-moves event from Metaworld state), per `docs/HANDOFF_BOUNDARY_FIX.md` §4. Success
criterion is pre-registered: BB drops in boundary regimes (re-run `12_boundary_diagnostic.py` on
the retrained head) and planning improves vs both the frozen baseline and the one-step-margin
fix-baseline. Secondary follow-ups: re-run the planning probe targeting BB (Action Error vs BB,
the C2 figure), and on the server complete `vjepa2_ac_droid` (03 → 05 → 12).

## 8. ADDENDUM (same day) — the C1 fix leg was built and run: a quantified structural null

C1 was implemented and trained the same day (`models/heads/mixture_predictor.py`,
`scripts/train_predictor_head.py`, `scripts/13_eval_fix_boundary.py`; 65/65 offline tests).
Four head variants on `dino_wm_metaworld` (frozen trunk, 12,312/1,368 trajectory split,
3 epochs each, ~40 min/run, peak ≤3.8 GB GPU): soft-NLL K=3; WTA/hard-EM K=3; C1+D
(state-slice context) WTA K=3; C1+D **supervised mode assignment** K=2 (components labeled by
the object-moves event). Every variant beat its K=1 control on val NLL; **none moved BB**.
BB-evaluated through the production pipeline: soft-NLL K1/K3
(`results/metaworld_boundary_fix_nll.csv`, incl. the frozen-base reproduction) and supervised
K1/K2+state (`results/metaworld_boundary_fix.csv`) — identical to the frozen base to ~3
decimals across all regimes. The two WTA variants were probe-screened (same μ-separation /
π-flip signature) and their BB pass skipped.

The probes (`scripts/_probe_head_modes.py`) isolate two measured causes:

1. **Latent-metric compression of the boundary subspace.** The supervised variant's component
   means are the conditional means of "object moves" vs "doesn't" futures by construction; they
   differ by **9.9 L2 units vs a 106-unit median prediction residual** (median true step 170).
   The bifurcation is ~9% of the residual in the latent's L2 geometry — the same metric CEM
   plans with and BB's spread measures. No predictor head can make the mode jump matter.
2. **No counterfactual action coverage in expert data.** π as the boundary-event classifier
   calibrates (mean [0.746, 0.254] vs 24.9% positives) but its action-flip rate is ~0 and CE
   0.487 barely beats the 0.562 base rate: at boundary states the experts' actions almost
   always succeed, so the *action*-dependence of the event is unlearnable from this data.

**Revised recommendation:** the fix must act on the **latent/metric** (encoder-level D: a
supervised projection that re-weights the boundary subspace, trained with the
object-displacement label) and/or the **data** (counterfactual boundary actions), not on the
predictor head. Head-level multimodality — any K, any objective — is structurally insufficient
on this representation; that is itself the falsification result the design pre-registered, with
mechanisms measured. Full narrative: `docs/FIX_C1_EXPLAINER.md` §6.

**Resolution (same day, later):** the revised direction was executed and closed. The
metric-only form is also null (`results/metaworld_boundary_metric.csv`), but the probe chain
(V1✓ decodable / V2✓ propagated / V3✗ counterfactual response = noise) localized the true
bottleneck, and the **grounded object-dynamics channel** `h(z,a)→Δobject` fixes it:
counterfactual tracking corr +0.035→**+0.682**, pre_grasp `bb_boundary` **1.323→0.660**
(`results/metaworld_boundary_dynamics.csv`); planning open-loop A/B no-harm/no-gain
(closed-loop success rate = next experiment). Full method + results: `docs/FIX_C1_EXPLAINER.md` §7.

## 9. Files changed in this session

- `diagnosis/metrics/boundary_blindness.py` — chunked the per-neighbourhood predictor forward
  (`CAI_JEPA_BB_PREDICT_ROWS`, default 256) to cap VRAM; numerically identical results.
- `diagnosis/stratification/boundary_regime.py` — ported the production hard_nn
  relax-to-nearest fallback into `state_neighbours` (fixes the degenerate all-zero BB run).
- `diagnosis/scripts/run_with_watchdog.ps1` — new: RAM-watchdog + BelowNormal-priority runner
  for GPU scripts on this 16 GB Windows box.
- `diagnosis/scripts/_gate_analysis.py` — new: CI-aware gate aggregation over a boundary CSV.
- `diagnosis/scripts/_make_bb_figure.py` — new: generates `figures/figure_bb_per_regime.pdf`.
- `diagnosis/results/metaworld_boundary.csv`, `diagnosis/results/droid_boundary.csv`,
  `diagnosis/results/droid_diagnostic.csv` — new/refreshed results (sources of every number above).
- `diagnosis/results/decision_report.md` — regenerated by `06`, then extended with the
  "Boundary Blindness (the gate)" section and the restored planning-probe section.
- `diagnosis/results/figures/figure_bb_per_regime.pdf` — new headline figure.

### Documentation updated with the measured results

- `diagnosis/results/decision_report.md` — added the "Boundary Blindness (the gate)" section
  (BB tables, PASS verdict, caveats, CRA cross-check, figure) and restored the planning-probe
  section that `06` does not render.
- `diagnosis/docs/PAPER_IDEA.md` — status line now records the gate PASS; C1's "Predicted
  result" replaced with the measured BB numbers (sources quoted); §4 falsification gate marked
  passed; §6 headline figure 1 marked real and pointed at `figure_bb_per_regime.pdf`.
- `diagnosis/docs/DIAGNOSIS_PLAN.md` — §4 gains the measured VERDICT: PASS block; §5 status
  table updated (BB run, DROID dino_wm complete, C1 green-lit); §7 next actions reordered
  (C1 first; server items remaining).
- `diagnosis/docs/HANDOFF_BOUNDARY_FIX.md` — §0 TL;DR now says "gate PASSED → start C1";
  STEP 2 marked DONE: PASS with the result block and disclosed fixes.
- `PROJECT_OVERVIEW_VI.md` — header alert, Phần 5.7, 9.5, 9.6 (full real BB table + verdict),
  10.1, file map, and glossary updated from "chưa chạy / chưa có số" to the measured PASS
  result with caveats.
- `CLAUDE.md` — execution-status block updated to 2026-06-10: BB gate PASS (with headline
  numbers), DROID dino_wm complete, vjepa2_ac_droid server-only, planning leg recast around BB.

### Files added/changed by the C1 fix leg (§8)

- `diagnosis/models/heads/__init__.py`, `diagnosis/models/heads/mixture_predictor.py` — the C1
  head: residual MDN (+ boundary head, + direction-D state slice, + three objectives:
  nll / wta / boundary-supervised) and the `MixturePredictorAdapter` wrapper.
- `diagnosis/scripts/train_predictor_head.py` — head training on cached latents (frozen trunk;
  trains the K≥2 hero and the K=1 control in one pass).
- `diagnosis/scripts/13_eval_fix_boundary.py` — BB before/after through the production pipeline.
- `diagnosis/scripts/_probe_head_modes.py` — μ-separation / π-flip diagnostics.
- `diagnosis/tests/test_mixture_predictor.py` — 9 synthetic mechanism tests (65 total green).
- `diagnosis/checkpoints/mdn_dino_wm_metaworld_*.pt` — trained heads (4 variants × {K hero, K1}).
- `diagnosis/results/metaworld_boundary_fix.csv` (+ `_nll.csv`) — BB after the fix (null).
- `diagnosis/docs/FIX_C1_EXPLAINER.md` — the full problem → mechanism → results narrative.
