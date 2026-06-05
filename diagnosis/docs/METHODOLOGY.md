# Methodology & Code Guide — how the CAI-JEPA diagnostic proves the action-grounding gap

This is the **conceptual + code map** for the diagnostic. Read it before touching the
pipeline; it explains *what* we measure, *why* each dataset / task / regime / negative
strategy exists, and *how* the numbers add up to a GO / NO-GO decision. For the *operational*
"how do I run it on this machine" steps, see `HANDOFF.md` (Metaworld) and `HANDOFF_DROID.md`
(DROID). For the upstream-API ground truth, see `plans/2026-06-01-real-api-rewrite-design.md`.

---

## 1. The idea in one paragraph (CAI-JEPA gap)

Action-conditioned JEPA world models (DINO-WM, V-JEPA-2-AC, JEPA-WM/Terver) predict the next
latent `ẑ_{t+1} = F_θ(z_t, a_t)`. They pass standard evals (held-out prediction MSE, planning
success, qualitative open-vs-close-gripper counterfactuals), but those evals never test whether
the predictor **reliably distinguishes the futures induced by different actions from the same
state** — especially in contact-rich / fine-precision regimes. The paper calls this property
**action-identifiability**. This diagnostic is a **go/no-go study**: on *frozen, pretrained*
checkpoints (nothing is trained), quantify whether action-grounding failures are real and
concentrated where the theory predicts. If yes → pursue the full paper (CounterfactualBench +
the CAI-JEPA loss); if not → pivot/abandon. Deliverable: `results/decision_report.md`.

**Falsifiable claim being tested:** a well-grounded model should rank the *factual* action's
prediction closest to the true `z_{t+1}` among a set of counterfactual actions. If it can't —
particularly when the action effect is small but consequential (pre-grasp, gripper, contact) —
that is the measurable failure the paper is built around.

---

## 2. What the diagnostic actually computes (vs. the paper)

The paper (proposal §4) defines **CounterfactualBench** with 4 metrics × 4 regimes × 3 negative
strategies. This diagnostic implements the *decision-relevant subset* of that on frozen models:

| Paper concept | In this diagnostic |
|---|---|
| CRA (Counterfactual Ranking Accuracy) | `metrics/cra.py` — **primary signal** |
| AUG (Action Usage Gap) | `metrics/aug.py` |
| ECS (Effect-Conditional Sensitivity) | `metrics/ecs.py` + effect-gating in the runner |
| CTD (trajectory divergence) | `metrics/ctd.py` — optional (`--ctd`), not in the decision |
| 4 state regimes | `stratification/` (per-dataset classifiers) |
| negative strategies | `metrics/negative_samplers.py` — **4** (we add `hard_effect`) |

Everything runs on a **latent cache** (encode once, never re-encode), so all metrics are cheap
and reproducible. The primary decision number is **effect-conditioned CRA under hard negatives
in contact regimes** — see §7.

---

## 3. Code architecture — the 6-step pipeline

Data flows one direction; each script reads the previous step's artifact. Nothing is trained.

```
checkpoints (torch.hub, frozen)          datasets (Metaworld HF / DROID gsutil)
        │                                          │
        ▼                                          ▼
 models/adapters/EncPredWMAdapter  ◄── drives ──►  data/loaders.py  (raw frames+actions+state)
        │  .encode / .unroll(predict)
        ▼
[03] scripts/03_extract_latents.py  ──►  data/precomputed_latents/{dataset}__{model}.h5
        │   (encode every frame once: z, action, proprio, state, gripper)
        ▼
[04] scripts/04_classify_regimes.py ──►  {…}.h5.regimes.json   (atomic sidecar, per-transition regime)
        │   stratification/{metaworld,droid,robocasa}_regimes.py
        ▼
[05] scripts/05_run_diagnostic.py   ──►  results/{dataset}_diagnostic.csv
        │   for (model × strategy × regime × task): sample negatives → adapter.predict →
        │   CRA / AUG / ECS, with trajectory-clustered bootstrap CIs
        ▼
[06] scripts/06_analyze_results.py  ──►  results/decision_report.md + figures/*.pdf
            CI-aware GO / CONDITIONAL_GO / PIVOT / ABANDON

  ── planning probe (DROID; closes CRA_eff → planning-failure link) ──
[08] scripts/08_planning_probe.py    ──►  results/droid_planning.csv + droid_planning_pertrans.npz
        │   per transition: CRA_eff(hard_nn) + faithful CEM plan → paper's DROID Action Error
        │   planning/cem_planner.py, metrics/action_score.py
        ▼
[09] scripts/09_correlate_planning.py ──► results/planning_correlation.md + figures/figure_c_*.pdf
            per-transition Spearman/Pearson(Action Error, CRA_eff) — expect clearly negative
```

Key module responsibilities:

- **`models/adapters/`** — one `EncPredWMAdapter` for all three baselines. They all load via
  `torch.hub.load('facebookresearch/jepa-wms', hub_id, trust_repo=True)` → `(EncPredWM,
  preprocessor)`. We drive the model **only** through its public `EncPredWM.encode` (raw
  `[0,255]` → `(B,T,V,H,W,D)`) and `EncPredWM.unroll` (the planner's prediction primitive) —
  never `.encoder`/`.predictor` directly. `frames_per_step = model_action_dim // action_dim`
  handles frameskip (Metaworld=5, DROID=1). Action normalization is the model's own
  `preprocessor.normalize_actions` — **the #1 bug source** (Metaworld = real shift+scale;
  DROID = identity).
- **`data/loaders.py`** — wraps the *real* upstream datasets (`MetaworldHFDataset`,
  `DROIDVideoDataset`) with `transform=None, normalize_action=False` so we get raw frames /
  actions / state, then the adapter applies the model's own transform + normalization (single
  source of truth).
- **`metrics/`** — each metric exposes a *per-transition* function the runner calls directly,
  so `07_validate_synthetic.py` tests the exact production path with synthetic models.
- **`stratification/`** — assigns each transition one of the 4 regimes (§5).
- **`scripts/06`** — turns the CSV into figures + a CI-aware decision (§7).
- **`planning/cem_planner.py` + `metrics/action_score.py`** — faithful port of the upstream
  `CEMPlanner` and the DROID Action Error; driven by `scripts/08`, correlated by `scripts/09`.

Offline correctness gates (no GPU, no data): `pytest tests/` (34 tests, incl. CEM /
action-score / grounded-vs-ignoring planning sign-check) and `scripts/07_validate_synthetic.py`
(PerfectModel → CRA≈1.0; ActionIgnoringModel → CRA≈chance).

---

## 4. Datasets & tasks — what we run and why it proves the gap

The thesis is **failures concentrate in contact-rich / fine-precision regimes**. We need data
that (a) exercises those regimes and (b) has signals to *detect* them. Two datasets, deliberately
chosen to be complementary; Push-T / PointMaze are sanity-only and never thesis evidence.

### 4a. Metaworld (primary) — breadth across the difficulty spectrum
- **12-task subset** spanning easy → hard (config `dataset.tasks`):
  - *easy*: `mw-reach`, `mw-push`, `mw-pick-place`
  - *medium*: `mw-door-open`, `mw-door-close`, `mw-drawer-close`, `mw-button-press`, `mw-window-close`
  - *hard*: `mw-peg-insert-side`, `mw-assembly`, `mw-hammer`, `mw-stick-pull`
- **Models:** `dino_wm_metaworld`, `jepa_wm_metaworld` (both fit any GPU).
- **Why:** many tasks → per-task CRA, and the *hard* tasks are the contact/precision ones where
  we expect grounding to break. The decision reads the **hard-task contact-regime** cells.
- **Known limitation:** the HF Metaworld release has **no MuJoCo contact GT**, so regimes are a
  *proxy* from the 39-dim state vector, and the dataset has **no usable gripper-actuation
  signal** → that regime is empty on Metaworld. This is exactly the hole DROID fills.

### 4b. DROID (secondary) — real Franka, real gripper, real contact
- **Models:** `dino_wm_droid` + `vjepa2_ac_droid` (both fit the A5000, 24 GB — the paper's
  headline DROID comparison). `jepa_wm_droid` stays out for a non-hardware reason (gated DINOv3
  weights); see `HANDOFF_DROID.md` §1.
- **Data:** a **333-episode** public subset (2 labs, wrist camera), built by hand — DROID has no
  HF download and the raw bucket is 5.6 TB (`HANDOFF_DROID.md` §3). One 8-frame clip per episode
  at **fps=4** (matches training so the pose-diff action scale is in-distribution) → ~2331
  transitions. There are **no task labels** in DROID; the loader treats it as one flat pool
  (`task = "droid"`).
- **Why DROID is the key add:** `gripper_position` is a **real** signal, so the
  `gripper_actuation` (16%) and `contact_manipulation` (18%) regimes actually populate — the two
  cells Metaworld leaves empty. DROID is also the dataset the paper's target models (V-JEPA-2-AC)
  were trained on, so it is the most relevant evidence for the planning claim.

**How the two datasets together prove the gap:** Metaworld gives breadth + a second model for
comparison; DROID gives the *real contact/gripper regimes* on a real robot. A consistent CRA
collapse in those regimes across both datasets is strong, hard-to-explain-away evidence.

---

## 5. State regimes — the 4 cells and the hypothesis

Every transition is labelled one of four regimes (`stratification/`). The thesis predicts a
specific **pattern**: grounding holds in clean regimes and degrades where the action effect is
small-in-pixels but consequential.

| Regime | Meaning | Detection (Metaworld proxy) | Detection (DROID) |
|---|---|---|---|
| `free_space` | arm moves through empty space | ee moving, no object displacement | low gripper-Δ, below-median latent change |
| `pre_grasp` | approaching an object, no contact yet | ee near object, object still | gripper **open** + above-median latent change |
| `gripper_actuation` | opening/closing the gripper | (no gripper signal → **empty**) | `|Δgripper| > 0.2` |
| `contact_manipulation` | in contact, action moves the object | object displacement > τ | gripper **closed** + above-median latent change |

**Hypothesis (proposal §4.5):** high CRA on `free_space` and (where visible) `contact`; degraded
CRA on `pre_grasp` and `gripper_actuation` (small visual effect, large task consequence). This
is *why* failures hide from qualitative open-vs-close demos but hurt fine grasping.

**DROID contact/pre-grasp are proxies** — no MuJoCo GT. The contact threshold is *encoder-
calibrated*: DINOv2 ViT-S patch-L2 has a narrow dynamic range (median≈622, max≈842), so the
original `1.5×median` gate produced **0% contact**; it was lowered to `1.0×median` (above-median
visual change while the gripper is closed). See `stratification/droid_regimes.py` and
`HANDOFF_DROID.md` §4. **Report this caveat** — it is a deliberate, documented calibration, not a
silent fudge.

---

## 6. Negative-action strategies — the 4 difficulty levels

For each factual transition `(z_t, a_t, z_{t+1})` we build K=16 counterfactual actions and ask:
is the factual action's prediction the closest to the true `z_{t+1}`? The **strategy** decides
how hard the counterfactuals are. This is the heart of the diagnostic — an easy strategy that
every model passes proves nothing; the hard ones expose the gap. Code: `metrics/negative_samplers.py`.

| Strategy | Negative `a⁻` | What it tests | Expected |
|---|---|---|---|
| `random` | uniform in action bounds (DROID: projected to L1 ball) | gross action sensitivity | easy — all models should pass |
| `opposite` | `−a_t + noise`, gripper dim flipped | direction reversal — the *quantitative* version of Terver's open-vs-close demo | easy — high CRA even for weak models |
| `hard_nn` | from a **similar-state** pool, the action **most different** from `a_t` | fine action distinctions a competent policy might take from a near-identical state | **hard** — where the gap shows |
| `hard_effect` | from a similar-state pool, the action whose **true effect Δz differs most** from the factual Δz, preferring actions **close** to `a_t` | "precise action matters": a *fair* hard negative whose real future genuinely differs from `z_{t+1}` | **hardest + fair** |

**Why `opposite` near-perfect but `hard_nn` collapsing is the proof.** On Metaworld both models
score ~0.97–0.99 on `opposite` (they pass the demo) yet drop to ~0.46–0.57 on `hard_nn` in
pre-grasp/contact (chance = 1/17 ≈ 0.059). That dissociation — pass the easy test, fail the hard
one in exactly the predicted regimes — *is* the action-grounding gap the paper claims.

**`hard_nn` vs `hard_effect` (the subtle but important distinction).**
- `hard_nn` maximises *action* difference and ignores the candidate's outcome. Risk: the chosen
  negative may, from this state, lead to *the same* future as `a_t` (e.g. in smooth free-space,
  many actions produce near-identical `z_{t+1}`). Then a low CRA is "unfair" — the model isn't
  wrong, the two futures really are indistinguishable.
- `hard_effect` fixes this by scoring candidates with
  `‖Δz_cand − Δz_factual‖ − action_penalty·‖a_cand − a_t‖` (both std-normalized). It picks a
  near-by action that *genuinely* leads to a different true future, so a well-grounded model
  **can** win the comparison. In smooth regimes no such negative exists, so `hard_effect`
  **self-selects toward contact/precision** transitions — precisely where the paper says
  grounding matters. Tune with `hard_effect.action_penalty` (0 = pure max-effect-divergence).
  Keeping both lets the report compare the "unfair-hard" and "fair-hard" definitions side by side.

---

## 7. Metrics & the decision logic

- **CRA (primary).** `P[ d(F(z_t,a_t), z_{t+1}) < min_k d(F(z_t,a⁻_k), z_{t+1}) ]`. Top-1 and
  MRR. Chance = 1/(K+1) ≈ 0.059. Distance is **L2** for every baseline (all upstream planning
  configs are `L2_cem`).
- **Effect-conditioned CRA (`cra_top1_eff`) — the decision number.** CRA computed only on
  transitions with `‖z_{t+1} − z_t‖ > τ` (τ = median Δz, calibrated per model). A low raw CRA in
  contact can just mean a tiny one-step delta; gating by effect isolates "the model fails to use
  the action *when something actually happens*."
- **AUG** = MSE(shuffled action) − MSE(factual action); **ECS** = AUG restricted to effectful
  transitions. Positive ⇒ the model uses actions; ≈0 ⇒ it ignores them.
- **CIs:** trajectory-clustered bootstrap (`metrics/bootstrap.py`, `n_resamples=1000`) — resample
  whole trajectories, not transitions, so within-trajectory correlation doesn't inflate
  significance.

**Decision (`scripts/06::make_decision`), read on the strongest available baseline, `hard_nn`,
contact regimes, hard tasks for Metaworld:**

| Outcome | Rule (effect-cond. CRA `c`, upper CI `hi`) |
|---|---|
| **GO** | strong pathology: `mw < 0.60` **and** `droid < 0.65` |
| **ABANDON** | only if **both** upper CIs are high: `mw_hi ≥ 0.85` **and** `droid_hi ≥ 0.85` |
| **CONDITIONAL_GO** | moderate pathology in **at least one** dataset (`c < 0.75`) |
| **PIVOT** | mixed signal otherwise |

The asymmetry is deliberate: ABANDON needs the *upper* CI bound confidently high in both
datasets, so a single noisy number can't kill the project. (06 auto-selects the DROID model
present in the CSV, so `dino_wm_droid` feeds the decision even though the constant names
`jepa_wm_droid`.)

---

## 8. The experimental matrix (what a full run produces)

Per dataset, the runner sweeps **model × strategy × regime × task**:

| Dataset | Models | Strategies | Regimes | Tasks | Cells |
|---|---|---|---|---|---|
| Metaworld | 2 | random, opposite, hard_nn (+hard_effect) | 4 | 12 | 2·3·4·12 = 288 (done) |
| DROID | 1 (`dino_wm_droid`) | random, opposite, hard_nn, **hard_effect** | 4 | 1 (`droid`) | 1·4·4·1 = 16 |

A cell with `< min_transitions_per_cell` rows is emitted as `insufficient_data` (not silently
dropped). The **per-transition** CRA/AUG/ECS plus trajectory-clustered CIs are written to
`results/{dataset}_diagnostic.csv`; `06` reduces them to figures + the decision report.

**Sanity gate before trusting any DROID number** (CLAUDE.md #6): `terver_gripper_test.py` —
the quantitative open-vs-close-gripper test; every DROID baseline must score **2-way CRA > 0.90**.
If it doesn't, suspect a pipeline bug (action normalization, frameskip), not a model failure.

---

## 9. Current status & where to continue

| Piece | State |
|---|---|
| Metaworld diagnostic | ✅ **complete** — `results/metaworld_diagnostic.csv`, decision **CONDITIONAL_GO** |
| Metaworld finding | `opposite` ~0.97–0.99 but `hard_nn` ~0.46–0.57 in pre-grasp/contact → gap is real; jepa_wm > dino_wm consistently |
| DROID env + data + latents + regimes | ✅ **done & cached** for `dino_wm_droid` (`HANDOFF_DROID.md`) |
| DROID `05`/`06` | ⏳ **ready to run** — now on the A5000 (24 GB); the old 8 GB GPU fault is resolved |
| `hard_effect` strategy | ✅ implemented (`negative_samplers.py`), wired through `05`, in the DROID config |
| Planning Action-Score probe (`08`/`09`) | ✅ **coded + offline-tested**; runs on server to correlate CRA_eff ↔ Action Error (§ HANDOFF_DROID §8) |

**To finish DROID** (latents cached for dino_wm_droid; run `03` first for `vjepa2_ac_droid`):
```bash
cd diagnosis && source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
python scripts/03_extract_latents.py --config configs/diagnostic_droid.yaml   # adds vjepa2_ac_droid cache
python scripts/05_run_diagnostic.py  --config configs/diagnostic_droid.yaml
python scripts/06_analyze_results.py --metaworld_csv results/metaworld_diagnostic.csv \
                                     --droid_csv     results/droid_diagnostic.csv
# Then close the planning loop (HANDOFF_DROID §8):
python scripts/08_planning_probe.py     --config configs/diagnostic_droid.yaml
python scripts/09_correlate_planning.py --planning_csv results/droid_planning.csv \
   --pertrans results/droid_planning_pertrans.npz --diagnostic_csv results/droid_diagnostic.csv
```

**What "proves the gap" looks like in the final report:** effect-conditioned CRA under `hard_nn`
(and `hard_effect`) that is (i) far above chance on `opposite`/`random` but (ii) low in
`pre_grasp` / `gripper_actuation` / `contact_manipulation`, (iii) consistent across Metaworld
*and* DROID. That pattern → **GO/CONDITIONAL_GO** → build CounterfactualBench + the CAI-JEPA loss.

---

## 10. Doc map for future agents

- **`METHODOLOGY.md`** (this file) — concepts, code map, experimental design, decision logic.
- **`HANDOFF.md`** — operational: run the Metaworld primary path on a fresh server.
- **`HANDOFF_DROID.md`** — operational: the DROID secondary run (env, data, recalibration; now on
  the A5000) + §8 the planning Action-Score probe.
- **`plans/2026-06-01-real-api-rewrite-design.md`** — the real upstream API + key design decisions.
- **`../../cai_jepa_paper_proposal.md`** — the research idea (the 4 contributions).
- **`../../diagnostic_implementation_plan_v2.md`** — the phased plan; §12 records v2.1 adjustments.
- **`../../CLAUDE.md`** — repo orientation; points here first.
</content>
</invoke>
