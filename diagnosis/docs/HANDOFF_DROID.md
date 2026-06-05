# Handoff — DROID secondary diagnostic (real Franka)

---

## 0. TL;DR — finish in 2 commands

```bash
cd <repo>/diagnosis && source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
python scripts/05_run_diagnostic.py  --config configs/diagnostic_droid.yaml
python scripts/06_analyze_results.py --metaworld_csv results/metaworld_diagnostic.csv \
                                     --droid_csv     results/droid_diagnostic.csv
```

Deliverable: `results/droid_diagnostic.csv` + an updated `results/decision_report.md`.

**Recommended sanity gate before trusting numbers** (CLAUDE.md #6): the Terver gripper test
(open-vs-close on real DROID, expect 2-way CRA > 0.90):

```bash
python scripts/terver_gripper_test.py --config configs/diagnostic_droid.yaml --model dino_wm_droid
```

---

## 1. Model selection — A5000 (24 GB) runs both VRAM-bound baselines

Hardware is **no longer the constraint** (we moved off the 8 GB box to an **A5000, 24 GB**). The
two VRAM-bound baselines both fit, which restores the paper's headline DROID comparison
(DINO-WM vs V-JEPA-2-AC). Only `jepa_wm_droid` stays out — for a **non-hardware** reason.


| hub id            | backbone                    | verdict on A5000 (24 GB)                                                                                                                                                                                                                                                       |
| ----------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dino_wm_droid`   | DINOv2 **ViT-S/14**         | ✅ **runs** — latents already cached (peak ~1.44 GB)                                                                                                                                                                                                                            |
| `vjepa2_ac_droid` | V-JEPA-2 **ViT-G/16** (~1B) | ✅ **runs** on 24 GB — **but** run `03_extract_latents.py` for it first (no cache yet)                                                                                                                                                                                          |
| `jepa_wm_droid`   | DINOv3 **ViT-L/16**         | ❌ blocked by **gated** DINOv3 `.pth` weights (Meta request-form), *not* hardware. HF mirror only ships `model.safetensors`; upstream `app/plan_common/models/dino.py` loads the original `.pth` via a local `~/dinov3` hub clone (`$JEPAWM_OSSCKPT/dinov3/...pth`).            |


Both `models:` entries are now enabled in `configs/diagnostic_droid.yaml`. **Before `05`/`08`**,
extract `vjepa2_ac_droid` latents (`03_extract_latents.py --config configs/diagnostic_droid.yaml`)
so its cache exists; otherwise `05` prints `[skip] … cache missing` for it and only scores
DINO-WM. To add `jepa_wm_droid` later: obtain gated DINOv3 ViT-L weights, clone
`facebookresearch/dinov3` to `~/dinov3`, set `JEPAWM_OSSCKPT` to the dir holding
`dinov3/dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth`, then add it to `models:`.

**Why DROID matters even with one model:** DROID has a **real gripper signal**
(`gripper_position`), so the `gripper_actuation` and `contact_manipulation` regimes populate —
exactly the cells the Metaworld HF proxy left at 0 % (Metaworld `gripper_actuation` had 0
effect transitions). This is the main value-add over the Metaworld-only result.

---

## 2. Environment (`diagnosis/.venv`, Python 3.10 — A5000 server)

We do **not** install the upstream `pyproject` (it drags `mujoco-py`/`d4rl`/`gym`/`dm-control`
sim stack we don't need). Lean venv with just what the encode/predict + DROID loader import:

```bash
cd <repo>/diagnosis
git clone --depth 1 https://github.com/facebookresearch/jepa-wms.git external/jepa-wms
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install "torch==2.7.*" "torchvision==0.22.*"          # CUDA 12.6 wheels
uv pip install timm tensordict einops h5py pandas scipy scikit-learn numpy pyyaml \
    omegaconf decord imageio imageio-ffmpeg huggingface-hub datasets matplotlib seaborn \
    tqdm Pillow gsutil opencv-python-headless ruamel.yaml termcolor lpips pytest
uv pip install -e .                                          # the diagnosis package
```

`.env` holds `HF_TOKEN=...` (gitignored; checkpoint weights come from the public
`dl.fbaipublicfiles.com` mirror, but HF is used for `dino_wm_droid.pth.tar`). For any gsutil
call use anonymous public access: `export BOTO_CONFIG=/dev/null`.

Sanity (no GPU, no data):

```bash
python -m pytest tests/ -q              # 34 passed (incl. planning probe: cem/action_score/sign-check)
python scripts/07_validate_synthetic.py # PASS (PerfectModel CRA=1.0, ActionIgnoring=chance)
```

---

## 3. DROID data — how the 333-episode subset was built

DROID has **no HF download** (the upstream `download_data.py` covers pusht/pointmaze/wall/
metaworld/robocasa/franka, not droid) and the raw bucket is **5.6 TB**. We pulled a small
public subset by hand:

1. **Anonymous gsutil** from the public bucket (no auth, `BOTO_CONFIG=/dev/null`):
  `gs://gresearch/robotics/droid_raw/1.0.1/<LAB>/success/<DATE>/<EPISODE>/`. Each episode dir
   has `metadata_*.json`, `trajectory.h5` (~~0.66 MB), `trajectory_im128.h5` (~~47 MB, **skip**),
   and `recordings/MP4/<serial>.mp4` (3 non-stereo cams ~1–3 MB each) + `-stereo.mp4` (**skip**).
2. `**rsync` two date dirs** (2 labs, for diversity), excluding stereo/im128/svo:
  ```bash
   export BOTO_CONFIG=/dev/null
   EXCL=".*-stereo\.mp4$|.*trajectory_im128\.h5$|.*\.svo$|.*\.svo2$|.*SVO.*"
   for P in "IRIS/success/2023-03-07" "TRI/success/2023-08-07"; do
     mkdir -p "data/droid_subset/$P"
     gsutil -m rsync -r -x "$EXCL" \
       "gs://gresearch/robotics/droid_raw/1.0.1/$P/" "data/droid_subset/$P"
   done
  ```
   → **415 episodes, ~4.3 GB.** (IRIS 2023-03-02 was avoided: those episodes are ~98 frames,
   too short — see below.)
3. **Build the paths CSV with a length filter** — `scripts/build_droid_paths.py` (NEW):
  ```bash
   python scripts/build_droid_paths.py --subset_root data/droid_subset \
       --output data/droid_subset/droid_paths.csv   # fpc=8 fps=4 vfps=60 → need ≥125 frames
  ```
   → **333 episodes kept** (82 too short), median length 167, **~2331 transitions**, gripper
   range `[0,1]`. `dataset.root` in the config points at this CSV (DROIDVideoDataset reads
   `data_path` as the **CSV file**, not a dir).

### Two non-obvious data gotchas (already handled)

- `**fps=4`, `frames_per_clip=8`** (NOT the config's original 16/5). These **match the upstream
DROID training config** (`droid_8fpcs_fps4_...`). `fps` sets the pose-diff action magnitude;
DROID action-norm is **identity** (mean 0/std 1), so a wrong fps feeds out-of-distribution
action scales and would invalidate CRA. Keep fps=4.
- **Infinite-loop trap:** `DROIDVideoDataset.__getitem__` retries on *any* load failure by
picking a new random index. With a CSV of all-too-short clips it loops forever (we hit this
with a 1-episode test CSV). `build_droid_paths.py` pre-filters to episodes long enough for
`frames_per_clip*ceil(vfps/fps)` frames, so no clip can raise "too short" → no loop.

---

## 4. Stratification recalibration (`stratification/droid_regimes.py`) — IMPORTANT

With the original thresholds, `**contact_manipulation` came out 0 %**. Root cause: DINOv2
ViT-S patch-token L2 over the full grid has a large near-constant component, so per-step latent
change has a **narrow dynamic range** (real DROID-wrist: median ≈ 622, max ≈ 842). The contact
rule required `latent_change > 1.5 × median = 933`, which is **never reached**.

Fix: `CONTACT_LATENT_RATIO` **1.5 → 1.0** (contact = stable gripper **closed** `g>0.5` AND
above-median visual change; pre-grasp = gripper **open** `g<0.3` AND above-median change). This
is encoder-dependent and documented in the module; a higher-dynamic-range encoder (DINOv3
ViT-L) would warrant a larger ratio. The resulting distribution over 2331 transitions:


| regime               | share            |
| -------------------- | ---------------- |
| free_space           | 43.3 % (1010)    |
| pre_grasp            | 22.7 % (528)     |
| gripper_actuation    | 16.3 % (380) — ` |
| contact_manipulation | 17.7 % (413)     |


All four cells clear `min_transitions_per_cell: 100`. **Scientific caveat for the report:**
contact/pre-grasp on DROID-wrist are *proxies* (gripper open/closed + above-median visual
change), not MuJoCo contact GT — consistent with the existing "no object GT, heuristic" framing.

---

## 5. Config state (`configs/diagnostic_droid.yaml`)

- `models: [dino_wm_droid, vjepa2_ac_droid]` (jepa_wm_droid still out — gated weights, §1).
- `dataset.root: data/droid_subset/droid_paths.csv`; `dataset_kwargs: {camera_views: [wrist_mp4_path], frames_per_clip: 8, fps: 4}`.
- `eval.device: cuda`, `batch_size: 16` (ample on the A5000), `min_transitions_per_cell: 100`.
- `negative_strategies: [random, opposite, hard_nn, hard_effect]`, `cra.K: 16`,
trajectory-clustered bootstrap CIs (`n_resamples: 1000`).
- `planning:` block (§8) — CEM params copied from the upstream DROID dino-wm config.

`**hard_effect` (new strategy).** `hard_nn` picks, from similar-state pool candidates, the
action *most different* from `a_t` and ignores the candidate's outcome. `hard_effect` instead
scores similar-state candidates by `||Δz_cand − Δz_factual|| − action_penalty·||a_cand − a_t||`
(both std-normalized) and takes top-K — an action that, from a similar state, leads to a
*genuinely different true future*, preferably while staying *close* to `a_t`. This is the
"precise action matters" negative: harder (fine action resolution) and fairer (the negative's
real outcome differs from `z_{t+1}`, so a grounded model can win the CRA; in smooth free-space
no such negative exists, so it self-selects toward contact/precision). Needs the pool's
next-latents (`pool_z1 = data["z_t1"][pool_indices]`, wired through `evaluate_cell`). Tune via
`hard_effect.action_penalty` (default 0.5; 0 = pure max-effect-divergence). `hard_nn` is kept
so the two definitions can be compared in the same CSV.

---

## 6. The GPU fault (historical — resolved by moving to the A5000)

On the **old 8 GB box**, mid-`05` in `cra_per_transition`, CUDA raised `RuntimeError: CUDA
error: unspecified launch failure` and `nvidia-smi` then reported `Unable to determine the
device handle for GPU0` — the GPU **fell off the bus** (an Xid driver/hardware fault, common on
consumer cards under load, aggravated by sharing the 8 GB card with `tts_stt_service`/`ragflow`).
It was **never** a code defect — the same metric path passes on CPU via `07_validate_synthetic.py`.

**This is no longer relevant on the dedicated A5000** (24 GB, not shared). Kept here only as a
note: if a similar Xid ever recurs, the fix is `reboot`, or `sudo fuser -k /dev/nvidia*` then
`sudo nvidia-smi --gpu-reset -i 0`. There is no longer a need to keep batch size tiny or stop
other services.

---

## 7. State checklist

- [x] Lean Python-3.10 venv (`diagnosis/.venv`) with CUDA torch + DROID/adapter deps.
- [x] `scripts/build_droid_paths.py` — length-filtered episode CSV builder.
- [x] DROID subset downloaded — 333 episodes / ~4.3 GB (`data/droid_subset/`, gitignored).
- [x] `03_extract_latents.py` → `data/precomputed_latents/droid__dino_wm_droid.h5` (~939 MB).
- [x] `04_classify_regimes.py` → regime sidecar; `CONTACT_LATENT_RATIO` recalibrated (1.5→1.0).
- [x] Offline: `pytest` 34/34; `07_validate_synthetic.py` PASS; adapter load+predict OK.
- [x] **Moved to A5000 (24 GB)** — the old 8 GB GPU fault (§6) no longer applies.
- [ ] `03_extract_latents.py` for **`vjepa2_ac_droid`** (now runnable on 24 GB) → its latent cache.
- [ ] `05_run_diagnostic.py` → `results/droid_diagnostic.csv` (both baselines).
- [ ] `terver_gripper_test.py` sanity (expect 2-way CRA > 0.90).
- [ ] `06_analyze_results.py` with **both** CSVs → updated `results/decision_report.md`.
- [ ] **Planning probe** — `08_planning_probe.py` + `09_correlate_planning.py` (§8).
- [ ] (Optional) add `jepa_wm_droid` once gated DINOv3 weights are obtained (§1).

---

## 8. Planning Action-Score probe — closing the CRA_eff → planning-failure link (NEW)

**Why:** the diagnostic shows DROID `CRA_eff ≈ chance` in contact/gripper regimes, but the
jump to "→ low planning success" was *inferential* (we never ran a planner). This probe runs a
real CEM planner on the cached latents, measures the paper's **DROID Action Error**, and
**correlates it per-transition with CRA_eff**. A strong negative correlation turns the
mechanistic argument into evidence. Design + exact upstream parameters:
`docs/plans/2026-06-05-planning-action-score-design.md`.

**Fidelity:** CEM hyper-params, the L2 objective (MSE to goal, last frame), and the Action
Error formula (`|Σ_t planned − Σ_t expert|` grouped xyz/orient/grip) are copied from the
upstream DROID dino-wm config and `plan_evaluator.py` — not invented. Code:
`planning/cem_planner.py`, `metrics/action_score.py`. Tunables live in the `planning:` block of
`configs/diagnostic_droid.yaml`.

**Caveats (record in any writeup):** DROID is offline → Action Score is the paper's proxy, not
a grasp/lift success rate. Action Error compares to a *single* expert trajectory (multimodal →
positive floor), so the **per-transition correlation** is the evidence, not the absolute score.
`max_planning_transitions` bounds CEM cost (`num_samples × iterations` unrolls per transition);
on the A5000 it is set high enough (1100) to cover every transition in each DROID regime.

**Prereq:** `05` must have produced `results/droid_diagnostic.csv` (used by `09` as a
cross-check). Offline correctness is covered by `pytest` (`test_cem_planner.py`,
`test_action_score.py`, `test_planning_probe_synthetic.py` — grounded vs action-ignoring sign
check); only `08` needs the GPU/cache.

```bash
cd <repo>/diagnosis && source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
python scripts/08_planning_probe.py     --config configs/diagnostic_droid.yaml
python scripts/09_correlate_planning.py --planning_csv   results/droid_planning.csv \
                                        --pertrans       results/droid_planning_pertrans.npz \
                                        --diagnostic_csv results/droid_diagnostic.csv
```

Deliverables: `results/droid_planning.csv`, `results/droid_planning_pertrans.npz`,
`results/planning_correlation.md`, `results/figures/figure_c_planning_vs_cra.pdf`.
Expected result if the thesis holds: per-transition Spearman(Action Error, CRA_eff) **clearly
negative**, with contact/gripper regimes showing *both* high Action Error and low CRA_eff.

> **First-pass smoke (optional):** to confirm the pipeline end-to-end before the full run, set
> `planning.max_planning_transitions: 30` and `planning.num_samples: 100`, then restore the
> config values (1100 / 300) for the real number. On the A5000 the full run is affordable, so
> this is only to fail fast on a wiring bug.

