# Handoff — DROID secondary diagnostic (real Franka, 8 GB-VRAM box)

**Status:** Everything up to the metric step is **done and cached on disk**. The env is built,
a 333-episode DROID subset is downloaded, latents are encoded (`03`), and regimes are
classified + recalibrated (`04`). The only step left is `05` (CRA/AUG/ECS) + `06` (report).
`05` was started but the **GPU fell off the bus mid-run** (hardware/driver Xid, not a code
bug — see §6). Offline validation (`pytest` 23/23, `07_validate_synthetic.py`) passes, so the
metric code path that crashed is proven correct; it just needs a healthy GPU (or `device: cpu`).

> Context: this ran on the user's **RTX 3070 Ti, 8 GB VRAM** desktop (shared with
> `tts_stt_service`/`ragflow`), not a server. That constraint drove the model selection (§1).
> The latent cache means a re-run only needs `05`+`06` — **no re-encode, no re-download.**

---

## 0. TL;DR — finish in 2 commands (after the GPU is healthy)

```bash
cd <repo>/diagnosis && source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
python scripts/05_run_diagnostic.py  --config configs/diagnostic_droid.yaml
python scripts/06_analyze_results.py --metaworld_csv results/metaworld_diagnostic.csv \
                                     --droid_csv     results/droid_diagnostic.csv
```
Deliverable: `results/droid_diagnostic.csv` + an updated `results/decision_report.md`.

If you don't want to fix the GPU first, set `eval.device: cpu` in
`configs/diagnostic_droid.yaml` — `05` then runs entirely off the cached latents (slower, no
GPU). Latents are already encoded, so CPU only does the predictor forward passes.

**Recommended sanity gate before trusting numbers** (CLAUDE.md #6): the Terver gripper test
(open-vs-close on real DROID, expect 2-way CRA > 0.90):
```bash
python scripts/terver_gripper_test.py --config configs/diagnostic_droid.yaml --model dino_wm_droid
```

---

## 1. Model selection — only `dino_wm_droid` runs on 8 GB (decision + why)

The DROID config originally listed three baselines; two are **dropped on this box**:

| hub id | backbone | verdict on 8 GB |
|---|---|---|
| `dino_wm_droid` | DINOv2 **ViT-S/14** | ✅ **runs** — load+encode+predict peak **1.44 GB** |
| `vjepa2_ac_droid` | V-JEPA-2 **ViT-G/16** (~1B) | ❌ does not fit 8 GB VRAM |
| `jepa_wm_droid` | DINOv3 **ViT-L/16** | ❌ needs **gated** DINOv3 `.pth` weights (Meta request-form). HF mirror only ships `model.safetensors`; the upstream `app/plan_common/models/dino.py` loads the original `.pth` via a local `~/dinov3` hub clone (`$JEPAWM_HOME/dinov3`, `$JEPAWM_OSSCKPT/dinov3/...pth`) → format + gating blocker. |

`dino_wm_droid` is the canonical DINO-WM baseline, so this still gives a real DROID result.
To add `jepa_wm_droid` later: get gated DINOv3 ViT-L weights, clone `facebookresearch/dinov3`
to `~/dinov3`, set `JEPAWM_OSSCKPT` to the dir holding `dinov3/dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth`,
then re-add it to `models:` in the config. (`vjepa2_ac_droid` needs a bigger GPU regardless.)

**Why DROID matters even with one model:** DROID has a **real gripper signal**
(`gripper_position`), so the `gripper_actuation` and `contact_manipulation` regimes populate —
exactly the cells the Metaworld HF proxy left at 0 % (Metaworld `gripper_actuation` had 0
effect transitions). This is the main value-add over the Metaworld-only result.

---

## 2. Environment (built on this box — `diagnosis/.venv`, Python 3.10)

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
python -m pytest tests/ -q              # 23 passed
python scripts/07_validate_synthetic.py # PASS (PerfectModel CRA=1.0, ActionIgnoring=chance)
```

---

## 3. DROID data — how the 333-episode subset was built

DROID has **no HF download** (the upstream `download_data.py` covers pusht/pointmaze/wall/
metaworld/robocasa/franka, not droid) and the raw bucket is **5.6 TB**. We pulled a small
public subset by hand:

1. **Anonymous gsutil** from the public bucket (no auth, `BOTO_CONFIG=/dev/null`):
   `gs://gresearch/robotics/droid_raw/1.0.1/<LAB>/success/<DATE>/<EPISODE>/`. Each episode dir
   has `metadata_*.json`, `trajectory.h5` (~0.66 MB), `trajectory_im128.h5` (~47 MB, **skip**),
   and `recordings/MP4/<serial>.mp4` (3 non-stereo cams ~1–3 MB each) + `-stereo.mp4` (**skip**).
2. **`rsync` two date dirs** (2 labs, for diversity), excluding stereo/im128/svo:
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
- **`fps=4`, `frames_per_clip=8`** (NOT the config's original 16/5). These **match the upstream
  DROID training config** (`droid_8fpcs_fps4_...`). `fps` sets the pose-diff action magnitude;
  DROID action-norm is **identity** (mean 0/std 1), so a wrong fps feeds out-of-distribution
  action scales and would invalidate CRA. Keep fps=4.
- **Infinite-loop trap:** `DROIDVideoDataset.__getitem__` retries on *any* load failure by
  picking a new random index. With a CSV of all-too-short clips it loops forever (we hit this
  with a 1-episode test CSV). `build_droid_paths.py` pre-filters to episodes long enough for
  `frames_per_clip*ceil(vfps/fps)` frames, so no clip can raise "too short" → no loop.

---

## 4. Stratification recalibration (`stratification/droid_regimes.py`) — IMPORTANT

With the original thresholds, **`contact_manipulation` came out 0 %**. Root cause: DINOv2
ViT-S patch-token L2 over the full grid has a large near-constant component, so per-step latent
change has a **narrow dynamic range** (real DROID-wrist: median ≈ 622, max ≈ 842). The contact
rule required `latent_change > 1.5 × median = 933`, which is **never reached**.

Fix: `CONTACT_LATENT_RATIO` **1.5 → 1.0** (contact = stable gripper **closed** `g>0.5` AND
above-median visual change; pre-grasp = gripper **open** `g<0.3` AND above-median change). This
is encoder-dependent and documented in the module; a higher-dynamic-range encoder (DINOv3
ViT-L) would warrant a larger ratio. The resulting distribution over 2331 transitions:

| regime | share |
|---|---|
| free_space | 43.3 % (1010) |
| pre_grasp | 22.7 % (528) |
| gripper_actuation | 16.3 % (380) — `|Δgripper| > 0.2` |
| contact_manipulation | 17.7 % (413) |

All four cells clear `min_transitions_per_cell: 100`. **Scientific caveat for the report:**
contact/pre-grasp on DROID-wrist are *proxies* (gripper open/closed + above-median visual
change), not MuJoCo contact GT — consistent with the existing "no object GT, heuristic" framing.

---

## 5. Config state (`configs/diagnostic_droid.yaml`)

- `models: [dino_wm_droid]` (the other two commented with the reason from §1).
- `dataset.root: data/droid_subset/droid_paths.csv`; `dataset_kwargs: {camera_views:
  [wrist_mp4_path], frames_per_clip: 8, fps: 4}`.
- `eval.device: cuda` (switch to `cpu` to run with no GPU — see §0), `batch_size: 8`,
  `min_transitions_per_cell: 100`.
- `negative_strategies: [random, opposite, hard_nn]`, `cra.K: 16`, trajectory-clustered
  bootstrap CIs (`n_resamples: 1000`).

---

## 6. The GPU fault (what happened, how to recover)

Mid-`05`, in `cra_per_transition`, CUDA raised `RuntimeError: CUDA error: unspecified launch
failure`, and `nvidia-smi` then reported `Unable to determine the device handle for GPU0:
Unknown Error` — i.e. the GPU **fell off the bus** (an Xid driver/hardware fault, common on
consumer cards under load, here aggravated by sharing the 8 GB card with `tts_stt_service`/
`ragflow`). This is **not** a code defect — the same metric path passes on CPU via
`07_validate_synthetic.py`.

Recovery (needs root, which the agent did not have):
- Simplest: **reboot**.
- Or, without reboot: free the GPU and reset —
  ```bash
  sudo fuser -k /dev/nvidia*                       # kills GPU-holding procs (tts_stt/ragflow)
  sudo nvidia-smi --gpu-reset -i 0                 # or reload modules:
  # sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
  ```
To lower the chance of a repeat on the re-run: keep `eval.batch_size` small (8), and consider
running while the other GPU services are stopped.

---

## 7. State checklist

- [x] Lean Python-3.10 venv (`diagnosis/.venv`) with CUDA torch + DROID/adapter deps.
- [x] `scripts/build_droid_paths.py` (NEW) — length-filtered episode CSV builder.
- [x] DROID subset downloaded — 333 episodes / ~4.3 GB (`data/droid_subset/`, gitignored).
- [x] `03_extract_latents.py` → `data/precomputed_latents/droid__dino_wm_droid.h5` (~939 MB).
- [x] `04_classify_regimes.py` → regime sidecar; `CONTACT_LATENT_RATIO` recalibrated (1.5→1.0).
- [x] Offline: `pytest` 23/23; `07_validate_synthetic.py` PASS; adapter load+predict on CPU OK.
- [ ] **GPU recovery** (reboot / `--gpu-reset`).
- [ ] `05_run_diagnostic.py` → `results/droid_diagnostic.csv` (GPU or `device: cpu`).
- [ ] `terver_gripper_test.py` sanity (expect 2-way CRA > 0.90).
- [ ] `06_analyze_results.py` with **both** CSVs → updated `results/decision_report.md`.
- [ ] (Optional) add `jepa_wm_droid` once gated DINOv3 weights are obtained (§1).
