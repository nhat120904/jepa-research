# Handoff — running the Metaworld diagnostic against the real checkpoints

**Status:** All code fixes needed to run the diagnostic against the real frozen `jepa-wms`
checkpoints on a GPU are **done and committed**. Both Metaworld checkpoints
(`jepa_wm_metaworld`, `dino_wm_metaworld`) load + `encode` + `predict` correctly. What
remains is purely operational on whatever server you run this on: build the env, get the
Metaworld dataset, then run the pipeline.

> The previous session ran on a constrained box. This doc is written so you can redo it on a
> **fresh server** without that context. Only the Metaworld primary path was validated;
> DROID/RoboCasa were out of scope (they need a lot of disk — ~150 GB for DROID).

---

## 0. TL;DR — finish in 4 steps

```bash
cd <repo>/diagnosis
# 1. env (see §2) — Python 3.10 venv at diagnosis/.venv, deps installed, HF token exported
# 2. dataset (see §4) — 126 Metaworld parquet shards -> data/hf_mw/metaworld/data/
# 3. gate check:
python scripts/check_normalization.py --config configs/diagnostic_metaworld.yaml --model jepa_wm_metaworld
# 4. full pipeline (detached, see §5):
setsid bash scripts/run_recovery.sh > logs/recovery.log 2>&1 < /dev/null &
```
Deliverable: `results/decision_report.md`. Done marker: `results/.pipeline_done`.

**Before any run, export the env block in §2** (venv, HF token, headless SDL, alloc guards).

---

## 1. Key facts you must know

- **Python 3.10 is required.** Upstream `jepa-wms` pins `<3.11`. Use a 3.10 venv at
  `diagnosis/.venv` (e.g. `uv venv --python 3.10 .venv`).
- **We do NOT `pip install` the upstream repo.** Its `pyproject.toml` pulls fragile sim deps
  (`d4rl`, `mujoco-py`, `gym`, `metaworld`, `dm-control`, `pybullet`) that need compilation
  and aren't needed. The adapters only put the clone on `sys.path`
  (`data.loaders.add_upstream_to_path`). We install just the runtime deps it imports (§2).
- **HF dataset repo `facebook/jepa-wms` is gated** — needs a token with access. Put it in
  `diagnosis/.env` as `HF_TOKEN=...` (already there in the previous box; re-create on a new
  server). `.env` is gitignored — never commit it.
- **Model loading is bypassed around the sim/planning stack** and around a non-public
  decoder head — both handled in code (`models/adapters/_torchhub.py`), so a clean checkout
  + the §2 deps is enough to load weights. You do **not** need gym/pygame/etc. for the
  diagnostic, but the previous env installed them anyway (harmless) — see §3c if curious.
- **DROID / RoboCasa:** out of scope unless you have the disk + time. The frameskip/device
  fixes apply to them too, but each model's `frames_per_step` must be re-derived (it differs
  from Metaworld's 5) — `adapter.frames_per_step` does this automatically.

---

## 2. Environment setup (fresh server)

```bash
cd <repo>/diagnosis
git clone --depth 1 https://github.com/facebookresearch/jepa-wms.git external/jepa-wms
uv venv --python 3.10 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e .          # diagnosis deps incl. torch (CUDA wheel)
# upstream runtime deps the encode/predict + MetaworldHFDataset path actually imports:
VIRTUAL_ENV=$PWD/.venv uv pip install timm einops tensordict omegaconf datasets \
    imageio imageio-ffmpeg decord lpips opencv-python-headless scikit-image termcolor \
    huggingface_hub hydra-core ruamel.yaml gym==0.23.1 pygame pymunk shapely pytest \
    torchcodec
```

> **`torchcodec` is required** (added 2026-06-03). The Metaworld parquet stores frames in a
> `video` column; current `datasets` (>=3.x) decodes it via `torchcodec`, not `decord`. Without
> it the dataset load dies with `ImportError: To support decoding videos, please install
> 'torchcodec'`. It needs system ffmpeg libs (`ffmpeg -version` should work — present on this
> box). The installed `torchcodec==0.13.0` binds against ffmpeg 6.

Create `diagnosis/.env` with `HF_TOKEN=<your gated token>`.

**Env block to export before any run** (the metric scripts need these):
```bash
cd <repo>/diagnosis
set -a; . ./.env; set +a
export HF_TOKEN HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=${HF_HOME:-$PWD/.hf_home}            # keep the checkpoint cache stable
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy  # pygame imports headless
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HDF5_USE_FILE_LOCKING=FALSE
source .venv/bin/activate
```

Sanity that the env is good (no dataset needed):
```bash
.venv/bin/python -m pytest tests/ -q            # expect 23 passed
.venv/bin/python scripts/07_validate_synthetic.py
python scripts/smoke_test.py                    # real checkpoints load + encode + predict
```
`smoke_test.py` downloads the Metaworld checkpoints (+ a ~3.4 GB ViT-L decoder head per
model) and should print `encode OK` / `predict OK` with no errors.

---

## 3. Code changes already made (committed) — WHAT and WHY

If you're just running the pipeline you can skip this. Read it if a load/predict step
misbehaves or you extend to other datasets.

### 3a. Frameskip-aware transitions (the important one)
The Metaworld checkpoints train with **frameskip=5** (`mw_4f_fsk5_ask1`), so
`model_action_dim = action_dim(4) × 5 = 20`: one model step advances **5 frames** and
conditions on a **stack of 5 raw actions**. The original code used 1-frame transitions with
a single 4-dim action → `predict` failed (`shape '[B,-1,20]' invalid`).
- `models/adapters/enc_pred_adapter.py`: exposes `adapter.frames_per_step =
  model_action_dim // action_dim`; `predict`/`predict_rollout` reshape to
  `(B, -1, action_dim)`, normalize each raw action, then fold into `model_action_dim` for
  `unroll` (mirrors `evals/unroll_decode/eval.py`'s `actions.reshape(B,-1,wm.action_dim)`).
- `scripts/05_run_diagnostic.py` (`load_cache_into_tensors(cache, regimes, step)`) and
  `scripts/sanity_check.py` (`check_action_normalization`) build **frameskip-spaced**
  transitions: `z_t=z[k·step]`, `z_t1=z[(k+1)·step]`, `a_t=stack(actions[k·step:(k+1)·step])`,
  regime taken at the start frame.
- `03_extract_latents.py` is **unchanged** — it still encodes every frame; `05` subsamples by
  `step`. So changing frameskip needs no re-encode.
- Metrics layer unchanged (CRA flattens negatives to `(B*K, A)`; 20-dim actions flow through).
  *Minor:* `opposite_negative` only flips the first sub-action's gripper dim — revisit if the
  opposite-strategy numbers look off.

### 3b. CPU/CUDA device fix (`enc_pred_adapter.py`)
The model keeps preprocessor norm stats on **CPU** and normalizes on CPU (`obs.cpu()`) before
moving to device. The adapter now leaves stats on CPU and has `normalize_action` /
`encode_proprio_features` normalize on CPU then `.to(device)`. (Removed the old
`_stats_to_device()` that wrongly pushed stats to GPU and broke the model's internal encode.)

### 3c. Load-path shims (`models/adapters/_torchhub.py`)
- `_install_lightweight_eval_stub()`: upstream `hubconf` does
  `from evals.simu_env_planning.eval import init_module`, which imports the whole sim/planning
  stack (`gym→pygame→pymunk→nevergrad`…) at import time. That module's `init_module` is just a
  dispatcher to `app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds.init_module`. We
  register a stub in `sys.modules` so the heavy module never runs. (Upstream docstring confirms
  envs aren't needed to load weights.)
- `_strip_nonpublic_head_checkpoints()`: `dino_wm_metaworld`'s config wants a `state_head`
  checkpoint at `${JEPAWM_LOGS}/.../jepa-latest.pth.tar` (a non-public training artifact). That
  head is an unused decoder. The shim removes `pretrain_dec_path` entries whose value is an
  unexpanded `${...}` local path from the cached config YAMLs, so `init_module` skips loading
  it. Idempotent.

### 3d. Atomic regime sidecar (`data/latent_cache.py`, `scripts/04`, `scripts/05`)
`04_classify_regimes.py` no longer reopens the big latent `.h5` in append mode (a kill
mid-write could truncate the expensive cache). It opens the cache **read-only** and writes
regimes to an atomic JSON sidecar `data/precomputed_latents/{dataset}__{model}.h5.regimes.json`
via `os.replace`; `05` reads it (`read_regimes`). New helpers: `regime_sidecar_path`,
`write_regimes`, `read_regimes` (exported from `data/`).

### 3e. Config + ops
- `configs/diagnostic_metaworld.yaml`: `max_trajectories_per_task` moved to the **`dataset:`**
  block (it was under `eval:`, which `03` ignores → silently defaulted to 1000); set **60** for
  a quick run (use **300** for the full thesis, then `rm -f data/precomputed_latents/*.h5`).
  `batch_size` 64→16. `dataset.root` → `data/hf_mw/metaworld/data`.
- `scripts/run_recovery.sh` (03→04→05→06) and `scripts/run_diagnostic_smallbatch.sh` (05→06):
  detached-friendly, with `PYTORCH_CUDA_ALLOC_CONF` + `HDF5_USE_FILE_LOCKING` guards and
  `.pipeline_done`/`.diagnostic_done` markers.
  **Caveat:** they `source external/jepa-wms/.venv/bin/activate` if present, but our venv is
  `diagnosis/.venv`. Either run them with `.venv` already active, or update that activation
  line to `source .venv/bin/activate`.

---

## 4. Getting the Metaworld dataset (0.74 GB, 126 parquet shards)

Target layout (this is `dataset.root`): `data/hf_mw/metaworld/data/train-*.parquet`.
`MetaworldHFDataset` reads **all** parquet in that dir then filters by the `task` column, so
all 126 shards are needed even with the 60-trajectory/task cap.

**Preferred (works on a normal HF connection):**
```bash
python external/jepa-wms/src/scripts/download_data.py --dataset metaworld --dataset-root data
# -> data/Metaworld/...  then point dataset.root at the dir containing train-*.parquet
```

> **Task-name convention (fixed 2026-06-03).** The released parquet's `task` column uses
> `mw-<name>` (e.g. `mw-reach`, `mw-peg-insert-side`) — **not** the `<name>-v2` Metaworld env
> ids. `MetaworldHFDataset` filters by **exact match**, so a `-v2` filter silently loads 0
> rollouts (then crashes in `torch.stack([])`). `configs/diagnostic_metaworld.yaml` and
> `HARD_TASKS` in `scripts/06_analyze_results.py` now use the `mw-` names. Also note
> `mw-window-open` is **absent** from the release (only `mw-window-close`); the medium set
> uses window-close. Full list of the 42 available tasks: inspect the `task` column of any
> shard with `pyarrow`.

**If `snapshot_download` / the dataset API fails** (e.g. the previous box had an HF mirror in
`/etc/hosts` that 401'd the dataset API and 0-bytes'd LFS streaming), download per shard with
`hf_hub_download` (same `resolve/` path that serves the checkpoints):
```bash
set -a; . ./.env; set +a; export HF_TOKEN HF_HOME=$PWD/.hf_home
.venv/bin/python - <<'PY'
import os, time
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
for i in range(126):
    fn = f"metaworld/data/train-{i:05d}-of-00126.parquet"
    for attempt in range(5):
        try:
            hf_hub_download(repo_id="facebook/jepa-wms", repo_type="dataset", filename=fn,
                            token=os.environ["HF_TOKEN"], local_dir="data/hf_mw"); break
        except Exception:
            time.sleep(2)
print("shards:", len(__import__("glob").glob("data/hf_mw/metaworld/data/*.parquet")))
PY
```
Idempotent (skips existing files). Verify: `ls data/hf_mw/metaworld/data/*.parquet | wc -l` == 126.
If you used the per-shard loop, `dataset.root` is already `data/hf_mw/metaworld/data` (current
config). If you used `download_data.py`, set `dataset.root` to its `Metaworld/data` dir.

---

## 5. Run the pipeline

```bash
cd <repo>/diagnosis
# (export the §2 env block first, and `source .venv/bin/activate`)
setsid bash -c '
  python scripts/03_extract_latents.py  --config configs/diagnostic_metaworld.yaml &&
  python scripts/04_classify_regimes.py --config configs/diagnostic_metaworld.yaml &&
  python scripts/05_run_diagnostic.py   --config configs/diagnostic_metaworld.yaml &&
  python scripts/06_analyze_results.py  --metaworld_csv results/metaworld_diagnostic.csv &&
  touch results/.pipeline_done
' > logs/pipeline.log 2>&1 < /dev/null &
tail -f logs/pipeline.log    # done when results/.pipeline_done appears
```
(Equivalently fix and use `scripts/run_recovery.sh`.) 03 is the expensive step (encodes every
frame for 2 models × the capped trajectories). 05 with frameskip=5 yields ~19 transitions/traj.

**Sanity expectations**
- Smoke / load: `frames_per_step=5`, `model_action_dim=20`, `uses_proprio=True`, `encode` →
  `(B,1,1,16,16,384)`, `predict` shape matches `z_t`.
- `check_normalization`: factual-action MSE should be clearly below shuffled-action MSE if the
  model uses actions. Where it *isn't*, that's the action-grounding failure the diagnostic is
  built to quantify.

---

## 6. State checklist

- [x] Code fixes (frameskip, device, load-path shims, regime sidecar, config, run scripts) —
      committed.
- [x] Offline: `pytest` 23 passed; `07_validate_synthetic.py` passes.
- [x] Real checkpoints: both Metaworld models load + encode + predict on GPU.
- [ ] Metaworld dataset present on the new server (`data/hf_mw/metaworld/data/`, 126 shards).
- [ ] `check_normalization` on a real transition.
- [ ] Full pipeline → `results/decision_report.md`.
- [ ] (Optional) point `run_recovery.sh` venv activation at `diagnosis/.venv`.
- [ ] (Optional) make `opposite_negative` frameskip-aware for gripper sub-dims.
</content>
</invoke>
