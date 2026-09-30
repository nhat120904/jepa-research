#!/usr/bin/env bash
#SBATCH --job-name=ti_ogb
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/ogb_%x_%j.out
# OGBench arena for CTA (docs/CTA_OGBENCH_PLAN.md). Official HIQL = baseline and subgoal proposal source.
# GPU modes: sbatch --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --time=... slurm_ogb.sh <mode>
#   setup (main, CPU)  venv (uv, python 3.11, OGBench impls requirements), ogbench repo at a pinned commit,
#                      visual-cube-single-play-v0 train/val from the HF mirror (official Berkeley server blocked)
#   smoke (mig GPU)    JAX sees the GPU, EGL renders, 1k HIQL steps + 2 eval episodes per task
#   hiql  (mig GPU)    official visual-cube-single-play HIQL command (impls/hyperparameters.sh), seed $SEED,
#                      500k steps, eval every 100k (50 episodes x 5 tasks), checkpoints every 100k
#   gcivl (mig GPU)    official visual-cube-single-play GCIVL command, seed $SEED: the self-supervised teacher value
#                      V(o, g) for the label-free CTA reader (docs/CTA_OGBENCH_PROTOCOL.md); same schedule as hiql
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
MODE=${1:?mode}
SEED=${SEED:-0}
BASE=/mnt/data/nhatnc129/jepa/ogbench
VENV="$BASE/venv"; REPO="$BASE/ogbench"; DATA="$BASE/data"
COMMIT=1d4140997f60c52c6fb0702ec100dc988b18c548
ENV_NAME=${ENV_NAME:-visual-cube-single-play-v0}
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85 PYTHONUNBUFFERED=1
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID MODE=$MODE $(date -u +%FT%TZ)"
case "$MODE" in
  setup)
    mkdir -p "$BASE" "$DATA"
    [[ -d "$REPO" ]] || git clone https://github.com/seohongpark/ogbench "$REPO"
    git -C "$REPO" fetch --quiet origin && git -C "$REPO" checkout --quiet "$COMMIT"
    # wandb pinned: OGBench log_utils passes wandb.Settings(start_method=...), rejected by wandb >= 0.19
    [[ -x "$VENV/bin/python" ]] || ~/.local/bin/uv venv --python 3.11 "$VENV"
    ~/.local/bin/uv pip install --python "$VENV/bin/python" --link-mode=copy \
      "jax[cuda12]==0.4.35" "flax==0.10.2" "distrax==0.1.5" "tensorflow-probability==0.24.0" ml_collections \
      "ogbench==1.2.1" matplotlib moviepy "wandb==0.17.9" huggingface_hub
    "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || true
    ~/.local/bin/uv pip freeze --python "$VENV/bin/python" > "$BASE/pip_freeze.txt"
    "$VENV/bin/python" - <<PY
import hashlib, json, numpy as np
from huggingface_hub import hf_hub_download
out = {}
for name in ("$ENV_NAME.npz", "$ENV_NAME-val.npz"):
    p = hf_hub_download("ryanhoangt/ogbench_data", name, repo_type="dataset", local_dir="$DATA")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    with np.load(p) as z:
        out[name] = {"sha256": h, "keys": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}}
json.dump(out, open("$BASE/data_manifest_$ENV_NAME.json", "w"), indent=2)
print(json.dumps(out, indent=2))
PY
    mkdir -p ~/.ogbench && ln -sfn "$DATA" ~/.ogbench/data      # ogbench's default dataset_dir
    ;;
  smoke|hiql|gcivl)
    # jax 0.4.35 reads nvidia.cuda_nvcc.__file__, which is None for the uv-installed namespace package; point it at
    # the pip CUDA toolkit directly (bin/ptxas, nvvm/libdevice)
    export CUDA_ROOT="$VENV/lib/python3.11/site-packages/nvidia/cuda_nvcc"
    cd "$REPO/impls"
    git -C "$REPO" rev-parse HEAD
    "$VENV/bin/python" -c "import jax; print('jax', jax.__version__, jax.devices())"
    # evaluation (250 episodes, ~1.3 h on a MIG slice) costs more than 100k training steps (~1.1 h): default is one
    # evaluation at the end (plus the step-1 evaluation main.py always runs); EVAL_INTERVAL overrides
    STEPS=500000; EVAL=${EVAL_INTERVAL:-500000}; SAVE=100000; EPIS=50; GROUP=cta_hiql
    if [[ "$MODE" == smoke ]]; then STEPS=1000; EVAL=1000; SAVE=1000; EPIS=2; GROUP=cta_smoke; fi
    AGENT=(--agent=agents/hiql.py --agent.batch_size=256 --agent.encoder=impala_small --agent.high_alpha=3.0
           --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.p_aug=0.5 --agent.subgoal_steps=10)
    if [[ "$MODE" == gcivl ]]; then GROUP=cta_gcivl
      AGENT=(--agent=agents/gcivl.py --agent.alpha=10.0 --agent.batch_size=256 --agent.encoder=impala_small
             --agent.p_aug=0.5); fi
    "$VENV/bin/python" /home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922/scripts/ogbench/run_offline.py --run_group="$GROUP" --seed="$SEED" --env_name="$ENV_NAME" \
      --save_dir="$BASE/exp" --train_steps="$STEPS" --eval_interval="$EVAL" --save_interval="$SAVE" \
      --eval_episodes="$EPIS" --video_episodes=0 --eval_on_cpu=0 "${AGENT[@]}"
    find "$BASE/exp" -name eval.csv -newer "$BASE/pip_freeze.txt" -exec sh -c 'echo "== $1"; tail -3 "$1"' _ {} \;
    ;;
  gcfbc_train|gcfbc_eval|gcfbc_distill)
    # proposal policy for CTA (scripts/ogbench/gcfbc.py), PyTorch in the CTA venv (ogbench 1.2.1, mujoco, torch)
    CTA_PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
    PROJ=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
    OUT=/mnt/data/nhatnc129/jepa/ogbench/gcfbc/${ENV_NAME}_${SLURM_JOB_ID}
    mkdir -p "$OUT"
    cp -r "$PROJ/scripts/ogbench" "$PROJ/ti_wm" "$OUT/" 2>/dev/null || true
    "$CTA_PY" -m unittest "$PROJ/cta_tests/test_ogb_gcfbc.py"
    if [[ "$MODE" == gcfbc_distill ]]; then
      "$CTA_PY" "$PROJ/scripts/ogbench/gcfbc.py" distill --run "$OUT" --env "$ENV_NAME" --checkpoint "${GC_CKPT:?GC_CKPT}" \
        --collect "${COLLECT_DIR:?}" --steps "${GC_STEPS:-50000}"
    elif [[ "$MODE" == gcfbc_train ]]; then
      "$CTA_PY" "$PROJ/scripts/ogbench/gcfbc.py" train --run "$OUT" --env "$ENV_NAME" ${GC_STEPS:+--steps "$GC_STEPS"}
    else
      "$CTA_PY" "$PROJ/scripts/ogbench/gcfbc.py" eval --run "$OUT" --env "$ENV_NAME" --checkpoint "${GC_CKPT:?GC_CKPT}" \
        --arms "${GC_ARMS:-P0,ORACLE8,ORACLE16}" --episodes "${GC_EPIS:-20}"
    fi
    ;;
  collect)
    # branched CTA data (scripts/ogbench/ogb_collect.py); array task index = OGBench task id - 1
    CTA_PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
    PROJ=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
    # CPU partition: rendering is software either way on this cluster; array index -> (task, part)
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    TASK=$((SLURM_ARRAY_TASK_ID % 5 + 1)); PART=$((SLURM_ARRAY_TASK_ID / 5))
    OUT=/mnt/data/nhatnc129/jepa/ogbench/collect/${ENV_NAME}_${SLURM_ARRAY_JOB_ID}
    mkdir -p "$OUT/code_$SLURM_ARRAY_TASK_ID"; cp -r "$PROJ/scripts/ogbench" "$PROJ/ti_wm" "$OUT/code_$SLURM_ARRAY_TASK_ID/"
    "$CTA_PY" "$PROJ/scripts/ogbench/ogb_collect.py" --out "$OUT" --env "$ENV_NAME" --checkpoint "${GC_CKPT:?GC_CKPT}" \
      --task "$TASK" --first "$(( ${COLLECT_FIRST:-1000} + PART * ${COLLECT_COUNT:?COLLECT_COUNT} ))" --count "$COLLECT_COUNT"
    ;;
  encode|cta_train|cta_eval)
    # CTA on the branched data: DINOv2/PCA tokens -> CTA + matched rerankers -> closed loop (docs/CTA_OGBENCH_PROTOCOL.md)
    CTA_PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
    PROJ=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
    export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
    "$CTA_PY" -m unittest discover -s "$PROJ/cta_tests" -p "test_cta_ogb.py"
    case "$MODE" in
      encode) "$CTA_PY" "$PROJ/scripts/ogbench/ogb_encode.py" --collect "${COLLECT_DIR:?}" --out "${ENC_DIR:?}" ;;
      cta_train) "$CTA_PY" "$PROJ/scripts/ogbench/ogb_cta_train.py" --run "${TRAIN_DIR:?}" --data "${ENC_DIR:?}" \
                   ${CTA_ARGS:-} ;;
      cta_eval) "$CTA_PY" "$PROJ/scripts/ogbench/ogb_cta_eval.py" --run "${EVAL_DIR:?}" --env "$ENV_NAME" \
                  --policy "${GC_CKPT:?}" --ckpt "${TRAIN_DIR:?}/cta_ogb.pt" --pca "${ENC_DIR:?}/pca.pt" ${EVAL_ARGS:-} ;;
    esac
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
