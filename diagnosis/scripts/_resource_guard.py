"""Lightweight resource preflight checks for heavy checkpoints."""

from __future__ import annotations

import os
import subprocess


HEAVY_MODEL_MIN_GPU_GIB = {
    # V-JEPA-2 ViT-G loads a separate ~15 GB base encoder plus the DROID
    # action-conditioned checkpoint. The DROID handoff assumes a 24 GB A5000;
    # on 12 GB desktop GPUs this can stall the whole machine before Python can
    # raise a clean CUDA OOM.
    "vjepa2_ac_droid": 22.0,
}


def _cuda_total_memory_gib() -> float | None:
    """Return total GPU memory via nvidia-smi without constructing a model."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    values = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line) / 1024.0)
        except ValueError:
            return None
    return max(values) if values else None


def preflight_model_load(model_name: str, device: str = "cuda") -> None:
    """Fail fast before loading known-heavy models on undersized local GPUs."""
    min_gib = HEAVY_MODEL_MIN_GPU_GIB.get(model_name)
    if min_gib is None:
        return
    if os.environ.get("CAI_JEPA_ALLOW_HEAVY_MODEL") == "1":
        return
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            f"{model_name} is a heavy ViT-G checkpoint. Refusing to load it on "
            f"device={device!r} without CAI_JEPA_ALLOW_HEAVY_MODEL=1."
        )

    total_gib = _cuda_total_memory_gib()
    if total_gib is None:
        raise RuntimeError(
            f"Could not verify GPU memory before loading {model_name}. Set "
            "CAI_JEPA_ALLOW_HEAVY_MODEL=1 only on the intended 24 GB GPU server."
        )
    if total_gib < min_gib:
        raise RuntimeError(
            f"Refusing to load {model_name}: detected GPU has {total_gib:.1f} GiB "
            f"total memory, but this run expects at least {min_gib:.1f} GiB. Run "
            "on the 24 GB server, or set CAI_JEPA_ALLOW_HEAVY_MODEL=1 if you are "
            "intentionally overriding the safety guard."
        )
