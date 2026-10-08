"""Check the local event_wm environment: GPU, packages, OGBench envs, dataset files.

    .venv\\Scripts\\python.exe local\\check_env.py [--no-render]
Exit code 0 only if everything required for the STATE track works.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
DATA = Path(os.environ.get("EVENT_WM_DATA", r"E:\jepa-data\ogbench\data"))
STATE = ["cube-triple-play-v0", "puzzle-4x5-play-v0", "scene-play-v0"]
EXTRA_STATE = ["puzzle-3x3-play-v0", "puzzle-4x4-play-v0", "puzzle-4x6-play-v0"]
VISUAL = ["visual-cube-triple-play-v0", "visual-scene-play-v0", "visual-puzzle-3x3-play-v0", "visual-puzzle-4x4-play-v0",
          "visual-puzzle-4x5-play-v0", "visual-puzzle-4x6-play-v0"]
ok = True


def row(name: str, good: bool, detail: str = "", optional: bool = False) -> None:
    global ok
    ok &= good or optional
    tag = "OK " if good else ("-- " if optional else "FAIL")
    print(f"[{tag}] {name:34s} {detail}", flush=True)


def npz_shapes(path: Path) -> dict[str, tuple]:
    """Array shapes of an .npz without decompressing the data."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            with z.open(n) as f:
                np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
                out[n[:-4]] = shape
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true", help="skip the pixel-env rendering test")
    a = ap.parse_args()

    print(f"python {sys.version.split()[0]}  exe={sys.executable}")
    import torch
    cuda = torch.cuda.is_available()
    row("torch", cuda, f"{torch.__version__} cuda={torch.version.cuda} avail={cuda}")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        x = torch.randn(1024, 1024, device="cuda")
        row("gpu matmul", bool(torch.isfinite(x @ x).all()), f"{p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f} GiB")
    import mujoco, gymnasium, ogbench, scipy  # noqa: F401
    row("packages", True, f"mujoco {mujoco.__version__} gymnasium {gymnasium.__version__} numpy {np.__version__} scipy {scipy.__version__}")

    for name in ["cube-triple-v0", "puzzle-4x5-v0", "scene-v0"]:
        try:
            t = time.time()
            env = gymnasium.make(name)
            ob, info = env.reset(seed=0, options=dict(task_id=1))
            for _ in range(5):
                ob, *_ = env.step(env.action_space.sample())
            row(f"env {name}", True, f"obs {ob.shape} act {env.action_space.shape} {time.time() - t:.1f}s")
            env.close()
        except Exception as e:  # noqa: BLE001
            row(f"env {name}", False, repr(e)[:150])
    if not a.no_render:
        try:
            env = gymnasium.make("visual-cube-triple-v0")
            ob, info = env.reset(seed=0, options=dict(task_id=1))
            row("pixel env render", ob.shape[-1] == 3 and ob.mean() > 20, f"visual-cube-triple-v0 obs {ob.shape} mean {ob.mean():.0f}")
            env.close()
        except Exception as e:  # noqa: BLE001
            row("pixel env render", False, repr(e)[:150] + "  (only needed for visual/pixel tracks)")

    print(f"\ndata dir: {DATA}")
    for group, names, required in [("state(unified)", STATE, True), ("state(puzzle sizes)", EXTRA_STATE, False), ("visual", VISUAL, False)]:
        for n in names:
            for suffix in ("", "-val"):
                f = DATA / f"{n}{suffix}.npz"
                if not f.exists():
                    row(f"{group}: {n}{suffix}", False, "missing", optional=not required)
                    continue
                try:
                    sh = npz_shapes(f)
                    obs = sh.get("observations", sh.get("obs"))
                    n_ep = None
                    row(f"{group}: {n}{suffix}", True, f"{f.stat().st_size / 1e9:.2f} GB  observations {obs}  keys {len(sh)}")
                except Exception as e:  # noqa: BLE001
                    row(f"{group}: {n}{suffix}", False, f"unreadable: {e!r}"[:120])
    free = shutil.disk_usage(DATA.anchor or "E:\\").free / 1e9
    row("disk free", free > 20, f"{free:.0f} GB on {DATA.anchor}")
    print("\nSTATE-TRACK READY" if ok else "\nPROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
