"""Gate 0 worker: one renderer/throughput probe per process.

Run as a subprocess so that a Vulkan abort() kills only this probe, not the driver.
Writes a single JSON file and exits. Never imports MuJoCo or any project code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--obs-mode", default="state", choices=["state", "rgb"])
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--sim-backend", default="gpu")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--measure-steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = {
        "task": args.task,
        "obs_mode": args.obs_mode,
        "num_envs": args.num_envs,
        "sim_backend": args.sim_backend,
        "ok": False,
        "stage_reached": "start",
        "error": None,
        "traceback": None,
        "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
    }

    def dump() -> None:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)

    try:
        import sapien  # noqa: F401

        result["sapien_version"] = getattr(sapien, "__version__", "unknown")
        result["stage_reached"] = "sapien_imported"

        # Defensive introspection: SAPIEN's device-reporting API has moved between
        # versions, so try several names and record whatever exists.
        for attr in ("get_device_summary", "get_devices"):
            fn = getattr(getattr(sapien, "render", None), attr, None)
            if callable(fn):
                try:
                    result[f"sapien_{attr}"] = str(fn())
                except Exception as exc:  # pragma: no cover - introspection only
                    result[f"sapien_{attr}_error"] = repr(exc)

        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import mani_skill

        result["mani_skill_version"] = getattr(mani_skill, "__version__", "unknown")
        result["stage_reached"] = "maniskill_imported"

        import torch

        result["torch_version"] = torch.__version__
        result["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["torch_device_name"] = torch.cuda.get_device_name(0)

        env = gym.make(
            args.task,
            num_envs=args.num_envs,
            obs_mode=args.obs_mode,
            sim_backend=args.sim_backend,
        )
        result["stage_reached"] = "env_created"

        env.reset(seed=0)
        result["stage_reached"] = "env_reset"

        action_space = env.action_space
        for _ in range(args.warmup_steps):
            env.step(action_space.sample())
        result["stage_reached"] = "warmup_done"

        t0 = time.perf_counter()
        for _ in range(args.measure_steps):
            env.step(action_space.sample())
        elapsed = time.perf_counter() - t0

        env_steps = args.measure_steps * args.num_envs
        result["elapsed_seconds"] = round(elapsed, 4)
        result["env_steps"] = env_steps
        result["fps_env_steps"] = round(env_steps / elapsed, 2)
        result["fps_control_steps"] = round(args.measure_steps / elapsed, 2)
        result["stage_reached"] = "measured"
        result["ok"] = True

        env.close()
    except BaseException as exc:  # noqa: BLE001 - we want SystemExit/abort info too
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()

    dump()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
