"""LIBERO L0 render probe (docs/LIBERO_QUALIFICATION_PROTOCOL.md). CPU only, MUJOCO_GL=osmesa.

Creates every LIBERO-Goal task, resets to init state 0, steps 50 zero/random actions with both
cameras rendered, and reports frame statistics and per-step timing.
"""

import argparse
import importlib.metadata as md
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

MAX_STEP_MS = 150.0
STEPS, WARMUP = 50, 5


def write_libero_config(cfg_dir):
    """LIBERO asks interactively for paths when its config is missing; write it up front."""
    pkg = Path(importlib.util.find_spec("libero").submodule_search_locations[0]) / "libero"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    paths = {"benchmark_root": pkg, "bddl_files": pkg / "bddl_files", "init_states": pkg / "init_files",
             "datasets": cfg_dir / "datasets", "assets": pkg / "assets"}
    (cfg_dir / "config.yaml").write_text("".join(f"{k}: {v}\n" for k, v in paths.items()))
    return {k: str(v) for k, v in paths.items()}


def versions():
    out = {}
    for name in ("lerobot", "hf-libero", "libero", "robosuite", "mujoco", "torch", "numpy"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def probe_task(benchmark, get_libero_path, env_cls, suite, i, rng):
    task = suite.get_task(i)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = env_cls(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        env.reset()
        init_states = suite.get_task_init_states(i)
        obs = env.set_init_state(init_states[0])
        times, frames = [], []
        for s in range(STEPS + WARMUP):
            action = np.zeros(7) if s % 2 == 0 else rng.uniform(-0.3, 0.3, 7)
            action[-1] = -1.0
            t0 = time.perf_counter()
            obs, _, _, _ = env.step(action)
            if s >= WARMUP:
                times.append(time.perf_counter() - t0)
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            frames.append(obs[key])
        return {"task": task.name, "language": task.language, "init_states": int(len(init_states)),
                "step_ms_mean": 1000 * float(np.mean(times)), "step_ms_p90": 1000 * float(np.percentile(times, 90)),
                "frame_shapes": [list(f.shape) for f in frames], "frame_std": [float(f.std()) for f in frames],
                "ok": True}, frames
    finally:
        env.close()


def main(run):
    assert os.environ.get("SLURM_JOB_ID"), "Run through sbatch"
    report = {"status": "RUNNING", "versions": versions(), "MUJOCO_GL": os.environ.get("MUJOCO_GL")}
    try:
        report["libero_paths"] = write_libero_config(Path(os.environ["LIBERO_CONFIG_PATH"]))
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        report["n_tasks"] = suite.n_tasks
        rng = np.random.default_rng(0)
        tasks, samples = [], {}
        for i in range(suite.n_tasks):
            try:
                res, frames = probe_task(benchmark, get_libero_path, OffScreenRenderEnv, suite, i, rng)
                samples[f"task{i}_agentview"], samples[f"task{i}_wrist"] = frames
            except Exception:
                res = {"ok": False, "error": traceback.format_exc()}
            tasks.append(res)
            print(json.dumps({i: {k: v for k, v in res.items() if k != "error"}}), flush=True)
        report["tasks"] = tasks
        np.savez_compressed(run / "sample_frames.npz", **samples)
        ok = [t for t in tasks if t["ok"]]
        worst = max((t["step_ms_mean"] for t in ok), default=float("inf"))
        rendered = all(min(t["frame_std"]) > 0 for t in ok)
        report["worst_step_ms_mean"] = worst
        report["verdict"] = ("PASS" if len(ok) == suite.n_tasks == 10 and rendered and worst <= MAX_STEP_MS
                             else "FAIL")
        report["status"] = "DONE"
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (run / "l0_report.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({k: v for k, v in report.items() if k != "tasks"}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run)
