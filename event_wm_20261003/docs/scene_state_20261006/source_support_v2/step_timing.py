#!/usr/bin/env python3
"""Time one closed-loop step: env.step (physics + render) and the model calls, per MUJOCO_GL backend."""
import json, os, sys, time
from pathlib import Path
import numpy as np


def main():
    import gymnasium, ogbench  # noqa: F401
    import torch
    env = gymnasium.make("visual-puzzle-4x5-v0")
    ob, info = env.reset(seed=0, options=dict(task_id=1))
    t0 = time.time()
    for _ in range(200):
        ob, *_ = env.step(env.action_space.sample())
    t_env = (time.time() - t0) / 200
    u = env.unwrapped
    t0 = time.time()
    for _ in range(200):
        u.compute_observation()
    t_render = (time.time() - t0) / 200
    rep = {"backend": os.environ.get("MUJOCO_GL"), "env_step_s": t_env, "render_only_s": t_render}
    if len(sys.argv) > 1:
        from common import load_code, tokens
        enc, logits, _ = load_code(Path(sys.argv[1]), "cuda")
        for _ in range(5):
            logits(tokens(enc, ob[None], "cuda"))
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(200):
            (logits(tokens(enc, ob[None], "cuda")) > 0).cpu()
        rep["observe_s"] = (time.time() - t0) / 200
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
