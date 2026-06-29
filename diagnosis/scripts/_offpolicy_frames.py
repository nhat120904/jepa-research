"""Shared off-policy frame sampler — Phase-3 readout diagnostic + fix.

The Phase-0 ladder localised the contact wall to the OFF-POLICY robustness of the
object readout: the probes are trained on EXPERT cache frames (Test-1b: 92% <5cm) but
the CEM planner scores the latents of arbitrary RANDOM action rollouts, where the
untrained probe degrades. This module generates exactly that off-policy distribution —
states reached by random action sequences, rendered + encoded through the real frozen
encoder, each labelled with the TRUE object and end-effector position from the sim.

Used by scripts/34 (3a off-policy precision diagnostic) and scripts/22 / scripts/19
(`--offpolicy-frac` robust probe training). MetaWorld only (needs sim-state labels);
needs the encoder + EGL render, hence a GPU.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


@torch.no_grad()
def collect_offpolicy_frames(adapter, device, *, tasks, episodes, max_steps=60,
                             collect_every=4, seed=20000, act_std=1.0, batch=64,
                             verbose=True):
    """Roll RANDOM actions in the sim and collect (z, obj, ee, task) at intervals.

    Returns a dict with CPU tensors ``z`` (N, *frame), ``obj`` (N,3), ``ee`` (N,3)
    and a list ``task`` (N,). Each frame is a state reached by a random action prefix
    — i.e. a sample from the CEM exploration distribution the probe is scored on but
    was never trained on."""
    cl = _load("closed_loop_eval", "18_closed_loop_eval.py")
    make_env, render, RAW_A = cl.make_env, cl.render, cl.RAW_A
    from stratification.metaworld_regimes import OBJECT_SLICE, EE_SLICE

    frames, props, objs, ees, tks = [], [], [], [], []
    for task in tasks:
        for e in range(episodes):
            env, _ = make_env(task, seed + e)
            obs, _ = env.reset()
            obs, _, _, _, _ = env.step(np.zeros(RAW_A))          # upstream warmup
            rng = np.random.default_rng((seed * 7 + e * 131 + (hash(task) % 997)) & 0xFFFFFFFF)
            for t in range(max_steps):
                a = np.clip(rng.normal(0.0, act_std, size=RAW_A), -1.0, 1.0)
                obs, _, _, _, _ = env.step(a)
                if t % collect_every == 0:
                    frames.append(render(env))
                    props.append(obs[:4].astype(np.float32))
                    objs.append(obs[OBJECT_SLICE].astype(np.float32))
                    ees.append(obs[EE_SLICE].astype(np.float32))
                    tks.append(task)
            env.close()

    uses_p = adapter.uses_proprio()
    zs = []
    for lo in range(0, len(frames), batch):
        fb, pb = frames[lo:lo + batch], props[lo:lo + batch]
        vis = torch.stack([torch.from_numpy(f.copy()).permute(2, 0, 1).float()
                           for f in fb]).unsqueeze(1)          # (b,1,C,H,W)
        prop = None
        if uses_p:
            prop = torch.stack([torch.from_numpy(p) for p in pb]).unsqueeze(1).to(device)
        z = adapter.encode(vis.to(device), prop)[:, 0]         # (b, *frame)
        zs.append(z.cpu())
        if verbose:
            print(f"  off-policy encode {min(lo + batch, len(frames))}/{len(frames)}",
                  flush=True)

    return {"z": torch.cat(zs), "obj": torch.tensor(np.stack(objs)),
            "ee": torch.tensor(np.stack(ees)), "task": tks}
