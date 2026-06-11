"""Diagnose why closed-loop reach success (6%) is far below the paper.

Discriminates the two suspects in one run:
  A) VISUAL DOMAIN SHIFT — today's gymnasium/MuJoCo renderer draws a scene the
     frozen encoder/predictor was not trained on. Symptoms: rendered frames'
     latents far from the dataset manifold; one-step prediction error much
     higher on rendered transitions than on dataset transitions.
  B) PROTOCOL — env/goal handling. If A comes back clean, the bug is here.

Measurements (all through the production adapter path, mirrors 18):
  1. one-step predict MSE on DATASET transitions   (reference: the model's home turf)
  2. one-step predict MSE on RENDERED expert transitions (same predict() call)
  3. latent NN distance: rendered frames -> dataset manifold, vs within-dataset NN
  4. side-by-side PNG (dataset frame vs render at episode start) + channel stats

Usage:
  python scripts/_baseline_probe.py --config configs/diagnostic_metaworld.yaml \
      --model dino_wm_metaworld --task mw-reach
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders import iterate_metaworld_trajectories  # noqa: E402
from models.adapters import build_adapter  # noqa: E402

FRAMESKIP = 5
RAW_A = 4


def flat(z):  # (.., V,H,W,D) -> vector
    return z.reshape(z.shape[0], -1) if z.ndim > 2 else z


def to_model_steps(frames, proprio, actions):
    """Raw-rate trajectory -> model-step series (mirrors 05's cache convention):
    frames/proprio every FRAMESKIP frames, actions = FRAMESKIP-stacked raws."""
    La = actions.shape[0]
    n = La // FRAMESKIP
    idx0 = torch.arange(n) * FRAMESKIP
    gather = idx0.unsqueeze(1) + torch.arange(FRAMESKIP)
    f = frames[torch.cat([idx0, idx0[-1:] + FRAMESKIP])]
    p = proprio[torch.cat([idx0, idx0[-1:] + FRAMESKIP])]
    a = actions[gather].reshape(n, -1)
    return f, p, a


@torch.no_grad()
def one_step_mse(adapter, frames, proprio, actions, device, max_n=40):
    """frames (T,C,H,W) [0,255], proprio (T,P), actions (T-1, A_model raw stack).
    Returns per-transition MSE list via adapter.predict (the planner primitive)."""
    n = min(max_n, actions.shape[0])
    z = adapter.encode(frames[None, :n + 1].to(device),
                       proprio[None, :n + 1].to(device))[0]      # (n+1,V,H,W,D)
    errs = []
    for t in range(n):
        pred = adapter.predict(z[t:t + 1], actions[t:t + 1].to(device),
                               proprio_t=proprio[t:t + 1].to(device))
        errs.append(float(((pred - z[t + 1]) ** 2).mean()))
    return np.array(errs), z


def rollout_expert_modelsteps(task: str, seed: int, max_raw: int = 100,
                              cam_tweak: bool = True):
    """Roll the scripted expert; return frames every FRAMESKIP raw steps,
    proprio at those frames, and the 5-stacked raw actions between them."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module
    cl = import_module("18_closed_loop_eval")  # reuse make_env/render/expert

    env, obs = cl.make_env(task, seed, cam_tweak=cam_tweak)
    pol = cl.expert_policy(task)
    frames = [cl.render(env)]
    proprios = [obs[:4].copy()]
    acts, chunk = [], []
    for t in range(max_raw):
        a = np.clip(pol.get_action(obs), -1, 1)
        obs, _, _, _, info = env.step(a)
        chunk.append(a)
        if len(chunk) == FRAMESKIP:
            frames.append(cl.render(env))
            proprios.append(obs[:4].copy())
            acts.append(np.concatenate(chunk))
            chunk = []
    env.close()
    f = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    p = torch.from_numpy(np.stack(proprios)).float()
    a = torch.from_numpy(np.stack(acts)).float()
    return f, p, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="mw-reach")
    ap.add_argument("--n-trajs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--no-cam-tweak", dest="cam_tweak", action="store_false",
                    help="render with the DEFAULT corner2 (pos 1.3,-0.2,1.1) "
                         "instead of the upstream wrapper's 0.75,0.075,0.7")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()

    root = cfg["dataset"]["root"]

    # ---- dataset side -------------------------------------------------------
    print(f"loading {args.n_trajs} dataset trajs for {args.task} ...", flush=True)
    ds_errs, z_chunks, ds_frame0 = [], [], None
    for i, tb in enumerate(iterate_metaworld_trajectories(
            root, [args.task], max_trajectories_per_task=args.n_trajs)):
        if ds_frame0 is None:
            ds_frame0 = tb.obs_visual[0].numpy()
        fm, pm, am = to_model_steps(tb.obs_visual, tb.proprio, tb.action)
        print(f"  traj {tb.traj_id}: raw frames={tuple(tb.obs_visual.shape)} -> "
              f"model-steps frames={tuple(fm.shape)} actions={tuple(am.shape)}",
              flush=True)
        errs, z = one_step_mse(adapter, fm, pm, am, device)
        ds_errs.append(errs)
        z_chunks.append((flat(z).cpu(), i))
    ds_errs = np.concatenate(ds_errs)
    print(f"DATASET one-step MSE: median={np.median(ds_errs):.5f} "
          f"p90={np.percentile(ds_errs, 90):.5f} n={len(ds_errs)}", flush=True)

    # ---- rendered side ------------------------------------------------------
    print("rolling scripted expert in the live env ...", flush=True)
    print(f"cam_tweak={args.cam_tweak}", flush=True)
    f, p, a = rollout_expert_modelsteps(args.task, args.seed, cam_tweak=args.cam_tweak)
    r_errs, z_r = one_step_mse(adapter, f, p, a, device)
    print(f"RENDER  one-step MSE: median={np.median(r_errs):.5f} "
          f"p90={np.percentile(r_errs, 90):.5f} n={len(r_errs)} "
          f"ratio_vs_dataset={np.median(r_errs) / max(np.median(ds_errs), 1e-12):.1f}x",
          flush=True)

    # ---- latent NN distances ------------------------------------------------
    Zd = torch.cat([c for c, _ in z_chunks])           # (Nd, D)
    owner = np.concatenate([[i] * len(c) for c, i in z_chunks])
    Zr = flat(z_r).cpu()
    d_cross = torch.cdist(Zr, Zd).min(dim=1).values    # render -> dataset
    d_within = []
    D = torch.cdist(Zd, Zd)
    for i in range(len(Zd)):
        mask = torch.from_numpy(owner != owner[i])     # other trajs only
        d_within.append(float(D[i][mask].min()))
    d_within = np.array(d_within)
    print(f"latent NN dist: render->dataset median={d_cross.median():.3f} | "
          f"within-dataset (cross-traj) median={np.median(d_within):.3f} | "
          f"ratio={float(d_cross.median()) / max(np.median(d_within), 1e-9):.2f}x",
          flush=True)

    # ---- pixel side-by-side -------------------------------------------------
    import imageio.v2 as imageio
    out = ROOT / "results" / "logs"
    rgb_ds = np.transpose(ds_frame0, (1, 2, 0)).astype(np.uint8)
    rgb_rd = f[0].permute(1, 2, 0).numpy().astype(np.uint8)
    imageio.imwrite(out / "probe_dataset_frame.png", rgb_ds)
    imageio.imwrite(out / "probe_render_frame.png", rgb_rd)
    side = np.concatenate([rgb_ds, rgb_rd], axis=1)
    imageio.imwrite(out / "probe_side_by_side.png", side)
    print(f"channel means dataset={rgb_ds.reshape(-1,3).mean(0).round(1)} "
          f"render={rgb_rd.reshape(-1,3).mean(0).round(1)}", flush=True)
    print("wrote results/logs/probe_side_by_side.png", flush=True)

    # ---- verdict -------------------------------------------------------------
    ratio = np.median(r_errs) / max(np.median(ds_errs), 1e-12)
    nn_ratio = float(d_cross.median()) / max(np.median(d_within), 1e-9)
    if ratio > 3 or nn_ratio > 2:
        print(f"VERDICT: DOMAIN SHIFT (pred-err ratio {ratio:.1f}x, NN ratio "
              f"{nn_ratio:.1f}x) — renderer != training data", flush=True)
    else:
        print(f"VERDICT: visuals OK (pred-err ratio {ratio:.1f}x, NN ratio "
              f"{nn_ratio:.1f}x) — suspect protocol/goal handling", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
