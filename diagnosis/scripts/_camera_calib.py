"""Calibrate the eval camera against the training dataset, pixel-level.

The arm's home pose and the table are identical across episodes, so the init
frame is directly comparable between the dataset and a fresh env render —
modulo the (small) randomized object/goal markers. Try every historically
plausible corner2 config, score pixel MSE against a few dataset init frames,
and save the renders for eyeballing.

Candidates (from Farama-Foundation/Metaworld git history + upstream wrapper):
  A_default : XML as installed   pos 1.3,-0.2,1.1  euler 3.9,2.3,0.6  fovy 60
  B_tweak   : upstream wrapper   pos 0.75,0.075,0.7 (euler/fovy from XML)
  C_pre348  : pre-July-2021 XML  pos 1.1,-0.4,0.6  euler 3.9,2.3,3.7  fovy 45
  + fovy variants of B and C.

No GPU needed. python scripts/_camera_calib.py --config configs/diagnostic_metaworld.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import decord  # noqa: F401  — its bundled ffmpeg DLLs let torchcodec load (HF video decode)
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders import iterate_metaworld_trajectories  # noqa: E402

CANDIDATES = {
    "A_default":  dict(pos=None,               euler=None,            fovy=None),
    "B_tweak":    dict(pos=[0.75, 0.075, 0.7], euler=None,            fovy=None),
    "B_fovy45":   dict(pos=[0.75, 0.075, 0.7], euler=None,            fovy=45),
    "C_pre348":   dict(pos=[1.1, -0.4, 0.6],   euler=[3.9, 2.3, 3.7], fovy=45),
    "C_fovy60":   dict(pos=[1.1, -0.4, 0.6],   euler=[3.9, 2.3, 3.7], fovy=60),
    # TDMPC2-era combo: old-XML orientation (pre-#348 euler) + the wrapper's
    # pos override — the dataset may predate the corner2 orientation change.
    "D_old45":    dict(pos=[0.75, 0.075, 0.7], euler=[3.9, 2.3, 3.7], fovy=45),
    "D_old60":    dict(pos=[0.75, 0.075, 0.7], euler=[3.9, 2.3, 3.7], fovy=60),
}


def make_env_with_cam(task: str, seed: int, cam: dict, img_size: int = 224):
    import mujoco
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

    env_id = task.split("-", 1)[-1] + "-v3-goal-observable"
    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=seed)
    env.seeded_rand_vec = False
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "corner2")
    # sanity: euler->quat convention must reproduce the XML-compiled quat
    chk = np.zeros(4)
    mujoco.mju_euler2Quat(chk, np.array([3.9, 2.3, 0.6]), "xyz")
    drift = float(min(np.abs(chk - env.model.cam_quat[cam_id]).max(),
                      np.abs(chk + env.model.cam_quat[cam_id]).max()))
    if drift > 1e-3:
        print(f"  [warn] euler2Quat 'xyz' drift vs XML quat: {drift:.4f}")
    if cam["pos"] is not None:
        env.model.cam_pos[cam_id] = cam["pos"]
    if cam["euler"] is not None:
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.asarray(cam["euler"], dtype=np.float64), "xyz")
        env.model.cam_quat[cam_id] = quat
    if cam["fovy"] is not None:
        env.model.cam_fovy[cam_id] = cam["fovy"]
    env.render_mode = "rgb_array"
    env.camera_name = "corner2"
    env.width = env.height = img_size
    env.mujoco_renderer = MujocoRenderer(
        env.model, env.data, env.mujoco_renderer.default_cam_config,
        width=img_size, height=img_size,
        max_geom=env.mujoco_renderer.max_geom,
        camera_id=None, camera_name="corner2")
    env.reset()
    frame = env.render()
    env.close()
    return frame.copy()


def refine(task: str, refs: np.ndarray, out: Path, img_size: int = 224):
    """Coordinate descent over (pos, euler, fovy) starting from B_tweak,
    minimizing pixel MSE against the dataset init frames. One env, re-render."""
    import mujoco
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

    env_id = task.split("-", 1)[-1] + "-v3-goal-observable"
    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=10000)
    env.seeded_rand_vec = False
    env.render_mode = "rgb_array"
    env.camera_name = "corner2"
    env.width = env.height = img_size
    env.mujoco_renderer = MujocoRenderer(
        env.model, env.data, env.mujoco_renderer.default_cam_config,
        width=img_size, height=img_size,
        max_geom=env.mujoco_renderer.max_geom,
        camera_id=None, camera_name="corner2")
    env.reset()
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "corner2")

    q0 = env.model.cam_quat[cam_id].copy()      # XML-compiled orientation

    def set_cam(p):
        env.model.cam_pos[cam_id] = p[:3]
        dq = np.zeros(4)
        mujoco.mju_euler2Quat(dq, np.asarray(p[3:6], dtype=np.float64), "xyz")
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q0, dq)            # delta rotation on top of XML quat
        env.model.cam_quat[cam_id] = q
        env.model.cam_fovy[cam_id] = p[6]
        mujoco.mj_forward(env.model, env.data)   # recompute data.cam_xpos/xmat

    def score(p):
        set_cam(p)
        frame = env.render().astype(np.float32)
        return float(((refs - frame[None]) ** 2).mean())

    # start: B_tweak = old pos + XML orientation (delta 0) + fovy 60
    p = np.array([0.75, 0.075, 0.7, 0.0, 0.0, 0.0, 60.0])
    best = score(p)
    print(f"refine start MSE={best:.1f}")
    deltas = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 8.0])
    for sweep in range(6):
        improved = False
        for i in range(len(p)):
            for sgn in (+1, -1):
                q = p.copy()
                q[i] += sgn * deltas[i]
                s = score(q)
                if s < best - 1e-6:
                    p, best, improved = q, s, True
        print(f"  sweep {sweep}: MSE={best:.1f} pos={p[:3].round(3)} "
              f"euler={p[3:6].round(3)} fovy={p[6]:.1f}", flush=True)
        if not improved:
            deltas *= 0.5
    set_cam(p)
    frame = env.render().copy()
    env.close()
    imageio.imwrite(out / "calib_refined.png", frame)
    side = np.concatenate([refs[0].astype(np.uint8), frame], axis=1)
    imageio.imwrite(out / "calib_refined_side_by_side.png", side)
    print(f"REFINED: MSE={best:.1f} pos={list(p[:3].round(4))} "
          f"euler={list(p[3:6].round(4))} fovy={p[6]:.2f}")
    return p, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="mw-reach")
    ap.add_argument("--n-trajs", type=int, default=4)
    ap.add_argument("--refine", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    refs = []
    for tb in iterate_metaworld_trajectories(
            cfg["dataset"]["root"], [args.task], max_trajectories_per_task=args.n_trajs):
        refs.append(tb.obs_visual[0].permute(1, 2, 0).numpy())   # (H,W,3) [0,255]
    refs = np.stack(refs).astype(np.float32)
    out = ROOT / "results" / "logs"
    imageio.imwrite(out / "calib_ref.png", refs[0].astype(np.uint8))

    print(f"{args.task}: {len(refs)} dataset init frames as reference")
    if args.refine:
        refine(args.task, refs, out)
        return 0
    scores = {}
    for name, cam in CANDIDATES.items():
        frame = make_env_with_cam(args.task, seed=10000, cam=cam).astype(np.float32)
        for tag, f in (("", frame), ("+fliplr", frame[:, ::-1]),
                       ("+flipud", frame[::-1])):
            mse = float(((refs - f[None]) ** 2).mean())
            scores[name + tag] = mse
            imageio.imwrite(out / f"calib_{name}{tag}.png",
                            np.ascontiguousarray(f).astype(np.uint8))
            print(f"  {name + tag:18s} pixel MSE vs dataset init frames: {mse:10.1f}")

    best = min(scores, key=scores.get)
    side = np.concatenate([refs[0],
                           imageio.imread(out / f"calib_{best}.png").astype(np.float32)],
                          axis=1).astype(np.uint8)
    imageio.imwrite(out / "calib_best_side_by_side.png", side)
    print(f"BEST: {best} (MSE {scores[best]:.1f}) — "
          f"see results/logs/calib_best_side_by_side.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
