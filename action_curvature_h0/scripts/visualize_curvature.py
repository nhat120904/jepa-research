#!/usr/bin/env python3
"""Render one snapshot's action triplet: video frames plus the latent bend.

For a chosen state we take the centre action chunk, nudge it symmetrically
(``a - d``, ``a``, ``a + d``), and execute all three in the simulator from the
identical restored start state, rendering EVERY step rather than only the
endpoint.  That gives the video.

We also run the frozen world model on the same three chunks and keep its three
predicted terminal latents.  Projected onto the plane spanned by
``v- = z(a) - z(a-d)`` and ``v+ = z(a+d) - z(a)``, those three points *are* what
"action-space curvature" measures: if the model's map were locally straight the
three would be collinear, and the angle between ``v-`` and ``v+`` is the bend.
The realized object positions are kept alongside so the model's bend can be put
next to the physical one.

Outputs an npz per snapshot; composition into figures and side-by-side video is
done separately, off the GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import inspect

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "diagnosis"
sys.path.insert(0, str(REPO))

from action_curvature_h0.core import (  # noqa: E402
    draw_unit_directions,
    make_feasible,
    scale_direction,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-index", type=int, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--population-dir", type=Path, required=True)
    p.add_argument("--population-index", type=int, default=1)
    p.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    p.add_argument("--checkpoint", default="quentinll/lewm-cube")
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--sigma", type=float, default=0.2,
                   help="probe size; the visual is clearest at the largest "
                        "preregistered scale")
    p.add_argument("--direction-seed", type=int, default=20260827)
    p.add_argument("--encode-batch", type=int, default=8)
    p.add_argument("--action-source", default="cem_fixed")
    p.add_argument("--cem-local-alpha", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def rollout_with_frames(raw_env: Any, init_row: Any, goal_row: Any, corrected: Any,
                        audit: Any, raw_actions: np.ndarray,
                        mc: Any) -> dict[str, Any]:
    """Execute a chunk from the restored start state, rendering every step."""
    corrected.restore_complete(
        raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit)
    frames = [audit.resize_render(raw_env.render())]
    obj = [mc.object_position(raw_env)]
    eff = [mc.effector_position(raw_env)]
    for action in raw_actions.reshape(-1, raw_actions.shape[-1]):
        _, _, terminated, truncated, _ = raw_env.step(action)
        frames.append(audit.resize_render(raw_env.render()))
        obj.append(mc.object_position(raw_env))
        eff.append(mc.effector_position(raw_env))
        if terminated or truncated:
            break
    return {"frames": np.stack(frames), "object": np.stack(obj),
            "effector": np.stack(eff)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("rendering the world model requires a GPU allocation")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "vz_audit")
    corrected = load_module(
        DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "vz_corrected")
    mc = load_module(Path(__file__).resolve().parent / "measure_curvature.py", "vz_mc")

    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset)
    init_row, goal_row = init_rows[0], goal_rows[0]
    action_data = np.asarray(dataset.get_col_data("action"))
    scaler = StandardScaler().fit(action_data[~np.isnan(action_data).any(axis=1)])

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # float64, matching how these states were MEASURED (slurm_heldout.sh passes
    # --model-dtype float64).  At a near-stationary state the latent
    # displacements are far below the float32 floor of this model's own forward
    # pass, so in float32 v- and v+ become numerical noise: the first version of
    # this figure reported a 116 deg bend and k=1.52 for a state whose measured
    # k is 0.0002.  Same upstream downcast patch as measure_curvature.py.
    model = model.double()
    from stable_worldmodel.wm.lewm.module import Embedder

    if "x = x.float()" not in inspect.getsource(Embedder.forward):
        raise RuntimeError("Embedder.forward changed; re-derive the precision patch")

    def _forward_dtype_preserving(self, x):
        x = x.to(next(self.parameters()).dtype)
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x

    Embedder.forward = _forward_dtype_preserving

    transform = audit.make_transform(224)
    world, raw_env, _, _ = corrected.make_world(swm, snapshot)
    try:
        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        low = np.asarray(world.envs.single_action_space.low,
                         dtype=np.float64).reshape(-1)
        high = np.asarray(world.envs.single_action_space.high,
                          dtype=np.float64).reshape(-1)

        base_evaluator = swm.planning.ShootingCostEvaluator(
            model, swm.planning.GoalMSE())
        config = swm.PlanConfig(
            horizon=args.horizon, receding_horizon=args.horizon,
            action_block=args.action_block, history_len=1, warm_start=True)
        solver = swm.planning.CEMSolver(
            cost=base_evaluator, batch_size=1, num_samples=1, n_steps=1, topk=1,
            device="cuda", seed=0)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process={"action": scaler},
            transform={"pixels": transform, "goal": transform})
        policy.set_env(world.envs)
        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
        }

        rng = np.random.default_rng(args.direction_seed + snapshot.order)
        centre, proposal_std = mc.resolve_centre(
            args, snapshot, action_data, scaler, raw_dim, rng)
        centre_raw = mc.normalized_to_raw_chunk(centre, scaler, args, raw_dim)
        centre_raw = np.clip(centre_raw, low, high).reshape(centre.shape[0], -1)
        centre = scaler.transform(
            centre_raw.reshape(-1, raw_dim)).reshape(centre.shape)

        # Mirrors measure_curvature.py exactly: per-component headroom to the
        # bounds, a cap on |base[i]| at this sigma, then a direction made
        # feasible by construction rather than drawn and rejected.
        raw_range_chunk = np.broadcast_to(
            np.tile(high - low, args.action_block), centre.shape).astype(np.float64)
        scaler_scale_chunk = np.broadcast_to(
            np.tile(scaler.scale_, args.action_block), centre.shape).astype(np.float64)
        centre_raw_chunk = mc.normalized_to_raw_chunk(
            centre, scaler, args, raw_dim).reshape(centre.shape)
        low_chunk = np.broadcast_to(np.tile(low, args.action_block), centre.shape)
        high_chunk = np.broadcast_to(np.tile(high, args.action_block), centre.shape)
        margin = np.minimum(high_chunk - centre_raw_chunk,
                            centre_raw_chunk - low_chunk).clip(min=0.0)
        if args.action_source == "cem_local":
            denom = args.cem_local_alpha * proposal_std * scaler_scale_chunk
        else:
            denom = args.sigma * raw_range_chunk
        cap = margin / np.maximum(denom, 1e-12)

        raw_base = draw_unit_directions(rng, 1, centre.shape)[0]
        adjusted, feas = make_feasible(raw_base.reshape(-1), cap.reshape(-1))
        if feas["feasible"] == 0.0:
            raise RuntimeError("centre is fully saturated; pick another snapshot")
        base = adjusted.reshape(centre.shape)
        delta = scale_direction(
            base, args.sigma, source=args.action_source,
            raw_range_chunk=raw_range_chunk, scaler_scale_chunk=scaler_scale_chunk,
            proposal_std=proposal_std, alpha=args.cem_local_alpha,
            sigma_max=args.sigma)

        chunks = np.stack([centre - delta, centre, centre + delta])
        rollouts = [rollout_with_frames(
            raw_env, init_row, goal_row, corrected, audit,
            np.clip(mc.normalized_to_raw_chunk(c, scaler, args, raw_dim), low, high),
            mc) for c in chunks]

        rolled = mc.model_terminal_latents(
            policy, raw_info, chunks, base_evaluator, torch.float64)
        latents = np.asarray(rolled["terminal"], dtype=np.float64).reshape(3, -1)
    finally:
        world.close()

    v_minus = latents[1] - latents[0]
    v_plus = latents[2] - latents[1]
    nm, np_ = np.linalg.norm(v_minus), np.linalg.norm(v_plus)
    cos = float(np.dot(v_minus, v_plus) / (nm * np_)) if nm > 0 and np_ > 0 else float("nan")

    # 2-D coordinates of the three latents in the plane spanned by v-, v+.
    e1 = v_minus / (nm + 1e-12)
    perp = v_plus - np.dot(v_plus, e1) * e1
    e2 = perp / (np.linalg.norm(perp) + 1e-12)
    coords = np.stack([[float(np.dot(z - latents[1], e1)),
                        float(np.dot(z - latents[1], e2))] for z in latents])

    k_viz = float(np.linalg.norm(v_plus - v_minus)
                  / (np.linalg.norm(latents[2] - latents[0]) + 1e-12))

    out = args.out_dir / f"snapshot_{snapshot.order:03d}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "viz.npz",
        frames_minus=rollouts[0]["frames"], frames_centre=rollouts[1]["frames"],
        frames_plus=rollouts[2]["frames"],
        object_minus=rollouts[0]["object"], object_centre=rollouts[1]["object"],
        object_plus=rollouts[2]["object"],
        latent_coords=coords, latent_cosine=np.asarray(cos),
        sigma=np.asarray(args.sigma), snapshot=np.asarray(snapshot.order))
    meta = {
        "snapshot": snapshot.order, "episode": snapshot.episode,
        "sigma": args.sigma, "latent_cosine": cos,
        "latent_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
        "n_frames": [int(r["frames"].shape[0]) for r in rollouts],
        "object_travel_mm": [float(1000 * np.linalg.norm(r["object"][-1] - r["object"][0]))
                             for r in rollouts],
        "effector_travel_mm": [float(1000 * np.linalg.norm(r["effector"][-1] - r["effector"][0]))
                               for r in rollouts],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
