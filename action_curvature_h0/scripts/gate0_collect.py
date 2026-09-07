#!/usr/bin/env python3
"""Gate 0: can ANY function of the frozen latent certify progress off-policy?

A control-Lyapunov cost must decrease whenever the true distance to the goal
decreases.  Before training such a cost it is worth asking whether the frozen
latent even admits one.  This collects the raw material for that question.

For each snapshot we take action sequences from the CACHED iteration-29 CEM
population -- the planner-induced distribution, i.e. off-policy, which is where
the deployed cost was measured to anti-correlate with progress (rho_final
-0.085) -- execute them in the simulator, and record per step:

  * the frozen encoder's latent of the rendered frame
  * the goal latent
  * the TRUE object distance to the goal
  * proprioception (effector position, gripper), which the world model never sees

Candidates are drawn across the whole cost ranking, not only the elites, so the
answer is about the latent rather than about one narrow region.

Analysis is separate and needs no GPU.
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

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "diagnosis"
sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-index", type=int, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--populations-dir", type=Path, required=True)
    p.add_argument("--population-index", type=int, default=1,
                   help="1 = the converged CEM population (planner-induced)")
    p.add_argument("--n-candidates", type=int, default=12)
    p.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    p.add_argument("--checkpoint", default="quentinll/lewm-cube")
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--encode-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260901)
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


@torch.inference_mode()
def encode_visual(model: Any, images: np.ndarray, transform: Any,
                  batch_size: int, audit: Any) -> np.ndarray:
    """Per-frame VISUAL embedding, for LeWM and PreJEPA alike.

    PreJEPA's encode concatenates a tiled action embedding onto the visual
    one, so passing a dummy action would contaminate the latent; emb_keys=[]
    skips the extra encoders entirely and leaves the visual embedding alone.
    LeWM's encode takes no such argument.

    PreJEPA returns patch tokens (B, T, P, d) while LeWM returns a pooled
    (B, T, d); patches are mean-pooled so both arms are one vector per frame
    and the readout comparison is like for like.  Pooling can only LOWER the
    measured upper bound, so a high score remains trustworthy and a low one
    carries this caveat.
    """
    out = []
    for i in range(0, len(images), batch_size):
        chunk = np.stack([transform(x) if callable(transform) else x
                          for x in images[i : i + batch_size]])
        px = torch.as_tensor(chunk, dtype=torch.float32).cuda()
        if px.ndim == 4:
            px = px[:, None]                       # (B, T=1, C, H, W)
        info = {"pixels": px}
        try:
            emb = model.encode(info, emb_keys=[])["emb"]
        except TypeError:
            emb = model.encode({"pixels": px})["emb"]
        emb = emb[:, -1]
        if emb.ndim == 3:                          # (B, P, d) -> (B, d)
            emb = emb.mean(dim=1)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out).reshape(len(images), -1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("encoding requires a GPU allocation")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "g0_audit")
    corrected = load_module(
        DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "g0_corrected")
    mc = load_module(Path(__file__).resolve().parent / "measure_curvature.py", "g0_mc")

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
    transform = audit.make_transform(224)

    pop = np.load(args.populations_dir / f"snapshot_{snapshot.order:03d}/populations.npz")
    actions = np.asarray(pop["actions_normalized"][args.population_index], dtype=np.float64)
    costs = np.asarray(pop["native_cost"][args.population_index], dtype=np.float64)
    # Spread across the whole ranking, not just the elites: the question is about
    # the latent, not about one narrow slice of action space.
    order = np.argsort(costs, kind="mergesort")
    picks = np.unique(np.linspace(0, len(order) - 1, args.n_candidates).astype(int))
    chosen = order[picks]

    world, raw_env, _, _ = corrected.make_world(swm, snapshot)
    records = []
    try:
        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        low = np.asarray(world.envs.single_action_space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(world.envs.single_action_space.high, dtype=np.float64).reshape(-1)
        target = np.asarray(audit.goal_field(goal_row, "block_0_pos"), dtype=np.float64)
        goal_frame = audit.resize_render(np.asarray(goal_row["goal"]))

        class _A:  # shape adapter for normalized_to_raw_chunk
            horizon = actions.shape[1]
            action_block = args.action_block
        for c in chosen:
            raw = mc.normalized_to_raw_chunk(actions[c], scaler, _A, raw_dim)
            raw = np.clip(raw, low, high)
            corrected.restore_complete(
                raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit)
            frames, dists, props = [], [], []
            frames.append(audit.resize_render(raw_env.render()))
            dists.append(float(audit.cube_distance(raw_env, target)))
            props.append(mc.effector_position(raw_env))
            for a in raw.reshape(-1, raw.shape[-1]):
                _, _, term, trunc, _ = raw_env.step(a)
                frames.append(audit.resize_render(raw_env.render()))
                dists.append(float(audit.cube_distance(raw_env, target)))
                props.append(mc.effector_position(raw_env))
                if term or trunc:
                    break
            z = encode_visual(model, np.stack(frames + [goal_frame]), transform,
                              args.encode_batch, audit)
            records.append({
                "candidate": int(c), "cost_rank": int(np.where(order == c)[0][0]),
                "z": z[:-1], "z_goal": z[-1],
                "true_distance_m": np.asarray(dists, dtype=np.float64),
                "proprio": np.asarray(props, dtype=np.float64),
            })
    finally:
        world.close()

    out = args.out_dir / f"snapshot_{snapshot.order:03d}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "gate0.npz",
        z=np.concatenate([r["z"] for r in records]),
        z_goal=np.stack([r["z_goal"] for r in records]),
        true_distance_m=np.concatenate([r["true_distance_m"] for r in records]),
        proprio=np.concatenate([r["proprio"] for r in records]),
        traj_id=np.concatenate([np.full(len(r["z"]), i) for i, r in enumerate(records)]),
        cost_rank=np.array([r["cost_rank"] for r in records]),
        snapshot=np.asarray(snapshot.order))
    meta = {"snapshot": snapshot.order, "n_candidates": len(records),
            "n_steps_total": int(sum(len(r["z"]) for r in records)),
            "latent_dim": int(records[0]["z"].shape[1]),
            "cost_ranks": [r["cost_rank"] for r in records],
            "population_index": args.population_index}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
