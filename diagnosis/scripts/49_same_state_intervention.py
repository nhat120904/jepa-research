"""Exact same-state interventional MetaWorld benchmark.

For every pre-grasp/contact anchor, this runner saves the complete MuJoCo
integration state, restores it before each member of a *fixed local action
fan*, and measures the true successors at model horizons 1/2/4/8.  It then
compares those outcomes to action-conditioned predictions from frozen JEPA-WM
checkpoints.  Unlike ``hard_nn``, no action is borrowed from a merely similar
state: all K+1 outcomes start from bit-identical simulator state.

This is a GPU/renderer workload.  Run only through
``scripts/slurm_same_state_intervention.sh``; the script refuses a login-node
launch unless ``CAI_JEPA_ALLOW_LOGIN_HEAVY=1`` is explicitly set.

Outputs for ``--out-prefix results/metaworld_same_state``:
  *_candidates.csv  one row per model/anchor/horizon/action candidate
  *_summary.csv     causal CRA and within-fan action-effect fidelity
  *_actions.csv     exact raw action sequences (model-independent manifest)
  *_metadata.json   protocol and contact-impulse caveats
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util as ilu
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from scripts._same_state_intervention import make_local_action_fan, summarize_fan  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def _load(modname: str, filename: str):
    spec = ilu.spec_from_file_location(modname, str(ROOT / "scripts" / filename))
    module = ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_cl = _load("closed_loop_eval_same_state", "18_closed_loop_eval.py")
make_env, expert_policy, render = _cl.make_env, _cl.expert_policy, _cl.render
FRAMESKIP, RAW_A = _cl.FRAMESKIP, _cl.RAW_A
HAND_SLICE = slice(0, 3)


def require_slurm_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("CAI_JEPA_ALLOW_LOGIN_HEAVY") != "1":
        raise RuntimeError(
            "Refusing to run the same-state GPU benchmark outside Slurm. Submit "
            "scripts/slurm_same_state_intervention.sh."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("same-state intervention requires a CUDA compute node")


def physics_snapshot(env) -> dict:
    """Capture MuJoCo's full integration state plus MetaWorld bookkeeping."""
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, spec)
    return {
        "integration": state,
        "curr_path_length": int(getattr(env, "curr_path_length", 0)),
    }


def physics_restore(env, snap: dict) -> None:
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(env.model, env.data, snap["integration"], spec)
    if hasattr(env, "curr_path_length"):
        env.curr_path_length = int(snap["curr_path_length"])
    mujoco.mj_forward(env.model, env.data)


_ROBOT_TOKENS = (
    "sawyer", "hand", "finger", "gripper", "wrist", "palm", "right_", "left_",
    "rightclaw", "leftclaw", "rightpad", "leftpad", "rail",
)
_STATIC_TOKENS = ("table", "floor", "wall", "base", "pedestal", "world")


def _geom_name(model, geom_id: int) -> str:
    import mujoco

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name or f"geom_{geom_id}"


def _body_name(model, geom_id: int) -> str:
    import mujoco

    body_id = int(model.geom_bodyid[int(geom_id)])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name or f"body_{body_id}"


def _is_robot(name: str) -> bool:
    lower = name.lower()
    return any(tok in lower for tok in _ROBOT_TOKENS)


def _is_static(name: str) -> bool:
    lower = name.lower()
    return any(tok in lower for tok in _STATIC_TOKENS)


def contact_observation(env) -> dict:
    """Read contacts at the current state, including sampled normal forces.

    MetaWorld exposes only the contact force at the state returned by each
    ``env.step``.  ``sampled_impulse_ns`` therefore sums force*env.dt at those
    endpoints; it is a reproducible approximation, not substep-integrated GT.
    """
    import mujoco

    normal_force = robot_object_force = 0.0
    robot_object_count = 0
    pairs: list[str] = []
    wrench = np.zeros(6, dtype=np.float64)
    for i in range(int(env.data.ncon)):
        c = env.data.contact[i]
        g1, g2 = _geom_name(env.model, c.geom1), _geom_name(env.model, c.geom2)
        b1, b2 = _body_name(env.model, c.geom1), _body_name(env.model, c.geom2)
        n1, n2 = f"{b1}/{g1}", f"{b2}/{g2}"
        pairs.append(f"{n1}|{n2}")
        wrench.fill(0.0)
        mujoco.mj_contactForce(env.model, env.data, i, wrench)
        force = abs(float(wrench[0]))
        normal_force += force
        r1, r2 = _is_robot(n1), _is_robot(n2)
        robot_object = (r1 != r2) and not _is_static(n1 if r2 else n2)
        if robot_object:
            robot_object_count += 1
            robot_object_force += force
    dt = float(getattr(env, "dt", env.model.opt.timestep))
    return {
        "contact_count": int(env.data.ncon),
        "robot_object_contact_count": robot_object_count,
        "normal_force_n": normal_force,
        "robot_object_force_n": robot_object_force,
        "sampled_impulse_ns": normal_force * dt,
        "robot_object_sampled_impulse_ns": robot_object_force * dt,
        "pairs": sorted(set(pairs)),
    }


def _regime(obs: np.ndarray, contact: dict, pregrasp_radius: float) -> str | None:
    if contact["robot_object_contact_count"] > 0:
        return "contact"
    if float(np.linalg.norm(obs[HAND_SLICE] - obs[OBJECT_SLICE])) <= pregrasp_radius:
        return "pre_grasp"
    return None


def collect_anchors(env, task: str, seed: int, *, max_steps: int,
                    anchors_per_episode: int, collect_every: int,
                    pregrasp_radius: float) -> list[dict]:
    """Collect deterministic expert-trajectory snapshots, balanced by regime."""
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    policy = expert_policy(task)
    candidates: dict[str, list[dict]] = {"pre_grasp": [], "contact": []}
    for step in range(max_steps):
        c = contact_observation(env)
        regime = _regime(obs, c, pregrasp_radius)
        if regime is not None and step % collect_every == 0:
            candidates[regime].append({
                "anchor_id": f"{task}__s{seed}__t{step:03d}",
                "task": task,
                "seed": seed,
                "anchor_step": step,
                "regime": regime,
                "snapshot": physics_snapshot(env),
                "obs": obs.copy(),
                "frame": render(env),
                "contact": c,
            })
        obs, _, terminated, truncated, _ = env.step(policy.get_action(obs))
        if terminated or truncated:
            break

    # Deterministic, approximately balanced selection spread over each phase.
    quota = max(1, anchors_per_episode // 2)
    selected: list[dict] = []
    for name in ("pre_grasp", "contact"):
        vals = candidates[name]
        if vals:
            idx = np.linspace(0, len(vals) - 1, min(quota, len(vals))).round().astype(int)
            selected.extend(vals[int(i)] for i in np.unique(idx))
    if len(selected) < anchors_per_episode:
        used = {x["anchor_id"] for x in selected}
        spare = [x for name in ("pre_grasp", "contact") for x in candidates[name]
                 if x["anchor_id"] not in used]
        selected.extend(spare[: anchors_per_episode - len(selected)])
    return sorted(selected[:anchors_per_episode], key=lambda x: x["anchor_step"])


def nominal_sequence(env, anchor: dict, task: str, raw_steps: int) -> np.ndarray:
    physics_restore(env, anchor["snapshot"])
    obs = env.unwrapped._get_obs()
    policy = expert_policy(task)
    actions = []
    for _ in range(raw_steps):
        action = np.asarray(policy.get_action(obs), dtype=np.float64)
        actions.append(action)
        obs, _, _, _, _ = env.step(action)
    physics_restore(env, anchor["snapshot"])
    return np.asarray(actions)


def execute_fan(env, anchor: dict, fan: np.ndarray, horizons: list[int]) -> dict[int, list[dict]]:
    endpoints = {h: [] for h in horizons}
    endpoint_raw = {h * FRAMESKIP: h for h in horizons}
    for candidate in fan:
        physics_restore(env, anchor["snapshot"])
        cumulative = {
            "sampled_impulse_ns": 0.0,
            "robot_object_sampled_impulse_ns": 0.0,
            "max_contact_count": 0,
            "max_robot_object_contact_count": 0,
            "pairs": set(),
        }
        obs = anchor["obs"]
        for raw_i, action in enumerate(candidate, start=1):
            obs, _, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
            c = contact_observation(env)
            cumulative["sampled_impulse_ns"] += c["sampled_impulse_ns"]
            cumulative["robot_object_sampled_impulse_ns"] += c["robot_object_sampled_impulse_ns"]
            cumulative["max_contact_count"] = max(cumulative["max_contact_count"], c["contact_count"])
            cumulative["max_robot_object_contact_count"] = max(
                cumulative["max_robot_object_contact_count"], c["robot_object_contact_count"])
            cumulative["pairs"].update(c["pairs"])
            if raw_i in endpoint_raw:
                h = endpoint_raw[raw_i]
                endpoints[h].append({
                    "obs": obs.copy(),
                    "frame": render(env),
                    "contact": {**cumulative, "pairs": sorted(cumulative["pairs"])},
                })
    physics_restore(env, anchor["snapshot"])
    return endpoints


@torch.no_grad()
def encode_frames(adapter, frames: list[np.ndarray], proprios: list[np.ndarray],
                  device: torch.device, chunk: int = 16) -> torch.Tensor:
    outs = []
    for start in range(0, len(frames), chunk):
        fs = frames[start:start + chunk]
        ps = proprios[start:start + chunk]
        visual = torch.stack([torch.from_numpy(x.copy()).permute(2, 0, 1).float() for x in fs])[:, None]
        proprio = torch.stack([torch.from_numpy(np.asarray(x, np.float32)) for x in ps])[:, None]
        z = adapter.encode(visual.to(device), proprio.to(device) if adapter.uses_proprio() else None)
        outs.append(z[:, 0].cpu())
    return torch.cat(outs)


def _hash_actions(seq: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(seq, np.float32).tobytes()).hexdigest()[:16]


def evaluate_anchor(env, adapter, device, anchor, *, task, horizons, n_candidates,
                    xyz_delta, noise_std, fan_seed, encode_chunk):
    max_h = max(horizons)
    nominal = nominal_sequence(env, anchor, task, max_h * FRAMESKIP)
    fan, labels = make_local_action_fan(
        nominal, n_candidates=n_candidates, xyz_delta=xyz_delta,
        noise_std=noise_std, seed=fan_seed + anchor["seed"] * 1000 + anchor["anchor_step"])
    endpoints = execute_fan(env, anchor, fan, horizons)

    z0 = encode_frames(adapter, [anchor["frame"]], [anchor["obs"][:4]], device, encode_chunk)[0]
    actions_model = torch.from_numpy(
        fan.reshape(n_candidates, max_h, FRAMESKIP * RAW_A).astype(np.float32)).to(device)
    prop0 = torch.from_numpy(anchor["obs"][:4].astype(np.float32))[None].repeat(n_candidates, 1).to(device)
    pred_rollout = adapter.predict_rollout(
        z0.to(device)[None].repeat(n_candidates, *([1] * z0.ndim)), actions_model,
        proprio_t=prop0 if adapter.uses_proprio() else None).cpu()

    candidate_rows, summary_rows = [], []
    init_obj, init_hand, init_grip = (anchor["obs"][OBJECT_SLICE], anchor["obs"][HAND_SLICE],
                                      float(anchor["obs"][3]))
    for h in horizons:
        ep = endpoints[h]
        true_z = encode_frames(adapter, [x["frame"] for x in ep],
                               [x["obs"][:4] for x in ep], device, encode_chunk)
        pred_z = pred_rollout[:, h]
        objects = np.stack([x["obs"][OBJECT_SLICE] for x in ep])
        stats = summarize_fan(pred_z.numpy(), true_z.numpy(), objects)
        summary_rows.append({
            "model": adapter.spec.name, "anchor_id": anchor["anchor_id"],
            "task": task, "seed": anchor["seed"], "anchor_step": anchor["anchor_step"],
            "regime": anchor["regime"], "horizon_model_steps": h,
            "horizon_raw_steps": h * FRAMESKIP, **asdict(stats),
        })
        pred_flat = pred_z.reshape(n_candidates, -1)
        true_flat = true_z.reshape(n_candidates, -1)
        z0_flat = z0.reshape(-1)
        factual_true = true_flat[0]
        for k, x in enumerate(ep):
            obs, c = x["obs"], x["contact"]
            obj, hand, grip = obs[OBJECT_SLICE], obs[HAND_SLICE], float(obs[3])
            candidate_rows.append({
                "model": adapter.spec.name, "anchor_id": anchor["anchor_id"],
                "task": task, "seed": anchor["seed"], "anchor_step": anchor["anchor_step"],
                "regime": anchor["regime"], "horizon_model_steps": h,
                "horizon_raw_steps": h * FRAMESKIP, "candidate_id": k,
                "candidate_label": labels[k], "is_factual": int(k == 0),
                "action_sequence_sha256_16": _hash_actions(fan[k, :h * FRAMESKIP]),
                "action_l2_from_factual": float(np.linalg.norm(fan[k, :h * FRAMESKIP]
                                                                 - fan[0, :h * FRAMESKIP])),
                "init_hand_object_dist_m": float(np.linalg.norm(init_hand - init_obj)),
                "object_x": float(obj[0]), "object_y": float(obj[1]), "object_z": float(obj[2]),
                "object_effect_m": float(np.linalg.norm(obj - init_obj)),
                "hand_effect_m": float(np.linalg.norm(hand - init_hand)),
                "gripper_effect": float(grip - init_grip),
                "robot_object_contact": int(c["max_robot_object_contact_count"] > 0),
                "max_contact_count": c["max_contact_count"],
                "max_robot_object_contact_count": c["max_robot_object_contact_count"],
                "sampled_contact_impulse_ns": c["sampled_impulse_ns"],
                "robot_object_sampled_impulse_ns": c["robot_object_sampled_impulse_ns"],
                "contact_pairs": ";".join(c["pairs"]),
                "prediction_error_l2": float(torch.linalg.vector_norm(pred_flat[k] - true_flat[k])),
                "prediction_error_mse": float(torch.mean((pred_flat[k] - true_flat[k]) ** 2)),
                "predicted_effect_l2": float(torch.linalg.vector_norm(pred_flat[k] - z0_flat)),
                "true_latent_effect_l2": float(torch.linalg.vector_norm(true_flat[k] - z0_flat)),
                "predicted_to_factual_true_l2": float(torch.linalg.vector_norm(
                    pred_flat[k] - factual_true)),
            })

    action_rows = []
    for k, seq in enumerate(fan):
        for raw_step, action in enumerate(seq):
            action_rows.append({
                "anchor_id": anchor["anchor_id"], "task": task, "seed": anchor["seed"],
                "anchor_step": anchor["anchor_step"], "regime": anchor["regime"],
                "candidate_id": k, "candidate_label": labels[k], "raw_step": raw_step,
                "action_0": action[0], "action_1": action[1],
                "action_2": action[2], "action_3": action[3],
            })
    return candidate_rows, summary_rows, action_rows


def _writer(path: Path, fields: list[str]):
    handle = open(path, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", default=["dino_wm_metaworld", "jepa_wm_metaworld"])
    ap.add_argument("--tasks", nargs="+", default=["mw-push", "mw-pick-place"])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--max-expert-steps", type=int, default=100)
    ap.add_argument("--anchors-per-episode", type=int, default=6)
    ap.add_argument("--collect-every", type=int, default=2)
    ap.add_argument("--pregrasp-radius", type=float, default=0.12)
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--n-candidates", type=int, default=17)
    ap.add_argument("--xyz-delta", type=float, default=0.25)
    ap.add_argument("--noise-std", type=float, default=0.12)
    ap.add_argument("--fan-seed", type=int, default=9173)
    ap.add_argument("--encode-chunk", type=int, default=16)
    ap.add_argument("--out-prefix", default="results/metaworld_same_state")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    require_slurm_gpu()
    horizons = sorted(set(args.horizons))
    if min(horizons) < 1:
        raise SystemExit("all horizons must be >=1")
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"].get("device", "cuda"))
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: Path(f"{prefix}_{name}") for name in
             ("candidates.csv", "summary.csv", "actions.csv", "metadata.json")}
    if not args.overwrite and any(p.exists() for p in paths.values()):
        raise SystemExit(f"output exists under {prefix}; pass --overwrite to replace")

    cand_handle = summary_handle = action_handle = None
    n_anchor_evals = 0
    anchor_ids: set[str] = set()
    try:
        for model_i, model_name in enumerate(args.models):
            print(f"=== loading {model_name} ===", flush=True)
            adapter = build_adapter(model_name, device=str(device)).eval()
            for task in args.tasks:
                for ep in range(args.episodes):
                    seed = args.seed0 + ep
                    env, _ = make_env(task, seed, img_size=adapter.spec.image_size)
                    anchors = collect_anchors(
                        env, task, seed, max_steps=args.max_expert_steps,
                        anchors_per_episode=args.anchors_per_episode,
                        collect_every=args.collect_every,
                        pregrasp_radius=args.pregrasp_radius)
                    print(f"  {model_name} {task} seed={seed}: {len(anchors)} anchors "
                          f"({','.join(a['regime'] for a in anchors)})", flush=True)
                    for anchor in anchors:
                        candidates, summaries, actions = evaluate_anchor(
                            env, adapter, device, anchor, task=task, horizons=horizons,
                            n_candidates=args.n_candidates, xyz_delta=args.xyz_delta,
                            noise_std=args.noise_std, fan_seed=args.fan_seed,
                            encode_chunk=args.encode_chunk)
                        if cand_handle is None:
                            cand_handle, cand_writer = _writer(paths["candidates.csv"], list(candidates[0]))
                            summary_handle, summary_writer = _writer(paths["summary.csv"], list(summaries[0]))
                        cand_writer.writerows(candidates); cand_handle.flush()
                        summary_writer.writerows(summaries); summary_handle.flush()
                        # Action manifest is model-independent.  Write only on
                        # the first model pass; later passes must regenerate it
                        # identically from the same task/seed/anchor protocol.
                        if model_i == 0:
                            if action_handle is None:
                                action_handle, action_writer = _writer(paths["actions.csv"], list(actions[0]))
                            action_writer.writerows(actions); action_handle.flush()
                            anchor_ids.add(anchor["anchor_id"])
                        n_anchor_evals += 1
                    env.close()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        for handle in (cand_handle, summary_handle, action_handle):
            if handle is not None:
                handle.close()

    metadata = {
        "protocol": "exact_same_mujoco_integration_state_local_action_fan",
        "models": args.models, "tasks": args.tasks, "episodes": args.episodes,
        "seed0": args.seed0, "horizons_model_steps": horizons,
        "frameskip_raw_steps_per_model_step": FRAMESKIP,
        "n_candidates_including_factual": args.n_candidates,
        "xyz_delta": args.xyz_delta, "noise_std": args.noise_std,
        "fan_seed": args.fan_seed, "pregrasp_radius_m": args.pregrasp_radius,
        "n_unique_anchors": len(anchor_ids), "n_model_anchor_evaluations": n_anchor_evals,
        "contact_impulse_note": (
            "sampled force multiplied by env.dt at each returned env.step; MetaWorld does not "
            "expose substep-integrated contact impulse, so this is an approximation"),
        "causal_scope_note": (
            "all candidates within an anchor start from the identical mjSTATE_INTEGRATION snapshot; "
            "claims remain local to the scripted-expert anchor distribution and finite action fan"),
    }
    with open(paths["metadata.json"], "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote {prefix}_{{candidates,summary,actions}}.csv and metadata.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
