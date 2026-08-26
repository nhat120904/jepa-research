#!/usr/bin/env python3
"""Stage-1 measurement: action-space curvature mismatch on OGBench-Cube.

For one snapshot, one action source, a grid of horizons and perturbation scales,
and several random directions, this compares

    model map     Phi_H(a) = F^H(E(o_t), a)      via LeWM's frozen rollout
    realized map  Psi_H(a) = E(o_H^sim(s_t, a))  via exact full-state reset

on the symmetric triplet ``(a - d, a, a + d)`` and writes every per-sample
quantity required by PROTOCOL.md readouts 1-8 and 10.

No training, no selection, no outcome is consulted.  All numerics come from
``action_curvature_h0.core``, which is unit-tested offline.

Perturbation scale convention: ``--sigmas`` are fractions of the RAW action
range ``high - low``, which is the only well-defined "range" here (the
normalized space is unbounded).  Directions are drawn isotropically in raw
action space and converted to normalized units through the StandardScaler, so
the model and the simulator see the same physical perturbation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from action_curvature_h0.core import (  # noqa: E402
    CEM_LOCAL_SOURCE,
    ContactTrace,
    Triplet,
    analyze_cost,
    analyze_triplet,
    classify_contact_mode,
    clip_validity,
    clip_to_bounds,
    cost_spread,
    draw_unit_directions,
    fit_scaling_exponent,
    make_feasible,
    ordinal_agreement,
    repeat_floor,
    scale_direction,
    state_anchor_curvature,
)

DIAGNOSIS_SCRIPTS = ROOT / "diagnosis" / "scripts"

DATASET_SOURCE = "dataset"
CEM_FIXED_SOURCE = "cem_fixed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "physical_search_distillation/outputs/h0/manifest.json")
    p.add_argument("--snapshot-index", type=int, required=True)
    p.add_argument("--n-snapshots", type=int, default=64,
                   help="Preregistered Stage-1 sample size; indices outside are refused.")
    p.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    p.add_argument("--checkpoint", default="quentinll/lewm-cube")
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--horizon", type=int, default=5,
                   help="Rollout horizon used by the frozen planner configuration.")
    p.add_argument("--horizons", default="1,3,5",
                   help="Horizons measured; each must be <= --horizon.")
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--sigmas",
                   default="0.00125,0.0025,0.005,0.01,0.025,0.05,0.10,0.20")
    p.add_argument("--n-directions", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--action-source", choices=[DATASET_SOURCE, CEM_FIXED_SOURCE, CEM_LOCAL_SOURCE],
                   default=DATASET_SOURCE)
    p.add_argument("--population-dir", type=Path,
                   default=ROOT / "physical_search_distillation/outputs/h0/populations")
    p.add_argument("--population-index", type=int, default=1,
                   help="0 = initial CEM population, 1 = final.")
    p.add_argument("--cem-local-alpha", type=float, default=1.0,
                   help="delta ~ N(0, alpha^2 diag(proposal_std^2)) for cem_local.")
    p.add_argument("--onset-tolerance", type=int, default=1,
                   help="Contact-onset steps differing by more than this are cross-mode.")
    p.add_argument("--repeats", type=int, default=3,
                   help="Repeated realized evaluations of the centre action (floor test).")
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--state-dict", type=Path, default=None,
                   help="checkpoint.pt from train_as.py; loaded on top of the "
                        "released weights so fine-tuned arms can be measured "
                        "through the identical diagnostic path.")
    p.add_argument("--probe", type=Path, default=None,
                   help="probe.npz from train_probe.py; enables the physical-space "
                        "bridge (Gate B and the metre-space primary metric).")
    p.add_argument("--model-dtype", choices=["float32", "float64"], default="float32",
                   help="Precision of the model forward pass. float64 tests whether "
                        "the ||D2 Phi|| floor is numerical rather than geometric.")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_script_module(filename: str, name: str) -> Any:
    source = DIAGNOSIS_SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "action_curvature_h0/core.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Simulator side
# --------------------------------------------------------------------------


def object_position(raw_env: Any) -> np.ndarray:
    """Declared task-state subset, part 1 (metres)."""
    return np.asarray(raw_env._data.joint("object_joint_0").qpos[:3], dtype=np.float64).copy()


def effector_position(raw_env: Any) -> np.ndarray:
    """Declared task-state subset, part 2 (metres); OGBench's ``proprio/effector_pos``."""
    return np.asarray(raw_env._data.site_xpos[raw_env._pinch_site_id], dtype=np.float64).copy()


def load_probe(path: Path) -> dict[str, np.ndarray]:
    """Load the affine latent -> physical-state bridge.

    P(z) = ((z - mu) / sigma) W + b.  Affine by construction, so second
    differences pass through exactly and the probe cannot manufacture
    curvature; see train_probe.py.
    """
    with np.load(path, allow_pickle=True) as z:
        return {"W": np.asarray(z["W"], dtype=np.float64),
                "b": np.asarray(z["b"], dtype=np.float64),
                "mu": np.asarray(z["mu"], dtype=np.float64),
                "sigma": np.asarray(z["sigma"], dtype=np.float64)}


def apply_probe(probe: dict[str, np.ndarray], z: np.ndarray) -> np.ndarray:
    """Decode latents to physical state, in metres."""
    zz = (np.asarray(z, dtype=np.float64) - probe["mu"]) / probe["sigma"]
    return zz @ probe["W"] + probe["b"]


def build_contact_classifier(raw_env: Any):
    """Resolve cube / static-scene / robot from MuJoCo body ids.

    No geom-name guessing: the cube body is the body of ``object_joint_0``, the
    static scene is the world body (id 0), and any other body touching the cube
    is the robot.

    Body resolution goes through ``mj_name2id`` + ``model.jnt_bodyid``, the
    canonical index arrays, rather than the named-view attribute
    ``model.joint(name).bodyid``.  In job 45721 that view returned six entries
    for this joint, so taking element zero would have silently selected the
    wrong body and turned the entire contact stratification into noise.  The
    size guard caught it; the index path removes the ambiguity.

    Returns the classifier plus the metadata it resolved, so the smoke output
    can be checked rather than trusted.
    """
    import mujoco

    model = raw_env._model
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint_0")
    if joint_id < 0:
        raise RuntimeError("object_joint_0 not found; contact stratification is impossible")
    cube_body = int(np.asarray(model.jnt_bodyid).reshape(-1)[joint_id])

    geom_body = np.asarray(model.geom_bodyid).reshape(-1)
    cube_geoms = frozenset(int(g) for g in np.nonzero(geom_body == cube_body)[0])
    if not cube_geoms:
        raise RuntimeError(
            f"body {cube_body} (from object_joint_0) owns no geoms; contact "
            "stratification is impossible and the run is refused"
        )

    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_body)
    geom_names = sorted(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)) for g in cube_geoms
    )
    world_geoms = int((geom_body == 0).sum())

    def classify(data: Any) -> tuple[bool, bool, set[int]]:
        """Returns (robot contact, static-scene contact, bodies touching the cube).

        The body set is reported, not just the booleans: in job 45863 every
        record had ``table_contact = False`` while the cube was expected to rest
        on a table, and "the cube is grasped" and "the table is not the world
        body, so its contact is being labelled robot" produce identical
        booleans.  Only the body identities separate them.
        """
        robot = table = False
        touched: set[int] = set()
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            in1, in2 = g1 in cube_geoms, g2 in cube_geoms
            if in1 == in2:  # cube-cube self contact, or no cube involved
                continue
            other = g2 if in1 else g1
            other_body = int(geom_body[other])
            touched.add(other_body)
            if other_body == 0:
                table = True
            else:
                robot = True
        return robot, table, touched

    def body_name_of(body_id: int) -> str:
        return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)))

    info = {
        "body_names": {
            str(b): body_name_of(b) for b in range(int(model.nbody))
        },
        "joint_id": int(joint_id),
        "cube_body_id": cube_body,
        "cube_body_name": str(body_name),
        "cube_geom_ids": sorted(cube_geoms),
        "cube_geom_names": geom_names,
        "n_world_geoms": world_geoms,
        "n_geoms_total": int(geom_body.size),
    }
    return classify, info


def true_rollout(
    raw_env: Any,
    init_row: dict,
    goal_row: dict,
    corrected: Any,
    audit: Any,
    raw_actions: np.ndarray,
    classify: Any,
) -> dict[str, Any]:
    """Exact-state reset, execute ``raw_actions``, return endpoint and contact trace.

    The trace is PER STEP, not a union over the rollout: two rollouts that touch
    the same bodies at different moments are not in the same dynamical mode, and
    a union would erase the onset time that distinguishes them.
    """
    corrected.restore_complete(
        raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit
    )
    robot_steps: list[bool] = []
    table_steps: list[bool] = []
    touched_bodies: set[int] = set()
    flat = raw_actions.reshape(-1, raw_actions.shape[-1])
    executed = 0
    for action in flat:
        _, _, terminated, truncated, _ = raw_env.step(action)
        executed += 1
        robot, table, touched = classify(raw_env._data)
        robot_steps.append(robot)
        table_steps.append(table)
        touched_bodies |= touched
        if terminated or truncated:
            break
    return {
        "frame": audit.resize_render(raw_env.render()),
        "object_pos": object_position(raw_env),
        "effector_pos": effector_position(raw_env),
        "trace": ContactTrace(robot=tuple(robot_steps), table=tuple(table_steps)),
        "touched_bodies": sorted(touched_bodies),
        "executed_steps": executed,
        "truncated_early": executed < flat.shape[0],
    }


# --------------------------------------------------------------------------
# Model side
# --------------------------------------------------------------------------


@torch.inference_mode()
def encode_images_dtype(
    model: Any, images: np.ndarray, transform: Any, batch_size: int, audit: Any
) -> np.ndarray:
    """Mirror of ``corrected.encode_images`` that respects the model dtype.

    The upstream helper hardcodes ``.float()`` on the embedding and feeds a
    float32 pixel tensor, which would silently defeat a float64 forward pass.
    """
    device = str(next(model.parameters()).device)
    dtype = next(model.parameters()).dtype
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    output: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        pixels = audit.transform_images(
            images[start : start + batch_size], transform, device
        ).to(dtype)
        emb = model.encode({"pixels": pixels})["emb"][:, -1].double()
        output.append(emb.reshape(len(emb), -1).cpu().numpy())
    return np.concatenate(output)


def model_terminal_latents(
    policy: Any,
    raw_info: dict,
    normalized_actions: np.ndarray,
    base: Any,
    model_dtype: Any,
) -> np.ndarray:
    """Frozen LeWM rollout for a batch of candidate action chunks.

    Reproduces the candidate/sample-axis expansion that ``CEMSolver.solve``
    performs internally; bypassing it sends a 3-D tensor into the ViT and
    silently reinterprets the RGB channel axis as LeWM's temporal axis.
    Mirrors ``rollout_repair_gate/scripts/collect_intermediates.py``.
    """
    n_samples = normalized_actions.shape[0]
    prepared = policy._prepare_info(raw_info)
    expanded: dict[str, Any] = {}
    for key, value in prepared.items():
        if torch.is_tensor(value):
            value = value.cuda()
            if value.is_floating_point():
                value = value.to(model_dtype)
            expanded[key] = value.unsqueeze(1).expand(
                value.shape[0], n_samples, *value.shape[1:]
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None, ...], n_samples, axis=1)
        else:
            expanded[key] = value
    action_np = np.ascontiguousarray(
        normalized_actions,
        dtype=np.float64 if model_dtype == torch.float64 else np.float32,
    )
    action_tensor = torch.from_numpy(action_np).cuda()[None]
    rolled = base._rollout(expanded, action_tensor)
    terminal = rolled["predicted_emb"][0, :, -1].double().cpu().numpy()
    goal = rolled["goal_emb"][0, -1].double().cpu().numpy()
    return {
        "terminal": np.asarray(terminal, dtype=np.float64),
        "goal": np.asarray(goal, dtype=np.float64),
    }


# --------------------------------------------------------------------------
# Triplet construction
# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 measurement requires the GPU nodes")

    horizons = [int(h) for h in args.horizons.split(",") if h]
    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    if max(horizons) > args.horizon:
        raise ValueError("every measured horizon must be <= the rollout horizon")

    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    audit = load_script_module("72_ogb_stage0_candidate_audit.py", "acm_audit")
    corrected = load_script_module("76_ogb_true_endpoint_corrected.py", "acm_corrected")
    corrected.load_stage0_transform_images = audit.transform_images

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < args.n_snapshots:
        raise ValueError(
            f"snapshot index outside the preregistered Stage-1 range "
            f"[0, {args.n_snapshots})"
        )
    if args.n_snapshots > len(manifest):
        raise ValueError("manifest is smaller than the preregistered sample size")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("manifest order/index mismatch")

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]

    # The scaler is fitted on finite rows only; indexing by storage_row must use
    # the UNFILTERED array, because the finite mask renumbers rows.
    action_data = np.asarray(dataset.get_col_data("action"))
    finite = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = StandardScaler().fit(finite)

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    embedder_patched = False
    if args.state_dict is not None:
        blob = torch.load(args.state_dict, map_location="cuda")
        missing, unexpected = model.load_state_dict(blob["model"], strict=True)
        state_dict_meta = {k: blob.get(k) for k in ("lambda_as", "seed", "steps")}
    else:
        state_dict_meta = None
    if args.model_dtype == "float64":
        model = model.double()
        # Upstream LeWM hardcodes a float32 downcast in the action embedder
        # (stable_worldmodel/wm/lewm/module.py:204, `x = x.float()`), the only
        # such cast in that module.  Left alone it silently defeats a float64
        # forward pass and raises at the Conv1d bias.  Patched at runtime rather
        # than by editing the vendored checkout, so every other pilot in this
        # repo keeps its provenance, and only under float64 so float32 runs stay
        # bit-identical to all earlier results.
        from stable_worldmodel.wm.lewm.module import Embedder

        src = inspect.getsource(Embedder.forward)
        if "x = x.float()" not in src:
            raise RuntimeError(
                "Embedder.forward no longer contains the expected float32 "
                "downcast; the precision patch must be re-derived"
            )

        def _forward_dtype_preserving(self, x):
            x = x.to(next(self.parameters()).dtype)
            x = x.permute(0, 2, 1)
            x = self.patch_embed(x)
            x = x.permute(0, 2, 1)
            x = self.embed(x)
            return x

        Embedder.forward = _forward_dtype_preserving
        embedder_patched = True
    model_dtype = next(model.parameters()).dtype
    if str(model_dtype) != f"torch.{args.model_dtype}":
        raise RuntimeError(
            f"requested {args.model_dtype} but model parameters are {model_dtype}; "
            "refusing to run a precision test at the wrong precision"
        )
    transform = audit.make_transform(224)
    probe = load_probe(args.probe) if args.probe is not None else None

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1_000 * args.snapshot_index)

    world, raw_env, visual_hash, _ = corrected.make_world(swm, snapshot)
    contact_info: dict[str, Any] = {}
    feasibility: list[dict[str, Any]] = []
    centre_clipping: dict[str, float] = {}
    clip_diagnostic: dict[str, Any] | None = None
    try:
        # Runtime check: the effector accessor is verified statically through
        # script 84's call path, but nothing may depend on that silently.
        if not hasattr(raw_env, "_pinch_site_id"):
            raise RuntimeError(
                "raw_env has no _pinch_site_id; the declared task-state subset "
                "cannot be read and the run is refused"
            )

        classify, contact_info = build_contact_classifier(raw_env)
        # Task target the physical cost is measured against, in metres.
        goal_object_pos = np.asarray(
            audit.goal_field(goal_row, "block_0_pos"), dtype=np.float64
        ).reshape(-1)

        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        low = np.asarray(world.envs.single_action_space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(world.envs.single_action_space.high, dtype=np.float64).reshape(-1)
        raw_range = high - low

        base_evaluator = swm.planning.ShootingCostEvaluator(
            model, swm.planning.GoalMSE()
        )
        config = swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.horizon,
            action_block=args.action_block,
            history_len=1,
            warm_start=True,
        )
        solver = swm.planning.CEMSolver(
            cost=base_evaluator, batch_size=1, num_samples=1, n_steps=1, topk=1,
            device="cuda", seed=0,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process={"action": scaler},
            transform={"pixels": transform, "goal": transform},
        )
        policy.set_env(world.envs)
        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
        }

        centre_normalized, proposal_std = resolve_centre(
            args, snapshot, action_data, scaler, raw_dim, rng
        )
        # A centre outside the action box cannot be rescued by any perturbation;
        # the simulator executes the clipped action regardless, so probe around
        # what actually runs.  A logged expert chunk is already interior, so this
        # is a no-op for the dataset source and the info line proves it.
        centre_raw_pre = normalized_to_raw_chunk(
            centre_normalized, scaler, args, raw_dim
        )
        centre_raw_clipped, centre_clipping = clip_to_bounds(
            centre_raw_pre.reshape(-1, raw_dim), low, high
        )
        centre_normalized = scaler.transform(centre_raw_clipped).reshape(
            centre_normalized.shape
        )

        records: list[dict[str, Any]] = []
        floor = measure_floor(
            args, corrected, audit, raw_env, init_row, goal_row, model,
            transform, scaler, centre_normalized, raw_dim, classify,
        )

        raw_range_chunk = np.broadcast_to(
            np.broadcast_to(raw_range, (args.action_block, raw_dim)).reshape(-1),
            centre_normalized.shape,
        ).astype(np.float64)
        scaler_scale_chunk = np.broadcast_to(
            np.tile(scaler.scale_, args.action_block), centre_normalized.shape
        ).astype(np.float64)
        sigma_max = max(sigmas)

        # Per-component headroom to the action bounds, in raw units.
        centre_raw_chunk = normalized_to_raw_chunk(
            centre_normalized, scaler, args, raw_dim
        ).reshape(centre_normalized.shape)
        low_chunk = np.broadcast_to(
            np.tile(low, args.action_block), centre_normalized.shape
        )
        high_chunk = np.broadcast_to(
            np.tile(high, args.action_block), centre_normalized.shape
        )
        margin = np.minimum(
            high_chunk - centre_raw_chunk, centre_raw_chunk - low_chunk
        ).clip(min=0.0)

        # Largest |base[i]| that still lands inside the bounds at the top sigma.
        if args.action_source == CEM_LOCAL_SOURCE:
            denom = args.cem_local_alpha * proposal_std * scaler_scale_chunk
        else:
            denom = sigma_max * raw_range_chunk
        cap = margin / np.maximum(denom, 1e-12)
        margin_diag = {
            "margin_min": float(margin.min()), "margin_max": float(margin.max()),
            "cap_min": float(cap.min()), "cap_max": float(cap.max()),
            "centre_clipped_dims": int((margin <= 0.0).sum()),
            "denom_min": float(denom.min()), "denom_max": float(denom.max()),
        }

        # Drawn once, then made feasible by construction rather than drawn and
        # rejected: job 45863 discarded every H=5 triplet under the old
        # all-or-nothing clip rule.
        raw_bases = draw_unit_directions(
            rng, args.n_directions, centre_normalized.shape
        )
        base_directions = []
        feasibility = []
        for k in range(args.n_directions):
            adjusted, info = make_feasible(
                raw_bases[k].reshape(-1), cap.reshape(-1)
            )
            base_directions.append(adjusted.reshape(centre_normalized.shape))
            feasibility.append({"direction": k, **info})
        feasibility[0] = {**feasibility[0], "margin_diagnostics": margin_diag}

        for horizon in horizons:
            for sigma in sigmas:
                for direction, base in enumerate(base_directions):
                    if feasibility[direction]["feasible"] == 0.0:
                        continue  # fully saturated chunk; recorded in the summary
                    delta = scale_direction(
                        base, sigma, source=args.action_source,
                        raw_range_chunk=raw_range_chunk,
                        scaler_scale_chunk=scaler_scale_chunk,
                        proposal_std=proposal_std,
                        alpha=args.cem_local_alpha, sigma_max=sigma_max,
                    )
                    record = measure_one(
                        args=args, corrected=corrected, audit=audit, raw_env=raw_env,
                        init_row=init_row, goal_row=goal_row, model=model,
                        transform=transform, scaler=scaler, policy=policy,
                        raw_info=raw_info, base=base_evaluator, classify=classify,
                        centre=centre_normalized, delta=delta, horizon=horizon,
                        raw_dim=raw_dim, low=low, high=high,
                        model_dtype=model_dtype, probe=probe,
                        goal_object_pos=goal_object_pos,
                    )
                    record.update(
                        {"horizon": horizon, "sigma": sigma, "direction": direction}
                    )
                    if (not record["valid_unclipped"] and sigma == min(sigmas)
                            and clip_diagnostic is None):
                        # A direction the feasibility pass called safe should
                        # never clip at its smallest scale.  Capture exactly why
                        # once, rather than reasoning about it after the fact.
                        centre_raw_probe = normalized_to_raw_chunk(
                            centre_normalized, scaler, args, raw_dim
                        )[:horizon].reshape(-1, raw_dim)
                        delta_raw_probe = (
                            normalized_to_raw_chunk(
                                centre_normalized + delta, scaler, args, raw_dim
                            )[:horizon]
                            - normalized_to_raw_chunk(
                                centre_normalized, scaler, args, raw_dim
                            )[:horizon]
                        ).reshape(-1, raw_dim)
                        over_high = centre_raw_probe + delta_raw_probe - high
                        under_low = low - (centre_raw_probe - delta_raw_probe)
                        clip_diagnostic = {
                            "horizon": horizon, "sigma": sigma, "direction": direction,
                            "max_over_high": float(np.max(over_high)),
                            "max_under_low": float(np.max(under_low)),
                            "feasibility_at_direction": feasibility[direction],
                        }
                    records.append(record)
    finally:
        world.close()

    write_outputs(args, snapshot, records, floor, visual_hash, horizons, sigmas,
                  contact_info, feasibility, centre_clipping, clip_diagnostic,
                  embedder_patched, state_dict_meta)


def resolve_centre(
    args: argparse.Namespace,
    snapshot: Any,
    action_data: np.ndarray,
    scaler: Any,
    raw_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Centre action chunk in normalized units, shape ``(horizon, block*dim)``."""
    n_steps = args.horizon * args.action_block
    if args.action_source == DATASET_SOURCE:
        start = int(snapshot.storage_row)
        chunk = np.asarray(action_data[start : start + n_steps], dtype=np.float64)
        if chunk.shape != (n_steps, raw_dim):
            raise RuntimeError(f"logged action chunk has shape {chunk.shape}")
        if np.isnan(chunk).any():
            raise RuntimeError("logged action chunk contains NaN; snapshot unusable")
        normalized = scaler.transform(chunk).reshape(
            args.horizon, args.action_block * raw_dim
        )
        return np.asarray(normalized, dtype=np.float64), None

    path = args.population_dir / f"snapshot_{snapshot.order:03d}/populations.npz"
    with np.load(path) as source:
        actions = np.asarray(source["actions_normalized"], dtype=np.float64)
        elite = np.asarray(source["native_elite"], dtype=np.int64)
        proposal_std = np.asarray(source["proposal_std"], dtype=np.float64)
    population = args.population_index
    # Centre on a uniformly drawn elite of the recorded population: the elites
    # are the candidates the planner actually retains and refits around.
    choice = int(rng.integers(elite.shape[1]))
    centre = actions[population, int(elite[population, choice])]
    if centre.shape != (args.horizon, args.action_block * raw_dim):
        raise RuntimeError(f"population action chunk has shape {centre.shape}")
    return centre, proposal_std[population]


def measure_floor(
    args, corrected, audit, raw_env, init_row, goal_row, model, transform,
    scaler, centre_normalized, raw_dim, classify,
) -> dict[str, float]:
    """Repeat the realized evaluation of the centre action (protocol floor test)."""
    raw = normalized_to_raw_chunk(centre_normalized, scaler, args, raw_dim)
    frames = []
    for _ in range(max(2, args.repeats)):
        result = true_rollout(
            raw_env, init_row, goal_row, corrected, audit, raw, classify
        )
        frames.append(result["frame"])
    embeddings = encode_images_dtype(
        model, np.stack(frames), transform, args.encode_batch, audit
    )
    value = repeat_floor([np.asarray(e, dtype=np.float64) for e in embeddings])
    return {"repeat_floor": float(value), "repeats": len(frames)}


def normalized_to_raw_chunk(
    normalized: np.ndarray, scaler: Any, args: argparse.Namespace, raw_dim: int
) -> np.ndarray:
    """``(horizon, block*dim)`` normalized -> ``(horizon, block, dim)`` raw."""
    flat = np.asarray(normalized, dtype=np.float64).reshape(-1, raw_dim)
    return scaler.inverse_transform(flat).reshape(
        normalized.shape[0], args.action_block, raw_dim
    )


def measure_one(
    *, args, corrected, audit, raw_env, init_row, goal_row, model, transform,
    scaler, policy, raw_info, base, classify, centre, delta, horizon, raw_dim,
    low, high, model_dtype, probe, goal_object_pos,
) -> dict[str, Any]:
    """One symmetric triplet at one horizon, one scale, one direction."""
    chunks_norm = np.stack([centre - delta, centre, centre + delta])
    chunks_raw = np.stack(
        [normalized_to_raw_chunk(c, scaler, args, raw_dim) for c in chunks_norm]
    )

    used_norm = chunks_norm[:, :horizon]
    used_raw = chunks_raw[:, :horizon]

    # Validity is decided in RAW action space, where the bounds live, and ONLY
    # over the actions this horizon actually executes.  Checking the full chunk
    # first would discard a valid short-horizon triplet because of a bound
    # violation in tail actions that are never stepped.
    centre_raw = used_raw[1].reshape(-1, raw_dim)
    delta_raw = (used_raw[2] - used_raw[1]).reshape(-1, raw_dim)
    valid = all(
        clip_validity(centre_raw[i], delta_raw[i], low, high)
        for i in range(centre_raw.shape[0])
    )

    rolled = model_terminal_latents(
        policy, raw_info, used_norm, base, model_dtype
    )
    model_terminal, goal_embedding = rolled["terminal"], rolled["goal"]

    true_results = [
        true_rollout(raw_env, init_row, goal_row, corrected, audit, used_raw[i], classify)
        for i in range(3)
    ]
    frames = np.stack([r["frame"] for r in true_results])
    true_terminal = np.asarray(
        encode_images_dtype(model, frames, transform, args.encode_batch, audit),
        dtype=np.float64,
    )

    # Resolution-floor diagnostic: the same rendered triplet the encoder just
    # consumed, characterized in raw pixel space, at zero extra render cost.
    # Distinguishes two very different explanations for a flat latent alpha:
    # the rasterizer never moved (sub-pixel motion, alpha_pixel ~ 0 too), or the
    # image moved smoothly but the encoder/patch tokenization discards it
    # (alpha_pixel >> alpha_latent).
    pixel_triplet = Triplet(
        frames[0].astype(np.float64).reshape(-1),
        frames[1].astype(np.float64).reshape(-1),
        frames[2].astype(np.float64).reshape(-1),
    )
    pixel_abs_diff = np.abs(
        frames[2].astype(np.float64) - frames[0].astype(np.float64)
    )

    delta_norm_value = float(np.linalg.norm(delta[:horizon]))
    latent_model = Triplet(model_terminal[0], model_terminal[1], model_terminal[2])
    latent_true = Triplet(true_terminal[0], true_terminal[1], true_terminal[2])

    report = analyze_triplet(latent_model, latent_true, delta_norm=delta_norm_value)
    cost = analyze_cost(latent_model, latent_true, goal_embedding)

    object_triplet = Triplet(*[r["object_pos"] for r in true_results])
    effector_triplet = Triplet(*[r["effector_pos"] for r in true_results])

    # Ordinal endpoint.  The model cost depends on the action ONLY through F:
    # the encoder is applied to the initial observation and to the goal, both
    # shared across the triplet, so the renderer contributes constants that
    # cancel out of any comparison between the three.  The physical cost comes
    # straight from simulator state.  Neither side passes a varying quantity
    # through the rasterizer, which is why this survives where both curvature
    # bridges failed.
    model_costs = tuple(
        float(np.sum((model_terminal[i] - goal_embedding) ** 2)) for i in range(3)
    )
    physical_costs = tuple(
        float(np.linalg.norm(true_results[i]["object_pos"] - goal_object_pos))
        for i in range(3)
    )
    ordinal = ordinal_agreement(model_costs, physical_costs)

    bridge: dict[str, Any] = {}
    if probe is not None:
        # Gate B isolates the BRIDGE: decode the encoded TRUE endpoints and
        # compare their curvature against the simulator's own, with no
        # predictor anywhere in the path.  A failure here indicts the probe,
        # not the world model -- which is the whole point of separating it from
        # the primary metric below.
        dec_true = apply_probe(probe, true_terminal)
        dec_model = apply_probe(probe, model_terminal)
        for name, sl, truth in (("object", slice(0, 3), object_triplet),
                                ("effector", slice(3, 6), effector_triplet)):
            bridge_tri = Triplet(dec_true[0, sl], dec_true[1, sl], dec_true[2, sl])
            model_tri = Triplet(dec_model[0, sl], dec_model[1, sl], dec_model[2, sl])
            d2_truth = truth.d2
            bridge.update({
                f"d2_true_state_{name}_m": float(np.linalg.norm(d2_truth)),
                f"d2_bridge_{name}_m": float(np.linalg.norm(bridge_tri.d2)),
                # Gate B numerator: curvature the bridge adds or loses.
                f"d2e_{name}_m": float(np.linalg.norm(bridge_tri.d2 - d2_truth)),
                # Primary metric, both sides in metres.
                f"d2_model_decoded_{name}_m": float(np.linalg.norm(model_tri.d2)),
                f"d2_mismatch_{name}_m": float(np.linalg.norm(model_tri.d2 - d2_truth)),
                # Absolute bridge fidelity at the centre, for reference only.
                f"bridge_centre_err_{name}_m": float(
                    np.linalg.norm(bridge_tri.center - truth.center)
                ),
            })

    mode = classify_contact_mode(
        true_results[0]["trace"], true_results[1]["trace"], true_results[2]["trace"],
        onset_tolerance=args.onset_tolerance,
    )

    row: dict[str, Any] = dict(report.as_dict())
    row.update(cost.as_dict())
    row.update(
        {
            "valid_unclipped": bool(valid),
            "mode": mode,
            "delta_norm": delta_norm_value,
            "d2_true_raw_norm": float(np.linalg.norm(latent_true.d2)),
            "k_true_state_object": state_anchor_curvature(object_triplet),
            "k_true_state_effector": state_anchor_curvature(effector_triplet),
            "object_span_m": float(np.linalg.norm(object_triplet.plus - object_triplet.minus)),
            "effector_span_m": float(
                np.linalg.norm(effector_triplet.plus - effector_triplet.minus)
            ),
            "truncated_early": any(r["truncated_early"] for r in true_results),
            "model_cost_minus": model_costs[0],
            "model_cost_centre": model_costs[1],
            "model_cost_plus": model_costs[2],
            "physical_cost_minus_m": physical_costs[0],
            "physical_cost_centre_m": physical_costs[1],
            "physical_cost_plus_m": physical_costs[2],
            **{f"ordinal_{k}": v for k, v in ordinal.items()},
            **bridge,
            "pixel_d2_norm": float(np.linalg.norm(pixel_triplet.d2)),
            "pixel_span": pixel_triplet.span,
            "pixel_max_abs_diff": float(pixel_abs_diff.max()),
            "pixel_nonzero_diff_fraction": float((pixel_abs_diff > 0.0).mean()),
            "pixel_dtype": str(frames.dtype),
            "pixel_value_max": float(frames.max()),
            # Covariates, not stratifiers: the cube rests on the table for
            # essentially every step, so table contact carries no mode signal.
            "table_contact_any": bool(
                any(any(r["trace"].table) for r in true_results)
            ),
            "touched_bodies": sorted(
                {b for r in true_results for b in r["touched_bodies"]}
            ),
            "onset_centre": (
                -1 if true_results[1]["trace"].onset is None
                else int(true_results[1]["trace"].onset)
            ),
        }
    )
    return row


def write_outputs(
    args: argparse.Namespace,
    snapshot: Any,
    records: list[dict[str, Any]],
    floor: dict[str, float],
    visual_hash: str,
    horizons: list[int],
    sigmas: list[float],
    contact_info: dict[str, Any],
    feasibility: list[dict[str, Any]],
    centre_clipping: dict[str, float],
    clip_diagnostic: dict[str, Any] | None,
    embedder_patched: bool,
    state_dict_meta: dict[str, Any] | None,
) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # The JSON is the authoritative record and is written FIRST.  In job 45974
    # the whole measurement completed and was then lost because a serialization
    # bug in the derived npz path ran before anything had been persisted.
    (out_dir / "records.json").write_text(json.dumps(records, indent=2, default=str))

    # Only scalar numeric fields go into the npz; dtype is decided by
    # inspection, not by a hardcoded exception list.
    keys = sorted({k for r in records for k in r})
    arrays: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for key in keys:
        values = [r.get(key) for r in records]
        sample = next((v for v in values if v is not None), None)
        if isinstance(sample, (str, list, tuple, dict)) or sample is None:
            skipped.append(key)
            continue
        arrays[key] = np.array(
            [np.nan if v is None else float(v) for v in values], dtype=np.float64
        )
    np.savez_compressed(out_dir / "curvature.npz", **arrays)

    # Readout 6: per (direction, horizon) log-log exponent over the scale sweep.
    # Pixel-space alpha is fit alongside latent alpha, on the identical triplet,
    # to localize a flat latent alpha to the rasterizer (pixel alpha flat too)
    # versus the encoder (pixel alpha smooth, latent alpha flat).
    fits: list[dict[str, Any]] = []
    pixel_fits: list[dict[str, Any]] = []
    for horizon in horizons:
        for direction in range(args.n_directions):
            chosen = [
                r for r in records
                if r["horizon"] == horizon and r["direction"] == direction
                and r["valid_unclipped"]
            ]
            if len(chosen) < 3:
                continue
            chosen.sort(key=lambda r: r["sigma"])
            fit = fit_scaling_exponent(
                [r["delta_norm"] for r in chosen],
                [r["d2_true_raw_norm"] for r in chosen],
                floor=floor["repeat_floor"],
            )
            fits.append(
                {"horizon": horizon, "direction": direction, "alpha": fit.alpha,
                 "r_squared": fit.r_squared, "n_used": fit.n_used,
                 "excluded_below_floor": fit.excluded_below_floor}
            )
            # Pixel values are integer-quantized (typically 8-bit), so the pixel
            # floor is not the repeat-evaluation floor; one raw grey-level step
            # is the natural unit and is used as the fit floor here.
            pfit = fit_scaling_exponent(
                [r["delta_norm"] for r in chosen],
                [r["pixel_d2_norm"] for r in chosen],
                floor=1.0,
            )
            pixel_fits.append(
                {"horizon": horizon, "direction": direction, "alpha": pfit.alpha,
                 "r_squared": pfit.r_squared, "n_used": pfit.n_used,
                 "excluded_below_floor": pfit.excluded_below_floor,
                 "max_abs_diff_at_largest_sigma": chosen[-1]["pixel_max_abs_diff"],
                 "max_abs_diff_at_smallest_sigma": chosen[0]["pixel_max_abs_diff"],
                 "nonzero_diff_fraction_at_smallest_sigma":
                     chosen[0]["pixel_nonzero_diff_fraction"]}
            )

    discard: dict[str, dict[str, int]] = {}
    for r in records:
        bucket = discard.setdefault(r["mode"], {"total": 0, "discarded": 0})
        bucket["total"] += 1
        if not r["valid_unclipped"]:
            bucket["discarded"] += 1

    summary = {
        "snapshot": {"order": snapshot.order, "episode": snapshot.episode,
                     "start_step": snapshot.start_step,
                     "storage_row": snapshot.storage_row,
                     "reset_seed": snapshot.reset_seed},
        "action_source": args.action_source,
        "population_index": args.population_index,
        "n_snapshots_preregistered": args.n_snapshots,
        "horizons": horizons,
        "sigmas": sigmas,
        "n_directions": args.n_directions,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "visual_signature": visual_hash,
        "contact_resolution": contact_info,
        "embedder_float32_patch_applied": embedder_patched,
        "probe_path": str(args.probe) if args.probe is not None else None,
        "state_dict": str(args.state_dict) if args.state_dict is not None else None,
        "state_dict_meta": state_dict_meta,
        "direction_feasibility": feasibility,
        "centre_clipping": centre_clipping,
        "clip_diagnostic": clip_diagnostic,
        "code_sha256": code_hash(),
        "floor": floor,
        "scaling_fits": fits,
        "pixel_scaling_fits": pixel_fits,
        "pixel_dtype": records[0]["pixel_dtype"] if records else None,
        "pixel_value_max": records[0]["pixel_value_max"] if records else None,
        # Near-bound actions are not uniformly distributed across contact
        # regimes, so a differential discard rate biases the mode comparison.
        "discard_by_mode": discard,
        "n_records": len(records),
        "npz_skipped_non_scalar_keys": skipped,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
