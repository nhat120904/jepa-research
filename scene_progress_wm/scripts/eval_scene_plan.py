"""Closed-loop planning evaluation on OGBench-Scene.

One shard = one contiguous range of the locked episode list, so the run can be split
across jobs without changing which episodes any arm sees. Every arm at a given
``--plan-seed`` and ``--goal-offset`` evaluates exactly the same (start, goal) pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.scene_eval import (  # noqa: E402
    PROTOCOL,
    FrameEncoder,
    PlanSettings,
    build_episode_specs,
    load_cache,
    run_episode,
)
from scene_progress_wm.scene_lewm import LeWMConfig, build_lewm  # noqa: E402
from scene_progress_wm.scene_render import make_scene_env  # noqa: E402
from scene_progress_wm.progress_head import (  # noqa: E402
    ProgressConfig,
    ProgressHead,
    ProgressTracker,
)
from scene_progress_wm.progress_objective import (  # noqa: E402
    ProgressCost,
    ScaledGoalMSE,
    build_mixture,
)

OBJECTIVES = ("latent_l2", "progress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="latent_l2")
    parser.add_argument("--progress-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--mixture-weight",
        type=float,
        default=0.0,
        help="weight on latent-L2; 1.0 reproduces the baseline exactly",
    )
    parser.add_argument(
        "--goal-scale",
        type=float,
        default=1.0,
        help="divisor putting latent-L2 on the progress scale; measured once on "
             "the baseline arm and then frozen, never refit per arm",
    )
    parser.add_argument("--goal-offset", type=int, required=True)
    parser.add_argument("--eval-budget", type=int, default=0, help="0 = 2x goal offset")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--episode-seed", type=int, default=90100)
    parser.add_argument("--plan-seed", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _latent_fn(model):
    """Encode one normalised frame to the planning latent, as the rollout does."""

    def latent_of(frame: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.encode({"pixels": frame.unsqueeze(0).unsqueeze(0)})["emb"][0, 0]

    return latent_of


def build_objective(args, device, config):
    """Return ``(objective, tracker, arm)``; tracker and arm are ``None`` for latent-L2.

    Only the objective differs between arms. The world model, the solver, the episode
    list and every seed are shared, which is what makes the ladder a controlled
    comparison rather than a collection of separate runs.
    """
    from stable_worldmodel.planning import GoalMSE

    if args.objective == "latent_l2":
        return GoalMSE(pred_key="predicted_emb", goal_key="goal_emb"), None, None

    if args.progress_checkpoint is None:
        raise ValueError("the progress objective needs --progress-checkpoint")
    payload = torch.load(args.progress_checkpoint, map_location="cpu", weights_only=False)
    head_config = ProgressConfig(**payload["config"])
    head = ProgressHead(head_config)
    head.load_state_dict(payload["state_dict"])
    head = head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    tracker = ProgressTracker(head, device)
    progress_term = ProgressCost(head, tracker, history_size=config.history_size)
    goal_term = ScaledGoalMSE(args.goal_scale)
    objective = build_mixture(goal_term, progress_term, args.mixture_weight)
    return objective, tracker, str(payload["arm"])


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("planning evaluation must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("planning evaluation requires a GPU allocation")

    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = LeWMConfig(**payload["config"])
    model = build_lewm(config)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    meta = json.loads((args.cache_dir / "meta.json").read_text())
    if int(meta["render_size"]) != config.img_size:
        raise RuntimeError("cache render size does not match the model image size")

    cache, track = load_cache(args.cache_dir)
    budget = args.eval_budget if args.eval_budget > 0 else 2 * args.goal_offset
    # Screen out starts that already satisfy their own goal. The replay gate measured
    # 17/50 such episodes at offset 25 and 8/50 at offset 100: free successes for every
    # arm that compress exactly the range the offset ladder exists to resolve. The
    # screen reads only the dataset, so all arms still see identical episodes.
    specs = build_episode_specs(
        num_rows=int(meta["num_rows"]),
        goal_offset=args.goal_offset,
        num_episodes=args.num_episodes,
        seed=args.episode_seed,
        track=track,
        buttons=cache["button_states"],
    )
    shard = specs[args.shard_index :: args.num_shards]

    settings = PlanSettings(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        num_samples=args.num_samples,
        cem_steps=args.cem_steps,
        topk=args.topk,
    )

    world, raw, render_info = make_scene_env(
        render_size=int(meta["render_size"]),
        quality=meta["render_quality"],
        camera=meta["camera"],
    )
    if render_info != meta["render_info"]:
        raise RuntimeError(
            f"renderer configuration drifted from the cache: {render_info} != {meta['render_info']}"
        )
    encoder = FrameEncoder(device)
    objective, tracker, progress_arm = build_objective(args, device, config)
    objective = objective.to(device)

    records = []
    started = time.time()
    for position, spec in enumerate(shard):
        # the plan seed is per-episode so that arms stay paired episode by episode
        plan_seed = args.plan_seed + 1_000_003 * spec.episode + 10_007 * args.goal_offset
        hooks = {}
        if tracker is not None:
            latent_of = _latent_fn(model)
            hooks = {
                "on_start": lambda frame: tracker.reset(latent_of(frame)),
                "on_step": lambda frame, block: tracker.observe(latent_of(frame), block),
            }
        record = run_episode(
            raw=raw,
            spec=spec,
            cache=cache,
            track=track,
            model=model,
            objective=objective,
            settings=settings,
            config=config,
            device=device,
            encoder=encoder,
            camera=meta["camera"],
            eval_budget=budget,
            plan_seed=plan_seed,
            **hooks,
        )
        record["plan_seed"] = plan_seed
        record["objective"] = args.objective
        if tracker is not None:
            # the head must have integrated exactly the executed transitions: one reset
            # plus one update per executed action block. A mismatch would mean CEM's
            # imagined rollouts had leaked into the memory of what actually happened.
            #
            # The count is over blocks *entered*, not whole blocks completed: the loop
            # in run_episode calls on_step once per block even when success or the
            # budget truncates that block mid-way, so a 11-step episode enters 3 blocks
            # of 5 and advances the tracker 4 times. Flooring instead of ceiling was
            # only ever right for episodes ending exactly on a block boundary, which is
            # why every 50-step baseline episode passed and the first truncated one did
            # not. Ceiling encodes the invariant the protocol actually specifies.
            blocks_entered = -(-record["env_steps"] // config.action_block)
            expected = blocks_entered + 1
            record["progress_updates"] = tracker.updates
            record["progress_updates_expected"] = expected
            if tracker.updates != expected:
                raise RuntimeError(
                    f"progress tracker advanced {tracker.updates} times, expected {expected}"
                )
        records.append(record)
        print(
            f"[{position + 1}/{len(shard)}] episode {spec.episode} "
            f"success={record['success']} steps={record['env_steps']} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    successes = [bool(r["success"]) for r in records]
    summary = {
        "protocol": PROTOCOL,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "objective": args.objective,
        "mixture_weight": args.mixture_weight,
        "goal_scale": args.goal_scale,
        "progress_checkpoint": (
            str(args.progress_checkpoint) if args.progress_checkpoint else None
        ),
        "progress_checkpoint_sha256": (
            file_sha256(args.progress_checkpoint) if args.progress_checkpoint else None
        ),
        "progress_arm": progress_arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "cache_meta": meta,
        "render_info": render_info,
        "goal_offset": args.goal_offset,
        "eval_budget": budget,
        "num_episodes": args.num_episodes,
        "episode_seed": args.episode_seed,
        "plan_seed": args.plan_seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "settings": settings.to_json(),
        "model_config": config.to_json(),
        "num_records": len(records),
        "num_success": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else None,
        "wall_seconds": time.time() - started,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # the arm name has to carry the weight: two arms of the same objective at different
    # mixture weights are different arms and must not overwrite each other
    arm_tag = args.objective if args.objective == "latent_l2" else (
        f"{progress_arm}_w{args.mixture_weight:.2f}"
    )
    name = f"{arm_tag}_off{args.goal_offset}_seed{args.plan_seed}_shard{args.shard_index}.json"
    (args.out_dir / name).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "objective": args.objective,
                "goal_offset": args.goal_offset,
                "plan_seed": args.plan_seed,
                "shard": args.shard_index,
                "num_records": len(records),
                "success_rate": summary["success_rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
