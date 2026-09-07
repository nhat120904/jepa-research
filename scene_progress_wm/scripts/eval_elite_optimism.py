#!/usr/bin/env python3
"""Instrument gate: does CEM select for the cost's blind spots?

The planning loop here mirrors ``scene_eval.run_episode`` step for step -- same
solver, same objective, same receding horizon, same success predicate -- and
adds one thing: at chosen replans it freezes the simulator, replays candidate
chunks from that frozen state, and finishes each one with OGBench's own Markov
controllers to obtain a true remaining cost in environment steps.

The comparison the gate exists for is *within* a replan.  Every stratum sees the
identical snapshot, goal and horizon, so the only thing separating them is where
in the CEM optimisation the candidate came from.  If the cost's ordering degrades
from the initial proposal distribution to the final elites, optimisation is
walking into the cost's error; if it does not, model exploitation is not what is
holding this arena back and a conservative objective has nothing to remove.

Running the same arm at two ``--plan-seed`` values, or two arms at one seed,
produces paired records over the same episodes.
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

from event_smdp_h0.scripts.run_scene_gate0 import SceneSnapshotManager  # noqa: E402
from scene_progress_wm.elite_optimism import (  # noqa: E402
    PROTOCOL,
    OracleRepair,
    ReplanRecord,
    draw_strata,
    make_recorder,
    summarise,
)
from scene_progress_wm.scene_eval import (  # noqa: E402
    FrameEncoder,
    HistoryWindow,
    PlanSettings,
    build_episode_specs,
    load_cache,
    make_solver,
    state_row,
)
from scene_progress_wm.scene_lewm import LeWMConfig, build_lewm  # noqa: E402
from scene_progress_wm.scene_render import (  # noqa: E402
    env_successes,
    make_scene_env,
    restore_state,
    set_goal_targets,
)
from scene_progress_wm.scripts.eval_scene_plan import (  # noqa: E402
    _latent_fn,
    build_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=("latent_l2", "progress"), default="latent_l2")
    parser.add_argument("--progress-checkpoint", type=Path, default=None)
    parser.add_argument("--mixture-weight", type=float, default=0.0)
    parser.add_argument("--goal-scale", type=float, default=1.0)
    parser.add_argument("--goal-offset", type=int, required=True)
    parser.add_argument("--eval-budget", type=int, default=0, help="0 = 2x goal offset")
    parser.add_argument("--num-episodes", type=int, default=12)
    parser.add_argument("--episode-seed", type=int, default=90100)
    parser.add_argument("--plan-seed", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    # -- instrument budget --------------------------------------------------
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=16,
        help="equal n drawn from each population stratum at every instrumented replan",
    )
    parser.add_argument(
        "--max-instrumented-replans",
        type=int,
        default=3,
        help="cap per episode; ground truth costs an oracle rollout per candidate",
    )
    parser.add_argument("--replan-stride", type=int, default=1)
    parser.add_argument("--max-repair-skills", type=int, default=8)
    parser.add_argument(
        "--deltas",
        default="0,10,25,50",
        help="true-cost margins in environment steps; swept, not chosen",
    )
    parser.add_argument(
        "--delta-probe-repeats",
        type=int,
        default=3,
        help="re-executions of one candidate per episode, to measure the noise "
             "floor the margin sweep is reported against",
    )
    parser.add_argument("--draw-seed", type=int, default=70100)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def execute_chunk(raw, actions_flat: np.ndarray, config: LeWMConfig, blocks: int) -> dict:
    """Run one candidate's committed blocks, exactly as the closed loop would.

    The clip is where the model and the simulator can disagree: upstream CEM
    samples an unbounded Gaussian, so a candidate may be scored outside the
    action bounds and executed inside them.  The fraction clipped is recorded
    rather than silently absorbed.
    """
    steps = 0
    clipped = 0
    total = 0
    for block_index in range(blocks):
        block = np.asarray(actions_flat[block_index], dtype=np.float32)
        block = block.reshape(config.action_block, config.action_dim)
        clipped += int(np.sum((block < -1.0) | (block > 1.0)))
        total += int(block.size)
        for action in np.clip(block, -1.0, 1.0).astype(np.float32):
            raw.step(action)
            steps += 1
            if env_successes(raw)["success"]:
                return {
                    "steps": steps,
                    "solved": True,
                    "clipped_frac": clipped / total,
                    "match": env_successes(raw),
                }
    return {
        "steps": steps,
        "solved": False,
        "clipped_frac": clipped / max(total, 1),
        # the component match the chunk leaves behind, read before any repair --
        # this is what says whether the chunk did anything at all, independently
        # of how coarse the oracle's remaining-cost estimate turns out to be
        "match": env_successes(raw),
    }


def component_delta(before: dict, after: dict) -> dict:
    """Which goal components the chunk flipped, in which direction.

    Job 49557/49558 could not separate "the candidates are genuinely equivalent"
    from "``c_true`` is too coarse to see the difference", because nothing was
    recorded between *the chunk solved the task* and *the oracle finished it*.
    This is that missing quantity, and it needs no oracle.
    """
    keys = ("cube", "button_0", "button_1", "drawer", "window")
    gained = [k for k in keys if after[k] and not before[k]]
    lost = [k for k in keys if before[k] and not after[k]]
    return {
        "gained": gained,
        "lost": lost,
        "changed": bool(gained or lost),
        "net": len(gained) - len(lost),
        "matched_after": sum(bool(after[k]) for k in keys),
    }


def score_candidate(
    raw,
    snapshots,
    snapshot,
    actions_flat,
    *,
    config,
    blocks,
    repair,
    goal_state,
    goal_buttons,
    match_before,
) -> dict:
    """True remaining cost of one candidate, from a frozen state.

    ``c_true`` is the chunk's own steps plus what OGBench's controllers then need
    to finish the task.  Charging the chunk itself is what makes a candidate that
    wanders and a candidate that advances comparable.
    """
    snapshots.restore(snapshot)
    executed = execute_chunk(raw, actions_flat, config, blocks)
    delta = component_delta(match_before, executed["match"])
    if executed["solved"]:
        return {
            "true_cost": float(executed["steps"]),
            "censored": False,
            "chunk_steps": executed["steps"],
            "chunk_solved": True,
            "repair_steps": 0,
            "clipped_frac": executed["clipped_frac"],
            "component": delta,
        }
    repaired = repair.steps_to_goal(goal_state, goal_buttons)
    return {
        "true_cost": float(executed["steps"] + repaired["steps"]),
        "censored": bool(repaired["censored"]),
        "chunk_steps": executed["steps"],
        "chunk_solved": False,
        "repair_steps": int(repaired["steps"]),
        "clipped_frac": executed["clipped_frac"],
        "component": delta,
    }


def run_episode(
    *,
    raw,
    spec,
    cache,
    track,
    model,
    objective,
    tracker,
    settings: PlanSettings,
    config: LeWMConfig,
    device,
    encoder: FrameEncoder,
    camera: str,
    eval_budget: int,
    plan_seed: int,
    args,
    deltas,
) -> tuple[dict, list[ReplanRecord]]:
    """Mirror of ``scene_eval.run_episode`` with candidate instrumentation."""
    qpos, qvel, buttons = cache["qpos"], cache["qvel"], cache["button_states"]
    goal_state = state_row(track, spec.goal_row)
    goal_buttons = buttons[spec.goal_row]

    restore_state(raw, qpos[spec.goal_row], qvel[spec.goal_row], goal_buttons)
    goal_frame = encoder(np.asarray(raw.render(camera=camera), dtype=np.uint8))

    restore_state(raw, qpos[spec.start_row], qvel[spec.start_row], buttons[spec.start_row])
    set_goal_targets(raw, goal_state, goal_buttons)
    if env_successes(raw)["success"]:
        return {"spec": spec.to_json(), "trivial": True, "success": True}, []

    solver, _evaluator, _plan_config = make_solver(
        model, objective, settings, config, device, plan_seed
    )
    recorder = make_recorder(
        record_steps=(0, settings.cem_steps // 2, settings.cem_steps - 1),
        topk=settings.topk,
    )
    solver.callbacks.append(recorder)

    snapshots = SceneSnapshotManager(raw)
    repair = OracleRepair(raw, max_skills=args.max_repair_skills)
    draw_rng = np.random.default_rng(args.draw_seed + 7919 * spec.episode)

    window = HistoryWindow(config.history_size, config.action_input_dim, device)
    first_frame = encoder(np.asarray(raw.render(camera=camera), dtype=np.uint8))
    window.reset(first_frame)
    latent_of = _latent_fn(model) if tracker is not None else None
    if tracker is not None:
        tracker.reset(latent_of(first_frame))

    goal_tensor = goal_frame.unsqueeze(0).unsqueeze(0)
    blocks_per_plan = min(settings.receding_horizon, settings.horizon)
    steps = 0
    replans = 0
    instrumented = 0
    observed_blocks = 0
    success = False
    records: list[ReplanRecord] = []

    while steps < eval_budget and not success:
        info = {
            "pixels": window.pixels(),
            "goal": goal_tensor,
            "action_history": window.action_history(),
            "action": torch.zeros(
                1, config.history_size, config.action_input_dim, device=device
            ),
        }
        outputs = solver.solve(info)
        plan = outputs["actions"][0]
        this_replan = replans
        replans += 1

        instrument = (
            instrumented < args.max_instrumented_replans
            and this_replan % args.replan_stride == 0
            and len(recorder.frames) == 3
        )
        if instrument:
            instrumented += 1
            record = ReplanRecord(episode=int(spec.episode), replan=int(this_replan))
            snapshot = snapshots.capture()
            match_before = env_successes(raw)
            record.match_before = match_before

            union_true: list[float] = []
            for draw in draw_strata(
                recorder.frames,
                n_per_stratum=args.n_per_stratum,
                cem_steps=settings.cem_steps,
                rng=draw_rng,
            ):
                scored = [
                    score_candidate(
                        raw,
                        snapshots,
                        snapshot,
                        draw.actions[i],
                        config=config,
                        blocks=blocks_per_plan,
                        repair=repair,
                        goal_state=goal_state,
                        goal_buttons=goal_buttons,
                        match_before=match_before,
                    )
                    for i in range(draw.actions.shape[0])
                ]
                record.strata[draw.stratum] = {
                    "source_step": draw.source_step,
                    "model_cost": [float(c) for c in draw.model_cost],
                    "true_cost": [s["true_cost"] for s in scored],
                    "censored": [s["censored"] for s in scored],
                    "chunk_solved": [s["chunk_solved"] for s in scored],
                    "clipped_frac": [s["clipped_frac"] for s in scored],
                    "component": [s["component"] for s in scored],
                }
                union_true.extend(
                    s["true_cost"] for s in scored if not s["censored"]
                )

            # the plan the loop is about to commit, scored the same way
            selected = score_candidate(
                raw,
                snapshots,
                snapshot,
                plan.cpu().numpy(),
                config=config,
                blocks=blocks_per_plan,
                repair=repair,
                goal_state=goal_state,
                goal_buttons=goal_buttons,
                match_before=match_before,
            )
            record.selected = dict(selected)
            record.selected["regret"] = (
                None
                if selected["censored"] or not union_true
                else float(selected["true_cost"] - min(union_true))
            )

            # noise floor: the same candidate, re-executed from the same state
            if args.delta_probe_repeats > 1 and instrumented == 1:
                repeats = [
                    score_candidate(
                        raw,
                        snapshots,
                        snapshot,
                        plan.cpu().numpy(),
                        config=config,
                        blocks=blocks_per_plan,
                        repair=repair,
                        goal_state=goal_state,
                        goal_buttons=goal_buttons,
                        match_before=match_before,
                    )["true_cost"]
                    for _ in range(args.delta_probe_repeats)
                ]
                record.delta_probe = {
                    "true_costs": repeats,
                    "spread": float(max(repeats) - min(repeats)),
                }

            records.append(record)
            snapshots.restore(snapshot)

        for block_index in range(blocks_per_plan):
            block = plan[block_index]
            actions = block.reshape(config.action_block, config.action_dim).cpu().numpy()
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            for action in actions:
                raw.step(action)
                steps += 1
                if env_successes(raw)["success"]:
                    success = True
                    break
            frame = encoder(np.asarray(raw.render(camera=camera), dtype=np.uint8))
            window.push(frame, block.to(device))
            if tracker is not None:
                tracker.observe(latent_of(frame), block.to(device))
            observed_blocks += 1
            if success or steps >= eval_budget:
                break

    if tracker is not None:
        # The invariant this whole program rests on: the head integrated exactly the
        # transitions that were executed, and none of the candidate rollouts replayed
        # through the simulator for ground truth.  Counted from the blocks actually
        # pushed rather than derived from ``env_steps``, which is not a multiple of
        # the block length when success lands mid-block.
        expected = observed_blocks + 1
        if tracker.updates != expected:
            raise RuntimeError(
                f"progress tracker advanced {tracker.updates} times, expected {expected}"
            )

    return (
        {
            "spec": spec.to_json(),
            "trivial": False,
            "success": bool(env_successes(raw)["success"]),
            "env_steps": int(steps),
            "replans": int(replans),
            "instrumented_replans": int(instrumented),
            "progress_updates": (None if tracker is None else int(tracker.updates)),
        },
        records,
    )


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("the elite-optimism gate must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("the elite-optimism gate requires a GPU allocation")

    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
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
    arm_tag = progress_arm or "latent_l2"

    episodes = []
    records: list[ReplanRecord] = []
    started = time.time()
    for spec in shard:
        plan_seed = args.plan_seed + 1_000_003 * spec.episode + 10_007 * args.goal_offset
        summary, episode_records = run_episode(
            raw=raw,
            spec=spec,
            cache=cache,
            track=track,
            model=model,
            objective=objective,
            tracker=tracker,
            settings=settings,
            config=config,
            device=device,
            encoder=encoder,
            camera=meta["camera"],
            eval_budget=budget,
            plan_seed=plan_seed,
            args=args,
            deltas=deltas,
        )
        episodes.append(summary)
        records.extend(episode_records)
        print(
            f"episode {spec.episode} success={summary['success']} "
            f"instrumented={summary.get('instrumented_replans', 0)} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    probes = [r.delta_probe["spread"] for r in records if r.delta_probe is not None]
    result = {
        "protocol": PROTOCOL,
        "arm": arm_tag,
        "objective": args.objective,
        "goal_offset": args.goal_offset,
        "plan_seed": args.plan_seed,
        "settings": settings.to_json(),
        "instrument": {
            "n_per_stratum": args.n_per_stratum,
            "max_instrumented_replans": args.max_instrumented_replans,
            "replan_stride": args.replan_stride,
            "max_repair_skills": args.max_repair_skills,
            "deltas": deltas,
            "repeat_probe_spreads": probes,
            "repeat_probe_max_spread": max(probes) if probes else None,
        },
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "progress_checkpoint_sha256": (
            file_sha256(args.progress_checkpoint) if args.progress_checkpoint else None
        ),
        "render_info": render_info,
        "episodes": episodes,
        "replans": [r.to_json() for r in records],
        "summary": summarise(records, deltas, seed=args.plan_seed),
        "wall_seconds": time.time() - started,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{arm_tag}_off{args.goal_offset}_seed{args.plan_seed}_shard{args.shard_index}.json"
    (args.out_dir / name).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out_dir / name}", flush=True)

    for delta_key, per_stratum in result["summary"]["per_delta"].items():
        amp = per_stratum["elite_amplification"]
        print(
            f"{delta_key}: random_init={amp['random_init_rate']} "
            f"elite={amp['elite_rate']} amplification={amp['ratio']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
