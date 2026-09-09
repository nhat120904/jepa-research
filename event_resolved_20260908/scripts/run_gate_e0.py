#!/usr/bin/env python3
"""Run the bounded ER-WM Gate E0 screen on a Slurm compute node."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import mani_skill
import mani_skill.envs  # noqa: F401

import strike_slide_env  # noqa: F401


@dataclass(frozen=True)
class StrikeParams:
    impact_x: float = 0.0
    standoff: float = 0.105
    strike_gain: float = 0.75


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


def vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1, np.asarray(value).shape[-1])[0]


def clone_tree(tree: Any) -> Any:
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone()
    if isinstance(tree, dict):
        return {key: clone_tree(value) for key, value in tree.items()}
    return copy.deepcopy(tree)


def make_env(seed: int, physics_hz: int):
    env = gym.make(
        "ERStrikeSlide-v0",
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_ee_target_delta_pos",
        render_mode=None,
        sim_backend="cpu",
        physics_hz=physics_hz,
    )
    env.reset(seed=seed)
    return env


def normalized_delta(delta_xyz: np.ndarray, max_norm: float = 0.9) -> np.ndarray:
    # ManiSkill EE delta controllers have normalized action spaces. We use only
    # direction plus a bounded magnitude and let feedback close small pose errors.
    delta = np.asarray(delta_xyz, dtype=np.float64)
    norm = np.linalg.norm(delta)
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (delta / norm * min(max_norm, norm * 12.0)).astype(np.float32)


def contact_force(unwrapped) -> float:
    forces = []
    for name in ("panda_leftfinger", "panda_rightfinger"):
        link = unwrapped.agent.robot.links_map[name]
        force = unwrapped.scene.get_pairwise_contact_forces(unwrapped.puck, link)
        forces.append(float(torch.linalg.norm(force, dim=-1).detach().cpu()[0]))
    return max(forces)


def policy_action(unwrapped, params: StrikeParams, control_index: int) -> np.ndarray:
    tcp = vector(unwrapped.agent.tcp.pose.p)[:3]
    puck = vector(unwrapped.puck.pose.p)[:3]
    goal = vector(unwrapped.goal_region.pose.p)[:3]
    direction = goal[:2] - puck[:2]
    direction /= max(np.linalg.norm(direction), 1e-9)
    lateral = np.array([-direction[1], direction[0]])
    hit_xy = puck[:2] - direction * params.standoff + lateral * params.impact_x
    hit_z = float(unwrapped.puck_half_sizes[2] + 0.035)

    if control_index < 16:
        target = np.array([hit_xy[0], hit_xy[1], hit_z + 0.08])
        max_norm = 0.70
    elif control_index < 24:
        target = np.array([hit_xy[0], hit_xy[1], hit_z])
        max_norm = 0.55
    elif control_index < 31:
        progress = (control_index - 23) / 7.0
        target_xy = hit_xy + direction * (params.standoff + 0.15 * params.strike_gain) * progress
        target = np.array([target_xy[0], target_xy[1], hit_z])
        max_norm = params.strike_gain
    else:
        target = tcp
        target[2] = max(target[2], hit_z + 0.10)
        max_norm = 0.35

    arm = normalized_delta(target - tcp, max_norm=max_norm)
    # Closed gripper makes a compact striking surface.
    return np.concatenate([arm, np.array([-1.0], dtype=np.float32)])


def run_rollout(seed: int, physics_hz: int, params: StrikeParams) -> dict[str, Any]:
    env = make_env(seed, physics_hz)
    unwrapped = env.unwrapped
    substeps = physics_hz // 20
    horizon_steps = int(round(4.0 * physics_hz))
    force_threshold = 0.5
    contacts: list[dict[str, Any]] = []
    in_contact = False
    last_distance = None
    last_action = np.array([0, 0, 0, -1], dtype=np.float32)
    success_any = False
    onset_approach_speeds: list[float] = []
    initial_state = clone_tree(unwrapped.get_state_dict())
    controller_index = -1
    terminal_info = None

    for step in range(horizon_steps):
        if step % substeps == 0:
            controller_index += 1
            last_action = policy_action(unwrapped, params, controller_index)
            action = last_action
        else:
            action = np.array([0, 0, 0, last_action[-1]], dtype=np.float32)

        _, _, _, _, info = env.step(action)
        terminal_info = info
        tcp = vector(unwrapped.agent.tcp.pose.p)[:3]
        puck = vector(unwrapped.puck.pose.p)[:3]
        distance = float(np.linalg.norm(tcp - puck))
        force = contact_force(unwrapped)
        now_contact = force > force_threshold
        event_time = (step + 1) / physics_hz
        if now_contact and not in_contact:
            approach = 0.0 if last_distance is None else max(0.0, (last_distance - distance) * physics_hz)
            onset_approach_speeds.append(approach)
            contacts.append({"kind": "onset", "time_s": event_time, "force_n": force, "approach_speed_mps": approach})
        elif in_contact and not now_contact:
            contacts.append({"kind": "release", "time_s": event_time, "force_n": force})
        in_contact = now_contact
        last_distance = distance
        success_any = success_any or bool(scalar(info["success"]))

    if in_contact:
        contacts.append({"kind": "open_at_horizon", "time_s": 4.0})
    puck_pos = vector(unwrapped.puck.pose.p)[:3]
    puck_vel = vector(unwrapped.puck.linear_velocity)[:3]
    goal_pos = vector(unwrapped.goal_region.pose.p)[:3]
    terminal_distance = float(np.linalg.norm(puck_pos[:2] - goal_pos[:2]))
    result = {
        "seed": seed,
        "physics_hz": physics_hz,
        "params": asdict(params),
        "events": contacts,
        "event_signature": [event["kind"] for event in contacts],
        "onset_times_s": [event["time_s"] for event in contacts if event["kind"] == "onset"],
        "release_times_s": [event["time_s"] for event in contacts if event["kind"] == "release"],
        "onset_approach_speeds_mps": onset_approach_speeds,
        "terminal_puck_pos": puck_pos.tolist(),
        "terminal_puck_vel": puck_vel.tolist(),
        "goal_pos": goal_pos.tolist(),
        "terminal_distance_m": terminal_distance,
        "success": success_any,
        "initial_state": initial_state,
    }
    env.close()
    return result


def stripped(result: dict[str, Any]) -> dict[str, Any]:
    output = dict(result)
    output.pop("initial_state", None)
    return output


def sample_params(rng: np.random.Generator, count: int) -> list[StrikeParams]:
    params = []
    for _ in range(count):
        params.append(
            StrikeParams(
                impact_x=float(rng.uniform(-0.075, 0.075)),
                standoff=float(rng.uniform(0.075, 0.135)),
                strike_gain=float(rng.uniform(0.40, 1.00)),
            )
        )
    return params


def objective(result: dict[str, Any]) -> float:
    speed = float(np.linalg.norm(result["terminal_puck_vel"][:2]))
    score = (result["terminal_distance_m"] / 0.05) ** 2 + 0.1 * (speed / 0.1) ** 2
    if abs(result["terminal_puck_pos"][0]) > 0.45 or not (-0.95 < result["terminal_puck_pos"][1] < 0.8):
        score += 100.0
    if result["success"]:
        score -= 100.0
    return score


def first_onset(result: dict[str, Any]) -> float | None:
    values = result["onset_times_s"]
    return None if not values else float(values[0])


def paired_mechanism(seed: int, physics_hz: int) -> list[dict[str, Any]]:
    variants = {
        "nominal": StrikeParams(),
        "impact_x_minus": StrikeParams(impact_x=-0.02),
        "impact_x_plus": StrikeParams(impact_x=0.02),
        "gain_minus": StrikeParams(strike_gain=0.67),
        "gain_plus": StrikeParams(strike_gain=0.83),
    }
    rows = []
    for label, params in variants.items():
        row = stripped(run_rollout(seed, physics_hz, params))
        row["variant"] = label
        rows.append(row)
    return rows


def median_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def summarize(
    fixed_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fixed_success = np.mean([row["success"] for row in fixed_rows])
    oracle_success = np.mean([row["success"] for row in oracle_rows])

    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in mechanism_rows:
        by_seed.setdefault(row["seed"], []).append(row)
    matched_roots = 0
    transversal = []
    timing_shifts = []
    for rows in by_seed.values():
        rows_by_variant = {row["variant"]: row for row in rows}
        signatures = [tuple(row["event_signature"]) for row in rows]
        matched = len(rows) == 5 and len(set(signatures)) == 1 and signatures[0] == ("onset", "release")
        if not matched:
            continue
        matched_roots += 1
        transversal.extend(
            speed > 0.05
            for row in rows
            for speed in row["onset_approach_speeds_mps"][:1]
        )
        for low, high in (("impact_x_minus", "impact_x_plus"), ("gain_minus", "gain_plus")):
            left, right = first_onset(rows_by_variant[low]), first_onset(rows_by_variant[high])
            if left is not None and right is not None and not math.isclose(left, right):
                timing_shifts.append(abs(right - left))

    freq_by_seed: dict[int, dict[int, dict[str, Any]]] = {}
    for row in frequency_rows:
        freq_by_seed.setdefault(row["seed"], {})[row["physics_hz"]] = row
    onset_disagreements: list[float] = []
    endpoint_disagreements: list[float] = []
    for cells in freq_by_seed.values():
        if 500 not in cells:
            continue
        ref = cells[500]
        ref_onset = first_onset(ref)
        for hz in (250, 1000):
            if hz not in cells:
                continue
            onset = first_onset(cells[hz])
            if ref_onset is not None and onset is not None:
                onset_disagreements.append(abs(ref_onset - onset))
            endpoint_disagreements.append(
                float(
                    np.linalg.norm(
                        np.asarray(ref["terminal_puck_pos"][:2])
                        - np.asarray(cells[hz]["terminal_puck_pos"][:2])
                    )
                )
            )

    replay_endpoint_errors = [row["endpoint_error_m"] for row in replay_rows]
    replay_success_mismatches = sum(row["success_mismatch"] for row in replay_rows)
    summary = {
        "fixed_success_rate": float(fixed_success),
        "oracle_success_rate": float(oracle_success),
        "oracle_headroom_points": float(100 * (oracle_success - fixed_success)),
        "matched_sequence_root_rate": matched_roots / max(1, len(by_seed)),
        "transversal_onset_rate": float(np.mean(transversal)) if transversal else None,
        "median_nonzero_onset_shift_s": median_or_none(timing_shifts),
        "median_cross_frequency_onset_disagreement_s": median_or_none(onset_disagreements),
        "median_cross_frequency_endpoint_disagreement_m": median_or_none(endpoint_disagreements),
        "max_replay_endpoint_error_m": max(replay_endpoint_errors, default=None),
        "replay_success_mismatches": int(replay_success_mismatches),
    }

    failures = []
    if oracle_success < 0.70:
        failures.append("oracle_search_below_70pct")
    if not (oracle_success - fixed_success >= 0.10 or (fixed_success < 0.70 <= oracle_success)):
        failures.append("insufficient_headroom_over_fixed_script")
    if summary["matched_sequence_root_rate"] < 0.60:
        failures.append("matched_event_coverage_below_60pct")
    if summary["transversal_onset_rate"] is None or summary["transversal_onset_rate"] < 0.50:
        failures.append("transversal_event_rate_below_50pct")
    if summary["median_nonzero_onset_shift_s"] is None or summary["median_nonzero_onset_shift_s"] < 0.004:
        failures.append("onset_shift_below_4ms")
    if summary["median_cross_frequency_onset_disagreement_s"] is None or summary["median_cross_frequency_onset_disagreement_s"] > 0.008:
        failures.append("cross_frequency_onset_disagreement_above_8ms")
    if summary["median_cross_frequency_endpoint_disagreement_m"] is None or summary["median_cross_frequency_endpoint_disagreement_m"] > 0.02:
        failures.append("cross_frequency_endpoint_disagreement_above_2cm")
    if summary["max_replay_endpoint_error_m"] is None or summary["max_replay_endpoint_error_m"] > 0.001 or replay_success_mismatches:
        failures.append("replay_determinism_failed")

    if not failures:
        decision = "E0_PASS_OPEN_E1"
    elif any("oracle" in item or "headroom" in item for item in failures):
        decision = "E0_STOP_TASK_OR_PLANNER"
    elif any("frequency" in item or "replay" in item for item in failures):
        decision = "E0_STOP_NUMERICS"
    else:
        decision = "E0_STOP_EVENT_SIGNAL"
    summary["decision"] = decision
    summary["failure_reasons"] = failures
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Gate E0 simulation must run in a Slurm job")
    args = parse_args()
    started = time.time()
    seeds = list(range(61000, 61000 + args.seeds))
    rng = np.random.default_rng(20260908)

    fixed_rows = [stripped(run_rollout(seed, 500, StrikeParams())) for seed in seeds]
    oracle_rows = []
    oracle_candidates = []
    for seed in seeds:
        candidates = [StrikeParams()] + sample_params(rng, args.candidates - 1)
        evaluated = [stripped(run_rollout(seed, 500, params)) for params in candidates]
        scores = [objective(row) for row in evaluated]
        best = int(np.argmin(scores))
        chosen = dict(evaluated[best])
        chosen["selected_index"] = best
        chosen["selected_objective"] = scores[best]
        oracle_rows.append(chosen)
        oracle_candidates.append(
            {"seed": seed, "objectives": scores, "selected_index": best}
        )

    mechanism_rows = [
        row for seed in seeds for row in paired_mechanism(seed, physics_hz=500)
    ]
    frequency_rows = [
        stripped(run_rollout(seed, hz, StrikeParams()))
        for seed in seeds[: min(4, len(seeds))]
        for hz in (250, 500, 1000)
    ]

    # Repeat from the same seed in fresh environments. This checks the whole reset,
    # controller, and physics path rather than only copying rigid-body tensors.
    replay_rows = []
    for seed in seeds[: min(4, len(seeds))]:
        first = stripped(run_rollout(seed, 500, StrikeParams()))
        second = stripped(run_rollout(seed, 500, StrikeParams()))
        replay_rows.append(
            {
                "seed": seed,
                "endpoint_error_m": float(
                    np.linalg.norm(
                        np.asarray(first["terminal_puck_pos"])
                        - np.asarray(second["terminal_puck_pos"])
                    )
                ),
                "success_mismatch": bool(first["success"] != second["success"]),
                "events_equal": first["events"] == second["events"],
            }
        )

    summary = summarize(
        fixed_rows, oracle_rows, mechanism_rows, frequency_rows, replay_rows
    )
    artifact = {
        "schema_version": 1,
        "gate": "ER_WM_E0",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "provenance": {
            "python": platform.python_version(),
            "mani_skill": getattr(mani_skill, "__version__", "unknown"),
            "torch": torch.__version__,
            "host": platform.node(),
            "seeds": seeds,
            "candidate_count_per_seed": args.candidates,
            "elapsed_seconds": time.time() - started,
        },
        "summary": summary,
        "fixed_script": fixed_rows,
        "oracle_search": oracle_rows,
        "oracle_candidate_scores": oracle_candidates,
        "mechanism": mechanism_rows,
        "frequency": frequency_rows,
        "replay": replay_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

