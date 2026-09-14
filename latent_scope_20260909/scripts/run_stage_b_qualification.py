#!/usr/bin/env python3
"""Bounded causal-prefix pilot. No success-conditioned anchors or automatic Stage-C gate.

Snapshot pickles are trusted, locally generated experiment artifacts only. Never load
an externally supplied pickle. All simulation and result aggregation run under Slurm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import time
import traceback
import types
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import mujoco
from robosuite.utils.binding_utils import MjSim

from run_baseline_sim_client import InferenceClient, load_multistep_wrapper
from run_stage_a4_collect import task_history, wait_for_server
from run_stage_a4_calibrate import progress_label, state_max_abs
from run_stage_a_preflight import capture_snapshot, restore_snapshot, object_simple_state, restore_sim_dynamics_state
from run_stage_b_profile import (create_env, policy_action_chunk, set_policy_seed,
                                 capture_images, image_alignment_metrics, progress_score,
                                 reset_to, write_json)


class BudgetStop(RuntimeError):
    pass


def check_budget(deadline):
    if time.time() >= deadline:
        raise BudgetStop("Predeclared soft budget reached; partial results retained")


def snapshot(wrapped, env, fields):
    packet = {"snapshot_version": 2, "mujoco_version": mujoco.__version__,
              "compiled_model": pickle.dumps(env.sim.model._model, protocol=pickle.HIGHEST_PROTOCOL),
              "full_mjdata": pickle.dumps(env.sim.data._data, protocol=pickle.HIGHEST_PROTOCOL),
              "native": capture_snapshot(env, fields),
              "wrapper": {name: copy.deepcopy(getattr(wrapped, name))
                          for name in ("obs", "reward", "done", "max_episode_steps")},
              "info": copy.deepcopy(dict(wrapped.info)),
              "obs_cache": copy.deepcopy(env._obs_cache),
              "observables": {name: object_simple_state(obj) for name, obj in env._observables.items()},
              "robots": []}
    for robot in env.robots:
        cc = robot.composite_controller
        packet["robots"].append({
            "controller": object_simple_state(cc),
            "parts": {name: object_simple_state(obj) for name, obj in cc.part_controllers.items()},
            "grippers": {name: object_simple_state(obj) for name, obj in robot.gripper.items()},
        })
    return packet


def set_attrs(obj, attrs):
    for key, value in attrs.items():
        setattr(obj, key, copy.deepcopy(value))


def restore(wrapped, env, packet):
    restore_snapshot(env, packet["native"])
    # mj_forward may modify warm starts; put the integration inputs back afterwards.
    restore_sim_dynamics_state(env, packet["native"]["sim_dynamics_state"])
    if "full_mjdata" in packet:
        # Preserve solver/derived state too, without an extra mj_forward afterwards.
        # This installed Python binding supports native MjData pickle round trips
        # but does not expose the C API's mj_copyData. Keep the robosuite wrapper
        # (and its references) intact, replacing only its owned native data object.
        env.sim.data._data = pickle.loads(packet["full_mjdata"])
    for robot, saved in zip(env.robots, packet["robots"]):
        set_attrs(robot.composite_controller, saved["controller"])
        for name, attrs in saved["parts"].items():
            set_attrs(robot.composite_controller.part_controllers[name], attrs)
        for name, attrs in saved["grippers"].items():
            set_attrs(robot.gripper[name], attrs)
    env._obs_cache = copy.deepcopy(packet["obs_cache"])
    for name, attrs in packet["observables"].items():
        set_attrs(env._observables[name], attrs)
    for name, value in packet["wrapper"].items():
        setattr(wrapped, name, copy.deepcopy(value))
    wrapped.info = defaultdict(lambda: deque(maxlen=wrapped.max_steps_needed + 1),
                               copy.deepcopy(packet["info"]))
    return wrapped._get_obs(wrapped.video_delta_indices, wrapped.state_delta_indices)


def open_restored(wrapper_class, cfg, source):
    wrapped, gym_env, env, _ = create_env(wrapper_class, cfg["task"]["name"], cfg["split"], 8)
    try:
        wrapped.reset(seed=source["seed"])
        packet = source["packet"]
        if "compiled_model" in packet:
            if packet["mujoco_version"] != mujoco.__version__:
                raise RuntimeError("Compiled snapshots require the identical MuJoCo version")
            # Keep RoboCasa's episode metadata and controller/observable binding lifecycle,
            # but bypass lossy XML re-compilation for the actual physics model.
            original_initialize = env._initialize_sim
            def initialize_compiled(this, xml_string=None):
                this.sim = MjSim(pickle.loads(packet["compiled_model"]))
                this.sim.forward()
                this.initialize_time(this.control_freq)
            env.set_ep_meta(copy.deepcopy(source["ep_meta"]))
            env.reset()
            env._initialize_sim = types.MethodType(initialize_compiled, env)
            try:
                env.reset_from_xml_string(source["model_xml"])
            finally:
                env._initialize_sim = original_initialize
        else:
            # Read-only legacy diagnosis only; new collection always emits version 2.
            reset_to(env, {"model": source["model_xml"], "ep_meta": json.dumps(source["ep_meta"]),
                           "states": packet["native"]["sim_state"]})
        obs = restore(wrapped, env, source["packet"])
        return wrapped, gym_env, env, obs
    except Exception:
        wrapped.close()
        raise


def monitor(task, env, info):
    # Never call update_state or reward here. _check_success is audited separately.
    success = bool(info.get("success", [False])[-1])
    return progress_label(task, env, success)


def record(wrapped, env, obs, cfg, reward=False, done=False):
    info = dict(wrapped.info)
    return {"state": np.asarray(env.sim.get_state().flatten()).copy(),
            "history": task_history(env, cfg["task"]["history_fields"]),
            "label": monitor(cfg["task"]["name"], env, info),
            "reward": bool(reward), "done": bool(done),
            "native_step": len(wrapped.reward),
            "images": capture_images(obs, cfg["camera_keys"])}


def step_recorded(wrapped, env, actions, cfg, deadline):
    """Same native steps and action values as a chunk, retaining ordered observations."""
    previous = wrapped.n_action_steps
    wrapped.n_action_steps = 1
    rows, observations = [], []
    try:
        for t in range(next(iter(actions.values())).shape[0]):
            check_budget(deadline)
            obs, reward, done, truncated, _ = wrapped.step({k: v[t:t + 1] for k, v in actions.items()})
            rows.append(record(wrapped, env, obs, cfg, reward, done or truncated))
            observations.append(copy.deepcopy(obs))
            if done or truncated:
                break
    finally:
        wrapped.n_action_steps = previous
    return rows, observations


def compare(first, second, cfg, exact_images=False):
    if len(first) != len(second) or not first:
        return {"pass": False, "reason": "suffix length mismatch"}
    states, mae, p99 = 0., 0., 0.
    image_max = 0
    equal = True
    for a, b in zip(first, second):
        if not np.isfinite(a["state"]).all() or not np.isfinite(b["state"]).all():
            return {"pass": False, "reason": "non-finite simulator state"}
        states = max(states, state_max_abs(a["state"], b["state"]))
        equal = equal and all(a[k] == b[k] for k in ("history", "label", "reward", "done", "native_step"))
        if set(a["images"]) != set(b["images"]) or set(a["images"]) != set(cfg["camera_keys"]):
            return {"pass": False, "reason": "camera mismatch"}
        for key in a["images"]:
            metric = image_alignment_metrics(a["images"][key], b["images"][key])
            mae, p99 = max(mae, metric["normalized_mae"]), max(p99, metric["p99_abs"])
            image_max = max(image_max, metric["max_abs"])
    passed = equal and states <= cfg["state_tolerance"]
    passed = passed and (image_max == 0 if exact_images else
                         mae <= cfg["source_image_max_normalized_mae"] and p99 <= cfg["source_image_max_p99_abs"])
    return {"pass": bool(passed), "state_max_abs": states, "history_labels_rewards_clocks_equal": bool(equal),
            "image_max_abs": image_max, "image_normalized_mae": mae, "image_p99_abs": p99,
            "image_gate": "exact" if exact_images else "predeclared_tolerance"}


def collect_source(client, wrapper_class, cfg, attempt, deadline):
    seed = cfg["environment_seed_base"] + attempt
    random.seed(seed)
    np.random.seed(seed)
    wrapped, gym_env, env, horizon = create_env(wrapper_class, cfg["task"]["name"], cfg["split"], 8)
    try:
        obs, _ = wrapped.reset(seed=seed)
        set_policy_seed(client, cfg["source_policy_seed_base"] + attempt)
        history = deque([copy.deepcopy(obs)], maxlen=3)
        prefix_actions = []
        latest = {"success": [False]}
        found = False
        while len(wrapped.reward) <= int(horizon * cfg["anchor_deadline_fraction"]):
            check_budget(deadline)
            label = monitor(cfg["task"]["name"], env, latest)
            # First observed partial contact milestone. No future rollout is inspected.
            if (len(wrapped.reward) >= 16 and 1 <= label["contact_milestone"] < 5
                    and not label["released_success"] and progress_score(cfg["task"]["name"], label) < 1):
                found = True
                break
            actions = policy_action_chunk(client, obs, 8)
            obs, _, done, truncated, latest = wrapped.step(actions)
            history.append(copy.deepcopy(obs))
            prefix_actions.append(actions)
            if done or truncated:
                break
        label = monitor(cfg["task"]["name"], env, latest)
        attempt_row = {"attempt": attempt, "seed": seed, "anchor_found": found,
                       "native_step": len(wrapped.reward), "label": label,
                       "future_success_used_for_selection": False}
        if not found:
            return None, attempt_row
        # Metadata may consume RNG; collect it before the snapshot, exactly once.
        source = {"seed": seed, "model_xml": env.sim.model.get_xml(), "ep_meta": env.get_ep_meta(),
                  "horizon": horizon, "history_observations": list(history),
                  "history_cadence_steps": 8, "prefix_actions": prefix_actions}
        source["packet"] = snapshot(wrapped, env, cfg["task"]["history_fields"])
        source["anchor_record"] = record(wrapped, env, obs, cfg)
        before = task_history(env, cfg["task"]["history_fields"])
        success_checks = [bool(env._check_success()), bool(env._check_success())]
        if before != task_history(env, cfg["task"]["history_fields"]) or success_checks[0] != success_checks[1]:
            raise RuntimeError("Success check mutates task history or is not repeatable")
        # Reference must remain the untouched live carrier, not a same-carrier restore.
        # The success checks above are read-only and have already been audited.
        set_policy_seed(client, cfg["proposal_seed_base"] + attempt * 100)
        suffix = policy_action_chunk(client, obs, 16)
        source["validation_actions"] = suffix
        source["live_suffix"], _ = step_recorded(wrapped, env, suffix, cfg, deadline)
        return source, attempt_row
    finally:
        wrapped.close()


def qualify(wrapper_class, cfg, source, deadline):
    replays = []
    initial = []
    for _ in range(2):
        check_budget(deadline)
        wrapped, _, env, obs = open_restored(wrapper_class, cfg, source)
        try:
            initial.append(record(wrapped, env, obs, cfg))
            rows, _ = step_recorded(wrapped, env, source["validation_actions"], cfg, deadline)
            replays.append(rows)
        finally:
            wrapped.close()
    result = {"initial": compare([source["anchor_record"]], initial[:1], cfg, True),
              "source_restore_h8": compare(source["live_suffix"][:8], replays[0][:8], cfg),
              "source_restore_h16": compare(source["live_suffix"], replays[0], cfg),
              "restore_repeat_h16": compare(replays[0], replays[1], cfg, True)}
    result["pass"] = all(row["pass"] for row in result.values())
    return result


def run_branch(client, wrapper_class, cfg, source, actions, continuation_seed, deadline):
    check_budget(deadline)
    started = time.time()
    wrapped, _, env, obs = open_restored(wrapper_class, cfg, source)
    try:
        alignment = compare([source["anchor_record"]], [record(wrapped, env, obs, cfg)], cfg, True)
        if not alignment["pass"]:
            raise RuntimeError(f"Candidate initial alignment failed: {alignment}")
        intervention, observations = step_recorded(wrapped, env, actions, cfg, deadline)
        obs = observations[-1]
        done = intervention[-1]["done"]
        success = any(row["reward"] for row in intervention)
        trajectory = [{k: v for k, v in row.items() if k not in ("images", "state")} for row in intervention]
        set_policy_seed(client, continuation_seed)
        while not done:
            check_budget(deadline)
            chunk = policy_action_chunk(client, obs, 8)
            obs, reward, terminated, truncated, info = wrapped.step(chunk)
            done = bool(terminated or truncated)
            success = success or bool(reward)
            trajectory.append({"native_step": len(wrapped.reward),
                               "label": monitor(cfg["task"]["name"], env, info),
                               "reward": bool(reward), "done": done})
        return {"success": bool(success), "continuation_seed": continuation_seed,
                "initial_alignment": alignment, "elapsed_seconds": time.time() - started,
                "trajectory": trajectory, "final_native_step": len(wrapped.reward),
                "intervention_state": intervention[-1]["state"]}, observations
    finally:
        wrapped.close()


def save_pickle(path, value):
    with path.open("xb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_observations(path, source, observations, actions, post_state):
    frames = source["history_observations"] + observations
    arrays = {}
    for key in frames[0]:
        if isinstance(frames[0][key], np.ndarray):
            arrays["observation::" + key] = np.stack([row[key][-1] for row in frames])
    arrays.update({"action::" + key: value for key, value in actions.items()})
    arrays["decision_step"] = np.asarray(len(source["history_observations"]) - 1)
    arrays["history_cadence_steps"] = np.asarray(8)
    arrays["post_state"] = post_state
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--server-state", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--restore-proof", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Qualification requires a bounded GPU sbatch job")
    config = json.loads(args.config.read_text())
    cfg = config["qualification"]
    proof = json.loads(args.restore_proof.read_text())
    if proof.get("verdict") != "COMPILED_RESTORE_PASS" or not proof["full_snapshot_gate"]["pass"]:
        raise RuntimeError("A successful CPU compiled-snapshot proof is required before pilot restart")
    verified_source_path = Path(proof["verified_source"])
    if (cfg["candidate_count"], cfg["intervention_steps"], cfg["continuation_cadence_steps"],
            cfg["replay_suffix_steps"], cfg["confirmation_seeds"]) != (8, 8, 8, 16, 2):
        raise ValueError("This runner is locked to N8/H8/cadence8/replay16/two confirmation seeds")
    if cfg["task"]["name"] != "ScrubCuttingBoard" or cfg["stage_c_auto_start"]:
        raise ValueError("Scrub qualification only; Stage C cannot auto-start")
    if not 0 < cfg["scoring_seconds"] < cfg["total_soft_seconds"] < cfg["slurm_hard_seconds"] <= 28800:
        raise ValueError("Budget must be ordered and capped at 8 allocated GPU-hours")
    if cfg["previous_tranche_gpu_seconds"] + cfg["slurm_hard_seconds"] > cfg["tranche_gpu_second_cap"]:
        raise ValueError("Retry would exceed the total approved tranche GPU budget")
    start = float(os.environ["QUALIFICATION_STARTED_UNIX"])
    scoring_deadline = start + cfg["scoring_seconds"]
    final_deadline = start + cfg["total_soft_seconds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "qualification_result.json"
    result = {"verdict": "QUALIFICATION_RUNNING", "job_id": os.environ["SLURM_JOB_ID"],
              "config": config, "attempts": [], "prefixes": [], "confirmation": [],
              "stage_c_authorized": False, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    result["restore_proof"] = {"path": str(args.restore_proof), "job_id": proof["job_id"],
                               "verified_source": str(verified_source_path),
                               "sha256": hashlib.sha256(args.restore_proof.read_bytes()).hexdigest()}
    write_json(result_path, result)
    client = InferenceClient("localhost", args.port, timeout_ms=1000)
    completed = []
    try:
        wait_for_server(client, args.server_pid, args.server_state)
        wrapper_class = load_multistep_wrapper(Path(config["runtime_root"]) / "src" / "Isaac-GR00T")
        try:
            for attempt in range(cfg["maximum_attempts"]):
                check_budget(scoring_deadline)
                if attempt == 0:
                    # Continue the exact already-qualified source, not a lucky regenerated seed.
                    with verified_source_path.open("rb") as handle:
                        source = pickle.load(handle)
                    if source["seed"] != cfg["environment_seed_base"] or source["packet"].get("snapshot_version") != 2:
                        raise RuntimeError("Verified source does not match the locked seed/snapshot protocol")
                    attempt_row = {"attempt": 0, "seed": source["seed"], "anchor_found": True,
                                   "native_step": source["anchor_record"]["native_step"],
                                   "label": source["anchor_record"]["label"],
                                   "future_success_used_for_selection": False,
                                   "resumed_verified_source": str(verified_source_path)}
                else:
                    source, attempt_row = collect_source(client, wrapper_class, cfg, attempt, scoring_deadline)
                result["attempts"].append(attempt_row)
                write_json(result_path, result)
                if source is None:
                    continue
                prefix_dir = args.output_dir / f"prefix_attempt_{attempt:02d}"
                prefix_dir.mkdir(exist_ok=False)
                save_pickle(prefix_dir / "source_snapshot.pkl", source)
                calibration = qualify(wrapper_class, cfg, source, scoring_deadline)
                write_json(prefix_dir / "replay_gate.json", calibration)
                if not calibration["pass"]:
                    raise RuntimeError(f"REPLAY_GATE_FAIL attempt={attempt}: {calibration}")
                # Use the identical recorded anchor observation for every proposal.
                obs = source["history_observations"][-1]
                banks = []
                for index in range(8):
                    set_policy_seed(client, cfg["proposal_seed_base"] + attempt * 100 + index)
                    banks.append(policy_action_chunk(client, obs, 8))
                save_pickle(prefix_dir / "candidate_bank.pkl", banks)
                row = {"attempt": attempt, "seed": source["seed"], "directory": str(prefix_dir),
                       "replay_gate": calibration, "candidates": [], "complete": False}
                result["prefixes"].append(row)
                write_json(result_path, result)
                post_states = []
                for index, bank in enumerate(banks):
                    branch, observations = run_branch(client, wrapper_class, cfg, source, bank,
                                                     cfg["scoring_seed_base"] + attempt, scoring_deadline)
                    branch["candidate_index"] = index
                    post_states.append(branch.pop("intervention_state"))
                    save_observations(prefix_dir / f"candidate_{index}_observations.npz", source, observations, bank, post_states[-1])
                    row["candidates"].append(branch)
                    write_json(result_path, result)
                    print(f"attempt={attempt} candidate={index} success={branch['success']}", flush=True)
                row["complete"] = True
                labels = [candidate["success"] for candidate in row["candidates"]]
                row["baseline_success"] = labels[0]
                row["oracle_success"] = any(labels)
                row["recoverable"] = not labels[0] and any(labels)
                row["outcome_varies"] = len(set(labels)) > 1
                row["candidate_state_diversity"] = max(state_max_abs(a, b) for a in post_states for b in post_states)
                # Success-only oracle with smallest-index tie-break, frozen before confirmation.
                row["selected_candidate"] = next((i for i, value in enumerate(labels) if value), 0)
                # Binary models can be large: retain paths, not all source snapshots in RAM.
                completed.append((row, prefix_dir / "source_snapshot.pkl", banks))
                write_json(result_path, result)
                if len(completed) >= cfg["target_prefixes"]:
                    break
        except BudgetStop as error:
            result["scoring_stop"] = str(error)
        result["selection_locked"] = True
        write_json(result_path, result)
        # Confirm observed recoveries first. Null panels still get one preselected random repeat.
        confirmation_panel = [(r, s, b) for r, s, b in completed if r["recoverable"]]
        if not confirmation_panel and completed:
            confirmation_panel = completed[:1]
        for row, source_path, banks in confirmation_panel:
            with source_path.open("rb") as handle:
                source = pickle.load(handle)
            selected = row["selected_candidate"] if row["recoverable"] else 1
            for repeat in range(2):
                seed = cfg["confirmation_seed_base"] + 10 * row["attempt"] + repeat
                pair = {"attempt": row["attempt"], "seed": seed, "selected_candidate": selected,
                        "purpose": "selected_recovery" if row["recoverable"] else "preselected_null_repeat",
                        "branches": []}
                result["confirmation"].append(pair)
                write_json(result_path, result)
                for index in (0, selected):
                    branch, _ = run_branch(client, wrapper_class, cfg, source, banks[index], seed, final_deadline)
                    branch.pop("intervention_state")
                    branch["candidate_index"] = index
                    pair["branches"].append(branch)
                    write_json(result_path, result)
        result["verdict"] = "QUALIFICATION_PILOT_COMPLETE" if len(completed) == cfg["target_prefixes"] else "QUALIFICATION_PILOT_INCOMPLETE"
    except BudgetStop as error:
        result["verdict"] = "QUALIFICATION_BUDGET_STOP"
        result["budget_stop"] = str(error)
    except Exception as error:
        result["verdict"] = "QUALIFICATION_ERROR"
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds_including_server_startup"] = time.time() - start
        rows = [r for r in result["prefixes"] if r["complete"]]
        result["summary"] = {"complete_prefixes": len(rows), "attempted_sources": len(result["attempts"]),
                             "recoverable_prefixes": sum(r["recoverable"] for r in rows),
                             "oracle_gain_percentage_points": 100 * sum(r["recoverable"] for r in rows) / len(rows) if rows else None,
                             "interpretation": "Exploratory conditional pilot, not a Stage-C gate or deployment estimate"}
        confirmed = [p for p in result["confirmation"] if len(p["branches"]) == 2 and p["purpose"] == "selected_recovery"]
        result["summary"]["complete_confirmation_pairs"] = len(confirmed)
        result["summary"]["conditional_confirmation_gain_percentage_points"] = (
            100 * sum(int(p["branches"][1]["success"]) - int(p["branches"][0]["success"]) for p in confirmed) / len(confirmed)
            if confirmed else None)
        write_json(result_path, result)
        client.close()
        print(result["verdict"], flush=True)


if __name__ == "__main__":
    main()
