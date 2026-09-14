#!/usr/bin/env python3
"""Verify untouched live versus binary snapshot restore using saved actions, CPU only."""
import argparse
import copy
import json
import os
import pickle
import random
import time
from pathlib import Path
import numpy as np
from run_stage_b_qualification import (create_env, load_multistep_wrapper, snapshot, record,
                                      compare, open_restored, step_recorded, qualify, write_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Compiled-restore tests require CPU sbatch")
    with args.source.open("rb") as handle:
        old = pickle.load(handle)
    config = json.loads(args.config.read_text())
    cfg = config["qualification"]
    wrapper = load_multistep_wrapper(Path(config["runtime_root"]) / "src/Isaac-GR00T")
    result = {"job_id": os.environ["SLURM_JOB_ID"], "verdict": "COMPILED_RESTORE_RUNNING"}
    deadline = time.time() + 1100
    random.seed(old["seed"])
    np.random.seed(old["seed"])
    live, _, env, _ = create_env(wrapper, cfg["task"]["name"], cfg["split"], 8)
    try:
        obs, _ = live.reset(seed=old["seed"])
        for bank in old["prefix_actions"]:
            obs, _, _, _, _ = live.step(bank)
        result["saved_source_anchor"] = compare([old["anchor_record"]], [record(live, env, obs, cfg)], cfg, True)
        if not result["saved_source_anchor"]["pass"]:
            raise RuntimeError("Cannot recreate the exact saved source anchor")
        source = copy.deepcopy(old)
        source["packet"] = snapshot(live, env, cfg["task"]["history_fields"])
        source["reference_provenance"] = "untouched_live_carrier_replayed_saved_actions"
        source["live_suffix"], _ = step_recorded(live, env, source["validation_actions"], cfg, deadline)
        result["old_reference_vs_untouched_live"] = compare(old["live_suffix"], source["live_suffix"], cfg)
        captured_path = args.output.with_suffix(".captured_source.pkl")
        with captured_path.open("xb") as handle:
            pickle.dump(source, handle, protocol=pickle.HIGHEST_PROTOCOL)
        result["captured_source"] = str(captured_path)
        write_json(args.output, result)
        result["full_snapshot_gate"] = qualify(wrapper, cfg, source, deadline)
        model_only = dict(source, packet=dict(source["packet"]))
        model_only["packet"].pop("full_mjdata")
        other, _, restored_env, _ = open_restored(wrapper, cfg, model_only)
        try:
            rows, _ = step_recorded(other, restored_env, source["validation_actions"], cfg, deadline)
            result["compiled_model_only"] = compare(source["live_suffix"], rows, cfg)
        finally:
            other.close()
        if not result["full_snapshot_gate"]["pass"]:
            raise RuntimeError("Compiled full snapshot did not pass the unchanged replay gate")
        source_path = args.output.with_suffix(".verified_source.pkl")
        with source_path.open("xb") as handle:
            pickle.dump(source, handle, protocol=pickle.HIGHEST_PROTOCOL)
        result["verified_source"] = str(source_path)
        result["verdict"] = "COMPILED_RESTORE_PASS"
    except Exception as error:
        result["verdict"] = "COMPILED_RESTORE_FAIL"
        result["error"] = repr(error)
        raise
    finally:
        live.close()
        write_json(args.output, result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
