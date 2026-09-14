#!/usr/bin/env python3
"""CPU-only saved-action replay and model-array audit; no policy server."""
import argparse
import copy
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
from run_stage_b_qualification import (create_env, load_multistep_wrapper, record, compare,
                                      open_restored, restore, step_recorded, write_json)


def model_arrays(env):
    model = env.sim.model._model
    return {name: getattr(model, name).copy() for name in dir(model)
            if isinstance(getattr(model, name), np.ndarray)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run saved-action diagnostics in CPU sbatch")
    with args.source.open("rb") as handle:
        source = pickle.load(handle)
    config = json.loads(args.config.read_text())
    cfg = config["qualification"]
    wrapper = load_multistep_wrapper(Path(config["runtime_root"]) / "src/Isaac-GR00T")
    result = {"job": os.environ["SLURM_JOB_ID"], "source": str(args.source)}
    deadline = time.time() + 1000
    random.seed(source["seed"])
    np.random.seed(source["seed"])
    live, _, env, _ = create_env(wrapper, cfg["task"]["name"], cfg["split"], 8)
    try:
        obs, _ = live.reset(seed=source["seed"])
        for bank in source["prefix_actions"]:
            obs, _, _, _, _ = live.step(bank)
        result["recreated_source_anchor"] = compare([source["anchor_record"]], [record(live, env, obs, cfg)], cfg)
        # Reapply the recorded packet: isolate model/carrier differences, not source RNG.
        restore(live, env, source["packet"])
        source_model = model_arrays(env)
        live_rows, _ = step_recorded(live, env, source["validation_actions"], cfg, deadline)
        result["recreated_source_suffix"] = compare(source["live_suffix"], live_rows, cfg)
        other, _, fresh, _ = open_restored(wrapper, cfg, source)
        try:
            fresh_model = model_arrays(fresh)
            differences = {}
            for name in source_model.keys() & fresh_model.keys():
                a, b = source_model[name], fresh_model[name]
                if a.shape != b.shape:
                    differences[name] = {"shape": [list(a.shape), list(b.shape)]}
                elif not np.array_equal(a, b):
                    differences[name] = {"max_abs": float(np.max(np.abs(a.astype(float) - b.astype(float)))),
                                         "different_elements": int(np.count_nonzero(a != b))}
            result["model_array_differences"] = differences
            rows, _ = step_recorded(other, fresh, source["validation_actions"], cfg, deadline)
            result["uncorrected_suffix"] = compare(source["live_suffix"], rows, cfg)
            # A controlled intervention: transfer every shape-compatible numerical model array.
            copied, failed = [], {}
            for name, array in source_model.items():
                try:
                    current = getattr(fresh.sim.model._model, name)
                    if current.shape != array.shape:
                        raise ValueError("shape mismatch")
                    current[...] = array
                    copied.append(name)
                except Exception as error:
                    failed[name] = repr(error)
            restore(other, fresh, source["packet"])
            fixed, _ = step_recorded(other, fresh, source["validation_actions"], cfg, deadline)
            result["model_array_transfer_suffix"] = compare(source["live_suffix"], fixed, cfg)
            result["model_array_transfer_failures"] = failed
            result["copied_model_arrays"] = copied
            if result["recreated_source_suffix"]["pass"] and result["model_array_transfer_suffix"]["pass"]:
                repaired = copy.deepcopy(source)
                repaired["packet"]["model_arrays"] = source_model
                path = args.output.with_suffix(".repaired_source.pkl")
                with path.open("xb") as handle:
                    pickle.dump(repaired, handle, protocol=pickle.HIGHEST_PROTOCOL)
                result["repaired_source"] = str(path)
        finally:
            other.close()
    finally:
        live.close()
        write_json(args.output, result)
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
