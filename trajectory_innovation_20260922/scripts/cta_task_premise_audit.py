#!/usr/bin/env python3
"""CPU-only native-task semantics and existing intervention audit; never trains.

The mock predicate traces test the native evaluator, not physical reachability,
visual observability, candidate availability, or a learned CTA representation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_intervention(branch, anchor_step, horizon):
    """Require every committed native step; never substitute continuation labels."""
    rows = [r for r in branch["trajectory"]
            if anchor_step < r["native_step"] <= anchor_step + horizon]
    if [r["native_step"] for r in rows] != list(range(anchor_step + 1, anchor_step + horizon + 1)):
        raise ValueError("Missing, duplicated, or unordered intervention steps")
    if not branch["initial_alignment"]["pass"]:
        raise ValueError("Candidate initial alignment failed")
    return rows


def audit_saved_banks(path):
    data = json.loads(path.read_text())
    cfg = data["config"]["qualification"]
    if cfg["task"]["name"] != "ScrubCuttingBoard":
        raise ValueError("Only the explicitly specified native Scrub labels are supported")
    if cfg["restore_mode"] != "compiled_mujoco_model_and_full_mjdata_same_runtime_version":
        raise ValueError("Require the verified compiled-model restore protocol")
    horizon, count = cfg["intervention_steps"], cfg["candidate_count"]
    anchors = {a["attempt"]: a for a in data["attempts"] if a["anchor_found"]}
    metrics = ("contact_count", "contact_milestone", "sweep_range", "sweep_milestone", "released_success")
    included, excluded = [], []
    for prefix in data["prefixes"]:
        attempt = prefix["attempt"]
        if not prefix["complete"] or len(prefix["candidates"]) != count:
            excluded.append({"attempt": attempt, "reason": "incomplete bank", "candidates": len(prefix["candidates"])})
            continue
        if not prefix["replay_gate"]["pass"]:
            raise ValueError(f"Replay gate failed: {path}, attempt {attempt}")
        anchor = anchors[attempt]
        if anchor["future_success_used_for_selection"]:
            raise ValueError("Future-selected anchor")
        bank = sorted(prefix["candidates"], key=lambda c: c["candidate_index"])
        if [c["candidate_index"] for c in bank] != list(range(count)):
            raise ValueError("Candidate IDs not complete and unique")
        if len({c["continuation_seed"] for c in bank}) != 1:
            raise ValueError("Expected matched continuation seed within bank")
        short = [extract_intervention(c, anchor["native_step"], horizon) for c in bank]
        result = {"attempt": attempt, "environment_seed": prefix["seed"],
                  "anchor_step": anchor["native_step"], "anchor_label": anchor["label"],
                  "horizon": horizon, "metrics": {}}
        for key in metrics:
            before = float(anchor["label"][key])
            values = [float(rows[-1]["label"][key]) for rows in short]
            # Native counts/latched flags and accumulated sweep span must not decrease.
            if min(values) < before - 1e-9:
                raise ValueError(f"Native accumulated label decreased: {key}")
            result["metrics"][key] = {
                "candidate_end_values": values,
                "candidate_increments": [v - before for v in values],
                "spread": max(values) - min(values),
                "oracle_over_candidate0": max(values) - values[0],
                "all_flat": max(values) - min(values) <= 1e-9,
            }
        result["continuation_diagnostic"] = {
            "candidate_terminal_success": [bool(c["success"]) for c in bank],
            "tail_steps": [c["final_native_step"] - anchor["native_step"] - horizon for c in bank],
            "scope": "One matched seed, descriptive only; not a reliable per-chunk preference label",
        }
        included.append(result)
    summary = {}
    for key in metrics:
        values = [p["metrics"][key] for p in included]
        summary[key] = {
            "banks": len(values),
            "informative_banks": sum(not v["all_flat"] for v in values),
            "banks_with_gain_over_candidate0": sum(v["oracle_over_candidate0"] > 1e-9 for v in values),
            "mean_oracle_gain": sum(v["oracle_over_candidate0"] for v in values) / len(values) if values else None,
        }
    return {"path": str(path), "sha256": digest(path), "horizon": horizon,
            "summary": summary, "prefixes": included, "excluded": excluded,
            "limitations": ["Selected partial-contact development prefixes, not all task phases",
                            "Native history labels are privileged; no visual-readability test",
                            "Short native progress is not terminal value or guaranteed control benefit",
                            "No endpoint observation equivalence or trajectory-code advantage tested",
                            "Incomplete prefixes remain reported; no independent test-set claim"]}


def rinse_semantics(source):
    """Execute three unmodified methods extracted from the pinned local task."""
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RinseBowls")
    names = {"_setup_scene", "update_state", "_check_success"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in cls.body} != names:
        raise ValueError("Native task API changed")

    class Kitchen:
        def _setup_scene(self):
            pass

        def update_state(self):
            pass

    namespace = {"Kitchen": Kitchen, "OU": SimpleNamespace(gripper_obj_far=lambda *a, **kw: True)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), namespace)

    def run(trace, initial_timer=0):
        env = namespace["RinseBowls"]()
        env._setup_scene()
        env.obj_body_id = {"dish0": 0, "dish1": 1}
        env.water_contact_timers["dish0"] = initial_timer
        # Identical shared context: the other bowl has already been rinsed.
        env.rinsed_objects["dish1"] = True
        env.water_contact_timers["dish1"] = 25
        for under_water in trace:
            env.sink = SimpleNamespace(check_obj_under_water=lambda e, obj: bool(under_water) if obj == "dish0" else False)
            env.update_state()
        first = env._check_success()
        second = env._check_success()
        if first != second:
            raise ValueError("Native success check unexpectedly non-idempotent")
        return {"success": bool(first), "rinsed": dict(env.rinsed_objects),
                "timers": dict(env.water_contact_timers), "total_under_water": sum(trace),
                "final_under_water": bool(trace[-1]), "steps": len(trace)}

    pairs = []
    for name, before, a, b in [
        ("same_duration_and_endpoint_predicate", 0, [1]*25+[0]*7, [1]*13+[0]*3+[1]*12+[0]*4),
        ("h8_near_native_threshold", 23, [1]*2+[0]*6, [1, 0, 1]+[0]*5),
    ]:
        left, right = run(a, before), run(b, before)
        assert left["success"] and not right["success"]
        assert left["total_under_water"] == right["total_under_water"]
        assert left["final_under_water"] == right["final_under_water"]
        pairs.append({"name": name, "initial_dish0_timer": before,
                      "trace_a": a, "trace_b": b, "result_a": left, "result_b": right})
    return {"source": str(source), "sha256": digest(source), "native_predicate_cases": pairs,
            "status": "NATIVE_SEMANTICS_ONLY",
            "not_established": ["Physical realizability of these traces", "Equal endpoint images or states",
                                "Readability from permitted cameras and proprioception",
                                "Availability in a policy candidate bank", "Training or control advantage"]}


def boundary_checks():
    branch = {"initial_alignment": {"pass": True}, "trajectory": [
        {"native_step": t, "label": {"contact_count": int(t > 18)}} for t in range(11, 20)]}
    assert extract_intervention(branch, 10, 8)[-1]["label"]["contact_count"] == 0
    for bad in [branch["trajectory"][1:], branch["trajectory"][:3]+branch["trajectory"][4:]]:
        try:
            extract_intervention({**branch, "trajectory": bad}, 10, 8)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete intervention accepted")


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this CPU analysis through sbatch")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-source", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    boundary_checks()
    report = {"job_id": os.environ["SLURM_JOB_ID"], "config": cfg,
              "audit_source_sha256": digest(Path(__file__)), "boundary_checks": "passed",
              "rinse_bowls": rinse_semantics(args.native_source or Path(cfg["rinse_bowls_source"])),
              "existing_banks": [audit_saved_banks(Path(p)) for p in cfg["saved_results"]],
              "decision": "No automatic training or rollout authorization; review premise evidence first"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"job": report["job_id"], "report": str(args.out),
                      "banks": [r["summary"] for r in report["existing_banks"]]}, indent=2))


if __name__ == "__main__":
    main()
