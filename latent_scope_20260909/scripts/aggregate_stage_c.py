#!/usr/bin/env python3
"""Aggregate fixed-candidate Stage-C arm results without changing the locked metric."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    arms = {}
    for arm in config["training"]["arms"]:
        path = args.results_root / arm / "stage_c_arm_result.json"
        result = json.loads(path.read_text())
        if result.get("verdict") != "STAGE_C_ARM_COMPLETE":
            raise RuntimeError(f"{arm} is incomplete: {result.get('verdict')}")
        if result.get("candidate_ranking") is None:
            raise RuntimeError(f"{arm} has no fixed-candidate ranking result")
        arms[arm] = result

    proposed = arms["compositional_segment"]["candidate_ranking"]
    controls = {
        name: result["candidate_ranking"]
        for name, result in arms.items()
        if name != "compositional_segment"
    }
    strongest_name, strongest = max(
        controls.items(), key=lambda item: item[1]["selected_success_rate"]
    )
    gain = 100.0 * (
        proposed["selected_success_rate"] - strongest["selected_success_rate"]
    )
    threshold = config["gate"]["minimum_selected_success_gain_percentage_points"]
    unstructured = controls["unstructured_segment"]
    composition_specific_gain = 100.0 * (
        proposed["selected_success_rate"] - unstructured["selected_success_rate"]
    )
    passes = gain >= threshold and composition_specific_gain > 0
    output = {
        "verdict": "STAGE_C_SCREEN_PASS" if passes else "STAGE_C_SCREEN_STOP",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_gain_threshold_percentage_points": threshold,
        "proposed": proposed,
        "strongest_control": strongest_name,
        "strongest_control_metrics": strongest,
        "gain_over_strongest_control_percentage_points": gain,
        "gain_over_unstructured_segment_percentage_points": composition_specific_gain,
        "arms": {
            name: {
                "candidate_ranking": result["candidate_ranking"],
                "best_validation": result["best_validation"],
                "trainable_parameters": result["trainable_parameters"],
            }
            for name, result in arms.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(output["verdict"])


if __name__ == "__main__":
    main()
