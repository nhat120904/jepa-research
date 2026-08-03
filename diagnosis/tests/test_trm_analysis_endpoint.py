"""Regression tests for the TRM report's strict endpoint convention."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "52_analyze_trm.py"
SPEC = importlib.util.spec_from_file_location("analyze_trm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYZE_TRM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE_TRM)


def test_success_vector_uses_endpoint_not_any_step(tmp_path: Path) -> None:
    csv_path = tmp_path / "cell.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "success", "success_end", "obj_goal_dist"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed": 30000,
                "success": 1,
                "success_end": 0,
                "obj_goal_dist": 0.1,
            }
        )

    rows = ANALYZE_TRM.load_cell(csv_path, seed0=30000, n=1)

    assert ANALYZE_TRM.SUCCESS_FIELD == "success_end"
    assert ANALYZE_TRM.success_vector(rows).tolist() == [0]
