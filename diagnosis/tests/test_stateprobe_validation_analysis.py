"""Unit checks for the optimizer-induced stateprobe validation metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "63_analyze_stateprobe_validation.py"
)
SPEC = importlib.util.spec_from_file_location("stateprobe_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_population_metrics_use_identical_candidates() -> None:
    population = pd.DataFrame(
        {
            "candidate": [0, 1, 2, 3],
            "proxy_cost": [0.0, 1.0, 2.0, 3.0],
            "true_shaped_cost": [3.0, 2.0, 1.0, 0.0],
            "obj_decode_error_cm": [1.0, 2.0, 3.0, 4.0],
            "ee_decode_error_cm": [2.0, 4.0, 6.0, 8.0],
        }
    )

    result = MODULE.summarize_population(population, topk_frac=0.5)

    assert result["n_candidate"] == 4
    assert result["object_error_cm"] == pytest.approx(2.5)
    assert result["hand_error_cm"] == pytest.approx(5.0)
    assert result["cost_spearman"] == pytest.approx(-1.0)
    assert result["reference_top10_recall"] == pytest.approx(0.0)
