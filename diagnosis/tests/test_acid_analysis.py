"""Offline tests for ACID paired inference helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "acid_analysis", ROOT / "scripts" / "53_analyze_acid_baseline.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_exact_mcnemar_known_cases():
    assert MOD.exact_mcnemar_p(0, 0) == 1.0
    assert MOD.exact_mcnemar_p(0, 5) == 0.0625
    assert MOD.exact_mcnemar_p(2, 2) == 1.0
    assert MOD.exact_mcnemar_p(1, 9) == 22 / 1024


def test_paired_bootstrap_is_deterministic_and_tracks_mean():
    diff = np.array([1.0, 1.0, -1.0, 1.0])
    first = MOD.paired_bootstrap_ci(diff, n_resamples=1000, seed=7)
    second = MOD.paired_bootstrap_ci(diff, n_resamples=1000, seed=7)
    assert first == second
    assert first[0] == 0.5
    assert first[1] <= first[0] <= first[2]


def test_parse_locked_cell_filename():
    parsed = MOD.parse_cell_path(Path(
        "acid_dino_wm_metaworld_oracle_mw-pick-place_seed22000_n32.csv"
    ))
    assert parsed == {
        "model": "dino_wm_metaworld",
        "dynamics": "oracle",
        "task": "mw-pick-place",
        "seed0": 22000,
        "n": 32,
    }
