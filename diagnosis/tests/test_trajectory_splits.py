from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from data.trajectory_splits import (
    build_trajectory_manifest,
    filter_records,
    load_manifest,
    write_manifest_once,
)


ROOT = Path(__file__).resolve().parents[1]


def _manifest(seed=7, *, model="dino_wm_droid"):
    return build_trajectory_manifest(
        [f"traj-{i:03d}" for i in range(100)], seed=seed,
        dataset="droid", model=model,
    )


def test_default_split_is_deterministic_disjoint_70_15_15():
    first = _manifest()
    second = _manifest()
    assert first == second
    assert {name: len(first["splits"][name]) for name in ("train", "val", "test")} == {
        "train": 70, "val": 15, "test": 15,
    }
    sets = [set(first["splits"][name]) for name in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set().union(*sets)) == 100


def test_manifest_is_write_once_and_rejects_mismatch(tmp_path):
    path = tmp_path / "split.json"
    write_manifest_once(path, _manifest())
    write_manifest_once(path, _manifest())  # an identical Slurm-array writer is safe
    assert load_manifest(path) == _manifest()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_manifest_once(path, _manifest(model="jepa_wm_droid"))


def test_filter_records_enforces_complete_cache_identity():
    manifest = _manifest()
    records = [{"tid": f"traj-{i:03d}", "i": i} for i in range(100)]
    test_records = filter_records(records, manifest, "test")
    assert {r["tid"] for r in test_records} == set(manifest["splits"]["test"])
    with pytest.raises(ValueError, match="manifest/cache trajectory mismatch"):
        filter_records(records[:-1], manifest, "test")


def _load_probe08():
    spec = importlib.util.spec_from_file_location(
        "planning_probe08", ROOT / "scripts" / "08_planning_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_manifest_can_come_from_checkpoint_and_external_must_match(tmp_path):
    probe = _load_probe08()
    manifest = _manifest()
    ckpt = tmp_path / "predictor.pt"
    torch.save({"data_split": {"manifest": manifest}}, ckpt)
    resolved = probe.resolve_eval_manifest(
        eval_split="test", split_manifest=None, predictor_lora=str(ckpt),
        model="dino_wm_droid", dataset="droid",
    )
    assert resolved["manifest_sha256"] == manifest["manifest_sha256"]

    other_path = tmp_path / "other.json"
    write_manifest_once(other_path, _manifest(model="jepa_wm_droid"))
    with pytest.raises(ValueError, match="does not match predictor checkpoint"):
        probe.resolve_eval_manifest(
            eval_split="test", split_manifest=str(other_path), predictor_lora=str(ckpt),
            model="dino_wm_droid", dataset="droid",
        )


def test_non_all_eval_requires_manifest():
    probe = _load_probe08()
    with pytest.raises(ValueError, match="requires --split-manifest"):
        probe.resolve_eval_manifest(
            eval_split="test", split_manifest=None, predictor_lora=None,
            model="dino_wm_droid", dataset="droid",
        )
