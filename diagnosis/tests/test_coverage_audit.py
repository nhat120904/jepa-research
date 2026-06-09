import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coverage_audit", ROOT / "scripts" / "10_audit_coverage.py"
)
coverage_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = coverage_audit
SPEC.loader.exec_module(coverage_audit)


def _write_config(root: Path, dataset: str, models: list[str]) -> None:
    cfg = {
        "dataset": {"name": dataset},
        "models": models,
        "latent_cache": {"root": "data/precomputed_latents"},
        "output": {"csv": f"results/{dataset}_diagnostic.csv"},
    }
    path = root / "configs" / f"diagnostic_{dataset}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))


def test_audit_fails_missing_outputs(tmp_path):
    _write_config(tmp_path, "pusht", ["jepa_wm_pusht"])
    rows = coverage_audit.audit_dataset(
        tmp_path,
        coverage_audit.DatasetExpectation("pusht", "configs/diagnostic_pusht.yaml"),
        allow_known_blockers=False,
    )
    failures = [r.check for r in rows if r.status == "FAIL"]
    assert "latent_cache:jepa_wm_pusht" in failures
    assert "csv_exists" in failures


def test_audit_passes_complete_minimal_dataset(tmp_path):
    model = "jepa_wm_pusht"
    _write_config(tmp_path, "pusht", [model])
    cache_root = tmp_path / "data" / "precomputed_latents"
    cache_root.mkdir(parents=True)
    cache = cache_root / f"pusht__{model}.h5"
    cache.write_bytes(b"placeholder")
    cache.with_name(cache.name + ".regimes.json").write_text("{}")

    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame({"dataset": ["pusht"], "model": [model], "status": ["ok"]}).to_csv(
        results / "pusht_diagnostic.csv", index=False
    )

    rows = coverage_audit.audit_dataset(
        tmp_path,
        coverage_audit.DatasetExpectation("pusht", "configs/diagnostic_pusht.yaml"),
        allow_known_blockers=False,
    )
    assert {r.status for r in rows} == {"PASS"}


def test_audit_accepts_configured_cache_when_override_is_set(tmp_path, monkeypatch):
    model = "jepa_wm_pusht"
    _write_config(tmp_path, "pusht", [model])
    cache_root = tmp_path / "data" / "precomputed_latents"
    cache_root.mkdir(parents=True)
    cache = cache_root / f"pusht__{model}.h5"
    cache.write_bytes(b"placeholder")
    cache.with_name(cache.name + ".regimes.json").write_text("{}")

    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame({"dataset": ["pusht"], "model": [model], "status": ["ok"]}).to_csv(
        results / "pusht_diagnostic.csv", index=False
    )
    monkeypatch.setenv("CAI_JEPA_LATENT_CACHE_ROOT", str(tmp_path / "alternate_cache"))

    rows = coverage_audit.audit_dataset(
        tmp_path,
        coverage_audit.DatasetExpectation("pusht", "configs/diagnostic_pusht.yaml"),
        allow_known_blockers=False,
    )
    assert {r.status for r in rows} == {"PASS"}


def test_known_blocker_can_be_waived(tmp_path):
    _write_config(tmp_path, "droid", ["dino_wm_droid", "vjepa2_ac_droid"])
    rows = coverage_audit.audit_dataset(
        tmp_path,
        coverage_audit.DatasetExpectation(
            "droid",
            "configs/diagnostic_droid.yaml",
            ("jepa_wm_droid",),
        ),
        allow_known_blockers=True,
    )
    waived = [r for r in rows if r.status == "WAIVED"]
    assert waived
    assert all("gated DINOv3" in r.detail for r in waived)


def test_audit_can_be_scoped_to_one_dataset(tmp_path):
    _write_config(tmp_path, "pusht", ["jepa_wm_pusht"])
    rows = coverage_audit.audit(
        tmp_path,
        allow_known_blockers=False,
        datasets=("pusht",),
    )
    assert rows
    assert {r.dataset for r in rows} == {"pusht"}
