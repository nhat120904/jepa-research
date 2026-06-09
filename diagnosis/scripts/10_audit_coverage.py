"""Audit whether requested diagnostic dataset/model coverage is complete.

This is a lightweight filesystem/CSV check. It does not import torch, load
checkpoints, open HDF5 caches, or touch CUDA. Use it after server runs to verify
that the broad coverage request is actually satisfied.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


@dataclass(frozen=True)
class DatasetExpectation:
    name: str
    config: str
    required_models: tuple[str, ...] | None = None


EXPECTED = (
    DatasetExpectation("metaworld", "configs/diagnostic_metaworld.yaml"),
    # User requested "all three baselines" for DROID. This deliberately includes
    # jepa_wm_droid even though the current config documents it as blocked by
    # gated DINOv3 .pth weights; use --allow-known-blockers to report that as a
    # waiver instead of a failure.
    DatasetExpectation(
        "droid",
        "configs/diagnostic_droid.yaml",
        ("dino_wm_droid", "vjepa2_ac_droid", "jepa_wm_droid"),
    ),
    DatasetExpectation("robocasa", "configs/diagnostic_robocasa.yaml"),
    DatasetExpectation("franka_custom", "configs/diagnostic_franka_custom.yaml"),
    DatasetExpectation("pusht", "configs/diagnostic_pusht.yaml"),
    DatasetExpectation("point_maze", "configs/diagnostic_point_maze.yaml"),
    DatasetExpectation("wall", "configs/diagnostic_wall.yaml"),
)

KNOWN_BLOCKED_MODELS = {
    ("droid", "jepa_wm_droid"): "blocked by gated DINOv3 ViT-L .pth weights",
    ("robocasa", "jepa_wm_droid"): "blocked by gated DINOv3 ViT-L .pth weights",
}


@dataclass
class AuditRow:
    dataset: str
    check: str
    status: str
    detail: str


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _read_config(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _expected_models(dataset: DatasetExpectation, cfg: dict) -> tuple[str, ...]:
    if dataset.required_models is not None:
        return dataset.required_models
    return tuple(cfg.get("models", ()))


def audit_dataset(root: Path, dataset: DatasetExpectation,
                  *, allow_known_blockers: bool) -> list[AuditRow]:
    rows: list[AuditRow] = []
    cfg_path = root / dataset.config
    cfg = _read_config(cfg_path)
    rows.append(AuditRow(dataset.name, "config_exists", _status(cfg is not None), str(cfg_path)))
    if cfg is None:
        return rows

    cfg_dataset = cfg.get("dataset", {}).get("name")
    rows.append(AuditRow(
        dataset.name,
        "config_dataset_name",
        _status(cfg_dataset == dataset.name),
        f"expected={dataset.name} actual={cfg_dataset}",
    ))

    configured_cache_root = root / cfg["latent_cache"]["root"]
    cache_roots = [configured_cache_root]
    cache_override = os.environ.get("CAI_JEPA_LATENT_CACHE_ROOT")
    if cache_override:
        override_root = Path(cache_override)
        if override_root not in cache_roots:
            cache_roots.insert(0, override_root)
    csv_path = root / cfg["output"]["csv"]
    models = _expected_models(dataset, cfg)
    config_models = set(cfg.get("models", ()))
    for model in models:
        blocker = KNOWN_BLOCKED_MODELS.get((dataset.name, model))
        is_waived = bool(blocker and allow_known_blockers)
        in_config = model in config_models
        rows.append(AuditRow(
            dataset.name,
            f"model_in_config:{model}",
            "WAIVED" if is_waived and not in_config else _status(in_config),
            blocker if is_waived and not in_config else dataset.config,
        ))

        cache_candidates = [r / f"{dataset.name}__{model}.h5" for r in cache_roots]
        cache = next((p for p in cache_candidates if p.exists()), cache_candidates[0])
        sidecar_candidates = [p.with_name(p.name + ".regimes.json") for p in cache_candidates]
        sidecar = next((p for p in sidecar_candidates if p.exists()), sidecar_candidates[0])
        rows.append(AuditRow(
            dataset.name,
            f"latent_cache:{model}",
            "WAIVED" if is_waived and not cache.exists() else _status(cache.exists()),
            blocker if is_waived and not cache.exists() else " | ".join(str(p) for p in cache_candidates),
        ))
        rows.append(AuditRow(
            dataset.name,
            f"regime_sidecar:{model}",
            "WAIVED" if is_waived and not sidecar.exists() else _status(sidecar.exists()),
            blocker if is_waived and not sidecar.exists() else " | ".join(str(p) for p in sidecar_candidates),
        ))

    rows.append(AuditRow(dataset.name, "csv_exists", _status(csv_path.exists()), str(csv_path)))
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            csv_dataset_ok = "dataset" in df.columns and set(df["dataset"].dropna()) == {dataset.name}
            rows.append(AuditRow(
                dataset.name,
                "csv_dataset_name",
                _status(csv_dataset_ok),
                f"values={sorted(df['dataset'].dropna().unique()) if 'dataset' in df.columns else 'missing column'}",
            ))
            present = set(df["model"].dropna()) if "model" in df.columns else set()
            for model in models:
                blocker = KNOWN_BLOCKED_MODELS.get((dataset.name, model))
                is_waived = bool(blocker and allow_known_blockers)
                rows.append(AuditRow(
                    dataset.name,
                    f"csv_model:{model}",
                    "WAIVED" if is_waived and model not in present else _status(model in present),
                    blocker if is_waived and model not in present else f"present={sorted(present)}",
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(AuditRow(dataset.name, "csv_readable", "FAIL", repr(exc)))
    return rows


def selected_expectations(names: tuple[str, ...] | None = None) -> tuple[DatasetExpectation, ...]:
    if not names:
        return EXPECTED
    wanted = set(names)
    known = {d.name for d in EXPECTED}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Known: {sorted(known)}")
    return tuple(d for d in EXPECTED if d.name in wanted)


def audit(root: Path, *, allow_known_blockers: bool,
          datasets: tuple[str, ...] | None = None) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for dataset in selected_expectations(datasets):
        rows.extend(audit_dataset(root, dataset, allow_known_blockers=allow_known_blockers))
    return rows


def render_markdown(rows: Iterable[AuditRow]) -> str:
    lines = [
        "# Diagnostic Coverage Audit",
        "",
        "| Dataset | Check | Status | Detail |",
        "|---|---|---|---|",
    ]
    for row in rows:
        detail = row.detail.replace("|", "\\|")
        lines.append(f"| {row.dataset} | `{row.check}` | {row.status} | {detail} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="diagnosis repo root")
    parser.add_argument("--out", default="results/coverage_audit.md")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[d.name for d in EXPECTED],
        help="limit audit to one dataset; may be repeated",
    )
    parser.add_argument(
        "--allow-known-blockers",
        action="store_true",
        help="waive explicitly documented external blockers such as gated DINOv3 weights",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = audit(
        root,
        allow_known_blockers=args.allow_known_blockers,
        datasets=tuple(args.dataset) if args.dataset else None,
    )
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(rows))
    print(f"Wrote {out}")

    failed = [r for r in rows if r.status == "FAIL"]
    if failed:
        print(f"Coverage audit FAILED ({len(failed)} failing checks).")
        for row in failed[:20]:
            print(f"  {row.dataset}: {row.check} -> {row.detail}")
        if len(failed) > 20:
            print(f"  ... {len(failed) - 20} more")
        return 1
    print("Coverage audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
