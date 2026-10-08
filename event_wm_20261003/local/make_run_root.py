"""Build a local run root: frozen source + protocol + stage drivers, the layout the cluster drivers expect.

    python local/make_run_root.py --profile adapter --name unified_local
    python local/make_run_root.py --profile generic --name generic_local --adapter-root unified_local

Why this exists: `scripts/` is NOT the code that produced the latest results. It lacks event_support.py,
skill_segments.py and train_event_support.py and holds older u_wm/u_skill/u_events/s_entities. The newest
self-contained snapshots are the frozen sources under docs/:
  adapter : docs/generic_state_20261007/base_source   (unified rerun source + the 2026-10-07 u_events fix)
  generic : docs/generic_state_20261007/source_generic (same, with the layout-free front end g_entities.py)
Drivers come from the matching docs directory. Only data_root and evaluation.workers are changed relative to the
frozen protocol.json (recorded in local_overrides.json); recipe, steps and seeds are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
PROFILES = {
    "adapter": dict(
        drivers=DOCS / "unified_rerun_20261006",
        source=DOCS / "generic_state_20261007" / "base_source",
        files=["run_prepare.py", "run_train.py", "run_eval.py", "stage_utils.py", "verify_prepared.py", "result_checks.py",
               "test_segments.py", "test_state_semantics.py", "test_results.py", "aggregate.py", "protocol.json"]),
    "generic": dict(
        drivers=DOCS / "generic_state_20261007",
        source=DOCS / "generic_state_20261007" / "source_generic",
        files=["run_prepare_generic.py", "run_train.py", "run_eval.py", "stage_utils.py", "verify_generic.py",
               "compare_events.py", "result_checks.py", "protocol.json"]),
}
SKIP_DIRS = {"prep", "runs", "logs", "__pycache__"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8, help="closed-loop worker processes (protocol default 10 on the cluster)")
    ap.add_argument("--adapter-root", help="generic profile: run-root name of the adapter run used for the matched comparison")
    a = ap.parse_args()
    runs = Path(os.environ.get("EVENT_WM_RUNS", r"E:\jepa-data\event_wm"))
    data = Path(os.environ.get("EVENT_WM_DATA", r"E:\jepa-data\ogbench\data"))
    root = runs / a.name
    if root.exists():
        print(f"{root} already exists; choose another --name (run roots are never overwritten)", file=sys.stderr)
        return 2
    prof = PROFILES[a.profile]
    root.mkdir(parents=True)
    shutil.copytree(prof["source"], root / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in prof["files"]:
        shutil.copy2(prof["drivers"] / name, root / name)

    proto = json.loads((root / "protocol.json").read_text())
    overrides = {"data_root": {"from": proto["data_root"], "to": str(data)},
                 "evaluation.workers": {"from": proto["evaluation"]["workers"], "to": a.workers}}
    proto["data_root"] = str(data)
    proto["evaluation"]["workers"] = a.workers
    if "adapter_prep" in proto:
        if not a.adapter_root:
            print("--adapter-root is needed for the generic profile (matched adapter comparison)", file=sys.stderr)
            return 2
        new = str(runs / a.adapter_root / "prep")
        overrides["adapter_prep"] = {"from": proto["adapter_prep"], "to": new}
        proto["adapter_prep"] = new
    (root / "protocol.json").write_text(json.dumps(proto, indent=2) + "\n")
    (root / "local_overrides.json").write_text(json.dumps(
        {"profile": a.profile, "source": str(prof["source"].relative_to(REPO)), "drivers": str(prof["drivers"].relative_to(REPO)),
         "overrides_vs_frozen_protocol": overrides}, indent=2) + "\n")

    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_dir() or rel.parts[0] in SKIP_DIRS or p.name in {"SOURCE_SHA256SUMS", "local_overrides.json"} or p.suffix == ".pyc":
            continue
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel.as_posix()}")
    (root / "SOURCE_SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"run root: {root}\n  profile={a.profile} files={len(lines)} workers={a.workers}\n  data_root={data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
