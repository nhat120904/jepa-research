#!/usr/bin/env python3
"""Merge encoded Stage-C sources while enforcing split and validation-branch rules."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    seen = set()
    encoder = None
    for path in args.inputs:
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != 1:
            raise RuntimeError(f"Unsupported manifest {path}")
        if encoder is None:
            encoder = manifest.get("encoder")
        elif manifest.get("encoder") != encoder:
            raise RuntimeError("All Stage-C sources must use the identical frozen encoder")
        for entry in manifest["episodes"]:
            uid = entry["episode_uid"]
            if uid in seen:
                raise RuntimeError(f"Duplicate episode_uid {uid}")
            seen.add(uid)
            if entry.get("source") == "stage_b_validation_branches" and entry.get(
                "allow_training", False
            ):
                raise RuntimeError(f"Validation branch {uid} cannot be training data")
            copied = dict(entry)
            source_file = path.parent / entry["path"]
            copied["path"] = os.path.relpath(
                source_file.resolve(), start=args.output.parent.resolve()
            )
            entries.append(copied)
    output = {"schema_version": 1, "encoder": encoder, "episodes": entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
