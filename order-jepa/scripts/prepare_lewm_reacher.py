#!/usr/bin/env python3
"""Download and validate the official LeWM Reacher checkpoint on Slurm."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from order_jepa.lewm_reacher import OFFICIAL_REPO, build_provenance  # noqa: E402


def main() -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("checkpoint preparation must run on a Slurm compute node")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-worldmodel-source", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(OFFICIAL_REPO)
    del model
    checkpoint_dir = (
        args.stablewm_home / "checkpoints/models--quentinll--lewm-reacher"
    )
    revision = None
    try:
        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{OFFICIAL_REPO}", timeout=30
        ) as response:
            revision = json.loads(response.read()).get("sha")
    except Exception as error:  # provenance stays usable when the API is transiently down
        print(f"WARNING: could not resolve Hugging Face revision: {error}")
    provenance = build_provenance(
        args.stable_worldmodel_source,
        checkpoint_dir,
        checkpoint_revision=revision,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(provenance.to_dict(), indent=2) + "\n")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
