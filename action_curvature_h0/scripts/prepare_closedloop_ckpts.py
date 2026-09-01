#!/usr/bin/env python3
"""Write our fine-tuned arms as folders that swm.wm.utils.load_pretrained accepts.

The official evaluator (`stable-worldmodel/scripts/plan/eval_wm.py`) loads a
policy through ``load_pretrained``, which resolves a folder holding exactly one
``.pt`` and a ``config.json``.  Our training saved ``{"model": state_dict, ...}``
instead, so each arm is rewritten into the released checkpoint's own container
format -- config copied verbatim, weights swapped -- and nothing about the
evaluation path is forked.

``original`` is the released checkpoint copied unchanged, so the control goes
through the identical code path as the arms.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--released", type=Path, required=True,
                   help="folder of the released checkpoint (config.json + *.pt)")
    p.add_argument("--arm", action="append", required=True, metavar="NAME[=CKPT]",
                   help="NAME alone copies the release; NAME=path swaps in that "
                        "state dict")
    p.add_argument("--out-root", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    released_pt = sorted(args.released.glob("*.pt"))
    if len(released_pt) != 1:
        raise RuntimeError(f"expected one .pt in {args.released}, got {released_pt}")
    container = torch.load(released_pt[0], map_location="cpu", weights_only=False)
    is_wrapped = isinstance(container, dict) and "model" in container
    reference = container["model"] if is_wrapped else container
    print(f"released container: {'wrapped {model: ...}' if is_wrapped else 'raw state_dict'}"
          f"  ({len(reference)} tensors)")

    for spec in args.arm:
        name, sep, path = spec.partition("=")
        dest = args.out_root / name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.released / "config.json", dest / "config.json")

        if not sep:
            shutil.copy2(released_pt[0], dest / "weights.pt")
            print(f"  {name:12s} <- released checkpoint, copied unchanged")
            continue

        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing = set(reference) - set(state)
        extra = set(state) - set(reference)
        if missing or extra:
            raise RuntimeError(
                f"{name}: state dict does not match the release "
                f"({len(missing)} missing, {len(extra)} unexpected)")
        changed = sum(1 for k in reference
                      if not torch.equal(reference[k].float(), state[k].float()))
        torch.save({"model": state} if is_wrapped else state, dest / "weights.pt")
        meta = {k: v for k, v in blob.items() if k != "model"} if isinstance(blob, dict) else {}
        (dest / "arm_meta.json").write_text(json.dumps(
            {"source": str(path), "tensors_changed_vs_release": changed,
             "n_tensors": len(reference), **{k: str(v) for k, v in meta.items()}},
            indent=2) + "\n")
        # A fine-tune that changed nothing would silently evaluate as the control.
        if changed == 0:
            raise RuntimeError(f"{name}: identical to the release; refusing to write")
        print(f"  {name:12s} <- {path}  ({changed}/{len(reference)} tensors differ)")


if __name__ == "__main__":
    main()
