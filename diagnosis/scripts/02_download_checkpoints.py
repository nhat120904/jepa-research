"""Pre-warm torch.hub cache for every checkpoint the diagnostic uses."""

from __future__ import annotations

import sys

import torch

from _resource_guard import preflight_model_load


CHECKPOINTS = [
    # Primary diagnostic: Metaworld
    ("facebookresearch/jepa-wms", "jepa_wm_metaworld"),
    ("facebookresearch/jepa-wms", "dino_wm_metaworld"),
    # Secondary diagnostic: DROID
    ("facebookresearch/jepa-wms", "jepa_wm_droid"),
    ("facebookresearch/jepa-wms", "dino_wm_droid"),
    ("facebookresearch/jepa-wms", "vjepa2_ac_droid"),
    # Sanity check baselines: Push-T
    ("facebookresearch/jepa-wms", "jepa_wm_pusht"),
    ("facebookresearch/jepa-wms", "dino_wm_pusht"),
]


def main() -> int:
    failures = []
    for repo, hub_id in CHECKPOINTS:
        print(f"[load] {hub_id}", flush=True)
        try:
            preflight_model_load(hub_id, "cuda" if torch.cuda.is_available() else "cpu")
            model, preprocessor = torch.hub.load(repo, hub_id, trust_repo=True)
            print(f"  OK  model={type(model).__name__} "
                  f"preprocessor={type(preprocessor).__name__}")
        except Exception as e:  # noqa: BLE001 — we want all errors reported
            print(f"  FAIL ({e})")
            failures.append((hub_id, repr(e)))
    if failures:
        print("\nFailures:")
        for fid, err in failures:
            print(f"  {fid}: {err}")
        print("\nIf you see 503s, delete external/jepa-wms/uv.lock and re-run uv sync.")
        return 1
    print("\nAll checkpoints loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
