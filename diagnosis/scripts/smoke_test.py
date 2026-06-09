"""Verify every checkpoint loads, encodes, and predicts on the REAL API.

Run FIRST after 02_download_checkpoints.py. If it fails, fix the adapter /
environment before anything else. Needs the upstream env (einops, tensordict,
omegaconf, decord/datasets) + the checkpoints — i.e. the server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from scripts._resource_guard import preflight_model_load  # noqa: E402


# (hub_id, action_dim, image_size)
SMOKE_TARGETS = [
    ("jepa_wm_metaworld", 4, 256),
    ("dino_wm_metaworld", 4, 224),
    ("jepa_wm_droid", 7, 256),
    ("dino_wm_droid", 7, 224),
    ("vjepa2_ac_droid", 7, 256),
]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    for hub_id, action_dim, image_size in SMOKE_TARGETS:
        print(f"\n[smoke] {hub_id} on {device}")
        try:
            preflight_model_load(hub_id, device)
            adapter = build_adapter(hub_id, device=device).eval()
            # Raw [0,255] frame, (B, T=1, C, H, W) — adapter.encode does /255 + transform.
            visual = torch.randint(0, 256, (2, 1, 3, image_size, image_size), dtype=torch.float32)
            with torch.no_grad():
                z = adapter.encode(visual)            # (2, 1, V, H, W, D)
                print(f"  encode OK  z.shape={tuple(z.shape)}  uses_proprio={adapter.uses_proprio()}")
                z_t = z[:, 0]                         # (2, V, H, W, D)
                a = torch.rand(2, action_dim) * 0.1
                z_next = adapter.predict(z_t, a)      # action normalized inside
                print(f"  predict OK z_next.shape={tuple(z_next.shape)}")
                assert z_next.shape == z_t.shape, "predict must preserve frame shape"
            del adapter
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL: {e}")
            failures.append((hub_id, repr(e)))

    if failures:
        print("\nSmoke FAILED:")
        for h, e in failures:
            print(f"  {h}: {e}")
        return 1
    print("\nSmoke PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
