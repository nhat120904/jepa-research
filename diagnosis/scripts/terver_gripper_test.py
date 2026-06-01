"""Terver gripper test — quantitative reproduction of the qualitative finding
(plan Task 3.4 / sanity check #6).

Terver et al. show qualitatively that DROID world models DO respond to the
binary "open gripper + move up" vs "close gripper + move up" counterfactual.
This script turns that into a number: a 2-way CRA on DROID transitions.

For each transition we build the two hardcoded action variants used by the
upstream ``evals/unroll_decode/eval.py::create_counterfactual_actions``:

    base = zeros; z-axis (idx 2) = +0.05 (up)
    open  = base, gripper (idx -1) = -0.75
    close = base, gripper (idx -1) = +0.75

We pick the variant whose gripper direction matches the *actual* action's
gripper sign as "factual", predict both, and check the factual one's latent
prediction is closer to the true next latent. All DROID baselines should score
**> 0.90** here. If they score ~0.50, the diagnostic pipeline has a bug.

Server-side only (needs the real checkpoint + DROID data).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import iterate_droid_trajectories  # noqa: E402
from metrics.distances import get_distance  # noqa: E402
from models.adapters import build_adapter  # noqa: E402

GRIPPER_IDX = -1
Z_IDX = 2


def _variants(B: int, A: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.zeros(B, A, device=device)
    base[:, Z_IDX] = 0.05
    open_a = base.clone(); open_a[:, GRIPPER_IDX] = -0.75
    close_a = base.clone(); close_a[:, GRIPPER_IDX] = +0.75
    return open_a, close_a


@torch.no_grad()
def main(config_path: str, model_name: str, max_transitions: int = 2000) -> int:
    cfg = yaml.safe_load(open(config_path))
    ds = cfg["dataset"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = build_adapter(model_name, device=device).eval()
    dist_fn = get_distance(adapter.spec.planning_distance)

    correct, total = 0, 0
    for traj in iterate_droid_trajectories(ds["root"], max_transitions=max_transitions,
                                           external_root=ds.get("external_root", "external/jepa-wms"),
                                           dataset_kwargs=ds.get("dataset_kwargs")):
        T = traj.action.shape[0]
        z = torch.cat([adapter.encode(traj.obs_visual[i:i + 1].unsqueeze(1))[:, 0].cpu()
                       for i in range(T + 1)], 0).to(device).float()
        z_t, z_t1 = z[:T], z[1:T + 1]
        a = traj.action[:T].to(device).float()
        proprio = traj.proprio[:T].to(device).float() if adapter.uses_proprio() else None

        open_a, close_a = _variants(T, a.shape[-1], device)
        # Factual = whichever gripper sign matches the real action's gripper.
        real_open = (a[:, GRIPPER_IDX] < 0)
        fact = torch.where(real_open[:, None], open_a, close_a)
        cf = torch.where(real_open[:, None], close_a, open_a)

        d_fact = dist_fn(adapter.predict(z_t, fact, proprio_t=proprio), z_t1)
        d_cf = dist_fn(adapter.predict(z_t, cf, proprio_t=proprio), z_t1)
        correct += int((d_fact < d_cf).sum().item())
        total += T
        if total >= max_transitions:
            break

    cra2 = correct / max(total, 1)
    print(f"[terver-gripper] {model_name}: 2-way CRA = {cra2:.3f} over {total} transitions")
    print("  PASS (>0.90)" if cra2 > 0.90 else "  ⚠ FAIL (<0.90) — pipeline bug suspected")
    return 0 if cra2 > 0.90 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/diagnostic_droid.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--max-transitions", type=int, default=2000)
    args = p.parse_args()
    sys.exit(main(args.config, args.model, args.max_transitions))
