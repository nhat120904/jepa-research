"""Test 2: multi-step rollout object-fidelity, frozen vs corrected predictor —
does the corrected predictor's object channel stay accurate over an H-step
imagined rollout, or does it compound? (the second confound on the option-C
closed-loop null; docs/plans/2026-06-18-residual-predictor-design.md §6 gate 2).

For held-out cache trajectories, roll BOTH the frozen predictor F and the
corrected F+Δ forward H model-steps on the recorded actions (teacher-forced
actions, free-running latent), decode the object at each step with the object
probe, and report the object-error growth curve vs the ground-truth state. If the
corrected curve stays ~flat near the one-step floor, the closed-loop failure is
NOT multi-step drift (-> precision/representation); if it blows up, compounding
is a contributor the planner can't escape.

Needs the predictor -> GPU. Run AFTER the closed-loop sweep (serialize GPU).

    .venv/Scripts/python.exe scripts/23_rollout_fidelity.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --probe checkpoints/object_probe_dino_wm_metaworld.pt \
        --residual-head checkpoints/residual_predictor_dino_wm_metaworld.pt --H 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.heads import load_residual_head  # noqa: E402
from models.probes import load_probe  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def build_sequences(cache, step, H, val_tids, stride):
    """(z0, prop0, actions (n,H,A_stack), obj_true (n,H+1,3)) from held-out trajs."""
    Z0, P0, ACTS, OBJ = [], [], [], []
    a_lo = OBJECT_SLICE
    for tid in val_tids:
        traj = cache.read_trajectory(tid)
        z, act = traj["z"], traj["action"]
        state = traj.get("state"); prop = traj.get("proprio")
        if state is None:
            continue
        La = len(act)
        for i in range(0, La - H * step + 1, stride * step):
            acts = np.stack([act[i + h * step: i + (h + 1) * step].reshape(-1)
                             for h in range(H)])                 # (H, A_stack)
            obj = np.stack([state[i + h * step][a_lo] for h in range(H + 1)])  # (H+1,3)
            Z0.append(z[i]); ACTS.append(acts); OBJ.append(obj)
            P0.append(prop[i] if prop is not None else np.zeros(1, np.float32))
    return (torch.tensor(np.stack(Z0)).float(), torch.tensor(np.stack(P0)).float(),
            torch.tensor(np.stack(ACTS)).float(), torch.tensor(np.stack(OBJ)).float())


@torch.no_grad()
def frozen_rollout(adapter, z0, acts, prop0):
    return adapter.predict_rollout(z0, acts, proprio_t=prop0 if adapter.uses_proprio() else None)


@torch.no_grad()
def corrected_rollout(adapter, rh, z0, acts, prop0):
    from einops import rearrange
    from tensordict.tensordict import TensorDict
    b = adapter
    B, H, _ = acts.shape
    z_cur = z0.to(b.device).float()
    a = acts.to(b.device).float().reshape(B, -1, b.action_dim())
    a = b.normalize_action(a).reshape(B, H, b._model_action_dim)
    uses_p = b.spec.uses_proprio
    pf = b.encode_proprio_features(prop0.reshape(B, 1, -1).to(b.device)) if uses_p else None
    outs = [z_cur]
    for t in range(H):
        a_t = a[:, t]
        if uses_p:
            ctxt = TensorDict({"visual": z_cur.unsqueeze(1), "proprio": pf}, batch_size=[])
            pred = b.encpred.unroll(ctxt, act_suffix=a_t.unsqueeze(0))
            z_next = pred["visual"][-1]
            pf = rearrange(pred["proprio"], "t b ... -> b t ...")[:, -1:]
        else:
            z_next = b.encpred.unroll(z_cur.unsqueeze(1), act_suffix=a_t.unsqueeze(0))[-1]
        z_cur = z_next + rh(z_cur, a_t)
        outs.append(z_cur)
    return torch.stack(outs, dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--residual-head", default=None)
    ap.add_argument("--predictor-lora", default=None,
                    help="LoRA ckpt (scripts/26): frozen=LoRA-off rollout, corrected=LoRA-on")
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4, help="start-index stride (in model-steps)")
    ap.add_argument("--max-seqs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rollout_fidelity.csv")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, cfg["dataset"]["name"])
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    probe, _ = load_probe(args.probe, device)
    lora = None
    rh = None
    if args.predictor_lora:
        from models.heads.lora_predictor import load_predictor_lora, set_lora_enabled
        lora, _ = load_predictor_lora(adapter, args.predictor_lora, device)
        print(f"predictor LoRA: {args.predictor_lora} — corrected = LoRA-on rollout", flush=True)
    elif args.residual_head:
        rh, _ = load_residual_head(args.residual_head, device)
    else:
        raise SystemExit("need --residual-head or --predictor-lora")

    with LatentCache(cache_path, mode="r") as cache:
        tids = sorted(cache.trajectory_ids())
        rng = np.random.default_rng(args.seed); rng.shuffle(tids)
        val_tids = tids[: max(1, int(len(tids) * args.val_frac))]
        z0, p0, acts, obj = build_sequences(cache, step, args.H, val_tids, args.stride)
    if len(z0) > args.max_seqs:
        sel = np.random.default_rng(1).choice(len(z0), args.max_seqs, replace=False)
        z0, p0, acts, obj = z0[sel], p0[sel], acts[sel], obj[sel]
    print(f"held-out sequences: {len(z0)} (H={args.H} steps)", flush=True)

    fro = np.zeros((args.H + 1,)); cor = np.zeros((args.H + 1,)); n = 0
    for s in range(0, len(z0), args.chunk):
        e = slice(s, s + args.chunk)
        if lora is not None:
            set_lora_enabled(lora, False)
            zf = frozen_rollout(adapter, z0[e], acts[e], p0[e])      # LoRA off = frozen
            set_lora_enabled(lora, True)
            zc = frozen_rollout(adapter, z0[e], acts[e], p0[e])      # LoRA on = corrected
        else:
            zf = frozen_rollout(adapter, z0[e], acts[e], p0[e])      # (b,H+1,...)
            zc = corrected_rollout(adapter, rh, z0[e], acts[e], p0[e])
        ot = obj[e].to(device)
        for h in range(args.H + 1):
            with torch.no_grad():
                fro[h] += (probe(zf[:, h]) - ot[:, h]).norm(dim=-1).sum().item()
                cor[h] += (probe(zc[:, h]) - ot[:, h]).norm(dim=-1).sum().item()
        n += zf.shape[0]
    fro /= n; cor /= n

    print(f"\nobject error (cm) by horizon step (n={n}):")
    print(f"{'step':>4s} {'frozen':>8s} {'corrected':>10s}")
    rows = []
    for h in range(args.H + 1):
        print(f"{h:>4d} {100*fro[h]:8.2f} {100*cor[h]:10.2f}", flush=True)
        rows.append({"step": h, "frozen_cm": 100*fro[h], "corrected_cm": 100*cor[h]})
    import pandas as pd
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
