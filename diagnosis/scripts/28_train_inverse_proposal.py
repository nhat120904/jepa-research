"""Train the INVERSE ACTION PROPOSAL h_inv(z_t, Δobj) → raw action — lever #2,
WAV-style (docs/plans/2026-06-24-inverse-action-proposal-design.md).

The closed-loop wall is no longer scoring (the dyn-head scores contact at cf-corr
+0.68) but SAMPLING: zero-mean Gaussian CEM essentially never proposes the
reach→grasp→lift sequence, so the grounded term has no contact to grade and
success stays 0/16. World Action Verifier's forward–inverse asymmetry says the
action is identifiable from a low-dimensional object/agent-relevant readout, so an
inverse over (latent, desired object motion) is reliable even where the frozen
forward predictor's counterfactual object channel is dead. We train that inverse on
the cached expert transitions and use it at plan time to SEED the CEM mean with a
contact-creating proposal toward the goal object (scripts/18 `l2inv`/`hdyninv`).

    h_inv(z_t, Δobj) → â        (raw stacked action, the space CEM samples in)
    Δobj = obj_{t+1} − obj_t    (cached sim-state object slice; test-time: spatial probe)
    L = w · ‖ h_inv(z_t, Δobj) − a_t ‖²   (w upweights object-moving / contact steps)

Cache-only; encoder + predictor are never touched (we don't even call the forward
model). The cache is materialised into RAM ONCE (the 26 GB HDF5 is far too slow to
re-stream every epoch), then training is in-memory; a checkpoint is saved every
epoch so a wall-clock timeout still leaves a usable head. Saves
checkpoints/inverse_proposal_<model>.pt in the format scripts/18 `--inverse-head`
loads (models.probes.load_inverse_head).

    .venv/bin/python scripts/28_train_inverse_proposal.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --epochs 20 --contact-weight 4 --move-thresh 0.005
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

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.probes import InverseProposalHead  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def split_by_trajectory(records, val_frac, seed):
    tids = sorted({r["tid"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(tids)
    val_tids = set(tids[: max(1, int(len(tids) * val_frac))])
    return ([r for r in records if r["tid"] not in val_tids],
            [r for r in records if r["tid"] in val_tids])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--load-chunk", type=int, default=4096,
                    help="cache-read chunk size for the one-time RAM materialisation")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--obj-emb-dim", type=int, default=64)
    ap.add_argument("--move-thresh", type=float, default=0.005,
                    help="‖Δobj‖ (m) above which a transition counts as object-moving "
                         "(contact); these steps are upweighted in the loss")
    ap.add_argument("--contact-weight", type=float, default=4.0,
                    help="loss weight multiplier for object-moving steps (1 + this)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--out", default=None,
                    help="explicit ckpt path (default checkpoints/inverse_proposal_<model>.pt)")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "4")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model,
                                   cfg["dataset"]["name"])
    # The adapter is only used for the cache stride (frames_per_step); h_inv never
    # calls the forward model. Raw stacked actions are the CEM's sampling space.
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    regime_by_traj = read_regimes(cache_path)

    # ---- one-time materialisation into RAM (no per-epoch cache re-streaming) ----
    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=True)
        train_recs, val_recs = split_by_trajectory(records, args.val_frac, args.seed)
        print(f"transitions: train={len(train_recs)} val={len(val_recs)}", flush=True)

        def load_all(recs):
            zs, as_, o0s, o1s = [], [], [], []
            for lo in range(0, len(recs), args.load_chunk):
                d = helpers.materialize_records(cache, recs[lo: lo + args.load_chunk], step,
                                                want_proprio=False, want_state=True)
                zs.append(d["z_t"].clone())
                as_.append(d["a_t"].float().clone())
                o0s.append(d["state_t"][:, OBJECT_SLICE].float().clone())
                o1s.append(d["state_t1"][:, OBJECT_SLICE].float().clone())
                del d
            return (torch.cat(zs), torch.cat(as_), torch.cat(o0s), torch.cat(o1s))

        z_tr, a_tr, o0_tr, o1_tr = load_all(train_recs)
        z_va, a_va, o0_va, o1_va = load_all(val_recs)
    print(f"materialised to RAM: train z {tuple(z_tr.shape)}  val z {tuple(z_va.shape)}",
          flush=True)

    latent_dim = int(z_tr.shape[-1])
    action_dim = int(a_tr.shape[-1])                 # raw stacked = RAW_A * step
    obj_dim = int(o0_tr.shape[-1])
    dobj_tr = o1_tr - o0_tr
    dobj_va = o1_va - o0_va
    obj_scale = float(dobj_tr.norm(dim=-1).std().clamp(min=1e-6))
    mean_action = a_tr.mean(dim=0)                    # constant-mean baseline
    moves_tr = (dobj_tr.norm(dim=-1) > args.move_thresh).float()
    moves_va = (dobj_va.norm(dim=-1) > args.move_thresh).float()

    head = InverseProposalHead(latent_dim=latent_dim, action_dim=action_dim,
                               obj_dim=obj_dim, hidden=args.hidden,
                               obj_emb_dim=args.obj_emb_dim,
                               obj_scale=obj_scale).to(device).train()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"inverse head params: {n_params/1e6:.2f}M  (latent_dim={latent_dim} "
          f"action_dim={action_dim} obj_dim={obj_dim} obj_scale={obj_scale:.4f})  "
          f"move_thresh={args.move_thresh} contact_weight={args.contact_weight} "
          f"contact_frac={float(moves_tr.mean()):.3f}", flush=True)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else out_dir / f"inverse_proposal_{args.model}.pt"

    @torch.no_grad()
    def val_metrics():
        head.eval()
        ve = vce = vcn = spread = 0.0
        for lo in range(0, z_va.shape[0], args.batch_size):
            sl = slice(lo, lo + args.batch_size)
            z = z_va[sl].to(device); a = a_va[sl].to(device)
            dj = dobj_va[sl].to(device); mv = moves_va[sl].to(device)
            se_i = ((head(z, dj) - a) ** 2).mean(dim=-1)
            spread_i = (head(z, dj) - head(z, torch.zeros_like(dj))).norm(dim=-1)
            ve += float(se_i.sum())
            vce += float((se_i * mv).sum()); vcn += float(mv.sum())
            spread += float((spread_i * mv).sum())
        head.train()
        n = z_va.shape[0]
        return ve / max(n, 1), vce / max(vcn, 1), spread / max(vcn, 1)

    def save(v_mse, v_cmse, v_spread, base_mse):
        torch.save({
            "model": args.model, "latent_dim": latent_dim, "action_dim": action_dim,
            "obj_dim": obj_dim, "hidden": args.hidden, "obj_emb_dim": args.obj_emb_dim,
            "obj_scale": obj_scale, "state_dict": head.state_dict(),
            "val_action_mse": v_mse, "val_contact_action_mse": v_cmse,
            "baseline_action_mse": base_mse, "contact_action_spread": v_spread,
            "move_thresh": args.move_thresh, "contact_weight": args.contact_weight,
        }, path)

    base_mse = float(((a_va - mean_action) ** 2).mean(dim=-1).mean())
    N = z_tr.shape[0]
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        head.train()
        order = rng.permutation(N)
        sw = wsum = 0.0
        for lo in range(0, N, args.batch_size):
            idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
            z = z_tr[idx].to(device); a = a_tr[idx].to(device)
            dj = dobj_tr[idx].to(device); mv = moves_tr[idx].to(device)
            se_i = ((head(z, dj) - a) ** 2).mean(dim=-1)
            w = 1.0 + args.contact_weight * mv
            loss = (w * se_i).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            sw += float((w * se_i).sum()); wsum += float(w.sum())
        v_mse, v_cmse, v_spread = val_metrics()
        save(v_mse, v_cmse, v_spread, base_mse)       # timeout-safe: latest always on disk
        print(f"epoch {ep+1}/{args.epochs}: train wMSE={sw/max(wsum,1):.5f} | "
              f"val MSE={v_mse:.5f} ({v_mse/max(base_mse,1e-9):.3f}x baseline) "
              f"contactMSE={v_cmse:.5f} contact_action_spread={v_spread:.4f}", flush=True)

    v_mse, v_cmse, v_spread = val_metrics()
    save(v_mse, v_cmse, v_spread, base_mse)
    print(f"\nwrote {path}\n"
          f"  val action-MSE = {v_mse:.5f}  (constant-mean baseline {base_mse:.5f}, "
          f"ratio {v_mse/max(base_mse,1e-9):.3f}x — < 1 means the inverse learned something)\n"
          f"  contact-subset action-MSE = {v_cmse:.5f}\n"
          f"  contact_action_spread = {v_spread:.4f}  "
          f"(>0: the inverse proposes a DIFFERENT action when object motion is "
          f"requested — the contact-creating signal CEM lacks)\n"
          f"  -> use as scripts/18 --inverse-head with arms l2inv/hdyninv "
          f"(seed the CEM mean; --probe must be the SPATIAL object probe).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
