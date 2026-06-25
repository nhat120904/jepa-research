"""Train a LoRA partial-unfreeze of the ViTPredictor with the counterfactual
object objective — model-side Rung A (docs/plans/2026-06-24-object-aware-predictor-design.md).

Option D (scripts/25) put the correction OUTSIDE the predictor (additive Δ), so the
corrected *readout* tracked the counterfactual object (cf-corr +0.37) but the
recursively-rolled latent was not dynamically consistent and success stayed 0/32,
saturating below the dyn-head teacher (0.68). Here the correction lives INSIDE the
predictor (LoRA in the transformer blocks), so the fixed dynamics propagate through
the rollout the planner searches. Encoder + predictor base weights stay frozen; only
the rank-r adapters train.

    ẑ(z,a) = Predictor_{frozen+LoRA}(z, a)      (LoRA enabled; one-step via adapter.predict)
    L = ‖ẑ − z_{t+1}‖²                                  (recon)
      + λ_obj · ‖g(ẑ(z,a))  − obj_{t+1}‖²              (FACTUAL object, spatial probe)
      + λ_cf  · ‖g(ẑ(z,a')) − (obj_t + h(z,a'))‖²      (COUNTERFACTUAL, dyn-head teacher)

Same loss family as Option D, but the trainable parameters are LoRA adapters inside
the predictor rather than an additive head — that is the ceiling-breaker. The cache
is materialised to RAM once (the 26 GB HDF5 is too slow to re-stream each epoch);
the per-batch cost is the predictor forward/backward on the GPU. A LoRA checkpoint is
saved every epoch (scripts/18/23/24 load it via `--predictor-lora`). GATE on
scripts/24 (V3-spatial corr) + scripts/23 (rollout fidelity) BEFORE any closed-loop.

    .venv/bin/python scripts/26_train_predictor_lora_cf.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --spatial-probe checkpoints/spatial_object_probe_dino_wm_metaworld.pt \
        --dyn-head checkpoints/object_dynamics_dino_wm_metaworld.pt \
        --epochs 8 --rank 8 --alpha 16 --lambda-obj 10 --lambda-cf 10
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
import torch.nn as nn  # noqa: E402
from models.heads.lora_predictor import (inject_lora, set_lora_enabled,  # noqa: E402
                                         lora_state_dict, _predictor_of)
from models.probes import load_probe, load_dynamics_head  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def predict_grad(adapter, z_t, a_t, proprio_t=None):
    """Grad-enabled clone of ``EncPredWMAdapter.predict`` (which is ``@torch.no_grad``,
    so the LoRA could never train through it). Drives the model through
    ``encpred.unroll`` under ``enable_grad`` so gradients reach the predictor's LoRA
    params; action/proprio normalisation stay detached (they are not trainable)."""
    from einops import rearrange
    from tensordict.tensordict import TensorDict
    dev = adapter.device
    B = z_t.shape[0]
    z_t = z_t.to(dev, torch.float32)
    a = a_t.to(dev, torch.float32).reshape(B, -1, adapter.spec.action_dim)
    a = adapter.normalize_action(a).reshape(B, -1, adapter._model_action_dim)
    act_suffix = rearrange(a, "b t a -> t b a")
    z_ctxt_visual = z_t.unsqueeze(1)
    with torch.enable_grad():
        if adapter.spec.uses_proprio and proprio_t is not None:
            prop_feat = adapter.encode_proprio_features(proprio_t.reshape(B, 1, -1))
            z_ctxt = TensorDict({"visual": z_ctxt_visual, "proprio": prop_feat}, batch_size=[])
            pred = adapter.encpred.unroll(z_ctxt, act_suffix=act_suffix)["visual"]
        else:
            pred = adapter.encpred.unroll(z_ctxt_visual, act_suffix=act_suffix)
    return pred[-1]


def object_saliency_mask(z, probe, k):
    """Zero the top-k object-salient patch tokens of the context ``z`` (B,1,H,W,D).
    The frozen spatial probe ``g`` localises the object: tokens with the largest
    ‖∂‖g(z)‖²/∂token‖ are the ones encoding object position. Masking them removes the
    'copy the object straight through' shortcut, so the corrected rollout must infer
    the object's future from the remaining context + the action — the teacher-FREE,
    ground-truth-supervised half of Rung A (design doc §2, λ_msk term). No camera
    calibration needed (the deleted-script blocker); the probe is the localiser."""
    B, T, H, W, D = z.shape
    zr = z.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        s = probe(zr).pow(2).sum()
    grad, = torch.autograd.grad(s, zr)
    sal = grad.reshape(B, H * W, D).norm(dim=-1)               # (B, HW) per-token saliency
    idx = sal.topk(min(k, H * W), dim=1).indices              # (B, k) object tokens
    zm = z.reshape(B, H * W, D).clone()
    zm.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, D), 0.0)   # mask -> 0
    return zm.reshape(B, T, H, W, D)


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
    ap.add_argument("--spatial-probe", required=True)
    ap.add_argument("--dyn-head", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--load-chunk", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lambda-obj", type=float, default=10.0)
    ap.add_argument("--lambda-cf", type=float, default=10.0)
    ap.add_argument("--lambda-msk", type=float, default=0.0,
                    help="weight of the teacher-FREE masked-object term (design §2 "
                         "λ_msk): mask the object's own tokens in the context and "
                         "require the rollout to recover obj_{t+1} from context+action. "
                         "0 = off (cf-distillation only, teacher-ceilinged).")
    ap.add_argument("--mask-topk", type=int, default=12,
                    help="number of object-salient patch tokens to mask for λ_msk.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "4")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model,
                                   cfg["dataset"]["name"])
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    a_dim = adapter.action_dim()
    ma_dim = adapter._model_action_dim
    # Freeze EVERYTHING, then inject LoRA (only the adapters will train).
    for p in adapter.encpred.parameters():
        p.requires_grad_(False)
    injected = inject_lora(adapter, r=args.rank, alpha=args.alpha)
    n_lora = sum(p.numel() for m in injected for p in (m.A, m.B))
    print(f"LoRA: {len(injected)} adapters, {n_lora/1e6:.3f}M trainable params "
          f"(rank={args.rank} alpha={args.alpha}); predictor base frozen", flush=True)

    probe, probe_meta = load_probe(args.spatial_probe, device)
    if probe_meta.get("probe_type") != "spatial":
        print("WARNING: --spatial-probe is not a spatial probe (needs the ungameable "
              "spatial readout, as in option D).", flush=True)
    dyn_head, dyn_meta = load_dynamics_head(args.dyn_head, device)
    for p in dyn_head.parameters():
        p.requires_grad_(False)
    print(f"teacher dyn-head: cf_corr={dyn_meta.get('cf_corr'):.3f}", flush=True)

    regime_by_traj = read_regimes(cache_path)

    def to_model_action(a_raw: torch.Tensor) -> torch.Tensor:
        B = a_raw.shape[0]
        a = adapter.normalize_action(a_raw.to(device).float().reshape(B, -1, a_dim))
        return a.reshape(B, ma_dim)

    # ---- materialise to RAM once ----
    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=True)
        train_recs, val_recs = split_by_trajectory(records, args.val_frac, args.seed)
        print(f"transitions: train={len(train_recs)} val={len(val_recs)}", flush=True)
        uses_prop = adapter.uses_proprio()

        def load_all(recs):
            zt, zt1, at, pt, o0, o1 = [], [], [], [], [], []
            for lo in range(0, len(recs), args.load_chunk):
                d = helpers.materialize_records(cache, recs[lo: lo + args.load_chunk], step,
                                                want_proprio=uses_prop, want_state=True)
                zt.append(d["z_t"].clone()); zt1.append(d["z_t1"].clone())
                at.append(d["a_t"].float().clone())
                pt.append(d["proprio_t"].clone() if d.get("proprio_t") is not None else None)
                o0.append(d["state_t"][:, OBJECT_SLICE].float().clone())
                o1.append(d["state_t1"][:, OBJECT_SLICE].float().clone())
                del d
            prop = torch.cat(pt) if pt and pt[0] is not None else None
            return (torch.cat(zt), torch.cat(zt1), torch.cat(at), prop,
                    torch.cat(o0), torch.cat(o1))

        z_tr, z1_tr, a_tr, p_tr, o0_tr, o1_tr = load_all(train_recs)
        z_va, z1_va, a_va, p_va, o0_va, o1_va = load_all(val_recs)
    latent_dim = int(z_tr.shape[-1])
    print(f"materialised: train z {tuple(z_tr.shape)} val z {tuple(z_va.shape)}", flush=True)

    opt = torch.optim.Adam((p for m in injected for p in (m.A, m.B)), lr=args.lr)

    def batch_losses(idx_t, z, z1, a, prop, o0, o1, train):
        z = z.to(device).float()
        base = predict_grad(adapter, z, a, proprio_t=prop)           # grad-enabled LoRA rollout
        recon = ((base - z1.to(device).float()) ** 2).mean()
        obj_fac = ((probe(base) - o1.to(device).float()) ** 2).mean()
        B = z.shape[0]
        perm = torch.randperm(B) if train else torch.roll(torch.arange(B), 1)
        a_cf = a[perm]
        base_cf = predict_grad(adapter, z, a_cf, proprio_t=prop)
        # SPREAD-PRESERVING cf loss: match the *change* in object prediction when the
        # action is swapped a->a', not the absolute position. Mean-L2 to absolute teacher
        # (the old form) lowers the average gap by regressing toward the per-anchor mean,
        # which FLATTENS action-dependence -> recreates Boundary Blindness at the fix layer
        # (22388: internal cf 0.36x but V3-spatial corr only 0.24). The V3 gate reads the
        # ordering of action-conditioned object *spread*; matching deltas targets it directly.
        with torch.no_grad():
            teacher_delta = (dyn_head(z, to_model_action(a_cf))
                             - dyn_head(z, to_model_action(a)))
        pred_delta = probe(base_cf) - probe(base)
        obj_cf = ((pred_delta - teacher_delta) ** 2).mean()
        loss = recon + args.lambda_obj * obj_fac + args.lambda_cf * obj_cf
        # TEACHER-FREE masked-object term (design §2, λ_msk): mask the object's own
        # tokens in the context, then require the rollout to still recover the REAL
        # observed obj_{t+1}. Ground-truth supervised -> not bounded by the dyn-head
        # teacher (which already scores push 0/16); this is the ceiling-breaker.
        obj_msk = 0.0
        if args.lambda_msk > 0.0:
            z_mask = object_saliency_mask(z, probe, args.mask_topk)
            base_msk = predict_grad(adapter, z_mask, a, proprio_t=prop)
            msk = ((probe(base_msk) - o1.to(device).float()) ** 2).mean()
            loss = loss + args.lambda_msk * msk
            obj_msk = float(msk)
        return loss, float(recon), float(obj_fac), float(obj_cf), obj_msk

    def run_split(z, z1, a, prop, o0, o1, train):
        N = z.shape[0]
        order = np.random.default_rng(0 if not train else None).permutation(N)
        sr = sf = sc = sm = n = 0.0
        for lo in range(0, N, args.batch_size):
            idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
            if len(idx) < 2:
                continue
            pr = prop[idx].to(device) if prop is not None else None
            loss, r, f, c, m = batch_losses(idx, z[idx], z1[idx], a[idx], pr,
                                            o0[idx], o1[idx], train)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            sr += r * len(idx); sf += f * len(idx); sc += c * len(idx)
            sm += m * len(idx); n += len(idx)
        return sr / max(n, 1), sf / max(n, 1), sc / max(n, 1), sm / max(n, 1)

    # frozen baseline (LoRA disabled) — the numbers the fix must move
    set_lora_enabled(injected, False)
    with torch.no_grad():
        b_r, b_f, b_c, b_m = run_split(z_va, z1_va, a_va, p_va, o0_va, o1_va, False)
    set_lora_enabled(injected, True)
    msk_on = args.lambda_msk > 0.0
    print(f"frozen baseline (LoRA off): val recon={b_r:.5f} obj_fac={b_f:.6f} "
          f"obj_cf={b_c:.6f}" + (f" obj_msk={b_m:.6f}" if msk_on else ""), flush=True)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else out_dir / f"predictor_lora_{args.model}.pt"

    def save(vr, vf, vc, vm):
        torch.save({
            "model": args.model, "r": args.rank, "alpha": args.alpha,
            "target_substrs": ["layers", "blocks"], "lora": lora_state_dict(injected, adapter),
            "latent_dim": latent_dim, "lambda_obj": args.lambda_obj, "lambda_cf": args.lambda_cf,
            "lambda_msk": args.lambda_msk, "mask_topk": args.mask_topk,
            "frozen_obj_fac": b_f, "corrected_obj_fac": vf,
            "frozen_obj_cf": b_c, "corrected_obj_cf": vc, "frozen_recon": b_r, "corrected_recon": vr,
            "frozen_obj_msk": b_m, "corrected_obj_msk": vm,
            "teacher_cf_corr": dyn_meta.get("cf_corr"),
        }, path)

    # ---- fail-fast grad-flow guard (post-mortem of 22378: a silent zero-grad gives
    # a fake null — val byte-identical to frozen across all epochs). Verify dL/dB != 0
    # on one batch BEFORE spending epochs; if zero, dump the predictor structure. ----
    set_lora_enabled(injected, True)
    g_idx = torch.arange(min(args.batch_size, z_tr.shape[0]), dtype=torch.long)
    g_pr = p_tr[g_idx].to(device) if p_tr is not None else None
    g_loss, *_ = batch_losses(g_idx, z_tr[g_idx], z1_tr[g_idx], a_tr[g_idx], g_pr,
                              o0_tr[g_idx], o1_tr[g_idx], True)
    opt.zero_grad(); g_loss.backward()
    gB = sum(float(m.B.grad.abs().sum()) for m in injected if m.B.grad is not None)
    gA = sum(float(m.A.grad.abs().sum()) for m in injected if m.A.grad is not None)
    print(f"grad-flow check: sum|dL/dB|={gB:.3e} sum|dL/dA|={gA:.3e} "
          f"loss={float(g_loss):.5f} base_frozen={not injected[0].base.weight.requires_grad}",
          flush=True)
    if gB == 0.0:
        lins = [n for n, m in _predictor_of(adapter).named_modules()
                if isinstance(m, (nn.Linear,)) or type(m).__name__ == "LoRALinear"]
        print(f"ZERO gradient to LoRA. Predictor Linear/LoRA modules ({len(lins)}):",
              flush=True)
        for nm in lins:
            print("   ", nm, flush=True)
        raise SystemExit("LoRA receives no gradient — fix the injection target before training.")
    opt.zero_grad()

    vr = vf = vc = vm = float("nan")
    for ep in range(args.epochs):
        tr, tf, tc, tm = run_split(z_tr, z1_tr, a_tr, p_tr, o0_tr, o1_tr, True)
        with torch.no_grad():
            vr, vf, vc, vm = run_split(z_va, z1_va, a_va, p_va, o0_va, o1_va, False)
        save(vr, vf, vc, vm)        # timeout-safe
        msk_str = (f" msk={tm:.6f}|{vm:.6f} ({vm/max(b_m,1e-9):.3f}x)") if msk_on else ""
        print(f"epoch {ep+1}/{args.epochs}: train recon={tr:.5f} fac={tf:.6f} cf={tc:.6f} | "
              f"val recon={vr:.5f} fac={vf:.6f} cf={vc:.6f} "
              f"(fac {vf/max(b_f,1e-9):.3f}x cf {vc/max(b_c,1e-9):.3f}x)" + msk_str, flush=True)

    print(f"\nwrote {path}\n"
          f"  factual object error corrected/frozen = {vf/max(b_f,1e-9):.3f}x ({b_f:.6f}->{vf:.6f})\n"
          f"  counterfactual-to-teacher gap corrected/frozen = {vc/max(b_c,1e-9):.3f}x "
          f"({b_c:.6f}->{vc:.6f})\n"
          f"  -> GATE on scripts/24 --predictor-lora {path} (cf-corr target >=0.5, beating "
          f"option-D's 0.37) + scripts/23 (factual drift) BEFORE closed-loop.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
