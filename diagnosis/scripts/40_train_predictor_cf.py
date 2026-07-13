"""Train a predictor-LoRA with the proposal's LATENT-SPACE counterfactual objective
(cai_jepa_paper_proposal.md §6, Contribution 3) — the beat-baseline method that the
oracle-ladder / cost-side program never tested.

Everything in Phase 0-G swapped only the planning COST under PERFECT latent dynamics
(the predictor F removed), and every cost was reward-hacked. This script attacks the
OTHER axis the oracle deliberately bypassed: the predictor's action-grounding. Unlike
scripts/26 (which distills a MetaWorld object-dynamics TEACHER and needs GT object
state), this objective is PURE LATENT and dataset-agnostic, so it runs on DROID —
whose published Action-Score is the beat-baseline arena (CRA_eff ≈ chance everywhere
today = the headroom).

    ẑ(z,a) = Predictor_{frozen+LoRA}(z, a)                    (LoRA enabled)
    d(a·)  = ‖ẑ(z,a·) − sg(z_{t+1})‖²   (mean over features; STOP-GRAD on the target)
    L_pred = mean_b d(a)                                       (recon: keep dynamics accurate)
    L_cf   = CE( softmax_a·[ −d(a·)/τ ], factual=0 )           (InfoNCE OVER ACTIONS)
    L      = L_pred + λ_cf · L_cf

L_cf is the proposal's counterfactual objective in its InfoNCE form: among the factual
action and K counterfactual actions (other actions applied to the SAME z_t), the
predictor must place the FACTUAL action's predicted latent closest to the true next
latent. This is exactly what the L2-CEM planner needs — pick the action whose predicted
rollout lands nearest the goal latent — so improving it should move DROID Action-Score
directly. The stop-gradient on z_{t+1} is essential (else the loss collapses the target
instead of grounding the prediction — proposal §6.2). Negatives are in-batch permuted
actions (random-over-ACTIONS, the axis that fixes action-grounding — not the
random-over-STATES axis of C-SWM/TWISTER). τ auto-scales to the batch's factual
distance so no per-model temperature tuning is needed.

Encoder + predictor base weights stay frozen; only the rank-r LoRA adapters train. The
checkpoint is `load_predictor_lora`-compatible → scripts/08 `--predictor-lora` runs the
Action-Score gate with it. GATE on the held-out cf-rank accuracy (does the predictor
now rank the factual action first?) before the expensive planning eval.

    .venv/bin/python scripts/40_train_predictor_cf.py \
        --config configs/diagnostic_droid.yaml --model dino_wm_droid \
        --epochs 8 --rank 8 --alpha 16 --lambda-cf 1.0 --num-neg 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from tensordict.tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (LatentCache, build_trajectory_manifest, filter_records,  # noqa: E402
                  latent_cache_path, read_regimes, write_manifest_once)
from models.adapters import build_adapter  # noqa: E402
from models.heads.lora_predictor import (inject_lora, set_lora_enabled,  # noqa: E402
                                         lora_state_dict)
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402


def predict_grad(adapter, z_t, a_t, proprio_t=None):
    """Grad-enabled clone of ``EncPredWMAdapter.predict`` (which is ``@torch.no_grad``,
    so LoRA could never train through it): drive the model through ``encpred.unroll``
    under ``enable_grad`` so gradients reach the predictor LoRA; action/proprio
    normalisation stay detached. Returns the one-step predicted latent (B, *frame)."""
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


def cf_infonce_loss(d_fac: torch.Tensor, d_neg: torch.Tensor, temp: float | None = None):
    """The counterfactual objective as InfoNCE over actions. Pure tensor (offline-testable).

    ``d_fac`` : (B,)   squared latent distance ‖F(z,a)−z_{t+1}‖² for the FACTUAL action.
    ``d_neg`` : (B,K)  the same distance for K counterfactual actions (same z_t).
    returns   : (loss, rank_acc, temp) where loss = CE(factual is closest), rank_acc =
                fraction of anchors whose factual action is already ranked first (a
                CRA-like train signal), temp = the temperature actually used.

    τ auto-scales to ``mean(d_fac)`` (detached) when not given, so the softmax sees
    O(1) logits regardless of the model's absolute latent-MSE scale."""
    if temp is None:
        temp = float(d_fac.mean().detach().clamp_min(1e-8))
    logits = -torch.cat([d_fac[:, None], d_neg], dim=1) / temp     # (B, 1+K), factual = col 0
    labels = torch.zeros(d_fac.shape[0], dtype=torch.long, device=d_fac.device)
    loss = F.cross_entropy(logits, labels)
    rank_acc = float((logits.argmax(dim=1) == 0).float().mean())
    return loss, rank_acc, temp


def split_by_trajectory(records, val_frac, test_frac, seed, *, dataset, model):
    """Return train/val/test records plus their immutable manifest payload."""
    manifest = build_trajectory_manifest(
        (r["tid"] for r in records), seed=seed, dataset=dataset, model=model,
        val_frac=val_frac, test_frac=test_frac,
    )
    return (
        filter_records(records, manifest, "train"),
        filter_records(records, manifest, "val"),
        filter_records(records, manifest, "test"),
        manifest,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--load-chunk", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lambda-cf", type=float, default=1.0,
                    help="weight of the InfoNCE-over-actions counterfactual term")
    ap.add_argument("--num-neg", type=int, default=4,
                    help="counterfactual actions per anchor (in-batch permutations)")
    ap.add_argument("--temp", type=float, default=None,
                    help="softmax temperature; default = auto (mean factual distance)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="trajectory split seed, intentionally independent of training seed")
    ap.add_argument("--split-manifest", default=None,
                    help="immutable JSON manifest path; default is <checkpoint>.split.json")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "4")))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model,
                                   cfg["dataset"]["name"])
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else out_dir / f"predictor_cf_{args.model}.pt"
    manifest_path = (Path(args.split_manifest) if args.split_manifest
                     else path.with_suffix(path.suffix + ".split.json"))
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    for p in adapter.encpred.parameters():
        p.requires_grad_(False)
    injected = inject_lora(adapter, r=args.rank, alpha=args.alpha)
    n_lora = sum(p.numel() for m in injected for p in (m.A, m.B))
    print(f"LoRA: {len(injected)} adapters, {n_lora/1e6:.3f}M trainable params "
          f"(rank={args.rank} alpha={args.alpha}); predictor base frozen", flush=True)

    regime_by_traj = read_regimes(cache_path)
    uses_prop = adapter.uses_proprio()

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=False)
        train_recs, val_recs, test_recs, split_manifest = split_by_trajectory(
            records, args.val_frac, args.test_frac, args.split_seed,
            dataset=cfg["dataset"]["name"], model=args.model,
        )
        write_manifest_once(manifest_path, split_manifest)
        split_counts = {
            "trajectories": {name: len(split_manifest["splits"][name])
                             for name in ("train", "val", "test")},
            "transitions": {"train": len(train_recs), "val": len(val_recs),
                            "test": len(test_recs)},
        }
        print(f"split manifest: {manifest_path} sha256="
              f"{split_manifest['manifest_sha256']}", flush=True)
        print(f"transitions: train={len(train_recs)} val={len(val_recs)} "
              f"test(UNTOUCHED)={len(test_recs)} trajectories={split_counts['trajectories']} "
              f"(uses_proprio={uses_prop})", flush=True)

        def load_all(recs):
            zt, zt1, at, pt = [], [], [], []
            for lo in range(0, len(recs), args.load_chunk):
                d = helpers.materialize_records(cache, recs[lo: lo + args.load_chunk], step,
                                                want_proprio=uses_prop, want_state=False)
                zt.append(d["z_t"].clone()); zt1.append(d["z_t1"].clone())
                at.append(d["a_t"].float().clone())
                pt.append(d["proprio_t"].clone() if d.get("proprio_t") is not None else None)
                del d
            prop = torch.cat(pt) if pt and pt[0] is not None else None
            return torch.cat(zt), torch.cat(zt1), torch.cat(at), prop

        z_tr, z1_tr, a_tr, p_tr = load_all(train_recs)
        z_va, z1_va, a_va, p_va = load_all(val_recs)
    latent_dim = int(z_tr.shape[-1])
    print(f"materialised: train z {tuple(z_tr.shape)} val z {tuple(z_va.shape)}", flush=True)

    opt = torch.optim.Adam((p for m in injected for p in (m.A, m.B)), lr=args.lr)

    def batch_losses(z, z1, a, prop, train):
        z = z.to(device).float()
        target = z1.to(device).float()
        B = z.shape[0]
        pred_fac = predict_grad(adapter, z, a, proprio_t=prop)
        # STOP-GRADIENT on the true next latent (proposal §6.2): the target must not
        # move — the prediction is pulled to it, not vice-versa.
        tgt = target.reshape(B, -1).detach()
        d_fac = ((pred_fac.reshape(B, -1) - tgt) ** 2).mean(-1)          # (B,)
        d_negs = []
        for k in range(args.num_neg):
            # perm indexes `a` (kept on CPU; predict_grad moves it to device itself),
            # so it must live on a.device — not z.device (already on the GPU).
            if train:
                perm = torch.randperm(B, device=a.device)
            else:
                perm = torch.roll(torch.arange(B, device=a.device), k + 1)
            pred_cf = predict_grad(adapter, z, a[perm], proprio_t=prop)
            d_negs.append(((pred_cf.reshape(B, -1) - tgt) ** 2).mean(-1))
        d_neg = torch.stack(d_negs, dim=1)                              # (B, K)
        l_cf, rank_acc, _ = cf_infonce_loss(d_fac, d_neg, temp=args.temp)
        recon = d_fac.mean()
        loss = recon + args.lambda_cf * l_cf
        return loss, float(recon), float(l_cf), rank_acc

    train_order_rng = np.random.default_rng(args.seed)

    def run_split(z, z1, a, prop, train):
        N = z.shape[0]
        order = (train_order_rng if train else np.random.default_rng(0)).permutation(N)
        sr = sc = sa = n = 0.0
        for lo in range(0, N, args.batch_size):
            idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
            if len(idx) < 2:
                continue
            pr = prop[idx].to(device) if prop is not None else None
            loss, r, c, ra = batch_losses(z[idx], z1[idx], a[idx], pr, train)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            sr += r * len(idx); sc += c * len(idx); sa += ra * len(idx); n += len(idx)
        return sr / max(n, 1), sc / max(n, 1), sa / max(n, 1)

    # frozen baseline (LoRA off) — the numbers the fix must move.
    set_lora_enabled(injected, False)
    with torch.no_grad():
        b_r, b_c, b_a = run_split(z_va, z1_va, a_va, p_va, False)
    set_lora_enabled(injected, True)
    print(f"frozen baseline (LoRA off): val recon={b_r:.6f} cf_ce={b_c:.4f} "
          f"cf_rank_acc={b_a:.3f} (chance={1.0/(1+args.num_neg):.3f})", flush=True)

    def save(vr, vc, va, *, epoch, val_objective):
        torch.save({
            "model": args.model, "r": args.rank, "alpha": args.alpha,
            "target_substrs": ["layers", "blocks"], "lora": lora_state_dict(injected, adapter),
            "latent_dim": latent_dim, "lambda_cf": args.lambda_cf, "num_neg": args.num_neg,
            "objective": "latent_cf_infonce",
            "frozen_recon": b_r, "corrected_recon": vr,
            "frozen_cf_ce": b_c, "corrected_cf_ce": vc,
            "frozen_cf_rank_acc": b_a, "corrected_cf_rank_acc": va,
            "selection": {"criterion": "val_objective", "mode": "min",
                          "best_epoch": int(epoch), "best_value": float(val_objective)},
            "data_split": {
                "manifest": split_manifest,
                "manifest_path": str(manifest_path),
                "manifest_sha256": split_manifest["manifest_sha256"],
                "counts": split_counts,
                "train_split": "train", "tuning_split": "val",
                "reserved_evaluation_split": "test",
            },
        }, path)

    # fail-fast grad-flow guard (post-mortem of 22378: a silent zero-grad gives a fake
    # null). Verify dL/dB != 0 on one batch BEFORE spending epochs.
    set_lora_enabled(injected, True)
    g = torch.arange(min(args.batch_size, z_tr.shape[0]), dtype=torch.long)
    g_pr = p_tr[g].to(device) if p_tr is not None else None
    g_loss, *_ = batch_losses(z_tr[g], z1_tr[g], a_tr[g], g_pr, True)
    opt.zero_grad(); g_loss.backward()
    gB = sum(float(m.B.grad.abs().sum()) for m in injected if m.B.grad is not None)
    if gB == 0.0:
        raise SystemExit("LoRA receives no gradient — fix the injection target before training.")
    print(f"grad-flow check: sum|dL/dB|={gB:.3e} loss={float(g_loss):.5f}", flush=True)
    opt.zero_grad()

    vr = vc = va = float("nan")
    best = None
    best_val_objective = float("inf")
    for ep in range(args.epochs):
        tr, tc, ta = run_split(z_tr, z1_tr, a_tr, p_tr, True)
        with torch.no_grad():
            vr, vc, va = run_split(z_va, z1_va, a_va, p_va, False)
        val_objective = vr + args.lambda_cf * vc
        if val_objective < best_val_objective:
            best_val_objective = val_objective
            best = (vr, vc, va, ep + 1)
            save(vr, vc, va, epoch=ep + 1, val_objective=val_objective)
        print(f"epoch {ep+1}/{args.epochs}: train recon={tr:.6f} cf_ce={tc:.4f} rank={ta:.3f} | "
              f"val recon={vr:.6f} cf_ce={vc:.4f} rank_acc={va:.3f} "
              f"objective={val_objective:.5f} (recon {vr/max(b_r,1e-9):.3f}x)"
              f"{' [BEST]' if best and best[3] == ep + 1 else ''}", flush=True)

    assert best is not None
    vr, vc, va, best_epoch = best
    print(f"\nwrote best validation-selected checkpoint {path} (epoch {best_epoch})\n"
          f"  immutable split: {manifest_path} (test trajectories untouched during training)\n"
          f"  cf-rank accuracy frozen->corrected = {b_a:.3f} -> {va:.3f} "
          f"(chance {1.0/(1+args.num_neg):.3f})\n"
          f"  factual recon corrected/frozen = {vr/max(b_r,1e-9):.3f}x ({b_r:.6f}->{vr:.6f})\n"
          f"  -> GATE: cf_rank_acc must rise well above frozen/chance AND recon must not\n"
          f"     blow up, THEN run scripts/08 --predictor-lora {path.name} "
          f"--eval-split test for held-out Action-Score.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
