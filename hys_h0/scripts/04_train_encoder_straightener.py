"""HyS-JEPA, full form: temporal straightening with a TRAINABLE encoder.

The frozen-projector form was refuted (`hys_h0/docs/VERDICT_FROZEN.md`): a matched random
drop beat contact-aligned gating on all three seeds. The one mechanism that could change
under fine-tuning is specific -- a projector can only SELECT a subspace of frozen DINOv2
features, so when temporal straightness and object precision are in tension inside that
space it must trade one away (object decodability went 2.9 cm -> 6.6-7.6 cm against a 5 cm
success radius). A trainable encoder can CREATE features instead of selecting them.

Arms (`--gate`), matching the frozen series so the two are readable side by side:
  off     prediction-only, no curvature term (control: does straightening add anything?)
  none    global straightening
  switch  straightening skipped at contact-mode switches (the HyS-JEPA proposal)
  random  straightening skipped on a MATCHED random fraction (the control that decides
          whether contact SEMANTICS matter -- it beat `switch` in the frozen form)

Health gate, checked before any planner run: does straightening still cost object precision?
If a fine-tuned encoder straightens while KEEPING object decode near the frozen ~2.9 cm, the
create-vs-select hypothesis is confirmed at the mechanism level. If object decode still
collapses to ~6.6 cm, the main reason to hope for this rung is gone.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parent.parent.parent / "diagnosis"
sys.path.insert(0, str(REPO))

from models.adapters import build_adapter  # noqa: E402
from models.heads.lora_encoder import (  # noqa: E402
    encoder_lora_state_dict, inject_encoder_lora)
from models.heads.straightening_projector import (  # noqa: E402
    curvature_loss, effective_rank, vicreg_terms)
from stratification.metaworld_regimes import (  # noqa: E402
    OBJECT_SLICE, classify_metaworld_regime)


def _load_script38():
    """Reuse script 38's raw-frame loader and grad-enabled encode verbatim."""
    p = REPO / "scripts" / "38_train_encoder_lora.py"
    spec = _ilu.spec_from_file_location("_s38", p)
    m = _ilu.module_from_spec(spec)
    sys.modules["_s38"] = m
    spec.loader.exec_module(m)
    return m


def pool(z: torch.Tensor) -> torch.Tensor:
    """(B, *frame) patch-token latent -> (B, 2D) mean+max over tokens."""
    D = z.shape[-1]
    t = z.reshape(z.shape[0], -1, D)
    return torch.cat([t.mean(1), t.amax(1)], dim=-1)


def regime_ids(states: np.ndarray) -> np.ndarray:
    """Per-transition regime label from the 39-dim state (the same proxy used throughout)."""
    from stratification import REGIMES
    idx = {r: i for i, r in enumerate(REGIMES)}
    return np.asarray([idx[classify_metaworld_regime(states[t], states[t + 1])]
                       for t in range(len(states) - 1)], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="dino_wm_metaworld")
    ap.add_argument("--tasks", nargs="*", default=["mw-push"])
    ap.add_argument("--gate", choices=["off", "none", "switch", "random"], required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-curve", type=float, default=10.0)
    ap.add_argument("--lambda-pred", type=float, default=10.0)
    ap.add_argument("--lambda-preserve", type=float, default=0.05,
                    help="anchor to the frozen encoder; with a trainable encoder the "
                         "curvature objective has a trivial collapse optimum, so this "
                         "matters MORE than it did for the projector")
    ap.add_argument("--lambda-var", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4, help="windows per step")
    ap.add_argument("--max-trajs-per-task", type=int, default=60)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-lora", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config))
    cfg["dataset"]["tasks"] = {"easy": args.tasks, "medium": [], "hard": []}

    s38 = _load_script38()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()

    trajs = s38.load_expert_trajs(cfg, args.max_trajs_per_task)
    print(f"gate={args.gate} seed={args.seed} trajs={len(trajs)} device={device}", flush=True)

    # windows that stay inside one trajectory, split by trajectory
    order = rng.permutation(len(trajs))
    n_val = max(1, int(round(args.holdout_frac * len(trajs))))
    val_t, train_t = set(order[:n_val].tolist()), set(order[n_val:].tolist())
    wins = {"train": [], "val": []}
    regs = []
    for ti, tr in enumerate(trajs):
        st = tr["obj"].numpy()
        full = tr.get("state")
        r = (regime_ids(full.numpy()) if full is not None
             else (np.linalg.norm(np.diff(st, axis=0), axis=-1) > 0.005).astype(np.int64))
        regs.append(r)
        T = tr["frames"].shape[0]
        for s in range(0, T - args.window + 1):
            wins["val" if ti in val_t else "train"].append((ti, s))
    switch_rate = float(np.mean([np.mean(r[:-1] != r[1:]) for r in regs if len(r) > 1]))
    print(f"  windows train={len(wins['train'])} val={len(wins['val'])} "
          f"switch_rate={switch_rate:.4f}", flush=True)

    injected = inject_encoder_lora(adapter, r=args.lora_r, alpha=args.lora_alpha)

    # The frozen encoder output is CONSTANT, so the preservation anchor does not need a
    # second forward pass per step. Cache it once; this halves the encoding cost.
    frozen_cache: dict = {}
    params = [p for lin in injected for p in (lin.A, lin.B)]
    D = None
    pred = None
    opt = None

    def encode_window(ti, s, grad=True):
        tr = trajs[ti]
        f = tr["frames"][s:s + args.window].float()
        p = tr["prop"][s:s + args.window]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            z = s38.encode_grad(adapter, f, p)         # (win, *frame)
        # window of W frames -> W-1 regime labels -> W-2 switch flags, which is exactly
        # the number of curvature triples. Slicing W-2 labels here was an off-by-one that
        # made the mask one short (job 44482 switch/random cells).
        return z, tr["action"][s:s + args.window - 1].to(device), \
            tr["obj"][s:s + args.window].to(device), regs[ti][s:s + args.window - 1]

    @torch.no_grad()
    def health(split="val", cap=48):
        P, Y, C = [], [], []
        for ti, s in wins[split][:cap]:
            z, _a, obj, _r = encode_window(ti, s, grad=False)
            p_ = pool(z)
            P.append(p_.cpu()); Y.append(obj.cpu())
            c, _ = curvature_loss(p_.unsqueeze(0)); C.append(float(c))
        if not P:
            return {}
        X = torch.cat(P).numpy(); Yv = torch.cat(Y).numpy()
        k = int(0.7 * len(X))
        X1 = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
        A = X1[:k].T @ X1[:k] + 1e-3 * k * np.eye(X1.shape[1])
        W = np.linalg.solve(A, X1[:k].T @ Yv[:k])
        err = np.linalg.norm(X1[k:] @ W - Yv[k:], axis=-1)
        return {"curv": float(np.mean(C)),
                "obj_decode_median_cm": float(np.median(err) * 100),
                "eff_rank": effective_rank(torch.from_numpy(X).float())}

    h0 = health()
    print(f"  frozen baseline: curv={h0.get('curv', float('nan')):.4f} "
          f"obj={h0.get('obj_decode_median_cm', float('nan')):.2f}cm "
          f"rank={h0.get('eff_rank', float('nan')):.1f}", flush=True)

    hist = []
    for ep in range(args.epochs):
        idx = rng.permutation(len(wins["train"]))
        agg = {"curve": 0.0, "pred": 0.0, "pres": 0.0, "n": 0}
        for bi in range(0, len(idx), args.batch_size):
            zs, acts, swm = [], [], []
            for j in idx[bi:bi + args.batch_size]:
                ti, s = wins["train"][j]
                z, a, _obj, r = encode_window(ti, s)
                zs.append(pool(z)); acts.append(a)
                swm.append(torch.from_numpy((r[:-1] != r[1:]).astype(np.float32)))
            P = torch.stack(zs)                                   # (B, win, 2D)
            A_ = torch.stack(acts)                                # (B, win-1, A)
            SW = torch.stack(swm).to(device)                      # (B, win-2)

            if D is None:
                D = P.shape[-1]
                pred = nn.Sequential(nn.Linear(D + A_.shape[-1], 512), nn.GELU(),
                                     nn.Linear(512, D)).to(device)
                opt = torch.optim.AdamW(params + list(pred.parameters()),
                                        lr=args.lr, weight_decay=1e-4)

            if args.gate == "off":
                lc = torch.zeros((), device=device)
            else:
                if args.gate == "switch":
                    keep = 1.0 - SW
                elif args.gate == "random":
                    keep = (torch.rand_like(SW) >= switch_rate).float()
                else:
                    keep = None
                lc, _ = curvature_loss(P, keep)

            tgt = P[:, 1:]
            ph = P[:, :-1] + pred(torch.cat([P[:, :-1], A_], -1))
            lp = ((ph - tgt) ** 2).mean() / (tgt.detach().var() + 1e-6)

            # preservation anchor against the cached frozen encoding of the same windows.
            # With a trainable encoder the curvature objective has a trivial collapse
            # optimum (map every frame onto one moving point), so this anchor matters
            # more here than it did for the projector.
            need = [j for j in idx[bi:bi + args.batch_size]
                    if wins["train"][j] not in frozen_cache]
            if need:
                with torch.no_grad():
                    for lin in injected:
                        lin.enabled = False
                    for j in need:
                        ti, s_ = wins["train"][j]
                        zf = s38.encode_grad(
                            adapter,
                            trajs[ti]["frames"][s_:s_ + args.window].float(),
                            trajs[ti]["prop"][s_:s_ + args.window])
                        frozen_cache[(ti, s_)] = pool(zf).detach().cpu()
                    for lin in injected:
                        lin.enabled = True
            Pf = torch.stack([frozen_cache[wins["train"][j]]
                              for j in idx[bi:bi + args.batch_size]]).to(device)
            lpres = ((P - Pf) ** 2).mean() / (Pf.var() + 1e-6)

            lv, lcov = vicreg_terms(P.reshape(-1, P.shape[-1]))
            loss = (args.lambda_curve * lc + args.lambda_pred * lp
                    + args.lambda_preserve * lpres + args.lambda_var * lv + lcov)
            opt.zero_grad(); loss.backward(); opt.step()
            agg["curve"] += float(lc); agg["pred"] += float(lp)
            agg["pres"] += float(lpres); agg["n"] += 1

        h = health()
        hist.append({"epoch": ep, **h, "train_curve": agg["curve"] / max(agg["n"], 1)})
        print(f"  ep{ep:02d} curve={agg['curve']/max(agg['n'],1):.4f} "
              f"pred={agg['pred']/max(agg['n'],1):.4f} pres={agg['pres']/max(agg['n'],1):.4f} "
              f"| val curv={h.get('curv', float('nan')):.4f} "
              f"obj={h.get('obj_decode_median_cm', float('nan')):.2f}cm "
              f"rank={h.get('eff_rank', float('nan')):.1f}", flush=True)

    meta = {"gate": args.gate, "seed": args.seed, "tasks": args.tasks,
            "lora_r": args.lora_r, "switch_rate": switch_rate,
            "lambda_curve": args.lambda_curve, "lambda_preserve": args.lambda_preserve,
            "frozen_baseline": h0, "final": hist[-1] if hist else {}, "history": hist}
    Path(args.out_lora).parent.mkdir(parents=True, exist_ok=True)
    # key names must match models/heads/lora_encoder.load_encoder_lora (it reads
    # ckpt["lora"], ckpt["r"], ckpt["alpha"], ckpt["target_substrs"]) so that
    # scripts/30 --encoder-lora and scripts/51 --encoder-lora can load this directly.
    torch.save({"model": args.model, "r": args.lora_r, "alpha": args.lora_alpha,
                "target_substrs": ("blocks", "layers"),
                "lora": encoder_lora_state_dict(injected, adapter),
                "lambda_preserve": args.lambda_preserve, "meta": meta},
               args.out_lora)
    Path(str(args.out_lora) + ".json").write_text(json.dumps(meta, indent=2))
    print("wrote", args.out_lora, flush=True)


if __name__ == "__main__":
    main()
