"""
Phase 0 / Gate 1 -- Observability diagnostic on ContactWorld.

Question: is the task-relevant hidden state recoverable from DEPLOYABLE observations,
and does tactile / history add information over current vision alone?

The target is `plug_pos` (and optionally plug_quat) because that is literally what the
ContactWorld success metric thresholds (eval_planner.py:313, plug_pos_err < 0.01 m).

Conditions (all inputs deployable at test time -- never privileged object state):
  A  pc_now            current pointcloud only
  B  pc_prop_now       current pointcloud + proprio
  C  pc_prop_hist      history of pointcloud + proprio            (GRU)
  D  pc_prop_tac_hist  history of pointcloud + proprio + tactile  (GRU)
  E  prop_tac_hist     history of proprio + tactile, NO vision    (GRU)

Split is by EPISODE. CIs are episode-clustered bootstrap. D-vs-C is the tactile-adds-info
test; it is evaluated as a PAIRED per-episode difference on the same held-out episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zarr

PROPRIO_KEYS = ["ee_pos", "ee_quat", "dof_pos", "dof_vel"]


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_task(root: Path, target_key: str):
    z = zarr.open(str(root), mode="r")
    g = z["data"]
    ends = np.asarray(z["meta"]["episode_ends"]).astype(int)
    starts = np.concatenate([[0], ends[:-1]])

    out = {
        "pc": np.asarray(g["pointcloud"]).astype(np.float32),
        "tac": np.asarray(g["tactile_force_field_right"]).astype(np.float32),
        "action": np.asarray(g["action"]).astype(np.float32),
        "target": np.asarray(g[target_key]).astype(np.float32),
    }
    out["proprio"] = np.concatenate(
        [np.asarray(g[k]).astype(np.float32) for k in PROPRIO_KEYS], axis=-1
    )
    out["tac"] = out["tac"].reshape(len(out["tac"]), -1)
    out["episodes"] = [(int(s), int(e)) for s, e in zip(starts, ends)]
    return out


def build_windows(data, hist: int, min_len: int):
    """Index windows [t-hist+1 .. t] that stay inside one episode."""
    idx, ep_id = [], []
    for e_i, (s, e) in enumerate(data["episodes"]):
        if e - s < max(min_len, hist):
            continue
        for t in range(s + hist - 1, e):
            idx.append(t)
            ep_id.append(e_i)
    return np.asarray(idx), np.asarray(ep_id)


class Norm:
    def __init__(self, x):
        self.m = x.mean(0, keepdims=True)
        self.s = x.std(0, keepdims=True) + 1e-6

    def __call__(self, x):
        return (x - self.m) / self.s


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class PointNet(nn.Module):
    def __init__(self, in_ch=6, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x):              # x [B,N,C]
        return self.net(x).max(dim=1).values


class Probe(nn.Module):
    def __init__(self, use_pc, use_prop, use_tac, hist, prop_dim, tac_dim, out_dim, hid=256):
        super().__init__()
        self.use_pc, self.use_prop, self.use_tac, self.hist = use_pc, use_prop, use_tac, hist
        feat = 0
        if use_pc:
            self.pc = PointNet(dim=128)
            feat += 128
        if use_prop:
            self.pr = nn.Sequential(nn.Linear(prop_dim, 128), nn.ReLU(), nn.Linear(128, 64))
            feat += 64
        if use_tac:
            self.tc = nn.Sequential(nn.Linear(tac_dim, 128), nn.ReLU(), nn.Linear(128, 64))
            feat += 64
        self.temporal = nn.GRU(feat, hid, batch_first=True) if hist > 1 else None
        head_in = hid if hist > 1 else feat
        self.head = nn.Sequential(nn.Linear(head_in, hid), nn.ReLU(), nn.Linear(hid, out_dim))

    def forward(self, pc, prop, tac):
        # inputs are [B,T,...]; T == hist
        B, T = prop.shape[0], prop.shape[1]
        parts = []
        if self.use_pc:
            parts.append(self.pc(pc.reshape(B * T, pc.shape[2], pc.shape[3])).reshape(B, T, -1))
        if self.use_prop:
            parts.append(self.pr(prop))
        if self.use_tac:
            parts.append(self.tc(tac))
        h = torch.cat(parts, dim=-1)
        if self.temporal is not None:
            h, _ = self.temporal(h)
        return self.head(h[:, -1])


CONDITIONS = {
    "A_pc_now":           dict(use_pc=True,  use_prop=False, use_tac=False, hist=1),
    "B_pc_prop_now":      dict(use_pc=True,  use_prop=True,  use_tac=False, hist=1),
    "C_pc_prop_hist":     dict(use_pc=True,  use_prop=True,  use_tac=False, hist=8),
    "D_pc_prop_tac_hist": dict(use_pc=True,  use_prop=True,  use_tac=True,  hist=8),
    "E_prop_tac_hist":    dict(use_pc=False, use_prop=True,  use_tac=True,  hist=8),
    "F_prop_hist":        dict(use_pc=False, use_prop=True,  use_tac=False, hist=8),
}


# --------------------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------------------
def run_condition(data, cfg, train_eps, test_eps, seed, device, epochs, bs, lr):
    torch.manual_seed(seed)
    np.random.seed(seed)
    hist = cfg["hist"]

    idx, ep_id = build_windows(data, hist, min_len=hist)
    tr = np.isin(ep_id, train_eps)
    te = np.isin(ep_id, test_eps)
    tr_idx, te_idx, te_ep = idx[tr], idx[te], ep_id[te]

    prop_n = Norm(data["proprio"][tr_idx])
    tac_n = Norm(data["tac"][tr_idx])
    tgt_n = Norm(data["target"][tr_idx])

    def gather(ix):
        off = np.arange(-hist + 1, 1)
        w = ix[:, None] + off[None, :]                      # [B,T]
        pc = torch.from_numpy(data["pc"][w.reshape(-1)]).reshape(len(ix), hist, data["pc"].shape[1], -1)
        pr = torch.from_numpy(prop_n(data["proprio"])[w.reshape(-1)]).reshape(len(ix), hist, -1)
        tc = torch.from_numpy(tac_n(data["tac"])[w.reshape(-1)]).reshape(len(ix), hist, -1)
        y = torch.from_numpy(data["target"][ix])
        return pc, pr, tc, y

    model = Probe(
        cfg["use_pc"], cfg["use_prop"], cfg["use_tac"], hist,
        data["proprio"].shape[1], data["tac"].shape[1], data["target"].shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ym = torch.from_numpy(tgt_n.m).to(device)
    ys = torch.from_numpy(tgt_n.s).to(device)

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr_idx))
        for i in range(0, len(perm), bs):
            b = tr_idx[perm[i:i + bs]]
            pc, pr, tc, y = gather(b)
            pc, pr, tc, y = pc.to(device), pr.to(device), tc.to(device), y.to(device)
            pred = model(pc, pr, tc)
            loss = nn.functional.mse_loss(pred, (y - ym) / ys)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 512):
            b = te_idx[i:i + 512]
            pc, pr, tc, y = gather(b)
            pred = model(pc.to(device), pr.to(device), tc.to(device)) * ys + ym
            errs.append(torch.linalg.norm(pred.cpu() - y, dim=-1).numpy())
    return np.concatenate(errs), te_ep


def baseline_proprio_linear(data, train_eps, test_eps, hist=1):
    """Least-squares plug_pos ~ proprio. Reference floor, no learning capacity."""
    idx, ep_id = build_windows(data, hist, min_len=hist)
    tr, te = np.isin(ep_id, train_eps), np.isin(ep_id, test_eps)
    Xtr = np.concatenate([data["proprio"][idx[tr]], np.ones((tr.sum(), 1), np.float32)], 1)
    Xte = np.concatenate([data["proprio"][idx[te]], np.ones((te.sum(), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xtr, data["target"][idx[tr]], rcond=None)
    err = np.linalg.norm(Xte @ W - data["target"][idx[te]], axis=-1)
    return err, ep_id[te]


def cluster_bootstrap(errs, ep_ids, stat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep_ids)
    by_ep = {e: errs[ep_ids == e] for e in eps}
    vals = []
    for _ in range(n):
        pick = rng.choice(eps, size=len(eps), replace=True)
        vals.append(stat(np.concatenate([by_ep[e] for e in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_cluster_bootstrap(err_a, err_b, ep_ids, stat, n=2000, seed=0):
    """Paired A-minus-B on identical held-out windows, resampled by episode."""
    rng = np.random.default_rng(seed)
    eps = np.unique(ep_ids)
    a = {e: err_a[ep_ids == e] for e in eps}
    b = {e: err_b[ep_ids == e] for e in eps}
    vals = []
    for _ in range(n):
        pick = rng.choice(eps, size=len(eps), replace=True)
        vals.append(stat(np.concatenate([a[e] for e in pick])) - stat(np.concatenate([b[e] for e in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--target-key", default="plug_pos")
    p.add_argument("--success-thresh", type=float, default=0.01)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_task(Path(args.data_root), args.target_key)
    n_ep = len(data["episodes"])
    print(f"task={args.task} episodes={n_ep} steps={len(data['target'])} device={device}", flush=True)

    results = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_ep)
        n_te = max(4, int(round(args.test_frac * n_ep)))
        test_eps, train_eps = perm[:n_te], perm[n_te:]

        err_lin, ep_lin = baseline_proprio_linear(data, train_eps, test_eps)
        results.setdefault("LIN_proprio", []).append((err_lin, ep_lin))
        print(f"  seed{seed} LIN_proprio      mean={err_lin.mean()*100:6.2f}cm "
              f"hit@1cm={(err_lin < args.success_thresh).mean():.3f}", flush=True)

        for name, cfg in CONDITIONS.items():
            errs, eps = run_condition(data, cfg, train_eps, test_eps, seed, device,
                                      args.epochs, args.batch_size, args.lr)
            results.setdefault(name, []).append((errs, eps))
            print(f"  seed{seed} {name:18s} mean={errs.mean()*100:6.2f}cm "
                  f"median={np.median(errs)*100:6.2f}cm hit@1cm={(errs < args.success_thresh).mean():.3f}",
                  flush=True)

    summary = {"task": args.task, "target": args.target_key, "n_episodes": n_ep,
               "success_thresh_m": args.success_thresh, "seeds": args.seeds, "conditions": {}}
    for name, runs in results.items():
        e = np.concatenate([r[0] for r in runs])
        ep = np.concatenate([r[1] + 1000 * i for i, r in enumerate(runs)])
        lo, hi = cluster_bootstrap(e, ep, np.mean)
        hlo, hhi = cluster_bootstrap((e < args.success_thresh).astype(float), ep, np.mean)
        summary["conditions"][name] = {
            "mean_err_cm": float(e.mean() * 100),
            "mean_err_cm_ci": [lo * 100, hi * 100],
            "median_err_cm": float(np.median(e) * 100),
            "hit_at_thresh": float((e < args.success_thresh).mean()),
            "hit_at_thresh_ci": [hlo, hhi],
            "per_seed_mean_cm": [float(r[0].mean() * 100) for r in runs],
        }

    # paired tactile test: D vs C on identical windows
    if "D_pc_prop_tac_hist" in results and "C_pc_prop_hist" in results:
        d = np.concatenate([r[0] for r in results["D_pc_prop_tac_hist"]])
        c = np.concatenate([r[0] for r in results["C_pc_prop_hist"]])
        ep = np.concatenate([r[1] + 1000 * i for i, r in enumerate(results["C_pc_prop_hist"])])
        lo, hi = paired_cluster_bootstrap(d, c, ep, np.mean)
        summary["tactile_effect_D_minus_C"] = {
            "delta_mean_err_cm": float((d.mean() - c.mean()) * 100),
            "delta_ci_cm": [lo * 100, hi * 100],
            "ci_excludes_zero": bool(lo * hi > 0),
            "interpretation": "negative delta = tactile REDUCES hidden-state error",
        }

    def paired(name_a, name_b, label, note):
        if name_a not in results or name_b not in results:
            return
        a = np.concatenate([r[0] for r in results[name_a]])
        b = np.concatenate([r[0] for r in results[name_b]])
        ep = np.concatenate([r[1] + 1000 * i for i, r in enumerate(results[name_b])])
        lo, hi = paired_cluster_bootstrap(a, b, ep, np.mean)
        summary[label] = {
            "delta_mean_err_cm": float((a.mean() - b.mean()) * 100),
            "delta_ci_cm": [lo * 100, hi * 100],
            "ci_excludes_zero": bool(lo * hi > 0),
            "interpretation": note,
        }

    paired("E_prop_tac_hist", "F_prop_hist", "tactile_effect_E_minus_F",
           "negative = tactile adds information over proprio history alone (no vision)")
    paired("C_pc_prop_hist", "F_prop_hist", "vision_effect_C_minus_F",
           "negative = pointcloud adds information over proprio history alone")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\nwrote", args.out)
    print(json.dumps(summary.get("tactile_effect_D_minus_C", {}), indent=2))


if __name__ == "__main__":
    main()
