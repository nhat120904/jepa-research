"""Pilot screen for planner-induced cost learning (round-0, frozen latents).

The selection-aware *encoder* sprint (scripts/60-62) gated
``STOP_METHOD_DIRECTION``: closed-loop push stayed at 0-5/16 across arms. The
open variant is iterative planner-induced cost learning — mine, relabel with
simulator truth, retrain, re-mine — which needs a GPU and a re-mining loop.

This script runs the cheap *necessary condition* for that loop, offline:

    on populations the planner already mined, does a regret-weighted
    (asymmetric) ranking objective select better than the objectives the
    sprint already tried, on a HELD-OUT episode?

If the asymmetric arm cannot beat uniform pairwise / regression / softmin here,
the iterative loop has nothing to bootstrap from and the direction closes for
the price of a CPU run.  A pass is *not* evidence of closed-loop success — the
encoder is frozen, there is no re-mining, and Phases 0/3/3b already showed
frozen-latent readouts fail closed-loop.  It only licenses the GPU experiment.

Protocol
--------
* Input is the sprint's own mined buffer (scripts/59), so the candidate
  populations, simulator truth and grouping are unchanged.
* The encoder is frozen and latents are cached once; every arm therefore gets
  identical features and differs only in objective.
* Leave-one-episode-out cross-validation over all 8 episodes, because the
  shipped 6/2 split leaves only two held-out episodes.
* Reported per held-out episode, then cluster-bootstrapped over episodes.

    .venv/bin/python scripts/64_planner_induced_cost_pilot.py \
        --buffer results/selection_populations_dino_push.pt \
        --model dino_wm_metaworld --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from models.heads.selection_cost import SelectionCostHead  # noqa: E402
from planning.selection_objectives import (  # noqa: E402
    grouped_pairwise_logistic, grouped_regression_huber,
    grouped_regret_weighted_pairwise, grouped_softmin_regret)


# --------------------------------------------------------------------------
# stage 1 — encode the mined frames once with the frozen encoder
# --------------------------------------------------------------------------

@torch.no_grad()
def encode_buffer(buffer: dict, model: str, device: str, batch: int) -> dict:
    from tensordict.tensordict import TensorDict

    adapter = build_adapter(model, device=device)
    uses_proprio = bool(adapter.spec.uses_proprio)

    def run(frames: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        out = []
        for start in range(0, len(frames), batch):
            vis = frames[start:start + batch].float().permute(0, 3, 1, 2)
            vis = vis.to(device).unsqueeze(1)                      # (B,1,C,H,W)
            if uses_proprio:
                obs = TensorDict(
                    {"visual": vis,
                     "proprio": prop[start:start + batch].float().to(device).unsqueeze(1)},
                    batch_size=[])
                z = adapter.encpred.encode(obs)["visual"]
            else:
                z = adapter.encpred.encode(vis)
                z = z["visual"] if hasattr(z, "keys") else z
            out.append(z[:, 0].to(torch.float16).cpu())
            print(f"    encoded {min(start + batch, len(frames))}/{len(frames)}",
                  end="\r", flush=True)
        print()
        return torch.cat(out)

    print("  candidate frames ...")
    z = run(buffer["frames"], buffer["prop"])
    print("  goal frames ...")
    z_goal = run(buffer["ep_goal_frames"], buffer["ep_goal_prop"])
    return {"z": z, "z_goal": z_goal, "model": model}


# --------------------------------------------------------------------------
# stage 2 — train one arm on frozen latents
# --------------------------------------------------------------------------

def make_objective(name: str):
    if name == "regression":
        return lambda p, t, g: grouped_regression_huber(p, t, g)
    if name == "pairwise":
        return lambda p, t, g: grouped_pairwise_logistic(p, t, g)
    if name == "softmin":
        return lambda p, t, g: grouped_softmin_regret(p, t, g)
    if name.startswith("regretw_k"):
        kappa = float(name.split("_k")[1])
        return lambda p, t, g: grouped_regret_weighted_pairwise(p, t, g, kappa=kappa)
    raise ValueError(f"unknown arm: {name}")


def train_arm(arm, z, z_goal, ep_idx, true_cost, gid, train_groups, *,
              latent_dim, epochs, lr, seed, device):
    torch.manual_seed(seed)
    head = SelectionCostHead(latent_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    objective = make_objective(arm)
    order = np.arange(len(train_groups))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        for oi in order:
            idx = train_groups[oi]
            zc = z[idx].to(device=device, dtype=torch.float32)
            zg = z_goal[ep_idx[idx[0]]].to(device=device, dtype=torch.float32)
            zg = zg.unsqueeze(0).expand(len(idx), *zg.shape)
            pred = head(zc, zg)
            loss = objective(pred, true_cost[idx].to(device),
                             gid[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
    return head.eval()


@torch.no_grad()
def evaluate(head, z, z_goal, ep_idx, true_cost, groups, device):
    """Per-population normalised regret and false-low rate."""
    regret, falselow = [], []
    for idx in groups:
        zc = z[idx].to(device=device, dtype=torch.float32)
        zg = z_goal[ep_idx[idx[0]]].to(device=device, dtype=torch.float32)
        zg = zg.unsqueeze(0).expand(len(idx), *zg.shape)
        pred = head(zc, zg).cpu()
        truth = true_cost[idx]
        base = float(truth.mean() - truth.min())
        if base <= 0:
            continue
        chosen = int(torch.argmin(pred))
        regret.append(float(truth[chosen] - truth.min()) / base)
        # false low = the pick lands in the worst tercile of the population
        falselow.append(float(truth[chosen] >= torch.quantile(truth, 2 / 3)))
    return float(np.mean(regret)), float(np.mean(falselow))


@torch.no_grad()
def evaluate_fixed(cost, true_cost, groups):
    """Same metrics for a cost that is already given per candidate."""
    regret, falselow = [], []
    for idx in groups:
        truth = true_cost[idx]
        base = float(truth.mean() - truth.min())
        if base <= 0:
            continue
        chosen = int(torch.argmin(cost[idx]))
        regret.append(float(truth[chosen] - truth.min()) / base)
        falselow.append(float(truth[chosen] >= torch.quantile(truth, 2 / 3)))
    return float(np.mean(regret)), float(np.mean(falselow))


def cluster_ci(values, boots=8000, seed=0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    b = [rng.choice(v, len(v)).mean() for _ in range(boots)]
    return v.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="results/selection_populations_dino_push.pt")
    ap.add_argument("--model", default="dino_wm_metaworld")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--latent-cache", default="results/selection_latents_dino_push.pt")
    ap.add_argument("--arms", nargs="+",
                    default=["regression", "pairwise", "softmin",
                             "regretw_k3", "regretw_k10"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--out", default="results/planner_induced_cost_pilot.md")
    args = ap.parse_args()

    buffer = torch.load(ROOT / args.buffer, map_location="cpu", weights_only=False)
    cache_path = ROOT / args.latent_cache
    if cache_path.exists():
        print(f"[1/3] latents from cache {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        print(f"[1/3] encoding {len(buffer['frames'])} frames on {args.device} ...")
        cache = encode_buffer(buffer, args.model, args.device, args.encode_batch)
        torch.save(cache, cache_path)
        print(f"      saved {cache_path}")

    z, z_goal = cache["z"], cache["z_goal"]
    latent_dim = int(z.shape[-1])
    ep_idx = buffer["ep_idx"].long()
    seed_of = buffer["seed"].long()
    true_cost = buffer["true_cost"].float()
    proxy_cost = buffer["proxy_cost"].float()
    gid = buffer["group_id"].long()

    groups_by_ep = {}
    for g in torch.unique(gid):
        idx = torch.nonzero(gid == g, as_tuple=False).flatten()
        if len(idx) >= 4:
            groups_by_ep.setdefault(int(ep_idx[idx[0]]), []).append(idx)
    episodes = sorted(groups_by_ep)
    print(f"[2/3] {len(episodes)} episodes, "
          f"{sum(len(v) for v in groups_by_ep.values())} populations, D={latent_dim}")

    per_arm = {a: {"regret": [], "falselow": []} for a in args.arms}
    baseline = {"regret": [], "falselow": []}
    for held in episodes:
        train_groups = [g for e in episodes if e != held for g in groups_by_ep[e]]
        test_groups = groups_by_ep[held]
        r, f = evaluate_fixed(proxy_cost, true_cost, test_groups)
        baseline["regret"].append(r)
        baseline["falselow"].append(f)
        for arm in args.arms:
            rs, fs = [], []
            for sd in args.seeds:
                head = train_arm(arm, z, z_goal, ep_idx, true_cost, gid,
                                 train_groups, latent_dim=latent_dim,
                                 epochs=args.epochs, lr=args.lr, seed=sd,
                                 device=args.device)
                a, b = evaluate(head, z, z_goal, ep_idx, true_cost,
                                test_groups, args.device)
                rs.append(a)
                fs.append(b)
            per_arm[arm]["regret"].append(float(np.mean(rs)))
            per_arm[arm]["falselow"].append(float(np.mean(fs)))
        print(f"      held-out ep {held}: " + "  ".join(
            f"{a}={per_arm[a]['regret'][-1]:.3f}" for a in args.arms)
            + f"   [planner l2 {r:.3f}]")

    lines = ["# Planner-induced cost pilot (round 0, frozen latents)", "",
             f"Buffer `{args.buffer}`, model `{args.model}`, "
             f"leave-one-episode-out over {len(episodes)} episodes, "
             f"{len(args.seeds)} head seeds per fold.", "",
             "Normalised selection regret: 1.00 = the expected regret of a "
             "uniformly random candidate in the same population; 0.00 = oracle. "
             "False-low rate = the pick lands in the worst tercile of true cost. "
             "CIs are a cluster bootstrap over held-out episodes.", "",
             "| arm | normalised regret [95% CI] | false-low rate [95% CI] |",
             "|---|---|---|"]
    m, lo, hi = cluster_ci(baseline["regret"])
    m2, lo2, hi2 = cluster_ci(baseline["falselow"])
    lines.append(f"| planner latent-L2 (as mined) | {m:.3f} [{lo:.3f}, {hi:.3f}] "
                 f"| {m2:.3f} [{lo2:.3f}, {hi2:.3f}] |")
    for arm in args.arms:
        m, lo, hi = cluster_ci(per_arm[arm]["regret"])
        m2, lo2, hi2 = cluster_ci(per_arm[arm]["falselow"])
        lines.append(f"| {arm} | {m:.3f} [{lo:.3f}, {hi:.3f}] "
                     f"| {m2:.3f} [{lo2:.3f}, {hi2:.3f}] |")
    lines += ["", "Per-episode values (for paired reading):", "",
              "```json",
              json.dumps({"episodes": episodes,
                          "planner_l2": baseline["regret"],
                          **{a: per_arm[a]["regret"] for a in args.arms}},
                         indent=2),
              "```"]
    (ROOT / args.out).write_text("\n".join(lines) + "\n")
    print(f"[3/3] wrote {args.out}")
    print("\n".join(lines[5:]))


if __name__ == "__main__":
    main()
