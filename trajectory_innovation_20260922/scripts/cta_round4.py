"""Round 4: the Round-3 parallel WM recipe on the enlarged, action-diverse data, with early stopping.

docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md. Same frozen source encoder / reader (checkpoint 55018), same warm start,
losses, batch (32 banks as 4 x 8) and learning rate as Round 3 (55077). Changes: training data (old P0 banks + new
standard banks + new perturbed banks), and per-network checkpoint selection by that network's own objective on the
offline dev banks (roots 2000-2099; closed-loop roots 2100-2199 are disjoint). Networks:
  nll      parallel FSQ WM, coordinate NLL only                      (all banks)
  task     NLL + score consistency + weighted ranking via frozen reader (all banks)   -> CTA4
  task_std same as task without the perturbed banks                     (standard banks) -> ablation CTA4S
  direct   direct scorer, weighted ranking                             (all banks)
"""
import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cta_closed_loop as closed  # noqa: E402
import cta_round3 as r3  # noqa: E402
import cta_train as ct  # noqa: E402
from cta_reader_refine import write_json  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.cta import Scorer, action_features, goal_scores, proprio  # noqa: E402
from ti_wm.cta_eval import ranking_metrics  # noqa: E402
from ti_wm.cta_parallel import ParallelFSQWM, score_consistency, weighted_rank  # noqa: E402

K = 8
NETS = ("nll", "task", "task_std", "direct")
POOL = {"nll": "all", "task": "all", "task_std": "std", "direct": "all"}
TIER = {"nll": "r4_nll", "task": "r4_task", "task_std": "r4_task_std", "direct": "r4_direct"}


class Plus:
    """Compact training set (scripts/cta_encode_plus.py) in RAM."""

    def __init__(self, path, codebook, limit=None):
        with np.load(path / "banks.npz") as z:
            self.b = {k: z[k] for k in z.files}
        if limit is not None:
            keep = np.concatenate([np.flatnonzero(self.b["source"] == s)[:limit] for s in np.unique(self.b["source"])])
            self.b = {k: v[keep] for k, v in self.b.items()}
        self.cur = torch.from_numpy(np.load(path / "cur.npy"))
        self.prev = torch.from_numpy(np.load(path / "prev.npy"))
        self.codebook = codebook
        self.n = len(self.b["root"])
        self.pools = {"all": np.arange(self.n), "std": np.flatnonzero(self.b["source"] != 2)}
        self.t = {k: torch.from_numpy(self.b[k]) for k in ("chunk", "ctx_pos", "geom", "src")}

    def batch(self, i, device):
        i = np.asarray(i)
        ci = torch.from_numpy(self.b["ctx_index"][i])
        pos = self.t["ctx_pos"][i]
        ctx = {"cur": self.cur[ci].to(device).repeat_interleave(K, 0),
               "prev": self.prev[ci].to(device).repeat_interleave(K, 0),
               "prop": proprio(pos[:, 0], pos[:, 1]).to(device).repeat_interleave(K, 0)}
        act = action_features(self.t["chunk"][i], pos[:, 0][:, None]).flatten(0, 1).to(device)
        src = self.codebook[self.t["src"][i].long().to(device)].flatten(0, 1)
        return ctx, act, src, self.t["geom"][i].to(device)


def build_nets(parent, device):
    cfg = parent["config"]
    nets = {name: ParallelFSQWM(m=cfg["m"]).to(device) for name in ("nll", "task", "task_std")}
    warm = nets["nll"].warm_start(parent["state"]["wm"])
    nets["task"].load_state_dict(nets["nll"].state_dict(), strict=True)
    nets["task_std"].load_state_dict(nets["nll"].state_dict(), strict=True)
    nets["direct"] = Scorer("action", layers=cfg["direct_layers"]).to(device)
    nets["direct"].load_state_dict(parent["state"]["direct"], strict=True)
    return nets, warm


def net_loss(name, net, reader, ctx, act, src, labels, goal, teacher, scale, pair_count, rank_cfg):
    """Round-3 objective of each network on one batch. Returns (unweighted sum, parts); callers combine parts.
    pair_count: denominator of the weighted ranking term (global informative-pair count of the effective batch)."""
    if name == "direct":
        scores = net(ctx, act, goal).float().view(-1, K)
        rank = weighted_rank(scores, labels, *rank_cfg, pair_count=pair_count)
        return rank, {"rank": rank}
    expected, logits = net(ctx, act)
    anchor = net.nll(logits, src)
    if name == "nll":
        return anchor, {"nll": anchor}
    scores = reader(ctx, expected, goal).float().view(-1, K)
    consistency = score_consistency(scores, teacher, scale)
    rank = weighted_rank(scores, labels, *rank_cfg, pair_count=pair_count)
    return anchor + consistency + rank, {"nll": anchor, "consistency": consistency, "rank": rank}


@torch.inference_mode()
def dev_objectives(nets, reader, dev, dev_src, codebook, goals, goal_idx, scales, rank_cfg, device, amp):
    """Each network's own training objective on all dev banks, fixed goal per bank (selection criterion)."""
    totals = {n: {"nll": 0., "consistency": 0., "rank_sum": 0.} for n in nets}
    pairs, rows = 0., 0
    for s in range(0, dev.n, 8):
        i = torch.arange(s, min(s + 8, dev.n))
        ctx, act = dev.context(i, device), dev.actions(i, device)
        src = codebook[dev_src[i].long().to(device)].flatten(0, 1)
        labels = dev.cov[i].to(device)
        goal = goals[torch.as_tensor(goal_idx[i.numpy()], device=device)].repeat_interleave(K, 0)
        pairs += float(((labels[:, :, None] - labels[:, None, :]) > rank_cfg[1]).sum())
        rows += len(i)
        one = torch.ones((), device=device)
        with amp():
            teacher = reader(ctx, src, goal).float().view(-1, K)
            for n, net in nets.items():
                _, parts = net_loss(n, net, reader, ctx, act, src, labels, goal, teacher, scales[POOL[n]], one, rank_cfg)
                totals[n]["rank_sum"] += float(parts["rank"]) if "rank" in parts else 0.
                totals[n]["nll"] += float(parts.get("nll", 0.)) * len(i)
                totals[n]["consistency"] += float(parts.get("consistency", 0.)) * len(i)
    out = {}
    for n, t in totals.items():
        nll, cons, rank = t["nll"] / rows, t["consistency"] / rows, t["rank_sum"] / max(pairs, 1.)
        objective = {"nll": nll, "direct": rank}.get(n, nll + cons + rank)
        out[n] = {"objective": objective, "nll_nats_per_token": nll * 3, "consistency": cons, "rank": rank}
    return out


@torch.inference_mode()
def dev_scores(nets, reader, dev, goals, device, amp, subset=None):
    idx = np.arange(dev.n) if subset is None else np.asarray(subset)
    out = {TIER[n]: np.zeros((len(idx), K), np.float32) for n in nets}
    for s in range(0, len(idx), 4):
        i = torch.as_tensor(idx[s:s + 4])
        ctx, act = dev.context(i, device), dev.actions(i, device)
        with amp():
            for n, net in nets.items():
                if n == "direct":
                    v = goal_scores(net, ctx, act, goals)
                else:
                    expected, _ = net(ctx, act)
                    v = goal_scores(reader, ctx, expected, goals)
                out[TIER[n]][s:s + len(i)] = v.view(len(i), K).cpu().numpy()
    return out


def train(a):
    report = {"status": "RUNNING", "job": os.environ.get("SLURM_JOB_ID"), "stage": "load"}
    path = a.run / "train_report.json"
    started = time.perf_counter()
    try:
        device = torch.device("cuda")
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        rng = np.random.default_rng(0)
        parent = torch.load(a.parent / "cta.pt", map_location="cpu")
        if parent["config"].get("target") != "negative_mean_vertex_distance_divided_by_512":
            raise ValueError("Expected geometry checkpoint")
        cfg = {"m": parent["config"]["m"], "direct_layers": parent["config"]["direct_layers"],
               "parent": str(a.parent.resolve()), "parent_sha256": closed.sha256(a.parent / "cta.pt"),
               "features": str(a.features.resolve()), "updates": a.updates, "banks": 32, "microbanks": 8,
               "lr": 1e-4, "wd": parent["config"]["wd"], "rank_scale": .01, "rank_margin": .001,
               "eval_every": a.eval_every, "selection": "per-network own objective on dev banks 2000-2099",
               "seed": 0}
        report["config"] = cfg
        write_json(path, report)
        frozen = ct.build(parent["config"], device)
        for name, model in frozen.items():
            model.load_state_dict(parent["state"][name], strict=True)
            model.eval().requires_grad_(False)
        reader, codebook = frozen["reader"], frozen["enc"].fsq.codebook
        nets, warm = build_nets(parent, device)
        report["warm_started_keys"] = warm
        opts = {n: torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) for n, net in nets.items()}
        data = Plus(a.features, codebook, a.limit)
        dev = ct.Split(a.old_features / "dev", a.dev_limit)
        dev_src = ct.source_indices(frozen, dev, device, amp)
        goals = torch.from_numpy(np.load(a.features / "goals.npy")).to(device)
        goal_idx = np.random.default_rng(777).integers(0, len(goals), dev.n)
        subset = np.sort(np.random.default_rng(778).choice(dev.n, min(800, dev.n), replace=False))
        rank_cfg = (cfg["rank_scale"], cfg["rank_margin"])
        report["data"] = {"banks": data.n, "by_source": {int(s): int((data.b["source"] == s).sum())
                          for s in np.unique(data.b["source"])}, "pools": {k: len(v) for k, v in data.pools.items()},
                          "dev_banks": dev.n}
        # consistency normalizer per pool, TRAIN only, independent RNG
        cal = np.random.default_rng(381)
        scales = {}
        for pool in ("all", "std"):
            vals = []
            for _ in range(32):
                ctx, act, src, _ = data.batch(cal.choice(data.pools[pool], cfg["microbanks"]), device)
                goal = ct.sample_goals(goals, cal, cfg["microbanks"], device)
                with torch.no_grad(), amp():
                    t = reader(ctx, src, goal).float().view(-1, K)
                vals.append(float((t - t.mean(1, keepdim=True)).square().mean()))
            scales[pool] = max(float(np.sqrt(np.mean(vals))), 1e-3)
        report.update(stage="train", scales=scales, dev_history=[])
        write_json(path, report)
        best = {n: {"objective": float("inf"), "update": None, "state": None} for n in nets}

        def evaluate(update):
            for net in nets.values():
                net.eval()
            t0 = time.perf_counter()
            obj = dev_objectives(nets, reader, dev, dev_src, codebook, goals, goal_idx, scales, rank_cfg, device, amp)
            sc = dev_scores(nets, reader, dev, goals, device, amp, subset)
            gap = {k: ranking_metrics(v, dev.cov8[subset], dev.root[subset], ci=False)["retained_gap"]["ratio"]
                   for k, v in sc.items()}
            for n in nets:
                if obj[n]["objective"] < best[n]["objective"]:
                    best[n] = {"objective": obj[n]["objective"], "update": update,
                               "state": copy.deepcopy({k: v.detach().cpu() for k, v in nets[n].state_dict().items()})}
            entry = {"update": update, "objectives": obj, "retained_gap_subset800": gap,
                     "seconds": time.perf_counter() - t0}
            report["dev_history"].append(entry)
            write_json(a.run / "dev_curve.json", report["dev_history"])
            print({"dev": entry}, flush=True)
            for net in nets.values():
                net.train()

        evaluate(0)
        accumulation = cfg["banks"] // cfg["microbanks"]
        train_seconds = 0.
        for step in range(cfg["updates"]):
            tick = time.perf_counter()
            batch = {pool: rng.choice(data.pools[pool], cfg["banks"]) for pool in ("all", "std")}
            goal_all = {pool: ct.sample_goals(goals, rng, cfg["banks"], device) for pool in ("all", "std")}
            pair_count = {pool: ((data.t["geom"][batch[pool]][:, :, None] - data.t["geom"][batch[pool]][:, None, :])
                                 > cfg["rank_margin"]).sum().to(device) for pool in ("all", "std")}
            for opt in opts.values():
                opt.zero_grad(set_to_none=True)
            values = {"update": step + 1}
            for off in range(0, cfg["banks"], cfg["microbanks"]):
                for pool in ("all", "std"):
                    ctx, act, src, labels = data.batch(batch[pool][off:off + cfg["microbanks"]], device)
                    goal = goal_all[pool][off * K:(off + cfg["microbanks"]) * K]
                    with torch.no_grad(), amp():
                        teacher = reader(ctx, src, goal).float().view(-1, K)
                    for n in (n for n in NETS if POOL[n] == pool):
                        with amp():
                            _, parts = net_loss(n, nets[n], reader, ctx, act, src, labels, goal, teacher,
                                                scales[pool], pair_count[pool], rank_cfg)
                            # NLL / consistency are per-microbatch means -> average over microbatches;
                            # ranking already divides by the effective batch's global pair count -> sum.
                            loss = sum(v for k, v in parts.items() if k != "rank") / accumulation
                            loss = loss + parts.get("rank", 0.)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(n)
                        loss.backward()
                        for k, v in parts.items():
                            values[f"{n}/{k}"] = values.get(f"{n}/{k}", 0.) + float(v) / (1 if k == "rank" else accumulation)
            for n, net in nets.items():
                values[f"{n}/grad"] = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
                opts[n].step()
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - tick
            values["mean_seconds_per_update"] = train_seconds / (step + 1)
            if step == 0 or (step + 1) % 100 == 0:
                with (a.run / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(values) + "\n")
                print(values, flush=True)
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["updates"]:
                evaluate(step + 1)
                report.update(last_update=step + 1, wall_seconds=time.perf_counter() - started)
                write_json(path, report)
        # select, save, full offline ladder
        for n, net in nets.items():
            net.load_state_dict(best[n]["state"], strict=True)
            net.eval()
        report["selected_update"] = {n: best[n]["update"] for n in nets}
        torch.save({"config": cfg, "networks": {n: net.state_dict() for n, net in nets.items()},
                    "selected_update": report["selected_update"], "dev_history": report["dev_history"]},
                   a.run / "round4.pt")
        report["checkpoint_sha256"] = closed.sha256(a.run / "round4.pt")
        report["frozen_unchanged"] = all(torch.equal(v.detach().cpu(), parent["state"][name][k])
                                         for name, model in frozen.items() for k, v in model.state_dict().items())
        if not report["frozen_unchanged"]:
            raise RuntimeError("Frozen parent models changed")
        report.update(stage="offline")
        write_json(path, report)
        scores = dev_scores(nets, reader, dev, goals, device, amp)
        if a.r3 is not None and a.dev_limit is None:
            with np.load(a.r3 / "dev_scores.npz") as saved:
                if not np.array_equal(saved["root"], dev.root):
                    raise ValueError("Round-3 dev scores do not match")
                scores.update({k: saved[k] for k in saved.files if k not in ("root", "decision", "cov8", "native_cov8")})
        masks = r3.group_masks(a.old_features, a.collection) if a.dev_limit is None else {"all": np.ones(dev.n, bool)}
        ladder = {}
        for g, mask in masks.items():
            ladder[g] = {}
            for tier, v in scores.items():
                m = ranking_metrics(v[mask], dev.cov8[mask], dev.root[mask], ci=True)
                m.pop("chosen")
                ladder[g][tier] = m
        final_obj = dev_objectives(nets, reader, dev, dev_src, codebook, goals, goal_idx, scales, rank_cfg, device, amp)
        write_json(a.run / "offline.json", {"ladder": ladder, "selected_objectives": final_obj,
                                            "groups": {g: int(m.sum()) for g, m in masks.items()}})
        np.savez(a.run / "dev_scores.npz", root=dev.root, cov8=dev.cov8, **scores)
        report.update(status="DONE", stage="done", seconds=time.perf_counter() - started)
        write_json(path, report)
        print(json.dumps({g: {t: round(v["retained_gap"]["ratio"], 3) for t, v in d.items()}
                          for g, d in ladder.items()}), flush=True)
    except Exception:
        report.update(status="FAILED", error=traceback.format_exc())
        write_json(path, report)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("run", "parent", "features", "old-features", "collection", "r3"):
        p.add_argument(f"--{key}", type=Path, required=key not in ("r3",))
    p.add_argument("--updates", type=int, default=8000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None, help="smoke only: banks per source")
    p.add_argument("--dev-limit", type=int, default=None, help="smoke only")
    args = p.parse_args()
    require_compute()
    args.run.mkdir(parents=True, exist_ok=True)
    train(args)
