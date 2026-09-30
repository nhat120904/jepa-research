"""Round 6: train for the 16-candidate deployment bank (policy samples + their perturbed copies).

docs/CTA_ROUND6_PROTOCOL.md. Round 4 ranked the policy bank and the perturbed bank of the SAME context as two separate
8-banks, so no loss ever compared a policy candidate with a perturbed one; Round 5 deployment requires exactly that
comparison (CTA4 19/50 vs P0 31/50, GEOM16 43/50). Here each training example is the paired 16-bank (the Round-4 cache
stores both halves with identical contexts, in the same order), and every ranking loss is computed over all 16:

  rank16  weighted pairwise ranking over all 16 candidates (Round-4 weighting, geometry labels)
  anchor  the same ranking restricted to pairs with candidate 0, the default the arm must beat

Stage A: the code reader D(C, S, g) is fine-tuned on source codes of the actual futures (frozen source encoder, cache
reused). Stage B: with that reader frozen, the CTA world model (Round-4 task recipe: NLL + score consistency + ranking
through the reader) and the direct scorer are trained on the same 16-banks. Checkpoints are selected per network by
its own objective on held-out dev banks (collection roots 32200-32249, excluded from training); closed-loop roots
2100-2299 are disjoint from every training and selection root. The source encoder, the FULL reader and the Round-4
networks stay frozen. Old policy-only 8-banks (source 0) are not used.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cta_closed_loop as closed  # noqa: E402
import cta_train as ct  # noqa: E402
from cta_reader_refine import write_json  # noqa: E402
from ti_wm.contract import require_compute, select_candidate  # noqa: E402
from ti_wm.cta import Scorer, action_features, goal_scores, proprio  # noqa: E402
from ti_wm.cta_eval import ranking_metrics  # noqa: E402
from ti_wm.cta_parallel import ParallelFSQWM, score_consistency, weighted_rank  # noqa: E402

K, BANK = 8, 16
DEV_ROOT = 32200                     # collection roots >= this are the dev banks
WORSE = 0.005                        # "chose a candidate clearly worse than the default" (geometry units, ~2.5 px)


def anchored_rank(scores, labels, scale=.01, margin=1e-3):
    """Pairwise ranking restricted to (candidate 0, candidate k) pairs, both directions, Round-4 weighting."""
    delta = labels[:, 1:] - labels[:, :1]                       # >0: k better than the default
    diff = scores[:, 1:] - scores[:, :1]
    weight = (delta.abs() / scale).clamp(0, 1) * (delta.abs() > margin)
    loss = weight * F.softplus(-torch.sign(delta) * diff)
    return loss.sum() / (delta.abs() > margin).sum().clamp_min(1)


def rank_losses(scores, labels, cfg):
    return {"rank16": weighted_rank(scores, labels, cfg["rank_scale"], cfg["rank_margin"]),
            "anchor": anchored_rank(scores, labels, cfg["rank_scale"], cfg["rank_margin"])}


class Pairs:
    """Paired 16-banks from the Round-4 cache (scripts/cta_encode_plus.py): source 1 (policy) + source 2 (perturbed)."""

    def __init__(self, path, codebook, limit=None):
        with np.load(path / "banks.npz") as z:
            b = {k: z[k] for k in z.files}
        std, pert = np.flatnonzero(b["source"] == 1), np.flatnonzero(b["source"] == 2)
        if not (np.array_equal(b["ctx_index"][std], b["ctx_index"][pert]) and np.array_equal(b["root"][std], b["root"][pert])
                and np.array_equal(b["decision"][std], b["decision"][pert])):
            raise ValueError("policy and perturbed banks are not aligned by context")
        self.root, self.decision = b["root"][std], b["decision"][std]
        self.ctx_index, self.ctx_pos = b["ctx_index"][std], torch.from_numpy(b["ctx_pos"][std])
        cat = lambda k: torch.from_numpy(np.concatenate([b[k][std], b[k][pert]], 1))
        self.chunk, self.src, self.geom = cat("chunk"), cat("src"), cat("geom").float()
        self.cur = torch.from_numpy(np.load(path / "cur.npy"))          # ~4.5 GB in RAM, as in Round 4
        self.prev = torch.from_numpy(np.load(path / "prev.npy"))
        self.codebook = codebook
        dev = self.root >= DEV_ROOT
        self.train_idx, self.dev_idx = np.flatnonzero(~dev), np.flatnonzero(dev)
        if limit is not None:
            self.train_idx, self.dev_idx = self.train_idx[:limit], self.dev_idx[:max(8, limit // 8)]

    def batch(self, i, device):
        i = np.sort(np.asarray(i))
        ci = self.ctx_index[i]
        pos = self.ctx_pos[i]
        ci = torch.from_numpy(ci)
        ctx = {"cur": self.cur[ci].to(device).repeat_interleave(BANK, 0),
               "prev": self.prev[ci].to(device).repeat_interleave(BANK, 0),
               "prop": proprio(pos[:, 0], pos[:, 1]).to(device).repeat_interleave(BANK, 0)}
        act = action_features(self.chunk[i], pos[:, 0][:, None]).flatten(0, 1).to(device)
        src = self.codebook[self.src[i].long().to(device)].flatten(0, 1)
        return ctx, act, src, self.geom[i].to(device)


def choice_stats(scores, labels, roots):
    """Offline decision quality on 16-banks: retained gap over all 16 and over the policy half, and how often the
    argmax is clearly worse than the default (the Round-5 failure)."""
    out = {}
    for name, lim in (("bank16", BANK), ("policy8", K)):
        s, y = scores[:, :lim], labels[:, :lim]
        m = ranking_metrics(s, y, roots, ci=False)
        c = np.array([select_candidate([float(x) for x in row]) for row in s])
        gain = y[np.arange(len(y)), c] - y[:, 0]
        out[name] = {"retained_gap": m["retained_gap"]["ratio"], "spearman": m["within_bank_spearman"]["ratio"],
                     "worse_than_default": float((gain < -WORSE).mean()), "mean_gain": float(gain.mean()),
                     "chose_perturbed": float((c >= K).mean()) if lim == BANK else 0.}
    return out


def net_scores(kind, net, reader, ctx, act, src, goal):
    if kind == "reader":
        return net(ctx, src, goal).float().view(-1, BANK), None
    if kind == "direct":
        return net(ctx, act, goal).float().view(-1, BANK), None
    expected, logits = net(ctx, act)
    return reader(ctx, expected, goal).float().view(-1, BANK), (expected, logits)


def objective(kind, net, reader, ctx, act, src, labels, goal, teacher, scale, cfg):
    scores, extra = net_scores(kind, net, reader, ctx, act, src, goal)
    parts = rank_losses(scores, labels, cfg)
    if kind == "task":
        parts["nll"] = net.nll(extra[1], src)
        parts["consistency"] = score_consistency(scores, teacher, scale)
    return sum(parts.values()), parts


@torch.inference_mode()
def evaluate(kind, net, reader, data, goals, goal_idx, scale, cfg, device, amp):
    """Objective (fixed goal per dev bank) and offline choice statistics (mean over the goal set, as deployed)."""
    idx = data.dev_idx
    tot, n = {}, 0
    scores = np.zeros((len(idx), BANK), np.float32)
    for s in range(0, len(idx), 4):
        i = idx[s:s + 4]
        ctx, act, src, labels = data.batch(i, device)
        goal = goals[torch.as_tensor(goal_idx[s:s + len(i)], device=device)].repeat_interleave(BANK, 0)
        with amp():
            teacher = reader(ctx, src, goal).float().view(-1, BANK) if kind == "task" else None
            _, parts = objective(kind, net, reader, ctx, act, src, labels, goal, teacher, scale, cfg)
            if kind == "reader":
                v = goal_scores(net, ctx, src, goals)
            elif kind == "direct":
                v = goal_scores(net, ctx, act, goals)
            else:
                v = goal_scores(reader, ctx, net(ctx, act)[0], goals)
        for k, x in parts.items():
            tot[k] = tot.get(k, 0.) + float(x) * len(i)
        n += len(i)
        scores[s:s + len(i)] = v.float().view(len(i), BANK).cpu().numpy()
    parts = {k: v / n for k, v in tot.items()}
    return {"objective": sum(parts.values()), **parts,
            "choice": choice_stats(scores, data.geom[idx].numpy(), data.root[idx])}, scores


def train_stage(kind, nets, reader, data, goals, cfg, device, amp, rng, run, report, scale):
    """Train the networks in `nets` (dict kind -> module) jointly, one optimizer each; select per network."""
    opts = {k: torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) for k, m in nets.items()}
    goal_idx = np.random.default_rng(777).integers(0, len(goals), len(data.dev_idx))
    best = {k: {"objective": float("inf"), "update": None, "state": None} for k in nets}
    updates = cfg[f"updates_{'A' if 'reader' in nets else 'B'}"]
    history = report.setdefault("dev_history", [])

    def dev(update):
        for m in nets.values():
            m.eval()
        entry = {"update": update, "stage": "A" if "reader" in nets else "B"}
        for k, m in nets.items():
            rd = m if k == "reader" else reader
            entry[k], _ = evaluate(k, m, rd, data, goals, goal_idx, scale, cfg, device, amp)
            if entry[k]["objective"] < best[k]["objective"]:
                best[k] = {"objective": entry[k]["objective"], "update": update,
                           "state": copy.deepcopy({a: b.detach().cpu() for a, b in m.state_dict().items()})}
        history.append(entry)
        write_json(run / "dev_curve.json", history)
        print({"dev": entry}, flush=True)
        for m in nets.values():
            m.train()

    dev(0)
    acc = cfg["banks"] // cfg["microbanks"]
    for step in range(updates):
        batch = rng.choice(data.train_idx, cfg["banks"], replace=False)
        goal_all = ct.sample_goals(goals, rng, cfg["banks"], device).view(cfg["banks"], K, -1, goals.shape[-1])
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        values = {"update": step + 1}
        for off in range(0, cfg["banks"], cfg["microbanks"]):
            ctx, act, src, labels = data.batch(batch[off:off + cfg["microbanks"]], device)
            # one goal image per bank, shared by its 16 candidates
            goal = goal_all[off:off + cfg["microbanks"], 0].repeat_interleave(BANK, 0)
            for k, m in nets.items():
                rd = m if k == "reader" else reader
                with amp():
                    teacher = None
                    if k == "task":
                        with torch.no_grad():
                            teacher = reader(ctx, src, goal).float().view(-1, BANK)
                    loss, parts = objective(k, m, rd, ctx, act, src, labels, goal, teacher, scale, cfg)
                    loss = loss / acc
                if not torch.isfinite(loss):
                    raise FloatingPointError(k)
                loss.backward()
                for a, b in parts.items():
                    values[f"{k}/{a}"] = values.get(f"{k}/{a}", 0.) + float(b) / acc
        for k, m in nets.items():
            values[f"{k}/grad"] = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True))
            opts[k].step()
        if step == 0 or (step + 1) % 100 == 0:
            with (run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(values) + "\n")
            print(values, flush=True)
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == updates:
            dev(step + 1)
            write_json(run / "train_report.json", report)
    for k, m in nets.items():
        m.load_state_dict(best[k]["state"], strict=True)
        m.eval().requires_grad_(False)
    return {k: best[k]["update"] for k in nets}


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
        r4 = torch.load(a.r4 / "round4.pt", map_location="cpu")
        cfg = {"m": parent["config"]["m"], "direct_layers": parent["config"]["direct_layers"],
               "parent_sha256": closed.sha256(a.parent / "cta.pt"), "r4_sha256": closed.sha256(a.r4 / "round4.pt"),
               "features": str(a.features.resolve()), "updates_A": a.updates_a, "updates_B": a.updates_b,
               "banks": 16, "microbanks": 4, "lr": 1e-4, "wd": parent["config"]["wd"], "rank_scale": .01,
               "rank_margin": .001, "eval_every": a.eval_every, "dev_roots": f">= {DEV_ROOT}", "seed": 0,
               "losses": "rank16 + anchor (+ nll + consistency for task)"}
        report["config"] = cfg
        write_json(path, report)
        frozen = ct.build(parent["config"], device)
        for name, model in frozen.items():
            model.load_state_dict(parent["state"][name], strict=True)
            model.eval().requires_grad_(False)
        codebook = frozen["enc"].fsq.codebook
        data = Pairs(a.features, codebook, a.limit)
        goals = torch.from_numpy(np.load(a.features / "goals.npy")).to(device)
        report["data"] = {"train_banks16": int(len(data.train_idx)), "dev_banks16": int(len(data.dev_idx))}
        reader = Scorer("code", m=cfg["m"]).to(device)
        reader.load_state_dict(parent["state"]["reader"], strict=True)
        reader.train()
        # consistency scale: spread of the (new) teacher scores, measured after stage A below
        report["stage"] = "A_reader"
        write_json(path, report)
        before, _ = evaluate("reader", reader.eval(), reader, data, goals,
                             np.random.default_rng(777).integers(0, len(goals), len(data.dev_idx)), 1., cfg, device, amp)
        report["reader_before"] = before
        reader.train()
        sel_a = train_stage("reader", {"reader": reader}, None, data, goals, cfg, device, amp, rng, a.run, report, 1.)
        cal = np.random.default_rng(381)
        vals = []
        for _ in range(32):
            ctx, _, src, _ = data.batch(cal.choice(data.train_idx, cfg["microbanks"], replace=False), device)
            goal = ct.sample_goals(goals, cal, cfg["microbanks"], device).view(cfg["microbanks"], K, -1, goals.shape[-1])[:, 0]
            with torch.no_grad(), amp():
                t = reader(ctx, src, goal.repeat_interleave(BANK, 0)).float().view(-1, BANK)
            vals.append(float((t - t.mean(1, keepdim=True)).square().mean()))
        scale = max(float(np.sqrt(np.mean(vals))), 1e-3)
        report.update(stage="B_wm_direct", consistency_scale=scale)
        write_json(path, report)
        task = ParallelFSQWM(m=cfg["m"]).to(device)
        task.load_state_dict(r4["networks"]["task"], strict=True)
        direct = Scorer("action", layers=cfg["direct_layers"]).to(device)
        direct.load_state_dict(r4["networks"]["direct"], strict=True)
        sel_b = train_stage("task+direct", {"task": task, "direct": direct}, reader, data, goals, cfg, device, amp,
                            rng, a.run, report, scale)
        report["selected_update"] = {**sel_a, **sel_b}
        torch.save({"config": cfg, "networks": {"reader": reader.state_dict(), "task": task.state_dict(),
                                                "direct": direct.state_dict()},
                    "selected_update": report["selected_update"]}, a.run / "round6.pt")
        report["checkpoint_sha256"] = closed.sha256(a.run / "round6.pt")
        report["frozen_unchanged"] = all(torch.equal(v.detach().cpu(), parent["state"][name][k])
                                         for name, model in frozen.items() for k, v in model.state_dict().items())
        if not report["frozen_unchanged"]:
            raise RuntimeError("Frozen parent models changed")
        # offline comparison on dev 16-banks: parent reader on actual codes, new reader, R4 task/direct, R6 task/direct
        goal_idx = np.random.default_rng(777).integers(0, len(goals), len(data.dev_idx))
        r4task = ParallelFSQWM(m=cfg["m"]).to(device).eval()
        r4task.load_state_dict(r4["networks"]["task"], strict=True)
        r4direct = Scorer("action", layers=cfg["direct_layers"]).to(device).eval()
        r4direct.load_state_dict(r4["networks"]["direct"], strict=True)
        offline = {"CODE_parent": evaluate("reader", frozen["reader"], frozen["reader"], data, goals, goal_idx, scale, cfg, device, amp)[0]["choice"],
                   "CODE6": evaluate("reader", reader, reader, data, goals, goal_idx, scale, cfg, device, amp)[0]["choice"],
                   "CTA4": evaluate("task", r4task, frozen["reader"], data, goals, goal_idx, scale, cfg, device, amp)[0]["choice"],
                   "CTA6": evaluate("task", task, reader, data, goals, goal_idx, scale, cfg, device, amp)[0]["choice"],
                   "DIRECT4": evaluate("direct", r4direct, None, data, goals, goal_idx, scale, cfg, device, amp)[0]["choice"],
                   "DIRECT6": evaluate("direct", direct, None, data, goals, goal_idx, scale, cfg, device, amp)[0]["choice"]}
        write_json(a.run / "offline16.json", offline)
        report.update(status="DONE", stage="done", seconds=time.perf_counter() - started)
        write_json(path, report)
        print(json.dumps({k: {h: {m: round(x, 3) for m, x in v.items()} for h, v in d.items()}
                          for k, d in offline.items()}), flush=True)
    except Exception:
        report.update(status="FAILED", error=traceback.format_exc())
        write_json(path, report)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("run", "parent", "features", "r4"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--updates-a", type=int, default=3000)
    p.add_argument("--updates-b", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--limit", type=int, default=None, help="smoke only: train banks")
    args = p.parse_args()
    require_compute()
    args.run.mkdir(parents=True, exist_ok=True)
    train(args)
