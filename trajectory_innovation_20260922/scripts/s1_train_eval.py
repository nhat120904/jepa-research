"""Gate S1: train the two sibling readers and evaluate within-bank resolution offline.

docs/SIBLING_READER_OFFLINE_PROTOCOL.md. One GPU job; reads the S1 collection shards.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from ti_wm.contract import require_compute, select_candidate
from ti_wm.gates import cluster_ratio
from ti_wm.progress import ProgressReader, pool_tokens, proprio_vector
from ti_wm.pusht_runtime import VisualScorer
from ti_wm.sibling import (
    RANK_MARGIN, SiblingReader, contrast_loss, contrast_targets, fit_pca, project, rank_loss, retained_gap,
)

K = 8
STEPS, DECISIONS, LR, WD = 10_000, 16, 3e-4, 0.05
PCA_FRAMES = 2000
ADV_RHO, ADV_GAP, PREFER_LABEL_FREE = 0.30, 0.40, 0.8
ENC_BATCH = 512


def load(run, lo, hi):
    shards = sorted(p for p in Path(run).glob("shard_*.npz") if lo <= int(p.stem.split("_")[1]) <= hi)
    parts = [dict(np.load(p)) for p in shards]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, [p.name for p in shards]


class Encoder:
    def __init__(self, visual):
        self.visual = visual
        self.mean = self.basis = None

    @torch.inference_mode()
    def raw(self, frames):
        return torch.cat([self.visual.features(frames[s:s + ENC_BATCH]).half()
                          for s in range(0, len(frames), ENC_BATCH)])

    def fit(self, frames):
        tokens = self.raw(frames).reshape(-1, 384)
        self.mean, self.basis = fit_pca(tokens)

    @torch.inference_mode()
    def pca(self, frames):
        """(N, ...) uint8 frames -> (N, ..., 256, PCA_DIM) fp16."""
        lead = frames.shape[:-3]
        flat = frames.reshape(-1, *frames.shape[-3:])
        out = torch.cat([project(self.visual.features(flat[s:s + ENC_BATCH]), self.mean, self.basis).half()
                         for s in range(0, len(flat), ENC_BATCH)])
        return out.reshape(*lead, *out.shape[1:])


def proprio(data):
    return torch.from_numpy(proprio_vector(data["end_pos"], data["end_prev_pos"]))


def train(kind, feats, data, goals, device, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SiblingReader().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    cov = torch.from_numpy(data["cov8"]).float().to(device)
    phys = torch.from_numpy(data["phys8"]).float().to(device)
    prop = proprio(data).to(device)
    pool = (np.arange(len(cov)) if kind == "a"
            else np.flatnonzero(np.ptp(data["cov8"], axis=1) > RANK_MARGIN))
    losses = []
    model.train()
    for step in range(STEPS):
        idx = torch.from_numpy(pool[rng.integers(0, len(pool), DECISIONS)]).to(device)
        end = feats["end"][idx].flatten(0, 1)
        ctx = feats["ctx"][idx][:, None].expand(-1, K, -1, -1).flatten(0, 1)
        if kind == "a":
            k = torch.from_numpy(rng.integers(0, K, DECISIONS)).to(device)
            goal = feats["goal16"][idx, k]
        else:
            goal = goals[torch.from_numpy(rng.integers(0, len(goals), DECISIONS)).to(device)]
        goal = goal[:, None].expand(-1, K, -1, -1).flatten(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(end, ctx, goal, prop[idx].flatten(0, 1)).float().view(DECISIONS, K)
        loss = contrast_loss(scores, contrast_targets(phys[idx], k)) if kind == "a" else rank_loss(scores, cov[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if step % 1000 == 0:
            print(f"S1-{kind} step {step} loss {np.mean(losses[-200:]):.4f}", flush=True)
    return model.eval(), float(np.mean(losses[-500:]))


@torch.inference_mode()
def score_goals(model, feats, data, goals, device, per_batch=8):
    """Mean over test goals of the reader output, for every held-out sibling: (N, K)."""
    prop = proprio(data).to(device)
    g = len(goals)
    out = []
    for s in range(0, len(prop), per_batch):
        idx = slice(s, s + per_batch)
        n = feats["end"][idx].shape[0]
        end = feats["end"][idx].flatten(0, 1).repeat_interleave(g, 0)
        ctx = feats["ctx"][idx][:, None].expand(-1, K, -1, -1).flatten(0, 1).repeat_interleave(g, 0)
        goal = goals.repeat(n * K, 1, 1)
        p = prop[idx].flatten(0, 1).repeat_interleave(g, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(end, ctx, goal, p).float().view(n, K, g).mean(-1).cpu())
    return torch.cat(out).numpy()


@torch.inference_mode()
def contrast_accuracy(model, feats, data, device, seed=1):
    rng = np.random.default_rng(seed)
    phys = torch.from_numpy(data["phys8"]).float().to(device)
    prop = proprio(data).to(device)
    hits = total = 0
    for s in range(0, len(prop), 64):
        idx = torch.arange(s, min(s + 64, len(prop)), device=device)
        n = len(idx)
        k = torch.from_numpy(rng.integers(0, K, n)).to(device)
        target = contrast_targets(phys[idx], k)
        informative = (target > 0).sum(-1) < K
        goal = feats["goal16"][idx, k][:, None].expand(-1, K, -1, -1).flatten(0, 1)
        ctx = feats["ctx"][idx][:, None].expand(-1, K, -1, -1).flatten(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sc = model(feats["end"][idx].flatten(0, 1), ctx, goal, prop[idx].flatten(0, 1)).float().view(n, K)
        hit = target[torch.arange(n), sc.argmax(-1)] > 0
        hits += int(hit[informative].sum())
        total += int(informative.sum())
    return {"top1": hits / max(total, 1), "chance_upper_bound": 1 / 2, "n": total}


def evaluate(scores, data, label):
    lab = data[label]
    roots = data["root"]
    uniq = np.unique(roots)
    rho_sum, rho_n, gap_num, gap_den = (np.zeros(len(uniq)) for _ in range(4))
    pos = {r: i for i, r in enumerate(uniq)}
    for n in range(len(lab)):
        i = pos[roots[n]]
        if np.ptp(lab[n]) > 0 and np.ptp(scores[n]) > 0:
            rho_sum[i] += spearmanr(scores[n], lab[n]).statistic
            rho_n[i] += 1
        num, den = retained_gap([float(x) for x in scores[n]], lab[n], select_candidate)
        gap_num[i] += num
        gap_den[i] += den
    keep = rho_n > 0
    return {"within_bank_spearman": cluster_ratio(rho_sum[keep], rho_n[keep]),
            "retained_gap": cluster_ratio(gap_num, gap_den),
            "informative_decisions": int(rho_n.sum()), "decisions": int(len(lab))}


def main(run, collect, prep, smoke, c2):
    require_compute()
    torch.backends.cudnn.benchmark = False
    report = {"status": "RUNNING"}
    try:
        t0 = time.perf_counter()
        train_data, train_shards = load(collect, 30000, 39999)
        val_data, val_shards = load(collect, 1500, 1599)
        assert len(train_shards) == 5 and len(val_shards) == 2, (train_shards, val_shards)
        report["shards"] = {"train": train_shards, "val": val_shards}
        report["decisions"] = {"train": int(len(train_data["root"])), "val": int(len(val_data["root"]))}
        visual = VisualScorer()
        device = visual.device
        enc = Encoder(visual)
        rng = np.random.default_rng(0)
        enc.fit(train_data["ctx"][rng.choice(len(train_data["ctx"]), PCA_FRAMES, replace=False)])
        goal_frames = np.load(smoke / "goal_frames.npz")["frames"]
        # clone() outside inference mode: the features are used as autograd inputs during training.
        goals = enc.pca(goal_frames).clone()
        feats = {k: enc.pca(train_data[k]).clone() for k in ("ctx", "end", "goal16")}
        vfeats = {k: enc.pca(val_data[k]).clone() for k in ("ctx", "end", "goal16")}
        report["encode_seconds"] = time.perf_counter() - t0

        # Reference scorers on the held-out siblings.
        goal_map = torch.load(smoke / "goal_map.pt").to(device)
        raw_goals = pool_tokens(enc.raw(goal_frames))
        c2_reader = ProgressReader().to(device)
        c2_reader.load_state_dict(torch.load(c2 / "reader.pt")["state_dict"])
        c2_reader.eval()
        vis, prog = [], []
        vprop = proprio(val_data).to(device)
        with torch.inference_mode():
            for s in range(0, len(val_data["root"]), 32):
                end = val_data["end"][s:s + 32]
                n = len(end)
                cur = enc.raw(end.reshape(-1, *end.shape[-3:]))
                prev = enc.raw(val_data["end_prev"][s:s + 32].reshape(-1, *end.shape[-3:]))
                vis.append((-(cur.float() - goal_map[None]).pow(2).mean(dim=(1, 2))).view(n, K).cpu())
                pc, pp, g = pool_tokens(cur.float()), pool_tokens(prev.float()), len(raw_goals)
                pred = c2_reader(pc.repeat_interleave(g, 0), pp.repeat_interleave(g, 0),
                                 raw_goals.repeat(n * K, 1, 1), vprop[s:s + 32].flatten(0, 1).repeat_interleave(g, 0))
                prog.append((-pred.view(n, K, g).mean(-1)).float().cpu())
        refs = {"c2_reader_r0": torch.cat(prog).numpy(), "gateC_L2": torch.cat(vis).numpy()}
        report["reference"] = {name: {lab: evaluate(sc, val_data, lab) for lab in ("cov8", "cov24")}
                               for name, sc in refs.items()}
        print(json.dumps(report["reference"], indent=1, default=float), flush=True)

        readers = {}
        for kind in ("a", "b"):
            t1 = time.perf_counter()
            model, final_loss = train(kind, feats, train_data, goals, device)
            sc = score_goals(model, vfeats, val_data, goals, device)
            res = {lab: evaluate(sc, val_data, lab) for lab in ("cov8", "cov24")}
            res["final_train_loss"] = final_loss
            res["train_seconds"] = time.perf_counter() - t1
            if kind == "a":
                res["heldout_contrast"] = contrast_accuracy(model, vfeats, val_data, device)
            rho = res["cov8"]["within_bank_spearman"]["ratio"]
            gap = res["cov8"]["retained_gap"]["ratio"]
            res["advances"] = bool(rho >= ADV_RHO and gap >= ADV_GAP)
            readers[f"S1-{kind}"] = res
            torch.save({"state_dict": model.state_dict(), "pca_mean": enc.mean.cpu(), "pca_basis": enc.basis.cpu()},
                       run / f"reader_s1{kind}.pt")
            np.save(run / f"heldout_scores_s1{kind}.npy", sc)
            print(json.dumps({f"S1-{kind}": res}, indent=1, default=float), flush=True)
        report["readers"] = readers
        a, b = readers["S1-a"], readers["S1-b"]
        if a["advances"] and b["advances"]:
            ga, gb = a["cov8"]["retained_gap"]["ratio"], b["cov8"]["retained_gap"]["ratio"]
            choice = "S1-a" if ga >= PREFER_LABEL_FREE * gb else "S1-b"
        elif a["advances"] or b["advances"]:
            choice = "S1-a" if a["advances"] else "S1-b"
        else:
            choice = None
        report["verdict"] = f"ADVANCE {choice}" if choice else "WEEK1_KILL"
        report["status"] = "DONE"
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (run / "s1_report.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "collect", "prep", "smoke", "c2"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    a = parser.parse_args()
    main(a.run, a.collect, a.prep, a.smoke, a.c2)
