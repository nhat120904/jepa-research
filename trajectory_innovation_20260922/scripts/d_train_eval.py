"""Gate D: train the six code/reader arms on actual futures and compare within-bank ranking offline.

docs/GATE_D_CODEC_PROTOCOL.md. One GPU job; reads the gate-D collection shards.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from ti_wm.codec import LEVELS, CodeDecoder, CodeEncoder, CodeReader, bits_per_token, pool_grid
from ti_wm.contract import require_compute, select_candidate
from ti_wm.gates import cluster_ratio
from ti_wm.progress import proprio_vector
from ti_wm.pusht_runtime import VisualScorer
from ti_wm.sibling import PCA_DIM, RANK_MARGIN, fit_pca, project, rank_loss

K = 8
STEPS, DECISIONS, LR, WD, LAMBDA = 10_000, 16, 3e-4, 0.05, 1.0
PCA_FRAMES, ENC_BATCH = 2000, 512
ARMS = {"FULL": None, "COND16": ("cond", 16), "COND4": ("cond", 4), "UNCOND16": ("uncond", 16),
        "TRAJ16": ("traj", 16), "POOL16": ("pool", 16)}
REF_MIN_GAP, PASS_RETENTION, PASS_LO = 0.40, 0.80, 0.50


def load(run, lo, hi):
    shards = sorted(p for p in Path(run).glob("shard_*.npz") if lo <= int(p.stem.split("_")[1]) <= hi)
    parts = [dict(np.load(p)) for p in shards]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, [p.name for p in shards]


class Features:
    def __init__(self, visual):
        self.visual = visual
        self.mean = self.basis = None

    @torch.inference_mode()
    def fit(self, frames):
        raw = torch.cat([self.visual.features(frames[s:s + ENC_BATCH]) for s in range(0, len(frames), ENC_BATCH)])
        self.mean, self.basis = fit_pca(raw.reshape(-1, raw.shape[-1]))

    @torch.inference_mode()
    def pca(self, frames, side=None):
        """(..., H, W, 3) uint8 -> (..., tokens, PCA_DIM) fp16; optionally pooled to side x side."""
        lead = frames.shape[:-3]
        flat = frames.reshape(-1, *frames.shape[-3:])
        out = []
        for s in range(0, len(flat), ENC_BATCH):
            x = project(self.visual.features(flat[s:s + ENC_BATCH]), self.mean, self.basis)
            out.append((pool_grid(x, side) if side else x).half())
        out = torch.cat(out)
        return out.reshape(*lead, *out.shape[1:])


def build(name, device):
    if ARMS[name] is None:
        return None, CodeReader(PCA_DIM, 256).to(device), None
    mode, m = ARMS[name]
    return (CodeEncoder(m, mode).to(device), CodeReader(len(LEVELS), m).to(device),
            CodeDecoder(len(LEVELS), m).to(device))


def code_of(enc, feats, idx):
    end = feats["end"][idx].flatten(0, 1)
    ctx = feats["ctx"][idx][:, None].expand(-1, K, -1, -1).flatten(0, 1)
    if enc is None:
        return ctx, end, end
    seg = feats["seg"][idx].flatten(0, 1) if enc.mode == "traj" else None
    return ctx, end, enc(ctx, end, seg)


def train(name, feats, data, goals, end_var, device, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, reader, dec = build(name, device)
    params = [p for mod in (enc, reader, dec) if mod is not None for p in mod.parameters()]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    cov = torch.from_numpy(data["cov8"]).float().to(device)
    prop = feats["prop"]
    pool = np.flatnonzero(np.ptp(data["cov8"], axis=1) > RANK_MARGIN)
    log = []
    for step in range(STEPS):
        idx = torch.from_numpy(pool[rng.integers(0, len(pool), DECISIONS)]).to(device)
        goal = goals[torch.from_numpy(rng.integers(0, len(goals), DECISIONS)).to(device)]
        goal = goal[:, None].expand(-1, K, -1, -1).flatten(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx, end, code = code_of(enc, feats, idx)
            scores = reader(ctx, code, goal, prop[idx].flatten(0, 1)).float().view(DECISIONS, K)
            rank = rank_loss(scores, cov[idx])
            aux = (((dec(ctx, code).float() - end.float()) ** 2).mean() / end_var) if dec is not None else rank * 0
        loss = rank + LAMBDA * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append((float(rank), float(aux)))
        if step % 1000 == 0:
            r, a = np.mean(log[-200:], axis=0)
            print(f"{name} step {step} rank {r:.4f} aux {a:.4f}", flush=True)
    for mod in (enc, reader, dec):
        if mod is not None:
            mod.eval()
    return (enc, reader, dec), np.mean(log[-500:], axis=0).tolist()


@torch.inference_mode()
def evaluate_arm(models, feats, goals, end_var, per_batch=8):
    enc, reader, dec = models
    g, n = len(goals), feats["end"].shape[0]
    scores, sse = [], 0.0
    for s in range(0, n, per_batch):
        idx = torch.arange(s, min(s + per_batch, n), device=goals.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx, end, code = code_of(enc, feats, idx)
            rows = len(idx) * K
            out = reader(ctx.repeat_interleave(g, 0), code.repeat_interleave(g, 0), goals.repeat(rows, 1, 1),
                         feats["prop"][idx].flatten(0, 1).repeat_interleave(g, 0))
            if dec is not None:
                sse += float(((dec(ctx, code).float() - end.float()) ** 2).mean()) * rows
        scores.append(out.float().view(len(idx), K, g).mean(-1).cpu())
    r2 = (1 - sse / n / K / end_var) if dec is not None else None
    return torch.cat(scores).numpy(), r2


def per_root(scores, data):
    roots, lab = data["root"], data["cov8"]
    uniq = np.unique(roots)
    pos = {r: i for i, r in enumerate(uniq)}
    num, den, rho_sum, rho_n = (np.zeros(len(uniq)) for _ in range(4))
    for n in range(len(lab)):
        i = pos[roots[n]]
        chosen = select_candidate([float(x) for x in scores[n]])
        num[i] += lab[n][chosen] - lab[n][0]
        den[i] += lab[n].max() - lab[n][0]
        if np.ptp(lab[n]) > 0 and np.ptp(scores[n]) > 0:
            rho_sum[i] += spearmanr(scores[n], lab[n]).statistic
            rho_n[i] += 1
    return num, den, rho_sum, rho_n


def main(run, collect, smoke, pipeline_check=False):
    global STEPS
    require_compute()
    report = {"status": "RUNNING", "bits_per_token": bits_per_token(), "pipeline_check": pipeline_check}
    try:
        t0 = time.perf_counter()
        if pipeline_check:
            # Code-path check only (2-root smoke shard used as both splits, 30 steps). Not a result.
            STEPS = 30
            train_data, tr = load(collect, 0, 99999)
            val_data, va = train_data, tr
        else:
            train_data, tr = load(collect, 30000, 39999)
            val_data, va = load(collect, 2000, 2099)
            assert len(tr) == 8 and len(va) == 2, (tr, va)
        report["decisions"] = {"train": int(len(train_data["root"])), "val": int(len(val_data["root"]))}
        visual = VisualScorer()
        device = visual.device
        fx = Features(visual)
        rng = np.random.default_rng(0)
        fx.fit(train_data["ctx"][rng.choice(len(train_data["ctx"]), PCA_FRAMES, replace=False)])
        # clone() outside inference mode: features are autograd inputs during training.
        goals = fx.pca(np.load(smoke / "goal_frames.npz")["frames"]).clone()

        def feats_of(data):
            return {"ctx": fx.pca(data["ctx"]).clone(), "end": fx.pca(data["end"]).clone(),
                    "seg": fx.pca(data["seg"], side=8).clone(),
                    "prop": torch.from_numpy(proprio_vector(data["end_pos"], data["end_prev_pos"])).to(device)}

        feats, vfeats = feats_of(train_data), feats_of(val_data)
        sample = feats["end"][:: max(1, len(feats["end"]) // 512)].float().flatten(0, 2)
        end_var = float(sample.var(0).mean())
        del sample
        report["end_token_var"] = end_var
        report["encode_seconds"] = time.perf_counter() - t0

        arms, per = {}, {}
        for name in ARMS:
            t1 = time.perf_counter()
            models, final = train(name, feats, train_data, goals, end_var, device)
            scores, r2 = evaluate_arm(models, vfeats, goals, end_var)
            num, den, rs, rn = per_root(scores, val_data)
            keep = rn > 0
            per[name] = (num, den)
            arms[name] = {"retained_gap": cluster_ratio(num, den),
                          "within_bank_spearman": cluster_ratio(rs[keep], rn[keep]),
                          "decoder_r2": r2, "final_train_losses": {"rank": final[0], "aux": final[1]},
                          "bits": None if ARMS[name] is None else ARMS[name][1] * bits_per_token(),
                          "train_seconds": time.perf_counter() - t1}
            np.save(run / f"heldout_scores_{name}.npy", scores)
            torch.save({k: m.state_dict() for k, m in zip(("encoder", "reader", "decoder"), models) if m is not None}
                       | {"pca_mean": fx.mean.cpu(), "pca_basis": fx.basis.cpu()}, run / f"arm_{name}.pt")
            print(json.dumps({name: arms[name]}, indent=1, default=float), flush=True)
        report["arms"] = arms
        full_num, den = per["FULL"]
        report["retention_vs_full"] = {n: cluster_ratio(per[n][0], full_num) for n in ARMS if n != "FULL"}
        report["paired_gap_differences"] = {
            f"{a}-{b}": cluster_ratio(per[a][0] - per[b][0], den)
            for a, b in (("COND16", "UNCOND16"), ("TRAJ16", "COND16"), ("COND16", "POOL16"))}
        ref_ok = arms["FULL"]["retained_gap"]["ratio"] >= REF_MIN_GAP
        ret = report["retention_vs_full"]["COND16"]
        if not ref_ok:
            verdict = "NOT_INTERPRETED_REFERENCE_FAILED"
        else:
            verdict = "PASS" if ret["ratio"] >= PASS_RETENTION and ret["lo"] >= PASS_LO else "FAIL"
        claims = {k: bool(v["lo"] > 0) for k, v in report["paired_gap_differences"].items()}
        report["verdict"], report["secondary_claims_supported"] = verdict, claims
        report["status"] = "DONE"
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (run / "d_report.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "collect", "smoke"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pipeline-check", action="store_true")
    a = parser.parse_args()
    main(a.run, a.collect, a.smoke, a.pipeline_check)
