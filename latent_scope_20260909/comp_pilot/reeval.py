"""Re-evaluate the frozen 52634 checkpoints with a corrected readout, plus raw-input probes.

Fixes relative to the 52634 evaluation (see docs/COMP_PILOT_REEVAL_PROTOCOL.md):
  * readout input normalization: train-set per-feature standardization instead of
    LayerNorm over the token dimension (LayerNorm(1) zeroed every map-cell token);
  * map quality by average precision, with a train cell-frequency prior baseline;
  * constant baselines fitted on train windows, reported for all and contact windows;
  * map_union native heads evaluated directly, before any readout;
  * two readout settings for long windows:
      S1 "unseen": readout trained on direct 32/64 predictions only (as locked before);
      S2 "calibrated": readout trained on train-episode representations from the same
         inference path, including 128-step totals; the predictor still never saw 128.
Raw-input probes (supervised, no world model): observed history + observed window frames
  dino_proprio_action, dino_action, proprio_action.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DIRECT_LENGTHS, EVAL_PARTS, HISTORY_TOKENS, TRAIN_LENGTHS, FeatureStore, _label_worker, eval_starts, train_starts
from .labels import MAP_CELLS
from .models import ARMS, MAP_SIZE, SUMMARY_TOKENS, FrameFuser, sinusoid, transformer
from .train import truncate

import multiprocessing as mp

SCALARS = ("count_increment", "count_empty", "area_empty")
ARM_NAMES = ("segment_nocomp", "segment_comp", "frame_rollout", "map_union")


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Labels for every window used here (train/val/test)


class Labels:
    def __init__(self, store: FeatureStore, workers: int) -> None:
        tasks = []
        for e in store.episodes:
            n = store.lengths[e]
            windows, hard_specs = set(), []
            for length in TRAIN_LENGTHS:
                for t in range(0, n - length, 8):
                    windows.add((t, t + length))
            for t in eval_starts(n):
                for length in DIRECT_LENGTHS:
                    windows.add((t, t + length))
                for name, parts in EVAL_PARTS.items():
                    cut = t
                    for p in parts:
                        windows.add((cut, cut + p))
                        cut += p
                    hard_specs.append((t, name, parts))
            tasks.append((str(store.label_root), e, sorted(windows), hard_specs))
        self.table, self.hard = {}, {}
        with mp.get_context("fork").Pool(workers) as pool:
            for episode, out, hard in pool.imap_unordered(_label_worker, tasks):
                for key, value in out.items():
                    self.table[(episode, *key)] = value
                for key, value in hard.items():
                    self.hard[(episode, *key)] = value
        self.device = store.device

    def get(self, eps, windows) -> dict:
        rows = [self.table[(e, a, b)] for e, (a, b) in zip(eps, windows)]
        scal = torch.tensor([[r[0], r[1], r[2], r[3]] for r in rows], dtype=torch.float32, device=self.device)
        maps = np.unpackbits(np.stack([r[4] for r in rows]), axis=1)[:, : MAP_CELLS * MAP_CELLS]
        return {
            "count_increment": scal[:, 0],
            "count_empty": scal[:, 1],
            "area_empty": scal[:, 2],
            "has_contact": scal[:, 3],
            "map": torch.from_numpy(maps.astype(np.float32)).to(self.device),
        }


# ---------------------------------------------------------------------------
# Corrected readout


class StandardizedReadout(nn.Module):
    """Same attention-pool readout as 52634, but inputs standardized with train statistics."""

    def __init__(self, token_dim: int, mean: torch.Tensor, std: torch.Tensor, width: int = 128, max_tokens: int = 256) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.inp = nn.Linear(token_dim, width)
        self.pos = nn.Parameter(torch.randn(1, max_tokens, width) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.body = transformer(width, 1, heads=4)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3 + MAP_SIZE))

    def forward(self, tokens):
        x = self.inp((tokens.float() - self.mean) / self.std) + self.pos[:, : tokens.shape[1]]
        x = self.body(torch.cat([self.query.expand(len(x), -1, -1), x], 1))[:, 0]
        out = self.head(x)
        return out[:, :3], out[:, 3:]


def loss_fn(scal, logits, target):
    goal = torch.stack([target["count_increment"], target["count_empty"], target["area_empty"] / 10.0], 1)
    return F.smooth_l1_loss(scal, goal) + F.binary_cross_entropy_with_logits(logits, target["map"])


def cat_targets(parts: list[dict]) -> dict:
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def fit_readout(train_sets: list[tuple[torch.Tensor, dict]], val_sets: list[tuple[torch.Tensor, dict]], cfg: dict, seed: int, device) -> tuple[nn.Module, dict]:
    """Each set holds one token count; batches alternate between sets."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    flat = torch.cat([t.reshape(-1, t.shape[-1]).float() for t, _ in train_sets])
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-4)
    readout = StandardizedReadout(flat.shape[-1], mean, std).to(device)
    opt = torch.optim.AdamW(readout.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state, steps = float("inf"), None, cfg["steps"]
    for step in range(steps):
        readout.train()
        tokens, target = train_sets[step % len(train_sets)]
        idx = torch.from_numpy(rng.integers(len(tokens), size=min(cfg["batch_size"], len(tokens)))).to(device)
        loss = loss_fn(*readout(tokens[idx]), {k: v[idx] for k, v in target.items()})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % cfg["val_every"] == 0 or step == steps - 1:
            readout.eval()
            with torch.no_grad():
                val = float(np.mean([float(batched_loss(readout, t, g)) for t, g in val_sets]))
            if val < best:
                best, best_state = val, {k: v.detach().clone() for k, v in readout.state_dict().items()}
    readout.load_state_dict(best_state)
    readout.eval()
    return readout, {"best_val_loss": best}


@torch.no_grad()
def batched_loss(readout, tokens, target, chunk=1024):
    total = 0.0
    for i in range(0, len(tokens), chunk):
        total += float(loss_fn(*readout(tokens[i : i + chunk]), {k: v[i : i + chunk] for k, v in target.items()})) * len(tokens[i : i + chunk])
    return total / len(tokens)


@torch.no_grad()
def apply_readout(readout, tokens, chunk=1024) -> dict:
    scal, logits = [], []
    for i in range(0, len(tokens), chunk):
        s, l = readout(tokens[i : i + chunk])
        scal.append(s)
        logits.append(l)
    s, l = torch.cat(scal), torch.cat(logits)
    return {"count_increment": s[:, 0], "count_empty": s[:, 1], "area_empty": s[:, 2] * 10.0, "map": torch.sigmoid(l)}


# ---------------------------------------------------------------------------
# Representation extraction for every inference path


@torch.no_grad()
def extract(model, store, pool, name: str) -> dict:
    """pool: list of (episode, t) with t+128 inside the episode. Returns path -> tokens (fp16)."""
    model.eval()
    out: dict[str, list] = {}

    def put(key, value):
        out.setdefault(key, []).append(value.half())

    for i in range(0, len(pool), 128):
        picks = pool[i : i + 128]
        batch = store.batch([p[0] for p in picks], [p[1] for p in picks], 128)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for length in DIRECT_LENGTHS:
                sub = truncate(batch, length)
                put(f"direct{length}", model.rollout(sub, [length], composed=False)["parts"][0])
                if hasattr(model, "true_representation"):
                    put(f"true{length}", model.true_representation(sub))
                if name == "map_union":
                    put(f"native_direct{length}", native_map_scalars(model, sub, [length])["direct"])
            for eval_name, parts in EVAL_PARTS.items():
                total = sum(parts)
                roll = model.rollout(truncate(batch, total), parts, composed=True)
                for k, rep in enumerate(roll["parts"]):
                    put(f"{eval_name}/part{k}", rep)
                if "composed" in roll:
                    put(f"{eval_name}/composed", roll["composed"])
                if name == "map_union":
                    put(f"{eval_name}/native", native_map_scalars(model, truncate(batch, total), parts)["combined"])
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def native_map_scalars(model, batch, parts) -> dict:
    """map_union heads before any readout: [B, 3 + 144] with sigmoid map; parts combined by OR / sum."""
    h, z = model.context(batch)
    maps, scal, start = [], [], 0
    for length in parts:
        out = model.predictor(h, z, batch["actions"][:, start : start + length])
        mp_, sc, e = model.heads(out)
        maps.append(torch.sigmoid(mp_.float()))
        scal.append(sc.float() * torch.tensor([1.0, 1.0, 10.0], device=sc.device))
        h, z, start = model.update(h, out[:, :SUMMARY_TOKENS], e), e, start + length
    direct = torch.cat([scal[0], maps[0]], 1)
    combined = torch.cat([sum(scal), torch.stack(maps).amax(0)], 1)
    return {"direct": direct, "combined": combined}


def native_to_pred(native: torch.Tensor) -> dict:
    n = native.float()
    return {"count_increment": n[:, 0], "count_empty": n[:, 1], "area_empty": n[:, 2], "map": n[:, 3:]}


# ---------------------------------------------------------------------------
# Metrics


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = labels.sum()
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / positives)


def window_metrics(pred: dict, target: dict, prior: np.ndarray) -> dict:
    """Per-window arrays plus pooled map AP."""
    p_map = pred["map"].float().cpu().numpy()
    t_map = target["map"].cpu().numpy()
    has = target["has_contact"].cpu().numpy() > 0
    per_ap = [average_precision(p_map[i], t_map[i]) for i in range(len(t_map))]
    return {
        "pred": {k: pred[k].float().cpu().numpy() for k in SCALARS},
        "true": {k: target[k].cpu().numpy() for k in SCALARS},
        "has_contact": has,
        "window_ap": np.array([np.nan if a is None else a for a in per_ap]),
        "micro_ap_contact": average_precision(p_map[has].reshape(-1), t_map[has].reshape(-1)) if has.any() else None,
        "micro_ap_prior_contact": average_precision(np.tile(prior, (int(has.sum()), 1)).reshape(-1), t_map[has].reshape(-1)) if has.any() else None,
    }


def summarize_metrics(m: dict, keep: np.ndarray, constants: dict) -> dict:
    out = {"windows": int(keep.sum())}
    for k in SCALARS:
        t, p = m["true"][k][keep], m["pred"][k][keep]
        out[f"mae_{k}"] = float(np.abs(t - p).mean()) if len(t) else None
        out[f"r_{k}"] = float(np.corrcoef(t, p)[0, 1]) if len(t) > 2 and t.std() > 0 and p.std() > 0 else None
        out[f"const_train_mae_{k}"] = float(np.abs(t - constants[k]).mean()) if len(t) else None
    ap = m["window_ap"][keep]
    out["mean_window_ap"] = float(np.nanmean(ap)) if np.isfinite(ap).any() else None
    return out


def episode_bootstrap_r(rows_a: dict, rows_b: dict, episodes: np.ndarray, keep: np.ndarray, metric: str, rng, resamples=1000) -> dict:
    ta, pa = rows_a["true"][metric], rows_a["pred"][metric]
    pb = rows_b["pred"][metric]
    uniq = np.unique(episodes[keep])
    index = {e: np.flatnonzero((episodes == e) & keep) for e in uniq}

    def stat(sel):
        t, x, y = ta[sel], pa[sel], pb[sel]
        if t.std() == 0 or x.std() == 0 or y.std() == 0:
            return np.nan, np.nan
        return np.corrcoef(t, x)[0, 1] - np.corrcoef(t, y)[0, 1], np.abs(t - x).mean() - np.abs(t - y).mean()

    base = stat(np.concatenate([index[e] for e in uniq]))
    boots = []
    for _ in range(resamples):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots.append(stat(np.concatenate([index[e] for e in pick])))
    boots = np.array(boots)
    return {
        "episodes": int(len(uniq)),
        "r_difference": float(base[0]),
        "r_ci95": [float(np.nanquantile(boots[:, 0], 0.025)), float(np.nanquantile(boots[:, 0], 0.975))],
        "mae_difference": float(base[1]),
        "mae_ci95": [float(np.nanquantile(boots[:, 1], 0.025)), float(np.nanquantile(boots[:, 1], 0.975))],
    }


# ---------------------------------------------------------------------------
# Raw-input probes


class RawProbe(nn.Module):
    """Supervised readout from observed inputs: stride-4 history + every window frame."""

    def __init__(self, variant: str, dim: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.variant = variant
        self.dim = dim
        self.use_visual = "dino" in variant
        self.use_proprio = "proprio" in variant
        self.visual = FrameFuser(dim) if self.use_visual else None
        self.proprio = nn.Linear(16, dim) if self.use_proprio else None
        self.action = nn.Linear(12, dim)
        self.segment = nn.Parameter(torch.randn(1, 2, 1, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.body = transformer(dim, depth)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3 + MAP_SIZE))

    def frames(self, visual, proprio, actions):
        x = self.action(actions.float())
        if self.use_visual:
            x = x + self.visual(visual, proprio if self.use_proprio else torch.zeros_like(proprio))
        if self.use_proprio:
            x = x + self.proprio(proprio.float())
        return x

    def forward(self, batch):
        length = batch["actions"].shape[1]
        hist = self.frames(batch["hist_visual"], batch["hist_proprio"], batch["hist_actions"])
        hist = hist + sinusoid(torch.arange(HISTORY_TOKENS, device=hist.device) - HISTORY_TOKENS, self.dim)[None] + self.segment[:, 0]
        win = self.frames(batch["fut_visual"], batch["fut_proprio"], batch["actions"])
        win = win + sinusoid(torch.arange(length, device=win.device), self.dim)[None] + self.segment[:, 1]
        b = len(hist)
        tokens = torch.cat([self.cls.expand(b, -1, -1), hist, win], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=hist.device), ~batch["hist_mask"], torch.zeros(b, length, dtype=torch.bool, device=hist.device)], 1)
        out = self.head(self.body(tokens, src_key_padding_mask=pad)[:, 0])
        return out[:, :3], out[:, 3:]


def train_probe(variant, store, labels, cfg, seed, device) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    probe = RawProbe(variant).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    pools = {L: [(e, t) for e in store.split_episodes("train") for t in eval_starts(store.lengths[e])] for L in DIRECT_LENGTHS}
    val_pool = [(e, t) for e in store.split_episodes("val") for t in eval_starts(store.lengths[e])]
    best, best_state, start = float("inf"), None, time.time()
    for step in range(steps):
        if time.time() - start > cfg["time_budget_seconds"]:
            break
        probe.train()
        length = DIRECT_LENGTHS[step % len(DIRECT_LENGTHS)]
        pick = rng.integers(len(pools[length]), size=cfg["batch_size"])
        eps, ts = [pools[length][i][0] for i in pick], [pools[length][i][1] for i in pick]
        batch = store.batch(eps, ts, length)
        target = labels.get(eps, [(t, t + length) for t in ts])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scal, logits = probe(batch)
        loss = loss_fn(scal.float(), logits.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % cfg["val_every"] == 0 or step == steps - 1:
            val = probe_loss(probe, store, labels, val_pool)
            if val < best:
                best, best_state = val, {k: v.detach().clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    probe.eval()
    return probe, {"best_val_loss": best, "steps_done": step + 1, "seconds": time.time() - start}


@torch.no_grad()
def probe_loss(probe, store, labels, pool) -> float:
    probe.eval()
    total, count = 0.0, 0
    for length in DIRECT_LENGTHS:
        for i in range(0, len(pool), 256):
            picks = pool[i : i + 256]
            eps, ts = [p[0] for p in picks], [p[1] for p in picks]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scal, logits = probe(store.batch(eps, ts, length))
            total += float(loss_fn(scal.float(), logits.float(), labels.get(eps, [(t, t + length) for t in ts]))) * len(picks)
            count += len(picks)
    return total / count


@torch.no_grad()
def probe_predict(probe, store, pool, length) -> dict:
    scal, maps = [], []
    for i in range(0, len(pool), 256):
        picks = pool[i : i + 256]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s, l = probe(store.batch([p[0] for p in picks], [p[1] for p in picks], length))
        scal.append(s.float())
        maps.append(torch.sigmoid(l.float()))
    s = torch.cat(scal)
    return {"count_increment": s[:, 0], "count_empty": s[:, 1], "area_empty": s[:, 2] * 10.0, "map": torch.cat(maps)}


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("re-evaluation must run inside an sbatch GPU job")
    cfg = json.loads(args.config.read_text())
    rcfg = cfg["reeval"]
    if args.profile:
        rcfg["readout"].update(rcfg["profile"]["readout"])
        rcfg["probe"].update(rcfg["profile"]["probe"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "reeval_result.json"
    result = {"verdict": "RUNNING", "job_id": os.environ["SLURM_JOB_ID"], "node": platform.node(),
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(args.source_run), "profile": args.profile, "reeval_config": rcfg}
    write_json(result_path, result)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = cfg["training"]["seed"]
    try:
        t0 = time.time()
        d = cfg["data"]
        store = FeatureStore(Path(d["feature_root"]), Path(d["label_root"]), Path(d["label_audit"]), device, rcfg["profile"]["max_episodes"] if args.profile else None)
        labels = Labels(store, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
        store.load()
        result["setup_seconds"] = time.time() - t0
        rng = np.random.default_rng(seed)

        pools = {s: [(e, t) for e in store.split_episodes(s) for t in eval_starts(store.lengths[e])] for s in ("train", "val", "test")}
        if args.profile:
            pools = {s: p[: rcfg["profile"]["max_windows"]] for s, p in pools.items()}
        test_eps = np.array([p[0] for p in pools["test"]])

        def targets(split, key):
            """Targets of total length `directN` on the pool of a split."""
            total = int(key.replace("direct", ""))
            pool = pools[split]
            return labels.get([p[0] for p in pool], [(p[1], p[1] + total) for p in pool])

        # train-set constants and cell prior per total length
        constants, priors = {}, {}
        for total in DIRECT_LENGTHS:
            tr = labels.get([p[0] for p in pools["train"]], [(p[1], p[1] + total) for p in pools["train"]])
            has = tr["has_contact"] > 0
            constants[total] = {
                "all": {k: float(tr[k].mean()) for k in SCALARS},
                "contact_fit": {k: float(tr[k][has].mean()) for k in SCALARS},
            }
            priors[total] = tr["map"].mean(0).cpu().numpy()
        result["train_constants"] = {str(k): v for k, v in constants.items()}

        test_targets = {total: labels.get([p[0] for p in pools["test"]], [(p[1], p[1] + total) for p in pools["test"]]) for total in DIRECT_LENGTHS}
        hard = {name: np.array([labels.hard[(e, t, name)] for e, t in pools["test"]]) for name in EVAL_PARTS}
        has_contact = {total: test_targets[total]["has_contact"].cpu().numpy() > 0 for total in DIRECT_LENGTHS}
        metrics_store: dict[str, dict] = {}
        table: dict[str, dict] = {}

        def record(label_key: str, pred: dict, total: int, eval_name: str | None):
            m = window_metrics(pred, test_targets[total], priors[total])
            metrics_store[label_key] = m
            strata = {"all": np.ones_like(has_contact[total]), "contact": has_contact[total]}
            if eval_name is not None:
                strata["hard"] = hard[eval_name] & has_contact[total]
                strata["easy_contact"] = (~hard[eval_name]) & has_contact[total]
            for stratum, keep in strata.items():
                const = constants[total]["contact_fit" if stratum != "all" else "all"]
                entry = summarize_metrics(m, keep.astype(bool), const)
                if stratum == "contact":
                    entry["micro_ap_contact"] = m["micro_ap_contact"]
                    entry["micro_ap_prior_contact"] = m["micro_ap_prior_contact"]
                table[f"{label_key}|{stratum}"] = entry

        rcfg_r = rcfg["readout"]
        for arm_index, name in enumerate(ARM_NAMES):
            ck = torch.load(args.source_run / name / "model.pt", map_location="cpu", weights_only=False)
            model = ARMS[name](cfg["training"]["dim"], cfg["training"]["predictor_depth"]).to(device)
            model.load_state_dict(ck["model"])
            model.eval()
            reps = {s: extract(model, store, pools[s], name) for s in ("train", "val", "test")}
            # S1 readout: direct 32/64 on stride-8 train windows (as locked in 52634), fixed normalization
            s1_train, s1_val = [], []
            for length in TRAIN_LENGTHS:
                for split, target_list in (("train", s1_train), ("val", s1_val)):
                    pool = train_starts(store, split, length, 8 if split == "train" else 16)
                    if args.profile:
                        pool = pool[: rcfg["profile"]["max_windows"]]
                    toks = []
                    for i in range(0, len(pool), 256):
                        picks = pool[i : i + 256]
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            toks.append(model.rollout(store.batch([p[0] for p in picks], [p[1] for p in picks], length), [length], composed=False)["parts"][0].float())
                    target_list.append((torch.cat(toks), labels.get([p[0] for p in pool], [(p[1], p[1] + length) for p in pool])))
            s1, s1_info = fit_readout(s1_train, s1_val, rcfg_r, seed + 1, device)
            # S2 calibrated readouts: same path on train episodes, including 128 totals
            s2_direct, _ = fit_readout(
                [(reps["train"][f"direct{L}"], targets("train", f"direct{L}")) for L in DIRECT_LENGTHS],
                [(reps["val"][f"direct{L}"], targets("val", f"direct{L}")) for L in DIRECT_LENGTHS], rcfg_r, seed + 2, device)
            arm_info = {"s1_readout": s1_info}
            for L in DIRECT_LENGTHS:
                for setting, ro in (("S1", s1), ("S2", s2_direct)):
                    record(f"{name}|direct{L}|direct|{setting}", apply_readout(ro, reps["test"][f"direct{L}"]), L, None)
                if f"true{L}" in reps["test"]:
                    for setting, train_lengths in (("S1", TRAIN_LENGTHS), ("S2", DIRECT_LENGTHS)):
                        ro, _ = fit_readout([(reps["train"][f"true{x}"], targets("train", f"direct{x}")) for x in train_lengths],
                                            [(reps["val"][f"true{x}"], targets("val", f"direct{x}")) for x in train_lengths], rcfg_r, seed + 3, device)
                        record(f"{name}|direct{L}|true_summary_ceiling|{setting}", apply_readout(ro, reps["test"][f"true{L}"]), L, None)
                if name == "map_union":
                    record(f"{name}|direct{L}|native_heads|none", native_to_pred(reps["test"][f"native_direct{L}"]), L, None)
            for eval_name, parts in EVAL_PARTS.items():
                total = sum(parts)
                for setting, ro in (("S1", s1), ("S2", s2_direct)):
                    preds = [apply_readout(ro, reps["test"][f"{eval_name}/part{k}"]) for k in range(len(parts))]
                    additive = {k: sum(p[k] for p in preds) for k in SCALARS}
                    additive["map"] = torch.stack([p["map"] for p in preds]).amax(0)
                    record(f"{name}|{eval_name}|additive|{setting}", additive, total, eval_name)
                    record(f"{name}|{eval_name}|direct_whole|{setting}", apply_readout(ro, reps["test"][f"direct{total}"]), total, eval_name)
                if f"{eval_name}/composed" in reps["test"]:
                    record(f"{name}|{eval_name}|composed|S1", apply_readout(s1, reps["test"][f"{eval_name}/composed"]), total, eval_name)
                    ro, _ = fit_readout([(reps["train"][f"{eval_name}/composed"], targets("train", f"direct{total}"))],
                                        [(reps["val"][f"{eval_name}/composed"], targets("val", f"direct{total}"))], rcfg_r, seed + 4, device)
                    record(f"{name}|{eval_name}|composed|S2", apply_readout(ro, reps["test"][f"{eval_name}/composed"]), total, eval_name)
                if name == "map_union":
                    record(f"{name}|{eval_name}|native_heads|none", native_to_pred(reps["test"][f"{eval_name}/native"]), total, eval_name)
            result.setdefault("arms", {})[name] = arm_info
            result["table"] = table
            write_json(result_path, result)
            del model, reps
            torch.cuda.empty_cache()
            print(f"re-evaluated {name}", flush=True)

        # raw-input probes
        result["probes"] = {}
        for variant in rcfg["probe"]["variants"]:
            probe, info = train_probe(variant, store, labels, rcfg["probe"], seed + 10, device)
            result["probes"][variant] = info
            for L in DIRECT_LENGTHS:
                record(f"probe_{variant}|direct{L}|observed|none", probe_predict(probe, store, pools["test"], L), L, None)
            result["table"] = table
            write_json(result_path, result)
            del probe
            torch.cuda.empty_cache()
            print(f"probe {variant} done", flush=True)

        # paired episode bootstrap, scale-free (Pearson r) and MAE, on key comparisons
        comparisons = []
        for eval_name in EVAL_PARTS:
            for setting in ("S1", "S2"):
                a = f"segment_comp|{eval_name}|composed|{setting}"
                for b in (f"segment_nocomp|{eval_name}|additive|{setting}", f"segment_nocomp|{eval_name}|direct_whole|{setting}",
                          f"segment_comp|{eval_name}|additive|{setting}", f"frame_rollout|{eval_name}|composed|{setting}",
                          f"map_union|{eval_name}|native_heads|none"):
                    comparisons.append((eval_name, a, b))
        for L in DIRECT_LENGTHS:
            for setting in ("S1", "S2"):
                comparisons.append((None, f"segment_comp|direct{L}|direct|{setting}", f"segment_nocomp|direct{L}|direct|{setting}"))
        paired = {}
        for eval_name, a, b in comparisons:
            if a not in metrics_store or b not in metrics_store:
                continue
            total = int(eval_name.split("=")[0]) if eval_name else int(a.split("|")[1].replace("direct", ""))
            strata = {"contact": has_contact[total]}
            if eval_name:
                strata["hard"] = hard[eval_name] & has_contact[total]
            for stratum, keep in strata.items():
                for metric in ("count_empty", "count_increment"):
                    paired[f"{a} vs {b}|{stratum}|{metric}"] = episode_bootstrap_r(metrics_store[a], metrics_store[b], test_eps, keep.astype(bool), metric, rng, 300 if args.profile else 1000)
        result["paired_episode_bootstrap"] = paired
        result["verdict"] = "PROFILE_COMPLETE" if args.profile else "REEVAL_COMPLETE"
        result["total_seconds"] = time.time() - t0
        write_json(result_path, result)
        print(f"WROTE {result_path}", flush=True)
    except Exception as error:
        result.update({"verdict": "ERROR", "error": repr(error), "traceback": traceback.format_exc()})
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
