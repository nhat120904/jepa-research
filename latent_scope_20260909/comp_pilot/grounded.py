"""Grounded shared-teacher prototype (docs/COMP_GROUNDED_PROTOCOL.md). sbatch GPU only.

Stage 1: a teacher reads observed window frames (fused DINO + proprio) and history memory.
         It is trained with physical supervision (history-free effect: count, area, contact map,
         first-contact time; history-conditioned contribution: count, area, new cells), a memory
         update U, and a spatial reconstruction anchor. Then it is frozen.
Stage 2: students predict the teacher summary from the observed prefix and proposed actions only.
         segment_nocomp, segment_comp (learned composer) and frame_teacher (ordered frame
         rollout in teacher space) share the teacher, data, heads and supervision. Only the
         composition terms differ between the two segment students.
Evaluation on val_decide (test episodes are dropped before loading); every arm is read through
the same frozen teacher heads.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
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

from .data import EVAL_PARTS, TRAIN_LENGTHS, FeatureStore, eval_starts, train_starts
from .labels import MAP_CELLS, EpisodeLabels
from .models import STRIDE, SUMMARY_TOKENS, Composer, FrameFuser, FramePredictor, Memory, SegmentPredictor, SpatialReconstructionDecoder, TrajectoryEncoder, variance_penalty
from .reeval import average_precision, episode_bootstrap_r
from .reeval_target import val_halves

CELLS = MAP_CELLS * MAP_CELLS
PRIMARY = ("128=64+64", "64=24+40")


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Labels with history-conditioned contribution


class HistoryLabels(EpisodeLabels):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.first_cover: dict[int, int] = {}
        for k, cells in enumerate(self.disc):
            for c in cells:
                self.first_cover.setdefault(c, k)
        self.first_map = np.full(CELLS, np.iinfo(np.int64).max, dtype=np.int64)
        for k, cell in enumerate(self.map_index):
            if cell >= 0 and self.first_map[cell] > k:
                self.first_map[cell] = k

    def grounded_window(self, a: int, b: int) -> tuple:
        base = self.window(a, b)
        p, q = self._points(a, b)
        cells: set[int] = set()
        for k in range(p, q):
            cells |= self.disc[k]
        area_inc = sum(1 for c in cells if self.first_cover[c] >= p) * 0.0025 * 0.0025 * 1e4
        map_new = np.zeros(CELLS, dtype=np.uint8)
        first_time = np.zeros(CELLS, dtype=np.float16)
        for k in range(p, q):
            cell = self.map_index[k]
            if cell < 0:
                continue
            if self.first_map[cell] >= p:
                map_new[cell] = 1
            value = (self.idx[k] - a) / max(b - a, 1)
            if first_time[cell] == 0 or value < first_time[cell]:
                first_time[cell] = max(value, 1e-3)
        return (base["count_increment"], base["count_empty"], base["area_empty"], area_inc, base["has_contact"],
                np.packbits(base["map"].astype(np.uint8)), np.packbits(map_new), first_time)


def _worker(task):
    root, episode, windows = task
    labels = HistoryLabels(Path(root) / f"episode_{episode:06d}.npz")
    return episode, {w: labels.grounded_window(*w) for w in windows}


class GroundedLabels:
    def __init__(self, store: FeatureStore, workers: int, eval_split: str) -> None:
        tasks = []
        for e in store.episodes:
            n, windows = store.lengths[e], set()
            if store.split[e] in ("train", "val_select"):
                for length in TRAIN_LENGTHS:
                    m = length // 2
                    for t in range(0, n - length, STRIDE):
                        windows |= {(t, t + length), (t, t + m), (t + m, t + length)}
            if store.split[e] == eval_split:
                for t in eval_starts(n):
                    for name, parts in EVAL_PARTS.items():
                        cut = t
                        for part in parts:
                            windows.add((cut, cut + part))
                            cut += part
                        windows.add((t, cut))
                    for length in (32, 64, 128):
                        windows.add((t, t + length))
            tasks.append((str(store.label_root), e, sorted(windows)))
        self.table = {}
        with mp.get_context("fork").Pool(workers) as pool:
            for episode, out in pool.imap_unordered(_worker, tasks):
                for key, value in out.items():
                    self.table[(episode, *key)] = value
        self.device = store.device

    def get(self, eps, windows) -> dict:
        rows = [self.table[(e, a, b)] for e, (a, b) in zip(eps, windows)]
        scal = torch.tensor([r[:5] for r in rows], dtype=torch.float32, device=self.device)
        maps = np.unpackbits(np.stack([r[5] for r in rows]), axis=1)[:, :CELLS]
        new = np.unpackbits(np.stack([r[6] for r in rows]), axis=1)[:, :CELLS]
        first = np.stack([r[7] for r in rows]).astype(np.float32)
        dev = self.device
        return {
            "count_increment": scal[:, 0], "count_empty": scal[:, 1], "area_empty": scal[:, 2], "area_increment": scal[:, 3],
            "has_contact": scal[:, 4], "map": torch.from_numpy(maps.astype(np.float32)).to(dev),
            "map_new": torch.from_numpy(new.astype(np.float32)).to(dev), "first_time": torch.from_numpy(first).to(dev),
        }


# ---------------------------------------------------------------------------
# Teacher


class EffectHeads(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        flat = dim * SUMMARY_TOKENS
        self.empty = nn.Sequential(nn.LayerNorm(flat), nn.Linear(flat, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 2 + 2 * CELLS))
        self.inc = nn.Sequential(nn.LayerNorm(flat + dim), nn.Linear(flat + dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 2 + CELLS))

    def forward_empty(self, s: torch.Tensor) -> dict:
        o = self.empty(s.flatten(1).float())
        return {"count_empty": o[:, 0], "area_empty": o[:, 1] * 10.0, "map_logits": o[:, 2 : 2 + CELLS], "first_time": torch.sigmoid(o[:, 2 + CELLS :])}

    def forward_inc(self, m: torch.Tensor, s: torch.Tensor) -> dict:
        o = self.inc(torch.cat([s.flatten(1), m], 1).float())
        return {"count_increment": o[:, 0], "area_increment": o[:, 1] * 10.0, "map_new_logits": o[:, 2:]}


POS_WEIGHT = 5.0


def empty_loss(pred, lab):
    occupied = lab["map"]
    time_loss = ((pred["first_time"] - lab["first_time"]) ** 2 * occupied).sum() / occupied.sum().clamp_min(1.0)
    return (F.smooth_l1_loss(pred["count_empty"], lab["count_empty"]) + F.smooth_l1_loss(pred["area_empty"] / 10, lab["area_empty"] / 10)
            + F.binary_cross_entropy_with_logits(pred["map_logits"], lab["map"], pos_weight=torch.tensor(POS_WEIGHT, device=occupied.device)) + time_loss)


def inc_loss(pred, lab):
    return (F.smooth_l1_loss(pred["count_increment"], lab["count_increment"]) + F.smooth_l1_loss(pred["area_increment"] / 10, lab["area_increment"] / 10)
            + F.binary_cross_entropy_with_logits(pred["map_new_logits"], lab["map_new"], pos_weight=torch.tensor(POS_WEIGHT, device=lab["map_new"].device)))


class SummaryUpdate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * (SUMMARY_TOKENS + 1), 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, m, s):
        return m + self.net(torch.cat([m, s.flatten(1)], 1).float())


class GroundedTeacher(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fuser = FrameFuser(dim)
        self.memory = Memory(dim)
        self.encoder = TrajectoryEncoder(dim, 4)
        self.heads = EffectHeads(dim)
        self.update = SummaryUpdate(dim)
        self.anchor = SpatialReconstructionDecoder(dim)

    def context(self, batch):
        hist = self.fuser(batch["hist_visual"], batch["hist_proprio"])
        m = self.memory(self.memory.tokens(hist, batch["hist_actions"]), batch["hist_mask"])
        return m, self.fuser(batch["cur_visual"], batch["cur_proprio"])

    def continue_memory(self, m, frames, actions, upto):
        pos = torch.arange(STRIDE - 1, upto, STRIDE, device=m.device)
        if len(pos) == 0:
            return m
        return self.memory(self.memory.tokens(frames[:, pos], actions[:, (pos + 1).clamp(max=actions.shape[1] - 1)]), None, m)

    def loss(self, batch, labs, weights):
        length = batch["actions"].shape[1]
        half = length // 2
        m_a, _ = self.context(batch)
        fr = self.fuser(batch["fut_visual"], batch["fut_proprio"])
        s_f, s_l, s_r = self.encoder(fr), self.encoder(fr[:, :half]), self.encoder(fr[:, half:])
        m_mid = self.continue_memory(m_a, fr, batch["actions"], half)
        m_end = self.continue_memory(m_a, fr, batch["actions"], length)
        terms = {
            "empty": empty_loss(self.heads.forward_empty(s_f), labs["full"]) + empty_loss(self.heads.forward_empty(s_l), labs["left"]) + empty_loss(self.heads.forward_empty(s_r), labs["right"]),
            "inc": inc_loss(self.heads.forward_inc(m_a, s_f), labs["full"]) + inc_loss(self.heads.forward_inc(m_a, s_l), labs["left"]) + inc_loss(self.heads.forward_inc(m_mid, s_r), labs["right"]),
            "update": F.mse_loss(self.update(m_a, s_l), m_mid.detach().float()) + F.mse_loss(self.update(m_a, s_f), m_end.detach().float()) + F.mse_loss(self.update(m_mid, s_r), m_end.detach().float()),
        }
        visual, proprio = self.anchor(s_f, length)
        frames = self.anchor.frames(length, s_f.device)
        terms["recon"] = F.mse_loss(visual.float(), F.layer_norm(batch["fut_visual"][:, frames].float(), (384,)))
        terms["recon_proprio"] = F.mse_loss(proprio.float(), batch["fut_proprio"].float())
        terms["variance"] = variance_penalty(s_f.mean(1))
        total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}

    @torch.no_grad()
    def targets(self, batch, half):
        length = batch["actions"].shape[1]
        m_t, z_t = self.context(batch)
        fr = self.fuser(batch["fut_visual"], batch["fut_proprio"])
        return {
            "m_t": m_t, "z_t": z_t, "frames": fr,
            "s_f": self.encoder(fr), "s_l": self.encoder(fr[:, :half]), "s_r": self.encoder(fr[:, half:]),
            "m_mid": self.continue_memory(m_t, fr, batch["actions"], half), "m_end": self.continue_memory(m_t, fr, batch["actions"], length),
        }


# ---------------------------------------------------------------------------
# Students (teacher frozen and not registered as a submodule)


class Student(nn.Module):
    def __init__(self, kind: str, teacher: GroundedTeacher, dim: int, depth: int) -> None:
        super().__init__()
        self.kind = kind
        object.__setattr__(self, "teacher", teacher)
        if kind == "frame_teacher":
            self.frame_predictor = FramePredictor(dim, depth)
        else:
            self.predictor = SegmentPredictor(dim, depth)
        self.composer = Composer(dim) if kind == "segment_comp" else None

    def predict(self, m, z, actions):
        """-> summary (4 tokens), endpoint latent, predicted frames or None."""
        if self.kind == "frame_teacher":
            frames = self.frame_predictor(m, z, actions)
            return self.teacher.encoder(frames), frames[:, -1], frames
        out = self.predictor(m, z, actions)
        return out[:, :SUMMARY_TOKENS], out[:, SUMMARY_TOKENS], None

    def advance(self, m, s, frames, actions):
        if frames is not None:
            # frozen teacher GRU is in eval mode; cuDNN RNN cannot backprop there, so use the native kernel
            with torch.backends.cudnn.flags(enabled=False):
                return self.teacher.continue_memory(m, frames, actions, frames.shape[1])
        return self.teacher.update(m, s)

    def loss(self, batch, labs, weights):
        length = batch["actions"].shape[1]
        half = length // 2
        tg = self.teacher.targets(batch, half)
        heads = self.teacher.heads
        act = batch["actions"]
        s_f, e_f, fr_f = self.predict(tg["m_t"], tg["z_t"], act)
        s_l, e_l, fr_l = self.predict(tg["m_t"], tg["z_t"], act[:, :half])
        s_rt, e_rt, fr_rt = self.predict(tg["m_mid"], tg["frames"][:, half - 1], act[:, half:])
        m_roll = self.advance(tg["m_t"], s_l, fr_l, act[:, :half])
        s_rr, e_rr, fr_rr = self.predict(m_roll, e_l, act[:, half:])

        def pair(s, e, st, et):
            return F.smooth_l1_loss(s.float(), st.float()) + F.smooth_l1_loss(e.float(), et.float())

        fr = tg["frames"]
        terms = {
            "pred": pair(s_f, e_f, tg["s_f"], fr[:, -1]) + pair(s_l, e_l, tg["s_l"], fr[:, half - 1])
            + pair(s_rt, e_rt, tg["s_r"], fr[:, -1]) + pair(s_rr, e_rr, tg["s_r"], fr[:, -1]),
            "heads": empty_loss(heads.forward_empty(s_f), labs["full"]) + inc_loss(heads.forward_inc(tg["m_t"], s_f), labs["full"])
            + inc_loss(heads.forward_inc(tg["m_t"], s_l), labs["left"]) + inc_loss(heads.forward_inc(m_roll, s_rr), labs["right"]),
            "variance": variance_penalty(s_f.mean(1)),
        }
        if fr_f is not None:
            terms["frames"] = (F.smooth_l1_loss(fr_f.float(), fr.float()) + F.smooth_l1_loss(fr_l.float(), fr[:, :half].float())
                               + F.smooth_l1_loss(fr_rt.float(), fr[:, half:].float()) + F.smooth_l1_loss(fr_rr.float(), fr[:, half:].float()))
        if self.composer is not None:
            c_obs = self.composer(tg["s_l"], tg["s_r"], half, length - half)
            c_hat = self.composer(s_l, s_rr, half, length - half)
            terms["comp_observed"] = F.smooth_l1_loss(c_obs.float(), tg["s_f"].float())
            terms["comp_predicted"] = F.smooth_l1_loss(c_hat.float(), tg["s_f"].float())
            terms["comp_heads"] = empty_loss(heads.forward_empty(c_hat), labs["full"]) + inc_loss(heads.forward_inc(tg["m_t"], c_hat), labs["full"])
            terms["comp_memory"] = F.mse_loss(self.teacher.update(tg["m_t"], c_hat), tg["m_end"].float())
        total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}

    @torch.no_grad()
    def evaluate_paths(self, batch, parts) -> dict:
        """All inference paths for one split; uses only the observed prefix and actions."""
        heads = self.teacher.heads
        m_t, z_t = self.teacher.context(batch)
        m, z, start = m_t, z_t, 0
        summaries, frames, seq = [], [], []
        for length in parts:
            actions = batch["actions"][:, start : start + length]
            s, e, fr = self.predict(m, z, actions)
            seq.append(heads.forward_inc(m, s))
            summaries.append(s)
            if fr is not None:
                frames.append(fr)
            m, z, start = self.advance(m, s, fr, actions), e, start + length
        out = {"sequential": combine(seq)}
        total = sum(parts)
        s_direct, _, _ = self.predict(m_t, z_t, batch["actions"][:, :total])
        out["direct_whole"] = {**heads.forward_inc(m_t, s_direct), **heads.forward_empty(s_direct)}
        if self.composer is not None:
            c, run = summaries[0], parts[0]
            for s, length in zip(summaries[1:], parts[1:]):
                c = self.composer(c, s, run, length)
                run += length
            out["composed"] = {**heads.forward_inc(m_t, c), **heads.forward_empty(c)}
        if frames:
            s_cat = self.teacher.encoder(torch.cat(frames, 1))
            out["composed"] = {**heads.forward_inc(m_t, s_cat), **heads.forward_empty(s_cat)}
        return out


def combine(parts: list[dict]) -> dict:
    return {
        "count_increment": sum(p["count_increment"] for p in parts),
        "area_increment": sum(p["area_increment"] for p in parts),
        "map_new_logits": torch.logit(torch.stack([torch.sigmoid(p["map_new_logits"]) for p in parts]).amax(0).clamp(1e-6, 1 - 1e-6)),
    }


# ---------------------------------------------------------------------------
# Training loop


def labels_for(labels, eps, ts, length):
    half = length // 2
    return {"full": labels.get(eps, [(t, t + length) for t in ts]), "left": labels.get(eps, [(t, t + half) for t in ts]),
            "right": labels.get(eps, [(t + half, t + length) for t in ts])}


def fit(model, store, labels, tcfg, name, seed, log) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps = tcfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tcfg["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    pools = {L: train_starts(store, "train", L, STRIDE) for L in TRAIN_LENGTHS}
    val_pools = {L: train_starts(store, "val_select", L, 16) for L in TRAIN_LENGTHS}
    vrng = np.random.default_rng(777)
    val_batches = [(L, [val_pools[L][i] for i in vrng.choice(len(val_pools[L]), size=min(tcfg["batch_size"], len(val_pools[L])), replace=False)]) for L in TRAIN_LENGTHS for _ in range(tcfg["val_batches"])]
    history, start, status = {"train": [], "val": []}, time.time(), "complete"
    for step in range(steps):
        if time.time() - start > tcfg["time_budget_seconds"]:
            status = f"incomplete_time_budget_at_step_{step}"
            break
        model.train()
        length = TRAIN_LENGTHS[step % len(TRAIN_LENGTHS)]
        pick = rng.integers(len(pools[length]), size=tcfg["batch_size"])
        eps, ts = [pools[length][i][0] for i in pick], [pools[length][i][1] for i in pick]
        batch = store.batch(eps, ts, length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, terms = model.loss(batch, labels_for(labels, eps, ts, length), tcfg["loss_weights"])
        if not torch.isfinite(loss):
            status = f"non_finite_at_step_{step}"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            history["train"].append({"step": step, "loss": float(loss.detach()), **terms})
            print(f"[{name}] step {step} loss {float(loss.detach()):.4f} {time.time() - start:.0f}s", flush=True)
        if (step + 1) % tcfg["val_every"] == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vals = []
                for L, picks in val_batches:
                    e2, t2 = [p[0] for p in picks], [p[1] for p in picks]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vals.append(model.loss(store.batch(e2, t2, L), labels_for(labels, e2, t2, L), tcfg["loss_weights"])[1])
            history["val"].append({"step": step + 1, **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}})
    history.update({"status": status, "seconds": time.time() - start, "parameters": sum(p.numel() for p in params)})
    return history


# ---------------------------------------------------------------------------
# Evaluation


def metric_rows(pred: dict, lab: dict) -> dict:
    out = {"has_contact": lab["has_contact"].cpu().numpy() > 0}
    for k in ("count_increment", "area_increment"):
        out[f"true_{k}"] = lab[k].cpu().numpy()
        out[f"pred_{k}"] = pred[k].float().cpu().numpy()
    probs = torch.sigmoid(pred["map_new_logits"].float()).cpu().numpy()
    truth = lab["map_new"].cpu().numpy()
    out["map_new_ap"] = average_precision(probs[out["has_contact"]].reshape(-1), truth[out["has_contact"]].reshape(-1)) if out["has_contact"].any() else None
    return out


def as_bootstrap_rows(m: dict, metric: str) -> dict:
    return {"true": {metric: m[f"true_{metric}"]}, "pred": {metric: m[f"pred_{metric}"]}}


def summarize(m: dict, keep: np.ndarray) -> dict:
    out = {"windows": int(keep.sum()), "map_new_micro_ap": m["map_new_ap"]}
    for k in ("count_increment", "area_increment"):
        t, p = m[f"true_{k}"][keep], m[f"pred_{k}"][keep]
        out[f"mae_{k}"] = float(np.abs(t - p).mean())
        out[f"r_{k}"] = float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("grounded prototype must run inside an sbatch GPU job")
    cfg = json.loads(args.config.read_text())
    gcfg = cfg["grounded"]
    if args.profile:
        for key in ("teacher", "student"):
            gcfg[key].update(gcfg["profile"][key])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.run_dir / "grounded_result.json"
    result = {"verdict": "RUNNING", "job_id": os.environ["SLURM_JOB_ID"], "node": platform.node(), "profile": args.profile,
              "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": "docs/COMP_GROUNDED_PROTOCOL.md", "grounded_config": gcfg}
    write_json(result_path, result)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = cfg["training"]["seed"]
    dim, depth = cfg["training"]["dim"], cfg["training"]["predictor_depth"]
    try:
        t0 = time.time()
        d = cfg["data"]
        store = FeatureStore(Path(d["feature_root"]), Path(d["label_root"]), Path(d["label_audit"]), device, gcfg["profile"]["max_episodes"] if args.profile else None)
        store.manifest = [m for m in store.manifest if m["split"] != "test"]
        store.episodes = [m["episode"] for m in store.manifest]
        select, decide = val_halves(store)
        for e in select:
            store.split[e] = "val_select"
        for e in decide:
            store.split[e] = "val_decide"
        labels = GroundedLabels(store, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), "val_decide")
        store.load()
        result.update({"setup_seconds": time.time() - t0, "episodes": {s: len(store.split_episodes(s)) for s in ("train", "val_select", "val_decide")}})
        write_json(result_path, result)

        # Stage 1: teacher
        teacher = GroundedTeacher(dim).to(device)
        result["teacher"] = fit(teacher, store, labels, gcfg["teacher"], "teacher", seed, None)
        torch.save(teacher.state_dict(), args.run_dir / "teacher.pt")
        teacher.eval().requires_grad_(False)
        write_json(result_path, result)

        # Stage 2: students
        students = {}
        result["students"] = {}
        for kind in ("segment_nocomp", "segment_comp", "frame_teacher"):
            student = Student(kind, teacher, dim, depth).to(device)
            result["students"][kind] = fit(student, store, labels, gcfg["student"], kind, seed + 1, None)
            torch.save(student.state_dict(), args.run_dir / f"student_{kind}.pt")
            student.eval()
            students[kind] = student
            write_json(result_path, result)

        # Evaluation on val_decide
        pool = [(e, t) for e in store.split_episodes("val_decide") for t in eval_starts(store.lengths[e])]
        if args.profile:
            pool = pool[: gcfg["profile"]["max_windows"]]
        episodes = np.array([p[0] for p in pool])
        collected: dict[str, list] = {}
        for i in range(0, len(pool), 128):
            picks = pool[i : i + 128]
            eps, ts = [p[0] for p in picks], [p[1] for p in picks]
            batch = store.batch(eps, ts, 128)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for name, parts in EVAL_PARTS.items():
                    total = sum(parts)
                    sub = {k: (v[:, :total] if k in ("fut_visual", "fut_proprio", "actions") else v) for k, v in batch.items()}
                    lab = labels.get(eps, [(t, t + total) for t in ts])
                    m_t, _ = teacher.context(sub)
                    s_true = teacher.encoder(teacher.fuser(sub["fut_visual"], sub["fut_proprio"]))
                    collected.setdefault(f"teacher_true_future|{name}", []).append((teacher.heads.forward_inc(m_t, s_true), lab))
                    for kind, student in students.items():
                        for path, pred in student.evaluate_paths(sub, parts).items():
                            collected.setdefault(f"{kind}:{path}|{name}", []).append((pred, lab))
        metrics, table = {}, {}
        for key, chunks in collected.items():
            pred = {k: torch.cat([c[0][k] for c in chunks]) for k in ("count_increment", "area_increment", "map_new_logits")}
            lab = {k: torch.cat([c[1][k] for c in chunks]) for k in chunks[0][1]}
            metrics[key] = metric_rows(pred, lab)
            table[key] = summarize(metrics[key], metrics[key]["has_contact"])
        result["table"] = table

        # locked decision
        rng = np.random.default_rng(seed)
        decision, paired = {}, {}
        for name in EVAL_PARTS:
            comp_key = f"segment_comp:composed|{name}"
            controls = [f"segment_nocomp:sequential|{name}", f"segment_nocomp:direct_whole|{name}", f"frame_teacher:composed|{name}", f"frame_teacher:sequential|{name}"]
            keep = metrics[comp_key]["has_contact"]
            entry = {"composed_mae": table[comp_key]["mae_count_increment"], "controls": {}}
            for ctrl in controls + [f"segment_comp:sequential|{name}"]:
                b = episode_bootstrap_r(as_bootstrap_rows(metrics[comp_key], "count_increment"), as_bootstrap_rows(metrics[ctrl], "count_increment"),
                                        episodes, keep, "count_increment", rng, 200 if args.profile else 1000)
                entry["controls"][ctrl] = {"mae": table[ctrl]["mae_count_increment"], **b}
                paired[f"{comp_key} vs {ctrl}"] = b
            best = min(controls, key=lambda c: table[c]["mae_count_increment"])
            entry["best_control"] = best
            entry["relative_mae_reduction_vs_best"] = (table[best]["mae_count_increment"] - entry["composed_mae"]) / table[best]["mae_count_increment"]
            entry["mae_ci_excludes_zero_vs_best"] = entry["controls"][best]["mae_ci95"][1] < 0
            decision[name] = entry
        passes = all(decision[n]["relative_mae_reduction_vs_best"] >= 0.10 for n in PRIMARY) and any(decision[n]["mae_ci_excludes_zero_vs_best"] for n in PRIMARY)
        result["decision"] = {"per_split": decision, "rule": "COMPOSITION_CLEARLY_BETTER_CONFIRM_3_SEEDS" if passes else "COMPOSITION_NOT_BETTER_THAN_MATCHED_CONTROLS"}
        result["verdict"] = "PROFILE_COMPLETE" if args.profile else "GROUNDED_PROTOTYPE_COMPLETE"
        result["total_seconds"] = time.time() - t0
        write_json(result_path, result)
        print(f"DECISION {result['decision']['rule']}", flush=True)
        print(f"WROTE {result_path}", flush=True)
    except Exception as error:
        result.update({"verdict": "ERROR", "error": repr(error), "traceback": traceback.format_exc()})
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
