"""Train, read out and evaluate the offline composition pilot arms (sbatch only).

Usage (inside a Slurm GPU job):
  python -m comp_pilot.train --config configs/comp_pilot.json --run-dir RUN --arms segment_nocomp,segment_comp,frame_rollout,map_union
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
import torch.nn.functional as F

from .data import DIRECT_LENGTHS, EVAL_PARTS, TRAIN_LENGTHS, FeatureStore, LabelTable, eval_starts, train_starts
from .models import ARMS, STRIDE, Readout

SCALARS = ("count_increment", "count_empty", "area_empty")


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


def with_labels(batch: dict, labels: LabelTable, eps: list[int], starts: list[int], length: int) -> dict:
    m = length // 2
    batch["labels_full"] = labels.get(eps, [(t, t + length) for t in starts])
    batch["labels_left"] = labels.get(eps, [(t, t + m) for t in starts])
    batch["labels_right"] = labels.get(eps, [(t + m, t + length) for t in starts])
    return batch


def truncate(batch: dict, length: int) -> dict:
    out = dict(batch)
    for key in ("fut_visual", "fut_proprio", "actions"):
        out[key] = batch[key][:, :length]
    return out


# ---------------------------------------------------------------------------
# World-model training


def train_arm(name: str, store: FeatureStore, labels: LabelTable, cfg: dict, arm_dir: Path, seed: int) -> tuple[torch.nn.Module, dict]:
    tcfg = cfg["training"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ARMS[name](tcfg["dim"], tcfg["predictor_depth"]).to(store.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps, warmup = tcfg["steps"], tcfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    pools = {L: train_starts(store, "train", L, STRIDE) for L in TRAIN_LENGTHS}
    val_pools = {L: train_starts(store, "val", L, 16) for L in TRAIN_LENGTHS}
    val_rng = np.random.default_rng(12345)
    val_batches = [
        (L, [val_pools[L][i] for i in val_rng.choice(len(val_pools[L]), size=tcfg["batch_size"], replace=False)])
        for L in TRAIN_LENGTHS
        for _ in range(tcfg["val_batches"])
    ]
    history = {"train": [], "val": []}
    start = time.time()
    status = "complete"
    model.train()
    for step in range(steps):
        if time.time() - start > tcfg["time_budget_seconds"]:
            status = f"incomplete_time_budget_at_step_{step}"
            break
        length = TRAIN_LENGTHS[step % len(TRAIN_LENGTHS)]
        pick = rng.integers(len(pools[length]), size=tcfg["batch_size"])
        eps = [pools[length][i][0] for i in pick]
        ts = [pools[length][i][1] for i in pick]
        batch = store.batch(eps, ts, length)
        if model.uses_labels:
            batch = with_labels(batch, labels, eps, ts, length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, terms = model.loss(batch, tcfg["loss_weights"])
        if not torch.isfinite(loss):
            status = f"non_finite_loss_at_step_{step}"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step()
        sched.step()
        model.ema_update(tcfg["ema_decay"])
        if step % 100 == 0:
            row = {"step": step, "loss": float(loss.detach()), "seconds": time.time() - start, **terms}
            history["train"].append(row)
            print(f"[{name}] step {step} loss {row['loss']:.4f} {row['seconds']:.0f}s", flush=True)
        if (step + 1) % tcfg["val_every"] == 0 or step == steps - 1:
            history["val"].append({"step": step + 1, **validate(model, store, labels, val_batches, tcfg)})
            model.train()
    history["status"] = status
    history["train_seconds"] = time.time() - start
    history["parameters"] = sum(p.numel() for p in params)
    torch.save({"model": model.state_dict(), "config": cfg, "arm": name, "seed": seed}, arm_dir / "model.pt")
    write_json(arm_dir / "train_history.json", history)
    return model, history


@torch.no_grad()
def validate(model, store, labels, val_batches, tcfg) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    for length, picks in val_batches:
        eps, ts = [p[0] for p in picks], [p[1] for p in picks]
        batch = store.batch(eps, ts, length)
        if model.uses_labels:
            batch = with_labels(batch, labels, eps, ts, length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, terms = model.loss(batch, tcfg["loss_weights"])
        for k, v in {"loss": float(loss), **terms}.items():
            totals[k] = totals.get(k, 0.0) + v / len(val_batches)
    return totals


# ---------------------------------------------------------------------------
# Readout (same architecture and procedure for every representation)


@torch.no_grad()
def representations(model, store, pool, length, true_rep: bool, chunk: int = 256) -> torch.Tensor:
    model.eval()
    reps = []
    for i in range(0, len(pool), chunk):
        picks = pool[i : i + chunk]
        batch = store.batch([p[0] for p in picks], [p[1] for p in picks], length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rep = model.true_representation(batch) if true_rep else model.rollout(batch, [length], composed=False)["parts"][0]
        reps.append(rep.float())
    return torch.cat(reps)


def readout_targets(labels, pool, length) -> dict:
    return labels.get([p[0] for p in pool], [(p[1], p[1] + length) for p in pool])


def readout_loss(readout, tokens, target) -> torch.Tensor:
    scal, logits = readout(tokens)
    goal = torch.stack([target["count_increment"], target["count_empty"], target["area_empty"] / 10.0], 1)
    return F.smooth_l1_loss(scal, goal) + F.binary_cross_entropy_with_logits(logits, target["map"])


def train_readout(model, store, labels, cfg: dict, true_rep: bool, seed: int) -> tuple[Readout, dict]:
    rcfg = cfg["readout"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = {}
    for split, stride in (("train", rcfg["train_stride"]), ("val", 16)):
        for length in TRAIN_LENGTHS:
            pool = train_starts(store, split, length, stride)
            data[(split, length)] = (representations(model, store, pool, length, true_rep), readout_targets(labels, pool, length))
    token_dim = data[("train", TRAIN_LENGTHS[0])][0].shape[-1]
    readout = Readout(token_dim).to(store.device)
    opt = torch.optim.AdamW(readout.parameters(), lr=rcfg["lr"], weight_decay=rcfg["weight_decay"])
    best, best_state, curve = float("inf"), None, []
    for step in range(rcfg["steps"]):
        readout.train()
        length = TRAIN_LENGTHS[step % len(TRAIN_LENGTHS)]
        tokens, target = data[("train", length)]
        idx = torch.from_numpy(rng.integers(len(tokens), size=rcfg["batch_size"])).to(store.device)
        loss = readout_loss(readout, tokens[idx], {k: v[idx] for k, v in target.items()})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % rcfg["val_every"] == 0:
            readout.eval()
            with torch.no_grad():
                val = sum(float(readout_loss(readout, *data[("val", L)])) for L in TRAIN_LENGTHS) / len(TRAIN_LENGTHS)
            curve.append({"step": step + 1, "val_loss": val})
            if val < best:
                best, best_state = val, {k: v.detach().clone() for k, v in readout.state_dict().items()}
    readout.load_state_dict(best_state)
    readout.eval()
    return readout, {"best_val_loss": best, "curve": curve, "token_dim": token_dim}


# ---------------------------------------------------------------------------
# Evaluation


def predict(readout, tokens) -> dict:
    scal, logits = readout(tokens.float())
    return {"count_increment": scal[:, 0], "count_empty": scal[:, 1], "area_empty": scal[:, 2] * 10.0, "map": torch.sigmoid(logits)}


def combine_additive(preds: list[dict]) -> dict:
    out = {k: sum(p[k] for p in preds) for k in SCALARS}
    out["map"] = torch.stack([p["map"] for p in preds]).amax(0)
    return out


@torch.no_grad()
def evaluate(name, model, readout, ceiling_readout, store, labels, split: str, max_windows: int | None) -> list[dict]:
    model.eval()
    pool = [(e, t) for e in store.split_episodes(split) for t in eval_starts(store.lengths[e])]
    if max_windows:
        pool = pool[:max_windows]
    rows = []

    def record(eps, ts, total, eval_name, method, pred, hard_name=None):
        target = labels.get(eps, [(t, t + total) for t in ts])
        pm = (pred["map"] > 0.5).float()
        tp = (pm * target["map"]).sum(1)
        f1 = (2 * tp / (pm.sum(1) + target["map"].sum(1)).clamp_min(1e-6)).cpu().numpy()
        for i, (e, t) in enumerate(zip(eps, ts)):
            row = {
                "arm": name,
                "eval": eval_name,
                "method": method,
                "episode": e,
                "t": t,
                "label_agree": store.label_agree.get(e, False),
                "has_contact": bool(target["has_contact"][i] > 0),
                "hard": None if hard_name is None else bool(labels.hard[(e, t, hard_name)]),
                "map_f1": float(f1[i]) if float(target["map"][i].sum()) > 0 else None,
            }
            for k in SCALARS:
                row[f"pred_{k}"] = float(pred[k][i])
                row[f"true_{k}"] = float(target[k][i])
            rows.append(row)

    for i in range(0, len(pool), 128):
        picks = pool[i : i + 128]
        eps, ts = [p[0] for p in picks], [p[1] for p in picks]
        batch = store.batch(eps, ts, 128)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for length in DIRECT_LENGTHS:
                sub = truncate(batch, length)
                rep = model.rollout(sub, [length], composed=False)["parts"][0]
                record(eps, ts, length, f"direct{length}", "direct", predict(readout, rep))
                if ceiling_readout is not None:
                    record(eps, ts, length, f"direct{length}", "true_summary_ceiling", predict(ceiling_readout, model.true_representation(sub)))
            for eval_name, parts in EVAL_PARTS.items():
                total = sum(parts)
                out = model.rollout(truncate(batch, total), parts, composed=True)
                record(eps, ts, total, eval_name, "additive", combine_additive([predict(readout, r) for r in out["parts"]]), eval_name)
                if "composed" in out:
                    record(eps, ts, total, eval_name, "composed", predict(readout, out["composed"]), eval_name)
                record(eps, ts, total, eval_name, "direct_whole", predict(readout, model.rollout(truncate(batch, total), [total], composed=False)["parts"][0]), eval_name)
    return rows


# ---------------------------------------------------------------------------
# Summary with episode bootstrap


def summarize(rows: list[dict], seed: int = 0, resamples: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        strata = ["all", "contact" if r["has_contact"] else "no_contact"]
        if r["hard"] is not None:
            strata.append("hard" if r["hard"] else ("easy_contact" if r["has_contact"] else "easy_no_contact"))
        for stratum in strata:
            for agree in ("all_labels", "label_agree" if r["label_agree"] else "label_disagree"):
                groups.setdefault((r["arm"], r["eval"], r["method"], stratum, agree), []).append(r)
    table = {}
    for key, rs in groups.items():
        entry = {"windows": len(rs), "episodes": len({r["episode"] for r in rs})}
        for k in SCALARS:
            entry[f"mae_{k}"] = float(np.mean([abs(r[f"pred_{k}"] - r[f"true_{k}"]) for r in rs]))
        f1 = [r["map_f1"] for r in rs if r["map_f1"] is not None]
        entry["map_f1"] = float(np.mean(f1)) if f1 else None
        table["|".join(key)] = entry

    def episode_errors(arm, eval_name, method, stratum, metric):
        per = {}
        for r in groups.get((arm, eval_name, method, stratum, "all_labels"), []):
            per.setdefault(r["episode"], []).append(abs(r[f"pred_{metric}"] - r[f"true_{metric}"]))
        return {e: float(np.mean(v)) for e, v in per.items()}

    comparisons = [
        (("segment_comp", "composed"), ("segment_nocomp", "additive")),
        (("segment_comp", "composed"), ("segment_nocomp", "direct_whole")),
        (("segment_comp", "composed"), ("segment_comp", "additive")),
        (("segment_comp", "composed"), ("frame_rollout", "composed")),
        (("segment_comp", "composed"), ("map_union", "composed")),
        (("segment_comp", "additive"), ("segment_nocomp", "additive")),
    ]
    paired = {}
    for eval_name in EVAL_PARTS:
        for stratum in ("hard", "contact", "all"):
            for metric in ("count_empty", "count_increment", "area_empty"):
                for (arm_a, meth_a), (arm_b, meth_b) in comparisons:
                    a = episode_errors(arm_a, eval_name, meth_a, stratum, metric)
                    b = episode_errors(arm_b, eval_name, meth_b, stratum, metric)
                    common = sorted(set(a) & set(b))
                    if len(common) < 5:
                        continue
                    diff = np.array([a[e] - b[e] for e in common])
                    boots = np.array([diff[rng.integers(len(diff), size=len(diff))].mean() for _ in range(resamples)])
                    paired[f"{eval_name}|{stratum}|{metric}|{arm_a}:{meth_a} - {arm_b}:{meth_b}"] = {
                        "episodes": len(common),
                        "mean_mae_difference": float(diff.mean()),
                        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                    }
    return {"table": table, "paired_episode_bootstrap": paired}


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", type=str, required=True)
    parser.add_argument("--profile", action="store_true", help="short run to measure step time and catch bugs")
    parser.add_argument("--train-only", action="store_true", help="save checkpoints; no readout or test evaluation")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("pilot training must run inside an sbatch job")
    if not torch.cuda.is_available():
        raise RuntimeError("pilot training requires a CUDA node")
    cfg = json.loads(args.config.read_text())
    if args.profile:
        cfg["training"].update(cfg["profile"]["training"])
        cfg["readout"].update(cfg["profile"]["readout"])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.run_dir / "pilot_result.json"
    result = {
        "verdict": "RUNNING",
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": platform.node(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "config": cfg,
        "arms": {},
    }
    write_json(result_path, result)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    data_cfg = cfg["data"]
    try:
        t0 = time.time()
        store = FeatureStore(Path(data_cfg["feature_root"]), Path(data_cfg["label_root"]), Path(data_cfg["label_audit"]), device,
                             cfg["profile"]["max_episodes"] if args.profile else None)
        labels = LabelTable(store, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
        store.load()
        result["setup_seconds"] = time.time() - t0
        result["episodes"] = {s: len(store.split_episodes(s)) for s in ("train", "val", "test")}
        write_json(result_path, result)
        all_rows = []
        seed = cfg["training"]["seed"]
        for name in args.arms.split(","):
            arm_dir = args.run_dir / name
            arm_dir.mkdir(exist_ok=True)
            arm_result = {"status": "running"}
            result["arms"][name] = arm_result
            write_json(result_path, result)
            model, history = train_arm(name, store, labels, cfg, arm_dir, seed)
            arm_result.update({"train_status": history["status"], "train_seconds": history["train_seconds"],
                               "parameters": history["parameters"], "final_val": history["val"][-1] if history["val"] else None})
            if args.train_only:
                arm_result["status"] = "trained"
                write_json(result_path, result)
                del model
                torch.cuda.empty_cache()
                continue
            readout, rinfo = train_readout(model, store, labels, cfg, true_rep=False, seed=seed + 1)
            arm_result["readout"] = {k: v for k, v in rinfo.items() if k != "curve"}
            ceiling = None
            if hasattr(model, "true_representation"):
                ceiling, cinfo = train_readout(model, store, labels, cfg, true_rep=True, seed=seed + 1)
                arm_result["ceiling_readout"] = {k: v for k, v in cinfo.items() if k != "curve"}
            rows = evaluate(name, model, readout, ceiling, store, labels, "test", cfg["profile"]["max_eval_windows"] if args.profile else None)
            (arm_dir / "test_rows.json").write_text(json.dumps(rows))
            all_rows.extend(rows)
            arm_result["status"] = "evaluated"
            arm_result["test_windows"] = len({(r["episode"], r["t"]) for r in rows})
            write_json(result_path, result)
            del model, readout, ceiling
            torch.cuda.empty_cache()
        if not args.train_only:
            result["summary"] = summarize(all_rows, seed=seed)
        result["verdict"] = "PROFILE_COMPLETE" if args.profile else "PILOT_COMPLETE"
        result["total_seconds"] = time.time() - t0
        write_json(result_path, result)
        print(f"WROTE {result_path}", flush=True)
    except Exception as error:
        result.update({"verdict": "ERROR", "error": repr(error), "traceback": traceback.format_exc()})
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
