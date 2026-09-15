"""Segment vs frame prediction from the 52655 checkpoints (docs/SEGMENT_VS_FRAME_EVAL_PROTOCOL.md).

Evaluation only, sbatch GPU. Test episodes are dropped before loading.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import EVAL_PARTS, FeatureStore, eval_starts
from .grounded import GroundedLabels, GroundedTeacher, Student
from .reeval import episode_bootstrap_r
from .reeval_target import val_halves

PRIMARY = ("64=24+40", "128=64+64")
METRICS = ("count_increment", "area_increment")


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


def sub_batch(batch, total):
    return {k: (v[:, :total] if k in ("fut_visual", "fut_proprio", "actions") else v) for k, v in batch.items()}


@torch.no_grad()
def sequential(student: Student, m, z, actions, parts):
    """Inference-only sequential path (cuDNN enabled for the teacher GRU)."""
    heads, teacher = student.teacher.heads, student.teacher
    start, totals = 0, {k: 0.0 for k in METRICS}
    for length in parts:
        act = actions[:, start : start + length]
        s, e, frames = student.predict(m, z, act)
        out = heads.forward_inc(m, s)
        for k in METRICS:
            totals[k] = totals[k] + out[k].float()
        m = teacher.continue_memory(m, frames, act, frames.shape[1]) if frames is not None else teacher.update(m, s)
        z, start = e, start + length
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("evaluation must run inside an sbatch GPU job")
    cfg = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "segment_vs_frame_result.json"
    result = {"verdict": "RUNNING", "job_id": os.environ["SLURM_JOB_ID"], "node": platform.node(), "profile": args.profile,
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(args.source_run),
              "protocol": "docs/SEGMENT_VS_FRAME_EVAL_PROTOCOL.md"}
    write_json(result_path, result)
    device = torch.device("cuda")
    dim, depth, seed = cfg["training"]["dim"], cfg["training"]["predictor_depth"], cfg["training"]["seed"]
    try:
        d = cfg["data"]
        store = FeatureStore(Path(d["feature_root"]), Path(d["label_root"]), Path(d["label_audit"]), device, 18 if args.profile else None)
        store.manifest = [m for m in store.manifest if m["split"] != "test"]
        store.episodes = [m["episode"] for m in store.manifest]
        select, decide = val_halves(store)
        for e in select:
            store.split[e] = "val_select"
        for e in decide:
            store.split[e] = "val_decide"
        # train windows are needed only for constants; label them like val_decide eval windows
        train_eps = store.split_episodes("train")
        for e in train_eps:
            store.split[e] = "train_const"
        labels = GroundedLabels(store, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), "val_decide")
        const_labels = GroundedLabels(store, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), "train_const")
        for e in train_eps:  # restore: feature normalization statistics come from the train split
            store.split[e] = "train"
        store.load()

        teacher = GroundedTeacher(dim).to(device)
        teacher.load_state_dict(torch.load(args.source_run / "teacher.pt", map_location=device, weights_only=True))
        teacher.eval().requires_grad_(False)
        students = {}
        for kind in ("segment_nocomp", "frame_teacher"):
            st = Student(kind, teacher, dim, depth).to(device)
            st.load_state_dict(torch.load(args.source_run / f"student_{kind}.pt", map_location=device, weights_only=True))
            students[kind] = st.eval().requires_grad_(False)
        result["parameters"] = {k: sum(p.numel() for p in s.parameters()) for k, s in students.items()}
        result["teacher_parameters"] = sum(p.numel() for p in teacher.parameters())

        pool = [(e, t) for e in store.split_episodes("val_decide") for t in eval_starts(store.lengths[e])]
        const_pool = [(e, t) for e in train_eps for t in eval_starts(store.lengths[e])]
        if args.profile:
            pool, const_pool = pool[:128], const_pool[:128]
        episodes = np.array([p[0] for p in pool])

        # 1. accuracy + 3. teacher decomposition
        store_rows: dict[str, dict] = {}

        def put(key, pred, lab):
            row = store_rows.setdefault(key, {"pred": {k: [] for k in METRICS}, "true": {k: [] for k in METRICS}, "has": []})
            for k in METRICS:
                row["pred"][k].append(pred[k].float().cpu().numpy())
                row["true"][k].append(lab[k].cpu().numpy())
            row["has"].append(lab["has_contact"].cpu().numpy() > 0)

        for i in range(0, len(pool), 128):
            picks = pool[i : i + 128]
            eps, ts = [p[0] for p in picks], [p[1] for p in picks]
            batch = store.batch(eps, ts, 128)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                m_t, z_t = teacher.context(batch)
                frames_true = teacher.fuser(batch["fut_visual"], batch["fut_proprio"])
                for name, parts in EVAL_PARTS.items():
                    total = sum(parts)
                    lab = labels.get(eps, [(t, t + total) for t in ts])
                    for kind, st in students.items():
                        put(f"{kind}:sequential|{name}", sequential(st, m_t, z_t, batch["actions"][:, :total], parts), lab)
                for total in (64, 128):
                    lab = labels.get(eps, [(t, t + total) for t in ts])
                    fr = frames_true[:, :total]
                    half = total // 2
                    whole = teacher.heads.forward_inc(m_t, teacher.encoder(fr))
                    s1, s2 = teacher.encoder(fr[:, :half]), teacher.encoder(fr[:, half:])
                    first = teacher.heads.forward_inc(m_t, s1)
                    m_obs = teacher.continue_memory(m_t, fr, batch["actions"], half)
                    m_upd = teacher.update(m_t, s1)
                    put(f"teacher_whole|{total}", whole, lab)
                    put(f"teacher_halves_observed_memory|{total}", {k: first[k] + teacher.heads.forward_inc(m_obs, s2)[k] for k in METRICS}, lab)
                    put(f"teacher_halves_update_U|{total}", {k: first[k] + teacher.heads.forward_inc(m_upd, s2)[k] for k in METRICS}, lab)
        rows = {k: {"pred": {m: np.concatenate(v["pred"][m]) for m in METRICS}, "true": {m: np.concatenate(v["true"][m]) for m in METRICS}, "has": np.concatenate(v["has"])} for k, v in store_rows.items()}

        constants = {}
        for total in (64, 128):
            lab = const_labels.get([p[0] for p in const_pool], [(p[1], p[1] + total) for p in const_pool])
            has = lab["has_contact"] > 0
            constants[total] = {"all": {k: float(lab[k].mean()) for k in METRICS}, "contact_fit": {k: float(lab[k][has].mean()) for k in METRICS}}
        result["train_constants"] = {str(k): v for k, v in constants.items()}

        table = {}
        for key, r in rows.items():
            total = int(key.split("|")[1].split("=")[0])
            for stratum, keep in (("all", np.ones_like(r["has"])), ("contact", r["has"])):
                keep = keep.astype(bool)
                entry = {"windows": int(keep.sum())}
                for k in METRICS:
                    t, p = r["true"][k][keep], r["pred"][k][keep]
                    c = constants[total]["all" if stratum == "all" else "contact_fit"][k]
                    entry[f"mae_{k}"] = float(np.abs(t - p).mean())
                    entry[f"r_{k}"] = float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else None
                    entry[f"const_train_mae_{k}"] = float(np.abs(t - c).mean())
                table[f"{key}|{stratum}"] = entry
        result["table"] = table

        rng = np.random.default_rng(seed)
        paired, decision = {}, {}
        for name in EVAL_PARTS:
            a, b = rows[f"segment_nocomp:sequential|{name}"], rows[f"frame_teacher:sequential|{name}"]
            for stratum, keep in (("all", np.ones_like(a["has"])), ("contact", a["has"])):
                for k in METRICS:
                    paired[f"{name}|{stratum}|{k}"] = episode_bootstrap_r(a, b, episodes, keep.astype(bool), k, rng, 200 if args.profile else 1000)
            main_ = paired[f"{name}|contact|count_increment"]
            frame_mae = table[f"frame_teacher:sequential|{name}|contact"]["mae_count_increment"]
            decision[name] = {"relative_mae_reduction": -main_["mae_difference"] / frame_mae, "mae_ci95": main_["mae_ci95"], "r_difference": main_["r_difference"], "r_ci95": main_["r_ci95"]}
        result["paired_episode_bootstrap"] = paired
        write_json(result_path, result)

        # 2. inference cost
        timing = {}
        warm, reps = (1, 3) if args.profile else (5, 20)
        for batch_size in (32, 256):
            picks = [pool[i % len(pool)] for i in range(batch_size)]
            batch = store.batch([p[0] for p in picks], [p[1] for p in picks], 128)
            for horizon, parts in ((64, [32, 32]), (128, [64, 64])):
                for kind, st in students.items():
                    def run(include_context: bool, ctx=None):
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            m, z = teacher.context(batch) if include_context else ctx
                            return sequential(st, m, z, batch["actions"][:, :horizon], parts)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        ctx = teacher.context(batch)
                    for include in (False, True):
                        for _ in range(warm):
                            run(include, ctx)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        times = []
                        for _ in range(reps):
                            t0 = time.perf_counter()
                            run(include, ctx)
                            torch.cuda.synchronize()
                            times.append(time.perf_counter() - t0)
                        timing[f"{kind}|batch{batch_size}|h{horizon}|{'with' if include else 'without'}_context"] = {
                            "median_ms": 1000 * float(np.median(times)), "p90_ms": 1000 * float(np.quantile(times, 0.9)),
                            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
        result["timing"] = timing
        seg = timing["segment_nocomp|batch256|h128|without_context"]["median_ms"]
        fra = timing["frame_teacher|batch256|h128|without_context"]["median_ms"]
        speedup = fra / seg

        accuracy_ok = all(decision[n]["relative_mae_reduction"] >= 0.05 and decision[n]["mae_ci95"][1] < 0 for n in PRIMARY)
        cost_ok = speedup >= 2.0 and all(decision[n]["mae_ci95"][1] <= 0 for n in PRIMARY)
        result["decision"] = {"per_split": decision, "speedup_frame_over_segment_b256_h128": speedup, "accuracy_rule": accuracy_ok, "cost_rule": cost_ok,
                              "rule": "KEEP_SEGMENT_PREDICTION_FOR_NEXT_STEP" if (accuracy_ok or cost_ok) else "CLOSE_SCRUB_BRANCH_AS_METHOD_DIRECTION"}
        result["verdict"] = "PROFILE_COMPLETE" if args.profile else "SEGMENT_VS_FRAME_COMPLETE"
        write_json(result_path, result)
        print(f"DECISION {result['decision']['rule']} speedup={speedup:.2f}", flush=True)
        print(f"WROTE {result_path}", flush=True)
    except Exception as error:
        result.update({"verdict": "ERROR", "error": repr(error), "traceback": traceback.format_exc()})
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
