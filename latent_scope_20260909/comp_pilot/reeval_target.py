"""Evaluate the target-corrected segment arms against the 52634 arms on val_decide (sbatch only).

Protocol: docs/COMP_PILOT_TARGET_FIX_PROTOCOL.md. Test episodes are dropped before loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import DIRECT_LENGTHS, EVAL_PARTS, TRAIN_LENGTHS, FeatureStore, eval_starts
from .models import ARMS
from .reeval import SCALARS, Labels, apply_readout, episode_bootstrap_r, extract, fit_readout, native_to_pred, summarize_metrics, window_metrics
from .train import truncate

PRIMARY_SPLITS = ("64=24+40", "128=64+64")
ALL_SPLITS = tuple(EVAL_PARTS)


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


def val_halves(store: FeatureStore) -> tuple[list[int], list[int]]:
    val = sorted(store.split_episodes("val"), key=lambda e: hashlib.sha256(f"20260916:val_half:{e}".encode()).hexdigest())
    half = len(val) // 2
    return sorted(val[:half]), sorted(val[half:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("evaluation must run inside an sbatch GPU job")
    cfg = json.loads(args.config.read_text())
    rcfg = cfg["reeval"]
    if args.profile:
        rcfg["readout"].update(rcfg["profile"]["readout"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "target_fix_result.json"
    result = {"verdict": "RUNNING", "job_id": os.environ["SLURM_JOB_ID"], "node": platform.node(), "profile": args.profile,
              "created_utc": datetime.now(timezone.utc).isoformat(), "new_run": str(args.new_run), "old_run": str(args.old_run),
              "protocol": "docs/COMP_PILOT_TARGET_FIX_PROTOCOL.md"}
    write_json(result_path, result)
    device = torch.device("cuda")
    seed = cfg["training"]["seed"]
    try:
        t0 = time.time()
        d = cfg["data"]
        store = FeatureStore(Path(d["feature_root"]), Path(d["label_root"]), Path(d["label_audit"]), device, rcfg["profile"]["max_episodes"] if args.profile else None)
        # drop test episodes entirely: they are not used for this decision
        store.manifest = [m for m in store.manifest if m["split"] != "test"]
        store.episodes = [m["episode"] for m in store.manifest]
        select, decide = val_halves(store)
        for e in select:
            store.split[e] = "val_select"
        for e in decide:
            store.split[e] = "val_decide"
        labels = Labels(store, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
        store.load()
        result["episodes"] = {s: len(store.split_episodes(s)) for s in ("train", "val_select", "val_decide")}
        result["val_select"], result["val_decide"] = select, decide
        pools = {s: [(e, t) for e in store.split_episodes(s) for t in eval_starts(store.lengths[e])] for s in ("train", "val_select", "val_decide")}
        if args.profile:
            pools = {s: p[: rcfg["profile"]["max_windows"]] for s, p in pools.items()}
        eps_decide = np.array([p[0] for p in pools["val_decide"]])

        def targets(split, total):
            pool = pools[split]
            return labels.get([p[0] for p in pool], [(p[1], p[1] + total) for p in pool])

        constants, priors = {}, {}
        for total in DIRECT_LENGTHS:
            tr = targets("train", total)
            has = tr["has_contact"] > 0
            constants[total] = {"all": {k: float(tr[k].mean()) for k in SCALARS}, "contact_fit": {k: float(tr[k][has].mean()) for k in SCALARS}}
            priors[total] = tr["map"].mean(0).cpu().numpy()
        decide_targets = {total: targets("val_decide", total) for total in DIRECT_LENGTHS}
        has_contact = {total: decide_targets[total]["has_contact"].cpu().numpy() > 0 for total in DIRECT_LENGTHS}
        hard = {name: np.array([labels.hard[(e, t, name)] for e, t in pools["val_decide"]]) for name in EVAL_PARTS}
        metrics_store, table = {}, {}

        def record(key, pred, total, eval_name):
            m = window_metrics(pred, decide_targets[total], priors[total])
            metrics_store[key] = m
            strata = {"all": np.ones_like(has_contact[total]), "contact": has_contact[total]}
            if eval_name is not None:
                strata["hard"] = hard[eval_name] & has_contact[total]
            for stratum, keep in strata.items():
                entry = summarize_metrics(m, keep.astype(bool), constants[total]["contact_fit" if stratum != "all" else "all"])
                if stratum == "contact":
                    entry["micro_ap_contact"], entry["micro_ap_prior_contact"] = m["micro_ap_contact"], m["micro_ap_prior_contact"]
                table[f"{key}|{stratum}"] = entry

        arms = [
            ("new", "segment_nocomp_spatial", args.new_run),
            ("new", "segment_comp_spatial", args.new_run),
            ("old", "segment_nocomp", args.old_run),
            ("old", "segment_comp", args.old_run),
            ("reference", "frame_rollout", args.old_run),
            ("reference", "map_union", args.old_run),
        ]
        rc = rcfg["readout"]
        for version, name, run in arms:
            ck = torch.load(run / name / "model.pt", map_location="cpu", weights_only=False)
            model = ARMS[name](cfg["training"]["dim"], cfg["training"]["predictor_depth"]).to(device)
            model.load_state_dict(ck["model"])
            model.eval()
            reps = {s: extract(model, store, pools[s], name) for s in ("train", "val_select", "val_decide")}
            label = f"{version}:{name}"
            # S1: direct 32/64 on stride-8 train windows; early stopping on val_select
            s1_sets = {}
            for split, stride in (("train", 8), ("val_select", 16)):
                sets = []
                for length in TRAIN_LENGTHS:
                    pool = [(e, t) for e in store.split_episodes(split) for t in range(0, store.lengths[e] - length, stride)]
                    if args.profile:
                        pool = pool[: rcfg["profile"]["max_windows"]]
                    toks = []
                    for i in range(0, len(pool), 256):
                        picks = pool[i : i + 256]
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            toks.append(model.rollout(store.batch([p[0] for p in picks], [p[1] for p in picks], length), [length], composed=False)["parts"][0].float())
                    sets.append((torch.cat(toks), labels.get([p[0] for p in pool], [(p[1], p[1] + length) for p in pool])))
                s1_sets[split] = sets
            s1, _ = fit_readout(s1_sets["train"], s1_sets["val_select"], rc, seed + 1, device)
            s2, _ = fit_readout([(reps["train"][f"direct{L}"], targets("train", L)) for L in DIRECT_LENGTHS],
                                [(reps["val_select"][f"direct{L}"], targets("val_select", L)) for L in DIRECT_LENGTHS], rc, seed + 2, device)
            for L in DIRECT_LENGTHS:
                for setting, ro in (("S1", s1), ("S2", s2)):
                    record(f"{label}|direct{L}|direct|{setting}", apply_readout(ro, reps["val_decide"][f"direct{L}"]), L, None)
                if name == "map_union":
                    record(f"{label}|direct{L}|native_heads|none", native_to_pred(reps["val_decide"][f"native_direct{L}"]), L, None)
            if f"true{DIRECT_LENGTHS[0]}" in reps["val_decide"]:
                ceiling, _ = fit_readout([(reps["train"][f"true{L}"], targets("train", L)) for L in DIRECT_LENGTHS],
                                         [(reps["val_select"][f"true{L}"], targets("val_select", L)) for L in DIRECT_LENGTHS], rc, seed + 3, device)
                for L in DIRECT_LENGTHS:
                    record(f"{label}|direct{L}|true_summary_ceiling|S2", apply_readout(ceiling, reps["val_decide"][f"true{L}"]), L, None)
            for eval_name, parts in EVAL_PARTS.items():
                total = sum(parts)
                for setting, ro in (("S1", s1), ("S2", s2)):
                    preds = [apply_readout(ro, reps["val_decide"][f"{eval_name}/part{k}"]) for k in range(len(parts))]
                    additive = {k: sum(p[k] for p in preds) for k in SCALARS}
                    additive["map"] = torch.stack([p["map"] for p in preds]).amax(0)
                    record(f"{label}|{eval_name}|additive|{setting}", additive, total, eval_name)
                    record(f"{label}|{eval_name}|direct_whole|{setting}", apply_readout(ro, reps["val_decide"][f"direct{total}"]), total, eval_name)
                if f"{eval_name}/composed" in reps["val_decide"]:
                    record(f"{label}|{eval_name}|composed|S1", apply_readout(s1, reps["val_decide"][f"{eval_name}/composed"]), total, eval_name)
                    ro, _ = fit_readout([(reps["train"][f"{eval_name}/composed"], targets("train", total))],
                                        [(reps["val_select"][f"{eval_name}/composed"], targets("val_select", total))], rc, seed + 4, device)
                    record(f"{label}|{eval_name}|composed|S2", apply_readout(ro, reps["val_decide"][f"{eval_name}/composed"]), total, eval_name)
                if name == "map_union":
                    record(f"{label}|{eval_name}|native_heads|none", native_to_pred(reps["val_decide"][f"{eval_name}/native"]), total, eval_name)
            result["table"] = table
            write_json(result_path, result)
            del model, reps
            torch.cuda.empty_cache()
            print(f"evaluated {label}", flush=True)

        rng = np.random.default_rng(seed)
        resamples = 200 if args.profile else 1000
        paired = {}

        def compare(a, b, total, stratum_keep, tag):
            for metric in ("count_empty", "count_increment"):
                paired[f"{a} vs {b}|{tag}|{metric}"] = episode_bootstrap_r(metrics_store[a], metrics_store[b], eps_decide, stratum_keep.astype(bool), metric, rng, resamples)

        # 1. target adequacy: new vs old true-summary readout, same arm
        adequacy = {}
        for new, old in (("new:segment_nocomp_spatial", "old:segment_nocomp"), ("new:segment_comp_spatial", "old:segment_comp")):
            deltas = []
            for L in DIRECT_LENGTHS:
                a, b = f"{new}|direct{L}|true_summary_ceiling|S2", f"{old}|direct{L}|true_summary_ceiling|S2"
                compare(a, b, L, has_contact[L], "contact")
                deltas.append(paired[f"{a} vs {b}|contact|count_empty"]["r_difference"])
                compare(f"{new}|direct{L}|direct|S2", f"{old}|direct{L}|direct|S2", L, has_contact[L], "contact")
            adequacy[new] = {"delta_r_by_length": dict(zip(map(str, DIRECT_LENGTHS), deltas)), "mean_delta_r": float(np.mean(deltas))}
        result["target_adequacy"] = adequacy

        # 2. composition within the new target
        composition = {}
        for eval_name in ALL_SPLITS:
            total = int(eval_name.split("=")[0])
            a = f"new:segment_comp_spatial|{eval_name}|composed|S2"
            row = {}
            for ref in ("additive", "direct_whole"):
                b = f"new:segment_nocomp_spatial|{eval_name}|{ref}|S2"
                compare(a, b, total, has_contact[total], "contact")
                compare(a, b, total, hard[eval_name] & has_contact[total], "hard")
                row[ref] = paired[f"{a} vs {b}|contact|count_empty"]
            compare(a, f"new:segment_comp_spatial|{eval_name}|additive|S2", total, has_contact[total], "contact")
            compare(a, f"old:segment_comp|{eval_name}|composed|S2", total, has_contact[total], "contact")
            composition[eval_name] = row
        result["composition"] = composition
        result["paired_episode_bootstrap"] = paired

        # locked decision rules
        target_improves = all(v["mean_delta_r"] >= 0.05 for v in adequacy.values())
        primary_ok = all(composition[s][ref]["r_difference"] > 0 for s in PRIMARY_SPLITS for ref in ("additive", "direct_whole"))
        some_ci = any(composition[s][ref]["r_ci95"][0] > 0 for s in PRIMARY_SPLITS for ref in ("additive", "direct_whole"))
        no_negative = all(composition[s][ref]["r_difference"] >= 0 for s in ALL_SPLITS for ref in ("additive", "direct_whole"))
        comp_better = primary_ok and some_ci and no_negative
        if not target_improves:
            decision = "TARGET_NOT_IMPROVED_STOP_SCRUB_FOR_THIS_DIRECTION"
        elif comp_better:
            decision = "COMPOSITION_CLEARLY_BETTER_CONFIRM_3_SEEDS_ON_UNUSED_SET"
        else:
            decision = "TARGET_IMPROVED_COMPOSITION_NOT_BETTER_CLOSE_VERSION_ON_SCRUB"
        result["decision"] = {"target_improves": target_improves, "composition_primary_positive": primary_ok,
                              "composition_some_ci_excludes_zero": some_ci, "composition_no_negative_point_estimate": no_negative,
                              "rule": decision}
        result["verdict"] = "PROFILE_COMPLETE" if args.profile else "TARGET_FIX_EVAL_COMPLETE"
        result["total_seconds"] = time.time() - t0
        write_json(result_path, result)
        print(f"DECISION {decision}", flush=True)
        print(f"WROTE {result_path}", flush=True)
    except Exception as error:
        result.update({"verdict": "ERROR", "error": repr(error), "traceback": traceback.format_exc()})
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
