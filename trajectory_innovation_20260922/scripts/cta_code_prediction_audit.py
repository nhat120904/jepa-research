"""Frozen code/reader-domain audit on selection banks; compute node only.

Source-hard-code conditions are privileged diagnostic oracles. Expected/MAP
conditions receive observed context and proposed actions only. This script
scores fixed offline banks and does not evaluate closed-loop success.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cta_factorized_direct_train import model_hash, sha256
from scripts.cta_hit_training_readout import metadata, metric_parts, provenance, ratio_stat
from scripts.cta_reader_calibrate import check_frozen
from scripts.cta_train_v2 import Bank
from ti_wm.contract import require_compute
from ti_wm.cta import Scorer, SourceEncoder, goal_scores
from ti_wm.cta_parallel import ParallelFSQWM
from ti_wm.sibling import PCA_DIM


TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")
K = 8
CONDITIONS = {
    "SRC_BASE": ("base", "source", True), "MEAN_BASE": ("base", "mean", False),
    "MAP_BASE": ("base", "map", False), "SRC_CTRL": ("control", "source", True),
    "MEAN_CTRL": ("control", "mean", False), "SRC_HIT": ("hit", "source", True),
    "MEAN_HIT": ("hit", "mean", False),
}


def map_code(logits, values):
    """Coordinate argmax on the predictor's exact FSQ grid; no future input."""
    if len(logits) != len(values) or not logits:
        raise ValueError("MAP logits/coordinate grids must have equal nonzero length")
    shape = logits[0].shape[:-1]
    if len(shape) != 2:
        raise ValueError("MAP logits require (candidates,tokens,levels) per coordinate")
    result = []
    for probability, grid in zip(logits, values):
        if probability.shape[:-1] != shape or grid.ndim != 1 or len(grid) != probability.shape[-1]:
            raise ValueError("MAP categorical logits/grid shapes differ")
        if probability.device != grid.device:
            raise ValueError("MAP categorical logits/grid devices differ")
        result.append(grid[probability.argmax(-1)])
    return torch.stack(result, -1)


def aligned_labels(root, decision, geometry, hits, score_root, score_decision):
    """Require explicit score-row IDs to match labels; never silently reorder."""
    root, decision, geometry, hits, score_root, score_decision = map(np.asarray,
        (root, decision, geometry, hits, score_root, score_decision))
    if root.ndim != 1 or root.shape != decision.shape:
        raise ValueError("Root/decision vectors must align")
    if not (np.array_equal(root, score_root) and np.array_equal(decision, score_decision)):
        raise ValueError("Score root/decision IDs differ from metadata label order")
    if geometry.ndim != 2 or len(geometry) != len(root) or hits.shape != geometry.shape:
        raise ValueError("Label bank shapes do not match row IDs")
    if not np.isfinite(geometry).all() or not np.isin(hits, (False, True)).all():
        raise ValueError("Invalid geometry/native-hit labels")
    if len(np.unique(np.stack([root, decision], 1), axis=0)) != len(root):
        raise ValueError("Duplicate root/decision rows")
    return geometry, hits.astype(np.bool_, copy=False)


def code_statistics(source, expected, hard, logits, levels):
    """One bank's prediction statistics; code MSE is not a task-value metric."""
    arrays = [value.detach().float().cpu().numpy().astype(np.float64) for value in (source, expected, hard)]
    centered = [value - value.mean(0, keepdims=True) for value in arrays]
    src, mean, mode = centered
    report = {"source_within_bank_variance": float(np.mean(src ** 2)),
              "mean_within_bank_variance": float(np.mean(mean ** 2)),
              "map_within_bank_variance": float(np.mean(mode ** 2)),
              "centered_mean_code_mse": float(np.mean((src - mean) ** 2)),
              "centered_map_code_mse": float(np.mean((src - mode) ** 2)),
              "raw_mean_code_mse": float(np.mean((arrays[0] - arrays[1]) ** 2)),
              "raw_map_code_mse": float(np.mean((arrays[0] - arrays[2]) ** 2)),
              "mean_map_code_mse": float(np.mean((arrays[1] - arrays[2]) ** 2))}
    nll, entropy, accuracy = [], [], []
    for coordinate, (probability, count) in enumerate(zip(logits, levels)):
        digits = (source[..., coordinate].float() * (count // 2) + count // 2).round().long()
        if not ((digits >= 0) & (digits < count)).all():
            raise ValueError("Source diagnostic target is outside FSQ coordinate grid")
        logprob = probability.float().log_softmax(-1)
        coordinate_nll = float(F.nll_loss(logprob.flatten(0, 1), digits.flatten()))
        coordinate_entropy = float(-(logprob.exp() * logprob).sum(-1).mean())
        coordinate_accuracy = float((probability.argmax(-1) == digits).float().mean())
        report[f"coordinate_{coordinate}_nll"] = coordinate_nll
        report[f"coordinate_{coordinate}_entropy_nats"] = coordinate_entropy
        report[f"coordinate_{coordinate}_entropy_fraction_of_max"] = coordinate_entropy / math.log(count)
        report[f"coordinate_{coordinate}_map_accuracy"] = coordinate_accuracy
        nll.append(coordinate_nll)
        entropy.append(coordinate_entropy)
        accuracy.append(coordinate_accuracy)
    report.update(mean_coordinate_nll=float(np.mean(nll)), mean_coordinate_entropy_nats=float(np.mean(entropy)),
                  mean_coordinate_map_accuracy=float(np.mean(accuracy)))
    return report


def config_check(args, base, last, final_config):
    cfg, other = base["config"], last["config"]
    calibration = other["reader_calibration"]
    if calibration["base_sha256"] != sha256(args.base) or Path(calibration["base"]).resolve() != args.base.resolve():
        raise ValueError("Reader calibration does not reference this exact BASE checkpoint")
    for key in ("sources", "selection", "chunk", "m", "conditional", "path"):
        if cfg.get(key) != other.get(key):
            raise ValueError(f"Reader calibration/BASE config differs for {key}")
    if cfg.get("chunk") != 15 or last["step"] != calibration["steps"]:
        raise ValueError("Require completed L15 final reader state")
    if set(last["readers"]) != {"reader_control", "reader_hit"}:
        raise ValueError("Calibration checkpoint lacks paired final readers")
    final = final_config["reader_calibration"]
    if final.get("actual_updates") != last["step"] or final["base_sha256"] != calibration["base_sha256"]:
        raise ValueError("Final calibration JSON disagrees with last-state checkpoint")
    frozen = final.get("frozen_check", {})
    if not all(frozen.get(key) is True for key in ("encoder_exactly_unchanged", "wm_exactly_unchanged", "encoder_wm_have_no_gradients")):
        raise ValueError("Calibration did not report a successful exact frozen check")
    for data in calibration["source_metadata"].values():
        if sha256(data["path"]) != data["sha256"]:
            raise ValueError(f"Calibration metadata changed: {data['path']}")
    return {"exact_base_checkpoint_sha256": calibration["base_sha256"],
            "calibration_last_step": last["step"], "calibration_selected_steps": final["selected_steps"],
            "calibration_trainer_reported_frozen_check": frozen,
            "scope": "audit loads BASE encoder/WM itself; last-reader snapshot does not contain encoder/WM tensors, so historical frozen status also relies on completed trainer check"}


def stat_text(stat, percent=False):
    if stat["estimate"] is None:
        return "không xác định"
    multiplier = 100 if percent else 1
    return f"{multiplier * stat['estimate']:+.3f} [{multiplier * stat['lo']:+.3f},{multiplier * stat['hi']:+.3f}]"


def markdown(report):
    rows = ["# CTA: code prediction và reader-domain audit", "",
            "Offline diagnostic trên selection split đã dùng chọn checkpoint; CI cluster theo root có tính mô tả. Source-hard-code là oracle dùng tương lai thật; MEAN/MAP chỉ dùng C,A. Không có closed-loop success mới.", "",
            f"{report['alignment']['rows']} banks / {report['alignment']['roots']} roots /16 goal images. CTRL/HIT dùng reader state cuối step{report['provenance']['reader_last_step']}, khác checkpoint được chọn nếu selectedstep0.", "",
            "| Condition | Privileged source | Retained geometry gap | Mixed hit capture |",
            "|---|---|---:|---:|"]
    for name, metric in report["conditions"].items():
        rows.append(f"| {name} | {metric['privileged_future_oracle']} | {stat_text(metric['metrics']['retained_geometry_gap'])} | {stat_text(metric['metrics']['mixed_hit_capture'], True)} |")
    rows += ["", "| Paired contrast | Δ retained gap | Δ mixed capture (pp) |", "|---|---:|---:|"]
    for name, metric in report["contrasts"].items():
        rows.append(f"| {name} | {stat_text(metric['retained_geometry_gap'])} | {stat_text(metric['mixed_hit_capture'], True)} |")
    signal = report["code_prediction"]
    rows += ["", "Code prediction signal (mean trên banks):", ""]
    for key in ("source_within_bank_variance", "mean_within_bank_variance", "map_within_bank_variance",
                "centered_mean_code_mse", "centered_map_code_mse", "mean_coordinate_nll", "mean_coordinate_entropy_nats"):
        rows.append(f"- {key}: {signal['mean_bank_statistics'][key]:.6f}")
    rows += ["", "Mean/MAP giữ coordinate grid khác nhau; code variance, NLL và accuracy là diagnostic, không chứng minh task information hay success. "
             "SOURCE changes đo semantic forgetting khi giữ encoder/labels cố định; MEAN changes đo readout của cùng predictor. MAP là input-domain probe, chưa phải control cải thiện được kiểm chứng.", ""]
    return "\n".join(rows)


def main(args):
    require_compute()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.resamples < 100 or args.seed < 0:
        raise ValueError("Require at least100 resamples and nonnegative seed")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    last = torch.load(args.reader_last, map_location="cpu", weights_only=False)
    final_config = json.loads((args.reader_last.parent / "config.json").read_text())
    checked = config_check(args, base, last, final_config)
    cfg = base["config"]
    meta = metadata(cfg, None)
    banks = [Bank(Path(path), name) for name, path in cfg["selection"].items()]
    goals = torch.from_numpy(np.load(args.goals, allow_pickle=False)).to(device)
    if tuple(goals.shape) != (16, 256, PCA_DIM) or not torch.isfinite(goals).all():
        raise ValueError("Expected all16 original goal feature images")
    original_goal = last["config"]["reader_calibration"]["goals"]
    if args.goals.resolve() != Path(original_goal["path"]).resolve() or sha256(args.goals) != original_goal["sha256"]:
        raise ValueError("Audit/calibration goal images differ")
    encoder = SourceEncoder(m=cfg["m"], conditional=bool(cfg.get("conditional", True)),
                            path=bool(cfg.get("path", True)), dropout=cfg["dropout"]).to(device)
    wm = ParallelFSQWM(m=cfg["m"], chunk=15, dropout=cfg["dropout"]).to(device)
    encoder.load_state_dict(base["stage1"]["enc"], strict=True)
    wm.load_state_dict(base["wms"]["cta"], strict=True)
    encoder.eval().requires_grad_(False)
    wm.eval().requires_grad_(False)
    readers = {}
    for name, states in (("base", base["stage1"]["reader"]), ("control", last["readers"]["reader_control"]),
                         ("hit", last["readers"]["reader_hit"])):
        reader = Scorer("code", m=cfg["m"], dropout=cfg["dropout"]).to(device)
        reader.load_state_dict(states, strict=True)
        readers[name] = reader.eval().requires_grad_(False)
    calibration_identity = last["config"]["reader_calibration"]["initial_state_sha256"]
    if model_hash(encoder) != calibration_identity["encoder"] or model_hash(wm) != calibration_identity["wm"]:
        raise ValueError("Audit frozen encoder/WM identity differs from calibration")
    check_frozen(encoder, wm, base)
    score_parts = {name: [] for name in CONDITIONS}
    code_rows = []
    score_roots, score_decisions = [], []
    with torch.inference_mode():
        for bank in banks:
            if bank.n != len(meta[bank.name]["root"]):
                raise ValueError("Bank/metadata selection row counts differ")
            values = tuple(getattr(wm, f"values_{coordinate}") for coordinate in range(len(wm.levels)))
            arrays = {name: np.empty((bank.n, K), np.float32) for name in CONDITIONS}
            for row in range(bank.n):
                ctx, future, actions, _ = bank.batch([row], device)
                with amp():
                    # Mean/MAP prediction receives no future argument.
                    expected, logits = wm(ctx, actions)
                    hard = map_code(logits, values)
                    source = encoder(ctx, future)  # privileged diagnostic target only
                    codes = {"source": source, "mean": expected, "map": hard}
                    for condition, (reader, domain, _) in CONDITIONS.items():
                        scores = goal_scores(readers[reader], ctx, codes[domain], goals)
                        arrays[condition][row] = scores.float().cpu().numpy().reshape(K)
                code_rows.append(code_statistics(source, expected, hard, logits, wm.levels))
                score_roots.append(int(bank.root[row]))
                score_decisions.append(int(meta[bank.name]["decision"][row]))
                if (row + 1) % 250 == 0:
                    print(json.dumps({"bank": bank.name, "rows": row + 1, "total": bank.n,
                                      "seconds": time.perf_counter() - started}), flush=True)
            for condition in CONDITIONS:
                score_parts[condition].append(arrays[condition])
    root = np.concatenate([data["root"] for data in meta.values()])
    decision = np.concatenate([data["decision"] for data in meta.values()])
    geometry = np.concatenate([data["geometry"] for data in meta.values()])
    hits = np.concatenate([data["hits"] for data in meta.values()])
    geometry, hits = aligned_labels(root, decision, geometry, hits, np.asarray(score_roots), np.asarray(score_decisions))
    scores = {condition: np.concatenate(parts) for condition, parts in score_parts.items()}
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Nonfinite audit scores")
    unique, inverse = np.unique(root, return_inverse=True)
    resamples = np.random.default_rng(args.seed).integers(0, len(unique), size=(args.resamples, len(unique)))
    parts = {name: metric_parts(value, geometry, hits, inverse, len(unique)) for name, value in scores.items()}
    conditions = {name: {"reader": CONDITIONS[name][0], "code_domain": CONDITIONS[name][1],
                         "privileged_future_oracle": CONDITIONS[name][2],
                         "metrics": {metric: ratio_stat(values, resamples) for metric, values in data.items()}}
                  for name, data in parts.items()}
    pairs = (("SRC_CTRL", "SRC_BASE"), ("SRC_HIT", "SRC_BASE"), ("MEAN_CTRL", "MEAN_BASE"),
             ("MEAN_HIT", "MEAN_BASE"), ("MAP_BASE", "MEAN_BASE"), ("SRC_BASE", "MEAN_BASE"),
             ("SRC_CTRL", "MEAN_CTRL"), ("SRC_HIT", "MEAN_HIT"), ("MEAN_HIT", "MEAN_CTRL"))
    contrasts = {f"{left} - {right}": {metric: ratio_stat(value, resamples, parts[right][metric])
                                        for metric, value in parts[left].items()} for left, right in pairs}
    code_arrays = {name: np.array([row[name] for row in code_rows], np.float64) for name in code_rows[0]}
    source_variance = code_arrays["source_within_bank_variance"].sum()
    signal = {"mean_bank_statistics": {name: float(value.mean()) for name, value in code_arrays.items()},
              "centered_mean_mse_over_source_variance": float(code_arrays["centered_mean_code_mse"].sum() / source_variance) if source_variance else None,
              "centered_map_mse_over_source_variance": float(code_arrays["centered_map_code_mse"].sum() / source_variance) if source_variance else None,
              "mean_variance_over_source_variance": float(code_arrays["mean_within_bank_variance"].sum() / source_variance) if source_variance else None,
              "map_variance_over_source_variance": float(code_arrays["map_within_bank_variance"].sum() / source_variance) if source_variance else None,
              "source_nonconstant_banks": int((code_arrays["source_within_bank_variance"] > 0).sum()),
              "map_nonconstant_banks": int((code_arrays["map_within_bank_variance"] > 0).sum()),
              "scope": "mean across equally weighted banks; variance centers across8candidates pertoken/coordinate; NLL/entropy average coordinates,candidates,tokens; not task sufficiency"}
    unchanged = check_frozen(encoder, wm, base)
    alignment = provenance(meta)
    alignment["alignment_scope"] = ("saved scores embed root/decision IDs checked against bank roots and configured metadata decision order; "
                                    "feature-array row alignment relies on the configured cache")
    report = {"job": os.environ["SLURM_JOB_ID"], "alignment": alignment, "config_check": checked,
              "provenance": {"base": str(args.base), "base_sha256": sha256(args.base),
                             "reader_last": str(args.reader_last), "reader_last_sha256": sha256(args.reader_last),
                             "reader_last_step": last["step"], "goal_path": str(args.goals), "goal_sha256": sha256(args.goals),
                             "encoder_state_sha256": model_hash(encoder), "wm_state_sha256": model_hash(wm),
                             "reader_state_sha256": {name: model_hash(reader) for name, reader in readers.items()}},
              "conditions": conditions, "contrasts": contrasts, "code_prediction": signal, "audit_frozen_check": unchanged,
              "bootstrap": {"unit": "selection root", "resamples": args.resamples, "seed": args.seed,
                            "scope": "paired descriptive selection uncertainty; already used for model selection; exploratory comparisons without multiplicity adjustment"},
              "backend": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                          "precision": "bfloat16 autocast,float32 logits/scores", "bank_batch": 1, "goal_images": 16,
                          "cudnn_deterministic": True, "cudnn_benchmark": False, "cudnn_tf32": True, "matmul_tf32": False},
              "seconds": time.perf_counter() - started,
              "limits": ["source code uses privileged future observations and is an oracle diagnostic",
                         "mean/MAP code uses C,A only; readers consume C,code,g only",
                         "trained readers are final-step states, not independently selected checkpoints",
                         "offline MAP/mean rankings do not establish closed-loop gains"]}
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / "scores.npz", root=root, decision=decision, geometry=geometry, native_hit=hits, **scores)
    np.savez(args.out / "code_statistics.npz", root=root, decision=decision, **code_arrays)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "audit_vi.md").write_text(markdown(report))
    print(json.dumps({"out": str(args.out), "rows": len(root), "conditions": conditions,
                      "code_prediction": signal}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=TI / "cta_replan15_train_56504/train/cta_v2.pt")
    parser.add_argument("--reader-last", type=Path, required=True)
    parser.add_argument("--goals", type=Path, default=TI / "cta_geometry_e2e_55018/features/goals.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
