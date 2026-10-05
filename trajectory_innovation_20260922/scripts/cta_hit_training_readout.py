"""Read saved continuation scores on a CPU compute node; never load models.

Reconstructs the independently selected CTA/ENDPOINT/DIRECT control and hit
states from saved evaluation points. Confidence intervals resample selection
roots jointly, keeping every comparison paired. They describe this selection
split; checkpoint selection on these same labels prevents a held-out claim.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ti_wm.contract import require_compute


TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")
FAMILIES = ("cta", "endpoint", "direct")
ARMS = tuple(f"{family}_{variant}" for family in FAMILIES for variant in ("control", "hit"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def optional_float(value):
    return float(value) if math.isfinite(value) else None


def metadata(config, limit):
    result = {}
    for name, directory in config["selection"].items():
        path = Path(directory) / "meta.npz"
        with np.load(path, allow_pickle=False) as archive:
            for key in ("root", "decision", "cov8", "done8"):
                if key not in archive.files:
                    raise ValueError(f"Missing {key} in {path}")
            root, decision = np.array(archive["root"]), np.array(archive["decision"])
            geometry = np.array(archive["cov8"], np.float32)
            flags = np.array(archive["done8"])
        count = len(root) if limit is None else min(int(limit), len(root))
        if root.shape != decision.shape or geometry.ndim != 2 or flags.shape != geometry.shape or len(geometry) != len(root):
            raise ValueError(f"Misaligned metadata in {path}")
        if not np.isfinite(geometry).all() or not np.isin(flags, (False, True)).all():
            raise ValueError(f"Invalid geometry/native termination metadata in {path}")
        keys = np.stack([root, decision], 1)
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError(f"Duplicate root/decision rows in {path}")
        result[name] = {"path": str(path), "sha256": sha256(path), "full_rows": len(root),
                        "root": root[:count], "decision": decision[:count],
                        "geometry": geometry[:count], "hits": flags[:count].astype(np.bool_)}
    return result


def load_scores(path, meta, arms):
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for arm in arms:
            arrays = []
            for bank, data in meta.items():
                key = f"{arm}__{bank}"
                if key not in archive.files:
                    raise ValueError(f"Missing {key} in {path}")
                scores = np.array(archive[key])
                if scores.shape != data["geometry"].shape or not np.isfinite(scores).all():
                    raise ValueError(f"Nonfinite or wrong-shape scores {key} in {path}: {scores.shape}")
                arrays.append(scores)
            result[arm] = np.concatenate(arrays)
    return result


def provenance(meta):
    rows = np.concatenate([np.stack([np.full(len(data["root"]), index), data["root"], data["decision"]], 1)
                           for index, data in enumerate(meta.values())])
    return {"row_order_sha256": hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
            "rows": len(rows), "roots": int(len(np.unique(rows[:, 1]))),
            "banks": {name: {"metadata": data["path"], "metadata_sha256": data["sha256"],
                              "selected_prefix_rows": len(data["root"]), "full_rows": data["full_rows"]}
                      for name, data in meta.items()},
            "alignment_scope": "score key/order/shape matched to configured cache root+decision prefix; score NPZs lack embedded row IDs"}


def root_sums(values, inverse, count):
    return np.bincount(inverse, weights=np.asarray(values, np.float64), minlength=count)


def metric_parts(scores, geometry, hits, inverse, count):
    chosen = scores.argmax(-1)
    index = np.arange(len(scores))
    picked = hits[index, chosen]
    eligible = hits.any(-1)
    mixed = eligible & ~hits.all(-1)
    crossing = eligible & ~hits[:, 0]
    return {
        "retained_geometry_gap": (root_sums(geometry[index, chosen] - geometry[:, 0], inverse, count),
                                  root_sums(geometry.max(-1) - geometry[:, 0], inverse, count)),
        "mixed_hit_capture": (root_sums(picked & mixed, inverse, count), root_sums(mixed, inverse, count)),
        "eligible_hit_capture": (root_sums(picked & eligible, inverse, count), root_sums(eligible, inverse, count)),
        "p0_crossing_hit_capture": (root_sums(picked & crossing, inverse, count), root_sums(crossing, inverse, count)),
    }


def ratio_stat(parts, resamples, difference=None):
    numerator, denominator = parts
    if difference is not None:
        other_num, other_den = difference
        if not np.array_equal(denominator, other_den):
            raise ValueError("Paired ratio denominators differ")
        numerator = numerator - other_num
    total = denominator.sum()
    sampled_den = denominator[resamples].sum(1)
    valid = sampled_den > 0
    boot = numerator[resamples].sum(1)[valid] / sampled_den[valid]
    interval = np.percentile(boot, [2.5, 97.5]) if len(boot) else (float("nan"), float("nan"))
    return {"estimate": optional_float(numerator.sum() / total) if total > 0 else None,
            "lo": optional_float(interval[0]), "hi": optional_float(interval[1]),
            "numerator": float(numerator.sum()), "denominator": float(total),
            "roots_with_denominator": int((denominator > 0).sum()),
            "undefined_resamples": int((~valid).sum())}


def assignment_path(source, target):
    """Resolve only literal Path/division assignments; do not execute source."""
    environment = {}
    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in environment:
            return environment[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1:
            return Path(value(node.args[0]))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return Path(value(node.left)) / value(node.right)
        raise ValueError("Not a literal path")
    for node in ast.parse(Path(source).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                environment[node.targets[0].id] = value(node.value)
            except (ValueError, KeyError, TypeError):
                pass
    return Path(environment[target]) if target in environment else None


def all_goal_call(source, function, position):
    tree = ast.parse(Path(source).read_text())
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function
               and len(node.args) > position and isinstance(node.args[position], ast.Name)
               and node.args[position].id == "goals" for node in ast.walk(tree))


def source_compatibility(args, config, base_config, meta):
    base_source = args.base_run.parent / "code/scripts/cta_train_v2.py"
    train_source = args.train_run.parent / "code/scripts/cta_hit_train.py"
    train_v2_source = args.train_run.parent / "code/scripts/cta_train_v2.py"
    reasons = []
    if config["selection"] != base_config.get("selection"):
        reasons.append("selection paths differ")
    if config["continuation"].get("selection_goal_images") != 16:
        reasons.append("new selection is not configured for all16 goal images")
    if any(len(data["root"]) != data["full_rows"] for data in meta.values()):
        reasons.append("new selection uses a truncated prefix")
    goals = None
    if not all(path.is_file() for path in (base_source, train_source, train_v2_source)):
        reasons.append("archived scoring source unavailable")
    else:
        original_goals = assignment_path(base_source, "GOALS")
        current_goals = assignment_path(train_v2_source, "GOALS")
        if original_goals is None or original_goals != current_goals:
            reasons.append("goal-file source provenance differs or cannot be resolved")
        else:
            goals = original_goals
        if not all_goal_call(base_source, "score_banks", 4):
            reasons.append("old final all-goal score call unavailable")
        if not all_goal_call(train_source, "select_scores", 3):
            reasons.append("new all-goal score call unavailable")
    goal_count = None
    if goals is not None:
        goal_array = np.load(goals, mmap_mode="r", allow_pickle=False)
        goal_count = len(goal_array)
        if goal_count != 16:
            reasons.append(f"goal-file length is {goal_count}, expected16")
    return {"compatible_for_supplemental_agreement": not reasons, "reasons": reasons,
            "goal_file": str(goals) if goals is not None else None, "goal_images": goal_count,
            "goal_file_sha256": sha256(goals) if goals is not None else None,
            "base_scoring_source": str(base_source), "new_scoring_source": str(train_source),
            "numerical_caveat": "old source scores use batch4; new prediction scores use fixed bank batch1, so numerical drift is possible",
            "historical_metadata_caveat": "old score archive has no row IDs/historical metadata hash; alignment relies on configured immutable feature-cache row order"}


def score_agreement(prediction, target, geometry, hits):
    a = prediction.astype(np.float64) - prediction.mean(-1, keepdims=True, dtype=np.float64)
    b = target.astype(np.float64) - target.mean(-1, keepdims=True, dtype=np.float64)
    target_rms = float(np.sqrt(np.mean(b * b)))
    norm = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    spread = np.ptp(geometry, axis=1) > 1e-3
    mixed = hits.any(-1) & ~hits.all(-1)
    match = prediction.argmax(-1) == target.argmax(-1)
    return {"bank_centered_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "bank_centered_normalized_rmse": float(np.sqrt(np.mean((a - b) ** 2)) / target_rms) if target_rms else None,
            "bank_centered_cosine": float(np.sum(a * b) / norm) if norm else None,
            "informative_bank_choice_agreement": float(match[spread].mean()) if spread.any() else None,
            "mixed_bank_choice_agreement": float(match[mixed].mean()) if mixed.any() else None,
            "comparison_scope": "paired fixed-reader target agreement; supplemental selection diagnostic"}


def score_scale(scores, hits, temperature):
    """Selection score spread/probabilities under a fixed training temperature.

    Selection averages16 goal images; training uses one goal image per bank.
    These are comparable state diagnostics, not estimates of training loss.
    """
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Saved native-hit temperature must be finite and positive")
    scores = scores.astype(np.float64)
    centered = scores - scores.mean(-1, keepdims=True)
    scaled = scores / temperature
    shifted = scaled - scaled.max(-1, keepdims=True)
    logprob = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    probability = np.exp(logprob)
    mixed = hits.any(-1) & ~hits.all(-1)
    logmass = np.logaddexp.reduce(np.where(hits[mixed], logprob[mixed], -np.inf), axis=-1)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    span = np.ptp(scores, axis=-1)
    return {"fixed_hit_temperature": temperature,
            "bank_centered_rms": rms, "bank_centered_rms_over_temperature": rms / temperature,
            "mean_bank_span": float(span.mean()), "mean_bank_span_over_temperature": float(span.mean() / temperature),
            "p90_bank_span_over_temperature": float(np.percentile(span / temperature, 90)),
            "mean_top1_softmax_probability": float(probability.max(-1).mean()),
            "mean_softmax_entropy_nats": float(-(probability * logprob).sum(-1).mean()),
            "mean_mixed_hit_softmax_mass": float(np.exp(logmass).mean()) if mixed.any() else None,
            "mean_mixed_hit_nll": float(-logmass.mean()) if mixed.any() else None,
            "scope": "full16-goal averaged selection score diagnostics, not one-goal training-loss estimates; controls use paired hit-arm temperature only for comparison"}


def exposure_readout(config, done, training):
    continuation = config["continuation"]
    expected_draws = continuation["steps"] * continuation["decisions"]
    actual = {}
    for group in ("control", "hit"):
        banks = done["exposure"][group]
        draws = sum(item["banks"] for item in banks.values())
        if draws != expected_draws:
            raise ValueError(f"{group} exposure totals {draws}, expected {expected_draws}")
        mixed = sum(item["mixed"] for item in banks.values())
        standard = sum(item["banks"] for name, item in banks.items() if name == "r4std")
        actual[group] = {"bank_draws": draws, "standard_fraction": standard / draws,
                         "mixed_fraction": mixed / draws, "mixed_draws": mixed,
                         "banks": banks}
    margin = float(config["rank_margin"])
    bank_counts = {}
    for name, directory in config["sources"].items():
        count = int(continuation["sampling"]["banks"][name]["banks"])
        with np.load(Path(directory) / "meta.npz", allow_pickle=False) as archive:
            geometry = np.array(archive["cov8"][:count], np.float32)
            hits = np.array(archive["done8"][:count], np.bool_)
        informative = np.ptp(geometry, -1) > margin
        mixed = hits.any(-1) & ~hits.all(-1)
        bank_counts[name] = {"rows": count, "informative": int(informative.sum()),
                             "mixed": int(mixed.sum()), "mixed_informative": int((mixed & informative).sum())}
    all_count = sum(item["rows"] for item in bank_counts.values())
    inf_count = sum(item["informative"] for item in bank_counts.values())
    if inf_count == 0:
        raise ValueError("Original sampler has no informative rows")
    spread = round(config["spread_frac"] * continuation["decisions"]) / continuation["decisions"]
    old_masses = {name: spread * item["informative"] / inf_count + (1 - spread) * item["rows"] / all_count
                  for name, item in bank_counts.items()}
    old_mixed = sum(spread * item["mixed_informative"] / inf_count + (1 - spread) * item["mixed"] / all_count
                    for item in bank_counts.values())
    return {"actual": actual, "expected_control": {"standard_fraction": old_masses.get("r4std", 0.),
                                                    "mixed_fraction": old_mixed, "mass_by_bank": old_masses},
            "expected_hit": {"standard_fraction": continuation["sampling"]["expected_standard_exposure"],
                             "mixed_fraction": continuation["sampling"]["expected_mixed_exposure"]},
            "native_training_bank_counts": bank_counts,
            "scope": "each exposure stream is shared by its three arms; repeated draws are not distinct episodes"}


def loss_readout(training):
    result = {}
    for arm in ARMS:
        records = [(event["step"], event[arm]) for event in training if arm in event]
        if not records:
            raise ValueError(f"No logged train losses for {arm}")
        parts = {key: [item["parts"][key] for _, item in records if key in item["parts"]]
                 for key in records[0][1]["parts"]}
        numbers = [item["loss"] for _, item in records]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f"Nonfinite saved losses for {arm}")
        result[arm] = {"logged_points": len(records), "first_step": records[0][0], "last_step": records[-1][0],
                       "first_loss": numbers[0], "last_loss": numbers[-1],
                       "parts_first": records[0][1]["parts"], "parts_last": records[-1][1]["parts"],
                       "parts_mean_of_logged_points": {key: float(np.mean(values)) for key, values in parts.items()},
                       "first_nonzero_logged_hit_step": next((step for step, item in records
                                                              if item["parts"].get("hit", 0.) > 0), None),
                       "gradient_norm_max_logged": max(item["gradient_norm_before_clip"] for _, item in records)}
    return {"arms": result,
            "interpretation": "control/hit losses are measured on different sampling distributions; decreases do not independently establish dynamics improvement",
            "sampling_scope": "sparse logged points, not a mean over every update"}


def interval_text(metric, percent=False):
    if metric["estimate"] is None:
        return "không xác định"
    factor = 100 if percent else 1
    label = " pp" if percent else ""
    return f"{metric['estimate'] * factor:+.3f}{label} [{metric['lo'] * factor:+.3f}, {metric['hi'] * factor:+.3f}]"


def point_text(value, percent=False):
    if value is None:
        return "không xác định"
    return f"{100 * value:.1f}%" if percent else f"{value:.3f}"


def markdown(report):
    rows = ["# CTA L=15: readout sau continuation", "",
            "Chỉ số offline trên selection split; cùng labels được dùng chọn checkpoint. CI bootstrap theo root, giữ pairing; chưa phải bằng chứng closed-loop hoặc test độc lập.", "",
            f"{report['alignment']['rows']} banks, {report['alignment']['roots']} roots; {report['goal_contract']['goal_images']} goal images. "
            f"Mỗi model đã train {report['updates']} updates; state được chọn có thể ở step0/1200/2400.", "",
            "| Model | Selected step | Geometry retained gap | Mixed hit capture |", "|---|---:|---:|---:|"]
    for arm, value in report["selected"].items():
        gap = value["metrics"]["retained_geometry_gap"]["estimate"]
        hit = value["metrics"]["mixed_hit_capture"]["estimate"]
        rows.append(f"| {arm} | {value['selected_step']} | {point_text(gap)} | {point_text(hit, True)} |")
    rows += ["", "| Paired contrast | Δ retained gap, 95% CI | Δ mixed capture, 95% CI |",
             "|---|---:|---:|"]
    for name, value in report["contrasts"].items():
        rows.append(f"| {name} | {interval_text(value['retained_geometry_gap'])} | {interval_text(value['mixed_hit_capture'], True)} |")
    rows += ["", "Exposure thực tế:", ""]
    for group, value in report["exposure"]["actual"].items():
        rows.append(f"- {group}: {value['bank_draws']} bank draws; standard {100 * value['standard_fraction']:.2f}%; mixed {100 * value['mixed_fraction']:.2f}%.")
    rows += ["", "Loss control/hit đo trên sampling distributions khác nhau; không suy dynamics tốt hơn từ loss thấp hơn. "
             "Can thiệp gộp replay/data rebalance với native-hit loss, chưa tách nguyên nhân gain.", "",
             "Frozen source/readers: trainer báo kiểm tra unchanged khi DONE; readout này không load checkpoint để kiểm chứng lại weights.", "",
             "Agreement với source scores lịch sử: " + ("đủ provenance để dùng như diagnostic bổ sung." if report["goal_contract"]["compatible_for_supplemental_agreement"]
                                                        else "không tính do thiếu compatibility: " + "; ".join(report["goal_contract"]["reasons"]) + "."),
             "Old source scores dùng batch4; prediction mới dùng batch1. Archive scores không chứa row IDs/hash metadata lịch sử; alignment theo configured immutable cache.", ""]
    if report.get("source_agreement"):
        rows += ["| Model | Source target | Centered normalized RMSE | Choice agreement trên mixed banks |",
                 "|---|---|---:|---:|"]
        for arm, value in report["source_agreement"].items():
            choice = value["mixed_bank_choice_agreement"]
            rows.append(f"| {arm} | {value['source_target']} | {point_text(value['bank_centered_normalized_rmse'])} | {point_text(choice, True)} |")
    rows += ["", "Curves theo state hiện tại (độc lập với state được chọn):", "",
             "| Step | Model | Score RMS / fixed T | Mean mixed-hit softmax mass | Source centered normalized RMSE |",
             "|---|---|---:|---:|---:|"]
    for step, models in report["score_scale_curve"].items():
        for arm, value in models.items():
            agreement = report["source_agreement_curve"].get(step, {}).get(arm, {})
            rows.append(f"| {step} | {arm} | {point_text(value['bank_centered_rms_over_temperature'])} | "
                        f"{point_text(value['mean_mixed_hit_softmax_mass'], True)} | {point_text(agreement.get('bank_centered_normalized_rmse'))} |")
    rows += ["", "Score RMS/softmax dùng mean của16 goal images trên selection. Training dùng một goal image mỗi bank; không xem đây là training-loss curve."]
    return "\n".join(rows) + "\n"


def main(args):
    require_compute()
    if args.resamples < 100 or args.seed < 0:
        raise ValueError("Require at least100 bootstrap resamples and a nonnegative seed")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = load_json(args.train_run / "config.json")
    continuation = config["continuation"]
    base_config = load_json(args.base_run / "config.json")
    events = [json.loads(line) for line in (args.train_run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    finished = [event for event in events if event.get("status") == "DONE"]
    if len(finished) != 1 or finished[0]["steps"] != continuation["steps"]:
        raise ValueError("Training lacks one complete matching DONE record")
    done = finished[0]
    selected_file = load_json(args.train_run / "selection.json")
    selected_steps = selected_file["selected_steps"]
    if selected_steps != done["selected_steps"] or set(selected_steps) != set(ARMS):
        raise ValueError("Final selected-step records disagree or miss arms")
    selection_events = [event for event in events if event.get("stage") == "selection"]
    steps = [event["step"] for event in selection_events]
    if len(set(steps)) != len(steps) or not steps or steps[0] != 0 or steps[-1] != continuation["steps"]:
        raise ValueError("Missing/duplicate selection curve endpoints")
    if any(step not in steps for step in selected_steps.values()):
        raise ValueError("Selected state has no saved evaluation scores")
    if selected_file["step"] != steps[-1] or selection_events[-1]["selected_steps"] != selected_steps:
        raise ValueError("Final selection JSON does not describe the final event")
    meta = metadata(config, continuation["selection_limit"])
    geometry = np.concatenate([data["geometry"] for data in meta.values()])
    hits = np.concatenate([data["hits"] for data in meta.values()])
    roots = np.concatenate([data["root"] for data in meta.values()])
    unique, inverse = np.unique(roots, return_inverse=True)
    resamples = np.random.default_rng(args.seed).integers(0, len(unique), size=(args.resamples, len(unique)))
    saved = {step: load_scores(args.train_run / f"selection_scores_{step}.npz", meta, ARMS) for step in steps}
    initial_pairs = {family: {"bitwise_equal": bool(np.array_equal(saved[0][f"{family}_control"], saved[0][f"{family}_hit"])),
                              "max_absolute_score_difference": float(np.max(np.abs(saved[0][f"{family}_control"] - saved[0][f"{family}_hit"]))) }
                     for family in FAMILIES}
    if not all(item["bitwise_equal"] for item in initial_pairs.values()):
        raise ValueError(f"Control/hit initial score pairing failed: {initial_pairs}")
    parts = {step: {arm: metric_parts(scores, geometry, hits, inverse, len(unique)) for arm, scores in arrays.items()}
             for step, arrays in saved.items()}
    # Recompute the fixed criterion from scores, including the first-state tie
    # rule. This checks that chosen-state provenance is not just a stale JSON.
    expected_steps, best = {}, {}
    for event in selection_events:
        step = event["step"]
        for arm in ARMS:
            metrics = parts[step][arm]
            num, den = metrics["retained_geometry_gap"]
            gap = num.sum() / den.sum() if den.sum() > 0 else float("nan")
            num, den = metrics["mixed_hit_capture"]
            capture = num.sum() / den.sum() if den.sum() > 0 else None
            logged = event["metrics"][arm]
            if not math.isclose(gap, logged["geometry"]["retained_gap"]["ratio"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Score/log geometry mismatch for {arm} at{step}")
            if capture != logged["native"]["mixed_capture"]:
                raise ValueError(f"Score/log native capture mismatch for {arm} at{step}")
            criterion = gap + .25 * (capture if capture is not None else 0.)
            if not math.isclose(criterion, logged["selection_criterion"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Score/log selection criterion mismatch for {arm} at{step}")
            if arm not in best or criterion > best[arm]:
                best[arm], expected_steps[arm] = criterion, step
    if expected_steps != selected_steps:
        raise ValueError(f"Saved selected-step mapping disagrees with recomputed criterion: {expected_steps} != {selected_steps}")
    selected_parts = {arm: parts[selected_steps[arm]][arm] for arm in ARMS}
    selected = {arm: {"selected_step": selected_steps[arm], "updates_executed": continuation["steps"],
                      "metrics": {name: ratio_stat(value, resamples) for name, value in selected_parts[arm].items()}}
                for arm in ARMS}
    contrasts = {}
    for family in FAMILIES:
        hit, control = f"{family}_hit", f"{family}_control"
        contrasts[f"{hit} - {control}"] = {name: ratio_stat(value, resamples, selected_parts[control][name])
                                           for name, value in selected_parts[hit].items()}
    for arm in ARMS:
        contrasts[f"{arm} - step0"] = {name: ratio_stat(value, resamples, parts[0][arm][name])
                                       for name, value in selected_parts[arm].items()}
    goal_contract = source_compatibility(args, config, base_config, meta)
    source_agreement, source_metrics, source_agreement_curve = {}, {}, {}
    if goal_contract["compatible_for_supplemental_agreement"]:
        target = load_scores(args.base_run / "selection_scores.npz", meta, ("code", "full"))
        for name, scores in target.items():
            target_parts = metric_parts(scores, geometry, hits, inverse, len(unique))
            source_metrics[name] = {metric: ratio_stat(value, resamples) for metric, value in target_parts.items()}
        for step, models in saved.items():
            source_agreement_curve[str(step)] = {}
            for arm, scores in models.items():
                source = "code" if arm.startswith("cta") else "full" if arm.startswith("endpoint") else None
                if source is not None:
                    value = score_agreement(scores, target[source], geometry, hits)
                    value["source_target"] = source
                    source_agreement_curve[str(step)][arm] = value
        for arm in ARMS:
            if arm in source_agreement_curve[str(selected_steps[arm])]:
                source_agreement[arm] = source_agreement_curve[str(selected_steps[arm])][arm]
    temperatures = continuation["hit_temperatures"]
    score_scale_curve = {str(step): {arm: score_scale(scores, hits, temperatures[arm.replace("_control", "_hit")])
                                   for arm, scores in models.items()}
                         for step, models in saved.items()}
    curve = {str(step): {arm: {name: optional_float(value[0].sum() / value[1].sum()) if value[1].sum() > 0 else None
                             for name, value in metrics.items()} for arm, metrics in models.items()}
             for step, models in parts.items()}
    report = {"job": os.environ["SLURM_JOB_ID"], "train_run": str(args.train_run), "base_run": str(args.base_run),
              "config_sha256": sha256(args.train_run / "config.json"), "updates": continuation["steps"],
              "alignment": provenance(meta), "goal_contract": goal_contract, "initial_pairing": initial_pairs,
              "selected_state_provenance": "selection.json/DONE/last selection event agree; selected steps also recomputed from every saved score array and fixed criterion",
              "selected": selected, "contrasts": contrasts, "state_curve": curve,
              "source_agreement": source_agreement, "source_agreement_curve": source_agreement_curve,
              "score_scale_curve": score_scale_curve, "source_metrics": source_metrics,
              "exposure": exposure_readout(config, done, [e for e in events if e.get("stage") == "train"]),
              "losses": loss_readout([e for e in events if e.get("stage") == "train"]),
              "trainer_reported_frozen_source_readers_unchanged": done.get("frozen_source_readers_unchanged"),
              "bootstrap": {"resamples": args.resamples, "seed": args.seed, "unit": "selection root",
                            "pairing": "same resampled root indices/denominators for every contrast",
                            "scope": "conditional descriptive selection uncertainty; checkpoints were selected on these labels; no multiple-comparison adjustment"},
              "limits": ["offline selection scores do not establish learned closed-loop success",
                         "selected checkpoints can come from different updates; end-state curve remains separately reported",
                         "combined data/replay/loss intervention does not isolate which component caused changes",
                         "readout loads no model checkpoints and does not independently attest frozen weights"]}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "training_readout.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "training_readout_vi.md").write_text(markdown(report))
    print(json.dumps({"out": str(args.out), "selected_steps": selected_steps,
                      "paired_contrasts": {key: {metric: value[metric] for metric in ("retained_geometry_gap", "mixed_hit_capture")}
                                           for key, value in contrasts.items() if "step0" not in key}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, default=TI / "cta_replan15_train_56504/train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
