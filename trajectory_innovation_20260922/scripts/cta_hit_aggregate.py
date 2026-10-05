"""Paired development readout for canonical L15 native-hit closed loops.

Array reads/statistics must run in an explicitly resourced CPU sbatch job.
Completed shards are required. Truncated smoke runs cannot be reported as a
success experiment. Root identities and experiment hashes are checked before
pooling; historical batched-proposal runs are not pooled with this pilot.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ti_wm.contract import require_compute
from ti_wm.cta_batch import chosen_retention, headroom
from ti_wm.cta_eval import ranking_metrics
from ti_wm.cta_hit_contracts import format_stat, json_ready, validate_decision_domains
from ti_wm.gates import cluster_ratio, mcnemar_exact, paired_diff, wilson


DEFAULT_PAIRS = ("CTA_HIT-CTA_CTRL,CTA_HIT-CTA_BASE,CTA_HIT-DIR_HIT,CTA_HIT-END_HIT,CTA_HIT-P0,CTA_HIT-GEOM8,"
                 "END_HIT-END_CTRL,DIR_HIT-DIR_CTRL,CTA_CTRL-DIR_CTRL,CTA_CTRL-END_CTRL")


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), indent=2, default=str, allow_nan=False) + "\n")


def identity(report):
    keys = ("schema", "arms", "log_scorers", "checkpoint_specs", "checkpoints", "n_exec", "draw", "bank",
            "proposal_mode", "proposal_microbatch", "score_root_microbatch", "goal_images", "hashes", "backend",
            "max_decisions", "learned_input_contract", "clone_method")
    return {k: report[k] for k in keys}


def shards(paths):
    out = []
    for path in paths:
        if (path / "closed_report.json").exists():
            out.append(path)
        else:
            found = sorted(path.glob("shard_*/closed_report.json"))
            if not found:
                raise FileNotFoundError(f"No pilot shards in {path}")
            out.extend(p.parent for p in found)
    return out


def load_runs(paths):
    reports, episodes, logs, expected = [], {}, {}, None
    for shard in shards(paths):
        report = json.loads((shard / "closed_report.json").read_text())
        if report["status"] != "DONE" or report["schema"] != "cta_hit_closed_v1":
            raise ValueError(f"Incomplete/wrong-format shard {shard}")
        if report.get("qualification", {}).get("status") != "PASS":
            raise ValueError(f"Missing numerical qualification in {shard}")
        now = identity(report)
        if expected is None:
            expected = now
        elif now != expected:
            raise ValueError(f"Experiment/backend/checkpoint identity differs in {shard}")
        if report["max_decisions"] is not None:
            raise ValueError("Smoke max-decisions outputs cannot be pooled as a completed success experiment")
        shard_eps = {arm: {} for arm in report["arms"]}
        for line in (shard / "episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            arm, root = ep["arm"], ep["root"]
            if any(not isinstance(ep[key], bool) for key in ("success", "terminated", "truncated_by_max_decisions", "reset_excluded")):
                raise ValueError(f"Nonboolean episode outcome for {arm}/{root}")
            if arm not in shard_eps or root in shard_eps[arm] or root in episodes.get(arm, {}):
                raise ValueError(f"Unknown arm/duplicate root {arm}/{root} in {shard}")
            if ep["truncated_by_max_decisions"] or not ep["terminated"]:
                raise ValueError(f"Truncated episode {arm}/{root}")
            if not ep["reset_excluded"] or not np.isfinite([ep["score"], ep["max_coverage"]]).all():
                raise ValueError(f"Invalid score/coverage contract for {arm}/{root}")
            shard_eps[arm][root] = ep
        for arm in report["arms"]:
            if sorted(shard_eps[arm]) != sorted(report["roots"]):
                raise ValueError(f"Episode root population mismatch for {arm} in {shard}")
            episodes.setdefault(arm, {}).update(shard_eps[arm])
            with np.load(shard / f"log_{arm}.npz", allow_pickle=False) as blob:
                part = {k: blob[k] for k in blob.files}
            if set(part["root"]) != set(report["roots"]):
                raise ValueError(f"Decision root population mismatch for {arm} in {shard}")
            required = {"root", "decision", "t", "chosen", "cov", "geom", "native_hit"}
            required.update(f"score_{name}" for name in report["log_scorers"])
            if set(part) != required:
                raise ValueError(f"Unexpected decision fields in {shard}/{arm}")
            validate_decision_domains(part["chosen"], part["native_hit"].flat)
            for key, value in part.items():
                if len(value) != len(part["root"]) or not np.isfinite(value).all():
                    raise ValueError(f"Unaligned/nonfinite {shard}/{arm}/{key}")
                if key in ("cov", "geom", "native_hit") or key.startswith("score_"):
                    if value.shape != (len(part["root"]), 8):
                        raise ValueError(f"Bank shape mismatch in {shard}/{arm}/{key}")
                logs.setdefault(arm, {}).setdefault(key, []).append(value)
            for root in report["roots"]:
                decisions = part["decision"][part["root"] == root]
                if not np.array_equal(decisions, np.arange(len(decisions))):
                    raise ValueError(f"Decision order/duplicates in {shard}/{arm}/{root}")
        reports.append({"path": str(shard), **report})
    roots = sorted(next(iter(episodes.values())))
    if any(sorted(e) != roots for e in episodes.values()):
        raise ValueError("Arms do not have exactly matched roots")
    merged = {arm: {k: np.concatenate(v) for k, v in fields.items()} for arm, fields in logs.items()}
    return roots, episodes, merged, reports, expected


def capture(hit, choices, roots):
    """Native-success opportunities where candidate 0 would not succeed."""
    hit = np.asarray(hit, bool)
    opportunity = hit.any(axis=1) & ~hit[:, 0]
    got = opportunity & hit[np.arange(len(hit)), choices]
    uniq = np.unique(roots)
    num = np.array([got[roots == r].sum() for r in uniq], float)
    den = np.array([opportunity[roots == r].sum() for r in uniq], float)
    interval = cluster_ratio(num, den) if den.sum() else {"ratio": None, "lo": None, "hi": None}
    return {"decisions": len(hit), "opportunities": int(opportunity.sum()), "captured": int(got.sum()),
            "roots_with_opportunity": int((den > 0).sum()), "capture": interval}


def strip(metrics):
    return {key: value for key, value in metrics.items() if key != "chosen"}


def aggregate(a):
    require_compute()
    roots, episodes, logs, reports, experiment = load_runs(a.closed_run)
    if a.expect_roots is not None and roots != list(range(a.expect_roots[0], a.expect_roots[1] + 1)):
        raise ValueError("Actual roots differ from requested pilot population")
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "summary.json").exists():
        raise FileExistsError("Refusing to overwrite an aggregate")
    names, arms = experiment["log_scorers"], experiment["arms"]
    vectors = {arm: {key: np.array([episodes[arm][r][key] for r in roots])
                     for key in ("success", "score", "max_coverage", "steps")} for arm in arms}
    summary = {"schema": "cta_hit_aggregate_v1", "scope": "development pilot, exploratory; no sealed claims",
               "roots": roots, "n": len(roots), "experiment": experiment, "source_shards": [r["path"] for r in reports],
               "arms": {}, "contrasts": {}, "onpolicy": {}, "cost": {}}
    for arm in arms:
        vec, log = vectors[arm], logs[arm]
        wins = int(vec["success"].sum())
        summary["arms"][arm] = {"successes": wins, "success_rate": wins / len(roots),
                                "success_wilson95": wilson(wins, len(roots)), "mean_score": float(vec["score"].mean()),
                                "mean_max_coverage": float(vec["max_coverage"].mean()),
                                "mean_steps": float(vec["steps"].mean()), "decisions": len(log["root"]),
                                "override_fraction": float((log["chosen"] != 0).mean())}
        entry = {"native_hit": {"own_choices": capture(log["native_hit"], log["chosen"], log["root"]),
                                "scorers": {}}}
        for label in ("geom", "cov"):
            entry[label] = {"headroom": headroom(log[label]),
                            "own_choices": chosen_retention(log[label], log["chosen"], log["root"]),
                            "scorers": {name: strip(ranking_metrics(log[f"score_{name}"], log[label], log["root"]))
                                        for name in names}}
        for name in names:
            pick = np.argmax(log[f"score_{name}"], axis=1)
            entry["native_hit"]["scorers"][name] = capture(log["native_hit"], pick, log["root"])
        summary["onpolicy"][arm] = entry
        cost = {"proposal_seconds": 0., "simulator_seconds": 0., "shared_context_seconds": 0.,
                "diagnostic_wall_seconds": 0., "proposal_root_banks": 0, "context_calls": 0,
                "scorer_seconds": {name: 0. for name in names}, "scorer_calls": {name: 0 for name in names}}
        for report in reports:
            source = report["timing"][arm]
            for key in cost:
                if isinstance(cost[key], dict):
                    for name in names:
                        cost[key][name] += source[key][name]
                else:
                    cost[key] += source[key]
        cost["proposal_ms_per_root_bank"] = 1000 * cost["proposal_seconds"] / cost["proposal_root_banks"]
        cost["shared_context_ms_per_root_bank"] = 1000 * cost["shared_context_seconds"] / cost["context_calls"]
        cost["scorer_ms_per_root_bank"] = {name: 1000 * cost["scorer_seconds"][name] / cost["scorer_calls"][name]
                                           for name in names}
        cost["deployment_timing_claim"] = False
        cost["native_k1_policy_latency_measured"] = False
        cost["notes"] = ("Simulator and cross-scoring all variants are diagnostic costs; per-net costs share context. "
                         "Canonical K1 pads to 8. This run does not benchmark native P0 latency or total deployment latency.")
        summary["cost"][arm] = cost
    for pair in a.pairs.split(","):
        left, right = pair.split("-", 1)
        if left not in vectors or right not in vectors:
            raise ValueError(f"Requested contrast missing an arm: {pair}")
        summary["contrasts"][pair] = {key: paired_diff(vectors[left][key], vectors[right][key])
                                      for key in ("success", "score", "max_coverage")}
        summary["contrasts"][pair]["mcnemar"] = mcnemar_exact(vectors[left]["success"], vectors[right]["success"])
    write_json(a.out / "summary.json", summary)
    lines = ["Development pilot: matched roots, canonical K=8, L=15. Confidence intervals resample roots.", "",
             "| Arm | Success | Mean score | Mean max coverage | Own geometry retention |", "|---|---:|---:|---:|---:|"]
    for arm, row in summary["arms"].items():
        gap = summary["onpolicy"][arm]["geom"]["own_choices"]["ratio"]
        lines.append(f"| {arm} | {row['successes']}/{len(roots)} | {row['mean_score']:.3f} | "
                     f"{row['mean_max_coverage']:.3f} | {format_stat(gap)} |")
    lines += ["", "Paired contrasts (success percentage points, bootstrap 95% CI, exact McNemar):", ""]
    for pair, result in summary["contrasts"].items():
        diff, test = result["success"], result["mcnemar"]
        lines.append(f"- {pair}: {100 * diff['mean']:+.1f} pp [{100 * diff['lo']:+.1f}, {100 * diff['hi']:+.1f}]; "
                     f"discordant {test['only_a']}/{test['only_b']}, p={test['p']:.4f}.")
    lines += ["", "Geometry ranking on P0 states:", ""]
    if "P0" in summary["onpolicy"]:
        for name in names:
            interval = summary["onpolicy"]["P0"]["geom"]["scorers"][name]["retained_gap"]
            hit = summary["onpolicy"]["P0"]["native_hit"]["scorers"][name]
            lines.append(f"- {name}: retained gap {format_stat(interval['ratio'])} "
                         f"[{format_stat(interval['lo'])}, {format_stat(interval['hi'])}]; "
                         f"native crossing capture {hit['captured']}/{hit['opportunities']}.")
    lines += ["", "Diagnostic timing excludes any deployment speed claim; simulator and cross-scoring costs are separate."]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--closed-run", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.add_argument("--expect-roots", nargs=2, type=int)
    aggregate(p.parse_args())
