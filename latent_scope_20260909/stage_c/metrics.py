"""Strict one-decision-per-candidate metrics; never silently drop malformed groups."""
import math


def candidate_metrics(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    groups = {}
    for row in rows:
        if not math.isfinite(row["score"]) or row["label"] not in (0, 1, False, True):
            raise ValueError("Non-finite score or non-binary continuation label")
        groups.setdefault((row["task"], row["group"]), []).append(row)
    baseline, selected, oracle = [], [], []
    for key, group in groups.items():
        n = int(group[0]["candidate_count"])
        if n < 2 or any(int(r["candidate_count"]) != n for r in group):
            raise ValueError(f"Inconsistent candidate count in {key}")
        group = sorted(group, key=lambda r: r["candidate"])
        if [r["candidate"] for r in group] != list(range(n)):
            raise ValueError(f"Duplicate/missing decision rows in {key}")
        baseline.append(group[0]["label"])
        # Stable tie-break: lowest candidate index, including the locked baseline 0.
        selected.append(max(group, key=lambda r: r["score"])["label"])
        oracle.append(max(r["label"] for r in group))
    b, s, o = (sum(values) / len(values) for values in (baseline, selected, oracle))
    return {"groups": len(groups), "baseline_success_rate": b,
            "selected_success_rate": s, "oracle_success_rate": o,
            "selected_gain_percentage_points": 100 * (s - b),
            "oracle_regret_percentage_points": 100 * (o - s)}
