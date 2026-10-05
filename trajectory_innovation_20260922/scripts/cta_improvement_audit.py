"""Read-only evidence for the 2026-10-02 CTA improvement review; submit on a CPU compute node.

Reads archived metadata and reports, never images, model checkpoints, or sealed roots.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")
MARGIN = 1e-3


def metadata(path):
    with np.load(path) as z:
        geom, native = z["cov8"], z["native_cov8"]
        roots = z["root"]
    hit = native > .95
    spread = np.ptp(geom, axis=1) > MARGIN
    mixed = hit.any(1) & ~hit.all(1)
    crossing = hit.any(1) & ~hit[:, 0]
    best = np.argmax(geom, axis=1)
    misses = hit.any(1) & ~hit[np.arange(len(hit)), best]
    return {
        "path": str(path), "banks": len(roots), "roots": len(np.unique(roots)),
        "geometry_informative": int(spread.sum()), "native_mixed_success": int(mixed.sum()),
        "native_crossings": int(crossing.sum()),
        "mixed_success_ignored_by_geometry_bank_margin": int((mixed & ~spread).sum()),
        "geometry_oracle_misses_available_success": int(misses.sum()),
        "mixed_success_fraction": float(mixed.mean()),
        "mixed_success_fraction_in_geometry_informative": float(mixed[spread].mean()),
    }


def episodes(path):
    out = {}
    for shard in sorted(path.glob("shard_*")):
        report = json.loads((shard / "closed_report.json").read_text())
        if report["status"] != "DONE":
            raise ValueError(f"Incomplete shard: {shard}")
        for line in (shard / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            key = (e["arm"], e["root"])
            if key in out:
                raise ValueError(f"Duplicate episode: {key}")
            out[key] = e
    return out


def compare_runs(left, right):
    a, b = episodes(left), episodes(right)
    report = {}
    for arm in ("P0", "GEOM8"):
        roots_a = {root for name, root in a if name == arm}
        roots_b = {root for name, root in b if name == arm}
        if roots_a != roots_b or roots_a != set(range(2200, 2400)):
            raise ValueError(f"Different root sets for {arm}")
        roots = sorted(roots_a)
        flips = [r for r in roots if a[arm, r]["success"] != b[arm, r]["success"]]
        steps = [r for r in roots if a[arm, r]["steps"] != b[arm, r]["steps"]]
        coverage = [abs(a[arm, r]["max_coverage"] - b[arm, r]["max_coverage"]) for r in roots]
        report[arm] = {
            "left_successes": sum(a[arm, r]["success"] for r in roots),
            "right_successes": sum(b[arm, r]["success"] for r in roots),
            "success_flips": len(flips), "success_flip_roots": flips,
            "step_mismatches": len(steps), "max_coverage_abs_difference": max(coverage),
            "exact_episode_reproduction": not flips and not steps and max(coverage) == 0,
        }
    return report


def main(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit with sbatch; bulk analysis is compute-node work")
    cfg_path = TI / "cta_replan15_train_56504/train/config.json"
    cfg = json.loads(cfg_path.read_text())
    splits = {n: metadata(Path(p) / "meta.npz") for n, p in cfg["sources"].items()}
    splits.update({n: metadata(Path(p) / "meta.npz") for n, p in cfg["selection"].items()})
    n_all = sum(splits[n]["banks"] for n in cfg["sources"])
    n_inf = sum(splits[n]["geometry_informative"] for n in cfg["sources"])
    mass = {n: cfg["spread_frac"] * splits[n]["geometry_informative"] / n_inf
            + (1 - cfg["spread_frac"]) * splits[n]["banks"] / n_all for n in cfg["sources"]}
    # Success probability under the actual two-pool sampler; pool membership differs from dataset membership.
    mix_prob = 0.0
    for n, p in cfg["sources"].items():
        with np.load(Path(p) / "meta.npz") as z:
            spread = np.ptp(z["cov8"], axis=1) > MARGIN
            hit = z["native_cov8"] > .95
            mixed = hit.any(1) & ~hit.all(1)
        mix_prob += cfg["spread_frac"] * int((mixed & spread).sum()) / n_inf
        mix_prob += (1 - cfg["spread_frac"]) * int(mixed.sum()) / n_all
    metrics = [json.loads(line) for line in (cfg_path.parent / "metrics.jsonl").read_text().splitlines()]
    curves = [m for m in metrics if "selection_gap" in m]
    report = {
        "job_id": os.environ["SLURM_JOB_ID"], "scope": "archived train/selection/development data only",
        "config": str(cfg_path), "splits": splits,
        "expected_training_mass_by_source": mass,
        "mixed_native_success_probability_under_sampler": mix_prob,
        "native_success_aux_enabled": cfg.get("success_bonus", 0) != 0,
        "hindsight_enabled": cfg.get("hindsight", 0) != 0,
        "selection_curves": curves,
        "reproduction": compare_runs(TI / "cta_replan15_headroom_56374", TI / "cta_replan15_closed_56597"),
        "reproduction_note": "Different batch sizes/code paths; this identifies differences, not their cause.",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
