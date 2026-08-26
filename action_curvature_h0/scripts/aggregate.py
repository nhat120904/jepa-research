#!/usr/bin/env python3
"""Stage-1 aggregation: pool shards, bootstrap, emit the locked readouts.

Consumes the per-snapshot shards written by ``measure_curvature.py`` and emits
PROTOCOL.md readouts 1-10 with snapshot-clustered bootstrap CIs, the descriptive
curvature taxonomy, the two boundary detectors' agreement, the offline-vs-CEM
region comparison, and the locked Stage-1 verdict.

Threshold-free quantities carry the CI-backed claims.  The taxonomy cross-tab
uses the two cut-points declared in the protocol before any measurement; they
label rows of a descriptive table and no claim rests on them alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnosis"))

from metrics.bootstrap import bootstrap_ci  # noqa: E402

from action_curvature_h0.core import (  # noqa: E402
    CROSS_MODE,
    SAME_MODE_CONTACT,
    SAME_MODE_NON_CONTACT,
)

STRATA = (SAME_MODE_NON_CONTACT, SAME_MODE_CONTACT, CROSS_MODE)
SAME_MODE = (SAME_MODE_NON_CONTACT, SAME_MODE_CONTACT)

# Continuous metrics summarized per stratum with clustered CIs.
METRICS = (
    "e_k", "e_j", "g_k", "k_model_self", "k_model_true", "k_true", "a_k",
    "model_ratio", "true_ratio", "d2c_mismatch_normalized",
    "s_model", "s_true", "s_ratio",
    "model_radial_fraction", "model_angular_fraction",
    "error_radial_fraction", "error_angular_fraction",
    "k_true_state_object", "k_true_state_effector",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-root", type=Path, required=True,
                   help="Directory holding <source>/snapshot_XXX/ shard directories.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260824)
    # Declared in PROTOCOL.md, second pre-execution amendment, table B.
    p.add_argument("--curvature-high", type=float, default=0.25)
    p.add_argument("--mismatch-high", type=float, default=0.25)
    p.add_argument("--effector-gate", type=float, default=0.25,
                   help="K_true_state_effector above this indicts the pipeline.")
    p.add_argument("--smooth-alpha-low", type=float, default=1.5,
                   help="alpha above this counts as the smooth (delta^2) regime.")
    p.add_argument("--min-r2", type=float, default=0.90,
                   help="No alpha is assigned below this fit quality.")
    p.add_argument("--min-sensitivity-quantile", type=float, default=0.10,
                   help="Drop records whose S_true is below this fraction of the "
                        "median S_true in their own (source, horizon, sigma) cell.")
    # Preregistered primary contrast cell (amendment E).
    p.add_argument("--primary-horizon", type=int, default=5)
    p.add_argument("--primary-sigma", type=float, default=0.10)
    p.add_argument("--primary-source", default="cem_fixed")
    p.add_argument("--reference-source", default="dataset")
    p.add_argument("--self-test", action="store_true",
                   help="Run the internal consistency checks on synthetic shards and exit.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_shards(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/snapshot_*/summary.json")):
        shard = summary_path.parent
        summary = json.loads(summary_path.read_text())
        rows = json.loads((shard / "records.json").read_text())
        order = int(summary["snapshot"]["order"])
        source = str(summary["action_source"])
        for row in rows:
            row["snapshot"] = order
            row["action_source"] = source
            records.append(row)
        summary["shard"] = str(shard.relative_to(root))
        summaries.append(summary)
    if not records:
        raise RuntimeError(f"no shards found under {root}")
    return records, summaries


def select(records: Iterable[dict[str, Any]], **conditions: Any) -> list[dict[str, Any]]:
    out = []
    for r in records:
        if all(r.get(k) == v for k, v in conditions.items()):
            out.append(r)
    return out


def valid_only(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clipped triplets are excluded from every scientific quantity."""
    return [r for r in records if bool(r.get("valid_unclipped"))]


def apply_sensitivity_gate(
    records: list[dict[str, Any]], quantile: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop directions where the realized map barely moves (amendment C).

    E_K's denominator vanishes at stationary points and null directions of the
    realized Jacobian, where a tiny second-order error inflates it without
    bound while the map stays smooth.  The rule is fixed in advance and the
    removal count is reported.
    """
    cells: dict[tuple[Any, int, float], list[float]] = defaultdict(list)
    for r in records:
        cells[(r["action_source"], r["horizon"], r["sigma"])].append(r["s_true"])
    medians = {
        key: float(np.median([v for v in values if np.isfinite(v)]) )
        for key, values in cells.items()
        if any(np.isfinite(v) for v in values)
    }
    kept, dropped = [], 0
    for r in records:
        key = (r["action_source"], r["horizon"], r["sigma"])
        threshold = quantile * medians.get(key, 0.0)
        if np.isfinite(r["s_true"]) and r["s_true"] >= threshold:
            kept.append(r)
        else:
            dropped += 1
    return kept, {"dropped": dropped, "kept": len(kept),
                  "quantile": quantile, "n_cells": len(medians)}


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def ci_of(records: list[dict[str, Any]], metric: str, args: argparse.Namespace,
          statistic=np.median) -> dict[str, Any]:
    values = np.array([r[metric] for r in records if np.isfinite(r.get(metric, np.nan))],
                      dtype=np.float64)
    groups = np.array([r["snapshot"] for r in records
                       if np.isfinite(r.get(metric, np.nan))])
    if values.size == 0:
        return {"n": 0, "point": None, "low": None, "high": None}
    result = bootstrap_ci(values, statistic=statistic, n_resamples=args.n_resamples,
                          seed=args.seed, groups=groups)
    return {"n": int(values.size), "n_snapshots": int(np.unique(groups).size),
            "point": result.point, "low": result.low, "high": result.high}


def stratified_table(records: list[dict[str, Any]], args: argparse.Namespace,
                     horizon: int | None = None) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for stratum in STRATA:
        chosen = [r for r in records if r["mode"] == stratum]
        if horizon is not None:
            chosen = [r for r in chosen if r["horizon"] == horizon]
        table[stratum] = {m: ci_of(chosen, m, args) for m in METRICS}
    return table


def taxonomy(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Descriptive cross-tab; cut-points declared before measurement."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        k_model = r["k_model_true"]
        k_true = r["k_true"]
        e_k = r["e_k"]
        if not all(np.isfinite(v) for v in (k_model, k_true, e_k)):
            continue
        hi_model = k_model > args.curvature_high
        hi_true = k_true > args.curvature_high
        hi_mismatch = e_k > args.mismatch_high
        if hi_mismatch and hi_model and not hi_true:
            label = "spurious_model_curvature"
        elif hi_mismatch and hi_true and not hi_model:
            label = "model_erases_real_nonlinearity"
        elif hi_mismatch:
            label = "same_magnitude_wrong_direction"
        elif hi_true and hi_model:
            label = "genuine_physical_nonlinearity"
        else:
            label = "model_genuinely_smooth"
        counts[r["mode"]][label] += 1
    return {mode: dict(labels) for mode, labels in counts.items()}


def horizon_curves(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Readout 5: model and realized curvature on shared axes, with sensitivity.

    Curvature falling with H while sensitivity also falls is a model going deaf
    over the rollout, not a model becoming well behaved.
    """
    out: dict[str, Any] = {}
    for horizon in sorted({r["horizon"] for r in records}):
        chosen = [r for r in records if r["horizon"] == horizon]
        out[str(horizon)] = {
            m: ci_of(chosen, m, args)
            for m in ("k_model_self", "k_true", "e_k", "s_model", "s_true")
        }
    return out


def detector_agreement(records: list[dict[str, Any]], summaries: list[dict[str, Any]],
                       args: argparse.Namespace) -> dict[str, Any]:
    """Readout 8: contact-pattern mode label vs the scaling exponent.

    Disagreement localizes mode changes the contact proxy misses (slip/roll
    transitions, joint limits) and is informative rather than an error.
    """
    # One alpha per FIT, not one per record: the sigma sweep contributes many
    # records to a single fit, and counting it once per record would inflate n
    # by the number of scales.
    fits: dict[tuple[int, str, int, int], dict[str, float]] = {}
    for summary in summaries:
        order = int(summary["snapshot"]["order"])
        source = str(summary["action_source"])
        for fit in summary.get("scaling_fits", []):
            fits[(order, source, int(fit["horizon"]), int(fit["direction"]))] = fit

    # Mode label is a property of the (snapshot, source, horizon, direction)
    # fit; take it from the primary sigma so one fit gets one label.
    mode_by_key: dict[tuple[int, str, int, int], str] = {}
    for r in records:
        key = (r["snapshot"], r["action_source"], int(r["horizon"]), int(r["direction"]))
        if r["sigma"] == args.primary_sigma or key not in mode_by_key:
            mode_by_key[key] = r["mode"]

    table = {"agree_smooth": 0, "agree_boundary": 0,
             "contact_only": 0, "exponent_only": 0,
             "unassigned_no_fit": 0, "unassigned_low_r2": 0}
    alphas_by_mode: dict[str, list[float]] = defaultdict(list)
    for key, mode in mode_by_key.items():
        fit = fits.get(key)
        if fit is None or not np.isfinite(fit.get("alpha", np.nan)):
            table["unassigned_no_fit"] += 1
            continue
        r2 = fit.get("r_squared", float("nan"))
        if not np.isfinite(r2) or r2 < args.min_r2:
            # Amendment D: R^2 gates, it does not merely get reported.  A sweep
            # that crosses regimes is not described by one slope and gets no
            # alpha rather than a misleading one.
            table["unassigned_low_r2"] += 1
            continue
        alpha = float(fit["alpha"])
        alphas_by_mode[mode].append(alpha)
        exponent_boundary = alpha < args.smooth_alpha_low
        contact_boundary = mode == CROSS_MODE
        if contact_boundary and exponent_boundary:
            table["agree_boundary"] += 1
        elif not contact_boundary and not exponent_boundary:
            table["agree_smooth"] += 1
        elif contact_boundary:
            table["contact_only"] += 1
        else:
            table["exponent_only"] += 1
    return {
        "cross_tab": table,
        "n_fits_used": sum(len(v) for v in alphas_by_mode.values()),
        "alpha_median_by_mode": {
            mode: float(np.median(values)) for mode, values in alphas_by_mode.items()
        },
        "alpha_n_by_mode": {mode: len(v) for mode, v in alphas_by_mode.items()},
    }


def region_comparison(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Readout 9: offline vs planner-visited, restricted to same-mode samples."""
    out: dict[str, Any] = {}
    for source in sorted({r["action_source"] for r in records}):
        chosen = [r for r in records
                  if r["action_source"] == source and r["mode"] in SAME_MODE]
        out[source] = {m: ci_of(chosen, m, args) for m in ("e_k", "k_model_true", "k_true")}
    return out


def gates(records: list[dict[str, Any]], summaries: list[dict[str, Any]],
          args: argparse.Namespace) -> dict[str, Any]:
    """Pipeline gates that must pass before any scientific reading."""
    effector = ci_of(records, "k_true_state_effector", args)
    floors = [s["floor"]["repeat_floor"] for s in summaries]
    discard: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "discarded": 0})
    for s in summaries:
        for mode, counts in s.get("discard_by_mode", {}).items():
            discard[mode]["total"] += int(counts["total"])
            discard[mode]["discarded"] += int(counts["discarded"])
    rates = {
        mode: (c["discarded"] / c["total"] if c["total"] else float("nan"))
        for mode, c in discard.items()
    }
    finite = [v for v in rates.values() if np.isfinite(v)]
    spread = (max(finite) - min(finite)) if finite else float("nan")
    return {
        "effector_curvature": effector,
        "effector_gate_pass": bool(
            effector["point"] is not None and effector["point"] <= args.effector_gate
        ),
        "repeat_floor_max": float(np.max(floors)) if floors else float("nan"),
        "repeat_floor_all_zero": bool(np.allclose(floors, 0.0)) if floors else False,
        "discard_rate_by_mode": rates,
        "discard_rate_spread": spread,
    }


def paired_contrast(
    records: list[dict[str, Any]], args: argparse.Namespace,
    metric: str, source_a: str, source_b: str,
) -> dict[str, Any]:
    """Snapshot-paired difference in one preregistered cell (amendment E).

    Per snapshot: the median over directions within the cell for each source,
    then the difference, then a bootstrap over snapshots.  The earlier rule
    compared unpaired point estimates and took a max over CEM sources, which is
    a selection bias.
    """
    def cell(source: str) -> dict[int, float]:
        by_snapshot: dict[int, list[float]] = defaultdict(list)
        for r in records:
            if (r["action_source"] == source
                    and r["horizon"] == args.primary_horizon
                    and r["sigma"] == args.primary_sigma
                    and r["mode"] in SAME_MODE
                    and np.isfinite(r.get(metric, np.nan))):
                by_snapshot[r["snapshot"]].append(r[metric])
        return {s: float(np.median(v)) for s, v in by_snapshot.items() if v}

    a, b = cell(source_a), cell(source_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n_pairs": 0, "point": None, "low": None, "high": None,
                "metric": metric, "contrast": f"{source_a} - {source_b}"}
    diffs = np.array([a[s] - b[s] for s in shared], dtype=np.float64)
    result = bootstrap_ci(diffs, statistic=np.median, n_resamples=args.n_resamples,
                          seed=args.seed, groups=np.array(shared))
    return {"n_pairs": len(shared), "point": result.point, "low": result.low,
            "high": result.high, "metric": metric,
            "contrast": f"{source_a} - {source_b}",
            "excludes_zero": bool(result.low > 0.0 or result.high < 0.0)}


def primary_cell(records: list[dict[str, Any]], args: argparse.Namespace,
                 source: str) -> list[dict[str, Any]]:
    return [r for r in records
            if r["action_source"] == source
            and r["horizon"] == args.primary_horizon
            and r["sigma"] == args.primary_sigma
            and r["mode"] in SAME_MODE]


def verdict(records: list[dict[str, Any]], agreement: dict[str, Any],
            args: argparse.Namespace) -> dict[str, Any]:
    """Locked Stage-1 decision rule, evaluated on the preregistered cell only.

    Horizons, sigmas and sources are NOT pooled: normalized curvature depends on
    the perturbation scale even in a smooth regime, so a pooled median mixes
    quantities that are not comparable.
    """
    cell = primary_cell(records, args, args.reference_source)
    e_same = ci_of(cell, "e_k", args)
    e_abs = ci_of(cell, "e_k_absolute", args)
    e_j = ci_of(cell, "e_j", args)
    k_model = ci_of(cell, "k_model_true", args)
    k_true = ci_of(cell, "k_true", args)
    s_model = ci_of(cell, "s_model", args)
    s_true = ci_of(cell, "s_true", args)

    contrast = paired_contrast(records, args, "e_k",
                               args.primary_source, args.reference_source)
    secondary = paired_contrast(records, args, "e_k",
                                "cem_local", args.reference_source)

    mismatch_present = bool(e_same["low"] is not None
                            and e_same["low"] > args.mismatch_high)
    # Amendment J: the surrogate losses drive D2 Phi -> 0, which only matches
    # the measured estimand where the realized curvature is small.  A large E_K
    # with K_true >= K_model is the "model erases real nonlinearity" row, and
    # straightening it further would be the wrong intervention.
    model_over_true = bool(
        k_model["point"] is not None and k_true["point"] is not None
        and k_model["point"] > k_true["point"]
    )
    cem_exceeds = bool(contrast["point"] is not None
                       and contrast.get("excludes_zero") and contrast["point"] > 0)

    alpha_n = agreement.get("alpha_n_by_mode", {})
    alpha_median = agreement.get("alpha_median_by_mode", {})
    usable = [m for m in SAME_MODE if alpha_n.get(m, 0) > 0]
    if not usable:
        # The previous `all(... if mode in ...)` was vacuously true when no
        # same-mode alpha existed, which could green-light on no evidence.
        same_mode_smooth: bool | None = None
    else:
        same_mode_smooth = all(
            alpha_median[m] >= args.smooth_alpha_low for m in usable
        )

    action_deaf = bool(
        s_model["point"] is not None and s_true["point"] is not None
        and s_model["point"] < 0.2 * s_true["point"]
    )

    if action_deaf:
        decision = "KILL_ACTION_DEAF_MODEL"
    elif not mismatch_present and not cem_exceeds:
        decision = "KILL_NO_SAME_MODE_MISMATCH"
    elif mismatch_present and not model_over_true:
        decision = "MISMATCH_BUT_MODEL_UNDER_TRUE"
    elif same_mode_smooth is None:
        decision = "INCONCLUSIVE_NO_USABLE_ALPHA"
    elif mismatch_present and model_over_true and cem_exceeds and same_mode_smooth:
        decision = "GREEN_LIGHT_STRONG"
    elif mismatch_present and model_over_true:
        decision = "GREEN_LIGHT_WEAK"
    else:
        decision = "INCONCLUSIVE"

    return {
        "decision": decision,
        "cell": {"source": args.reference_source, "horizon": args.primary_horizon,
                 "sigma": args.primary_sigma, "strata": list(SAME_MODE)},
        "e_k": e_same, "e_k_absolute": e_abs, "e_j": e_j,
        "k_model_true": k_model, "k_true": k_true,
        "s_model": s_model, "s_true": s_true,
        "primary_contrast": contrast,
        "secondary_contrast_cem_local": secondary,
        "mismatch_present": mismatch_present,
        "model_curvature_exceeds_true": model_over_true,
        "same_mode_alpha_is_smooth": same_mode_smooth,
        "action_deaf": action_deaf,
    }


def self_test(args: argparse.Namespace) -> None:
    """Internal consistency checks on synthetic shards, run inside the smoke job."""
    rng = np.random.default_rng(0)
    records = []
    for snapshot in range(6):
      for source in ("dataset", "cem_fixed"):
        for direction in range(4):
            records.append({
                "snapshot": snapshot, "action_source": source, "horizon": 5,
                "direction": direction, "sigma": 0.1, "valid_unclipped": True,
                "mode": SAME_MODE_NON_CONTACT if direction % 2 else CROSS_MODE,
                "e_k": float(abs(rng.normal(0.3, 0.05))),
                "e_j": float(abs(rng.normal(0.05, 0.01))),
                "model_ratio": -0.5, "true_ratio": -0.2,
                "d2c_mismatch_normalized": 0.1,
                "g_k": float(rng.normal(0.1, 0.05)),
                "k_model_self": 0.3, "k_model_true": 0.3, "k_true": 0.1,
                "a_k": 0.5, "s_model": 1.0, "s_true": 1.0, "s_ratio": 1.0,
                "model_radial_fraction": 0.4, "model_angular_fraction": 0.6,
                "error_radial_fraction": 0.4, "error_angular_fraction": 0.6,
                "k_true_state_object": 0.1, "k_true_state_effector": 0.01,
                "delta_norm": 0.1, "d2_true_raw_norm": 0.01,
                "e_k_absolute": 0.5, "e_k_pathlen": 0.2,
            })
    summaries = [{
        "snapshot": {"order": s}, "action_source": src,
        "floor": {"repeat_floor": 0.0},
        "discard_by_mode": {SAME_MODE_NON_CONTACT: {"total": 10, "discarded": 1},
                            CROSS_MODE: {"total": 10, "discarded": 1}},
        "scaling_fits": [{"horizon": 5, "direction": d, "alpha": 2.0,
                          "r_squared": 1.0, "n_used": 4,
                          "excluded_below_floor": 0} for d in range(4)],
    } for s in range(6) for src in ("dataset", "cem_fixed")]

    args.n_resamples = 200
    table = stratified_table(records, args)
    assert set(table) == set(STRATA), table.keys()
    ci = table[SAME_MODE_NON_CONTACT]["e_k"]
    assert ci["n_snapshots"] == 6, ci
    assert ci["low"] <= ci["point"] <= ci["high"], ci
    tax = taxonomy(records, args)
    assert "spurious_model_curvature" in tax[SAME_MODE_NON_CONTACT], tax
    agreement = detector_agreement(records, summaries, args)
    # alpha = 2 everywhere, so every cross-mode row must be a contact-only call.
    assert agreement["cross_tab"]["exponent_only"] == 0, agreement
    assert agreement["cross_tab"]["contact_only"] > 0, agreement
    g = gates(records, summaries, args)
    assert g["effector_gate_pass"], g
    assert g["repeat_floor_all_zero"], g
    regions = region_comparison(records, args)
    gated, gate_info = apply_sensitivity_gate(records, args.min_sensitivity_quantile)
    assert gate_info["kept"] == len(records), gate_info
    contrast = paired_contrast(gated, args, "e_k", "cem_fixed", "dataset")
    assert contrast["n_pairs"] == 6, contrast
    v = verdict(gated, agreement, args)
    assert v["decision"] in {
        "KILL_ACTION_DEAF_MODEL", "KILL_NO_SAME_MODE_MISMATCH",
        "MISMATCH_BUT_MODEL_UNDER_TRUE", "INCONCLUSIVE_NO_USABLE_ALPHA",
        "GREEN_LIGHT_STRONG", "GREEN_LIGHT_WEAK", "INCONCLUSIVE",
    }, v
    # The vacuous-truth path must report unknown, never green-light on nothing.
    empty_alpha = dict(agreement, alpha_n_by_mode={}, alpha_median_by_mode={})
    assert verdict(gated, empty_alpha, args)["same_mode_alpha_is_smooth"] is None
    print("aggregate self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args)
        return

    all_records, summaries = load_shards(args.shard_root)
    unclipped = valid_only(all_records)
    records, sensitivity_gate = apply_sensitivity_gate(
        unclipped, args.min_sensitivity_quantile
    )
    agreement = detector_agreement(records, summaries, args)
    regions = region_comparison(records, args)

    report = {
        "n_records_total": len(all_records),
        "n_records_unclipped": len(unclipped),
        "n_records_after_sensitivity_gate": len(records),
        "sensitivity_gate": sensitivity_gate,
        "n_snapshots": len({r["snapshot"] for r in records}),
        "sources": sorted({r["action_source"] for r in records}),
        "thresholds": {
            "curvature_high": args.curvature_high,
            "mismatch_high": args.mismatch_high,
            "effector_gate": args.effector_gate,
            "smooth_alpha_low": args.smooth_alpha_low,
            "min_r2": args.min_r2,
            "min_sensitivity_quantile": args.min_sensitivity_quantile,
        },
        "primary_cell": {"horizon": args.primary_horizon, "sigma": args.primary_sigma,
                         "source": args.primary_source,
                         "reference": args.reference_source},
        "gates": gates(records, summaries, args),
        "stratified": stratified_table(records, args),
        "taxonomy": taxonomy(records, args),
        "horizon_curves": horizon_curves(records, args),
        "detector_agreement": agreement,
        "region_comparison": regions,
        "verdict": verdict(records, agreement, args),
        "code_hashes": sorted({s.get("code_sha256", "") for s in summaries}),
        "shards": [s["shard"] for s in summaries],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
