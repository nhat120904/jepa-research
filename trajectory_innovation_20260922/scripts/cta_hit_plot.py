"""Plot completed native-hit pilot success outcomes from CPU-generated JSON.

No models, trajectories or candidate arrays are read. Run through CPU sbatch;
the plotted intervals are each arm's Wilson intervals, not paired contrast CIs.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ti_wm.contract import require_compute


ORDER = ("P0", "GEOM8", "CTA_BASE", "CTA_CTRL", "CTA_HIT", "END_BASE", "END_CTRL", "END_HIT",
         "DIR_BASE", "DIR_CTRL", "DIR_HIT")
FAMILY = {"CTA": "cta", "END": "endpoint", "DIR": "direct"}
COLORS = {"BASE": "#58636e", "CTRL": "#2872b8", "HIT": "#d17b25"}
MARKERS = {"BASE": "o", "CTRL": "D", "HIT": "^"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(summary_path, fallback_path):
    summary = json.loads(Path(summary_path).read_text())
    fallback = json.loads(Path(fallback_path).read_text())
    if summary.get("schema") != "cta_hit_aggregate_v1":
        raise ValueError("Expected a completed native-hit aggregate summary")
    if fallback.get("schema") != "cta_hit_fallback_check_v1" or fallback.get("status") not in ("PASS", "NOT_APPLICABLE"):
        raise ValueError("Fallback equivalence is missing or failed")
    if summary["n"] != 20 or sorted(summary["roots"]) != sorted(fallback["roots"]):
        raise ValueError("Expected the completed paired 20-root pilot and matching fallback roots")
    if set(summary["source_shards"]) != set(fallback["source_shards"]):
        raise ValueError("Summary/fallback checks refer to different source shards")
    if set(summary["arms"]) != set(ORDER):
        raise ValueError("Expected all eleven acting arms; partial results cannot be plotted")
    experiment = summary["experiment"]
    if experiment["draw"] != 8 or experiment["n_exec"] != 15 or experiment["max_decisions"] is not None:
        raise ValueError("Expected full K8/L15 episodes")
    if experiment["proposal_mode"] != "canonical":
        raise ValueError("Expected canonical proposals")
    aliases = {item["alias"]: item for item in fallback["relationships"]}
    selected = {}
    for arm in ORDER:
        value = summary["arms"][arm]
        count, rate = value["successes"], value["success_rate"]
        interval = value["success_wilson95"]
        if type(count) is not int or not 0 <= count <= summary["n"] or not math.isclose(rate, count / summary["n"], abs_tol=1e-12):
            raise ValueError(f"Invalid success count/rate for {arm}")
        if len(interval) != 2 or not all(math.isfinite(x) for x in interval):
            raise ValueError(f"Invalid Wilson interval for {arm}")
        lo, hi = interval
        if not (-1e-12 <= lo <= rate + 1e-12 and rate - 1e-12 <= hi <= 1 + 1e-12):
            raise ValueError(f"Wilson interval excludes the arm's success rate: {arm}")
        if arm in ("P0", "GEOM8") or arm.endswith("_BASE"):
            continue
        prefix, label = arm.split("_", 1)
        key = f"{FAMILY[prefix]}_{'control' if label == 'CTRL' else 'hit'}"
        cfg = experiment["checkpoints"][label]["config"]["continuation"]
        selected[arm] = cfg["selected_steps"][key]
        if selected[arm] == 0:
            alias = aliases.get(arm)
            if not alias or alias["status"] != "PASS" or alias["baseline"] != f"{prefix}_BASE":
                raise ValueError(f"Step-zero arm lacks a matching passed fallback check: {arm}")
    seeds = {experiment["checkpoints"][label]["config"].get("seed") for label in ("BASE", "CTRL", "HIT")}
    if len(seeds) != 1 or None in seeds:
        raise ValueError("Plot's one-training-seed label requires a consistent saved seed")
    return summary, fallback, aliases, selected, next(iter(seeds))


def main(args):
    require_compute()
    summary, fallback, aliases, selected, seed = load_inputs(args.summary, args.fallback)
    args.out.mkdir(parents=True, exist_ok=False)
    # Agg is a file backend; rendering stays on this CPU compute node.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#aaaaaa"})
    ys, labels, families, previous, cursor = [], [], [], None, 0.
    for arm in ORDER:
        family = "anchors" if arm in ("P0", "GEOM8") else arm.split("_")[0]
        if previous is not None and family != previous:
            cursor += .65
        ys.append(cursor)
        families.append(family)
        cursor += 1.
        previous = family
        if arm == "GEOM8":
            label = "GEOM8 · privileged simulator oracle"
        elif arm in aliases:
            label = f"{arm} = {aliases[arm]['baseline']} (step 0)"
        elif arm in selected:
            label = f"{arm} (step {selected[arm]})"
        else:
            label = arm
        labels.append(label)
    fig, ax = plt.subplots(figsize=(11.5, 8.1))
    fig.subplots_adjust(left=.35, right=.85, top=.86, bottom=.24)
    for family in ("CTA", "DIR"):
        members = [y for y, name in zip(ys, families) if name == family]
        ax.axhspan(min(members) - .45, max(members) + .45, color="#f3f5f7", zorder=0)
    for arm, y in zip(ORDER, ys):
        value = summary["arms"][arm]
        percent = 100 * value["success_rate"]
        lo, hi = (100 * x for x in value["success_wilson95"])
        if arm == "GEOM8":
            color, marker, fill = "#a84242", "s", "white"
        elif arm == "P0":
            color, marker, fill = "#303b45", "s", "#303b45"
        else:
            role = arm.split("_")[1]
            color, marker = COLORS[role], MARKERS[role]
            fill = "white" if arm in aliases else color
        ax.errorbar(percent, y, xerr=[[max(0., percent - lo)], [max(0., hi - percent)]], fmt=marker,
                    color=color, markerfacecolor=fill, markeredgewidth=1.5, markersize=7,
                    elinewidth=1.5, capsize=3, zorder=3)
        ax.text(102, y, f"{value['successes']}/{summary['n']} · {percent:.0f}%", va="center", ha="left",
                fontsize=10, color="#303b45", clip_on=False)
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_ylim(max(ys) + .6, -.6)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Closed-loop success rate (%) · Wilson 95% intervals", labelpad=10)
    ax.xaxis.grid(True, color="#dce1e6", linewidth=.7, zorder=0)
    handles = [Line2D([0], [0], marker=MARKERS[role], color=COLORS[role], linewidth=0, markersize=7,
                      label=label) for role, label in (("BASE", "BASE"), ("CTRL", "Matched continuation CTRL"),
                                                     ("HIT", "Sampling + native-hit intervention HIT"))]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.05, .145), frameon=False,
               ncol=3, fontsize=9, borderaxespad=0)
    fig.suptitle("Closed-loop development pilot", x=.05, y=.975, ha="left", fontsize=16, fontweight="bold")
    fig.text(.05, .935, f"PushT · K=8 · L=15 · n=20 paired development roots · one training seed ({seed})",
             ha="left", fontsize=11, color="#4b5662")
    fig.text(.05, .105, "Hollow learned-model markers indicate selected-step0 aliases verified against BASE.\n"
             "GEOM8 uses simulated future states. It is a privileged diagnostic oracle.\n"
             "Intervals are marginal Wilson intervals; paired contrasts are reported in the aggregate.\n"
             "This development pilot is not a sealed evaluation or a multi-seed result.",
             ha="left", va="top", fontsize=9, color="#4b5662", linespacing=1.45)
    for extension in ("png", "svg"):
        fig.savefig(args.out / f"success_closed_loop.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    provenance = {"schema": "cta_hit_plot_v1", "job": os.environ["SLURM_JOB_ID"],
                  "summary": str(args.summary.resolve()), "summary_sha256": sha256(args.summary),
                  "fallback": str(args.fallback.resolve()), "fallback_sha256": sha256(args.fallback),
                  "fallback_status": fallback["status"], "arms": list(ORDER), "roots": summary["roots"],
                  "n": summary["n"], "training_seeds": [seed], "selected_steps": selected,
                  "fallback_aliases": {name: item["baseline"] for name, item in aliases.items()},
                  "statistics": "Saved marginal Wilson success intervals; no on-policy ranking metrics or new contrasts computed",
                  "matplotlib": matplotlib.__version__, "script_sha256": sha256(Path(__file__)),
                  "outputs": {name: sha256(args.out / name) for name in ("success_closed_loop.png", "success_closed_loop.svg")}}
    (args.out / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "fallback_aliases": provenance["fallback_aliases"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory for PNG, SVG and provenance")
    main(parser.parse_args())
