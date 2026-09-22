"""Small human-readable development report, executed inside the compute job."""
import argparse
import json
from pathlib import Path
from .runtime import require_slurm


def make_report(result):
    lines = ["# Prospective predictive-abstraction pilot", "",
        f"Status: **{result['status']}**; seed {result['seed']}.", "",
        "One-seed development result, 12 independent validation prefixes. "
        "No physical success or closed-loop planning measured. Scores are RGB-query "
        "scores in [0,1], NOT percentages of task success.", "",
        "## A-before-B prediction and selection", "",
        "Smaller is better for MAE/MSE and regret. Regret = oracle RGB score minus "
        "score of the selected candidate. Observed codecs read the real future; "
        "they are diagnostics, not deployable predictors.", ""]
    arms = ["constant", "query_only", "no_action", "observed_query_codec",
            "observed_generic_codec", "query_summary", "generic_summary",
            "frame", "endpoint", "direct", "query_summary_shuffled"]
    for h in ("32", "48", "64"):
        metric, selection = result["metrics"][h], result["selection"][h]
        bank = selection["bank"]
        gap = bank["mean_oracle_order_score"] - bank["mean_default_order_score"]
        lines += [f"### Horizon {h}" + (" (unseen in training)" if h=="64" else ""), "",
            f"Default {bank['mean_default_order_score']:.6f}; "
            f"oracle {bank['mean_oracle_order_score']:.6f}; headroom {gap:.6f}. "
            f"Informative prefixes: {bank['informative_prefixes']}/{bank['prefixes']}.", "",
            "| Arm | Train MSE | Validation MSE | Validation MAE | Selection regret |",
            "|---|---:|---:|---:|---:|"]
        for arm in arms:
            fit = result.get("train_fit_mse", {}).get(h, {}).get(arm)
            fit_text = "—" if fit is None else f"{fit[6]:.6f}"
            lines.append(f"| {arm} | {fit_text} | {metric[arm]['mse'][6]:.6f} | "
                         f"{metric[arm]['mae'][6]:.6f} | {selection[arm]['mean_order_regret']:.6f} |")
        lines += ["", "Paired prefix-bootstrap differences in regret "
                  "(baseline − query summary; positive favors summary):", ""]
        for arm, values in result["paired_selection_contrasts"][h].items():
            ci = values["paired_prefix_ci"]
            lines.append(f"- {arm}: {values['regret_reduction']:.6f}, "
                         f"95% interval [{ci[0]:.6f}, {ci[1]:.6f}].")
        lines.append("")
    lines += ["## Interpretation order", "",
        "1. Does the observed-future codec read the targets better than constant/query-only? "
        "Use train versus validation to distinguish fit from generalization.",
        "2. Does action-conditioned forecasting beat the trained no-action baseline, "
        "and does within-prefix action permutation degrade it?",
        "3. Does query-summary beat generic summary, ordered spatial frames and direct "
        "query prediction? All query families are in result.json, not only A-before-B.",
        "4. Does lower prediction error actually reduce within-bank regret? The tiny "
        "default-to-oracle gap constrains how much selection can improve here.", "",
        "No automated PASS/FAIL or paper claim. A profile is never an accuracy result. "
        "No continuation jobs are submitted by this report.", ""]
    return "\n".join(lines)


def main():
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    report = make_report(result)
    output = args.result.parent / "REPORT.md"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
