"""Offline paper-hardening analysis — NO GPU, NO re-run needed.

Two reviewer-facing gaps, both answerable from artifacts already on disk:

  #1  Sim-state cross-check of the exploitation claim (Table `tab:exploit`).
      The published elite-decode number (24% <5cm) is the PROBE's accuracy on
      the CEM elites, so it conflates "object really far" with "probe fails
      OOD". `results/cem_exploit_lite.npz` already stores, per elite, the TRUE
      simulator object position (`obj`) and TRUE goal (`goal_obj`). We compute
      the true object->goal distance on the planner's committed (final-iter)
      elites directly from sim state — a verifier that never passes through the
      probe. If the object is far from goal here too, exploitation is confirmed
      on ground truth, closing the confound.

  #4  Per-arm confidence intervals for the oracle-ladder / mitigation table.
      Each arm's per-episode `success_end` is in `results/latent_oracle_*.csv`.
      Episodes are independent (paired env init + CEM noise per seed), so the
      exact binomial Wilson score interval is the right CI for k/n. We emit
      k/n + 95% Wilson CI per arm (mw-push), so "all inside 0-2/16" can be
      stated with intervals instead of bare point estimates.

Run (login node, CPU):
    .venv/bin/python scripts/45_simstate_and_cis.py
Writes results/exploit_simstate_crosscheck.csv and results/ladder_mitigation_cis.csv.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PUSH_RADIUS, PICK_RADIUS = 0.05, 0.07  # metres, matching scripts/35


# --------------------------------------------------------------------------- #
# #1  Sim-state cross-check of the exploitation claim
# --------------------------------------------------------------------------- #
def simstate_crosscheck() -> pd.DataFrame:
    npz_path = RES / "cem_exploit_lite.npz"
    d = np.load(npz_path, allow_pickle=True)
    obj, goal, it, task = d["obj"], d["goal_obj"], d["it"], d["task"].astype(str)
    true_gap = np.linalg.norm(obj - goal, axis=-1)  # TRUE object->goal, sim state
    last_it = int(it.max())

    def row(name, mask):
        g = true_gap[mask]
        if g.size == 0:
            return None
        return {
            "group": name, "n": int(g.size),
            "true_obj_goal_median_cm": round(100 * float(np.median(g)), 2),
            "true_obj_goal_p90_cm": round(100 * float(np.percentile(g, 90)), 2),
            "frac_true_within_5cm": round(float((g < PUSH_RADIUS).mean()), 4),
            "frac_true_within_7cm": round(float((g < PICK_RADIUS).mean()), 4),
        }

    rows = [row("ALL_elites", np.ones_like(it, bool)),
            row("FINAL_iter_elites", it == last_it)]
    for t in sorted(set(task)):
        rows.append(row(f"final_{t}", (it == last_it) & (task == t)))
    return pd.DataFrame([r for r in rows if r])


# --------------------------------------------------------------------------- #
# #4  Wilson score interval for each ladder / mitigation arm (mw-push)
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Exact-ish binomial Wilson score 95% interval. Deterministic in (k, n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def ladder_cis(task: str = "mw-push") -> pd.DataFrame:
    rows = []
    for csv in sorted(glob.glob(str(RES / "latent_oracle_*.csv"))):
        df = pd.read_csv(csv)
        if "success_end" not in df.columns or task not in set(df.get("task", [])):
            continue
        sub = df[df["task"] == task]
        n = len(sub)
        k = int(sub["success_end"].sum())
        lo, hi = wilson(k, n)
        rows.append({
            "arm": Path(csv).stem.replace("latent_oracle_", ""),
            "task": task, "k": k, "n": n,
            "success_pct": round(100 * k / n, 1) if n else float("nan"),
            "wilson_lo_pct": round(100 * lo, 1), "wilson_hi_pct": round(100 * hi, 1),
            "wilson_hi_k_of_n": round(hi * n, 2),  # upper bound expressed as #/n
        })
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)


def main() -> int:
    print("=" * 72)
    print("#1  SIM-STATE cross-check of exploitation (true object->goal, no probe)")
    print("=" * 72)
    cc = simstate_crosscheck()
    print(cc.to_string(index=False))
    cc.to_csv(RES / "exploit_simstate_crosscheck.csv", index=False)
    print(f"\n-> wrote {RES/'exploit_simstate_crosscheck.csv'}")
    print("Compare to PROBE-based numbers in results/cem_exploit_precision.csv "
          "(final-iter <5cm = 19.2%).")

    print("\n" + "=" * 72)
    print("#4  Wilson 95% CIs per ladder/mitigation arm (mw-push, success_end)")
    print("=" * 72)
    ci = ladder_cis("mw-push")
    print(ci.to_string(index=False))
    ci.to_csv(RES / "ladder_mitigation_cis.csv", index=False)
    print(f"\n-> wrote {RES/'ladder_mitigation_cis.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
