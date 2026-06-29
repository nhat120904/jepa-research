"""Phase-3 3a — is the readout precise OFF-POLICY? (the localised contact wall).

Test-1b (`scripts/21`) showed the spatial object probe localises the STATIC object to
92% <5cm on EXPERT held-out frames. Yet the `stateprobe` gate (the exact state-oracle
cost with probe readouts, perfect dynamics) gets push 2/16 vs the true-readout
state-oracle's 16/16. The reconciliation: the planner scores OFF-POLICY frames (latents
of random CEM action rollouts) where the probe was never trained.

This measures the probe decode error on exactly that off-policy distribution
(`scripts/_offpolicy_frames.collect_offpolicy_frames`) vs the 5/7 cm success radii —
the same table as scripts/21, so off-policy degradation is read directly against the
92% expert number.

  off-policy <5cm  COLLAPSES vs 92%  -> root cause confirmed; fix = off-policy-robust
                                        readout (scripts/22/19 --offpolicy-frac, then
                                        re-run scripts/30 --cost stateprobe).
  off-policy <5cm  stays high        -> readout is fine off-policy; the wall is the
                                        planner/cost-noise interaction, re-open that.

    python scripts/34_offpolicy_precision.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --probe checkpoints/spatial_object_probe_dino_wm_metaworld.pt \
        --target obj --out results/offpolicy_precision_spatial.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from models.probes import load_probe  # noqa: E402
from scripts._offpolicy_frames import collect_offpolicy_frames  # noqa: E402

PUSH_RADIUS, PICK_RADIUS = 0.05, 0.07


def _rows(err, tasks):
    def line(name, e):
        return {"group": name, "n": int(len(e)), "median_cm": 100 * float(np.median(e)),
                "mean_cm": 100 * float(e.mean()), "p90_cm": 100 * float(np.percentile(e, 90)),
                "frac_within_push_5cm": float((e < PUSH_RADIUS).mean()),
                "frac_within_pick_7cm": float((e < PICK_RADIUS).mean())}
    tks = np.asarray(tasks)
    rows = [line("ALL", err)]
    for t in sorted(set(tasks)):
        rows.append(line(t, err[tks == t]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--target", choices=["obj", "ee"], default="obj",
                    help="compare the probe output to the true object or end-effector")
    ap.add_argument("--tasks", nargs="+", default=["mw-push", "mw-pick-place"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--collect-every", type=int, default=4)
    ap.add_argument("--act-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="results/offpolicy_precision.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    probe, _ = load_probe(args.probe, device)
    print(f"probe: {args.probe} target={args.target}", flush=True)

    buf = collect_offpolicy_frames(adapter, device, tasks=args.tasks,
                                   episodes=args.episodes, max_steps=args.max_steps,
                                   collect_every=args.collect_every, seed=args.seed,
                                   act_std=args.act_std, batch=args.batch)
    target = buf["obj"] if args.target == "obj" else buf["ee"]
    z = buf["z"]
    errs = []
    with torch.no_grad():
        for lo in range(0, z.shape[0], args.batch):
            s = slice(lo, lo + args.batch)
            e = (probe(z[s].to(device)) - target[s].to(device)).norm(dim=-1)
            errs.append(e.cpu().numpy())
    err = np.concatenate(errs)
    rows = _rows(err, buf["task"])

    print(f"\nOFF-POLICY decode ({args.target}) — cf. Test-1b EXPERT spatial 92% <5cm:")
    print(f"{'group':16s} {'med':>6s} {'p90':>6s} {'<5cm':>6s} {'<7cm':>6s}")
    for r in rows:
        print(f"{r['group']:16s} {r['median_cm']:6.2f} {r['p90_cm']:6.2f} "
              f"{r['frac_within_push_5cm']*100:5.1f}% {r['frac_within_pick_7cm']*100:5.1f}%",
              flush=True)

    import pandas as pd
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
