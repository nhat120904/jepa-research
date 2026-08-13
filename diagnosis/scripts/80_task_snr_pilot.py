"""Gate 1 (goal-marginalization design, 2026-08-11) — task-level SNR law.

For each episode, roll the scripted expert exactly as scripts/29/30 do, but
keep every intermediate raw 39-dim state (not just the final one). Classify
every transition with the existing regime proxy
(stratification.metaworld_regimes.classify_metaworld_regime) to find the last
pre-contact frame, then encode three frames with the real frozen encoder:
episode start, last pre-contact frame, and the expert's final/goal frame.

    SNR = ||z_goal - z_precontact|| / ||z_goal - z_start||

High SNR: most goal-latent displacement happens during/after contact -> the
plain L2 terminal cost is predicted to carry a usable signal. Low SNR: most
goal-latent displacement is arm transport that finishes before contact -> the
cost is front-loaded on something the planner can match without touching the
object -> predicted failure. See
docs/plans/2026-08-11-goal-marginalization-design.md for the full protocol,
task list, and decision rule. This script only produces the per-episode CSV;
scripts/81_analyze_task_snr.py does the correlation against the already-
measured l2-arm outcomes.

    python scripts/80_task_snr_pilot.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --tasks mw-push mw-reach --episodes 16 \
        --seed0 80000 --out results/task_snr_pilot.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util as _ilu
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from stratification.metaworld_regimes import classify_metaworld_regime  # noqa: E402


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_cl = _load("closed_loop_eval", "18_closed_loop_eval.py")
make_env = _cl.make_env
render = _cl.render
encode_frame = _cl.encode_frame
expert_policy = _cl.expert_policy


def roll_expert_full(env, init_obs: np.ndarray, task: str, max_steps: int):
    """Like scripts/18's rollout_expert, but keeps every state AND every
    rendered frame (not just the final one) so a post-hoc pre-contact index
    can pick out the right frame. Same convention: run the FULL max_steps,
    do not stop at first success."""
    pol = expert_policy(task)
    obs = init_obs
    states = [obs.copy()]
    frames = [render(env)]
    succ_step = None
    for t in range(1, max_steps + 1):
        obs, _, _, _, info = env.step(pol.get_action(obs))
        states.append(obs.copy())
        frames.append(render(env))
        if succ_step is None and info.get("success", 0) > 0.5:
            succ_step = t
    return states, frames, succ_step


def find_precontact_index(states: list[np.ndarray]) -> tuple[int, bool]:
    """Return (index into `states` of the last pre-contact frame,
    no_contact_detected). Transition i is states[i] -> states[i+1]."""
    regimes = [classify_metaworld_regime(states[i], states[i + 1])
               for i in range(len(states) - 1)]
    first_contact = next(
        (i for i, r in enumerate(regimes)
         if r in ("gripper_actuation", "contact_manipulation")), None)
    if first_contact is None:
        return max(0, len(states) - 2), True
    return first_contact, False


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="dino_wm_metaworld")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=80000)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--out", default="results/task_snr_pilot.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "seed", "t_precontact", "t_final", "no_contact_detected",
              "snr", "num_start_goal", "denom_start_goal", "expert_success_step"]
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for task in args.tasks:
            for e in range(args.episodes):
                seed = args.seed0 + e
                t0 = time.time()
                env, init_state = make_env(task, seed)
                states, frames, succ_step = roll_expert_full(
                    env, init_state, task, args.max_steps)
                env.close()
                t_pre, no_contact = find_precontact_index(states)
                t_fin = len(states) - 1

                z_start = encode_frame(adapter, frames[0], states[0][:4], device)
                z_pre = encode_frame(adapter, frames[t_pre], states[t_pre][:4], device)
                z_goal = encode_frame(adapter, frames[t_fin], states[t_fin][:4], device)

                num = float((z_goal.reshape(-1) - z_pre.reshape(-1)).norm())
                denom = float((z_goal.reshape(-1) - z_start.reshape(-1)).norm())
                snr = num / denom if denom > 1e-8 else float("nan")

                row = {"task": task, "seed": seed, "t_precontact": t_pre,
                       "t_final": t_fin, "no_contact_detected": int(no_contact),
                       "snr": snr, "num_start_goal": num, "denom_start_goal": denom,
                       "expert_success_step": succ_step}
                w.writerow(row); f.flush(); rows.append(row)
                print(f"  {task:16s} seed={seed} t*={t_pre:3d}/{t_fin:3d} "
                      f"no_contact={no_contact} snr={snr:.3f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n=== task SNR pilot: per-task mean ===")
    for task in args.tasks:
        tr = [r for r in rows if r["task"] == task]
        vals = [r["snr"] for r in tr if r["snr"] == r["snr"]]  # drop NaN
        nc = sum(r["no_contact_detected"] for r in tr)
        mean_snr = float(np.mean(vals)) if vals else float("nan")
        print(f"  {task:16s} mean_snr={mean_snr:.3f}  n={len(tr)}  no_contact={nc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
