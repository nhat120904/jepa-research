"""Planning budget of each deployable PushT scorer: parameters, FLOPs and wall time per decision (compute node only).

Same modules and code paths as the closed loop (scripts/cta_diag_onpolicy.py, ti_wm/cta_batch.BatchScorer): one state,
a bank of K policy chunks, G goal images (the scores are averaged over goals as in deployment). Components timed with
CUDA synchronization, median over states x repeats:
  context: DINOv2/PCA tokens of the current and previous frame + action features, shared by CTA/ENDPOINT/DIRECT;
  <scorer>: world model (if any) + reader over K candidates x G goals;
  DINOWM: DINO-WM's own encoding of the current frame + rollout + latent cost (it does not use our tokens);
  policy: drawing the K chunks (common to every arm, reported once).
G in {1, 16} shows what an extra goal costs each scorer ("predict once, query many"). FLOPs from
torch.utils.flop_counter on one call (matmul/conv/attention ops only).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cta_diag_onpolicy import R4_NETS, load_nets, load_v2  # noqa: E402
from ti_wm.contract import candidate_seed, require_compute  # noqa: E402
from ti_wm.cta import action_features  # noqa: E402
from ti_wm.cta_batch import BatchScorer  # noqa: E402
from ti_wm.cta_runtime import Planner  # noqa: E402
from ti_wm.pusht_runtime import PolicyRunner, VisualScorer, reset_branch  # noqa: E402

SCORERS = ("CTAV2", "ENDV2", "DIRV2", "CTA4", "DINOWM")


def nparams(*mods):
    return int(sum(p.numel() for m in mods for p in m.parameters()))


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def timed(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(reps):
        t0 = sync()
        fn()
        out.append(1000 * (sync() - t0))
    return out


def flops(fn):
    try:
        with FlopCounterMode(display=False) as fc:
            fn()
        return int(fc.get_total_flops())
    except Exception as e:                       # report, do not fail the benchmark
        return f"unavailable: {type(e).__name__}: {e}"


@torch.inference_mode()
def main(a):
    require_compute()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    goal_frames = np.load(a.smoke / "goal_frames.npz")["frames"]
    runner = PolicyRunner(a.prep / "checkpoint", "cuda")
    visual = VisualScorer("cuda")
    planner = Planner(a.parent / "cta.pt", visual, goal_frames)
    extra = load_v2(a.v2 / "cta_v2.pt", device)
    r4, _, _ = load_nets(a.r4 / "round4.pt", R4_NETS, device)
    extra["CTA4"] = r4["CTA4"]
    from ti_wm.dinowm_scorer import DinoWMScorer
    render = reset_branch(a.first)
    dino = DinoWMScorer(a.dino_root, a.dinowm, device, goal_frames, render.env)
    render.env.close()
    extra["DINOWM"] = ("dinowm", dino)
    scorer = BatchScorer(planner, extra)

    m = dino.model
    params = {
        "shared_token_encoder (frozen DINOv2 + PCA, ours)": nparams(visual.model),
        "CTAV2 (WM + code reader)": nparams(*extra["CTAV2"][1]),
        "ENDV2 (endpoint WM + future reader)": nparams(*extra["ENDV2"][1]),
        "DIRV2 (direct scorer)": nparams(extra["DIRV2"][1]),
        "CTA4 (WM + parent code reader)": nparams(extra["CTA4"][1], planner.models["reader"]),
        "DINOWM predictor + action/proprio encoders": nparams(m.predictor, m.action_encoder, m.proprio_encoder),
        "DINOWM frozen encoder (DINOv2)": nparams(m.encoder),
        "policy (diffusion, common)": int(runner.parameters),
    }
    goals_all, zgoal_all = planner.goals, dino.z_goal
    states = [reset_branch(r) for r in range(a.first, a.first + a.states)]
    report = {"params": params, "states": [a.first, a.first + a.states - 1], "reps": a.reps, "results": {}}
    for k in a.ks:
        chunks = [np.asarray(runner.draw([s.hist] * k, [candidate_seed(a.first + i, 0, j) for j in range(k)]))
                  for i, s in enumerate(states)]
        report["results"].setdefault(f"K{k}", {})["policy_ms"] = statistics.median(
            [t for i, s in enumerate(states) for t in
             timed(lambda: runner.draw([s.hist] * k, [candidate_seed(a.first + i, 0, j) for j in range(k)]), 3, 1)])
        for g in a.goals:
            planner.goals = goals_all[:g]
            dino.z_goal = {kk: v[:g] for kk, v in zgoal_all.items()}
            cell, times = {}, {n: [] for n in ("context",) + SCORERS}
            for s, ch in zip(states, chunks):
                ctx_act = {}

                def context():
                    ctx, agent = scorer.context([s], k)
                    ctx_act["ctx"] = ctx
                    ctx_act["act"] = action_features(torch.as_tensor(ch[None], device=device), agent[:, None]).flatten(0, 1)

                times["context"] += timed(context, a.reps)
                for n in SCORERS:
                    if n == "DINOWM":
                        fn = lambda: dino.score([s.env], ch[None])
                    else:
                        def fn(n=n):
                            with planner.amp():
                                return scorer._one(n, ctx_act["ctx"], ctx_act["act"], None)
                    try:
                        times[n] += timed(fn, a.reps)
                    except torch.cuda.OutOfMemoryError:
                        times[n] = None
                        torch.cuda.empty_cache()
            s, ch = states[0], chunks[0]
            ctx, agent = scorer.context([s], k)
            act = action_features(torch.as_tensor(ch[None], device=device), agent[:, None]).flatten(0, 1)
            for n in ("context",) + SCORERS:
                if n == "context":
                    fl = flops(lambda: scorer.context([s], k))
                elif n == "DINOWM":
                    fl = flops(lambda: dino.score([s.env], ch[None]))
                else:
                    def call(n=n):
                        with planner.amp():
                            scorer._one(n, ctx, act, None)
                    fl = flops(call)
                t = times[n]
                cell[n] = {"ms_median": statistics.median(t) if t else "oom",
                           "ms_p90": float(np.percentile(t, 90)) if t else "oom", "flops": fl}
            report["results"][f"K{k}"][f"G{g}"] = cell
            print(f"K={k} G={g}", json.dumps(cell), flush=True)
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "budget.json").write_text(json.dumps(report, indent=1))
    for s in states:
        s.env.close()
    lines = ["| component | params |", "|---|---:|"] + [f"| {k} | {v / 1e6:.2f} M |" for k, v in params.items()]
    for kk, res in report["results"].items():
        lines += ["", f"### {kk} (policy sampling {res['policy_ms']:.1f} ms, common to all arms)", "",
                  "| scorer | G | ms/decision (median) | p90 | GFLOPs |", "|---|---:|---:|---:|---:|"]
        for gg, cell in res.items():
            if not gg.startswith("G"):
                continue
            for n, v in cell.items():
                ms = f"{v['ms_median']:.2f}" if isinstance(v["ms_median"], float) else v["ms_median"]
                p90 = f"{v['ms_p90']:.2f}" if isinstance(v["ms_p90"], float) else v["ms_p90"]
                fl = f"{v['flops'] / 1e9:.2f}" if isinstance(v["flops"], int) else "n/a"
                lines.append(f"| {n} | {gg[1:]} | {ms} | {p90} | {fl} |")
    (a.out / "budget.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("out", "parent", "v2", "r4", "dinowm", "prep", "smoke"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--dino-root", type=Path, default=Path("/mnt/data/nhatnc129/jepa/dino_wm_original"))
    p.add_argument("--first", type=int, default=2200)
    p.add_argument("--states", type=int, default=10)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--ks", type=int, nargs="+", default=[8, 16, 64])
    p.add_argument("--goals", type=int, nargs="+", default=[1, 16])
    main(p.parse_args())
