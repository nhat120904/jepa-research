"""PushT demo videos and figures for CTA (Round 5 deployment: 8 policy samples + 8 perturbed copies).

Replays chosen roots with the same deterministic batched closed loop as scripts/cta_diag_onpolicy.py and renders every
executed step at 384 x 384. Each frame overlays the decision's whole candidate bank as agent target paths:
policy samples in blue, perturbed copies in orange, line width by the acting scorer's rank, the executed chunk in
green (P0: candidate 0). Outputs per root: <arm>_root<r>.mp4, side_by_side_root<r>.mp4 (arms left to right) and
decision_root<r>_<arm>_d<d>.png stills (paper figure candidates). demo_report.json records whether every replay
reproduced the closed-loop outcome it was selected from.

Roots: --roots explicit, or --from-run <closed-loop run> picks CTA-win / both-success / CTA-loss cases.
Compute node only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cta_diag_onpolicy as diag  # noqa: E402
from cta_reader_refine import write_json  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.cta_batch import K, BatchScorer, base_name, mixed_bank, run_arm  # noqa: E402
from ti_wm.cta_runtime import Planner, run_segment  # noqa: E402
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer, done, reset_branch  # noqa: E402

WORLD = 512.
SIZE = 384
BLUE, ORANGE, GREEN, WHITE = (40, 110, 230), (240, 140, 30), (20, 170, 60), (255, 255, 255)


def pick_roots(run, arm, per_case):
    """CTA-only successes first (the result), then both-success and CTA-only failures (honest counter-examples)."""
    eps, _, _ = diag.load_shards(run)
    p0, cta = eps["P0"], eps[arm]
    roots = sorted(set(p0) & set(cta))
    cases = {"cta_only_success": [r for r in roots if cta[r]["success"] and not p0[r]["success"]],
             "both_success_cta_faster": sorted((r for r in roots if cta[r]["success"] and p0[r]["success"]
                                                and cta[r]["steps"] < p0[r]["steps"]),
                                               key=lambda r: cta[r]["steps"] - p0[r]["steps"]),
             "p0_only_success": [r for r in roots if p0[r]["success"] and not cta[r]["success"]]}
    chosen = {k: v[:per_case] for k, v in cases.items()}
    expected = {r: {"P0": p0[r]["success"], arm: cta[r]["success"]} for v in chosen.values() for r in v}
    return chosen, expected, {k: len(v) for k, v in cases.items()}


class Recorder:
    """record hook for run_arm: renders the executed chunk on an exact clone of the pre-decision state."""

    def __init__(self, cloner, arm, scorer_name, stills):
        self.cloner, self.arm, self.name, self.stills = cloner, arm, scorer_name, stills
        self.frames, self.pngs = {}, {}

    def __call__(self, root, d, state, chunks, scores, chosen):
        from PIL import Image, ImageDraw

        env = self.cloner(state.env)
        order = np.argsort(np.argsort(-np.asarray(scores[self.name]))) if self.name in scores else np.zeros(len(chunks))
        cov = state.coverage
        for step, action in enumerate(chunks[chosen], start=1):
            if state.success or state.t + step - 1 >= 300:
                break
            _, _, terminated, _, info = env.step(np.asarray(action, np.float32))
            img = Image.fromarray(np.ascontiguousarray(env.render())).resize((SIZE, SIZE))
            draw = ImageDraw.Draw(img, "RGBA")
            for k, path in enumerate(chunks):
                if k == chosen:
                    continue
                pts = [tuple(p * SIZE / WORLD) for p in path]
                color = (*(BLUE if k < K else ORANGE), 150)
                draw.line(pts, fill=color, width=max(1, 4 - int(order[k]) // 4))
            draw.line([tuple(p * SIZE / WORLD) for p in chunks[chosen]], fill=(*GREEN, 255), width=5)
            cov = float(info["coverage"])
            draw.rectangle([0, 0, SIZE, 20], fill=(0, 0, 0, 150))
            kind = "policy" if chosen < K else "perturbed"
            draw.text((4, 4), f"{self.arm}  t={state.t + step}  coverage={cov:.2f}  chose #{chosen} ({kind})",
                      fill=WHITE)
            frame = np.asarray(img.convert("RGB"))
            self.frames.setdefault(root, []).append(frame)
            if step == 1 and d in self.stills:
                self.pngs[(root, d)] = frame
            if terminated:
                break
        env.close()


def write_video(path, frames, fps=10):
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as w:
        for f in frames:
            w.append_data(f)


def main(a):
    require_compute()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    smoke = json.loads((a.smoke / "smoke.json").read_text())
    cloner = CLONERS[smoke["clone_method"]]
    runner = PolicyRunner(a.prep / "checkpoint", "cuda")
    planner = Planner(a.parent / "cta.pt", VisualScorer("cuda"), np.load(a.smoke / "goal_frames.npz")["frames"])
    extra = {}
    if a.r3:
        more, _, _ = diag.load_nets(a.r3 / "round3.pt", diag.R3_NETS, device)
        extra.update(more)
    more, _, _ = diag.load_nets(a.r4 / "round4.pt", diag.R4_NETS, device)
    extra.update(more)
    if a.r6:
        extra.update(diag.load_round6(a.r6 / "round6.pt", device))
    scorer = BatchScorer(planner, extra)
    arms, names = a.arms.split(","), a.log_scorers.split(",")
    acting = arms[-1]
    if a.from_run:
        cases, expected, counts = pick_roots(a.from_run, acting, a.per_case)
        roots = sorted({r for v in cases.values() for r in v})
    else:
        cases, expected, counts, roots = {"explicit": a.roots}, {}, {}, a.roots
    report = {"status": "RUNNING", "job": os.environ.get("SLURM_JOB_ID"), "cases": cases, "case_counts": counts,
              "arms": arms, "roots": roots, "replay": {}}
    write_json(a.run / "demo_report.json", report)
    if not roots:
        report["status"] = "NO_ROOTS"
        write_json(a.run / "demo_report.json", report)
        return
    frames = {}
    for arm in arms:
        rec = Recorder(cloner, arm, base_name(arm) if arm != "P0" else acting, set(a.stills))
        episodes, _, _ = run_arm(arm, roots, runner, cloner, scorer, names, reset_branch, done, run_segment,
                                 a.max_decisions, bank=mixed_bank if a.bank == "mixed" else None, record=rec)
        for e in episodes:
            r = e["root"]
            want = expected.get(r, {}).get(arm)
            report["replay"].setdefault(str(r), {})[arm] = {"success": e["success"], "steps": e["steps"],
                                                            "matches_closed_loop": None if want is None else want == e["success"]}
            write_video(a.run / f"{arm}_root{r}.mp4", rec.frames.get(r, []))
            frames[(arm, r)] = rec.frames.get(r, [])
        for (r, d), img in rec.pngs.items():
            from PIL import Image
            Image.fromarray(img).save(a.run / f"decision_root{r}_{arm}_d{d}.png")
        write_json(a.run / "demo_report.json", report)
    for r in roots:
        seqs = [frames[(arm, r)] for arm in arms]
        n = max(len(s) for s in seqs)
        if n == 0:
            continue
        pad = [s + [s[-1]] * (n - len(s)) if s else [np.zeros((SIZE, SIZE, 3), np.uint8)] * n for s in seqs]
        write_video(a.run / f"side_by_side_root{r}.mp4", [np.concatenate(f, 1) for f in zip(*pad)])
    report["status"] = "DONE"
    write_json(a.run / "demo_report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("run", "parent", "r3", "r4", "r6", "prep", "smoke", "from-run"):
        p.add_argument(f"--{key}", type=Path, required=key in ("run", "parent", "r4", "prep", "smoke"))
    p.add_argument("--roots", type=int, nargs="*", default=[])
    p.add_argument("--per-case", type=int, default=3)
    p.add_argument("--arms", default="P0,CTA4", help="last arm is the acting CTA arm used for selection and overlay")
    p.add_argument("--log-scorers", default="CTA4")
    p.add_argument("--bank", choices=("policy", "mixed"), default="mixed")
    p.add_argument("--stills", type=int, nargs="*", default=[0, 5, 10])
    p.add_argument("--max-decisions", type=int, default=None, help="smoke only")
    args = p.parse_args()
    require_compute()
    args.run.mkdir(parents=True, exist_ok=True)
    main(args)
