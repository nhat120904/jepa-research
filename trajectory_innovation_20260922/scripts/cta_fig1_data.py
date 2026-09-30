"""Paper Fig. 1 assets: real PushT decisions rendered at the simulator's native 512 x 512 resolution, with the CTA
world model's predicted codes and scores. Compute node only (loads the policy, DINOv2, CTA networks and pymunk).

Pass 1 runs the policy-only arm (K = 8 policy samples, the paper's main bank) on --roots with the lockstep closed loop
of scripts/cta_diag_onpolicy.py, logging geometry labels and the scores of --log-scorers at every decision, and keeps
an exact clone of every pre-decision state.
Decisions are ranked by a rule that reads no learned score: the spread (max - min) of the geometry label across the
proposals, among decisions >= --min-decision in which no proposal ends the episode. The top --n decisions (at most one
per root) are rendered into <run>/r<root>_d<decision>/:
  prev.png, cur.png      native frames at t-1 and t (the two images of the context)
  cand<k>_s<j>.png       native frame after step j = 1..8 of proposal k
  decision.json          chunks (world units = native pixels, y down), agent/block trajectories, labels, scores,
                         expected predicted codes (CTA world model), source codes of the actual futures, and checks
                         against the pass-1 log and the observations the models saw.
<run>/goal<i>.png are the reader's goal images (the 96 x 96 observations it uses, upscaled 4x for display).
"""
import argparse
import hashlib
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
from ti_wm.cta import action_features, goal_scores  # noqa: E402
from ti_wm.cta_batch import K, BatchScorer, run_arm  # noqa: E402
from ti_wm.cta_geometry import registration_score  # noqa: E402
from ti_wm.cta_runtime import Planner, run_segment  # noqa: E402
from ti_wm.pusht_runtime import MAX_STEPS, CLONERS, Branch, PolicyRunner, VisualScorer, done, reset_branch  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def native_frame(env):
    """The 512 x 512 surface that gym-pusht resizes to the 96 x 96 observation; no action marker."""
    import pygame

    return np.ascontiguousarray(np.transpose(pygame.surfarray.array3d(env._draw()), (1, 0, 2)))


def save_png(path, img, scale=1):
    from PIL import Image

    im = Image.fromarray(np.asarray(img, np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
    im.save(path, optimize=True)


def pose(env):
    return [float(env.block.position[0]), float(env.block.position[1]), float(env.block.angle)]


class Stash:
    """record hook for run_arm: keeps an exact clone of every pre-decision state (arguments are not modified)."""

    def __init__(self, cloner):
        self.cloner, self.items = cloner, {}

    def __call__(self, root, d, state, chunks, scores, chosen):
        self.items[(root, d)] = {"env": self.cloner(state.env), "hist": list(state.hist), "t": int(state.t),
                                 "coverage": float(state.coverage), "max_coverage": float(state.max_coverage),
                                 "chunks": np.array(chunks, np.float32, copy=True),
                                 "scores": {n: np.array(v, np.float64, copy=True) for n, v in scores.items()},
                                 "chosen": int(chosen)}


@torch.inference_mode()
def render_decision(key, stash, cloner, planner, scorer, wm, log_row, out):
    root, d = key
    item, prev = stash.items[key], stash.items.get((root, d - 1))
    env0, chunks = item["env"], item["chunks"]
    out.mkdir(parents=True, exist_ok=True)
    checks = {"cur_obs_max_abs": float(np.abs(env0._render().astype(int) - item["hist"][-1]["pixels"].astype(int)).max())}
    save_png(out / "cur.png", native_frame(env0))
    if prev is not None:
        e = cloner(prev["env"])
        for action in prev["chunks"][prev["chosen"]][:-1]:          # t - 1 = after 7 of the previous 8 steps
            e.step(np.asarray(action, np.float32))
        checks["prev_obs_max_abs"] = float(np.abs(e._render().astype(int) - item["hist"][-2]["pixels"].astype(int)).max())
        save_png(out / "prev.png", native_frame(e))
        e.close()
    agent, block, labels = [], [], []
    for k in range(len(chunks)):
        e = cloner(env0)
        a_k, b_k = [], []
        for j, action in enumerate(chunks[k], start=1):
            e.step(np.asarray(action, np.float32))
            save_png(out / f"cand{k}_s{j}.png", native_frame(e))
            a_k.append([float(e.agent.position[0]), float(e.agent.position[1])])
            b_k.append(pose(e))
        agent.append(a_k)
        block.append(b_k)
        labels.append(float(registration_score(np.array(b_k[-1]))))
        e.close()
    checks["label_vs_log_max_abs"] = float(np.abs(np.array(labels) - log_row["geom"]).max())

    # model quantities through the evaluation pipeline (same calls as BatchScorer)
    state = Branch(env0, list(item["hist"]), item["t"], False, item["coverage"], item["max_coverage"])
    ctx, agent_pos = scorer.context([state], len(chunks))
    act = action_features(torch.as_tensor(chunks[None], device=planner.device), agent_pos[:, None]).flatten(0, 1)
    sims = [run_segment(state, chunks[k], cloner) for k in range(len(chunks))]
    branches, segs = [s[0] for s in sims], [s[1] for s in sims]
    checks["segment_vs_render_block_max_abs"] = float(max(
        np.abs(np.array(pose(b.env)) - np.array(block[k][-1])).max() for k, b in enumerate(branches)))
    fut = planner.future(branches, segs)
    enc, reader = planner.models["enc"], planner.models["reader"]
    with planner.amp():
        expected, _ = wm(ctx, act)
        source = enc(ctx, fut)
        s_cta = goal_scores(reader, ctx, expected, planner.goals).float().cpu().numpy()
        s_code = goal_scores(reader, ctx, source, planner.goals).float().cpu().numpy()
        s_full = goal_scores(planner.models["full"], ctx, {"end": fut["end"], "prop": fut["prop"]},
                             planner.goals).float().cpu().numpy()
    for b in branches:
        b.env.close()
    logged = {n[len("score_"):]: log_row[n].tolist() for n in log_row if n.startswith("score_")}
    if "CTA4" in logged:
        checks["cta_vs_log_max_abs"] = float(np.abs(s_cta - np.array(logged["CTA4"])).max())
    rec = {"root": root, "decision": d, "t": item["t"], "coverage_before": item["coverage"],
           "chunks": chunks.tolist(), "agent": agent, "block": block, "block_start": pose(env0),
           "agent_start": [float(env0.agent.position[0]), float(env0.agent.position[1])],
           "goal_pose": [float(x) for x in env0.goal_pose],
           "labels": labels, "coverage_end": log_row["cov"].tolist(),
           "scores": {"CTA": s_cta.tolist(), "CODE": s_code.tolist(), "FULL": s_full.tolist(), "logged": logged},
           "cta_choice": int(np.argmax(s_cta)), "oracle_choice": int(np.argmax(labels)),
           "cta_choice_label_rank": int((np.array(labels) > labels[int(np.argmax(s_cta))]).sum()) + 1,
           "expected_codes": expected.float().cpu().tolist(), "source_codes": source.float().cpu().tolist(),
           "source_indices": enc.fsq.codes_to_indices(source).cpu().tolist(), "checks": checks}
    write_json(out / "decision.json", rec)
    return rec


def main(a):
    require_compute()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    report = {"status": "RUNNING", "job": os.environ.get("SLURM_JOB_ID"), "roots": a.roots,
              "rule": "top label spread (max - min geometry label over the K proposals), decision >= "
                      f"{a.min_decision}, no proposal ends the episode, at most one decision per root; "
                      "no learned score is read by the rule",
              "checkpoints": {"parent": str(a.parent / "cta.pt"), "parent_sha256": sha256(a.parent / "cta.pt"),
                              "r4": str(a.r4 / "round4.pt"), "r4_sha256": sha256(a.r4 / "round4.pt"),
                              "world_model": a.cta}}
    write_json(a.run / "fig1_report.json", report)
    smoke = json.loads((a.smoke / "smoke.json").read_text())
    cloner = CLONERS[smoke["clone_method"]]
    runner = PolicyRunner(a.prep / "checkpoint", "cuda")
    goal_frames = np.load(a.smoke / "goal_frames.npz")["frames"]
    planner = Planner(a.parent / "cta.pt", VisualScorer("cuda"), goal_frames)
    extra, _, _ = diag.load_nets(a.r4 / "round4.pt", diag.R4_NETS, device)
    kind, wm = extra[a.cta]
    if kind != "parallel":
        raise ValueError(f"{a.cta} is not a parallel code world model")
    scorer = BatchScorer(planner, extra)
    for i, frame in enumerate(goal_frames):
        save_png(a.run / f"goal{i}.png", frame, scale=4)
    stash = Stash(cloner)
    episodes, log, timing = run_arm("P0", a.roots, runner, cloner, scorer, [a.cta], reset_branch, done, run_segment,
                                    a.max_decisions, record=stash)
    report.update(episodes=episodes, timing=timing, decisions=int(len(log["root"])))
    geom, cov = np.asarray(log["geom"]), np.asarray(log["cov"])
    spread = geom.max(1) - geom.min(1)
    eligible = (np.asarray(log["decision"]) >= a.min_decision) & (cov.max(1) < 0.95) \
        & (np.asarray(log["t"]) + 8 <= MAX_STEPS)
    order = [i for i in np.argsort(-spread) if eligible[i]]
    picked, seen = [], set()
    for i in order:
        r = int(log["root"][i])
        if r not in seen:
            picked.append(i)
            seen.add(r)
        if len(picked) == a.n:
            break
    report["spread"] = {"eligible": int(eligible.sum()), "quantiles": np.quantile(spread[eligible], [.5, .9, .99]).tolist()
                        if eligible.any() else None}
    report["rendered"] = []
    for i in picked:
        key = (int(log["root"][i]), int(log["decision"][i]))
        row = {k: np.asarray(v)[i] for k, v in log.items()}
        rec = render_decision(key, stash, cloner, planner, scorer, wm, row, a.run / f"r{key[0]}_d{key[1]}")
        report["rendered"].append({"root": key[0], "decision": key[1], "spread": float(spread[i]),
                                   "cta_choice": rec["cta_choice"], "oracle_choice": rec["oracle_choice"],
                                   "cta_choice_label_rank": rec["cta_choice_label_rank"], "checks": rec["checks"]})
        write_json(a.run / "fig1_report.json", report)
    report["status"] = "DONE"
    write_json(a.run / "fig1_report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("run", "parent", "r4", "prep", "smoke"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--roots", type=int, nargs="+", required=True)
    p.add_argument("--cta", default="CTA4", help="parallel code world model from round4.pt, read by the parent reader")
    p.add_argument("--max-decisions", type=int, default=20)
    p.add_argument("--min-decision", type=int, default=3)
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()
    require_compute()
    args.run.mkdir(parents=True, exist_ok=True)
    main(args)
