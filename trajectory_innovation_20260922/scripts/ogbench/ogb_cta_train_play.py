"""CTA trained on the official OGBench play dataset (1M transitions), evaluated on the branched policy banks.

Why: in an offline dataset every state has ONE logged action. A direct scorer D(C, A, g) must learn how actions change
progress from that alone; CTA factorizes the problem -- the world model p(S | C, A) learns what an action chunk does
to the future (dense target: the code of the actual future), and the reader D(C, S, g) learns progress from actual
futures, no action involved. This is the regime where the factorization can matter; branched sibling banks (ogb_cta_
train_v2.py) give every scorer direct action contrasts and favour the direct scorer.

Same modules, losses and arms as v2; only the data changes:
  decision i (any play step with a previous frame and 5 more steps in its episode): C = frame i (16x16) + frame i-1
  (8x8); A = actions i..i+4; tau = frames i+2..i+4 (8x8) and i+5 (16x16).
  Goals per step, shared by the batch: official task goals and hindsight goals (play frames with their cube position).
  Label of (i, goal): progress change over the chunk, prog(cube_{i+5}, g) - prog(cube_i, g), with prog the environment's
  progress (1 within 4 cm, else minus the cube-target distance). Ranking pairs: decisions of the batch, per goal.
Selection and evaluation: the branched held-out banks (the designated checkpoint-selection split), exactly as v2.
--heldout-goals excludes those official goals, and hindsight goals within --exclude-radius of them, from training.
The checkpoint has the v2 format (loads in ogb_cta_eval.py and as --init of ogb_cta_train_v2.py).
Compute node only.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import ogb_cta_train_v2 as v2  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.cta import saturation_penalty  # noqa: E402
from ti_wm.cta_ogb import zeros_prop  # noqa: E402
from ti_wm.cta_parallel import score_consistency, weighted_rank  # noqa: E402
from ti_wm.sibling import rank_loss  # noqa: E402

H = 5


class Play:
    def __init__(self, path, in_ram=False, max_frames=0):
        meta = np.load(path / "meta.npz")
        ep = meta["ep"]
        n = len(ep) if not max_frames else min(max_frames, len(ep))
        ep = ep[:n]
        self.actions = torch.from_numpy(meta["actions"][:n])
        self.cube = meta["cube"][:n].astype(np.float32)
        mm = lambda k: np.load(path / f"{k}.npy", mmap_mode="r")[:n]
        self.full, self.small = (np.ascontiguousarray(mm("full")), np.ascontiguousarray(mm("small"))) if in_ram \
            else (mm("full"), mm("small"))
        i = np.arange(1, n - H)
        ok = (ep[i - 1] == ep[i]) & (ep[i + H] == ep[i])
        self.valid = i[ok]
        moved = np.linalg.norm(self.cube[self.valid + H] - self.cube[self.valid], axis=-1) > 1e-3
        self.moving = self.valid[moved]
        self.n = n

    def batch(self, i, device):
        i = np.sort(np.asarray(i))
        f = lambda arr, j: torch.from_numpy(np.asarray(arr[j])).to(device)
        ctx = {"cur": f(self.full, i), "prev": f(self.small, i - 1), "prop": zeros_prop(len(i), device)}
        frames = torch.stack([f(self.full, i + k) for k in (2, 3, 4, 5)], 1)             # (B, 4, 256, D)
        fut = {"end": frames[:, 3], "seg": torch.stack([f(self.small, i + k) for k in (2, 3, 4)], 1),
               "prop": zeros_prop(len(i), device)}
        act = self.actions[torch.from_numpy(np.stack([i + k for k in range(H)], 1))].to(device)   # (B, 5, 5)
        return i, ctx, fut, frames, act


def sample(play, banks, rng, cfg, device):
    b, g = cfg["decisions"], cfg["goals"]
    n_mv = int(round(cfg["moving_frac"] * b))
    i = np.concatenate([rng.choice(play.moving, n_mv), rng.choice(play.valid, b - n_mv)])
    i, ctx, fut, frames, act = play.batch(i, device)
    toks, pos = [], np.zeros((g, 3), np.float32)
    for gi in range(g):
        if rng.random() < cfg["p_official"]:
            j = int(rng.choice(cfg["train_goals"]))
            toks.append(banks.goal_tok[j])
            pos[gi] = v2.GOAL_XYZ[j - 1]
        else:
            h = int(rng.integers(0, len(cfg["_hind"])))
            toks.append(torch.from_numpy(np.asarray(play.full[cfg["_hind"][h]])))
            pos[gi] = play.cube[cfg["_hind"][h]]
    goal = torch.stack(toks).to(device)                                                      # (G, 256, D)
    now, end = play.cube[i], play.cube[i + H]
    y = v2.progress_label(end[None], pos[:, None]) - v2.progress_label(now[None], pos[:, None])    # (G, B)
    return ctx, fut, frames, act, goal, torch.from_numpy(y).to(device)


def rep_goal(x, g):
    """Rows (b) -> (g, b): the batch repeated once per goal (goal-major)."""
    if isinstance(x, dict):
        return {k: rep_goal(v, g) for k, v in x.items()}
    return x.repeat(g, *([1] * (x.dim() - 1)))


def stage1(mods, play, banks, held, cfg, device, amp, rng, norms, log):
    params = [p for m in mods.values() for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = v2.schedule(opt, cfg["steps1"], cfg["warmup"])
    best = {grp: (-float("inf"), None, None) for grp in v2.GROUPS}
    g = cfg["goals"]
    for step in range(cfg["steps1"]):
        for m in mods.values():
            m.train()
        ctx, fut, frames, act, goal, y = sample(play, banks, rng, cfg, device)
        b = len(act)
        goal_r = goal.repeat_interleave(b, 0)
        with amp():
            code, pre = mods["enc"](ctx, fut, return_pre=True)
            end_hat, seg_hat = mods["dec"](ctx, code)
            l_rec = 0.5 * (((end_hat - fut["end"].float()) ** 2).mean() / norms["end"]
                           + ((seg_hat - fut["seg"].float()) ** 2).mean() / norms["seg"])
            ctx_r = rep_goal(ctx, g)
            s_code = mods["reader"](ctx_r, rep_goal(code, g), goal_r).float().view(g, b)
            s_full = mods["full"](ctx_r, {"end": rep_goal(fut["end"], g), "prop": rep_goal(fut["prop"], g)}, goal_r).float().view(g, b)
            s_dir = mods["direct"](ctx_r, rep_goal(act, g), goal_r).float().view(g, b)
        l_code, l_full, l_dir = rank_loss(s_code, y), rank_loss(s_full, y), rank_loss(s_dir, y)
        loss = l_code + cfg["rec"] * l_rec + cfg["sat"] * saturation_penalty(pre) + l_full + l_dir
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            idx = mods["enc"].fsq.codes_to_indices(code.detach()).cpu().numpy()
            log({"stage": 1, "step": step, "rank_code": float(l_code), "rec": float(l_rec), "rank_full": float(l_full),
                 "rank_direct": float(l_dir), "distinct_per_code": v2.code_distinct(idx),
                 "saturated_frac": float((pre.detach().abs() > 1.5).float().mean())})
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps1"]:
            for m in mods.values():
                m.eval()
            sc = v2.score_split(held, ("code", "full", "direct"), cfg["train_goals"], mods, {}, device, amp)
            rep = v2.gaps(held, sc, ("code", "full", "direct"), cfg["train_goals"])
            val = {grp: v2.mean_gap(rep, v2.GROUP_TIER[grp], cfg["train_goals"]) for grp in v2.GROUPS}
            for grp, v in val.items():
                if v > best[grp][0] or best[grp][2] is None:
                    best[grp] = (v, step + 1, {n: copy.deepcopy({k: t.detach().cpu() for k, t in mods[n].state_dict().items()})
                                                for n in v2.GROUPS[grp]})
            log({"stage": 1, "step": step + 1, "heldout_mean_gap": val})
    for grp, (_, _, states) in best.items():
        for n, st in states.items():
            mods[n].load_state_dict(st)
    return {grp: {"gap": best[grp][0], "step": best[grp][1]} for grp in v2.GROUPS}


def stage2(mods, wms, play, banks, held, cfg, device, amp, rng, norms, scale, log):
    opts = {n: torch.optim.AdamW(w.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) for n, w in wms.items()}
    scheds = {n: v2.schedule(o, cfg["steps2"], cfg["warmup"]) for n, o in opts.items()}
    best = {n: (-float("inf"), None, None) for n in wms}
    rank_cfg = (cfg["rank_scale"], cfg["rank_margin"])
    g = cfg["goals"]
    for step in range(cfg["steps2"]):
        ctx, fut, frames, act, goal, y = sample(play, banks, rng, cfg, device)
        b = len(act)
        goal_r, ctx_r = goal.repeat_interleave(b, 0), rep_goal(ctx, g)
        with torch.no_grad(), amp():
            src = mods["enc"](ctx, fut)
            t_code = mods["reader"](ctx_r, rep_goal(src, g), goal_r).float().view(g, b)
            t_full = mods["full"](ctx_r, {"end": rep_goal(fut["end"], g), "prop": rep_goal(fut["prop"], g)}, goal_r).float().view(g, b)
        rec = {"stage": 2, "step": step}
        for n, w in wms.items():
            w.train()
            with amp():
                if n == "cta":
                    expected, logits = w(ctx, act)
                    pred = w.nll(logits, src.float())
                    s = mods["reader"](ctx_r, rep_goal(expected, g), goal_r).float().view(g, b)
                    teacher = t_code
                else:
                    if n == "endpoint":
                        out = w(ctx, act)
                        pred = ((out["end"] - fut["end"].float()) ** 2).mean() / norms["end"]
                        end = out["end"]
                    else:
                        tf = w.rollout(ctx, act, teacher=frames)
                        pred = ((tf - frames.float()) ** 2).mean() / norms["frame"]
                        end = w.rollout(ctx, act)[:, 3]
                    s = mods["full"](ctx_r, {"end": rep_goal(end, g), "prop": rep_goal(zeros_prop(b, device), g)},
                                     goal_r).float().view(g, b)
                    teacher = t_full
                parts = {"pred": pred, "consistency": score_consistency(s, teacher, scale[v2.TEACHER[n]]),
                         "rank": weighted_rank(s, y, *rank_cfg)}
                loss = sum(parts.values())
            opts[n].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(w.parameters(), 1.0)
            opts[n].step()
            scheds[n].step()
            rec.update({f"{n}/{k}": float(v) for k, v in parts.items()})
        if step % 500 == 0:
            log(rec)
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps2"]:
            for w in wms.values():
                w.eval()
            sc = v2.score_split(held, tuple(wms), cfg["train_goals"], mods, wms, device, amp)
            rep = v2.gaps(held, sc, tuple(wms), cfg["train_goals"])
            val = {n: v2.mean_gap(rep, n, cfg["train_goals"]) for n in wms}
            for n, v in val.items():
                if v > best[n][0] or best[n][2] is None:
                    best[n] = (v, step + 1, copy.deepcopy({k: t.detach().cpu() for k, t in wms[n].state_dict().items()}))
            log({"stage": 2, "step": step + 1, "heldout_mean_gap": val})
    for n, w in wms.items():
        w.load_state_dict(best[n][2])
    return {n: {"gap": best[n][0], "step": best[n][1]} for n in wms}


def copy_norms(play, rng, n=256):
    i = np.sort(rng.choice(play.valid, n))
    cur = torch.from_numpy(np.asarray(play.full[i])).float()
    fr = torch.stack([torch.from_numpy(np.asarray(play.full[i + k])).float() for k in (2, 3, 4, 5)], 1)
    small = torch.from_numpy(np.asarray(play.small[i])).float()
    seg = torch.stack([torch.from_numpy(np.asarray(play.small[i + k])).float() for k in (2, 3, 4)], 1)
    return {"end": float(((fr[:, 3] - cur) ** 2).mean()), "seg": float(((seg - small[:, None]) ** 2).mean()),
            "frame": float(((fr - cur[:, None]) ** 2).mean())}


def main(a):
    require_compute()
    device = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    held_out = [int(x) for x in a.heldout_goals.split(",")] if a.heldout_goals else []
    train_goals = [j for j in range(1, 6) if j not in held_out]
    cfg = {"version": "play", "max_frames": a.max_frames, "m": 16, "direct_layers": 8, "lr": 3e-4, "wd": 0.05, "warmup": 500, "dropout": a.dropout,
           "steps1": a.steps1, "steps2": a.steps2, "decisions": a.decisions, "goals": a.goals, "moving_frac": .75,
           "p_official": a.p_official, "rec": 1.0, "sat": a.sat, "rank_scale": .01, "rank_margin": .001,
           "eval_every": a.eval_every, "seed": a.seed, "play": str(a.play), "banks": str(a.banks),
           "train_goals": train_goals, "heldout_goals": held_out, "exclude_radius": a.exclude_radius,
           "job": os.environ.get("SLURM_JOB_ID")}
    a.run.mkdir(parents=True, exist_ok=True)
    logf = (a.run / "metrics.jsonl").open("a")
    log = lambda d: (logf.write(json.dumps(d) + "\n"), logf.flush(), print(d, flush=True))
    t0 = time.perf_counter()
    play = Play(a.play, in_ram=a.in_ram, max_frames=a.max_frames)
    banks = v2.Data(a.banks / "train")            # only for the official goal images (no bank enters training)
    held = v2.Data(a.banks / "heldout")
    hind = rng.choice(play.valid, min(200_000, len(play.valid)), replace=False)
    if held_out:
        xy = v2.GOAL_XYZ[np.asarray(held_out) - 1, :2]
        near = (np.linalg.norm(play.cube[hind][:, None, :2] - xy[None], axis=-1) < a.exclude_radius).any(1)
        cfg["hindsight_excluded"] = int(near.sum())
        hind = hind[~near]
    cfg["_hind"] = hind
    norms = copy_norms(play, rng)
    cfg.update(norms=norms, play_valid=int(len(play.valid)), play_moving=int(len(play.moving)),
               heldout_decisions=held.n, load_seconds=time.perf_counter() - t0)
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (a.run / "config.json").write_text(json.dumps(public, indent=2))
    log({"config": public})
    mods = v2.build_stage1(cfg, device)
    t1 = time.perf_counter()
    cfg["selected_stage1"] = stage1(mods, play, banks, held, cfg, device, amp, rng, norms, log)
    cfg["stage1_hours"] = (time.perf_counter() - t1) / 3600
    for m in mods.values():
        m.eval().requires_grad_(False)
    spread = {"reader": [], "full": []}
    with torch.no_grad():
        for _ in range(8):
            ctx, fut, frames, act, goal, y = sample(play, banks, rng, cfg, device)
            b, g = len(act), cfg["goals"]
            with amp():
                t_code = mods["reader"](rep_goal(ctx, g), rep_goal(mods["enc"](ctx, fut), g), goal.repeat_interleave(b, 0))
                t_full = mods["full"](rep_goal(ctx, g), {"end": rep_goal(fut["end"], g), "prop": rep_goal(fut["prop"], g)},
                                      goal.repeat_interleave(b, 0))
            # within-"bank" spread in the bank sense is not available on play data: use the per-goal spread across the
            # batch, which upper-bounds it; the consistency term only needs a common normalizer per reader
            for n, t in (("reader", t_code), ("full", t_full)):
                t = t.float().view(g, b)
                spread[n].append(float((t - t.mean(1, keepdim=True)).square().mean()))
    scale = {n: max(float(np.sqrt(np.mean(v))), 1e-3) for n, v in spread.items()}
    cfg["consistency_scale"] = scale
    wms = v2.build_wms(cfg, device)
    t2 = time.perf_counter()
    cfg["selected_stage2"] = stage2(mods, wms, play, banks, held, cfg, device, amp, rng, norms, scale, log)
    cfg["stage2_hours"] = (time.perf_counter() - t2) / 3600
    for w in wms.values():
        w.eval().requires_grad_(False)
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    torch.save({"config": public, "stage1": {k: m.state_dict() for k, m in mods.items()},
                "wms": {n: w.state_dict() for n, w in wms.items()}, "consistency_scale": scale}, a.run / "cta_ogb.pt")
    tiers = ("full", "code", "direct", "cta", "endpoint", "frame")
    goal_names = ["own"] + list(range(1, 6))
    sc = v2.score_split(held, tiers, goal_names, mods, wms, device, amp)
    np.savez(a.run / "heldout_scores.npz", root=held.s.root, task=held.task, cubes=held.cubes,
             **{f"{t}__{gn}": v for (t, gn), v in sc.items()})
    rep = v2.gaps(held, sc, tiers, goal_names, ci=True)
    summary = {t: {"own": rep[t]["own"]["retained_gap"]["ratio"], "train_goals": v2.mean_gap(rep, t, train_goals),
                   **({"heldout_goals": v2.mean_gap(rep, t, held_out)} if held_out else {})} for t in tiers}
    if held_out:
        mask = np.isin(held.task, held_out)
        for t in tiers:
            m = v2.ranking_metrics(sc[(t, "own")][mask], held.s.label.numpy()[mask], held.s.root[mask], ci=True)
            m.pop("chosen")
            rep[t]["own_on_heldout_tasks"] = m
            summary[t]["own_on_heldout_tasks"] = m["retained_gap"]["ratio"]
    src_held = v2.source_codes(mods, held, device, amp)
    report = {"summary_retained_gap": summary, "ladder": rep, "codes": v2.code_report(mods, wms, held, src_held, device, amp),
              "config": public}
    (a.run / "offline_ladder.json").write_text(json.dumps(report, indent=2))
    (a.run / "config.json").write_text(json.dumps(public, indent=2))
    print(json.dumps({"summary": summary, "codes": report["codes"]}, indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--play", type=Path, required=True)
    p.add_argument("--banks", type=Path, required=True, help="branched encode dir (held-out split + goal images)")
    p.add_argument("--steps1", type=int, default=12000)
    p.add_argument("--steps2", type=int, default=8000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--decisions", type=int, default=64)
    p.add_argument("--goals", type=int, default=4)
    p.add_argument("--p-official", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sat", type=float, default=1.0)
    p.add_argument("--heldout-goals", default="")
    p.add_argument("--exclude-radius", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--in-ram", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="use the first N play frames (whole episodes of 1001)")
    main(p.parse_args())
