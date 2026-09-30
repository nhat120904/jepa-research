"""One bounded continuation on CTA4-visited, policy-only PushT banks.

CTA, direct, and endpoint models see the same candidate banks and task labels.
All frozen target/readers retain the original geometry checkpoint.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cta_train as ct
from ti_wm.contract import require_compute
from ti_wm.cta import Scorer
from ti_wm.cta_parallel import EndpointWM, ParallelFSQWM, score_consistency, weighted_rank


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def build(a, device):
    parent = torch.load(a.parent / "cta.pt", map_location="cpu")
    old = torch.load(a.r4 / "round4.pt", map_location="cpu")
    r3 = torch.load(a.r3 / "round3.pt", map_location="cpu")
    cfg = parent["config"]
    frozen = ct.build(cfg, device)
    for key, net in frozen.items():
        net.load_state_dict(parent["state"][key], strict=True)
        net.eval().requires_grad_(False)
    nets = {"task": ParallelFSQWM(m=cfg["m"]),
            "direct": Scorer("action", layers=cfg["direct_layers"]),
            "frame": EndpointWM()}
    for key, state in (("task", old["networks"]["task"]),
                       ("direct", old["networks"]["direct"]),
                       ("frame", r3["networks"]["frame"])):
        nets[key].load_state_dict(state, strict=True)
        nets[key].to(device)
    return cfg, frozen, nets


def losses(nets, frozen, ctx, fut, act, labels, goals, scales, image_norm, pair_count):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        src = frozen["enc"](ctx, fut)
        code_teacher = frozen["reader"](ctx, src, goals).float().view(-1, ct.K)
        full_teacher = frozen["full"](ctx, fut, goals).float().view(-1, ct.K)
    rank = lambda s: weighted_rank(s, labels, .01, .001, pair_count=pair_count)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted, logits = nets["task"](ctx, act)
        code_scores = frozen["reader"](ctx, predicted, goals).float().view(-1, ct.K)
        task = nets["task"].nll(logits, src) + score_consistency(code_scores, code_teacher, scales["code"])
        task_rank = rank(code_scores)
        direct_scores = nets["direct"](ctx, act, goals).float().view(-1, ct.K)
        direct_rank = rank(direct_scores)
        endpoint = nets["frame"](ctx, act)
        image = F.mse_loss(endpoint["end"].float(), fut["end"].float()) / image_norm
        prop = F.mse_loss(endpoint["prop"].float(), fut["prop"].float()) / scales["prop"]
        frame_scores = frozen["full"](ctx, endpoint, goals).float().view(-1, ct.K)
        frame = (image + prop) / 2 + score_consistency(frame_scores, full_teacher, scales["full"])
        frame_rank = rank(frame_scores)
    return {"task": (task, task_rank), "direct": (direct_rank * 0, direct_rank),
            "frame": (frame, frame_rank)}


def calibrate(data, frozen, goals, device):
    rng = np.random.default_rng(381)
    values = {"code": [], "full": [], "prop": []}
    for _ in range(20):
        ids = torch.from_numpy(rng.integers(0, data.n, 4))
        ctx, fut = data.context(ids, device), data.future(ids, device)
        goal = ct.sample_goals(goals, rng, len(ids), device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            src = frozen["enc"](ctx, fut)
            for key, scorer, target in (("code", frozen["reader"], src),
                                         ("full", frozen["full"], fut)):
                score = scorer(ctx, target, goal).float().view(-1, ct.K)
                values[key].append(float((score - score.mean(1, keepdim=True)).square().mean()))
            values["prop"].append(float((fut["prop"] - ctx["prop"]).square().mean()))
    return {"code": max(float(np.sqrt(np.mean(values["code"]))), 1e-3),
            "full": max(float(np.sqrt(np.mean(values["full"]))), 1e-3),
            "prop": max(float(np.mean(values["prop"])), 1e-6)}


def main(a):
    require_compute()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cuda")
    cfg, frozen, nets = build(a, device)
    train = ct.Split(a.features / "train", a.limit)
    dev = ct.Split(a.features / "dev", max(8, a.limit // 4) if a.limit else None)
    if not (set(train.root).isdisjoint(dev.root) and set(dev.root).isdisjoint(range(2000, 2400))):
        raise ValueError("Train, selection and closed-loop roots must be disjoint")
    goals = torch.from_numpy(np.load(a.features / "goals.npy")).to(device)
    scales = calibrate(train, frozen, goals, device)
    image_norm = float(torch.load(a.parent / "cta.pt", map_location="cpu")["norms"]["end"])
    opts = {n: torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=cfg["wd"]) for n, net in nets.items()}
    selection = np.sort(np.random.default_rng(779).choice(dev.n, min(256, dev.n), replace=False))
    best = {n: {"loss": float("inf"), "step": -1, "state": None} for n in nets}
    report = {"status": "RUNNING", "train_banks": train.n, "dev_banks": dev.n,
              "selection_banks": len(selection), "steps": a.steps, "lr": a.lr, "scales": scales,
              "image_normalizer": image_norm, "history": []}
    save_json(a.out / "train_report.json", report)

    def evaluate(step):
        for net in nets.values():
            net.eval()
        sums = {n: 0. for n in nets}
        with torch.no_grad():
            for group in np.array_split(selection, max(1, int(np.ceil(len(selection) / 4)))):
                ids = torch.from_numpy(group)
                ctx, fut, act = dev.context(ids, device), dev.future(ids, device), dev.actions(ids, device)
                labels = dev.cov[ids].to(device)
                goal = ct.sample_goals(goals, np.random.default_rng(1000 + int(group[0])), len(ids), device)
                parts = losses(nets, frozen, ctx, fut, act, labels, goal, scales, image_norm, None)
                for name, (regular, ranking) in parts.items():
                    sums[name] += float(regular + ranking) * len(ids)
        metrics = {n: sums[n] / len(selection) for n in nets}
        for name in nets:
            if metrics[name] < best[name]["loss"]:
                best[name] = {"loss": metrics[name], "step": step,
                              "state": copy.deepcopy({k: v.detach().cpu() for k, v in nets[name].state_dict().items()})}
        report["history"].append({"step": step, "objective": metrics})
        save_json(a.out / "train_report.json", report)
        print(f"dev step={step} objective={metrics}", flush=True)
        for net in nets.values():
            net.train()

    evaluate(0)
    start = time.perf_counter()
    for step in range(1, a.steps + 1):
        ids_all = rng.integers(0, train.n, a.batch)
        goals_all = ct.sample_goals(goals, rng, a.batch, device)
        labels_all = train.cov[ids_all].to(device)
        pair_count = (labels_all[:, :, None] - labels_all[:, None, :] > .001).sum()
        for opt in opts.values():
            opt.zero_grad(set_to_none=True)
        for offset in range(0, a.batch, a.micro):
            ids = torch.as_tensor(ids_all[offset:offset + a.micro])
            ctx, fut, act = train.context(ids, device), train.future(ids, device), train.actions(ids, device)
            labels = train.cov[ids].to(device)
            goal = goals_all[offset * ct.K:(offset + len(ids)) * ct.K]
            parts = losses(nets, frozen, ctx, fut, act, labels, goal, scales, image_norm, pair_count)
            for name, (regular, ranking) in parts.items():
                loss = regular * (len(ids) / a.batch) + ranking
                if not torch.isfinite(loss):
                    raise FloatingPointError(name)
                loss.backward()
        for name, net in nets.items():
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opts[name].step()
        if step == 1 or step % 100 == 0:
            print(f"step {step}/{a.steps}, elapsed={time.perf_counter() - start:.1f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
    for name, net in nets.items():
        net.load_state_dict(best[name]["state"], strict=True)
    saved = {"config": {"m": cfg["m"], "direct_layers": cfg["direct_layers"],
                        "source": "CTA4 states, policy bank K=8", "selection": "own held-out objective"},
             "networks": {n: net.cpu().state_dict() for n, net in nets.items()},
             "selected_step": {n: best[n]["step"] for n in nets}}
    torch.save(saved, a.out / "onpolicy8.pt")
    report.update(status="DONE", selected_step=saved["selected_step"], seconds=time.perf_counter() - start)
    save_json(a.out / "train_report.json", report)
    print(f"TRAIN_OK {saved['selected_step']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("out", "parent", "r4", "r3", "features"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--limit", type=int, help="Smoke mode: truncate banks")
    args = p.parse_args()
    if args.batch % args.micro:
        raise ValueError("batch must be divisible by micro")
    main(args)
