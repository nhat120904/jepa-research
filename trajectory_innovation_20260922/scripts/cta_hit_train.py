"""Bounded L=15 continuation with native-hit supervision and matched baselines.

The source encoder, code reader and FULL reader remain frozen. Actual futures
and termination flags are training/selection targets only. Checkpoints retain
the v2 deployment schema, whose CTA reader never receives proposed actions.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cta_train_v2 import Bank, GOALS, K, Pool, build_stage1, build_wms, schedule
from ti_wm.contract import require_compute
from ti_wm.cta import goal_scores
from ti_wm.cta_eval import ranking_metrics
from ti_wm.cta_hit_training import HitBank, StratifiedHitPool, native_hit_loss, native_hit_metrics
from ti_wm.cta_hit_contracts import validate_bank_shapes, validate_train_arguments
from ti_wm.cta_parallel import score_consistency, weighted_rank
from ti_wm.sibling import rank_loss


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def backend_config():
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "cudnn": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "training_precision": "bfloat16 autocast; float32 losses"}


def old_batch(pool, banks, rng, decisions, device):
    ctx, fut, act, labels, rows = pool.sample(rng, decisions, .75, device, return_rows=True)
    hits = torch.cat([torch.as_tensor(banks[j].hits[rows[rows[:, 0] == j, 1]], device=device)
                      for j in np.unique(rows[:, 0])])
    return ctx, fut, act, labels, hits, rows


def choose_goals(goals, rng, count, device):
    indices = torch.as_tensor(rng.integers(0, len(goals), count), device=device)
    return goals[indices].repeat_interleave(K, 0)


def teacher_scores(mods, ctx, future, goal, amp):
    # no_grad, not inference_mode: these targets participate in autograd losses.
    with torch.no_grad(), amp():
        source = mods["enc"](ctx, future)
        code = mods["reader"](ctx, source, goal).float().view(-1, K)
        full = mods["full"](ctx, {"end": future["end"], "prop": future["prop"]}, goal).float().view(-1, K)
    return source, code, full


def forward_loss(name, model, mods, batch, goal, teachers, cfg, temperatures, amp):
    ctx, future, actions, labels, hits = batch[:5]
    source, t_code, t_full = teachers
    with amp():
        if name.startswith("cta"):
            expected, logits = model(ctx, actions)
            scores = mods["reader"](ctx, expected, goal).float().view(-1, K)
            parts = {"pred": model.nll(logits, source.float()),
                     "consistency": score_consistency(scores, t_code, cfg["consistency_scale"]["reader"])}
        elif name.startswith("endpoint"):
            predicted = model(ctx, actions)
            scores = mods["full"](ctx, predicted, goal).float().view(-1, K)
            parts = {"pred": ((predicted["end"] - future["end"].float()).square().mean() / cfg["norms"]["end"]
                              + (predicted["prop"] - future["prop"].float()).square().mean() / cfg["norms"]["prop"]),
                     "consistency": score_consistency(scores, t_full, cfg["consistency_scale"]["full"])}
        else:
            scores = model(ctx, actions, goal).float().view(-1, K)
            parts = {}
        # DIRECT continuation retains its original stage-1 dense objective;
        # the WM pairs retain the original stage-2 weighted ranking objective.
        parts["rank"] = (rank_loss(scores, labels) if name.startswith("direct")
                         else weighted_rank(scores, labels, cfg["rank_scale"], cfg["rank_margin"]))
        if name.endswith("_hit"):
            parts["hit"] = cfg["continuation"]["hit_weight"] * native_hit_loss(scores, hits, temperatures[name])
        loss = sum(parts.values())
    return loss, parts, scores


@torch.inference_mode()
def select_scores(banks, models, mods, goals, device, amp, limit=None):
    result = {name: [] for name in models}
    for bank in banks:
        n = bank.n if limit is None else min(limit, bank.n)
        scores = {name: np.empty((n, K), np.float32) for name in models}
        # Fixed one-bank batch, same full 16-goal reader contract as deployment.
        for row in range(n):
            ctx, _, actions, _ = bank.batch([row], device)
            with amp():
                for name, model in models.items():
                    if name.startswith("cta"):
                        value = goal_scores(mods["reader"], ctx, model(ctx, actions)[0], goals)
                    elif name.startswith("endpoint"):
                        value = goal_scores(mods["full"], ctx, model(ctx, actions), goals)
                    else:
                        value = goal_scores(model, ctx, actions, goals)
                    scores[name][row] = value.float().cpu().numpy().reshape(K)
        for name in models:
            result[name].append(scores[name])
    return result


def selection_report(banks, scores, limit):
    n = lambda bank: bank.n if limit is None else min(limit, bank.n)
    labels = np.concatenate([bank.geom[:n(bank)] for bank in banks])
    roots = np.concatenate([bank.root[:n(bank)] for bank in banks])
    hits = np.concatenate([bank.hits[:n(bank)] for bank in banks])
    report = {}
    for name, arrays in scores.items():
        prediction = np.concatenate(arrays)
        ranking = ranking_metrics(prediction, labels, roots, ci=False)
        ranking.pop("chosen")
        native = native_hit_metrics(prediction, hits)
        gap = float(ranking["retained_gap"]["ratio"])
        capture = native["mixed_capture"]
        # Fixed before closed loop: dense performance plus bounded native capture.
        criterion = gap + .25 * (capture if capture is not None else 0.)
        report[name] = {"geometry": ranking, "native": native, "selection_criterion": criterion,
                        "criterion_definition": "retained_geometry_gap + 0.25 * mixed_hit_capture"}
    return report


def save_v2(path, cfg, base_stage1, base_wms, model_states, selected_steps, intervention):
    stage1, wms = dict(base_stage1), dict(base_wms)
    if intervention:
        stage1["direct"] = model_states["direct_hit"]
        wms["cta"], wms["endpoint"] = model_states["cta_hit"], model_states["endpoint_hit"]
    else:
        stage1["direct"] = model_states["direct_control"]
        wms["cta"], wms["endpoint"] = model_states["cta_control"], model_states["endpoint_control"]
    own_cfg = copy.deepcopy(cfg)
    own_cfg["continuation"]["intervention"] = intervention
    own_cfg["continuation"]["selected_steps"] = selected_steps
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"config": own_cfg, "stage1": stage1, "wms": wms}, temporary)
    temporary.replace(path)


def main(args):
    require_compute()
    validate_train_arguments(args)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    started = time.perf_counter()
    blob = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(blob["config"])
    if cfg.get("chunk") != 15 or cfg.get("hindsight", 0.) != 0 or cfg.get("success_bonus", 0.) != 0:
        raise ValueError("Pilot requires the original L15 recipe without altered goal/geometry labels")
    banks = [HitBank(Bank(Path(path), name), Path(path), standard=(name == "r4std"))
             for name, path in cfg["sources"].items()]
    selection = [HitBank(Bank(Path(path), name), Path(path), standard=True)
                 for name, path in cfg["selection"].items()]
    for bank in banks + selection:
        validate_bank_shapes(bank.name, bank.geom.shape, bank.hits.shape, bank.chunk.shape, K, cfg["chunk"],
                             {key: value.shape for key, value in bank.mm.items()})
    if set(cfg["sources"]) != {"r4std", "r4pert"}:
        raise ValueError("Expected only L15 standard/perturbed training splits")
    if args.train_limit:
        for bank in banks:
            bank.base.n = min(bank.n, args.train_limit)
            bank.base.spread = bank.spread[bank.spread < bank.n]
    pool = StratifiedHitPool(banks, args.standard_mass, args.mixed_frac, .75)
    old_pool = Pool(banks)
    disjoint = pool.assert_disjoint(selection, forbidden_roots=range(2100, 34000))
    # The reserved interval includes development and sealed roots; selection 2000-2099 is permitted.
    goals = torch.from_numpy(np.load(GOALS)).to(device)
    if len(goals) != 16:
        raise ValueError("Expected the original 16 goal/background images")
    mods = build_stage1(cfg, device)
    for name, model in mods.items():
        model.load_state_dict(blob["stage1"][name], strict=True)
        model.eval().requires_grad_(False)
    base_wms = build_wms(cfg, device)
    for name, model in base_wms.items():
        model.load_state_dict(blob["wms"][name], strict=True)
    models = {"cta_control": base_wms["cta"], "cta_hit": copy.deepcopy(base_wms["cta"]),
              "endpoint_control": base_wms["endpoint"], "endpoint_hit": copy.deepcopy(base_wms["endpoint"]),
              "direct_control": copy.deepcopy(mods["direct"]), "direct_hit": copy.deepcopy(mods["direct"])}
    for model in models.values():
        model.requires_grad_(True).eval()
    temperatures = {"cta_hit": cfg["consistency_scale"]["reader"],
                    "endpoint_hit": cfg["consistency_scale"]["full"]}
    # Fix DIRECT's score scale on train-only draws before adaptation. No selection labels are used.
    calibration_rng = np.random.default_rng(args.seed + 811)
    variance = []
    with torch.no_grad():
        for _ in range(8):
            batch = old_batch(old_pool, banks, calibration_rng, args.decisions, device)
            goal = choose_goals(goals, calibration_rng, len(batch[3]), device)
            with amp():
                value = models["direct_hit"](batch[0], batch[2], goal).float().view(-1, K)
            variance.append(float((value - value.mean(1, keepdim=True)).square().mean()))
    temperatures["direct_hit"] = max(float(np.sqrt(np.mean(variance))), 1e-3)
    cfg["continuation"] = {"job": os.environ["SLURM_JOB_ID"], "base": str(args.base),
                           "base_sha256": sha256(args.base), "seed": args.seed, "steps": args.steps,
                           "lr": args.lr, "warmup": args.warmup, "decisions": args.decisions,
                           "hit_weight": args.hit_weight, "hit_temperatures": temperatures,
                           "sampling": pool.report(args.decisions), "root_split": disjoint,
                           "selection_limit": args.selection_limit, "selection_goal_images": 16,
                           "selection_batch": 1, "selection_rule": "retained_geometry_gap + 0.25 * mixed_hit_capture",
                           "source_reader_frozen": True, "independent_optimizers_and_clip": True,
                           "control_models": ["cta", "endpoint", "direct"],
                           "dense_objectives": {"cta": "weighted_rank", "endpoint": "weighted_rank",
                                                "direct": "original_stage1_rank_loss"},
                           "comparison_scope": "combined sampling/native-hit intervention; same updates and selection rule",
                           "bank_shapes": {bank.name: {"actions": list(bank.chunk.shape),
                                                        "native_hit": list(bank.hits.shape)} for bank in banks + selection},
                           "backend": backend_config()}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    metrics_file = (args.out / "metrics.jsonl").open("a")
    def log(record):
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        print(json.dumps(record), flush=True)
    log({"config": cfg})
    opts = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["wd"])
            for name, model in models.items()}
    schedules = {name: schedule(opt, args.steps, args.warmup) for name, opt in opts.items()}
    rng_old, rng_hit = np.random.default_rng(args.seed), np.random.default_rng(args.seed)
    best, selected, selected_steps = {}, {}, {}
    def cpu_states():
        return {name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                for name, model in models.items()}
    def evaluate(step):
        for model in models.values():
            model.eval()
        scores = select_scores(selection, models, mods, goals, device, amp, args.selection_limit)
        report = selection_report(selection, scores, args.selection_limit)
        for name, value in report.items():
            criterion = value["selection_criterion"]
            if not math.isfinite(criterion):
                raise RuntimeError(f"Nonfinite selection criterion for {name}")
            if name not in best or criterion > best[name]:
                best[name], selected_steps[name] = criterion, step
                selected[name] = {key: value.detach().cpu().clone() for key, value in models[name].state_dict().items()}
        log({"stage": "selection", "step": step, "metrics": report, "selected_steps": selected_steps.copy()})
        np.savez(args.out / f"selection_scores_{step}.npz",
                 **{f"{name}__{bank.name}": scores[name][j] for name in models for j, bank in enumerate(selection)})
        save_v2(args.out / "control/cta_v2.pt", cfg, blob["stage1"], blob["wms"], selected, selected_steps, False)
        save_v2(args.out / "hit/cta_v2.pt", cfg, blob["stage1"], blob["wms"], selected, selected_steps, True)
        (args.out / "selection.json").write_text(json.dumps({"step": step, "metrics": report,
                                                              "selected_steps": selected_steps}, indent=2) + "\n")
    evaluate(0)
    exposure = {group: {bank.name: {"banks": 0, "mixed": 0} for bank in banks} for group in ("control", "hit")}
    for step in range(1, args.steps + 1):
        control = old_batch(old_pool, banks, rng_old, args.decisions, device)
        intervention = pool.sample(rng_hit, args.decisions, device, return_rows=True)
        control_goal = choose_goals(goals, rng_old, len(control[3]), device)
        hit_goal = choose_goals(goals, rng_hit, len(intervention[3]), device)
        control_teacher = teacher_scores(mods, control[0], control[1], control_goal, amp)
        hit_teacher = teacher_scores(mods, intervention[0], intervention[1], hit_goal, amp)
        for group, batch in (("control", control), ("hit", intervention)):
            rows = batch[-1]
            for j in np.unique(rows[:, 0]):
                indices = rows[rows[:, 0] == j, 1]
                hits = banks[j].hits[indices]
                exposure[group][banks[j].name]["banks"] += len(indices)
                exposure[group][banks[j].name]["mixed"] += int((hits.any(1) & ~hits.all(1)).sum())
        record = {"stage": "train", "step": step}
        for name, model in models.items():
            model.train()
            batch, goal, teachers = ((control, control_goal, control_teacher) if name.endswith("_control")
                                     else (intervention, hit_goal, hit_teacher))
            loss, parts, scores = forward_loss(name, model, mods, batch, goal, teachers, cfg, temperatures, amp)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {name} loss at step {step}")
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opts[name].step()
            schedules[name].step()
            if step == 1 or step % args.log_every == 0:
                record[name] = {"loss": float(loss), "gradient_norm_before_clip": float(norm),
                                "parts": {key: float(value) for key, value in parts.items()},
                                "native": native_hit_metrics(scores, batch[4])}
        if step == 1 or step % args.log_every == 0:
            record["exposure"] = copy.deepcopy(exposure)
            record["elapsed_seconds"] = time.perf_counter() - started
            log(record)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            torch.save({"config": cfg, "step": step, "models": cpu_states(),
                        "optimizers": {name: opt.state_dict() for name, opt in opts.items()},
                        "schedulers": {name: scheduler.state_dict() for name, scheduler in schedules.items()},
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                        "rng_old": rng_old.bit_generator.state, "rng_hit": rng_hit.bit_generator.state},
                       args.out / "continuation_last.pt")
    frozen_changed = [name for name in ("enc", "reader", "full")
                      if any(not torch.equal(value.detach().cpu(), blob["stage1"][name][key])
                             for key, value in mods[name].state_dict().items())]
    if frozen_changed:
        raise RuntimeError(f"Frozen source/readers changed: {frozen_changed}")
    log({"status": "DONE", "steps": args.steps, "selected_steps": selected_steps,
         "frozen_source_readers_unchanged": True, "exposure": exposure,
         "seconds": time.perf_counter() - started})
    metrics_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--eval-every", type=int, default=1200)
    parser.add_argument("--decisions", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--hit-weight", type=float, default=1.)
    parser.add_argument("--standard-mass", type=float, default=.8)
    parser.add_argument("--mixed-frac", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()
    try:
        validate_train_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    main(args)
