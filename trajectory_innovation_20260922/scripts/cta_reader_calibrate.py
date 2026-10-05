"""Matched reader calibration on hard source and frozen-WM expected codes.

Only the two readers are updated. Each sees the same observed context/goal and
the same two code domains. Hard future codes are training targets only; model
selection and deployment consume predicted codes only. The averaged source
branch encourages retention but does not guarantee it.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cta_factorized_direct_train import TaskOnlyBank, model_hash, schedule, selection_metrics, sha256
from scripts.cta_train_v2 import Bank
from ti_wm.contract import require_compute
from ti_wm.cta import Scorer, SourceEncoder, goal_scores
from ti_wm.cta_hit_training import HitBank, StratifiedHitPool, native_hit_loss, native_hit_metrics
from ti_wm.cta_hit_contracts import validate_bank_shapes
from ti_wm.cta_parallel import ParallelFSQWM
from ti_wm.sibling import PCA_DIM, rank_loss


TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")
K = 8


def random_state():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_random_state(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def frozen_codes(encoder, wm, ctx, future, actions, amp):
    # Regular no_grad tensors can be saved by the trainable reader backward;
    # inference_mode tensors cannot. Goals are absent from both predictors.
    with torch.no_grad(), amp():
        source = encoder(ctx, future)
        predicted = wm(ctx, actions)[0]
    return source.detach(), predicted.detach()


def reader_objective(reader, ctx, source, predicted, goal, labels, hits, temperature, hit_weight, amp):
    """Average dense objective over both code domains, with optional hit loss.

    No proposed actions or future observations are arguments to the reader.
    ``source`` is a frozen hard code and ``predicted`` a frozen continuous code.
    """
    candidates = labels.shape[1]
    with amp():
        source_scores = reader(ctx, source, goal).float().view(-1, candidates)
        predicted_scores = reader(ctx, predicted, goal).float().view(-1, candidates)
        source_rank = rank_loss(source_scores, labels)
        predicted_rank = rank_loss(predicted_scores, labels)
        dense = .5 * (source_rank + predicted_rank)
        if hit_weight:
            source_hit = native_hit_loss(source_scores, hits, temperature)
            predicted_hit = native_hit_loss(predicted_scores, hits, temperature)
        else:
            with torch.no_grad():
                source_hit = native_hit_loss(source_scores, hits, temperature)
                predicted_hit = native_hit_loss(predicted_scores, hits, temperature)
        hit = .5 * (source_hit + predicted_hit)
        loss = dense + hit_weight * hit
    parts = {"source_rank": source_rank, "predicted_rank": predicted_rank, "dense_average": dense,
             "source_hit": source_hit, "predicted_hit": predicted_hit, "hit_average": hit,
             "weighted_hit_average": hit_weight * hit}
    return loss, parts, source_scores, predicted_scores


def check_frozen(encoder, wm, base):
    changed = []
    for name, module, states in (("encoder", encoder, base["stage1"]["enc"]),
                                 ("wm", wm, base["wms"]["cta"])):
        if any(parameter.requires_grad or parameter.grad is not None for parameter in module.parameters()):
            changed.append(f"{name}:gradient")
        if module.training:
            changed.append(f"{name}:train_mode")
        if any(not torch.equal(value.detach().cpu(), states[key]) for key, value in module.state_dict().items()):
            changed.append(f"{name}:weights_or_buffers")
    if changed:
        raise RuntimeError(f"Frozen modules changed: {changed}")
    return {"encoder_exactly_unchanged": True, "wm_exactly_unchanged": True,
            "encoder_wm_have_no_gradients": True}


@torch.inference_mode()
def predicted_selection(readers, wm, banks, goals, device, amp, limit=None):
    result = {name: {} for name in readers}
    for reader in readers.values():
        reader.eval()
    for bank in banks:
        count = bank.n if limit is None else min(limit, bank.n)
        arrays = {name: np.empty((count, K), np.float32) for name in readers}
        for row in range(count):
            ctx, future, actions, _ = bank.batch([row], device)
            if future:
                raise RuntimeError("Predicted-only selection received future features")
            with amp():
                predicted = wm(ctx, actions)[0]
                for name, reader in readers.items():
                    arrays[name][row] = goal_scores(reader, ctx, predicted, goals).float().cpu().numpy().reshape(K)
        for name in readers:
            result[name][bank.name] = arrays[name]
    return result


def validate(args):
    for name in ("steps", "eval_every", "warmup", "decisions", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("train_limit", "selection_limit"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive or None")
    if args.seed < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Invalid seed or learning rate")
    for name in ("hit_weight", "standard_mass", "mixed_frac"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if args.hit_weight < 0 or not 0 <= args.standard_mass <= 1 or not 0 <= args.mixed_frac <= 1:
        raise ValueError("Invalid native-hit/sampling settings")


def main(args):
    require_compute()
    validate(args)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(base["config"])
    if cfg.get("chunk") != 15 or cfg.get("hindsight", 0.) or cfg.get("success_bonus", 0.):
        raise ValueError("Expected the factual BASE L15 checkpoint with original goal/geometry labels")
    if set(cfg["sources"]) != {"r4std", "r4pert"}:
        raise ValueError("Expected only existing L15 standard/perturbed sources")
    temperature = float(cfg["consistency_scale"]["reader"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid fixed BASE reader temperature")
    banks = [HitBank(Bank(Path(path), name), path, standard=(name == "r4std"))
             for name, path in cfg["sources"].items()]
    selection = [HitBank(TaskOnlyBank(path, name), path, standard=True)
                 for name, path in cfg["selection"].items()]
    for bank in banks:
        validate_bank_shapes(bank.name, bank.geom.shape, bank.hits.shape, bank.chunk.shape, K, 15,
                             {key: value.shape for key, value in bank.mm.items()})
    if args.train_limit:
        for bank in banks:
            bank.base.n = min(bank.n, args.train_limit)
            bank.base.spread = bank.spread[bank.spread < bank.n]
    pool = StratifiedHitPool(banks, args.standard_mass, args.mixed_frac, .75)
    split = pool.assert_disjoint(selection, forbidden_roots=range(2100, 34000))
    goals = torch.from_numpy(np.load(args.goals, allow_pickle=False)).to(device)
    if tuple(goals.shape) != (16, 256, PCA_DIM) or not torch.isfinite(goals).all():
        raise ValueError("Expected all16 original goal images")
    encoder = SourceEncoder(m=cfg["m"], conditional=bool(cfg.get("conditional", True)),
                            path=bool(cfg.get("path", True)), dropout=cfg["dropout"]).to(device)
    wm = ParallelFSQWM(m=cfg["m"], chunk=15, dropout=cfg["dropout"]).to(device)
    initial_reader = Scorer("code", m=cfg["m"], dropout=cfg["dropout"]).to(device)
    encoder.load_state_dict(base["stage1"]["enc"], strict=True)
    wm.load_state_dict(base["wms"]["cta"], strict=True)
    initial_reader.load_state_dict(base["stage1"]["reader"], strict=True)
    encoder.eval().requires_grad_(False)
    wm.eval().requires_grad_(False)
    readers = {"reader_control": initial_reader, "reader_hit": copy.deepcopy(initial_reader)}
    initial_identity = {"encoder": model_hash(encoder), "wm": model_hash(wm),
                        "readers": {name: model_hash(reader) for name, reader in readers.items()}}
    if len(set(initial_identity["readers"].values())) != 1:
        raise RuntimeError("Initial reader pairing failed")
    cfg["method"] = "cta_reader_domain_calibration"
    cfg["reader_calibration"] = {"base": str(args.base), "base_sha256": sha256(args.base),
                                 "job": os.environ["SLURM_JOB_ID"], "steps": args.steps, "seed": args.seed,
                                 "lr": args.lr, "warmup": args.warmup, "decisions": args.decisions,
                                 "hit_weight": args.hit_weight, "fixed_base_reader_temperature": temperature,
                                 "sampling": pool.report(args.decisions), "root_split": split,
                                 "initial_state_sha256": initial_identity,
                                 "dense_objective": "(source hard-code rank_loss + frozen-WM expected-code rank_loss)/2 for both readers",
                                 "hit_objective": "HIT only: hit_weight*(source native_hit_loss + predicted native_hit_loss)/2",
                                 "control_scope": "domain calibration and data are common; native-hit objective isolated",
                                 "source_retention_scope": "source branch is averaged task supervision, not a guaranteed retention constraint",
                                 "reader_inputs": ["observed context", "code", "goal"], "actions_bypass_reader": False,
                                 "selection_domain": "predicted frozen-WM code only; no actual future features opened",
                                 "selection_goal_images": 16, "selection_limit": args.selection_limit,
                                 "selection_batch": 1, "selection_rule": "retained_geometry_gap + 0.25 * mixed_hit_capture",
                                 "matched_dropout_rng": True, "independent_optimizers_and_clip": True,
                                 "source_metadata": {bank.name: {"path": str(bank.feature_path / "meta.npz"),
                                                                  "sha256": sha256(bank.feature_path / "meta.npz"), "active_rows": bank.n}
                                                     for bank in banks + selection},
                                 "goals": {"path": str(args.goals), "sha256": sha256(args.goals)},
                                 "backend": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                             "gpu": torch.cuda.get_device_name(), "precision": "bfloat16 autocast,float32 losses",
                                             "cudnn_deterministic": True, "cudnn_benchmark": False,
                                             "cudnn_tf32": True, "matmul_tf32": False}}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    log_file = (args.out / "metrics.jsonl").open("a")
    def log(value):
        log_file.write(json.dumps(value) + "\n")
        log_file.flush()
        print(json.dumps(value), flush=True)
    log({"config": cfg})
    optimizers = {name: torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=cfg["wd"])
                  for name, reader in readers.items()}
    schedulers = {name: schedule(optimizer, args.steps, args.warmup) for name, optimizer in optimizers.items()}
    rng = np.random.default_rng(args.seed)
    best, selected, selected_steps = {}, {}, {}
    exposure = {bank.name: {"banks": 0, "mixed": 0} for bank in banks}
    def save_selected():
        for name, states in selected.items():
            path = args.out / ("hit" if name == "reader_hit" else "control") / "cta_v2.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            own = copy.deepcopy(cfg)
            own["reader_calibration"].update(variant=name, selected_steps=selected_steps.copy())
            temporary = path.with_suffix(".tmp")
            torch.save({"config": own, "stage1": {"reader": states}, "wms": {"cta": base["wms"]["cta"]}}, temporary)
            temporary.replace(path)
    def evaluate(step):
        arrays = predicted_selection(readers, wm, selection, goals, device, amp, args.selection_limit)
        metrics = {name: selection_metrics(selection, scores, args.selection_limit) for name, scores in arrays.items()}
        if step == 0 and any(not np.array_equal(arrays["reader_control"][bank.name], arrays["reader_hit"][bank.name]) for bank in selection):
            raise RuntimeError("Initial predicted-score reader pairing failed")
        for name, value in metrics.items():
            if name not in best or value["selection_criterion"] > best[name]:
                best[name], selected_steps[name] = value["selection_criterion"], step
                selected[name] = {key: value.detach().cpu().clone() for key, value in readers[name].state_dict().items()}
        np.savez(args.out / f"selection_scores_{step}.npz", **{f"{name}__{bank}": value
                                                             for name, scores in arrays.items() for bank, value in scores.items()})
        save_selected()
        log({"stage": "selection", "step": step, "metrics": metrics, "selected_steps": selected_steps.copy()})
        (args.out / "selection.json").write_text(json.dumps({"step": step, "metrics": metrics,
                                                            "selected_steps": selected_steps}, indent=2) + "\n")
    evaluate(0)
    for step in range(1, args.steps + 1):
        ctx, future, actions, labels, hits, rows = pool.sample(rng, args.decisions, device, return_rows=True)
        goal = goals[torch.as_tensor(rng.integers(0, len(goals), len(labels)), device=device)].repeat_interleave(K, 0)
        source, predicted = frozen_codes(encoder, wm, ctx, future, actions, amp)
        before, after = random_state(), None
        record = {"stage": "train", "step": step}
        for name, reader in readers.items():
            if name == "reader_hit":
                restore_random_state(before)
            reader.train()
            weight = args.hit_weight if name == "reader_hit" else 0.
            loss, parts, source_scores, predicted_scores = reader_objective(reader, ctx, source, predicted, goal,
                                                                          labels, hits, temperature, weight, amp)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {name} loss at{step}")
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1., error_if_nonfinite=True)
            optimizers[name].step()
            schedulers[name].step()
            if name == "reader_control":
                after = random_state()
            if step == 1 or step % args.log_every == 0:
                record[name] = {"loss": float(loss), "parts": {key: float(value) for key, value in parts.items()},
                                "gradient_norm_before_clip": float(norm),
                                "native_source": native_hit_metrics(source_scores, hits),
                                "native_predicted": native_hit_metrics(predicted_scores, hits)}
        restore_random_state(after)
        for index in np.unique(rows[:, 0]):
            indices = rows[rows[:, 0] == index, 1]
            flags = banks[index].hits[indices]
            exposure[banks[index].name]["banks"] += len(indices)
            exposure[banks[index].name]["mixed"] += int((flags.any(-1) & ~flags.all(-1)).sum())
        if step == 1 or step % args.log_every == 0:
            record.update(exposure=copy.deepcopy(exposure), elapsed_seconds=time.perf_counter() - started)
            log(record)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            torch.save({"config": cfg, "step": step,
                        "readers": {name: {key: value.detach().cpu() for key, value in reader.state_dict().items()}
                                    for name, reader in readers.items()},
                        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                        "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
                        "random_state": random_state(), "data_rng": rng.bit_generator.state}, args.out / "reader_calibration_last.pt")
    unchanged = check_frozen(encoder, wm, base)
    cfg["reader_calibration"].update(actual_updates=args.steps, selected_steps=selected_steps,
                                      exposure=exposure, wall_seconds=time.perf_counter() - started,
                                      frozen_check=unchanged)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_selected()
    log({"status": "DONE", "updates": args.steps, "selected_steps": selected_steps,
         "exposure": exposure, "frozen_check": unchanged, "wall_seconds": time.perf_counter() - started})
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=TI / "cta_replan15_train_56504/train/cta_v2.pt")
    parser.add_argument("--goals", type=Path, default=TI / "cta_geometry_e2e_55018/features/goals.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--decisions", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--hit-weight", type=float, default=1.)
    parser.add_argument("--standard-mass", type=float, default=.8)
    parser.add_argument("--mixed-frac", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--selection-limit", type=int, default=None)
    args = parser.parse_args()
    try:
        validate(args)
    except ValueError as error:
        parser.error(str(error))
    main(args)
