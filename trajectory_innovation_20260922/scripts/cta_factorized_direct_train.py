"""Fresh task-only factorized DIRECT on factual L15 CTA feature banks.

Read source configuration JSON, never a CTA/model checkpoint. Only observed
cur/prev tokens, observed agent positions and proposed actions enter the model.
Geometry and done8 remain supervised labels. Future feature arrays are not even
opened. Submit training/model work through sbatch on a compute node.
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
from ti_wm.contract import require_compute
from ti_wm.cta import action_features, goal_scores, proprio
from ti_wm.cta_eval import ranking_metrics
from ti_wm.cta_factorized_direct import FactorizedDirect
from ti_wm.cta_hit_training import HitBank, StratifiedHitPool, native_hit_loss, native_hit_metrics
from ti_wm.sibling import PCA_DIM, RANK_MARGIN, rank_loss


TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")
K = 8


class TaskOnlyBank:
    """Observed-input bank with no future-feature storage or accesses."""
    def __init__(self, path, name):
        self.path, self.name = Path(path), name
        with np.load(self.path / "meta.npz", allow_pickle=False) as meta:
            self.root = np.array(meta["root"], copy=True)
            self.decision = np.array(meta["decision"], copy=True)
            self.geom = np.array(meta["cov8"], np.float32)
            self.ctx_pos = torch.from_numpy(np.array(meta["ctx_pos"], np.float32))
            self.chunk = torch.from_numpy(np.array(meta["chunk"], np.float32))
        self.n = len(self.root)
        self.spread = np.flatnonzero(np.ptp(self.geom, axis=1) > RANK_MARGIN)
        self.mm = {key: np.load(self.path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                   for key in ("cur", "prev")}
        if self.geom.shape != (self.n, K) or tuple(self.chunk.shape) != (self.n, K, 15, 2):
            raise ValueError(f"{name}: expected factual K8/L15 geometry/action banks")
        if tuple(self.ctx_pos.shape) != (self.n, 2, 2):
            raise ValueError(f"{name}: invalid observed agent history")
        if self.mm["cur"].shape != (self.n, 256, PCA_DIM) or self.mm["prev"].shape != (self.n, 64, PCA_DIM):
            raise ValueError(f"{name}: incompatible observed feature dimensions")
        if not np.isfinite(self.geom).all() or not torch.isfinite(self.chunk).all() or not torch.isfinite(self.ctx_pos).all():
            raise ValueError(f"{name}: nonfinite metadata")

    def batch(self, indices, device):
        indices = np.asarray(indices)
        ctx = {key: torch.from_numpy(np.asarray(self.mm[key][indices])).to(device).repeat_interleave(K, 0)
               for key in ("cur", "prev")}
        ctx["prop"] = proprio(self.ctx_pos[indices, 0], self.ctx_pos[indices, 1]).to(device).repeat_interleave(K, 0)
        actions = action_features(self.chunk[indices], self.ctx_pos[indices, 0][:, None]).flatten(0, 1).to(device)
        labels = torch.from_numpy(self.geom[indices]).to(device)
        return ctx, {}, actions, labels


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def schedule(optimizer, steps, warmup):
    def multiplier(step):
        return min(1., (step + 1) / warmup) * .5 * (1 + math.cos(math.pi * min(step, steps) / steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def validate(args):
    for name in ("steps", "eval_every", "warmup", "decisions", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    for name in ("train_limit", "selection_limit"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive or None")
    for name in ("lr", "weight_decay", "hit_weight", "hit_temperature", "dropout", "standard_mass", "mixed_frac"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if args.lr <= 0 or args.weight_decay < 0 or args.hit_weight < 0 or args.hit_temperature <= 0:
        raise ValueError("Invalid optimizer/native-hit settings")
    if not 0 <= args.dropout < 1 or not 0 <= args.standard_mass <= 1 or not 0 <= args.mixed_frac <= 1:
        raise ValueError("Invalid dropout/sampling fractions")


@torch.inference_mode()
def selection_scores(model, banks, goals, device, amp, limit=None):
    model.eval()
    result = {}
    for bank in banks:
        count = bank.n if limit is None else min(limit, bank.n)
        scores = np.empty((count, K), np.float32)
        for row in range(count):
            ctx, unused_future, actions, _ = bank.batch([row], device)
            if unused_future:
                raise RuntimeError("Task-only bank exposed future features")
            with amp():
                code = model.encode(ctx, actions)
                value = goal_scores(model.reader, ctx, code, goals)
            scores[row] = value.float().cpu().numpy().reshape(K)
        result[bank.name] = scores
    return result


def selection_metrics(banks, scores, limit=None):
    size = lambda bank: bank.n if limit is None else min(limit, bank.n)
    geometry = np.concatenate([bank.geom[:size(bank)] for bank in banks])
    hits = np.concatenate([bank.hits[:size(bank)] for bank in banks])
    roots = np.concatenate([bank.root[:size(bank)] for bank in banks])
    ranking = ranking_metrics(np.concatenate([scores[bank.name] for bank in banks]), geometry, roots, ci=False)
    ranking.pop("chosen")
    native = native_hit_metrics(np.concatenate([scores[bank.name] for bank in banks]), hits)
    capture = native["mixed_capture"]
    criterion = ranking["retained_gap"]["ratio"] + .25 * (capture if capture is not None else 0.)
    if not math.isfinite(criterion):
        raise RuntimeError("Nonfinite factorized DIRECT selection criterion")
    return {"geometry": ranking, "native": native, "selection_criterion": criterion,
            "criterion_definition": "retained_geometry_gap + 0.25 * mixed_hit_capture"}


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
    factual = json.loads(args.data_config.read_text())
    if factual.get("chunk") != 15 or set(factual["sources"]) != {"r4std", "r4pert"}:
        raise ValueError("Expected factual L15 standard/perturbed source configuration")
    banks = [HitBank(TaskOnlyBank(path, name), path, standard=(name == "r4std"))
             for name, path in factual["sources"].items()]
    selection = [HitBank(TaskOnlyBank(path, name), path, standard=True)
                 for name, path in factual["selection"].items()]
    if args.train_limit:
        for bank in banks:
            bank.base.n = min(bank.n, args.train_limit)
            bank.base.spread = bank.spread[bank.spread < bank.n]
    pool = StratifiedHitPool(banks, args.standard_mass, args.mixed_frac, .75)
    split = pool.assert_disjoint(selection, forbidden_roots=range(2100, 34000))
    goals = torch.from_numpy(np.load(args.goals, allow_pickle=False)).to(device)
    if tuple(goals.shape) != (16, 256, PCA_DIM) or not torch.isfinite(goals).all():
        raise ValueError("Expected all16 original goal feature images")
    # The only model constructors in this script are fresh random predictor and
    # reader. No source/decoder/FULL reader or pretrained planner is instantiated.
    model = FactorizedDirect(m=16, chunk=15, dropout=args.dropout).to(device)
    parameters = {"predictor": sum(p.numel() for p in model.predictor.parameters()),
                  "reader": sum(p.numel() for p in model.reader.parameters())}
    cfg = {"method": "task_only_factorized_direct", "m": 16, "chunk": 15,
           "sources": factual["sources"], "selection": factual["selection"],
           "seed": args.seed, "job": os.environ["SLURM_JOB_ID"], "steps": args.steps,
           "lr": args.lr, "weight_decay": args.weight_decay, "warmup": args.warmup,
           "dropout": args.dropout, "decisions": args.decisions,
           "initialization": "fresh random predictor and reader; no model checkpoint loading",
           "initial_state_sha256": model_hash(model), "supervision": ["factual_geometry_rank_loss", "native_done8_hit_loss"],
           "future_feature_supervision": False, "future_code_target_supervision": False,
           "future_feature_files_opened": False, "task_labels_are_future_derived": True,
           "shared_visual_features": "same frozen DINO/PCA cur/prev caches as CTA; vision features are not trained here",
           "bottleneck": "same categorical-head expected code as CTA,16x3 continuous values; no code targets/NLL",
           "parameters": parameters | {"total": sum(parameters.values())},
           "task_loss": {"dense": "original DIRECT rank_loss", "hit_weight": args.hit_weight,
                         "hit_temperature": args.hit_temperature,
                         "temperature_scope": "fixed1 by default in geometry-score units; not pretrained CTA-reader score normalization"},
           "sampling": pool.report(args.decisions), "root_split": split,
           "selection_limit": args.selection_limit, "selection_goal_images": 16,
           "selection_batch": 1, "selection_rule": "retained_geometry_gap + 0.25 * mixed_hit_capture",
           "data_configuration": {"path": str(args.data_config), "sha256": sha256(args.data_config),
                                  "original_stage1_updates": factual.get("steps1"), "original_stage2_updates": factual.get("steps2")},
           "input_metadata": {bank.name: {"path": str(bank.feature_path / "meta.npz"),
                                           "sha256": sha256(bank.feature_path / "meta.npz"), "active_rows": bank.n,
                                           "observed_features": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                                                        "bytes": (bank.feature_path / f"{key}.npy").stat().st_size}
                                                                 for key, value in bank.mm.items()}}
                              for bank in banks + selection},
           "goals": {"path": str(args.goals), "sha256": sha256(args.goals), "images": 16},
           "architecture": model.architecture,
           "comparison_scope": "deployment architecture/bottleneck/data matched; fresh task-only recipe versus future-feature/code-target recipe; total training compute, initialization, dense loss and hit temperature are not matched",
           "backend": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                       "precision": "bfloat16 autocast,float32 task losses", "matmul_tf32": False,
                       "cudnn_deterministic": True, "cudnn_benchmark": False, "cudnn_tf32": True}}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    log_file = (args.out / "metrics.jsonl").open("a")
    def log(value):
        log_file.write(json.dumps(value) + "\n")
        log_file.flush()
        print(json.dumps(value), flush=True)
    log({"config": cfg})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = schedule(optimizer, args.steps, args.warmup)
    rng = np.random.default_rng(args.seed)
    exposure = {bank.name: {"banks": 0, "mixed": 0} for bank in banks}
    seen = [np.zeros(bank.n, np.bool_) for bank in banks]
    best, chosen_step, best_bundle = -float("inf"), None, None
    def evaluate(step):
        nonlocal best, chosen_step, best_bundle
        arrays = selection_scores(model, selection, goals, device, amp, args.selection_limit)
        metric = selection_metrics(selection, arrays, args.selection_limit)
        np.savez(args.out / f"selection_scores_{step}.npz", **{f"fdirect__{name}": value for name, value in arrays.items()})
        if metric["selection_criterion"] > best:
            best, chosen_step = metric["selection_criterion"], step
            cfg["selected_step"] = step
            best_bundle = model.deployable_checkpoint(cfg)
            temporary = args.out / "cta_v2.tmp"
            torch.save(best_bundle, temporary)
            temporary.replace(args.out / "cta_v2.pt")
        log({"stage": "selection", "step": step, "metrics": metric, "selected_step": chosen_step})
        (args.out / "selection.json").write_text(json.dumps({"step": step, "metrics": metric,
                                                            "selected_step": chosen_step}, indent=2) + "\n")
    evaluate(0)
    for step in range(1, args.steps + 1):
        ctx, future, actions, labels, hits, rows = pool.sample(rng, args.decisions, device, return_rows=True)
        if future:
            raise RuntimeError("Task-only sampler exposed future observations")
        goal_indices = torch.as_tensor(rng.integers(0, len(goals), len(labels)), device=device)
        goal = goals[goal_indices].repeat_interleave(K, 0)
        model.train()
        with amp():
            scores = model(ctx, actions, goal).float().view(-1, K)
            dense = rank_loss(scores, labels)
            hit = native_hit_loss(scores, hits, args.hit_temperature)
            loss = dense + args.hit_weight * hit
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite factorized DIRECT loss at{step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        for index in np.unique(rows[:, 0]):
            indices = rows[rows[:, 0] == index, 1]
            flags = banks[index].hits[indices]
            exposure[banks[index].name]["banks"] += len(indices)
            exposure[banks[index].name]["mixed"] += int((flags.any(-1) & ~flags.all(-1)).sum())
            seen[index][indices] = True
        if step == 1 or step % args.log_every == 0:
            log({"stage": "train", "step": step, "loss": float(loss), "rank": float(dense), "hit": float(hit),
                 "gradient_norm_before_clip": float(norm), "native": native_hit_metrics(scores, hits),
                 "exposure": copy.deepcopy(exposure), "unique_bank_rows_seen": {bank.name: int(mask.sum()) for bank, mask in zip(banks, seen)},
                 "elapsed_seconds": time.perf_counter() - started})
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            torch.save({"config": cfg, "step": step, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                        "data_rng": rng.bit_generator.state}, args.out / "factorized_last.pt")
    cfg.update(selected_step=chosen_step, actual_updates=args.steps, wall_seconds=time.perf_counter() - started,
               exposure=exposure, unique_bank_rows_seen={bank.name: int(mask.sum()) for bank, mask in zip(banks, seen)})
    # Preserve selected weights while attaching the actual completed budget
    # and exposure to the deployable checkpoint's config. No model is reloaded.
    best_bundle["config"].update(copy.deepcopy(cfg))
    temporary = args.out / "cta_v2.tmp"
    torch.save(best_bundle, temporary)
    temporary.replace(args.out / "cta_v2.pt")
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    log({"status": "DONE", "selected_step": chosen_step, "updates": args.steps, "exposure": exposure,
         "unique_bank_rows_seen": cfg["unique_bank_rows_seen"], "wall_seconds": cfg["wall_seconds"],
         "deployable_checkpoint": str(args.out / "cta_v2.pt"), "arm_spec": "FDIRECT=FD:cta"})
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=TI / "cta_replan15_train_56504/train/config.json")
    parser.add_argument("--goals", type=Path, default=TI / "cta_geometry_e2e_55018/features/goals.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--decisions", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--hit-weight", type=float, default=1.)
    parser.add_argument("--hit-temperature", type=float, default=1.)
    parser.add_argument("--standard-mass", type=float, default=.8)
    parser.add_argument("--mixed-frac", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--selection-limit", type=int, default=None)
    args = parser.parse_args()
    try:
        validate(args)
    except ValueError as error:
        parser.error(str(error))
    main(args)
