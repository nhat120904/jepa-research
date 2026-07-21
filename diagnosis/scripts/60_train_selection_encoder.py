"""Train the four pre-registered selection-aware encoder arms.

Input is the grouped raw-frame buffer from scripts/59.  The encoder is run live;
cached latents would become stale as soon as an encoder parameter changes.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from models.heads.encoder_adaptation import (  # noqa: E402
    trainable_encoder_state_dict, unfreeze_last_encoder_blocks)
from models.heads.lora_encoder import (  # noqa: E402
    encoder_lora_state_dict, inject_encoder_lora)
from models.heads.selection_cost import SelectionCostHead  # noqa: E402
from planning.selection_objectives import (  # noqa: E402
    grouped_hard_selection_regret, grouped_pairwise_logistic,
    grouped_regression_huber, grouped_softmin_regret)


def _load(modname: str, filename: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / filename))
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


encode_grad = _load("selection_train_encoder_grad", "38_train_encoder_lora.py").encode_grad


def group_index(buffer: dict, seeds: set[int]):
    seed = buffer["seed"].numpy()
    gid = buffer["group_id"].numpy()
    groups = []
    for group in np.unique(gid[np.isin(seed, list(seeds))]):
        idx = np.flatnonzero(gid == group)
        if len(idx) >= 2:
            groups.append(torch.from_numpy(idx))
    return groups


def materialize_group(buffer: dict, idx: torch.Tensor):
    frames = buffer["frames"][idx].float().permute(0, 3, 1, 2)
    prop = buffer["prop"][idx].float()
    ep = int(buffer["ep_idx"][idx[0]])
    if not torch.all(buffer["ep_idx"][idx] == ep):
        raise RuntimeError("a population group crosses episode goals")
    goal_frame = buffer["ep_goal_frames"][ep:ep + 1].float().permute(0, 3, 1, 2)
    goal_prop = buffer["ep_goal_prop"][ep:ep + 1].float()
    truth = buffer["true_cost"][idx].float()
    return frames, prop, goal_frame, goal_prop, truth


def objective_loss(name, prediction, truth, group, args):
    if name == "regression":
        return grouped_regression_huber(prediction, truth, group, beta=args.huber_beta)
    if name == "pairwise":
        return grouped_pairwise_logistic(
            prediction, truth, group, min_true_gap=args.min_true_gap)
    if name == "tail":
        return grouped_softmin_regret(
            prediction, truth, group, temperature=args.temperature)
    raise ValueError(name)


def weight_anchor(named_parameters, initial):
    return torch.stack([
        (parameter - initial[name]).square().mean()
        for name, parameter in named_parameters
    ]).mean()


@torch.no_grad()
def validate(adapter, head, buffer, groups, device, max_groups: int):
    head.eval()
    regrets, maes = [], []
    for idx in groups[:max_groups]:
        frames, prop, gf, gp, truth = materialize_group(buffer, idx)
        z = encode_grad(adapter, frames, prop)
        zg = encode_grad(adapter, gf, gp)
        pred = head(z, zg.expand(z.shape[0], *([-1] * (zg.ndim - 1))))
        local_group = torch.zeros(len(idx), dtype=torch.long, device=device)
        truth = truth.to(device)
        regrets.append(float(grouped_hard_selection_regret(pred, truth, local_group)))
        maes.append(float((pred - truth).abs().mean()))
    head.train()
    return {"selected_regret": float(np.mean(regrets)), "mae": float(np.mean(maes)),
            "n_groups": min(len(groups), max_groups)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="dino_wm_metaworld")
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--adaptation", choices=["lora", "last_blocks"], required=True)
    ap.add_argument("--objective", choices=["regression", "pairwise", "tail"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(62000, 62006)))
    ap.add_argument("--val-seeds", type=int, nargs="+", default=[62006, 62007])
    ap.add_argument("--last-blocks", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--lambda-anchor", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--huber-beta", type=float, default=0.05)
    ap.add_argument("--min-true-gap", type=float, default=0.005)
    ap.add_argument("--head-hidden", type=int, default=384)
    ap.add_argument("--pool-hidden", type=int, default=128)
    ap.add_argument("--val-groups", type=int, default=160)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "4")))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    buffer = torch.load(args.buffer, map_location="cpu", weights_only=False)
    train_groups = group_index(buffer, set(args.train_seeds))
    val_groups = group_index(buffer, set(args.val_seeds))
    fixed_rng = np.random.default_rng(12345)
    fixed_rng.shuffle(val_groups)
    if not train_groups or not val_groups:
        raise SystemExit(f"empty split: train_groups={len(train_groups)} val_groups={len(val_groups)}")
    print(f"buffer rows={len(buffer['true_cost'])} train_groups={len(train_groups)} "
          f"val_groups={len(val_groups)}", flush=True)

    adapter = build_adapter(args.model, device=str(device)).eval()
    adaptation_meta = {}
    if args.adaptation == "lora":
        injected = inject_encoder_lora(adapter, r=args.lora_r, alpha=args.lora_alpha)
        named_encoder = []
        for index, module in enumerate(injected):
            named_encoder.extend([(f"lora.{index}.A", module.A), (f"lora.{index}.B", module.B)])
        adaptation_meta = {"n_lora_modules": len(injected)}
    else:
        injected = None
        named_encoder, adaptation_meta = unfreeze_last_encoder_blocks(
            adapter, n_blocks=args.last_blocks)
    initial = {name: parameter.detach().clone() for name, parameter in named_encoder}
    print(f"adaptation={args.adaptation} objective={args.objective} "
          f"trainable_encoder={sum(p.numel() for _, p in named_encoder)/1e6:.2f}M "
          f"meta={adaptation_meta}", flush=True)

    # Infer the live frame-latent width after adaptation is installed.
    f0, p0, gf0, gp0, _ = materialize_group(buffer, train_groups[0][:1])
    with torch.no_grad():
        latent_dim = int(encode_grad(adapter, f0, p0).shape[-1])
    head = SelectionCostHead(latent_dim, hidden=args.head_hidden,
                             pool_hidden=args.pool_hidden).to(device).train()
    optimizer = torch.optim.AdamW([
        {"params": [p for _, p in named_encoder], "lr": args.encoder_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=1e-4)
    rng = np.random.default_rng(args.seed + 70000)
    best = None

    for epoch in range(args.epochs):
        losses, task_losses, anchors = [], [], []
        for _ in range(args.steps_per_epoch):
            idx = train_groups[int(rng.integers(len(train_groups)))]
            frames, prop, gf, gp, truth = materialize_group(buffer, idx)
            z = encode_grad(adapter, frames, prop)
            zg = encode_grad(adapter, gf, gp)
            prediction = head(z, zg.expand(z.shape[0], *([-1] * (zg.ndim - 1))))
            truth = truth.to(device)
            group = torch.zeros(len(idx), dtype=torch.long, device=device)
            task_loss = objective_loss(args.objective, prediction, truth, group, args)
            anchor = weight_anchor(named_encoder, initial)
            loss = task_loss + args.lambda_anchor * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in named_encoder] + list(head.parameters()), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            task_losses.append(float(task_loss.detach()))
            anchors.append(float(anchor.detach()))

        metrics = validate(adapter, head, buffer, val_groups, device, args.val_groups)
        print(f"epoch={epoch+1}/{args.epochs} loss={np.mean(losses):.6f} "
              f"task={np.mean(task_losses):.6f} anchor={np.mean(anchors):.6g} "
              f"val_regret={metrics['selected_regret']:.6f} val_mae={metrics['mae']:.6f}",
              flush=True)
        if best is None or metrics["selected_regret"] < best["val"]["selected_regret"]:
            if args.adaptation == "lora":
                encoder_state = encoder_lora_state_dict(injected, adapter)
            else:
                encoder_state = trainable_encoder_state_dict(adapter)
            best = {
                "format_version": 1,
                "model": args.model,
                "adaptation": args.adaptation,
                "objective": args.objective,
                "seed": args.seed,
                "latent_dim": latent_dim,
                "head_hidden": args.head_hidden,
                "pool_hidden": args.pool_hidden,
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "encoder": encoder_state,
                "last_blocks": args.last_blocks,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "temperature": args.temperature,
                "lambda_anchor": args.lambda_anchor,
                "train_seeds": args.train_seeds,
                "val_seeds": args.val_seeds,
                "epoch": epoch + 1,
                "val": metrics,
                "adaptation_meta": adaptation_meta,
            }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(best, temporary)
    os.replace(temporary, path)
    print(f"saved {path} best_epoch={best['epoch']} val={best['val']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
