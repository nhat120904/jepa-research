#!/usr/bin/env python3
"""Arms 2 and 3: frozen-encoder predictor continuation, with or without cosine AS.

ONE script for both arms; `--lambda-as 0` is arm 2 and `> 0` is arm 3.  Arm 2
still builds the symmetric triplet and still runs the same three H-step
rollouts, so the arms are matched in graph and compute and differ only by the
AS gradient.

  base loss : open-loop H-step rollout prediction on the CENTRE chunk, so both
              arms receive multi-step gradient.  Without this, arm 3 would
              differ from arm 2 by two things -- multi-step training and AS --
              and no gain could be attributed.
  AS term   : 1 - cos(v-, v+) on the terminal predictions of the triplet, the
              angular half of the exact curvature identity, which is the half
              the confirmatory analysis localised the false-valley effect to.

The encoder is frozen throughout so the latent chart is identical across arms
and curvature stays comparable.  Seeds are paired: the same --seed gives the
same batch order and the same perturbations in both arms, so the primary
contrast is a within-seed paired difference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "action_curvature_h0" / "scripts"))

from action_curvature_h0.core import make_feasible  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="quentinll/lewm-cube")
    p.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-as", type=float, required=True,
                   help="0 = arm 2 (continuation), > 0 = arm 3 (continuation + AS)")
    p.add_argument("--sigma-min", type=float, default=0.00125)
    p.add_argument("--sigma-max", type=float, default=0.20)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def cosine_as_loss(v_minus: torch.Tensor, v_plus: torch.Tensor) -> torch.Tensor:
    """1 - cos(v-, v+), the angular half of ||D2||^2 = radial + angular."""
    return (1.0 - torch.nn.functional.cosine_similarity(v_minus, v_plus, dim=-1)).mean()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("training requires the GPU nodes")

    import stable_pretraining as spt
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    upstream = ROOT / "diagnosis/external/stable-worldmodel/scripts/train"
    sys.path.insert(0, str(upstream))
    from lewm import get_img_preprocessor  # noqa: E402
    # Same alias the upstream trainer uses (scripts/train/lewm.py:14); the
    # symbol lives in stable_worldmodel.data, not data.utils.
    from stable_worldmodel.data import column_normalizer as get_column_normalizer  # noqa: E402

    # Window長 = history + horizon so one sample carries the context frames and
    # every frame the open-loop rollout must predict.
    num_steps = args.history_size + args.horizon
    dataset = swm.data.load_dataset(
        args.dataset, transform=None, num_steps=num_steps,
        frameskip=args.frameskip,
        keys_to_load=["pixels", "action"], keys_to_cache=["action"],
    )
    transforms = [get_img_preprocessor("pixels", "pixels", args.img_size),
                  get_column_normalizer(dataset, "action", "action")]
    dataset.transform = spt.data.transforms.Compose(*transforms)

    gen = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True, generator=gen,
    )

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda()
    model.interpolate_pos_encoding = True

    # Freeze everything on the encoder side; the latent chart must be identical
    # across arms or curvature is not comparable between them.
    frozen, trained = [], []
    for name, module in (("encoder", model.encoder), ("projector", model.projector)):
        for prm in module.parameters():
            prm.requires_grad_(False)
        frozen.append(name)
    model.encoder.eval()
    params = [prm for prm in model.parameters() if prm.requires_grad]
    for name in ("predictor", "action_encoder", "pred_proj"):
        if any(p.requires_grad for p in getattr(model, name).parameters()):
            trained.append(name)
    if not params:
        raise RuntimeError("nothing left trainable after freezing the encoder")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Perturbation RNG is seeded from the same --seed, so paired arms see the
    # identical delta sequence and the contrast is within-seed.
    rng = np.random.default_rng(args.seed)
    action_dim = dataset.get_dim("action") * args.frameskip

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    step = 0
    model.train()
    model.encoder.eval()

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            pixels = batch["pixels"].cuda(non_blocking=True)
            actions = torch.nan_to_num(batch["action"].cuda(non_blocking=True), 0.0)
            B = pixels.shape[0]
            H, T = args.history_size, args.horizon

            # Targets: encoded future frames, encoder frozen so no grad needed.
            with torch.no_grad():
                enc = model.encode({"pixels": pixels})
                target_emb = enc["emb"][:, H:H + T].detach()

            act_hist = actions[:, : H - 1]
            centre = actions[:, H - 1 : H - 1 + T]

            # One symmetric triplet per batch element, feasible by construction.
            base_dir = rng.normal(size=(B, T, action_dim))
            base_dir /= np.maximum(
                np.linalg.norm(base_dir.reshape(B, -1), axis=1), 1e-12
            )[:, None, None]
            sigma = float(np.exp(rng.uniform(np.log(args.sigma_min),
                                             np.log(args.sigma_max))))
            delta = torch.from_numpy(
                (base_dir * sigma).astype(np.float32)
            ).to(centre.device)

            # Triplet occupies the candidate axis S, exactly as the diagnostic
            # and the planner do, so training and measurement share one path.
            cand = torch.stack([centre - delta, centre, centre + delta], dim=1)
            info = {
                "pixels": pixels[:, :H].unsqueeze(1).expand(B, 3, H, *pixels.shape[2:]),
                "action_history": act_hist.unsqueeze(1).expand(B, 3, H - 1, action_dim),
            }
            rolled = model.rollout(info, cand)
            pred = rolled["predicted_emb"][:, :, H:]        # (B, 3, T, D)

            base_loss = (pred[:, 1] - target_emb).pow(2).mean()

            terminal = pred[:, :, -1]                       # (B, 3, D)
            v_minus = terminal[:, 1] - terminal[:, 0]
            v_plus = terminal[:, 2] - terminal[:, 1]
            as_loss = cosine_as_loss(v_minus, v_plus)

            loss = base_loss + args.lambda_as * as_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    # Guards: curvature falling is only good if the predictor
                    # stays responsive and still predicts.
                    d2 = (v_plus - v_minus).norm(dim=-1)
                    span = (terminal[:, 2] - terminal[:, 0]).norm(dim=-1)
                    history.append({
                        "step": step,
                        "loss": float(loss),
                        "base_loss": float(base_loss),
                        "as_loss": float(as_loss),
                        "sigma": sigma,
                        "curvature_norm": float((d2 / (span + 1e-12)).mean()),
                        "action_sensitivity": float(
                            (span / (2.0 * delta.flatten(1).norm(dim=1) + 1e-12)).mean()
                        ),
                    })
                    print(json.dumps(history[-1]), flush=True)
            step += 1

    torch.save({"model": model.state_dict(),
                "lambda_as": args.lambda_as, "seed": args.seed,
                "steps": args.steps}, args.out_dir / "checkpoint.pt")
    (args.out_dir / "train_log.json").write_text(json.dumps({
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "frozen_modules": frozen, "trained_modules": trained,
        "n_trainable_params": int(sum(p.numel() for p in params)),
        "history": history,
    }, indent=2))
    print("TRAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
