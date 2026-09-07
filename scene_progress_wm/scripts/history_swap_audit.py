#!/usr/bin/env python3
"""Does the filter recover task state the current frame does not carry?

This is the gate the whole predictive-state claim rests on, and it is built to
fail loudly rather than quietly.  The method says a belief filtered from the
observation history holds something a single frame cannot: which persistent
events have happened.  On Scene those are the buttons, which lock the drawer and
the window, and whose effect on what an action will do outlives their appearance
in any one image.

Two stages, cheapest first.

**Stage 0 -- is there anything to recover?**  Probe the frozen LeWM latent for
each latched variable.  If a single frame already reveals the button states
almost perfectly, there is no aliasing on this arena, no hidden state for a
filter to reconstruct, and the premise is dead regardless of anything downstream.
That outcome is worth ten minutes to find.

**Stage 1 -- does the filter beat the frame?**  Probe each arm's belief for the
same variables.  The contrast that matters is ``ps_jepa`` against ``ps_frame``:
both are 256-dimensional beliefs from the same architecture at the same parameter
count, and they differ only in whether the recurrence is connected.  The raw
latent is 192-dimensional and is reported as context, not as the matched control,
because a probe on a wider input is not a fair fight.

``ps_frame`` is also the harness check: its belief is a function of the current
latent alone, so it must land near the raw-latent probe.  If it does not, the
belief extraction is wrong and no other number here means anything.

Splits are by **episode**, never by row: neighbouring grid rows are nearly the
same state, so a random row split would leak the answer across it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.ps_jepa import PSJEPAConfig, BeliefEncoder  # noqa: E402

PROTOCOL = "scene_progress_wm_history_swap_v1"

#: The latched variables. The buttons are the lock state the method is about;
#: drawer and window are included because they are what the buttons gate.
TARGETS = ("button_0", "button_1", "drawer_open", "window_open")

DRAWER_OPEN_THRESHOLD = -0.12
WINDOW_OPEN_THRESHOLD = 0.16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", default="ps_jepa,ps_frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-blocks", type=int, default=48)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--probe-steps", type=int, default=3000)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def latched_labels(cache_dir: Path, meta: dict) -> dict[str, np.ndarray]:
    """Ground-truth latched variables on the render grid, from the dataset."""
    stride = int(meta["pixel_stride"])
    buttons = np.load(cache_dir / "button_states.npy", mmap_mode="r")
    state = np.load(cache_dir / "state.npy", mmap_mode="r")
    rows = np.arange(0, len(buttons), stride)
    columns = list(meta["state_columns"])
    drawer = np.asarray(state[rows, columns.index("drawer")], dtype=np.float64)
    window = np.asarray(state[rows, columns.index("window")], dtype=np.float64)
    return {
        "button_0": np.asarray(buttons[rows, 0], dtype=np.int64),
        "button_1": np.asarray(buttons[rows, 1], dtype=np.int64),
        "drawer_open": (drawer <= DRAWER_OPEN_THRESHOLD).astype(np.int64),
        "window_open": (window >= WINDOW_OPEN_THRESHOLD).astype(np.int64),
    }


def sample_positions(
    num_grid: int, meta: dict, context: int, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Grid positions with a full in-episode context behind them."""
    stride = int(meta["pixel_stride"])
    episode_len = int(meta["episode_len"])
    grid = np.arange(num_grid, dtype=np.int64)
    # the window's first frame still needs the block that led into it, so the
    # usable span starts one block earlier than the frame itself
    first_row = (grid - context) * stride
    last_row = grid * stride
    ok = first_row >= 0
    ok[ok] = (first_row[ok] // episode_len) == (last_row[ok] // episode_len)
    usable = np.flatnonzero(ok)
    take = min(count, usable.size)
    return np.sort(rng.choice(usable, size=take, replace=False))


@torch.no_grad()
def encode_beliefs(
    encoder: BeliefEncoder,
    latents: np.ndarray,
    actions: np.ndarray,
    positions: np.ndarray,
    *,
    config: PSJEPAConfig,
    context: int,
    stride: int,
    device,
    batch: int = 256,
) -> np.ndarray:
    """The belief at each position, filtered over the preceding context."""
    out = np.empty((positions.size, config.belief_dim), dtype=np.float32)
    for start in range(0, positions.size, batch):
        chunk = positions[start : start + batch]
        window = np.stack(
            [np.asarray(latents[p - context + 1 : p + 1], dtype=np.float32) for p in chunk]
        )
        blocks = np.empty((chunk.size, context, config.action_input_dim), dtype=np.float32)
        for i, position in enumerate(chunk):
            for t in range(context):
                row = (position - context + 1 + t) * stride
                blocks[i, t] = np.asarray(
                    actions[row - stride : row], dtype=np.float32
                ).reshape(-1)
        beliefs, _ = encoder.scan(
            torch.from_numpy(window).to(device), torch.from_numpy(blocks).to(device)
        )
        out[start : start + chunk.size] = beliefs[:, -1].cpu().numpy()
    return out


def balanced_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean of per-class recall, so a skewed variable cannot be gamed by majority."""
    predicted = (logits > 0).astype(np.int64)
    scores = []
    for value in (0, 1):
        mask = labels == value
        if mask.sum() == 0:
            continue
        scores.append(float((predicted[mask] == value).mean()))
    return float(np.mean(scores)) if scores else float("nan")


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    device,
    args,
    seed: int,
) -> dict:
    """One linear probe. Features are standardised on the training split only."""
    torch.manual_seed(seed)
    mean = features[train_mask].mean(axis=0, keepdims=True)
    std = features[train_mask].std(axis=0, keepdims=True) + 1e-6
    scaled = (features - mean) / std

    x_train = torch.from_numpy(scaled[train_mask]).float().to(device)
    y_train = torch.from_numpy(labels[train_mask]).float().to(device)
    x_val = torch.from_numpy(scaled[~train_mask]).float().to(device)
    y_val = labels[~train_mask]

    probe = torch.nn.Linear(features.shape[1], 1).to(device)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    # class weighting so a rare positive is not simply ignored
    positive = float(y_train.mean().clamp(1e-3, 1 - 1e-3))
    weight = torch.tensor((1 - positive) / positive, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(args.probe_steps):
        index = torch.randint(
            0, x_train.shape[0], (min(args.batch_size, x_train.shape[0]),), generator=generator
        ).to(device)
        loss = loss_fn(probe(x_train[index]).squeeze(-1), y_train[index])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        logits = probe(x_val).squeeze(-1).cpu().numpy()
    return {
        "balanced_accuracy": balanced_accuracy(logits, y_val),
        "positive_rate": float(y_val.mean()),
        "n_train": int(train_mask.sum()),
        "n_val": int((~train_mask).sum()),
    }


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("the history-swap audit must run inside a Slurm compute job")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    meta = json.loads((args.cache_dir / "meta.json").read_text())
    stride = int(meta["pixel_stride"])
    episode_len = int(meta["episode_len"])
    latents = np.load(args.cache_dir / "latents.npy", mmap_mode="r")
    actions = np.load(args.cache_dir / "actions.npy", mmap_mode="r")

    labels = latched_labels(args.cache_dir, meta)
    positions = sample_positions(len(latents), meta, args.context_blocks, rng, args.samples)

    # split by episode: neighbouring grid rows are nearly the same state, so a
    # row-wise split would put the answer on both sides of it
    episodes = (positions * stride) // episode_len
    unique = np.unique(episodes)
    held = set(rng.choice(unique, size=max(1, unique.size // 4), replace=False).tolist())
    train_mask = np.array([e not in held for e in episodes])

    features: dict[str, np.ndarray] = {
        "frame_latent": np.asarray(latents[positions], dtype=np.float32)
    }
    for arm in [a for a in args.arms.split(",") if a]:
        payload = torch.load(
            args.checkpoint_root / arm / f"seed{args.seed}" / "ps_jepa.pt",
            map_location="cpu",
            weights_only=False,
        )
        config = PSJEPAConfig(**payload["config"])
        encoder = BeliefEncoder(config)
        encoder.load_state_dict(payload["encoder"])
        encoder = encoder.to(device).eval()
        features[arm] = encode_beliefs(
            encoder,
            latents,
            actions,
            positions,
            config=config,
            context=args.context_blocks,
            stride=stride,
            device=device,
        )

    report: dict = {
        "protocol": PROTOCOL,
        "cache_dir": str(args.cache_dir),
        "samples": int(positions.size),
        "context_blocks": args.context_blocks,
        "held_out_episodes": len(held),
        "train_episodes": int(unique.size - len(held)),
        "feature_dims": {k: int(v.shape[1]) for k, v in features.items()},
        "probes": {},
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }

    for name, matrix in features.items():
        report["probes"][name] = {}
        for target in TARGETS:
            report["probes"][name][target] = train_probe(
                matrix, labels[target][positions], train_mask, device, args, args.seed
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "history_swap_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    header = f"{'features':>18}{'dim':>6}" + "".join(f"{t:>14}" for t in TARGETS)
    print(header)
    print("-" * len(header))
    for name in features:
        row = f"{name:>18}{report['feature_dims'][name]:>6}"
        for target in TARGETS:
            row += f"{report['probes'][name][target]['balanced_accuracy']:>14.4f}"
        print(row)

    print("\npositive rate:  " + "  ".join(
        f"{t}={report['probes']['frame_latent'][t]['positive_rate']:.3f}" for t in TARGETS
    ))

    # the matched contrast: same architecture, same width, recurrence on or off
    if "ps_jepa" in features and "ps_frame" in features:
        print("\nps_jepa - ps_frame (matched, the contrast that counts):")
        for target in TARGETS:
            gap = (
                report["probes"]["ps_jepa"][target]["balanced_accuracy"]
                - report["probes"]["ps_frame"][target]["balanced_accuracy"]
            )
            print(f"  {target:>14}: {gap:+.4f}")
    print(f"\nwrote {args.out_dir / 'history_swap_audit.json'}")


if __name__ == "__main__":
    main()
