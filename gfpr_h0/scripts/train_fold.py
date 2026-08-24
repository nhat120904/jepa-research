#!/usr/bin/env python3
"""Train one outer GFPR fold and emit episode-held-out selections."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gfpr_h0.core import farthest_point_indices, select_with_gate  # noqa: E402


FEATURE_VIEWS = ("action_only", "proxy_action", "latent_context")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument(
        "--features-root", type=Path, default=REPO / "gfpr_h0/outputs/features_v3"
    )
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--margin-cm", type=float, default=2.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


class RegretMLP(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(128, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


def load_snapshot(root: Path, index: int) -> dict[str, np.ndarray]:
    path = root / str(index) / "features.npz"
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_one(
    x: np.ndarray,
    y: np.ndarray,
    margin_cm: float,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> RegretMLP:
    set_seed(seed)
    model = RegretMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    positives = max(float(np.sum(y >= margin_cm)), 1.0)
    negatives = max(float(np.sum(y < margin_cm)), 1.0)
    pos_weight = torch.tensor(negatives / positives, device=device)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(x_tensor), generator=generator)
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_tensor[idx].to(device, non_blocking=True)
            yb = y_tensor[idx].to(device, non_blocking=True)
            pred = model(xb)
            regression = F.smooth_l1_loss(pred, yb, beta=2.0)
            corrective = (yb >= margin_cm).float()
            classification = F.binary_cross_entropy_with_logits(
                pred - margin_cm, corrective, pos_weight=pos_weight
            )
            loss = regression + 0.25 * classification
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict(model: RegretMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    return model(tensor).detach().float().cpu().numpy()


def outcome(snapshot: dict[str, np.ndarray], selected: int) -> dict[str, float | int]:
    anchor_distance = float(snapshot["anchor_physical_distance_m"])
    anchor_success = int(anchor_distance <= 0.05)
    if selected < 0:
        distance = anchor_distance
        success = anchor_success
    else:
        distance = float(snapshot["candidate_physical_distance_m"][selected])
        success = int(snapshot["candidate_success"][selected])
    gain_cm = (anchor_distance - distance) * 100.0
    return {
        "selected_index": int(selected),
        "physical_distance_m": distance,
        "success": success,
        "gain_cm": gain_cm,
        "corrective": int(gain_cm >= 2.0),
        "switch": int(selected >= 0),
        "harmful_switch": int(selected >= 0 and anchor_success == 1 and success == 0),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("fold training requires a GPU Slurm allocation")
    if args.ensemble_size < 2 or args.epochs <= 0 or args.margin_cm <= 0:
        raise ValueError("invalid training configuration")
    device = torch.device("cuda")
    test_indices = [index for index in range(32) if index % 4 == args.fold]
    train_indices = [index for index in range(32) if index not in test_indices]
    snapshots = {index: load_snapshot(args.features_root, index) for index in range(32)}
    if len({int(snapshots[i]["episode"]) for i in range(32)}) != 32:
        raise RuntimeError("snapshot episodes are not unique")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.out_dir / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    view_predictions: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for view_offset, view in enumerate(FEATURE_VIEWS):
        x_train_raw = np.concatenate([snapshots[i][view] for i in train_indices])
        y_train = np.concatenate([snapshots[i]["physical_gain_cm"] for i in train_indices])
        mean = x_train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x_train_raw.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, 1e-5)
        x_train = (x_train_raw - mean) / std
        ensemble_test: dict[int, list[np.ndarray]] = {i: [] for i in test_indices}
        for member in range(args.ensemble_size):
            seed = 20260819 + args.fold * 1000 + view_offset * 100 + member
            model = fit_one(
                x_train,
                y_train,
                args.margin_cm,
                seed,
                args.epochs,
                args.batch_size,
                device,
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_view": view,
                    "feature_mean": mean,
                    "feature_std": std,
                    "fold": args.fold,
                    "train_snapshots": train_indices,
                    "seed": seed,
                },
                checkpoint_root / f"{view}_member{member}.pt",
            )
            for index in test_indices:
                x_test = (snapshots[index][view] - mean) / std
                ensemble_test[index].append(predict(model, x_test, device))
        view_predictions[view] = {}
        for index in test_indices:
            stacked = np.stack(ensemble_test[index], axis=0)
            view_predictions[view][index] = (
                stacked.mean(axis=0),
                stacked.std(axis=0, ddof=1),
            )

    state_records: list[dict[str, object]] = []
    for index in test_indices:
        snapshot = snapshots[index]
        anchor = outcome(snapshot, -1)
        actions = snapshot["candidate_actions"]
        diverse = farthest_point_indices(actions, snapshot["anchor_action"], 8)
        diverse_best = int(
            diverse[np.argmin(snapshot["candidate_physical_distance_m"][diverse])]
        )
        if float(snapshot["candidate_physical_distance_m"][diverse_best]) >= float(
            snapshot["anchor_physical_distance_m"]
        ):
            diverse_best = -1
        oracle = int(np.argmin(snapshot["candidate_physical_distance_m"]))
        if float(snapshot["candidate_physical_distance_m"][oracle]) >= float(
            snapshot["anchor_physical_distance_m"]
        ):
            oracle = -1
        arms: dict[str, object] = {
            "native": anchor,
            "dino_best": outcome(
                snapshot, int(np.argmin(snapshot["candidate_auxiliary_cost"]))
            ),
            "action_diverse_oracle8": outcome(snapshot, diverse_best),
            "physical_oracle_full": outcome(snapshot, oracle),
        }
        for view in FEATURE_VIEWS:
            mean_gain, std_gain = view_predictions[view][index]
            ungated, gated = select_with_gate(mean_gain, std_gain, args.margin_cm)
            arms[f"{view}_ungated"] = outcome(snapshot, ungated)
            arms[f"{view}_gated"] = outcome(snapshot, gated)
        state_records.append(
            {
                "snapshot_index": index,
                "episode": int(snapshot["episode"]),
                "outer_fold": args.fold,
                "anchor_success": int(anchor["success"]),
                "arms": arms,
            }
        )

    (args.out_dir / "state_records.json").write_text(
        json.dumps(state_records, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "fold": args.fold,
        "train_snapshots": train_indices,
        "test_snapshots": test_indices,
        "feature_views": list(FEATURE_VIEWS),
        "ensemble_size": args.ensemble_size,
        "epochs": args.epochs,
        "margin_cm": args.margin_cm,
        "records": len(state_records),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
