"""Held-out trajectory validation of the complete deployed stateprobe cost.

The object and end-effector probes were trained with a deterministic 90/10
trajectory split.  This script reconstructs that split and evaluates the exact
scalar composition used by the planner,

    ||g_obj(z) - g_obj(z_goal)|| + w_hand ||g_ee(z) - g_obj(z)||,

against the same expression in simulator state.  Goal frames are the final
frames of the held-out expert trajectories.  The script never loads the world
model; it reads the large latent cache and therefore must run on a compute node.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from models.probes import load_probe  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
from stratification.metaworld_regimes import EE_SLICE, OBJECT_SLICE  # noqa: E402


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    lr = pd.Series(left).rank(method="average").to_numpy(float)
    rr = pd.Series(right).rank(method="average").to_numpy(float)
    return float(np.corrcoef(lr, rr)[0, 1])


def top_fraction_recall(left: np.ndarray, right: np.ndarray, frac: float) -> float:
    k = max(1, int(np.ceil(frac * len(left))))
    li = set(np.argsort(left, kind="mergesort")[:k].tolist())
    ri = set(np.argsort(right, kind="mergesort")[:k].tolist())
    return float(len(li & ri) / k)


def bootstrap_trajectory_mean(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    if len(values) == 1:
        return point, point, point
    draws = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


@torch.no_grad()
def decode(probe, z: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(z, dtype=np.float32))
    pieces = []
    for lo in range(0, len(tensor), batch_size):
        pieces.append(probe(tensor[lo : lo + batch_size].to(device)).cpu().numpy())
    return np.concatenate(pieces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--object-probe", required=True)
    parser.add_argument("--ee-probe", required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--w-hand", type=float, default=0.5)
    parser.add_argument("--topk-frac", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=79001)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache_path = latent_cache_path(
        cfg["latent_cache"]["root"], args.model, cfg["dataset"]["name"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    object_probe, object_meta = load_probe(args.object_probe, device)
    ee_probe, ee_meta = load_probe(args.ee_probe, device)
    for kind, metadata in (("object", object_meta), ("ee", ee_meta)):
        if metadata.get("model") != args.model:
            raise ValueError(
                f"{kind} probe model {metadata.get('model')!r} != {args.model!r}"
            )
    helpers = _load_runner_helpers()
    regime_by_traj = read_regimes(cache_path)

    rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(
            cache, regime_by_traj, args.step, per_task=True
        )
        tids = sorted({record["tid"] for record in records})
        split_rng = np.random.default_rng(args.split_seed)
        split_rng.shuffle(tids)
        val_tids = set(tids[: max(1, int(len(tids) * args.val_frac))])
        val_records = [record for record in records if record["tid"] in val_tids]
        by_tid: dict[str, list[dict]] = {}
        for record in val_records:
            # Reach uses a different deployed true-state cost (hand-to-goal).
            if record["task"] != "mw-reach":
                by_tid.setdefault(record["tid"], []).append(record)

        for tid, selected in by_tid.items():
            selected.sort(key=lambda record: int(record["idx0"]))
            group = cache.h5["trajectories"][LatentCache._safe_key(tid)]
            endpoints = np.asarray(
                [int(record["idx0"]) + args.step for record in selected], dtype=np.int64
            )
            z = np.asarray(group["z"][endpoints], dtype=np.float32)
            state = np.asarray(group["state"][endpoints], dtype=np.float32)
            z_goal = np.asarray(group["z"][-1:], dtype=np.float32)
            state_goal = np.asarray(group["state"][-1], dtype=np.float32)

            decoded_obj = decode(object_probe, z, device, args.batch_size)
            decoded_ee = decode(ee_probe, z, device, args.batch_size)
            decoded_goal_obj = decode(object_probe, z_goal, device, 1)[0]
            true_obj = state[:, OBJECT_SLICE]
            true_ee = state[:, EE_SLICE]
            true_goal_obj = state_goal[OBJECT_SLICE]

            proxy_obj = np.linalg.norm(decoded_obj - decoded_goal_obj[None], axis=1)
            proxy_hand = np.linalg.norm(decoded_ee - decoded_obj, axis=1)
            true_obj_term = np.linalg.norm(true_obj - true_goal_obj[None], axis=1)
            true_hand_term = np.linalg.norm(true_ee - true_obj, axis=1)
            proxy_cost = proxy_obj + args.w_hand * proxy_hand
            true_cost = true_obj_term + args.w_hand * true_hand_term
            task = str(selected[0]["task"])

            for index, record in enumerate(selected):
                rows.append({
                    "model": args.model, "task": task, "tid": tid,
                    "idx0": int(record["idx0"]),
                    "proxy_cost_m": float(proxy_cost[index]),
                    "true_cost_m": float(true_cost[index]),
                    "cost_error_m": float(proxy_cost[index] - true_cost[index]),
                    "abs_cost_error_m": float(abs(proxy_cost[index] - true_cost[index])),
                    "object_term_error_m": float(proxy_obj[index] - true_obj_term[index]),
                    "hand_term_error_m": float(proxy_hand[index] - true_hand_term[index]),
                    "object_decode_error_m": float(
                        np.linalg.norm(decoded_obj[index] - true_obj[index])
                    ),
                    "ee_decode_error_m": float(
                        np.linalg.norm(decoded_ee[index] - true_ee[index])
                    ),
                })
            trajectory_rows.append({
                "model": args.model, "task": task, "tid": tid,
                "n_transition": int(len(selected)),
                "cost_mae_m": float(np.mean(np.abs(proxy_cost - true_cost))),
                "cost_bias_m": float(np.mean(proxy_cost - true_cost)),
                "cost_rmse_m": float(np.sqrt(np.mean((proxy_cost - true_cost) ** 2))),
                "cost_spearman": spearman(proxy_cost, true_cost),
                "reference_top10_recall": top_fraction_recall(
                    proxy_cost, true_cost, args.topk_frac
                ),
            })
            del z, state, decoded_obj, decoded_ee
            gc.collect()

    transitions = pd.DataFrame(rows)
    trajectories = pd.DataFrame(trajectory_rows)
    metric_names = [
        "cost_mae_m", "cost_bias_m", "cost_rmse_m", "cost_spearman",
        "reference_top10_recall",
    ]
    summary_rows = []
    primary_contact = trajectories[
        trajectories["task"].isin(["mw-push", "mw-pick-place"])
    ]
    groups = [
        ("ALL_MANIPULATION", trajectories),
        ("PRIMARY_PUSH_PICK", primary_contact),
    ] + list(trajectories.groupby("task", sort=True))
    for group_index, (task, group) in enumerate(groups):
        row: dict[str, object] = {
            "model": args.model, "task": task,
            "n_trajectory": int(group.tid.nunique()),
            "n_transition": int(group.n_transition.sum()),
        }
        for metric_index, metric in enumerate(metric_names):
            point, lo, hi = bootstrap_trajectory_mean(
                group[metric].to_numpy(float), args.n_bootstrap,
                np.random.default_rng(
                    args.bootstrap_seed + group_index * 101 + metric_index
                ),
            )
            row[metric] = point
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(prefix.with_name(prefix.name + "_transitions.csv.gz"), index=False)
    trajectories.to_csv(prefix.with_name(prefix.name + "_trajectories.csv"), index=False)
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    metadata = pd.DataFrame([{
        "model": args.model, "cache": str(cache_path), "step": args.step,
        "split_seed": args.split_seed, "val_frac": args.val_frac,
        "w_hand": args.w_hand, "object_probe": args.object_probe,
        "ee_probe": args.ee_probe,
        "object_probe_val_mse": object_meta.get("val_mse"),
        "ee_probe_val_mse": ee_meta.get("val_mse"),
    }])
    metadata.to_csv(prefix.with_name(prefix.name + "_metadata.csv"), index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"wrote held-out scalar-cost validation under {prefix.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
