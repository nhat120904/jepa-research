"""Core of the boundary diagnostic (design 2026-06-09 §3) — importable & testable.

Produces a **Boundary Blindness per regime** table on frozen baselines, the gate
that proves the contact-boundary gap is real before any fix is trained. Mirrors
``05_run_diagnostic.py``: streams one (task, regime) cell at a time (only scalar
per-transition sensitivities are retained, so RAM stays bounded), then standardises
``S_true``/``S_model`` over the whole model population and bootstraps ``BB`` per cell.

The neighbourhood for every anchor is drawn from the same shared hard-negative
pool the samplers use; ``S_true`` is the spread of the pool neighbours' **true
outcome** (object displacement on Metaworld, ``‖Δz‖`` proxy elsewhere) and
``S_model`` the spread of the model's ``F(z_t, a_neighbour)`` predictions.

This module deliberately does *not* touch ``05``'s output; it writes a sibling
``{dataset}_boundary.csv`` so the core decision pipeline carries zero regression
risk. Only ``materialize_records`` gained a backward-compatible ``want_state`` flag.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from data.latent_cache import REGIME_TO_ID, ID_TO_REGIME  # noqa: E402
from metrics import (  # noqa: E402
    boundary_sensitivities_per_transition,
    boundary_blindness,
    bootstrap_ci,
)
from models.adapters import build_adapter  # noqa: E402
from stratification import (  # noqa: E402
    state_neighbours,
    boundary_score_per_transition,
    calibrate_boundary_threshold,
    boundary_mask,
)
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def _load_runner_helpers():
    """Reuse 05's cache-streaming helpers without importing it as ``05.*``."""
    spec = importlib.util.spec_from_file_location(
        "_run_diagnostic_helpers", ROOT / "scripts" / "05_run_diagnostic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# True outcome (the bifurcation signal)
# ---------------------------------------------------------------------------

def compute_true_outcome(
    dataset_name: str,
    z_t: torch.Tensor,
    z_t1: torch.Tensor,
    state_t: Optional[torch.Tensor] = None,
    state_t1: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Per-transition true outcome used to detect/measure the boundary.

    Metaworld: **object displacement** ``‖obj_{t+1} − obj_t‖`` from the 39-dim
    state — the only signal that actually labels the grasp bifurcation (and is
    invisible to a vision-only latent). Everywhere else: the ``‖Δz‖`` latent proxy
    (design §3.2 — explicitly weaker; DROID is a transfer check, not the proof).
    """
    if dataset_name == "metaworld" and state_t is not None and state_t1 is not None:
        obj_t = state_t[:, OBJECT_SLICE]
        obj_t1 = state_t1[:, OBJECT_SLICE]
        return (obj_t1 - obj_t).norm(dim=-1).cpu().numpy()
    diff = (z_t1 - z_t).reshape(z_t.shape[0], -1)
    return diff.norm(dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Per-cell accumulation (adapter-driven; the synthetic-test entry point)
# ---------------------------------------------------------------------------

@torch.no_grad()
def accumulate_cell(
    adapter,
    cell_data: dict,
    pool_z: torch.Tensor,
    pool_a: torch.Tensor,
    pool_outcome: np.ndarray,
    *,
    device,
    similarity_radius: float,
    max_neighbours: int,
    min_neighbours: int = 2,
) -> dict:
    """Per-transition ``(S_true, S_model, boundary_score)`` for one cell.

    Neighbours come from the shared pool (cross-set ``state_neighbours``). Returns
    scalar arrays only — the caller concatenates them across cells for the global
    standardisation, so memory stays flat.
    """
    z_t = cell_data["z_t"]
    a_t = cell_data["a_t"]
    proprio = cell_data.get("proprio_t")

    idx, mask, _valid = state_neighbours(
        z_t, similarity_radius, max_neighbours=max_neighbours,
        min_neighbours=min_neighbours, pool_z=pool_z,
    )
    neigh_a = pool_a[idx]                                   # (B, M, A)
    pool_out_t = torch.as_tensor(pool_outcome, dtype=torch.float32)
    neigh_out = pool_out_t[idx]                             # (B, M)

    boundary_score = boundary_score_per_transition(a_t, neigh_a, neigh_out, mask)

    proprio_dev = proprio.to(device).float() if proprio is not None else None
    s_true, s_model = boundary_sensitivities_per_transition(
        adapter, z_t.to(device).float(), neigh_a, neigh_out, mask, proprio_t=proprio_dev,
    )
    return {
        "s_true": s_true,
        "s_model": s_model,
        "boundary_score": boundary_score,
        "traj_tag": np.asarray(cell_data["traj_tag"]),
    }


# ---------------------------------------------------------------------------
# Finalise: global standardisation + per-cell bootstrap
# ---------------------------------------------------------------------------

def finalize_rows(
    dataset_name: str,
    model_name: str,
    *,
    s_true: np.ndarray,
    s_model: np.ndarray,
    boundary_score: np.ndarray,
    task: np.ndarray,
    regime: np.ndarray,
    traj_tag: np.ndarray,
    boundary_quantile: float = 0.75,
    n_resamples: int = 1000,
    ci: float = 0.95,
    min_boundary: int = 5,
) -> tuple[list[dict], float]:
    """Standardise sensitivities globally, then bootstrap BB per (task, regime).

    Reports ``bb`` over all transitions in the cell and ``bb_boundary`` over the
    boundary-flagged subset (top ``boundary_quantile`` of the boundary score). The
    gate: ``bb_boundary`` elevated in contact-rich regimes vs. free_space.
    """
    bb = boundary_blindness(s_true, s_model)
    threshold = calibrate_boundary_threshold(boundary_score, boundary_quantile)
    bmask = boundary_mask(boundary_score, threshold)

    rows: list[dict] = []
    for tk, rg in sorted(set(zip(task.tolist(), regime.tolist()))):
        sel = (task == tk) & (regime == rg)
        if not sel.any():
            continue
        groups = traj_tag[sel]
        ci_all = bootstrap_ci(bb[sel], n_resamples=n_resamples, ci=ci, seed=1, groups=groups)

        bsel = sel & bmask
        n_b = int(bsel.sum())
        ci_b = (bootstrap_ci(bb[bsel], n_resamples=n_resamples, ci=ci, seed=2,
                             groups=traj_tag[bsel]) if n_b >= min_boundary else None)

        rows.append({
            "dataset": dataset_name, "task": tk, "model": model_name, "regime": rg,
            "n_transitions": int(sel.sum()), "n_boundary": n_b,
            "boundary_threshold": threshold,
            "mean_boundary_score": float(np.nanmean(boundary_score[sel])),
            "bb": ci_all.point, "bb_lo": ci_all.low, "bb_hi": ci_all.high,
            "bb_boundary": ci_b.point if ci_b else float("nan"),
            "bb_boundary_lo": ci_b.low if ci_b else float("nan"),
            "bb_boundary_hi": ci_b.high if ci_b else float("nan"),
            "status": "ok",
        })
    return rows, threshold


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def main(config_path: str) -> int:
    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(config_path))
    helpers = _load_runner_helpers()

    only_model = os.environ.get("CAI_JEPA_ONLY_MODEL")
    models = [m for m in cfg["models"] if (not only_model or m == only_model)]

    dataset_name = cfg["dataset"]["name"]
    action_dim = cfg["dataset"]["action_dim"]
    cache_root = cfg["latent_cache"]["root"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available()
                          or cfg["eval"]["device"] == "cpu" else "cpu")
    min_n = cfg["eval"]["min_transitions_per_cell"]
    similarity_radius = cfg["hard_nn"]["similarity_radius"]
    pool_size = cfg["hard_nn"]["pool_size"]
    bcfg = cfg.get("boundary", {})
    max_neighbours = int(bcfg.get("max_neighbours", cfg["cra"]["K"]))
    boundary_quantile = float(bcfg.get("quantile", 0.75))
    n_resamples = cfg["bootstrap"]["n_resamples"]
    ci = cfg["bootstrap"]["ci"]

    all_rows: list[dict] = []
    for model_name in models:
        cache_path = latent_cache_path(cache_root, model_name, dataset_name)
        if not cache_path.exists():
            print(f"[skip] {cache_path} missing")
            continue
        print(f"\n=== Boundary diagnostic: {model_name} on {dataset_name} ===", flush=True)
        adapter = build_adapter(model_name, device=str(device)).eval()
        step = adapter.frames_per_step
        want_state = dataset_name == "metaworld"

        regime_by_traj = read_regimes(cache_path)
        with LatentCache(cache_path, mode="r") as cache:
            per_task = dataset_name == "metaworld"
            records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=per_task)
            if not records:
                print(f"  [warn] no transitions for {cache_path.name}", flush=True)
                continue

            rng = np.random.default_rng(0)
            pool_idx = rng.choice(np.arange(len(records)),
                                  size=min(pool_size, len(records)), replace=False)
            pool_records = [records[int(i)] for i in pool_idx]
            pool_data = helpers.materialize_records(cache, pool_records, step,
                                                    want_proprio=False, want_state=want_state)
            pool_outcome = compute_true_outcome(
                dataset_name, pool_data["z_t"], pool_data["z_t1"],
                pool_data.get("state_t"), pool_data.get("state_t1"),
            )
            pool_z, pool_a = pool_data["z_t"], pool_data["a_t"]
            print(f"  pool={len(pool_records)} transitions; max_neighbours={max_neighbours}", flush=True)

            acc: dict[str, list] = {"s_true": [], "s_model": [], "boundary_score": [],
                                    "traj_tag": [], "task": [], "regime": []}
            tasks = sorted({r["task"] for r in records})
            for regime_name in cfg["regimes"]:
                regime_id = REGIME_TO_ID[regime_name]
                for task in tasks:
                    cell_records = [r for r in records
                                    if r["regime"] == regime_id and r["task"] == task]
                    if len(cell_records) < min_n:
                        continue
                    cell_data = helpers.materialize_records(
                        cache, cell_records, step,
                        want_proprio=adapter.uses_proprio(), want_state=False,
                    )
                    out = accumulate_cell(
                        adapter, cell_data, pool_z, pool_a, pool_outcome, device=device,
                        similarity_radius=similarity_radius, max_neighbours=max_neighbours,
                    )
                    n = len(out["s_true"])
                    acc["s_true"].append(out["s_true"])
                    acc["s_model"].append(out["s_model"])
                    acc["boundary_score"].append(out["boundary_score"])
                    acc["traj_tag"].append(out["traj_tag"])
                    acc["task"].append(np.array([task] * n))
                    acc["regime"].append(np.array([regime_name] * n))
                    del cell_data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not acc["s_true"]:
                print("  [warn] no cells met min_transitions_per_cell", flush=True)
                del adapter
                continue

            rows, thr = finalize_rows(
                dataset_name, model_name,
                s_true=np.concatenate(acc["s_true"]),
                s_model=np.concatenate(acc["s_model"]),
                boundary_score=np.concatenate(acc["boundary_score"]),
                task=np.concatenate(acc["task"]),
                regime=np.concatenate(acc["regime"]),
                traj_tag=np.concatenate(acc["traj_tag"]),
                boundary_quantile=boundary_quantile, n_resamples=n_resamples, ci=ci,
            )
            all_rows.extend(rows)
            for r in rows:
                print(f"  {r['task']:22s} {r['regime']:20s} n={r['n_transitions']:5d} "
                      f"BB={r['bb']:+.3f} BB_boundary={r['bb_boundary']:+.3f} "
                      f"(n_b={r['n_boundary']})", flush=True)

        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import pandas as pd
    out_path = Path(os.environ.get(
        "CAI_JEPA_BOUNDARY_CSV",
        Path(cfg["output"]["csv"]).with_name(f"{dataset_name}_boundary.csv"),
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    raise SystemExit(main(args.config))
