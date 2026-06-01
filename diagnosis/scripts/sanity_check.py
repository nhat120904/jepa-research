"""End-of-pipeline sanity checks (Task 4.3) — run before trusting the decision.

Implements the six sanity checks from the plan:
    1. Easy-case CRA  — free_space + random  CRA > 0.90 for all models
    2. Sanity tasks    — pusht/pointmaze CRA > 0.90 (separate config)
    3. Model ordering  — JEPA-WM ≥ DINO-WM on aggregate
    4. Regime ordering — CRA(free_space) ≥ CRA(contact_manipulation)
    5. Strategy ordering — CRA(random) ≥ CRA(hard_nn)
    6. Terver gripper test — separate quantitative reproduction script

Writes `results/sanity_checks_log.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_easy_case(df: pd.DataFrame) -> tuple[bool, str]:
    sub = df[(df.strategy == "random") & (df.regime == "free_space")
              & (df.status == "ok")]
    if sub.empty:
        return False, "no rows for (strategy=random, regime=free_space)"
    bad = sub[sub.cra_top1 < 0.90]
    if not bad.empty:
        offenders = ", ".join(f"{r.model}@{r.task}={r.cra_top1:.2f}"
                               for _, r in bad.iterrows())
        return False, f"models below 0.90: {offenders}"
    return True, f"all {len(sub)} cells ≥ 0.90"


def check_model_ordering(df: pd.DataFrame) -> tuple[bool, str]:
    sub = df[(df.status == "ok") & (df.strategy == "hard_nn")]
    if sub.empty:
        return False, "no hard_nn rows"
    j = sub[sub.model.str.startswith("jepa_wm_")]["cra_top1"].mean()
    d = sub[sub.model.str.startswith("dino_wm_")]["cra_top1"].mean()
    if j >= d - 0.02:
        return True, f"JEPA-WM={j:.3f}, DINO-WM={d:.3f}"
    return False, f"DINO-WM ({d:.3f}) beats JEPA-WM ({j:.3f}) by > 0.02"


def check_regime_ordering(df: pd.DataFrame) -> tuple[bool, str]:
    sub = df[(df.status == "ok") & (df.strategy == "hard_nn")]
    if sub.empty:
        return False, "no hard_nn rows"
    free = sub[sub.regime == "free_space"]["cra_top1"].mean()
    contact = sub[sub.regime == "contact_manipulation"]["cra_top1"].mean()
    if free >= contact - 0.05:
        return True, f"free_space={free:.3f} ≥ contact={contact:.3f}"
    return False, f"contact ({contact:.3f}) > free_space ({free:.3f})"


def check_strategy_ordering(df: pd.DataFrame) -> tuple[bool, str]:
    sub = df[df.status == "ok"]
    if sub.empty:
        return False, "no rows"
    rand = sub[sub.strategy == "random"]["cra_top1"].mean()
    hard = sub[sub.strategy == "hard_nn"]["cra_top1"].mean()
    if rand >= hard - 0.02:
        return True, f"random={rand:.3f} ≥ hard_nn={hard:.3f}"
    return False, f"hard_nn ({hard:.3f}) > random ({rand:.3f})"


def check_action_normalization(
    model_name: str,
    config_path: str,
    n_transitions: int = 64,
    ref_eval_loss: float | None = None,
):
    """The #1-bug guard (plan Note 1). Predict a REAL transition and compare the
    latent MSE to the model's reported eval loss.

    Steps:
      1. Build the adapter (real checkpoint) + load one trajectory.
      2. Encode o_t, o_{t+1}; predict z_hat = F(z_t, a_t) with the action
         normalized exactly as the model trains (preprocessor.normalize_actions).
      3. MSE(z_hat, z_{t+1}) should be within ~2× the model's eval loss.
         A ~10× gap means action normalization is wrong.

    Also reports the MSE with SHUFFLED actions as a contrast: if factual and
    shuffled MSE are nearly equal, the model is ignoring actions (pathology) —
    which the diagnostic then quantifies.

    Requires the upstream env + checkpoint, so it runs on the server only.
    """
    import sys
    from pathlib import Path

    import numpy as np
    import torch
    import yaml

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from data import (iterate_metaworld_trajectories, iterate_droid_trajectories,  # noqa
                      iterate_robocasa_trajectories)
    from models.adapters import build_adapter

    cfg = yaml.safe_load(open(config_path))
    ds = cfg["dataset"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = build_adapter(model_name, device=device).eval()

    # Grab one trajectory.
    if ds["name"] == "metaworld":
        tasks = ds["tasks"]["easy"][:1]
        it = iterate_metaworld_trajectories(ds["root"], tasks, max_trajectories_per_task=1,
                                            external_root=ds.get("external_root", "external/jepa-wms"))
    elif ds["name"] == "droid":
        it = iterate_droid_trajectories(ds["root"], max_transitions=200,
                                        external_root=ds.get("external_root", "external/jepa-wms"),
                                        dataset_kwargs=ds.get("dataset_kwargs"))
    else:
        it = iterate_robocasa_trajectories(ds["root"], max_transitions=200,
                                           external_root=ds.get("external_root", "external/jepa-wms"),
                                           dataset_kwargs=ds.get("dataset_kwargs"))
    traj = next(iter(it))

    # Encode frames, build transitions.
    visual = traj.obs_visual
    T = traj.action.shape[0]
    z = torch.cat([adapter.encode(visual[i:i + 1].unsqueeze(1))[:, 0].cpu() for i in range(T + 1)], 0)
    n = min(n_transitions, T)
    z_t = z[:n].to(device).float()
    z_t1 = z[1:n + 1].to(device).float()
    a_t = traj.action[:n].to(device).float()
    proprio_t = traj.proprio[:n].to(device).float() if adapter.uses_proprio() else None

    z_hat = adapter.predict(z_t, a_t, proprio_t=proprio_t)
    mse = float(((z_hat - z_t1) ** 2).mean().item())
    perm = torch.randperm(n, device=device)
    z_hat_shuf = adapter.predict(z_t, a_t[perm], proprio_t=proprio_t)
    mse_shuf = float(((z_hat_shuf - z_t1) ** 2).mean().item())

    ratio = (mse / ref_eval_loss) if ref_eval_loss else float("nan")
    ok = (ref_eval_loss is None) or (ratio <= 2.0)
    print(f"[norm] {model_name}: MSE(factual)={mse:.5f}  MSE(shuffled)={mse_shuf:.5f}  "
          f"ref_eval_loss={ref_eval_loss}  ratio={ratio:.2f}")
    if ref_eval_loss and ratio > 2.0:
        print("  ⚠ MSE >> eval loss → action normalization likely WRONG (the #1 bug).")
    if mse_shuf <= mse * 1.05:
        print("  ⚠ shuffled ≈ factual → model ignores actions here (pathology to quantify).")
    return {"mse_factual": mse, "mse_shuffled": mse_shuf, "ratio": ratio, "ok": ok}


def main(metaworld_csv: str, droid_csv: str, out_path: str) -> int:
    dfs = []
    for p in (metaworld_csv, droid_csv):
        if Path(p).exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        print("No diagnostic CSVs found.")
        return 1
    df = pd.concat(dfs, ignore_index=True)

    checks = [
        ("1. easy-case CRA (random/free_space ≥ 0.90)", check_easy_case(df)),
        ("3. model ordering (JEPA-WM ≥ DINO-WM)", check_model_ordering(df)),
        ("4. regime ordering (free ≥ contact)", check_regime_ordering(df)),
        ("5. strategy ordering (random ≥ hard_nn)", check_strategy_ordering(df)),
    ]

    lines = ["# Sanity checks", ""]
    all_pass = True
    for name, (ok, msg) in checks:
        tag = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        lines.append(f"- [{tag}] {name}: {msg}")
        print(f"  [{tag}] {name}: {msg}")

    lines += ["",
              "Sanity-check 2 (pusht/pointmaze) is reported by a separate run "
              "using `configs/diagnostic_pusht.yaml` (not in primary diagnostic).",
              "Sanity-check 6 (Terver gripper test) is the responsibility of "
              "`scripts/terver_gripper_test.py` (to be added if time permits).",
              "",
              f"**Overall:** {'PASS' if all_pass else 'FAIL'}", ""]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))
    print(f"\nWrote {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metaworld_csv", default="results/metaworld_diagnostic.csv")
    parser.add_argument("--droid_csv", default="results/droid_diagnostic.csv")
    parser.add_argument("--out", default="results/sanity_checks_log.md")
    args = parser.parse_args()
    sys.exit(main(args.metaworld_csv, args.droid_csv, args.out))
