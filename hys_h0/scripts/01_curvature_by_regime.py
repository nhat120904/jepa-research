"""
HyS-JEPA pre-gate: is the frozen latent trajectory straight WITHIN a dynamical mode
and kinked AT mode boundaries?

The HyS-JEPA premise is that global temporal straightening is miscalibrated because it
straightens across contact-mode switches. That premise is only true if the frozen latent
actually kinks at switches more than it does inside a mode. This measures that directly on
the existing latent cache -- no training, no new labels.

Curvature, exactly the ICML-2026 temporal-straightening form:

    c_t = 1 - cos( z_{t+1} - z_t ,  z_{t+2} - z_{t+1} )

Reported per regime and split by whether a contact-mode switch occurs at t.

Two controls that decide whether any difference is real:
  * chance level -- cosine between displacement pairs drawn from DIFFERENT trajectories.
    In ~1e5 dimensions random directions are near-orthogonal, so c -> 1. Any within-mode
    straightness must be read against this, not against 0.
  * effect conditioning -- curvature is undefined/noisy when the latent barely moves, so
    everything is also reported restricted to transitions with ||dz|| above the median.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent / "diagnosis"
sys.path.insert(0, str(REPO))

from data.latent_cache import LatentCache  # noqa: E402
from stratification import REGIMES  # noqa: E402


def traj_curvature(z: np.ndarray):
    """z [T, ...] -> (curvature[T-2], dz_norm[T-1])."""
    zf = z.reshape(len(z), -1).astype(np.float64)
    dz = np.diff(zf, axis=0)                      # [T-1, D]
    n = np.linalg.norm(dz, axis=1)                # [T-1]
    a, b = dz[:-1], dz[1:]
    na, nb = n[:-1], n[1:]
    denom = na * nb
    cos = np.zeros(len(a))
    ok = denom > 0
    cos[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / denom[ok]
    return 1.0 - cos, n


def clustered_ci(values, traj_ids, n_boot=2000, seed=0, stat=np.mean):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    traj_ids = np.asarray(traj_ids)
    uniq = np.unique(traj_ids)
    by = {t: values[traj_ids == t] for t in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        cat = np.concatenate([by[t] for t in pick])
        if len(cat):
            out.append(stat(cat))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_diff_ci(v_a, id_a, v_b, id_b, n_boot=2000, seed=0):
    """Bootstrap mean(A) - mean(B), resampling trajectories jointly."""
    rng = np.random.default_rng(seed)
    v_a, id_a, v_b, id_b = map(np.asarray, (v_a, id_a, v_b, id_b))
    uniq = np.unique(np.concatenate([id_a, id_b]))
    A = {t: v_a[id_a == t] for t in uniq}
    B = {t: v_b[id_b == t] for t in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ca = np.concatenate([A[t] for t in pick]) if any(len(A[t]) for t in pick) else np.array([])
        cb = np.concatenate([B[t] for t in pick]) if any(len(B[t]) for t in pick) else np.array([])
        if len(ca) and len(cb):
            out.append(ca.mean() - cb.mean())
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--regimes", default=None)
    p.add_argument("--tasks", nargs="*", default=None, help="task prefixes; default all")
    p.add_argument("--max-traj-per-task", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    regimes_path = args.regimes or (args.cache + ".regimes.json")
    regime_map = json.loads(Path(regimes_path).read_text())

    all_pairs = {"c": [], "n": []}
    rows = defaultdict(list)          # key -> list of curvature values
    rowids = defaultdict(list)        # key -> list of traj ids
    dz_by_regime = defaultdict(list)
    seen = defaultdict(int)
    sample_dz = []                    # for the chance-level control

    with LatentCache(args.cache, "r") as c:
        tids = c.trajectory_ids()
        for tid in tids:
            task = str(tid).split("/")[0]
            if args.tasks and task not in args.tasks:
                continue
            if args.max_traj_per_task and seen[task] >= args.max_traj_per_task:
                continue
            seen[task] += 1

            traj = c.read_trajectory(tid)
            z = traj["z"]
            reg = np.asarray(regime_map.get(str(tid), []), dtype=int)
            curv, dzn = traj_curvature(z)
            st = traj.get("state")
            curv_obj = curv_ee = None
            if st is not None and st.shape[-1] >= 7:
                curv_obj, _ = traj_curvature(np.asarray(st)[:, 4:7])   # object xyz
                curv_ee, _ = traj_curvature(np.asarray(st)[:, 0:3])    # hand xyz
            T = len(curv)                       # transitions t = 0..T-1 usable
            if T < 2 or len(reg) < T + 1:
                continue

            # one random displacement per trajectory for the chance control
            if len(dzn) > 2:
                zf = z.reshape(len(z), -1).astype(np.float64)
                k = np.random.default_rng(abs(hash(tid)) % 2**31).integers(0, len(zf) - 1)
                sample_dz.append(zf[k + 1] - zf[k])

            med = np.median(dzn[dzn > 0]) if np.any(dzn > 0) else 0.0
            for t in range(T):
                r_t, r_t1 = int(reg[t]), int(reg[t + 1])
                switch = r_t != r_t1
                big = dzn[t] >= med and dzn[t + 1] >= med
                for scope in ([""] if not big else ["", "|big"]):
                    rows[f"ALL{scope}"].append(curv[t]); rowids[f"ALL{scope}"].append(tid)
                    key = "switch" if switch else "within"
                    rows[f"{key}{scope}"].append(curv[t]); rowids[f"{key}{scope}"].append(tid)
                    rows[f"regime:{REGIMES[r_t]}{scope}"].append(curv[t])
                    rowids[f"regime:{REGIMES[r_t]}{scope}"].append(tid)
                    rows[f"task:{task}|{key}{scope}"].append(curv[t])
                    rowids[f"task:{task}|{key}{scope}"].append(tid)
                    if curv_obj is not None and t < len(curv_obj):
                        rows[f"STATEOBJ:{key}{scope}"].append(curv_obj[t])
                        rowids[f"STATEOBJ:{key}{scope}"].append(tid)
                        rows[f"STATEEE:{key}{scope}"].append(curv_ee[t])
                        rowids[f"STATEEE:{key}{scope}"].append(tid)
                dz_by_regime[REGIMES[r_t]].append(dzn[t])
                all_pairs["c"].append(curv[t]); all_pairs["n"].append(min(dzn[t], dzn[t + 1]))

    # chance level: cosine between displacements from different trajectories
    chance = None
    if len(sample_dz) > 4:
        S = np.stack(sample_dz)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        rng = np.random.default_rng(0)
        i = rng.integers(0, len(S), 4000)
        j = rng.integers(0, len(S), 4000)
        m = i != j
        chance = float((1.0 - np.einsum("ij,ij->i", S[i[m]], S[j[m]])).mean())

    res = {"cache": args.cache, "n_traj": int(sum(seen.values())),
           "tasks": dict(seen), "chance_curvature_cross_traj": chance, "groups": {}}

    for k in sorted(rows):
        v = np.asarray(rows[k])
        if len(v) < 30:
            continue
        lo, hi = clustered_ci(v, rowids[k])
        res["groups"][k] = {"n": int(len(v)), "mean": float(v.mean()),
                            "median": float(np.median(v)), "ci": [lo, hi]}

    # noise test: if curvature > chance is encoder jitter, it must fall as ||dz|| grows
    if all_pairs["c"]:
        cc = np.asarray(all_pairs["c"]); nn = np.asarray(all_pairs["n"])
        qs = np.quantile(nn, np.linspace(0, 1, 11))
        dec = []
        for i in range(10):
            m = (nn >= qs[i]) & (nn <= qs[i + 1] if i == 9 else nn < qs[i + 1])
            if m.sum() > 20:
                dec.append({"decile": i + 1, "dz_lo": float(qs[i]), "dz_hi": float(qs[i + 1]),
                            "n": int(m.sum()), "mean_curv": float(cc[m].mean())})
        res["curvature_by_dz_decile"] = dec

    res["dz_norm_by_regime"] = {r: {"n": len(v), "mean": float(np.mean(v)),
                                    "median": float(np.median(v))}
                                for r, v in dz_by_regime.items() if v}

    # THE pre-gate test: switch vs within (latent, and physical-state controls)
    for prefix in ["", "STATEOBJ:", "STATEEE:"]:
      for scope in ["", "|big"]:
        a, b = f"{prefix}switch{scope}", f"{prefix}within{scope}"
        if a in rows and b in rows:
            lo, hi = paired_diff_ci(rows[a], rowids[a], rows[b], rowids[b])
            d = float(np.mean(rows[a]) - np.mean(rows[b]))
            res[f"{prefix}switch_minus_within{scope}"] = {
                "delta": d, "ci": [lo, hi], "ci_excludes_zero": bool(lo * hi > 0),
                "interpretation": "positive = latent kinks MORE at contact-mode switches",
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))

    print(f"cache={Path(args.cache).name}  n_traj={res['n_traj']}")
    print(f"chance curvature (cross-trajectory displacements) = {chance:.4f}"
          if chance else "chance n/a")
    for k in ["ALL", "within", "switch", "ALL|big", "within|big", "switch|big"]:
        if k in res["groups"]:
            g = res["groups"][k]
            print(f"  {k:14s} n={g['n']:7d} mean={g['mean']:.4f} CI[{g['ci'][0]:.4f},{g['ci'][1]:.4f}]")
    for k in sorted(res["groups"]):
        if k.startswith("regime:") and "|big" not in k:
            g = res["groups"][k]
            print(f"  {k:34s} n={g['n']:7d} mean={g['mean']:.4f}")
    for k in sorted(res["groups"]):
        if k.startswith(("STATEOBJ:", "STATEEE:")):
            g = res["groups"][k]
            print(f"  {k:28s} n={g['n']:7d} mean={g['mean']:.4f} CI[{g['ci'][0]:.4f},{g['ci'][1]:.4f}]")
    for key in [k for k in res if k.endswith(("switch_minus_within", "switch_minus_within|big"))]:
        if True:
            e = res[key]
            print(f"  {key}: delta={e['delta']:+.4f} CI[{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}] "
                  f"excl0={e['ci_excludes_zero']}")
    if "curvature_by_dz_decile" in res:
        print("  curvature by ||dz|| decile (noise test):")
        for d in res["curvature_by_dz_decile"]:
            print(f"    d{d['decile']:2d} dz>={d['dz_lo']:8.2f} n={d['n']:6d} curv={d['mean_curv']:.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
