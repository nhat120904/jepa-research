"""Auxiliary PushT labels for CTA v2.1, aligned with the v2 feature caches (CPU only, no images).

For every feature split (Round-0 train/dev, on-policy train/dev, Round-4 standard/perturbed) this writes, in the
split's own row order (checked against its meta.npz root/decision):
  end_pose  (N, K, 3)  simulated end block pose of every candidate (phys8[..., 4:7]); training labels only
  native    (N, K)     native coverage at the end of every candidate
  success   (N, K)     the candidate reached the success threshold (native coverage > 0.95; gym-pusht terminates there)
  hind_frame (H,), hind_pose (H, 3): hindsight goals = decision frames of the split whose block pose is known (the
                       executed candidate's end pose of the previous decision of the same episode)
Output: <out>/<split>.npz. Compute node (sbatch) only by the repo's rules, though it reads small arrays only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")
# name: (raw collection, root range, feature split (None: Round-4 encode), bank index, executed key, native-coverage key)
# Raw "cov8" is native coverage in Round 0 / Round 4 but the geometry label in the on-policy collection.
SPLITS = {
    "r0": (TI / "cta_collect_54489", (30250, 31049), TI / "cta_geometry_e2e_55018/features/train", None, "p0", "cov8"),
    "r0dev": (TI / "cta_collect_54489", (2000, 2099), TI / "cta_geometry_e2e_55018/features/dev", None, "p0", "cov8"),
    "onpolicy": (TI / "cta_onpolicy8_collect_55346", (32250, 32649), TI / "cta_onpolicy8_encode_55347/features/train", None,
                 "chosen", "native_cov8"),
    "onpolicydev": (TI / "cta_onpolicy8_collect_55346", (32650, 32699), TI / "cta_onpolicy8_encode_55347/features/dev",
                    None, "chosen", "native_cov8"),
    "r4std": (TI / "cta_plus_collect_55147", (31050, 32249), None, 0, "executed", "cov8"),
    "r4pert": (TI / "cta_plus_collect_55147", (31050, 32249), None, 1, "executed", "cov8"),
}
SUCCESS = 0.95


def shards(collect, lo, hi):
    return sorted(p for p in Path(collect).glob("shard_*.npz") if lo <= int(p.stem.split("_")[1]) <= hi)


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, (collect, (lo, hi), feat, bank, exe, native_key) in SPLITS.items():
        rows = {k: [] for k in ("root", "decision", "end_pose", "native", "executed", "exec_pose_bank")}
        for f in shards(collect, lo, hi):
            with np.load(f) as z:
                phys, cov = z["phys8"], z[native_key]
                executed_bank = phys[:, 0] if bank is not None else phys          # Round 4 executes the standard bank
                if bank is not None:                                            # Round-4: (D, 2 banks, K, ...)
                    phys, cov = phys[:, bank], cov[:, bank]
                rows["root"].append(z["root"])
                rows["decision"].append(z["decision"])
                rows["end_pose"].append(phys[..., 4:7])
                rows["native"].append(cov)
                rows["executed"].append(np.zeros(len(z["root"]), int) if exe == "p0" else z[exe])
                rows["exec_pose_bank"].append(executed_bank[..., 4:7])
        r = {k: np.concatenate(v) for k, v in rows.items()}
        meta_path = (feat if feat is not None else a.r4 / name) / "meta.npz"
        meta = np.load(meta_path)
        if not (np.array_equal(meta["root"], r["root"]) and np.array_equal(meta["decision"], r["decision"])):
            raise ValueError(f"{name}: raw rows do not align with {meta_path}")
        if "native_cov8" in meta.files and np.abs(meta["native_cov8"] - r["native"]).max() > 1e-6:
            raise ValueError(f"{name}: native coverage differs from the feature cache")
        # hindsight goals: frame of decision i+1 in the same episode, block pose = end pose of decision i's executed chunk
        # rows are in lockstep order (all roots at decision d, then d+1): find decision d+1 of the same root by key
        index = {(int(rt), int(dc)): k for k, (rt, dc) in enumerate(zip(r["root"], r["decision"]))}
        pairs = [(k, index[(int(rt), int(dc) + 1)]) for k, (rt, dc) in enumerate(zip(r["root"], r["decision"]))
                 if (int(rt), int(dc) + 1) in index]
        i, nxt = (np.array(x, int) for x in zip(*pairs)) if pairs else (np.zeros(0, int), np.zeros(0, int))
        pose = r["exec_pose_bank"][i, r["executed"][i]]
        still = np.ptp(r["end_pose"][nxt], axis=1).max(-1) < 1e-6          # checkable: no candidate moves the block
        err = np.abs(r["end_pose"][nxt, 0] - pose).max(-1)
        bad = still & (err > 1e-3)
        i, nxt, pose = i[~bad], nxt[~bad], pose[~bad]
        success = r["native"] > SUCCESS
        np.savez(a.out / f"{name}.npz", end_pose=r["end_pose"].astype(np.float32), native=r["native"].astype(np.float32),
                 success=success, hind_frame=nxt, hind_pose=pose.astype(np.float32))
        report[name] = {"rows": int(len(r["root"])), "hindsight": int(len(i)), "checkable": int(still.sum()),
                        "max_err_checkable": float(err[still].max()) if still.any() else None,
                        "dropped_inconsistent": int(bad.sum()), "success_candidates": float(success.mean()),
                        "mixed_success_banks": int((success.any(1) & ~success.all(1)).sum())}
        print(name, report[name], flush=True)
    (a.out / "aux_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--r4", type=Path, required=True, help="scripts/cta_encode_r4_full.py output (Round-4 meta)")
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
