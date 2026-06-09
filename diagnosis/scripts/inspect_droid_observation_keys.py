"""Introspect what observation channels a raw DROID episode actually contains.

Motivation: the upstream ``DROIDVideoDataset`` builds its 7-dim proprio/state from
ONLY ``cartesian_position`` (6) + ``gripper_position`` (1) — see
``droid_dset.py:259-262`` (HF path) and ``:316-319`` (decord path). It never touches
joint state, velocities, or any force/torque channel. This script answers the
empirical question the diagnostic redesign hinges on:

    "Does the raw DROID episode expose joint_position/velocity or any
     force/torque/wrench signal that we could fuse into the latent (direction D)?"

It does NOT decode video and does NOT load any model — it just opens a handful of
episode HDF5 files and walks the observation group, printing every key with shape +
dtype, then flags the channels relevant to a contact-boundary grounded latent.

Run on the server (where the DROID subset lives):

    python scripts/inspect_droid_observation_keys.py \
        --paths-csv data/droid_subset/droid_paths.csv --n 5

Or point straight at episode files / a directory:

    python scripts/inspect_droid_observation_keys.py --glob 'data/droid_subset/**/*.h5' --n 5
"""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path

import h5py

# Channels that would (or would not) let us resolve a contact boundary in the latent.
WANTED = {
    "force/torque (the sharp boundary signal — direction D's strongest form)": [
        "force", "torque", "wrench", "ft", "external_wrench", "o_f_ext", "ext_force",
        "contact", "tactile",
    ],
    "joint state (richer proprio, NOT force — fallback for D)": [
        "joint_position", "joint_positions", "joint_velocity", "joint_velocities",
        "joint_torque", "joint_torques",  # joint_torque DOES exist on some Franka logs
    ],
    "already-used pose+gripper (the current 7-dim proprio)": [
        "cartesian_position", "cartesian_velocity", "gripper_position",
        "gripper_velocity",
    ],
}


def _collect_keys(h5obj, prefix=""):
    """Recursively walk an h5py group; yield (path, shape, dtype) for datasets."""
    out = []
    for key in h5obj.keys():
        item = h5obj[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Group):
            out.extend(_collect_keys(item, path))
        else:
            try:
                out.append((path, tuple(item.shape), str(item.dtype)))
            except Exception as e:  # noqa: BLE001
                out.append((path, f"<unreadable: {e}>", "?"))
    return out


def _find_observation_group(f: h5py.File):
    """Return (group, label) for the observation group, trying both upstream layouts."""
    # HF layout used by loadvideo_hf
    if "episode_data" in f and "observation" in f["episode_data"]:
        return f["episode_data"]["observation"], "episode_data/observation (HF)"
    # decord layout used by loadvideo_decord
    if "observation" in f:
        obs = f["observation"]
        if "robot_state" in obs:
            return obs["robot_state"], "observation/robot_state (decord)"
        return obs, "observation (decord)"
    return None, None


def inspect_one(path: str) -> None:
    print(f"\n{'=' * 78}\nEPISODE: {path}")
    try:
        f = h5py.File(path, "r")
    except Exception as e:  # noqa: BLE001
        print(f"  !! could not open as h5: {e}")
        return
    with f:
        print("  top-level keys:", list(f.keys()))
        grp, label = _find_observation_group(f)
        if grp is None:
            print("  !! no observation group found in either upstream layout.")
            print("     Full tree (datasets only):")
            for p, shape, dt in _collect_keys(f):
                print(f"       {p:55s} {str(shape):20s} {dt}")
            return

        print(f"  observation group: {label}")
        keys = _collect_keys(grp)
        print(f"  --- all {len(keys)} observation datasets ---")
        for p, shape, dt in keys:
            print(f"    {p:50s} {str(shape):20s} {dt}")

        # Flag the channels we care about.
        flat = {p.lower() for p, _, _ in keys}
        print("  --- verdict for the latent-grounding redesign ---")
        for category, needles in WANTED.items():
            hits = sorted({p for p in flat if any(n in p for n in needles)})
            mark = "FOUND" if hits else "absent"
            print(f"    [{mark:6s}] {category}")
            for h in hits:
                print(f"             - {h}")


def gather_paths(args) -> list[str]:
    paths: list[str] = []
    if args.paths_csv:
        csv_path = Path(args.paths_csv)
        for line in csv_path.read_text().splitlines():
            line = line.strip().strip(",")
            if line and not line.lower().startswith("path"):
                # build_droid_paths.py may emit "<episode_path>,<something>" rows.
                paths.append(line.split(",")[0])
    if args.glob:
        paths.extend(sorted(globlib.glob(args.glob, recursive=True)))
    if args.episode:
        paths.extend(args.episode)
    # de-dup, preserve order
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths-csv", help="droid_paths.csv produced by build_droid_paths.py")
    ap.add_argument("--glob", help="glob for episode .h5 files, e.g. 'data/**/*.h5'")
    ap.add_argument("--episode", nargs="*", help="explicit episode .h5 path(s)")
    ap.add_argument("--n", type=int, default=3, help="how many episodes to inspect")
    args = ap.parse_args()

    paths = gather_paths(args)
    if not paths:
        print("No episode paths found. Pass --paths-csv, --glob, or --episode.")
        return 1
    print(f"Found {len(paths)} candidate episode(s); inspecting first {args.n}.")
    for p in paths[: args.n]:
        inspect_one(p)
    print(
        "\nNOTE: even if joint_* appears, that is richer proprio, NOT contact force. "
        "If the force/torque category is 'absent' across episodes, direction D's "
        "force-grounded form is not possible on this data — see "
        "docs/plans/2026-06-09-action-identifiability-fix-design.md §2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
