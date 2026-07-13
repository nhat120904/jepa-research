"""Deterministic, trajectory-level train/validation/test split manifests.

The manifest is deliberately JSON-only and contains trajectory identifiers, not
transition indices.  This makes it possible for every downstream evaluation to
enforce the same held-out split after rebuilding transition records from a cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")


def _canonical_payload(manifest: Mapping) -> bytes:
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def manifest_sha256(manifest: Mapping) -> str:
    """Hash the deterministic manifest payload (excluding its hash field)."""
    return hashlib.sha256(_canonical_payload(manifest)).hexdigest()


def validate_manifest(manifest: Mapping) -> dict:
    """Validate and normalize a trajectory split manifest.

    Raises ``ValueError`` for overlap, incomplete coverage, or a stale digest.
    A normalized plain ``dict`` is returned so callers can safely serialize it.
    """
    m = dict(manifest)
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported split manifest schema: {m.get('schema_version')!r}")
    if m.get("split_unit") != "trajectory":
        raise ValueError("split manifest must use split_unit='trajectory'")
    raw_splits = m.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(SPLIT_NAMES):
        raise ValueError(f"split manifest must contain exactly {SPLIT_NAMES}")
    splits = {name: [str(t) for t in raw_splits[name]] for name in SPLIT_NAMES}
    sets = {name: set(tids) for name, tids in splits.items()}
    if any(len(sets[name]) != len(splits[name]) for name in SPLIT_NAMES):
        raise ValueError("duplicate trajectory id within a split")
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"trajectory split overlap ({left}/{right}): {sorted(overlap)[:3]}")
    all_tids = sorted(set().union(*sets.values()))
    if int(m.get("trajectory_count", -1)) != len(all_tids):
        raise ValueError("trajectory_count does not match split contents")
    expected_tid_hash = hashlib.sha256("\n".join(all_tids).encode("utf-8")).hexdigest()
    if m.get("trajectory_sha256") != expected_tid_hash:
        raise ValueError("trajectory_sha256 does not match split contents")
    m["splits"] = splits
    expected_manifest_hash = manifest_sha256(m)
    if m.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("manifest_sha256 does not match manifest payload")
    return m


def build_trajectory_manifest(
    trajectory_ids: Iterable[str],
    *,
    seed: int,
    dataset: str,
    model: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict:
    """Build a deterministic trajectory-level split (default 70/15/15).

    The training fraction is the remainder after validation and test.  Validation
    and test are guaranteed non-empty, so at least three trajectories are required.
    """
    tids = sorted({str(t) for t in trajectory_ids})
    if len(tids) < 3:
        raise ValueError("at least three trajectories are required for train/val/test")
    if not (0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0
            and val_frac + test_frac < 1.0):
        raise ValueError("val_frac and test_frac must be positive and sum to less than 1")
    shuffled = list(tids)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    n_val = max(1, int(np.floor(len(tids) * val_frac)))
    n_test = max(1, int(np.floor(len(tids) * test_frac)))
    if n_val + n_test >= len(tids):
        raise ValueError("split fractions leave no training trajectories")
    n_train = len(tids) - n_val - n_test
    splits = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }
    fractions = {name: len(splits[name]) / len(tids) for name in SPLIT_NAMES}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_unit": "trajectory",
        "dataset": str(dataset),
        "model": str(model),
        "seed": int(seed),
        "requested_fractions": {
            "train": float(1.0 - val_frac - test_frac),
            "val": float(val_frac),
            "test": float(test_frac),
        },
        "realized_fractions": fractions,
        "trajectory_count": len(tids),
        "trajectory_sha256": hashlib.sha256("\n".join(tids).encode("utf-8")).hexdigest(),
        "splits": splits,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return validate_manifest(manifest)


def write_manifest_once(path: str | Path, manifest: Mapping) -> Path:
    """Create an immutable manifest, accepting an identical existing file.

    ``os.link`` publishes the fully-written temporary file atomically.  This also
    makes shared manifests safe when several Slurm array tasks start together.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = validate_manifest(manifest)
    encoded = (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "xb") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            existing = load_manifest(path)
            if existing["manifest_sha256"] != normalized["manifest_sha256"]:
                raise ValueError(
                    f"refusing to overwrite non-matching immutable split manifest: {path}"
                )
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_manifest(path: str | Path) -> dict:
    with open(path) as f:
        return validate_manifest(json.load(f))


def filter_records(records: Sequence[Mapping], manifest: Mapping, split: str) -> list:
    """Return records whose ``tid`` belongs to ``split`` and verify cache coverage."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; choose from {SPLIT_NAMES}")
    m = validate_manifest(manifest)
    record_tids = {str(r["tid"]) for r in records}
    manifest_tids = set().union(*(set(m["splits"][s]) for s in SPLIT_NAMES))
    missing = manifest_tids - record_tids
    extra = record_tids - manifest_tids
    if missing or extra:
        raise ValueError(
            "split manifest/cache trajectory mismatch: "
            f"missing_in_cache={sorted(missing)[:3]} extra_in_cache={sorted(extra)[:3]}"
        )
    keep = set(m["splits"][split])
    return [r for r in records if str(r["tid"]) in keep]
