"""Strict provenance helpers for the official LeWM Reacher release."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import subprocess
from pathlib import Path


OFFICIAL_REPO = "quentinll/lewm-reacher"
EXPECTED_TARGET = "stable_worldmodel.wm.lewm.LeWM"
EXPECTED_IMAGE_SIZE = 224
EXPECTED_HISTORY = 3
EXPECTED_RAW_ACTION_DIM = 2
EXPECTED_ACTION_BLOCK = 5
EXPECTED_BLOCKED_ACTION_DIM = EXPECTED_RAW_ACTION_DIM * EXPECTED_ACTION_BLOCK


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LeWMReacherProvenance:
    implementation: str
    checkpoint_repo: str
    checkpoint: str
    checkpoint_sha256: str
    config: str
    config_sha256: str
    stable_worldmodel_source: str
    source_commit: str
    checkpoint_revision: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def validate_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    checks = {
        "target": config.get("_target_"),
        "image_size": config.get("encoder", {}).get("image_size"),
        "history": config.get("predictor", {}).get("num_frames"),
        "blocked_action_dim": config.get("action_encoder", {}).get("input_dim"),
    }
    expected = {
        "target": EXPECTED_TARGET,
        "image_size": EXPECTED_IMAGE_SIZE,
        "history": EXPECTED_HISTORY,
        "blocked_action_dim": EXPECTED_BLOCKED_ACTION_DIM,
    }
    if checks != expected:
        raise ValueError(f"official Reacher config mismatch: got {checks}, expected {expected}")
    return config


def build_provenance(
    stable_worldmodel_source: Path,
    checkpoint_dir: Path,
    *,
    checkpoint_revision: str | None = None,
) -> LeWMReacherProvenance:
    source = stable_worldmodel_source.resolve()
    checkpoint = (checkpoint_dir / "weights.pt").resolve()
    config = (checkpoint_dir / "config.json").resolve()
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(f"incomplete official checkpoint directory: {checkpoint_dir}")
    validate_config(config)
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError(f"stable-worldmodel checkout is dirty: {source}")
    return LeWMReacherProvenance(
        implementation="lucas-maes/le-wm:LeWM",
        checkpoint_repo=OFFICIAL_REPO,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        config=str(config),
        config_sha256=sha256(config),
        stable_worldmodel_source=str(source),
        source_commit=commit,
        checkpoint_revision=checkpoint_revision,
    )


__all__ = [
    "EXPECTED_ACTION_BLOCK",
    "EXPECTED_HISTORY",
    "EXPECTED_IMAGE_SIZE",
    "EXPECTED_RAW_ACTION_DIM",
    "LeWMReacherProvenance",
    "OFFICIAL_REPO",
    "build_provenance",
    "validate_config",
]
