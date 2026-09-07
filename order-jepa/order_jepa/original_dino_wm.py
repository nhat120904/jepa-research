"""Strict adapter for the authors' original DINO-WM PushT release.

This module intentionally does not import ``facebookresearch/jepa-wms``.  It
loads the pickled modules and resolved ``hydra.yaml`` emitted by
``gaoyuezhou/dino_wm`` and drives ``VWorldModel.rollout`` exactly through that
implementation's action grouping and preprocessing conventions.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


PINNED_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


@dataclass(frozen=True)
class OriginalDinoProvenance:
    implementation: str
    source_commit: str
    source_origin: str
    checkpoint: str
    checkpoint_sha256: str
    config: str
    config_sha256: str
    checkpoint_epoch: int


class OriginalDinoWMPushT:
    """Load and score the official PushT checkpoint from the original paper."""

    raw_action_dim = 2

    def __init__(
        self,
        source_root: Path,
        model_dir: Path,
        *,
        checkpoint: Path | None = None,
        device: str = "cuda",
        strict_commit: bool = True,
    ) -> None:
        self.source_root = source_root.resolve()
        self.model_dir = model_dir.resolve()
        self.device = torch.device(device)
        self.config_path = self.model_dir / "hydra.yaml"
        self.checkpoint_path = (
            checkpoint.resolve()
            if checkpoint is not None
            else self.model_dir / "checkpoints" / "model_latest.pth"
        )
        self._validate_paths(strict_commit=strict_commit)

        # The release checkpoint pickles classes under top-level names such as
        # ``models.*``.  Its source must be first during torch.load.
        source = str(self.source_root)
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)

        import hydra
        from omegaconf import OmegaConf

        self.cfg = OmegaConf.load(self.config_path)
        self._validate_config()
        payload = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        required = {"predictor", "proprio_encoder", "action_encoder", "epoch"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"official checkpoint is missing keys: {missing}")

        encoder = payload.get("encoder")
        if encoder is None:
            encoder = hydra.utils.instantiate(self.cfg.encoder)
        encoder.requires_grad_(False)
        decoder = None  # Scoring is entirely in latent space.
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=encoder,
            proprio_encoder=payload["proprio_encoder"],
            action_encoder=payload["action_encoder"],
            predictor=payload["predictor"],
            decoder=decoder,
            proprio_dim=payload["proprio_encoder"].emb_dim,
            action_dim=payload["action_encoder"].emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        ).to(self.device)
        self.model.eval()
        self.transform = hydra.utils.instantiate(self.cfg.env.dataset.transform)

        # Exact constants in the original datasets/pusht_dset.py release.
        self.action_mean = torch.tensor([-0.0087, 0.0068], dtype=torch.float32)
        self.action_std = torch.tensor([0.2019, 0.2002], dtype=torch.float32)
        self.proprio_mean = torch.tensor(
            [236.6155, 264.5674, -2.93032027, 2.54307914], dtype=torch.float32
        )
        self.proprio_std = torch.tensor(
            [101.1202, 87.0112, 74.84556075, 74.14009094], dtype=torch.float32
        )
        self.epoch = int(payload["epoch"])
        self.provenance = OriginalDinoProvenance(
            implementation="gaoyuezhou/dino_wm",
            source_commit=_git_value(self.source_root, "rev-parse", "HEAD"),
            source_origin=_git_value(self.source_root, "remote", "get-url", "origin"),
            checkpoint=str(self.checkpoint_path),
            checkpoint_sha256=sha256(self.checkpoint_path),
            config=str(self.config_path),
            config_sha256=sha256(self.config_path),
            checkpoint_epoch=self.epoch,
        )

    def _validate_paths(self, *, strict_commit: bool) -> None:
        if not (self.source_root / "models" / "visual_world_model.py").exists():
            raise FileNotFoundError(
                f"not an original gaoyuezhou/dino_wm checkout: {self.source_root}"
            )
        if not self.config_path.exists():
            raise FileNotFoundError(f"missing released resolved config: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"missing released checkpoint: {self.checkpoint_path}")
        commit = _git_value(self.source_root, "rev-parse", "HEAD")
        if strict_commit and commit != PINNED_COMMIT:
            raise ValueError(
                f"DINO-WM source commit {commit} != pinned paper implementation {PINNED_COMMIT}"
            )
        origin = _git_value(self.source_root, "remote", "get-url", "origin")
        if "gaoyuezhou/dino_wm" not in origin:
            raise ValueError(f"unexpected DINO-WM git origin: {origin}")
        dirty = _git_value(
            self.source_root, "status", "--porcelain", "--untracked-files=no"
        )
        if dirty:
            raise ValueError("original DINO-WM checkout has modified tracked files")

    def _validate_config(self) -> None:
        checks = {
            "env.name": str(self.cfg.env.name) == "pusht",
            "frameskip": int(self.cfg.frameskip) == 5,
            "num_hist": int(self.cfg.num_hist) == 3,
            "encoder.name": str(self.cfg.encoder.name) == "dinov2_vits14",
            "encoder.feature_key": str(self.cfg.encoder.feature_key)
            == "x_norm_patchtokens",
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(
                "checkpoint hydra.yaml is not the official PushT recipe; failed: "
                + ", ".join(failed)
            )
        self.frameskip = int(self.cfg.frameskip)

    def _visual(self, value: torch.Tensor) -> torch.Tensor:
        """Raw uint8/float B,T,H,W,C -> original normalized B,T,C,H,W."""

        x = value.to(dtype=torch.float32)
        if x.ndim != 5 or x.shape[-1] != 3:
            raise ValueError(f"expected B,T,H,W,C visual input, got {tuple(x.shape)}")
        x = x.permute(0, 1, 4, 2, 3) / 255.0
        return self.transform(x).to(self.device)

    def _proprio(self, value: torch.Tensor) -> torch.Tensor:
        x = value.to(dtype=torch.float32, device="cpu")
        x = (x - self.proprio_mean) / self.proprio_std
        return x.to(self.device)

    def _actions(self, value: torch.Tensor) -> torch.Tensor:
        """Raw B,H,5,2 controls -> normalized B,H,10 model actions."""

        x = value.to(dtype=torch.float32, device="cpu")
        if x.ndim != 4 or x.shape[-2:] != (self.frameskip, self.raw_action_dim):
            raise ValueError(
                f"expected B,H,{self.frameskip},{self.raw_action_dim} actions, got {tuple(x.shape)}"
            )
        x = (x - self.action_mean) / self.action_std
        return x.flatten(start_dim=2).to(self.device)

    @torch.no_grad()
    def encode_observations(
        self, visual: torch.Tensor, proprio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        obs = {"visual": self._visual(visual), "proprio": self._proprio(proprio)}
        return self.model.encode_obs(obs)

    @torch.no_grad()
    def rollout(
        self,
        anchor_visual: torch.Tensor,
        anchor_proprio: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Original planner path: one current observation and grouped actions."""

        if anchor_visual.shape[1] != 1 or anchor_proprio.shape[1] != 1:
            raise ValueError("the original DINO-WM planner conditions on one current frame")
        obs = {
            "visual": self._visual(anchor_visual),
            "proprio": self._proprio(anchor_proprio),
        }
        z_obses, _ = self.model.rollout(obs_0=obs, act=self._actions(actions))
        return z_obses
