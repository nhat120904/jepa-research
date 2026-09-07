"""LeWM on OGBench-Scene: model construction, windowed dataset, loss.

The recipe is the published LeWM one -- ViT encoder trained from scratch, CLS token
through a projector, causal predictor with AdaLN action conditioning, and
``pred_loss + 0.09 * SIGReg`` -- with two declared deviations, both forced by the arena:

* image size 64 with patch size 8 (64 patches) instead of 224/14, because the arena is
  OGBench's visual-Scene resolution;
* actions are **not** normalised. OGBench actions already live in the environment's own
  ``Box(-1, 1)``, so the planner searches exactly the space the model was trained on.
  Introducing a normaliser here is the single most common source of silent train/plan
  mismatch in this codebase, and it buys nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROTOCOL = "scene_progress_wm_lewm_v1"


@dataclass(frozen=True)
class LeWMConfig:
    img_size: int = 64
    patch_size: int = 8
    encoder_scale: str = "tiny"
    embed_dim: int = 192
    history_size: int = 3
    num_preds: int = 1
    action_dim: int = 5
    action_block: int = 5
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.0
    projector_hidden: int = 2048
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_proj: int = 1024

    @property
    def num_frames(self) -> int:
        """Frames per training window: context plus the prediction target."""
        return self.history_size + self.num_preds

    @property
    def action_input_dim(self) -> int:
        return self.action_dim * self.action_block

    @property
    def window_rows(self) -> int:
        """Environment steps a training window spans."""
        return self.num_frames * self.action_block

    def to_json(self) -> dict:
        return asdict(self)


def build_lewm(config: LeWMConfig) -> nn.Module:
    from stable_pretraining.backbone.utils import vit_hf
    from stable_worldmodel.wm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor

    encoder = vit_hf(
        size=config.encoder_scale,
        patch_size=config.patch_size,
        image_size=config.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    predictor = Predictor(
        num_frames=config.history_size,
        input_dim=config.embed_dim,
        hidden_dim=config.embed_dim,
        output_dim=config.embed_dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
    )
    action_encoder = Embedder(
        input_dim=config.action_input_dim, emb_dim=config.embed_dim
    )
    projector = MLP(
        input_dim=config.embed_dim,
        hidden_dim=config.projector_hidden,
        output_dim=config.embed_dim,
        norm_fn=nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=config.embed_dim,
        hidden_dim=config.projector_hidden,
        output_dim=config.embed_dim,
        norm_fn=nn.BatchNorm1d,
    )
    return LeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


def image_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
    return mean, std


def normalise_pixels(pixels: torch.Tensor, mean, std) -> torch.Tensor:
    """``(B, T, H, W, 3)`` uint8 -> ``(B, T, 3, H, W)`` normalised float."""
    x = pixels.permute(0, 1, 4, 2, 3).float().div_(255.0)
    return (x - mean) / std


class SceneWindowDataset(Dataset):
    """Windows of ``num_frames`` frames spaced ``action_block`` environment steps apart.

    Frame ``k`` sits at row ``t + k * action_block``; action block ``k`` is the
    ``action_block`` primitive actions leaving that frame, flattened. Windows never
    straddle an episode boundary, and never start off the cache's render grid --
    the cache renders one frame every ``pixel_stride`` rows and leaves the rest zero,
    so an ungridded start would silently train on blank images.
    """

    def __init__(
        self,
        cache_dir: Path,
        config: LeWMConfig,
        episode_len: int,
        pixel_stride: int = 1,
    ) -> None:
        cache_dir = Path(cache_dir)
        self.pixels = np.load(cache_dir / "pixels.npy", mmap_mode="r")
        self.actions = np.load(cache_dir / "actions.npy", mmap_mode="r")
        self.config = config
        self.episode_len = episode_len
        self.pixel_stride = int(pixel_stride)

        # Only rows on the cache's render grid hold a frame; every other row of
        # pixels.npy is still zero. Frames within a window are action_block apart,
        # so a window is renderable exactly when the block divides the stride and
        # the start sits on the grid.
        if config.action_block % self.pixel_stride != 0:
            raise RuntimeError(
                f"action block {config.action_block} is not a multiple of the "
                f"cache render stride {self.pixel_stride}"
            )
        total = self.pixels.shape[0]
        if total % episode_len != 0:
            raise RuntimeError(f"{total} rows is not a whole number of episodes")

        # The render grid is absolute (`arange(0, total, stride)`), not per-episode, and
        # 1001 % 5 != 0, so each episode meets the grid at a different phase. Select
        # starts by the absolute row, then keep the ones that leave a whole window
        # inside their own episode.
        span = config.window_rows  # last block ends at t + num_frames * action_block
        grid = np.arange(0, total, self.pixel_stride, dtype=np.int64)
        offset_in_episode = grid % episode_len
        self.starts = grid[offset_in_episode <= episode_len - span]

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        cfg = self.config
        start = int(self.starts[index])
        rows = start + np.arange(cfg.num_frames, dtype=np.int64) * cfg.action_block
        pixels = np.asarray(self.pixels[rows])
        blocks = np.stack(
            [
                np.asarray(
                    self.actions[row : row + cfg.action_block], dtype=np.float32
                ).reshape(-1)
                for row in rows
            ]
        )
        return {
            "pixels": torch.from_numpy(pixels),
            "action": torch.from_numpy(blocks),
            "start": torch.tensor(start, dtype=torch.long),
        }


def lewm_losses(
    model: nn.Module,
    sigreg: nn.Module,
    pixels: torch.Tensor,
    action: torch.Tensor,
    config: LeWMConfig,
) -> dict[str, torch.Tensor]:
    """The published LeWM objective, unchanged."""
    info = model.encode({"pixels": pixels, "action": action})
    emb = info["emb"]
    act_emb = info["act_emb"]

    ctx_emb = emb[:, : config.history_size]
    ctx_act = act_emb[:, : config.history_size]
    tgt_emb = emb[:, config.num_preds :]
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    return {
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "loss": pred_loss + config.sigreg_weight * sigreg_loss,
    }


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PROTOCOL",
    "LeWMConfig",
    "SceneWindowDataset",
    "build_lewm",
    "image_stats",
    "lewm_losses",
    "normalise_pixels",
]
