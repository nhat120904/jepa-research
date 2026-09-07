"""Predictive-state JEPA: predict the *filtered belief*, not the next frame.

The claim this module exists to make testable is not "add memory to LeWM".  A
recurrent LeWM still predicts the embedding of a future *frame*; two futures that
look alike get the same target however differently they were reached.  Here the
target is the embedding of a future *history* -- what a filter that has watched
the whole episode would hold at ``t + k`` -- so two visually similar frames
reached through different pasts can carry different targets, and the predictor is
forced to represent whatever separates them.

Three commitments follow from what the sibling programs already measured, and
each is left falsifiable by an arm flag rather than assumed:

* **State inference reads observations, not the actions taken.**  On the oracle
  arena the observation prefix carried the entire history advantage (+24.48
  points, CI [+17.45, +32.03]) while action tokens added nothing given vision
  (-2.86, CI [-6.25, +0.26]) and actively caused over-reading, because a skill
  that was *called* is not a skill that *succeeded*.  So ``H_phi`` sees latents
  only, and ``ps_action_history`` puts the action prefix back to re-measure that
  here, where the executor is a learned model rather than an oracle.

* **Counterfactual prediction is action-conditioned, and must be.**  The
  predictor rolls forward one action block at a time, so a planner can ask what a
  candidate chunk would do.  Without it every candidate receives the same
  prediction and the model is not controllable at all; ``ps_noaction_pred`` is
  that degenerate control, and it should be far worse.

* **Recurrence is trained at deployment length.**  The progress head was trained
  on 8-block windows from a zero state and deployed carrying state for 40-80
  blocks.  Segments here are contiguous and long, with a burn-in prefix that
  contributes no loss, so the state a target is read from was warmed the way a
  deployed state is.

The collapse risk is real and specific: the target encoder is an EMA of the
online one, so a constant belief satisfies the prediction loss exactly.  SIGReg
is carried over from the LeWM recipe at the same weight, and ``collapse_metrics``
reports effective rank and the prediction error *relative to* belief variance --
a small absolute error means nothing if the beliefs it is measured between are
all the same point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


PROTOCOL = "scene_progress_wm_ps_jepa_v1"


@dataclass(frozen=True)
class PSJEPAConfig:
    latent_dim: int = 192  # frozen LeWM embedding
    belief_dim: int = 256
    action_input_dim: int = 25  # action_block (5) x action_dim (5)
    action_embed: int = 64
    predict_blocks: int = 5  # K: the planner's horizon, in action blocks
    segment_blocks: int = 48  # contiguous grid steps per training segment
    burn_in: int = 16  # leading blocks that warm the state and carry no loss
    ema_decay: float = 0.996
    dropout: float = 0.05
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_proj: int = 1024
    # -- arm flags, all defaulting to the method --------------------------
    use_history: bool = True  # recurrence in H_phi
    use_action_history: bool = False  # action prefix into H_phi
    use_candidate_action: bool = True  # candidate actions into the predictor

    def to_json(self) -> dict:
        return asdict(self)


#: Every arm is the same architecture at the same parameter count; they differ
#: only in which inputs are readable and whether the recurrence is severed.
ARMS: dict[str, dict] = {
    "ps_jepa": {},
    "ps_frame": {"use_history": False},
    "ps_action_history": {"use_action_history": True},
    "ps_noaction_pred": {"use_candidate_action": False},
}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class BeliefEncoder(nn.Module):
    """``H_phi``: the filter. Recurrent over observation latents."""

    def __init__(self, config: PSJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.belief_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        # present at every arm so the parameter count is matched; zeroed when the
        # arm may not read the action prefix
        self.action_embed = nn.Linear(config.action_input_dim, config.action_embed)
        self.cell = nn.GRUCell(config.belief_dim + config.action_embed, config.belief_dim)
        self.norm = nn.LayerNorm(config.belief_dim)

    def initial_state(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch, self.config.belief_dim, device=device, dtype=dtype)

    def step(self, state: torch.Tensor, latent: torch.Tensor, block: torch.Tensor):
        cfg = self.config
        token = self.proj(latent)
        action = self.action_embed(block)
        if not cfg.use_action_history:
            action = torch.zeros_like(action)
        if not cfg.use_history:
            # frame-conditioned control: the filter is severed, so the belief is a
            # function of the current observation alone
            state = torch.zeros_like(state)
        return self.cell(torch.cat([token, action], dim=-1), state)

    def scan(self, latents: torch.Tensor, blocks: torch.Tensor, state=None):
        """``(B, T, ...)`` in, ``(B, T, belief_dim)`` plus the final state out."""
        batch, steps = latents.shape[0], latents.shape[1]
        if state is None:
            state = self.initial_state(batch, latents.device, latents.dtype)
        beliefs = []
        for t in range(steps):
            state = self.step(state, latents[:, t], blocks[:, t])
            beliefs.append(self.norm(state))
        return torch.stack(beliefs, dim=1), state


class BeliefPredictor(nn.Module):
    """``P_theta``: action-conditioned dynamics in belief space, one block at a time."""

    def __init__(self, config: PSJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.action_embed = nn.Linear(config.action_input_dim, config.action_embed)
        self.cell = nn.GRUCell(config.action_embed, config.belief_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(config.belief_dim),
            nn.Linear(config.belief_dim, config.belief_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.belief_dim, config.belief_dim),
        )
        self.norm = nn.LayerNorm(config.belief_dim)

    def step(self, belief: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        action = self.action_embed(block)
        if not self.config.use_candidate_action:
            # degenerate control: every candidate chunk gets the same prediction,
            # which is what "not action-conditioned" actually costs a planner
            action = torch.zeros_like(action)
        hidden = self.cell(action, belief)
        return self.norm(hidden + self.head(hidden))

    def rollout(self, belief: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """``blocks`` is ``(B, K, action_input_dim)``; returns ``(B, K, belief_dim)``."""
        out = []
        for k in range(blocks.shape[1]):
            belief = self.step(belief, blocks[:, k])
            out.append(belief)
        return torch.stack(out, dim=1)


class PSJEPA(nn.Module):
    """Online filter, its EMA target, and the action-conditioned predictor."""

    def __init__(self, config: PSJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = BeliefEncoder(config)
        self.predictor = BeliefPredictor(config)
        self.target = BeliefEncoder(config)
        self.target.load_state_dict(self.encoder.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        decay = self.config.ema_decay
        for online, target in zip(self.encoder.parameters(), self.target.parameters()):
            target.mul_(decay).add_(online.detach(), alpha=1.0 - decay)
        for online, target in zip(self.encoder.buffers(), self.target.buffers()):
            target.copy_(online)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class SceneSegmentDataset(Dataset):
    """Contiguous grid segments that never cross an episode boundary.

    The render grid is absolute -- ``arange(0, total, stride)`` -- and
    ``1001 % 5 != 0``, so each episode meets it at a different phase.  Segments
    are therefore selected by absolute grid index and rejected whenever the block
    leading into the first frame, or the last frame itself, would fall in a
    neighbouring episode.  Job 49520 died on assuming otherwise.

    ``blocks[t]`` is the action block leading *into* frame ``t``, matching the
    convention the closed loop and the LeWM window dataset both use.
    """

    def __init__(self, cache_dir: Path, config: PSJEPAConfig, latents_name: str = "latents.npy"):
        import json

        self.config = config
        self.meta = json.loads((cache_dir / "meta.json").read_text())
        self.stride = int(self.meta["pixel_stride"])
        self.episode_len = int(self.meta["episode_len"])
        self.latents = np.load(cache_dir / latents_name, mmap_mode="r")
        self.actions = np.load(cache_dir / "actions.npy", mmap_mode="r")

        length = config.segment_blocks
        grid = np.arange(len(self.latents), dtype=np.int64)
        first_row = (grid - 1) * self.stride  # the block leading into frame `grid`
        last_row = (grid + length - 1) * self.stride
        ok = first_row >= 0
        ok &= last_row < len(self.actions)
        # same episode at both ends, which also covers everything between
        ok[ok] = (first_row[ok] // self.episode_len) == (last_row[ok] // self.episode_len)
        self.starts = np.flatnonzero(ok)
        if self.starts.size == 0:
            raise RuntimeError("no usable segments in this cache")

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        length = self.config.segment_blocks
        latents = np.asarray(self.latents[start : start + length], dtype=np.float32)
        blocks = np.empty((length, self.config.action_input_dim), dtype=np.float32)
        for t in range(length):
            row = (start + t) * self.stride
            blocks[t] = np.asarray(
                self.actions[row - self.stride : row], dtype=np.float32
            ).reshape(-1)
        return {
            "latents": torch.from_numpy(latents),
            "blocks": torch.from_numpy(blocks),
            "start": torch.tensor(start, dtype=torch.long),
        }


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
def ps_jepa_losses(
    model: PSJEPA,
    sigreg: nn.Module,
    latents: torch.Tensor,
    blocks: torch.Tensor,
    config: PSJEPAConfig,
) -> dict[str, torch.Tensor]:
    """Predict the EMA filter's future belief from the current one and a chunk.

    Anchored at every post-burn-in step that leaves room for the full horizon, so
    one segment supplies many ``(t, k)`` pairs rather than one.  The target is
    stop-gradient by construction: it comes from the EMA encoder under
    ``no_grad``.
    """
    beliefs, _ = model.encoder.scan(latents, blocks)
    with torch.no_grad():
        targets, _ = model.target.scan(latents, blocks)

    horizon = config.predict_blocks
    first = config.burn_in
    last = latents.shape[1] - horizon  # exclusive
    if last <= first:
        raise ValueError("segment is too short for the burn-in and horizon")

    # Every anchor runs the same K-step rollout, so they are folded into the batch
    # rather than looped: K sequential steps on ``B * anchors`` rows instead of
    # ``anchors * K`` sequential steps on ``B``.
    batch, _, belief_dim = beliefs.shape
    anchors = torch.arange(first, last, device=beliefs.device)
    count = int(anchors.numel())
    # blocks[t + 1 : t + 1 + K] are the chunks leading from frame t forward
    index = anchors[:, None] + 1 + torch.arange(horizon, device=beliefs.device)[None, :]

    start = beliefs[:, anchors].reshape(batch * count, belief_dim)
    actions = blocks[:, index].reshape(batch * count, horizon, config.action_input_dim)
    target = targets[:, index].reshape(batch * count, horizon, belief_dim)

    predicted = model.predictor.rollout(start, actions)
    per_horizon = (predicted - target).pow(2).mean(dim=(0, 2))  # (K,)
    per_step = [per_horizon]
    pred_loss = per_horizon.mean()
    # SIGReg wants (T, B, D); the online beliefs are what must stay spread out
    sigreg_loss = sigreg(beliefs[:, first:].transpose(0, 1))
    return {
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "loss": pred_loss + config.sigreg_weight * sigreg_loss,
        "per_horizon": torch.stack(per_step).mean(dim=0).detach(),
        "beliefs": beliefs.detach(),
        "targets": targets.detach(),
    }


@torch.no_grad()
def collapse_metrics(beliefs: torch.Tensor, pred_loss: torch.Tensor) -> dict[str, float]:
    """Is the prediction loss small because it is right, or because nothing moved?

    A constant belief satisfies the objective exactly, so absolute error is not
    evidence on its own.  ``normalised_error`` divides it by the variance of the
    beliefs it is measured between; effective rank -- the entropy of the
    normalised singular spectrum -- says how many directions the belief space
    actually uses out of ``belief_dim``.
    """
    flat = beliefs.reshape(-1, beliefs.shape[-1]).float()
    variance = float(flat.var(dim=0).mean())
    centred = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred)
    spectrum = singular / singular.sum().clamp_min(1e-12)
    entropy = -(spectrum * spectrum.clamp_min(1e-12).log()).sum()
    return {
        "belief_variance": variance,
        "effective_rank": float(entropy.exp()),
        "dims": int(flat.shape[-1]),
        "normalised_error": float(pred_loss) / max(variance, 1e-12),
    }


__all__ = [
    "ARMS",
    "PROTOCOL",
    "BeliefEncoder",
    "BeliefPredictor",
    "PSJEPA",
    "PSJEPAConfig",
    "SceneSegmentDataset",
    "collapse_metrics",
    "ps_jepa_losses",
]
