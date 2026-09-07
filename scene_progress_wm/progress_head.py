"""The method: a latched, history-conditioned progress head over a frozen world model.

Three findings from the Event-SMDP scaffold on this same environment motivate the shape
of this module, and each is left falsifiable by an ablation flag rather than assumed:

* progress in Scene is **latched** -- the buttons that lock the drawer and window stay
  flipped, and the current frame does not say whether they ever were. A cost read off a
  single predicted frame cannot represent that, so the head carries recurrent state
  across the whole episode (``use_history``), and its progress vector is non-decreasing
  by construction rather than by penalty (``monotone``);
* **action dead reckoning is not a substitute** for watching the world: integrating the
  actions taken, without the observations, collapsed under execution failure. So the
  head consumes both, and either can be ablated (``use_obs``, ``use_action``);
* cost designs that look good under oracle state can invert under learned state, so the
  progress signal here is learned from pixels only and is never handed the simulator.

The readout is goal-conditioned normalised remaining time: ``v`` near 0 means "this state
is at the goal", near 1 means "far". Planning minimises it, so it drops into the same
``Objective`` slot as latent-L2 with no other change to the loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

PROTOCOL = "scene_progress_wm_progress_head_v1"


@dataclass(frozen=True)
class ProgressConfig:
    latent_dim: int = 192
    action_input_dim: int = 25
    action_embed: int = 32
    hidden: int = 256
    progress_dim: int = 8
    dropout: float = 0.05
    aux_dim: int = 5  # cube, button_0, button_1, drawer, window
    use_obs: bool = True
    use_action: bool = True
    use_history: bool = True
    monotone: bool = True

    def to_json(self) -> dict:
        return asdict(self)


ARMS: dict[str, dict] = {
    # the arm names the protocol uses; every one shares this architecture and differs
    # only in which inputs it may read and whether progress is forced to latch
    "prog_ssl": {},
    "prog_sup": {},
    "prog_frame": {"use_history": False},
    "prog_action_only": {"use_obs": False},
    "prog_nomono": {"monotone": False},
}


class ProgressHead(nn.Module):
    """Recurrent progress state plus a goal-conditioned remaining-time readout."""

    def __init__(self, config: ProgressConfig) -> None:
        super().__init__()
        self.config = config

        self.action_embed = nn.Linear(config.action_input_dim, config.action_embed)
        token_dim = config.latent_dim + config.action_embed
        self.token_norm = nn.LayerNorm(token_dim)
        self.cell = nn.GRUCell(token_dim, config.hidden)

        # progress increments are non-negative, so the accumulated vector can only
        # rise: latching is a property of the architecture, not of a loss term
        self.increment = nn.Linear(config.hidden, config.progress_dim)

        readout_in = config.hidden + config.progress_dim + config.latent_dim
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_in),
            nn.Linear(readout_in, config.hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, 1),
        )
        # auxiliary head for the supervised arm; unused by prog_ssl
        self.aux = nn.Linear(config.progress_dim, config.aux_dim)

    # -- state ------------------------------------------------------------
    def initial_state(self, batch: int, device, dtype=torch.float32):
        cfg = self.config
        return (
            torch.zeros(batch, cfg.hidden, device=device, dtype=dtype),
            torch.zeros(batch, cfg.progress_dim, device=device, dtype=dtype),
        )

    def token(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if not cfg.use_obs:
            latent = torch.zeros_like(latent)
        embedded = self.action_embed(action)
        if not cfg.use_action:
            embedded = torch.zeros_like(embedded)
        return self.token_norm(torch.cat([latent, embedded], dim=-1))

    def step(self, state, latent: torch.Tensor, action: torch.Tensor):
        """Advance one action block. ``latent`` is the frame arrived at."""
        hidden, progress = state
        if not self.config.use_history:
            # frame-conditioned control: the recurrence is severed, so the head sees
            # only the current observation and the block that produced it
            hidden = torch.zeros_like(hidden)
            progress = torch.zeros_like(progress)
        hidden = self.cell(self.token(latent, action), hidden)
        raw = self.increment(hidden)
        progress = progress + F.softplus(raw) if self.config.monotone else raw
        return hidden, progress

    def value(self, state, goal_latent: torch.Tensor) -> torch.Tensor:
        """Predicted normalised remaining time to ``goal_latent``; lower is closer."""
        hidden, progress = state
        return self.readout(torch.cat([hidden, progress, goal_latent], dim=-1)).squeeze(-1)

    def aux_logits(self, state) -> torch.Tensor:
        return self.aux(state[1])

    # -- sequence helpers --------------------------------------------------
    def scan(self, latents: torch.Tensor, actions: torch.Tensor, state=None):
        """Run the head over ``(B, T, ...)`` sequences, returning every intermediate.

        Used by training, and by the planner to roll predicted latents forward.
        """
        batch, steps = latents.shape[0], latents.shape[1]
        if state is None:
            state = self.initial_state(batch, latents.device, latents.dtype)
        hiddens, progresses = [], []
        for t in range(steps):
            state = self.step(state, latents[:, t], actions[:, t])
            hiddens.append(state[0])
            progresses.append(state[1])
        return torch.stack(hiddens, dim=1), torch.stack(progresses, dim=1), state


def expectile_loss(
    prediction: torch.Tensor, target: torch.Tensor, tau: float = 0.2
) -> torch.Tensor:
    """Asymmetric L2 used instead of a mean.

    The play data is not expert data, so the number of steps a trajectory happened to
    take between two states is an upper-biased estimate of how far apart they are. An
    expectile with ``tau < 0.5`` leans on the short observations, approximating the
    minimum rather than the average.
    """
    residual = target - prediction
    weight = torch.where(residual > 0, tau, 1.0 - tau)
    return (weight * residual.pow(2)).mean()


class ProgressTracker:
    """The executed-history state the planner reads, updated once per replan.

    The distinction matters: CEM calls the cost hundreds of times per replan, and if the
    recurrent state advanced on every one of those calls the head would be integrating
    imagined rollouts into its memory of what actually happened. This object is stepped
    only by the environment loop, and the objective treats it as a constant.
    """

    def __init__(self, head: ProgressHead, device, dtype=torch.float32) -> None:
        self.head = head
        self.device = device
        self.dtype = dtype
        self.state = head.initial_state(1, device, dtype)
        self.updates = 0

    def reset(self, latent: torch.Tensor) -> None:
        cfg = self.head.config
        self.state = self.head.initial_state(1, self.device, self.dtype)
        zero_block = torch.zeros(
            1, cfg.action_input_dim, device=self.device, dtype=self.dtype
        )
        with torch.no_grad():
            self.state = self.head.step(self.state, latent.view(1, -1), zero_block)
        self.updates = 1

    def observe(self, latent: torch.Tensor, action_block: torch.Tensor) -> None:
        with torch.no_grad():
            self.state = self.head.step(
                self.state,
                latent.view(1, -1).to(self.dtype),
                action_block.view(1, -1).to(self.dtype),
            )
        self.updates += 1

    def frozen(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state[0].detach(), self.state[1].detach()

    def fingerprint(self) -> tuple[int, float]:
        """Cheap identity of the current state, for the invariance check in smoke runs."""
        return self.updates, float(self.state[0].sum()) + float(self.state[1].sum())


__all__ = [
    "ARMS",
    "PROTOCOL",
    "ProgressConfig",
    "ProgressHead",
    "ProgressTracker",
    "expectile_loss",
]
