"""State-grounded latent metric — the encoder/metric-level fix (direction D).

Measured premise (2026-06-10, `docs/FIX_C1_EXPLAINER.md` §6): in the frozen
latent's **L2 geometry** the contact bifurcation is ~9% of the prediction
residual (conditional "object moves / doesn't" future-means 9.9 apart vs a 106
median residual), so every L2-based consumer — the CEM planning cost, BB's
model-spread, any head trained against L2/NLL on the full latent — is blind to
the boundary *even when the information is present in the latent*.

The fix re-weights the metric instead of touching the model:

1. **ObjectProbe** ``g(z) → object xyz`` — a small supervised readout trained on
   the cached (latent, sim-state) pairs. Its held-out error answers, before
   anything else: *is the boundary-relevant state decodable from the frozen
   latent at all?*
2. **Boundary-aware representation** ``φ(z) = [z/s_z ‖ β·g(z)/s_g]`` — L2 in φ
   equals the original latent metric **plus** an object-displacement term
   re-amplified to comparable scale (s_z, s_g are dataset scales; β the
   amplification knob). Applied to *predictions*:
   - BB's ``S_model`` becomes "spread of the predicted object position across
     neighbour actions" (+ the original term) — exactly the world-side quantity
     ``S_true`` already measures.
   - the CEM planning cost becomes ``MSE(z)/s_z² + β²·MSE(g)/s_g²`` — candidates
     that move the *object* to the goal separate from candidates that merely
     match pixels-at-large.

Everything trains on the existing cache; the encoder and predictor stay frozen.
DROID/real-world transfer caveat: the probe needs a state label, so this exact
form is Metaworld-only (the paper states it as such; the principle — supervise
a metric, not the model — is the transferable claim).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models.adapters import WorldModelAdapter
from models.heads.mixture_predictor import flatten_tokens


class ObjectProbe(nn.Module):
    """Mean+max-pooled token MLP: ``(B, N, D) tokens → (B, out_dim)`` readout."""

    def __init__(self, latent_dim: int, out_dim: int = 3, hidden: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: any single-frame latent ``(B, *frame_dims)``."""
        tok = flatten_tokens(z, self.latent_dim)
        pooled = torch.cat([tok.mean(dim=1), tok.max(dim=1).values], dim=-1)
        return self.net(pooled)


def load_probe(ckpt_path: str | Path, device) -> tuple[ObjectProbe, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    probe = ObjectProbe(latent_dim=ckpt["latent_dim"], out_dim=ckpt["out_dim"],
                        hidden=ckpt["hidden"])
    probe.load_state_dict(ckpt["state_dict"])
    return probe.to(device).eval(), ckpt


class BoundaryAwareMetricAdapter(WorldModelAdapter):
    """Wraps a frozen adapter so its predictions live in φ-space.

    ``predict`` returns ``[γ·flatten(ẑ)/s_z ‖ β·g(ẑ)/s_g]`` — a flat vector.
    Every BB/metric consumer that takes L2 over flattened predictions then
    measures the boundary-aware distance. ``γ=0`` gives the object-only metric
    (the maximally interpretable variant: "spread of the predicted object
    position").
    """

    def __init__(self, base: WorldModelAdapter, probe: ObjectProbe, *,
                 s_z: float, s_g: float, beta: float = 1.0, gamma: float = 1.0,
                 device: Optional[str] = None):
        self.base = base
        self.probe = probe.eval()
        self.spec = base.spec
        self.s_z, self.s_g = float(s_z), float(s_g)
        self.beta, self.gamma = float(beta), float(gamma)
        self.device = torch.device(device) if device else getattr(
            base, "device", torch.device("cpu"))
        self.probe.to(self.device)

    @torch.no_grad()
    def predict(self, z_t: torch.Tensor, a_t: torch.Tensor,
                proprio_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = self.base.predict(z_t, a_t, proprio_t=proprio_t).float()
        B = pred.shape[0]
        parts = []
        if self.gamma != 0.0:
            parts.append(self.gamma * pred.reshape(B, -1) / self.s_z)
        parts.append(self.beta * self.probe(pred) / self.s_g)
        return torch.cat(parts, dim=-1)

    # ----- pass-throughs -------------------------------------------------------
    def encode(self, visual, proprio=None):
        return self.base.encode(visual, proprio)

    def action_dim(self) -> int:
        return self.base.action_dim()

    def uses_proprio(self) -> bool:
        return self.base.uses_proprio()

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return self.base.normalize_action(a)

    def to(self, device):
        self.device = torch.device(device)
        self.probe.to(self.device)
        self.base.to(device)
        return self

    def eval(self):
        self.probe.eval()
        self.base.eval()
        return self


class ObjectDynamicsHead(nn.Module):
    """``h(z_t, a) → Δobject`` — the action-conditioned grounded dynamics channel.

    Why this exists (measured chain, 2026-06-10): V1 the latent encodes the object
    position; V2 the frozen predictor propagates it for the *factual* action; V3
    its **counterfactual** object response is noise (corr with true outcome spread
    ≈ 0.03). The hard_nn neighbourhoods prove the cache contains similar-state /
    different-action / different-outcome triples, so the action-dependence is
    learnable **across samples** even though every individual sample is factual —
    it just has to be supervised directly on the object quantity instead of buried
    in a 98k-dim L2. h is that supervision. At inference it reads only (latent,
    action); the state label is training-time-only (Metaworld direction D).
    """

    def __init__(self, latent_dim: int, action_dim: int, out_dim: int = 3,
                 hidden: int = 512, a_emb_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.out_dim = out_dim
        self.a_emb = nn.Sequential(nn.Linear(action_dim, a_emb_dim), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim + a_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        tok = flatten_tokens(z, self.latent_dim)
        pooled = torch.cat([tok.mean(dim=1), tok.max(dim=1).values,
                            self.a_emb(action)], dim=-1)
        return self.net(pooled)


def load_dynamics_head(ckpt_path: str | Path, device) -> tuple["ObjectDynamicsHead", dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    h = ObjectDynamicsHead(latent_dim=ckpt["latent_dim"], action_dim=ckpt["action_dim"],
                           out_dim=ckpt["out_dim"], hidden=ckpt["hidden"])
    h.load_state_dict(ckpt["state_dict"])
    return h.to(device).eval(), ckpt


class ObjectDynamicsAdapter(WorldModelAdapter):
    """Exposes ``h`` behind the standard interface: ``predict(z_t, a) = h(z_t, a)``
    (the predicted object displacement, a 3-vector). BB's ``S_model`` then measures
    the spread of the predicted object motion across neighbour actions — directly
    commensurate with ``S_true`` (the true object-displacement spread)."""

    def __init__(self, base: WorldModelAdapter, head: ObjectDynamicsHead,
                 device: Optional[str] = None):
        self.base = base
        self.head = head.eval()
        self.spec = base.spec
        self.device = torch.device(device) if device else getattr(
            base, "device", torch.device("cpu"))
        self.head.to(self.device)

    @torch.no_grad()
    def predict(self, z_t: torch.Tensor, a_t: torch.Tensor,
                proprio_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = a_t.shape[0]
        a = a_t.to(self.device).float().reshape(B, -1, self.base.action_dim())
        a = self.base.normalize_action(a).reshape(B, -1)
        return self.head(z_t.to(self.device).float(), a)

    def encode(self, visual, proprio=None):
        return self.base.encode(visual, proprio)

    def action_dim(self) -> int:
        return self.base.action_dim()

    def uses_proprio(self) -> bool:
        return False        # h reads latent + action only

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return self.base.normalize_action(a)

    def to(self, device):
        self.device = torch.device(device)
        self.head.to(self.device)
        self.base.to(device)
        return self

    def eval(self):
        self.head.eval()
        self.base.eval()
        return self


def grounded_dynamics_cost(probe: ObjectProbe, head: ObjectDynamicsHead,
                           base: WorldModelAdapter, z_init: torch.Tensor,
                           z_goal: torch.Tensor, *, s_z: float, s_g: float,
                           beta: float = 1.0):
    """CEM ``traj_cost_fn`` for the grounded-dynamics fix: track the OBJECT.

    Predicted object at the horizon = ``g(z_init) + Σ_t h(ẑ_t, a_t)`` (the h
    channel integrated along the rollout); cost = the upstream latent MSE term
    plus the squared object-goal error, each on its dataset scale:

        cost = MSE(z_H, z_goal)/s_z² + β²·‖obĵ_H − g(z_goal)‖²/s_g²

    h is the channel whose counterfactual spread actually tracks the true
    outcome (corr +0.68 vs +0.03 for the frozen predictor), so near a grasp
    boundary the second term separates "this action moves the object to the
    goal" from "this one misses" — the distinction the plain L2 surface lacks.
    """
    with torch.no_grad():
        g_goal = probe(z_goal.unsqueeze(0))                       # (1, out)
        g_init = probe(z_init.unsqueeze(0))                       # (1, out)

    def cost(pred: torch.Tensor, actions: torch.Tensor, z_goal_: torch.Tensor) -> torch.Tensor:
        B, H1 = pred.shape[0], pred.shape[1]
        H = H1 - 1
        obj = g_init.expand(B, -1).clone()
        for t in range(H):
            a = actions[:, t].reshape(B, -1, base.action_dim())
            a = base.normalize_action(a).reshape(B, -1)
            obj = obj + head(pred[:, t], a)
        # Squared NORMS over their scale norms (NOT per-dim MSE: the two terms
        # have 98k vs 3 dims — per-dim means differ by 4–5 orders of magnitude
        # and silently turn the blend into object-only; measured 2026-06-10).
        sq_z = ((pred[:, -1].reshape(B, -1) - z_goal_.reshape(1, -1)) ** 2).sum(-1)
        sq_g = ((obj - g_goal) ** 2).sum(-1)
        return sq_z / (s_z ** 2) + (beta ** 2) * sq_g / (s_g ** 2)

    return cost


def boundary_aware_cost(probe: ObjectProbe, z_goal: torch.Tensor, *,
                        s_z: float, s_g: float, beta: float = 1.0):
    """CEM ``cost_fn``: ``MSE(z)/s_z² + β²·MSE(g(z))/s_g²`` against the goal.

    Pre-computes the goal's object readout once. Returns a callable
    ``(pred_last (B,*frame), z_goal_) -> (B,)`` matching ``cem_plan(cost_fn=)``.
    """
    with torch.no_grad():
        g_goal = probe(z_goal.unsqueeze(0))                       # (1, out_dim)

    def cost(pred_last: torch.Tensor, z_goal_: torch.Tensor) -> torch.Tensor:
        B = pred_last.shape[0]
        # Squared norms over scale norms — see grounded_dynamics_cost for why
        # per-dim MSE must not be used here.
        sq_z = ((pred_last.reshape(B, -1) - z_goal_.reshape(1, -1)) ** 2).sum(dim=-1)
        g_pred = probe(pred_last)                                 # (B, out_dim)
        sq_g = ((g_pred - g_goal) ** 2).sum(dim=-1)
        return sq_z / (s_z ** 2) + (beta ** 2) * sq_g / (s_g ** 2)

    return cost
