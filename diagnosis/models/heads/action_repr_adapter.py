"""Action-conditioned representation adapter φ = A(z) — Track 2, the heavy fix
(docs/plans/2026-07-01-adversarial-cost-and-repr-design.md).

The oracle ladder (results/oracle_ladder_cost_report.md) closed every frozen-encoder
lever: dynamics (predictor), cost formula, hand-approach term, and even an
off-policy-ROBUST readout (Phase-3 3b: obj 78→92% <5cm off-policy) still re-gates
push 1/16. CEM — searching for the cost MINIMUM over 100×6 candidates — exploits
whatever residual-error pockets a POST-HOC cost on the frozen latent has. That is a
reward-hacking failure, not a missing-information one (the object IS recoverable at
~2cm off-policy).

This module instead RESHAPES the representation: φ is a small learned head over the
frozen patch tokens (encoder + predictor stay frozen; only φ trains,
``scripts/37_train_repr_adapter.py``), trained with four losses so plain L2 in
φ-space (``scripts/30 --cost phi``) is a plannable, harder-to-exploit cost:

  1. grounding      — ``φ[:, :obj_dim]`` regresses the true object xyz. Keeps the
                       ~2cm object precision the frozen z already carries
                       (Test-1b), now at ``obj_dim`` dims instead of buried in the
                       full ~98k-dim latent.
  2. cf-contrastive  — for HARD-EFFECT negatives (same z_t, different action,
                       genuinely different true outcome —
                       ``metrics.negative_samplers.hard_effect_negative``, the
                       exact geometry CRA measures as near-chance in
                       decision_report.md), the object estimate of the two true
                       next states must be MARGIN-separated by at least their true
                       gap. This hardens exactly the channel Phase-3 tried (and
                       failed) to harden via plain MSE alone — a margin constraint
                       directly targets residual-error pockets, not just average
                       accuracy.
  3. temporal        — a RANKING loss (not a literal step-gap regression, which
                       would fight term 1's metres-scale grounding — see
                       ``scripts/37``'s docstring): within a trajectory, a frame
                       temporally closer to the goal must have smaller φ-distance
                       to the goal than one further away. Lives entirely in the
                       "extra" (non-object) subspace of φ so it cannot dilute the
                       object signal.
  4. adversarial     — CEM-elite frames mined by ``scripts/_cem_mining`` (the ones
                       a frozen-encoder cost got EXPLOITED on) are margin-pushed
                       away from their episode's goal in the object subspace, in
                       proportion to their TRUE object→goal distance — directly
                       closing the pockets that broke Phase-3 3b. Uses the same
                       margin mechanism as term 2, applied to an adversarially
                       mined (rather than in-batch) negative source.

``φ``'s object subspace (metres) and its free "extra" subspace (its own learned
scale, ``extra_scale``, calibrated at train time) must never be summed unscaled —
the same "squared norms over scale norms" lesson already documented in
``models/probes/object_probe.py``'s ``grounded_dynamics_cost``. ``scripts/30``'s
``phi`` cost arm applies that scaling.

The trunk is a CLS-attention transformer over patch tokens (the same architecture
as ``models.probes.SpatialObjectProbe``, which Test-1b showed can localise the
object at 92% <5cm — mean/max pooling cannot, since it discards which patch lit up).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.heads.mixture_predictor import flatten_tokens


class ActionReprAdapter(nn.Module):
    """φ = A(z): CLS-attention over patch tokens -> a ``phi_dim`` embedding whose
    first ``obj_dim`` components are the grounded object-position estimate; the
    remaining ``phi_dim - obj_dim`` ("extra") components are free to absorb the
    temporal / adversarial signal (terms 3-4)."""

    def __init__(self, latent_dim: int, phi_dim: int = 64, obj_dim: int = 3, *,
                n_layers: int = 2, n_heads: int = 6, ff_mult: int = 2, hidden: int = 512,
                extra_scale: float = 1.0):
        super().__init__()
        assert phi_dim > obj_dim, "phi_dim must leave room for the 'extra' subspace"
        self.latent_dim = latent_dim
        self.phi_dim = phi_dim
        self.obj_dim = obj_dim
        self.cls = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=ff_mult * latent_dim,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(),
                                  nn.Linear(hidden, phi_dim))
        # Calibrated at train time (std of the "extra" subspace over held-out data)
        # so scripts/30's phi cost can put the object and extra terms on comparable
        # footing; see the module docstring's scale-norm warning.
        self.register_buffer("extra_scale", torch.tensor(float(extra_scale)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: single-frame latent ``(B, *frame_dims)`` -> ``(B, phi_dim)``."""
        tok = flatten_tokens(z, self.latent_dim)
        B = tok.shape[0]
        h = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)
        h = self.encoder(h)
        return self.head(h[:, 0])

    def split(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return phi[..., : self.obj_dim], phi[..., self.obj_dim:]


def load_repr_adapter(ckpt_path: str | Path, device) -> tuple["ActionReprAdapter", dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ActionReprAdapter(latent_dim=ckpt["latent_dim"], phi_dim=ckpt.get("phi_dim", 64),
                          obj_dim=ckpt.get("obj_dim", 3), n_layers=ckpt.get("n_layers", 2),
                          n_heads=ckpt.get("n_heads", 6), hidden=ckpt.get("hidden", 512),
                          extra_scale=ckpt.get("extra_scale", 1.0))
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval(), ckpt


def margin_loss(a: torch.Tensor, b: torch.Tensor, true_gap: torch.Tensor,
                cap: float) -> torch.Tensor:
    """Shared term-2/term-4 mechanism: require ``‖a-b‖`` (both in the same metres
    units, e.g. two object estimates, or an estimate vs a raw true target) to be at
    least the TRUE gap between them (capped so outliers don't dominate). Zero loss
    once the representation already separates the pair correctly — this is a
    minimum-separation constraint, not an exact-match regression, so it stacks with
    term 1's MSE grounding loss without fighting it."""
    d = (a - b).norm(dim=-1)
    margin = true_gap.clamp(max=cap)
    return torch.relu(margin - d).mean()
