"""Counterfactual action samplers used by CRA/CTD.

Strategies:

- random:    a^- uniform within the model's action bounds
- opposite:  a^- = -a + N(0, sigma^2) with gripper-dim flip
- hard_nn:   a^- drawn from a candidate pool whose latent z is close to z_t
             but whose action is maximally different from a  (paper Section 4.3)
- hard_effect: a^- drawn from a candidate pool whose latent z is close to z_t
             AND whose *true effect* (Δz) differs most from the factual Δz —
             optionally preferring actions *close* to a (small Δaction). This
             targets precision-sensitive regions where a small action change
             yields a large outcome change, and is "fair" by construction (the
             negative's true future genuinely differs from z_{t+1}, so a
             well-grounded model can win the CRA comparison). In smooth regimes
             (free-space) no such negative exists, so it self-selects toward
             contact / fine-precision transitions.
"""

from __future__ import annotations

from typing import Optional

import torch


# ---------- random ----------------------------------------------------------

def random_negative(
    a_t: torch.Tensor,
    action_bounds: tuple[float, float] | torch.Tensor,
    K: int = 16,
    l1_radius: Optional[float] = None,
) -> torch.Tensor:
    """Uniform sample over the action space.

    Returns: (B, K, A).

    If `l1_radius` is given (DROID), the samples are projected onto the L1 ball.
    """
    B, A = a_t.shape
    if isinstance(action_bounds, tuple):
        lo, hi = action_bounds
        out = torch.empty(B, K, A, device=a_t.device).uniform_(lo, hi)
    else:
        bounds = action_bounds.to(a_t.device)  # (A, 2)
        lo = bounds[:, 0].view(1, 1, A)
        hi = bounds[:, 1].view(1, 1, A)
        out = lo + (hi - lo) * torch.rand(B, K, A, device=a_t.device)
    if l1_radius is not None:
        l1 = out.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (l1_radius / l1).clamp(max=1.0)
        out = out * scale
    return out


# ---------- opposite --------------------------------------------------------

def opposite_negative(
    a_t: torch.Tensor,
    sigma: float = 0.1,
    K: int = 16,
    gripper_dim: Optional[int] = None,
    gripper_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """a^- = -a + N(0, sigma^2 I). Gripper dim is explicitly flipped to (1 - g).

    Returns: (B, K, A).
    """
    B, A = a_t.shape
    a_exp = a_t.unsqueeze(1).expand(B, K, A)
    noise = torch.randn(B, K, A, device=a_t.device) * sigma
    out = -a_exp + noise
    if gripper_dim is not None:
        g_lo, g_hi = gripper_range
        flipped = (g_hi - a_exp[..., gripper_dim]) + g_lo
        out[..., gripper_dim] = flipped + noise[..., gripper_dim] * 0.1
    return out


# ---------- hard NN ---------------------------------------------------------

def hard_nn_negative(
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    pool_z: torch.Tensor,
    pool_a: torch.Tensor,
    K: int = 16,
    similarity_radius: float = 0.5,
    distance_fn=None,
) -> torch.Tensor:
    """For each (z_t, a_t), find K actions from the candidate pool whose
    latent z is within `similarity_radius` of z_t but whose action is maximally
    different from a_t.

    Args:
        z_t:    (B, ..., D) anchor latents
        a_t:    (B, A)
        pool_z: (N, ..., D) candidate latents
        pool_a: (N, A) candidate actions
        K:      number of negatives per anchor
        similarity_radius: max latent distance to consider a candidate "similar"

    Returns: (B, K, A) counterfactual actions.
    """
    from .distances import l2_distance

    B = z_t.shape[0]
    N = pool_z.shape[0]
    A = a_t.shape[-1]

    z_flat = z_t.reshape(B, -1)
    pool_flat = pool_z.reshape(N, -1)

    # All-pairs latent distance (B, N).
    z_dist = torch.cdist(z_flat, pool_flat, p=2)
    # All-pairs action distance (B, N): bigger = more different.
    a_dist = torch.cdist(a_t, pool_a.to(a_t.device), p=2)

    # Mask out candidates whose latent is too far.
    sim_mask = z_dist <= similarity_radius
    # Guarantee at least K candidates per anchor: if too few, relax to the
    # K nearest by z regardless of radius.
    counts = sim_mask.sum(dim=-1)
    too_few = counts < K
    if too_few.any():
        # For each underfilled row, take the K nearest by z.
        nearest_z = z_dist.topk(K, dim=-1, largest=False).indices  # (B, K)
        fill_mask = torch.zeros_like(sim_mask)
        fill_mask.scatter_(1, nearest_z, True)
        sim_mask = torch.where(too_few.unsqueeze(-1), fill_mask, sim_mask)

    # Score candidates: prefer large action distance, restrict to sim_mask.
    scores = a_dist.clone()
    scores[~sim_mask] = -float("inf")

    topk_idx = scores.topk(K, dim=-1).indices  # (B, K)
    return pool_a.to(a_t.device)[topk_idx]      # (B, K, A)


# ---------- hard effect (small Δaction, large Δeffect) ----------------------

def hard_effect_negative(
    z_t: torch.Tensor,
    z_t1: torch.Tensor,
    a_t: torch.Tensor,
    pool_z: torch.Tensor,
    pool_z1: torch.Tensor,
    pool_a: torch.Tensor,
    K: int = 16,
    similarity_radius: float = 0.5,
    action_penalty: float = 0.5,
) -> torch.Tensor:
    """Pick negatives from the pool that, from a *similar state*, lead to a
    *different true outcome* than the factual transition — optionally preferring
    actions close to ``a_t``.

    Scoring among candidates with ``||z_t - z_{t'}|| <= similarity_radius``:

        score(t') = norm(|| Δz_{t'} - Δz_t ||)  -  action_penalty · norm(|| a_{t'} - a_t ||)

    where ``Δz = z_{t+1} - z_t`` is the true effect and ``norm`` rescales each
    term by its std over the in-radius candidates so ``action_penalty`` is on a
    comparable scale to the effect term. Large effect-divergence makes the
    negative *answerable* (a grounded model should rank it worse); the action
    penalty makes it a *precision* test (the negative action is near a_t).

    Args:
        z_t, z_t1:   (B, ...) anchor current/next latents
        a_t:         (B, A)
        pool_z, pool_z1: (N, ...) candidate current/next latents
        pool_a:      (N, A)

    Returns: (B, K, A).
    """
    B = z_t.shape[0]
    N = pool_z.shape[0]

    z_flat = z_t.reshape(B, -1)
    pool_flat = pool_z.reshape(N, -1)
    z_dist = torch.cdist(z_flat, pool_flat, p=2)            # (B, N) state similarity

    sim_mask = z_dist <= similarity_radius
    # Guarantee >= K candidates per anchor: relax underfilled rows to K-nearest by z.
    counts = sim_mask.sum(dim=-1)
    too_few = counts < K
    if too_few.any():
        nearest_z = z_dist.topk(K, dim=-1, largest=False).indices
        fill_mask = torch.zeros_like(sim_mask)
        fill_mask.scatter_(1, nearest_z, True)
        sim_mask = torch.where(too_few.unsqueeze(-1), fill_mask, sim_mask)

    # True-effect vectors Δz, compared as (B, N) divergence between the anchor's
    # Δz and each candidate's Δz (isolates the *change*, removing the residual
    # state offset that survives the similarity gate).
    d_anchor = (z_t1 - z_t).reshape(B, -1)
    d_pool = (pool_z1 - pool_z).reshape(N, -1)
    eff_div = torch.cdist(d_anchor, d_pool, p=2)            # (B, N) bigger = more different effect
    a_dist = torch.cdist(a_t, pool_a.to(a_t.device), p=2)   # (B, N) bigger = more different action

    def _rescale(x: torch.Tensor) -> torch.Tensor:
        vals = x[sim_mask]
        s = vals.std() if vals.numel() > 1 else x.new_tensor(1.0)
        return x / (s + 1e-8)

    scores = _rescale(eff_div) - action_penalty * _rescale(a_dist)
    scores = scores.masked_fill(~sim_mask, -float("inf"))

    topk_idx = scores.topk(K, dim=-1).indices              # (B, K)
    return pool_a.to(a_t.device)[topk_idx]                 # (B, K, A)


# ---------- dispatch --------------------------------------------------------

def sample_negatives(
    strategy: str,
    *,
    a_t: torch.Tensor,
    K: int = 16,
    action_bounds=None,
    l1_radius: Optional[float] = None,
    sigma: float = 0.1,
    gripper_dim: Optional[int] = None,
    z_t: Optional[torch.Tensor] = None,
    z_t1: Optional[torch.Tensor] = None,
    pool_z: Optional[torch.Tensor] = None,
    pool_a: Optional[torch.Tensor] = None,
    pool_z1: Optional[torch.Tensor] = None,
    similarity_radius: float = 0.5,
    action_penalty: float = 0.5,
) -> torch.Tensor:
    if strategy == "random":
        assert action_bounds is not None, "random_negative needs action_bounds"
        return random_negative(a_t, action_bounds, K=K, l1_radius=l1_radius)
    if strategy == "opposite":
        return opposite_negative(a_t, sigma=sigma, K=K, gripper_dim=gripper_dim)
    if strategy == "hard_nn":
        assert z_t is not None and pool_z is not None and pool_a is not None
        return hard_nn_negative(z_t, a_t, pool_z, pool_a, K=K,
                                similarity_radius=similarity_radius)
    if strategy == "hard_effect":
        assert (z_t is not None and z_t1 is not None and pool_z is not None
                and pool_z1 is not None and pool_a is not None), \
            "hard_effect needs z_t, z_t1, pool_z, pool_z1, pool_a"
        return hard_effect_negative(z_t, z_t1, a_t, pool_z, pool_z1, pool_a, K=K,
                                    similarity_radius=similarity_radius,
                                    action_penalty=action_penalty)
    raise ValueError(f"Unknown negative strategy: {strategy}")
