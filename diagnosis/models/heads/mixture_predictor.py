"""C1 fix — mixture-density (distributional) latent prediction + boundary supervision.

Design 2026-06-09 §4 / PAPER_IDEA.md C3. The diagnosed failure: a unimodal point
predictor trained with L2 *must* output the conditional mean of `z_{t+1}`. Near a
contact bifurcation (gripper centred → object lifts; 2–3° off → it doesn't) that
conditional distribution is bimodal, so the mean sits **between** the two futures —
the model is Boundary Blind (high BB) no matter how well it was trained. The fix is
representational, not more training signal:

* **MixtureDensityHead** — a small residual head on top of the *frozen* base
  predictor (trunk). For `(z_t, ẑ_base = F(z_t, a), a)` it outputs `K` Gaussian
  components `{π_k, μ_k = ẑ_base + Δ_k, σ_k}` over `z_{t+1}` and is trained with
  NLL. With K ≥ 2 the head can place probability mass on *both* sides of the
  bifurcation and let `π(a)` decide — representing "lift OR not-lift" instead of
  their L2 average.
* **Boundary-supervision head** — an auxiliary BCE head predicting the sharp
  boundary event `g_{t+1}` (object-moves indicator from Metaworld state). It shares
  the context trunk with the mixture parameters, forcing capacity onto exactly the
  transitions the smoothing failure hides (design §4.1; also the R3 mitigation —
  it discourages the mixture from collapsing to one component at the boundary).
* **MixturePredictorAdapter** — wraps (frozen base adapter + trained head) behind
  the standard ``WorldModelAdapter.predict`` contract, returning the **mode**
  (highest-π component mean). The BB runner and CEM planner consume it unchanged;
  near a boundary the argmax-π prediction switches *discontinuously* with the
  action, which is precisely the action-sensitivity that BB measures and the
  baselines lack.

Everything here trains on cached latents only — the encoder and base predictor stay
frozen (the "force-free hero": no new data, no force sensing, no re-encoding).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adapters import WorldModelAdapter

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -4.0, 4.0

# Metaworld 39-dim state layout (stratification/metaworld_regimes.py):
# ee xyz = [0:3], gripper = 3, object xyz = [4:7].
_MW_EE, _MW_GRIP, _MW_OBJ = slice(0, 3), slice(3, 4), slice(4, 7)


def metaworld_boundary_state_slice(state: torch.Tensor) -> torch.Tensor:
    """Direction D's boundary-relevant feature from the raw Metaworld state:
    ``[ee(3) ‖ gripper(1) ‖ object(3) ‖ object−ee(3)]`` → (B, 10).

    This is the information a vision-only latent demonstrably under-resolves
    (C1-only null result, 2026-06-10): whether the gripper is *about to make or
    miss contact*. The cache already stores ``state``; no visual re-encoding.
    """
    ee, grip, obj = state[:, _MW_EE], state[:, _MW_GRIP], state[:, _MW_OBJ]
    return torch.cat([ee, grip, obj, obj - ee], dim=-1)


MW_STATE_SLICE_DIM = 10


def flatten_tokens(z: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """View a single-frame latent ``(B, *frame_dims)`` as tokens ``(B, N, D)``.

    Real caches store ``(B, V, H, W, D)`` patch grids (N = V*H*W); synthetic
    tests use ``(B, d)`` (N = 1). The head is token-wise, so both work.
    """
    B = z.shape[0]
    return z.reshape(B, -1, latent_dim)


class MixtureDensityHead(nn.Module):
    """Token-wise residual MDN over the frozen base prediction.

    forward(z_t_tokens, base_tokens, action) ->
        pi_logits (B, K), mu (B, K, N, D), log_sigma (B, K), boundary_logit (B,)

    ``mu_k = base + Δ_k``: the trunk keeps doing what it learned; the head only
    has to model the *split* around it. σ is a per-component scalar (the NLL is
    over N·D dims, so a scalar already calibrates the residual scale).
    """

    def __init__(self, latent_dim: int, action_dim: int, K: int = 3,
                 hidden: int = 512, ctx_dim: int = 256, a_emb_dim: int = 64,
                 state_dim: int = 0, s_emb_dim: int = 64):
        super().__init__()
        if K < 1:
            raise ValueError("K must be >= 1")
        self.K = K
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.a_emb = nn.Sequential(nn.Linear(action_dim, a_emb_dim), nn.SiLU())
        # Direction D (Metaworld only): φ(boundary-relevant state slice) joins the
        # context, giving the head the geometry the vision latent under-resolves.
        self.s_emb = (nn.Sequential(nn.Linear(state_dim, s_emb_dim), nn.SiLU())
                      if state_dim > 0 else None)
        s_extra = s_emb_dim if state_dim > 0 else 0
        # Shared context from pooled state/motion features + action: every output —
        # Δ, π, σ AND the boundary logit — reads it, so the boundary BCE shapes
        # the same features the mixture uses (that is the "forces capacity onto
        # the boundary" mechanism, not a separate detached probe). Pooling is
        # mean+max over z_t tokens plus the mean of the trunk's predicted motion
        # (base − z_t): mean-pooling alone washes out the local ee–object
        # geometry the boundary event depends on.
        self.ctx = nn.Sequential(
            nn.Linear(3 * latent_dim + a_emb_dim + s_extra, ctx_dim), nn.SiLU(),
            nn.Linear(ctx_dim, ctx_dim), nn.SiLU(),
        )
        self.delta = nn.Sequential(
            nn.Linear(2 * latent_dim + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, K * latent_dim),
        )
        self.pi = nn.Linear(ctx_dim, K)
        self.log_sigma = nn.Linear(ctx_dim, K)
        self.boundary = nn.Linear(ctx_dim, 1)
        # Start near the base prediction (Δ ≈ 0) so early training is a calibrated
        # unimodal Gaussian around the trunk. The per-component bias is given a
        # small random offset rather than exact zeros: winner-take-all training
        # needs the symmetry broken or one component wins every sample forever.
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.normal_(self.delta[-1].bias, std=0.02)

    def forward(self, z_t_tokens: torch.Tensor, base_tokens: torch.Tensor,
                action: torch.Tensor, state: Optional[torch.Tensor] = None) -> dict:
        B, N, D = z_t_tokens.shape
        a = self.a_emb(action)                                    # (B, E)
        parts = [z_t_tokens.mean(dim=1),
                 z_t_tokens.max(dim=1).values,
                 (base_tokens - z_t_tokens).mean(dim=1), a]
        if self.s_emb is not None:
            if state is None:
                raise ValueError("head was built with state_dim > 0; pass state")
            parts.append(self.s_emb(state))
        c = self.ctx(torch.cat(parts, dim=-1))                    # (B, C)

        tok = torch.cat([z_t_tokens, base_tokens,
                         c.unsqueeze(1).expand(B, N, c.shape[-1])], dim=-1)
        delta = self.delta(tok).reshape(B, N, self.K, D).permute(0, 2, 1, 3)
        mu = base_tokens.unsqueeze(1) + delta                     # (B, K, N, D)

        return {
            "pi_logits": self.pi(c),                              # (B, K)
            "mu": mu,
            "log_sigma": self.log_sigma(c).clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX),
            "boundary_logit": self.boundary(c).squeeze(-1),       # (B,)
        }


def component_log_likelihoods(out: dict, z_t1_tokens: torch.Tensor) -> torch.Tensor:
    """Per-component Gaussian log-likelihoods ``log N(z; μ_k, σ_k² I)``. (B, K)."""
    B, K, N, D = out["mu"].shape
    z = z_t1_tokens.reshape(B, 1, N * D)
    mu = out["mu"].reshape(B, K, N * D)
    log_sigma = out["log_sigma"]                                  # (B, K)
    nd = float(N * D)
    sq = ((z - mu) ** 2).sum(dim=-1)                              # (B, K)
    return (-0.5 * sq / torch.exp(2 * log_sigma)
            - nd * log_sigma - 0.5 * nd * math.log(2 * math.pi))


def mixture_nll(out: dict, z_t1_tokens: torch.Tensor) -> torch.Tensor:
    """Per-sample NLL of ``z_{t+1}`` under the isotropic Gaussian mixture.

    -log Σ_k π_k N(z; μ_k, σ_k² I) over the N·D flattened latent. Returns (B,).
    """
    log_comp = component_log_likelihoods(out, z_t1_tokens)
    log_mix = torch.logsumexp(F.log_softmax(out["pi_logits"], dim=-1) + log_comp, dim=-1)
    return -log_mix


def total_loss(out: dict, z_t1_tokens: torch.Tensor,
               boundary_label: Optional[torch.Tensor] = None,
               lambda_b: float = 0.1, objective: str = "nll") -> dict:
    """L = L_pred(z_{t+1}) + λ_b · BCE(g_{t+1})  (design §4.2). Returns scalars.

    objective:
        "nll" — soft mixture NLL (the textbook MDN loss). On high-dim latents
            where the bifurcation lives in a small subspace this is prone to
            **mean collapse**: components share one μ and split only σ (a
            variance mixture), which improves NLL but leaves the mode prediction
            — and BB — unchanged. Measured on dino_wm_metaworld 2026-06-10.
        "wta" — winner-take-all / hard-EM: only each sample's best-likelihood
            component receives the regression gradient, and π is trained as a
            classifier of the winner. Specialises the means and makes π(z, a) a
            discrete mode selector. **Also measured null on real latents**: the
            winner is decided by the full 98k-dim residual, in which the
            bifurcation subspace (a few object tokens) is negligible, so the
            winner label is noise w.r.t. the boundary.
        "boundary" — supervised mode assignment (K=2): component 1 ≡ the
            "object moves" future, component 0 ≡ "doesn't move", assigned by the
            boundary label g (not by EM); π is trained as the g classifier, so
            mode switching ≡ predicted boundary-event switching. This removes
            the noisy-winner problem by *telling* the mixture which futures to
            separate — the most literal form of the design's boundary
            supervision.
    """
    if objective == "boundary":
        if boundary_label is None:
            raise ValueError("objective='boundary' needs boundary_label")
        if out["mu"].shape[1] < 2:
            raise ValueError("objective='boundary' needs K >= 2")
        g = boundary_label.long()                                 # (B,) in {0,1}
        log_comp = component_log_likelihoods(out, z_t1_tokens)    # (B, K)
        pred_nll = -log_comp.gather(1, g.unsqueeze(1)).squeeze(1).mean()
        pi_ce = F.cross_entropy(out["pi_logits"], g)
        loss = pred_nll + pi_ce
        nll = pred_nll
        bce = pi_ce.detach()
        if lambda_b > 0:
            head_bce = F.binary_cross_entropy_with_logits(
                out["boundary_logit"], boundary_label.float())
            loss = loss + lambda_b * head_bce
        return {"loss": loss, "nll": nll.detach(), "bce": bce}
    if objective == "wta":
        log_comp = component_log_likelihoods(out, z_t1_tokens)    # (B, K)
        winner = log_comp.argmax(dim=-1).detach()                 # (B,)
        pred_nll = -log_comp.gather(1, winner.unsqueeze(1)).squeeze(1).mean()
        pi_ce = F.cross_entropy(out["pi_logits"], winner)
        loss = pred_nll + pi_ce
        nll = pred_nll
    elif objective == "nll":
        nll = mixture_nll(out, z_t1_tokens).mean()
        loss = nll
    else:
        raise ValueError(f"unknown objective {objective!r}")
    bce = torch.tensor(0.0, device=loss.device)
    if boundary_label is not None and lambda_b > 0:
        bce = F.binary_cross_entropy_with_logits(
            out["boundary_logit"], boundary_label.float())
        loss = loss + lambda_b * bce
    return {"loss": loss, "nll": nll.detach(), "bce": bce.detach()}


class MixturePredictorAdapter(WorldModelAdapter):
    """Frozen base adapter + trained MDN head behind the standard interface.

    ``predict`` returns the highest-π component mean (mode-based scoring, design
    §4.3 — the drop-in choice for both the BB runner and CEM). The action handed
    to the head is normalized exactly as the base model normalizes it.
    """

    def __init__(self, base: WorldModelAdapter, head: MixtureDensityHead,
                 device: Optional[str] = None, state_via_proprio: bool = False,
                 base_proprio_dim: int = 4):
        """``state_via_proprio`` (the C1+D variant): callers hand the **full raw
        state** through the ``proprio_t`` argument (the diagnostic runner only
        forwards proprio); the wrapper passes ``state[:, :base_proprio_dim]`` on
        to the base model (Metaworld proprio = the first 4 state dims) and the
        boundary slice to the head."""
        self.base = base
        self.head = head.eval()
        self.spec = base.spec
        self.state_via_proprio = state_via_proprio
        self.base_proprio_dim = base_proprio_dim
        self.device = torch.device(device) if device else getattr(
            base, "device", torch.device("cpu"))
        self.head.to(self.device)

    def _head_action(self, a_t: torch.Tensor) -> torch.Tensor:
        B = a_t.shape[0]
        a = a_t.to(self.device).float().reshape(B, -1, self.base.action_dim())
        return self.base.normalize_action(a).reshape(B, -1)

    @torch.no_grad()
    def predict(self, z_t: torch.Tensor, a_t: torch.Tensor,
                proprio_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        state = None
        if self.state_via_proprio and proprio_t is not None:
            state_full = proprio_t.to(self.device).float()
            state = metaworld_boundary_state_slice(state_full)
            proprio_t = state_full[:, : self.base_proprio_dim]
        base_pred = self.base.predict(z_t, a_t, proprio_t=proprio_t)
        zt = flatten_tokens(z_t.to(self.device).float(), self.head.latent_dim)
        bt = flatten_tokens(base_pred.to(self.device).float(), self.head.latent_dim)
        out = self.head(zt, bt, self._head_action(a_t), state=state)
        k = out["pi_logits"].argmax(dim=-1)                       # (B,)
        mode = out["mu"][torch.arange(len(k), device=k.device), k] # (B, N, D)
        return mode.reshape(base_pred.shape)

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
        self.head.to(self.device)
        self.base.to(device)
        return self

    def eval(self):
        self.head.eval()
        self.base.eval()
        return self
