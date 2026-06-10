"""Synthetic validation of the C1 fix (mixture-density + boundary supervision).

The mechanism the tests must prove (mirrors test_boundary_diagnostic.py):

* A bifurcating world (small action change → one of two far-apart futures) is
  exactly where a unimodal (K=1) head is forced to predict the average. A K≥2
  mixture head trained with NLL on the same data must (a) reach a much better
  NLL, (b) produce **mode** predictions that switch with the action, and (c)
  therefore recover the model-side action sensitivity that Boundary Blindness
  measures — BB(fixed) < BB(frozen action-ignoring base) on boundary anchors.
"""

import numpy as np
import torch

from metrics import boundary_sensitivities_per_transition, boundary_blindness
from models.adapters.synthetic import ActionIgnoringModel
from models.heads import (
    MixtureDensityHead,
    MixturePredictorAdapter,
    mixture_nll,
    total_loss,
    flatten_tokens,
)

D, A = 8, 2


def _bifurcation_batch(n, seed=0, boundary=True):
    """World: z_{t+1} = z_t + u if a[0] > 0 else z_t - u (+ small noise).

    The frozen base trunk in these tests is ActionIgnoringModel (returns z_t),
    so *all* action sensitivity must come from the head — the hardest case the
    critique raises (the trunk carries zero boundary signal to amplify).
    """
    g = torch.Generator().manual_seed(seed)
    z_t = torch.randn(n, D, generator=g)
    # Boundary membership must be readable from the state (as it is in the real
    # data: ee–object geometry); encode it in z[:, 1].
    z_t[:, 1] = 1.0 if boundary else -1.0
    a = torch.randn(n, A, generator=g)
    u = torch.zeros(D)
    u[0] = 3.0
    if boundary:
        sign = torch.where(a[:, 0] > 0, 1.0, -1.0).unsqueeze(-1)
        z_t1 = z_t + sign * u + 0.05 * torch.randn(n, D, generator=g)
        g_label = torch.ones(n)
    else:  # smooth region: future does not depend on the action
        z_t1 = z_t + 0.05 * torch.randn(n, D, generator=g)
        g_label = torch.zeros(n)
    return z_t, a, z_t1, g_label


def _train_head(K, n_steps=400, lambda_b=0.1, seed=0, objective="nll"):
    torch.manual_seed(seed)
    head = MixtureDensityHead(latent_dim=D, action_dim=A, K=K, hidden=64,
                              ctx_dim=32, a_emb_dim=16)
    base = ActionIgnoringModel(action_dim=A)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    for step in range(n_steps):
        zb, ab, z1b, gb = _bifurcation_batch(128, seed=step, boundary=(step % 2 == 0))
        base_pred = base.predict(zb, ab)
        out = head(flatten_tokens(zb, D), flatten_tokens(base_pred, D), ab)
        losses = total_loss(out, flatten_tokens(z1b, D), gb, lambda_b=lambda_b,
                            objective=objective)
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
    return head, base


def _eval_nll(head, base, seed=999):
    z, a, z1, _ = _bifurcation_batch(512, seed=seed, boundary=True)
    with torch.no_grad():
        out = head(flatten_tokens(z, D), flatten_tokens(base.predict(z, a), D), a)
        return mixture_nll(out, flatten_tokens(z1, D)).mean().item()


# ---------------------------------------------------------------------------
# Module mechanics
# ---------------------------------------------------------------------------

def test_shapes_and_zero_init_starts_at_base():
    head = MixtureDensityHead(latent_dim=D, action_dim=A, K=3, hidden=32, ctx_dim=16)
    z, a, z1, _ = _bifurcation_batch(5)
    out = head(flatten_tokens(z, D), flatten_tokens(z, D), a)
    assert out["pi_logits"].shape == (5, 3)
    assert out["mu"].shape == (5, 3, 1, D)
    assert out["log_sigma"].shape == (5, 3)
    assert out["boundary_logit"].shape == (5,)
    # Δ weights are zero-initialised → every component mean starts ≈ the base
    # prediction, up to the small random per-component bias that breaks the
    # winner-take-all symmetry (std 0.02).
    assert torch.allclose(out["mu"],
                          flatten_tokens(z, D).unsqueeze(1).expand(5, 3, 1, D),
                          atol=0.1)


def test_total_loss_runs_without_boundary_label():
    head = MixtureDensityHead(latent_dim=D, action_dim=A, K=2, hidden=32, ctx_dim=16)
    z, a, z1, _ = _bifurcation_batch(8)
    out = head(flatten_tokens(z, D), flatten_tokens(z, D), a)
    losses = total_loss(out, flatten_tokens(z1, D), boundary_label=None)
    assert torch.isfinite(losses["loss"])
    assert losses["bce"].item() == 0.0


# ---------------------------------------------------------------------------
# The mechanism: K>=2 resolves the bifurcation that K=1 must average over
# ---------------------------------------------------------------------------

def test_mixture_beats_unimodal_nll_on_bifurcation():
    head2, base = _train_head(K=2)
    head1, _ = _train_head(K=1)
    nll2 = _eval_nll(head2, base)
    nll1 = _eval_nll(head1, base)
    # The bimodal future is ~6 latent units apart; a unimodal Gaussian must
    # widen σ to cover both modes, paying heavily in NLL.
    assert nll2 < nll1 - 1.0, f"K=2 NLL {nll2} not clearly below K=1 NLL {nll1}"


def test_mode_prediction_switches_with_action():
    head, base = _train_head(K=2)
    adapter = MixturePredictorAdapter(base, head)
    z = torch.randn(64, D)
    z[:, 1] = 1.0                       # boundary-region anchors (see _bifurcation_batch)
    a_pos = torch.zeros(64, A); a_pos[:, 0] = +0.8
    a_neg = torch.zeros(64, A); a_neg[:, 0] = -0.8
    pred_pos = adapter.predict(z, a_pos)
    pred_neg = adapter.predict(z, a_neg)
    gap = (pred_pos - pred_neg).norm(dim=-1).mean().item()
    # True modes are 2*3 = 6 apart along dim 0; the frozen base predicts a 0 gap.
    assert gap > 3.0, f"mode gap {gap} — the head did not split on the boundary"


def test_wta_objective_also_splits_modes():
    """The hard-EM objective (the real-data remedy for mean collapse) must
    reproduce the mode-switching mechanism on the synthetic bifurcation."""
    head, base = _train_head(K=2, objective="wta")
    adapter = MixturePredictorAdapter(base, head)
    z = torch.randn(64, D)
    z[:, 1] = 1.0
    a_pos = torch.zeros(64, A); a_pos[:, 0] = +0.8
    a_neg = torch.zeros(64, A); a_neg[:, 0] = -0.8
    gap = (adapter.predict(z, a_pos) - adapter.predict(z, a_neg)).norm(dim=-1).mean().item()
    assert gap > 3.0, f"WTA mode gap {gap}"


def test_boundary_head_learns_the_event():
    head, base = _train_head(K=2, lambda_b=0.3)
    zb, ab, _, _ = _bifurcation_batch(256, seed=1234, boundary=True)
    zs, as_, _, _ = _bifurcation_batch(256, seed=4321, boundary=False)
    with torch.no_grad():
        lb = head(flatten_tokens(zb, D), flatten_tokens(base.predict(zb, ab), D), ab)
        ls = head(flatten_tokens(zs, D), flatten_tokens(base.predict(zs, as_), D), as_)
    assert lb["boundary_logit"].mean() > ls["boundary_logit"].mean() + 1.0


def test_state_conditioning_resolves_aliased_boundary():
    """Failure point (a), the C1-only null measured on real latents 2026-06-10:
    when the latent does not encode which side of the boundary the system is on
    (state aliasing), no predictor head can decide which future to select — but
    conditioning the head on the boundary-relevant state slice (direction D)
    restores it. Dynamics: z1 = z ± u with the branch decided by a[0] > s_thr,
    where s_thr exists ONLY in the side-channel state, not in z."""
    u = torch.zeros(D); u[0] = 3.0

    def batch(n, seed):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, D, generator=g)
        a = torch.randn(n, A, generator=g)
        s_thr = torch.rand(n, 1, generator=g) * 2 - 1          # state-only info
        sign = torch.where(a[:, 0:1] > s_thr, 1.0, -1.0)
        z1 = z + sign * u + 0.05 * torch.randn(n, D, generator=g)
        return z, a, z1, torch.cat([s_thr, a[:, 0:1] - s_thr], dim=-1)

    def train(state_dim):
        torch.manual_seed(0)
        head = MixtureDensityHead(latent_dim=D, action_dim=A, K=2, hidden=64,
                                  ctx_dim=32, a_emb_dim=16, state_dim=state_dim)
        base = ActionIgnoringModel(action_dim=A)
        opt = torch.optim.Adam(head.parameters(), lr=5e-3)
        for step in range(500):
            z, a, z1, s = batch(128, seed=step)
            out = head(flatten_tokens(z, D), flatten_tokens(base.predict(z, a), D),
                       a, state=s if state_dim else None)
            losses = total_loss(out, flatten_tokens(z1, D), objective="wta")
            opt.zero_grad(); losses["loss"].backward(); opt.step()
        return head, base

    def mode_error(head, base, state_dim):
        z, a, z1, s = batch(512, seed=9999)
        with torch.no_grad():
            out = head(flatten_tokens(z, D), flatten_tokens(base.predict(z, a), D),
                       a, state=s if state_dim else None)
            k = out["pi_logits"].argmax(-1)
            mode = out["mu"][torch.arange(len(k)), k].reshape(z.shape)
        return (mode - z1).norm(dim=-1).mean().item()

    h_c1, base = train(state_dim=0)
    h_c1d, _ = train(state_dim=2)
    err_c1 = mode_error(h_c1, base, 0)
    err_c1d = mode_error(h_c1d, base, 2)
    # C1-only guesses the branch from the action alone (s_thr unknown) and must
    # be wrong often; C1+D should be near the noise floor (measured ≈0.16 vs ≈1.15).
    assert err_c1d < 0.5 * err_c1, f"C1+D {err_c1d} not clearly below C1 {err_c1}"


def test_boundary_supervised_assignment_splits_and_switches():
    """objective='boundary' (supervised mode assignment): component identity is
    given by the event label, π is the event classifier — mode switching equals
    predicted event switching. Must solve the aliased-boundary world when the
    state slice is provided."""
    u = torch.zeros(D); u[0] = 3.0

    def batch(n, seed):
        g_ = torch.Generator().manual_seed(seed)
        z = torch.randn(n, D, generator=g_)
        a = torch.randn(n, A, generator=g_)
        s_thr = torch.rand(n, 1, generator=g_) * 2 - 1
        branch = (a[:, 0:1] > s_thr).float()                   # the event label
        sign = branch * 2 - 1
        z1 = z + sign * u + 0.05 * torch.randn(n, D, generator=g_)
        return z, a, z1, branch.squeeze(1), torch.cat([s_thr, a[:, 0:1] - s_thr], -1)

    torch.manual_seed(0)
    head = MixtureDensityHead(latent_dim=D, action_dim=A, K=2, hidden=64,
                              ctx_dim=32, a_emb_dim=16, state_dim=2)
    base = ActionIgnoringModel(action_dim=A)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    for step in range(500):
        z, a, z1, g, s = batch(128, seed=step)
        out = head(flatten_tokens(z, D), flatten_tokens(base.predict(z, a), D), a, state=s)
        losses = total_loss(out, flatten_tokens(z1, D), g, objective="boundary")
        opt.zero_grad(); losses["loss"].backward(); opt.step()

    z, a, z1, g, s = batch(512, seed=9999)
    with torch.no_grad():
        out = head(flatten_tokens(z, D), flatten_tokens(base.predict(z, a), D), a, state=s)
        k = out["pi_logits"].argmax(-1)
        mode = out["mu"][torch.arange(len(k)), k].reshape(z.shape)
    err = (mode - z1).norm(dim=-1).mean().item()
    acc = (k.float() == g).float().mean().item()
    assert acc > 0.9, f"π did not learn the event (acc {acc})"
    assert err < 1.0, f"supervised mode prediction error {err}"


def test_bb_drops_versus_frozen_base():
    """End-to-end through the production BB machinery (the success criterion)."""
    head, base = _train_head(K=2)
    fixed = MixturePredictorAdapter(base, head)

    g = torch.Generator().manual_seed(7)
    B, M = 48, 8
    z = torch.randn(B, D, generator=g)
    z[:, 1] = 1.0                       # boundary-region anchors (see _bifurcation_batch)
    neigh_a = torch.randn(B, M, A, generator=g)
    # True outcome across the neighbourhood bifurcates with the action sign.
    u = torch.zeros(D); u[0] = 3.0
    sign = torch.where(neigh_a[:, :, 0] > 0, 1.0, -1.0)
    neigh_out = (sign.unsqueeze(-1) * u).norm(dim=-1) * sign  # (B, M), ±3
    mask = torch.ones(B, M, dtype=torch.bool)

    st_b, sm_b = boundary_sensitivities_per_transition(base, z, neigh_a, neigh_out, mask)
    st_f, sm_f = boundary_sensitivities_per_transition(fixed, z, neigh_a, neigh_out, mask)

    assert np.allclose(sm_b, 0.0, atol=1e-6)        # frozen base: blind
    assert sm_f.mean() > 1.0                        # fixed model: tracks the split
    # Standardise over the joint population (same reference for both models).
    st = np.concatenate([st_b, st_f]); sm = np.concatenate([sm_b, sm_f])
    bb = boundary_blindness(st, sm)
    bb_base, bb_fixed = bb[:B], bb[B:]
    assert bb_fixed.mean() < bb_base.mean() - 0.5
