"""Synthetic validation of the state-grounded metric fix (models/probes).

Mechanism the tests must prove: when the boundary-relevant quantity occupies a
small subspace of the latent (so plain L2 underweights it), (1) the probe can
recover it; (2) the boundary-aware φ-metric makes a predictor's *object-channel*
action-sensitivity visible where plain L2 spread is noise-dominated; (3) the
φ planning cost prefers the action that moves the object to the goal even when
a distractor candidate is closer in plain L2.
"""

import numpy as np
import torch

from models.adapters.synthetic import PerfectModel
from models.probes import (
    ObjectProbe,
    ObjectDynamicsHead,
    BoundaryAwareMetricAdapter,
    boundary_aware_cost,
)

D = 64
OBJ = slice(0, 3)          # the planted "object" subspace (3 of 64 dims)


def _make_world(scale_obj=0.1, noise=1.0, seed=0):
    """Latents where dims 0:3 hold the object position at small scale and the
    remaining 61 dims are large-scale nuisance — the measured real-data geometry."""
    g = torch.Generator().manual_seed(seed)

    def sample(n):
        z = noise * torch.randn(n, D, generator=g)
        z[:, OBJ] = scale_obj * torch.randn(n, 3, generator=g)
        return z

    return sample


def _train_probe(sample, n_steps=300):
    torch.manual_seed(0)
    probe = ObjectProbe(latent_dim=D, out_dim=3, hidden=64)
    opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(n_steps):
        z = sample(128)
        loss = ((probe(z) - z[:, OBJ]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return probe.eval()


def test_probe_recovers_planted_subspace():
    sample = _make_world()
    probe = _train_probe(sample)
    z = sample(256)
    err = (probe(z) - z[:, OBJ]).norm(dim=-1).median().item()
    sd = z[:, OBJ].std().item()
    assert err < 0.5 * sd, f"probe error {err} vs object sd {sd}"


def test_metric_adapter_amplifies_object_sensitivity():
    """A predictor that moves ONLY the object dims under action changes is
    invisible to plain-L2 spread (nuisance dominates) but visible under φ."""
    sample = _make_world()
    probe = _train_probe(sample)

    def lookup(z, a):
        out = z.clone()
        out[:, 0] = out[:, 0] + 0.2 * torch.sign(a[:, 0])   # object responds
        out = out + 0.0 * a.sum(-1, keepdim=True)
        return out

    base = PerfectModel(lookup_fn=lookup, action_dim=2)
    wrapped = BoundaryAwareMetricAdapter(base, probe, s_z=1.0, s_g=0.1,
                                         beta=1.0, gamma=0.0)
    z = sample(64)
    a_pos = torch.zeros(64, 2); a_pos[:, 0] = +1
    a_neg = torch.zeros(64, 2); a_neg[:, 0] = -1
    gap_phi = (wrapped.predict(z, a_pos) - wrapped.predict(z, a_neg)).norm(dim=-1).mean()
    # φ-space gap ≈ 0.4/s_g = 4 sigma-units; plain latent gap is 0.4 vs ~11 of
    # typical inter-state L2 distance — the re-amplification factor is ~10×.
    assert gap_phi.item() > 2.0


def test_dynamics_head_learns_action_dependence_from_factual_data():
    """The grounded-dynamics fix: the branch is decided by an interaction of a
    latent-readable quantity (z[:,1] — the real analogue is ee–object geometry,
    decodable per V1) and the action. Training sees only FACTUAL (z, a, Δobj)
    samples — no counterfactual pairs — yet supervising Δobj directly lets h
    learn the interaction across samples (the cross-sample signal the 98k-dim
    L2 objective buries)."""
    g = torch.Generator().manual_seed(0)

    def batch(n, seed):
        gg = torch.Generator().manual_seed(seed)
        z = torch.randn(n, D, generator=gg)
        a = torch.randn(n, 2, generator=gg)
        branch = (a[:, 0] > z[:, 1]).float().unsqueeze(-1)
        dobj = torch.zeros(n, 3)
        dobj[:, 0] = (branch.squeeze(-1) * 2 - 1) * 0.05      # ±5 cm-scale move
        dobj += 0.002 * torch.randn(n, 3, generator=gg)
        return z, a, dobj

    torch.manual_seed(0)
    h = ObjectDynamicsHead(latent_dim=D, action_dim=2, hidden=64, a_emb_dim=16)
    opt = torch.optim.Adam(h.parameters(), lr=3e-3)
    for step in range(500):
        z, a, dobj = batch(128, seed=step)
        loss = ((h(z, a) - dobj) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    h.eval()

    z = torch.randn(128, D, generator=g)
    a_pos = torch.zeros(128, 2); a_pos[:, 0] = z[:, 1] + 0.5   # just across the boundary
    a_neg = torch.zeros(128, 2); a_neg[:, 0] = z[:, 1] - 0.5   # just below it
    with torch.no_grad():
        gap = (h(z, a_pos) - h(z, a_neg))[:, 0].mean().item()
    assert gap > 0.05, f"h did not learn the action interaction (gap {gap})"


def test_boundary_aware_cost_prefers_object_match():
    sample = _make_world()
    probe = _train_probe(sample)
    z_goal = sample(1)[0]

    # Candidate A: matches the goal's nuisance dims but the object is wrong.
    cand_a = z_goal.clone()
    cand_a[OBJ] = z_goal[OBJ] + 0.3
    # Candidate B: nuisance dims off by typical noise, object exactly right.
    cand_b = z_goal.clone() + 0.2 * torch.randn(D, generator=torch.Generator().manual_seed(5))
    cand_b[OBJ] = z_goal[OBJ]

    preds = torch.stack([cand_a, cand_b])
    plain = ((preds.reshape(2, -1) - z_goal.reshape(1, -1)) ** 2).mean(-1)
    cost = boundary_aware_cost(probe, z_goal, s_z=1.0, s_g=0.1, beta=1.0)
    aware = cost(preds, z_goal.reshape(1, -1))
    # Plain L2 prefers the object-wrong candidate A; the aware cost flips it.
    assert plain[0] < plain[1]
    assert aware[1] < aware[0]
