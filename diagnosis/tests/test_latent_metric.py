"""Synthetic validation of the learned latent metric (Track B, models/heads/latent_metric).

Mechanism the tests must prove on a planted world where the task-relevant quantity
(an "object" position) drifts monotonically along each trajectory while the rest of
the latent is i.i.d. nuisance — the measured real-data geometry (object = a few
patch tokens, the rest is arm-pose / background that carries no temporal signal):

1. ``d_θ`` trained *only* on the temporal order of frames (no object label) puts its
   minimum at the goal frame: ``d_θ(g, g) ≈ 0``.
2. ``d_θ(z_t, z_T)`` is monotone in steps-to-go (averaged over held-out trajectories,
   as the scripts/33 gate measures it) — a usable planning cost, unlike L2.
3. Because nuisance is temporally meaningless, the metric must key on the object
   subspace: its cost prefers an object-correct candidate over an L2-closer but
   object-wrong one — the exact flip plain L2 cannot make (cf. test_object_probe).

The metric is trained ONCE (module-scoped fixture) and shared across the tests.
"""

import numpy as np
import pytest
import torch

from models.heads.latent_metric import LatentMetric

D = 16
OBJ = slice(0, 3)          # planted "object" subspace (3 of 16 dims)
NOISE = 0.5                # nuisance per-dim std (object per-dim std is 0.1 -> 5:1)
T = 12
K = 10


def _make_trajectories(n_traj, obj_scale=0.1, noise=NOISE, seed=0):
    """Each trajectory: object moves linearly start→goal; the other dims are fresh
    noise every frame (no temporal structure → the only learnable temporal signal
    is the object subspace)."""
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n_traj):
        start = obj_scale * torch.randn(3, generator=g)
        end = obj_scale * torch.randn(3, generator=g)
        zs = []
        for t in range(T):
            frac = t / (T - 1)
            z = noise * torch.randn(D, generator=g)
            z[OBJ] = (1 - frac) * start + frac * end
            zs.append(z)
        trajs.append(torch.stack(zs))                      # (T, D)
    return trajs


def _train_metric(trajs, steps=1200, seed=0):
    torch.manual_seed(seed)
    m = LatentMetric(latent_dim=D, embed_dim=64, hidden=128)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    huber = torch.nn.SmoothL1Loss()
    rng = np.random.default_rng(seed)
    far = float(K + 4)
    for _ in range(steps):
        zi, zj, gaps = [], [], []
        for _ in range(64):
            tr = trajs[rng.integers(len(trajs))]
            i = int(rng.integers(T - 1))
            gp = int(rng.integers(1, min(K, T - 1 - i) + 1))
            zi.append(tr[i]); zj.append(tr[i + gp]); gaps.append(float(gp))
        zi = torch.stack(zi); zj = torch.stack(zj); gaps = torch.tensor(gaps)
        d = m(zi, zj)
        loss = huber(d, gaps) + torch.relu(far - m(zi, torch.roll(zj, 1, 0))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


@pytest.fixture(scope="module")
def trained():
    """Train d_θ once; return (metric, train_trajs)."""
    trajs = _make_trajectories(n_traj=48, seed=0)
    return _train_metric(trajs), trajs


def test_metric_zero_at_goal(trained):
    m, trajs = trained
    zg = trajs[0][-1:]
    with torch.no_grad():
        assert abs(m(zg, zg).item()) < 1e-3


def test_metric_monotone_to_goal(trained):
    """d_θ(z_t, z_T) must rank-order with steps-to-go, averaged over held-out
    trajectories (the scripts/33 gate quantity; per-trajectory is high-variance)."""
    m, _ = trained
    held = _make_trajectories(n_traj=12, seed=999)       # disjoint from training seed
    steps_to_go = np.arange(T)[::-1].astype(np.float64)
    sps = []
    for tr in held:
        with torch.no_grad():
            d = m(tr, tr[-1:].expand(T, D)).numpy()
        sps.append(_spearman(d, steps_to_go))
    assert float(np.mean(sps)) > 0.7, f"mean monotone Spearman {np.mean(sps):.3f}"


def test_metric_cost_prefers_object_progress(trained):
    """The metric keys on the object subspace, so it flips L2's wrong preference:
    an object-correct candidate scores lower than an L2-closer object-wrong one."""
    m, trajs = trained
    z_goal = trajs[0][-1]

    # A: matches goal nuisance dims, object wrong (L2-close). B: nuisance off by
    # typical noise, object exactly right.
    cand_a = z_goal.clone(); cand_a[OBJ] = z_goal[OBJ] + 0.3
    cand_b = z_goal.clone() + 0.2 * torch.randn(D, generator=torch.Generator().manual_seed(5))
    cand_b[OBJ] = z_goal[OBJ]

    preds = torch.stack([cand_a, cand_b])
    plain = ((preds - z_goal[None]) ** 2).mean(-1)
    with torch.no_grad():
        d = m(preds, z_goal[None].expand(2, D))
    assert plain[0] < plain[1]          # L2 prefers the object-wrong candidate A
    assert d[1] < d[0]                  # the learned metric flips it to B
