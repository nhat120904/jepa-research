"""Synthetic validation of the action-conditioned representation adapter φ = A(z)
(Track 2, models/heads/action_repr_adapter.py) — proves each of its four training
terms moves the model in the right direction, on a planted world where the
mechanism is known, mirroring tests/test_latent_metric.py's approach.

Planted world: an "object" subspace (3 of D dims) that carries the real per-frame
signal; the rest is noise. Each test isolates ONE loss term (the others would need
the real cache / a MuJoCo env, exercised only in tests/test_cem_mining.py and on
the GPU pipeline) and shows it improves a held-out BEFORE-vs-AFTER metric with the
exact loss function `scripts/37_train_repr_adapter.py` uses. This CLS-attention
transformer (matching `SpatialObjectProbe`'s architecture) converges slowly from a
random init on a handful of SGD steps — production training runs full epochs over
thousands of real transitions (Test-1b already showed the same architecture reaches
92% <5cm there). A before/after relative check is therefore the right correctness
signal here: it catches a broken loss (no improvement) without requiring the toy
run to reach production-grade absolute precision.
"""

from __future__ import annotations

import numpy as np
import torch

from metrics.negative_samplers import hard_effect_negative
from models.heads.action_repr_adapter import ActionReprAdapter, margin_loss

D = 16
OBJ = slice(0, 3)
OBJ_DIM = 3
NOISE = 0.3


def _make_model(phi_dim=10, seed=0):
    torch.manual_seed(seed)
    return ActionReprAdapter(latent_dim=D, phi_dim=phi_dim, obj_dim=OBJ_DIM,
                             n_layers=1, n_heads=2, hidden=32)


# ---------------------------------------------------------------------------
# Term 1: grounding (MSE regression of phi[:, :obj_dim] to the true object).
# ---------------------------------------------------------------------------

def _make_grounding_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    obj = 0.3 * torch.randn(n, 3, generator=g)
    z = NOISE * torch.randn(n, D, generator=g)
    z[:, OBJ] = obj
    return z, obj


def test_term1_grounding_reduces_held_out_error():
    model = _make_model()
    z_train, obj_train = _make_grounding_data(256, seed=0)
    z_val, obj_val = _make_grounding_data(64, seed=999)

    model.eval()
    with torch.no_grad():
        err_before = (model(z_val)[:, :OBJ_DIM] - obj_val).norm(dim=-1).median()

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(200):
        idx = torch.randperm(256)[:32]
        pred = model(z_train[idx])[:, :OBJ_DIM]
        loss = ((pred - obj_train[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        err_after = (model(z_val)[:, :OBJ_DIM] - obj_val).norm(dim=-1).median()
    assert float(err_after) < 0.7 * float(err_before), (
        f"grounding should cut held-out median error: before={err_before:.4f} after={err_after:.4f}")


# ---------------------------------------------------------------------------
# Term 2: cf-contrastive margin (in-batch hard-effect negatives).
# ---------------------------------------------------------------------------

def _cf_satisfied_frac(model, sample_batch, seed, K, cap):
    zt, at, zt1, objt1 = sample_batch(seed)
    with torch.no_grad():
        obj_est = model(zt1)[:, :OBJ_DIM]
        _, neg_idx = hard_effect_negative(zt, zt1, at, pool_z=zt, pool_z1=zt1, pool_a=at,
                                          K=K, similarity_radius=10.0,   # D is tiny noise: no radius gate
                                          action_penalty=0.0, return_indices=True)
        B = zt.shape[0]
        obj_neg_true = objt1[neg_idx.reshape(-1)].reshape(B, K, -1)
        phi_neg = model(zt1[neg_idx.reshape(-1)])[:, :OBJ_DIM].reshape(B, K, -1)
        true_gap = (objt1.unsqueeze(1) - obj_neg_true).norm(dim=-1)
        d_cf = (obj_est.unsqueeze(1) - phi_neg).norm(dim=-1)
        margin = true_gap.clamp(max=cap)
        return float((d_cf >= margin - 1e-3).float().mean())


def test_term2_cf_margin_improves_separation():
    """An untrained model has no reason to separate the object estimates of two
    truly-different-outcome transitions (same z_t, different a, different z_t1).
    Training with margin_loss on hard_effect_negative pairs must raise the
    held-out margin-satisfied fraction."""
    model = _make_model()
    B, K, cap = 24, 4, 0.15

    def sample_batch(seed):
        gg = torch.Generator().manual_seed(seed)
        zt = NOISE * torch.randn(B, D, generator=gg)
        at = torch.randn(B, 4, generator=gg)
        objt1 = 0.3 * torch.randn(B, 3, generator=gg)
        zt1 = NOISE * torch.randn(B, D, generator=gg)
        zt1[:, OBJ] = objt1
        return zt, at, zt1, objt1

    model.eval()
    before = _cf_satisfied_frac(model, sample_batch, seed=12345, K=K, cap=cap)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for step in range(150):
        zt, at, zt1, objt1 = sample_batch(step)
        obj_est = model(zt1)[:, :OBJ_DIM]
        _, neg_idx = hard_effect_negative(zt, zt1, at, pool_z=zt, pool_z1=zt1, pool_a=at,
                                          K=K, similarity_radius=10.0, action_penalty=0.0,
                                          return_indices=True)
        obj_neg_true = objt1[neg_idx.reshape(-1)].reshape(B, K, -1)
        phi_neg = model(zt1[neg_idx.reshape(-1)])[:, :OBJ_DIM].reshape(B, K, -1)
        true_gap = (objt1.unsqueeze(1) - obj_neg_true).norm(dim=-1).reshape(-1)
        obj_est_rep = obj_est.unsqueeze(1).expand(-1, K, -1).reshape(-1, OBJ_DIM)
        loss = margin_loss(obj_est_rep, phi_neg.reshape(-1, OBJ_DIM), true_gap, cap)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    after = _cf_satisfied_frac(model, sample_batch, seed=12345, K=K, cap=cap)
    assert after > before + 0.15, (
        f"cf-margin satisfied fraction should improve: before={before:.2f} after={after:.2f}")


# ---------------------------------------------------------------------------
# Term 3: temporal ranking on phi_extra.
# ---------------------------------------------------------------------------

def _make_trajectories(n_traj, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n_traj):
        start = 0.3 * torch.randn(3, generator=g)
        end = 0.3 * torch.randn(3, generator=g)
        zs = []
        for t in range(T):
            frac = t / (T - 1)
            z = NOISE * torch.randn(D, generator=g)
            z[OBJ] = (1 - frac) * start + frac * end
            zs.append(z)
        trajs.append(torch.stack(zs))
    return trajs


def _sample_ranking(trajs, batch, rng):
    near, far, goal = [], [], []
    n = len(trajs)
    for _ in range(batch):
        tr = trajs[rng.integers(n)]
        T = tr.shape[0]
        i, j = (int(x) for x in rng.integers(0, T - 1, size=2))
        while j == i:
            j = int(rng.integers(0, T - 1))
        ni, fi = (i, j) if (T - 1 - i) < (T - 1 - j) else (j, i)
        near.append(tr[ni]); far.append(tr[fi]); goal.append(tr[-1])
    return torch.stack(near), torch.stack(far), torch.stack(goal)


def _ranking_accuracy(model, trajs, n, seed):
    with torch.no_grad():
        near, far, goal = _sample_ranking(trajs, n, np.random.default_rng(seed))
        phi_goal = model(goal)[:, OBJ_DIM:]
        d_near = (model(near)[:, OBJ_DIM:] - phi_goal).norm(dim=-1)
        d_far = (model(far)[:, OBJ_DIM:] - phi_goal).norm(dim=-1)
        return float((d_near < d_far).float().mean())


def test_term3_temporal_ranking_improves_accuracy():
    model = _make_model(phi_dim=10)
    trajs = _make_trajectories(20, seed=0)
    held = _make_trajectories(10, seed=999)
    margin = 1.0

    model.eval()
    before = _ranking_accuracy(model, held, 200, seed=1)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    rng = np.random.default_rng(0)
    model.train()
    for _ in range(200):
        near, far, goal = _sample_ranking(trajs, 24, rng)
        phi_goal = model(goal)[:, OBJ_DIM:]
        d_near = (model(near)[:, OBJ_DIM:] - phi_goal).norm(dim=-1)
        d_far = (model(far)[:, OBJ_DIM:] - phi_goal).norm(dim=-1)
        loss = torch.relu(margin + d_near - d_far).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    after = _ranking_accuracy(model, held, 200, seed=1)
    assert after > before + 0.1, (
        f"temporal ranking accuracy should improve: before={before:.3f} after={after:.3f}")
    assert after > 0.6, f"held-out temporal ranking accuracy too low: {after:.3f}"


# ---------------------------------------------------------------------------
# Term 4: adversarial margin (mined "exploited pocket" pairs).
# ---------------------------------------------------------------------------

def _sample_pockets(n, seed):
    g = torch.Generator().manual_seed(seed)
    z_goal = NOISE * torch.randn(n, D, generator=g)
    obj_goal = 0.3 * torch.randn(n, 3, generator=g)
    z_goal[:, OBJ] = obj_goal
    # "exploited" frame: encoder-level noise differs but object is FAR.
    z_adv = NOISE * torch.randn(n, D, generator=g)
    obj_adv = obj_goal + 0.25 * torch.sign(torch.randn(n, 3, generator=g))
    z_adv[:, OBJ] = obj_adv
    return z_adv, z_goal, obj_adv, obj_goal


def _adv_satisfied_frac(model, n, seed, cap):
    z_adv, z_goal, obj_adv, obj_goal = _sample_pockets(n, seed)
    with torch.no_grad():
        phi_a = model(z_adv)[:, :OBJ_DIM]
        phi_g = model(z_goal)[:, :OBJ_DIM]
        d = (phi_a - phi_g).norm(dim=-1)
        margin = (obj_adv - obj_goal).norm(dim=-1).clamp(max=cap)
        return float((d >= margin - 1e-3).float().mean())


def test_term4_adversarial_margin_improves_separation():
    """Construct a pocket the untrained model collapses (phi_obj(z_adv) ~=
    phi_obj(z_goal) despite a large TRUE object gap) and show the margin loss
    raises the held-out margin-satisfied fraction on pockets of the same
    construction."""
    model = _make_model()
    cap = 0.2

    model.eval()
    before = _adv_satisfied_frac(model, 64, seed=54321, cap=cap)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for step in range(150):
        z_adv, z_goal, obj_adv, obj_goal = _sample_pockets(24, step)
        phi_a = model(z_adv)[:, :OBJ_DIM]
        phi_g = model(z_goal)[:, :OBJ_DIM]
        true_gap = (obj_adv - obj_goal).norm(dim=-1)
        loss = margin_loss(phi_a, phi_g, true_gap, cap)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    after = _adv_satisfied_frac(model, 64, seed=54321, cap=cap)
    assert after > before + 0.15, (
        f"adversarial margin satisfied fraction should improve: before={before:.2f} after={after:.2f}")
