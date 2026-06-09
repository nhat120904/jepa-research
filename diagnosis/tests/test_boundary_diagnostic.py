"""Synthetic validation of the boundary diagnostic (design 2026-06-09 §3).

Mirrors test_metrics_synthetic.py: drive the production per-transition functions
with controllable synthetic predictors and assert the mechanism.

The thesis the metric must capture:
  * In **boundary** regions (true outcome is sensitive to small action changes) an
    action-ignoring model is *boundary blind* → high BB.
  * A model that uses actions correctly tracks the true sensitivity → low BB, and
    is **not** elevated in boundary regions.
"""

import numpy as np
import torch

from stratification import (
    state_neighbours,
    boundary_score_per_transition,
    calibrate_boundary_threshold,
    boundary_mask,
)
from metrics import boundary_sensitivities_per_transition, boundary_blindness
from models.adapters.synthetic import PerfectModel, ActionIgnoringModel


# ---------------------------------------------------------------------------
# boundary_regime: neighbourhood + score
# ---------------------------------------------------------------------------

def test_state_neighbours_stay_within_cluster():
    # Two tight clusters far apart → neighbours must come from the same cluster.
    g = torch.Generator().manual_seed(0)
    c0 = torch.randn(20, 8, generator=g) * 0.01
    c1 = torch.randn(20, 8, generator=g) * 0.01 + 10.0
    z = torch.cat([c0, c1], dim=0)
    idx, mask, valid = state_neighbours(z, similarity_radius=1.0, max_neighbours=5)
    assert valid.all()
    # Every in-radius neighbour of a cluster-0 anchor is also in cluster 0.
    for i in range(20):
        real = idx[i][mask[i]]
        assert (real < 20).all()
    # Self is never selected.
    assert not any(i in idx[i].tolist() for i in range(40))


def test_boundary_score_high_for_bifurcation_low_for_smooth():
    # Hand-built neighbourhood: anchor 0's neighbours have near-identical actions
    # but wildly different outcomes (a bifurcation); anchor 3's outcomes are flat.
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]])
    outcome = torch.tensor([0.0, 1.0, 0.0, 0.0])     # anchors 0/1 differ in neighbour spread
    # anchor 0 neighbours = {1, 2}: outcomes {1.0, 0.0} spread high, action spread tiny
    # anchor 3 neighbours = {0, 2}: outcomes {0.0, 0.0} spread zero
    idx = torch.tensor([[1, 2], [0, 2], [0, 1], [0, 2]])
    mask = torch.ones(4, 2, dtype=torch.bool)
    scores = boundary_score_per_transition(a, outcome, idx, mask)
    assert scores[0] > scores[3]
    assert scores[3] == 0.0           # flat outcomes → zero boundary score


def test_boundary_score_nan_without_neighbours():
    a = torch.zeros(3, 2)
    outcome = torch.zeros(3)
    idx = torch.zeros(3, 2, dtype=torch.long)
    mask = torch.zeros(3, 2, dtype=torch.bool)        # no real neighbours
    scores = boundary_score_per_transition(a, outcome, idx, mask)
    assert np.isnan(scores).all()


def test_calibrate_and_mask_select_top_scores():
    scores = np.array([0.1, 0.2, np.nan, 5.0, 4.0])
    thr = calibrate_boundary_threshold(scores, quantile=0.5)
    m = boundary_mask(scores, thr)
    assert m.dtype == bool
    assert m[3] and m[4]              # the two large scores are boundary
    assert not m[2]                  # nan is never boundary


# ---------------------------------------------------------------------------
# boundary_blindness: the metric mechanism
# ---------------------------------------------------------------------------

def _boundary_dataset(n_boundary=40, n_smooth=40, D=16, A=4, M=8, seed=0):
    """Build anchors + neighbourhoods where the per-anchor action→outcome gain is
    high (boundary) or near-zero (smooth). The gain is encoded in z[...,0] so a
    *perfect* model can reproduce it; an action-ignoring model cannot."""
    g = torch.Generator().manual_seed(seed)
    N = n_boundary + n_smooth
    W = torch.randn(D, A, generator=g) * 0.3
    gain = torch.cat([torch.ones(n_boundary), torch.full((n_smooth,), 0.05)])
    z = torch.randn(N, D, generator=g)
    z[:, 0] = gain                                    # gain readable from state
    is_boundary = torch.cat([torch.ones(n_boundary, dtype=torch.bool),
                             torch.zeros(n_smooth, dtype=torch.bool)])

    neigh_a = torch.randn(N, M, A, generator=g).clamp(-1, 1)
    # True outcome of each neighbour action from the anchor's local gain.
    delta = gain.view(N, 1, 1) * (neigh_a @ W.T)      # (N, M, D)
    neigh_out = delta.norm(dim=-1)                    # (N, M) scalar true outcome
    mask = torch.ones(N, M, dtype=torch.bool)

    def lookup(zb, ab):                               # F(z,a) = z + z[...,0]*(a@W.T)
        return zb + zb[:, 0:1] * (ab @ W.T)

    return z, neigh_a, neigh_out, mask, is_boundary.numpy(), lookup, A


def test_action_ignoring_is_boundary_blind():
    z, neigh_a, neigh_out, mask, is_boundary, _, A = _boundary_dataset()
    m = ActionIgnoringModel(action_dim=A)
    s_true, s_model = boundary_sensitivities_per_transition(m, z, neigh_a, neigh_out, mask)
    # Action-ignoring → identical predictions for every action → zero model spread.
    assert np.allclose(s_model, 0.0, atol=1e-6)
    bb = boundary_blindness(s_true, s_model)
    assert bb[is_boundary].mean() > bb[~is_boundary].mean()
    assert bb[is_boundary].mean() > 0.0


def test_perfect_model_tracks_sensitivity():
    z, neigh_a, neigh_out, mask, is_boundary, lookup, A = _boundary_dataset()
    perfect = PerfectModel(lookup_fn=lookup, action_dim=A)
    ignoring = ActionIgnoringModel(action_dim=A)

    st_p, sm_p = boundary_sensitivities_per_transition(perfect, z, neigh_a, neigh_out, mask)
    st_i, sm_i = boundary_sensitivities_per_transition(ignoring, z, neigh_a, neigh_out, mask)
    bb_p = boundary_blindness(st_p, sm_p)
    bb_i = boundary_blindness(st_i, sm_i)

    # The perfect model's prediction spread responds to action (non-zero).
    assert sm_p[is_boundary].mean() > sm_p[~is_boundary].mean()
    # It is far less boundary-blind than the action-ignoring model.
    assert bb_p.mean() < bb_i.mean()
    assert bb_p[is_boundary].mean() < bb_i[is_boundary].mean()


def test_boundary_blindness_constant_inputs_are_zero():
    # Degenerate population (no spread) must not divide-by-zero.
    bb = boundary_blindness(np.ones(10), np.ones(10))
    assert np.allclose(bb, 0.0)
