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
    # Two anchors with M=2 neighbours each (gathered form). Anchor 0: neighbour
    # actions ~identical but outcomes far apart (a bifurcation). Anchor 1: outcomes
    # flat (smooth).
    anchor_a = torch.zeros(2, 3)
    neigh_a = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]])
    neigh_out = torch.tensor([[0.0, 1.0],     # spread high → boundary
                              [0.0, 0.0]])    # spread zero → smooth
    mask = torch.ones(2, 2, dtype=torch.bool)
    scores = boundary_score_per_transition(anchor_a, neigh_a, neigh_out, mask)
    assert scores[0] > scores[1]
    assert scores[1] == 0.0           # flat outcomes → zero boundary score


def test_boundary_score_nan_without_neighbours():
    anchor_a = torch.zeros(3, 2)
    neigh_a = torch.zeros(3, 2, 2)
    neigh_out = torch.zeros(3, 2)
    mask = torch.zeros(3, 2, dtype=torch.bool)        # no real neighbours
    scores = boundary_score_per_transition(anchor_a, neigh_a, neigh_out, mask)
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


# ---------------------------------------------------------------------------
# Runner core: true outcome, per-cell accumulation, finalisation
# ---------------------------------------------------------------------------

from scripts._boundary_diagnostic import (   # noqa: E402
    compute_true_outcome,
    accumulate_cell,
    finalize_rows,
    _load_runner_helpers,
)
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def test_compute_true_outcome_metaworld_uses_object_displacement():
    N, S = 6, 39
    z_t = torch.zeros(N, 8)
    z_t1 = torch.randn(N, 8)          # latent moves, but outcome must ignore it
    state_t = torch.zeros(N, S)
    state_t1 = torch.zeros(N, S)
    state_t1[:, OBJECT_SLICE] = torch.tensor([[0.3, 0.4, 0.0]]).repeat(N, 1)  # ‖·‖ = 0.5
    out = compute_true_outcome("metaworld", z_t, z_t1, state_t, state_t1)
    assert np.allclose(out, 0.5, atol=1e-5)


def test_compute_true_outcome_falls_back_to_latent_delta():
    z_t = torch.zeros(5, 4)
    z_t1 = torch.full((5, 4), 0.5)    # ‖Δz‖ = sqrt(4*0.25) = 1.0
    out = compute_true_outcome("droid", z_t, z_t1)
    assert np.allclose(out, 1.0, atol=1e-5)


def _pool_with_two_clusters(D=8, A=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    # Cluster A near 0 with high-variance outcomes (a bifurcation neighbourhood).
    za = torch.randn(30, D, generator=g) * 0.05
    outa = torch.tensor([0.0, 1.0]).repeat(15)
    # Cluster B near 10 with flat outcomes (a smooth neighbourhood).
    zb = torch.randn(30, D, generator=g) * 0.05 + 10.0
    outb = torch.full((30,), 0.5)
    pool_z = torch.cat([za, zb], dim=0)
    pool_a = torch.randn(60, A, generator=g).clamp(-1, 1)
    pool_outcome = torch.cat([outa, outb]).numpy()
    return pool_z, pool_a, pool_outcome, D, A


def test_accumulate_and_finalize_flags_boundary_regime():
    pool_z, pool_a, pool_outcome, D, A = _pool_with_two_clusters()
    m = ActionIgnoringModel(action_dim=A)
    dev = torch.device("cpu")
    g = torch.Generator().manual_seed(2)

    def cell(z_centre, n, tag):
        return {
            "z_t": torch.randn(n, D, generator=g) * 0.05 + z_centre,
            "a_t": torch.randn(n, A, generator=g).clamp(-1, 1),
            "proprio_t": None,
            "traj_tag": np.array([f"{tag}/{i % 3}" for i in range(n)]),
        }

    # Boundary-regime anchors sit near cluster A; free-space anchors near cluster B.
    acc = {"s_true": [], "s_model": [], "boundary_score": [],
           "traj_tag": [], "task": [], "regime": []}
    for centre, regime, tag in [(0.0, "contact_manipulation", "c"), (10.0, "free_space", "f")]:
        cdata = cell(centre, 18, tag)
        out = accumulate_cell(m, cdata, pool_z, pool_a, pool_outcome, device=dev,
                              similarity_radius=2.0, max_neighbours=8)
        n = len(out["s_true"])
        acc["s_true"].append(out["s_true"])
        acc["s_model"].append(out["s_model"])
        acc["boundary_score"].append(out["boundary_score"])
        acc["traj_tag"].append(out["traj_tag"])
        acc["task"].append(np.array(["all"] * n))
        acc["regime"].append(np.array([regime] * n))

    rows, thr = finalize_rows(
        "metaworld", "synthetic_action_ignoring",
        s_true=np.concatenate(acc["s_true"]), s_model=np.concatenate(acc["s_model"]),
        boundary_score=np.concatenate(acc["boundary_score"]),
        task=np.concatenate(acc["task"]), regime=np.concatenate(acc["regime"]),
        traj_tag=np.concatenate(acc["traj_tag"]), n_resamples=200,
    )
    by_regime = {r["regime"]: r for r in rows}
    # Action-ignoring model: predictions never move, so S_model ≈ 0 everywhere.
    assert np.allclose(np.concatenate(acc["s_model"]), 0.0, atol=1e-6)
    # The bifurcation neighbourhood (contact) is flagged boundary-blind; smooth isn't.
    assert by_regime["contact_manipulation"]["bb"] > by_regime["free_space"]["bb"]


def test_materialize_records_loads_state_for_outcome(tmp_path):
    """Integration: a fake HDF5 cache → materialize(want_state) → object-Δ outcome."""
    from data import LatentCache

    T, D, A, S = 6, 4, 3, 39
    g = torch.Generator().manual_seed(0)
    z = torch.randn(T, D, generator=g)
    action = torch.randn(T - 1, A, generator=g)
    proprio = torch.randn(T, 7, generator=g)
    state = torch.zeros(T, S)
    state[:, OBJECT_SLICE] = torch.cumsum(torch.ones(T, 3) * 0.1, dim=0)  # object drifts

    cache_path = tmp_path / "metaworld__synthetic.h5"
    with LatentCache(cache_path, mode="w") as c:
        c.write_trajectory("task0/0", z=z, action=action, proprio=proprio, state=state)

    helpers = _load_runner_helpers()
    with LatentCache(cache_path, mode="r") as c:
        records = helpers.build_transition_records(c, None, step=1, per_task=True)
        data = helpers.materialize_records(c, records, 1, want_proprio=True, want_state=True)

    assert data["state_t"] is not None and data["state_t1"].shape[-1] == S
    out = compute_true_outcome("metaworld", data["z_t"], data["z_t1"],
                               data["state_t"], data["state_t1"])
    # object moves a constant 0.1 in each of 3 dims per step → ‖Δobj‖ = sqrt(3)*0.1.
    assert np.allclose(out, np.sqrt(3) * 0.1, atol=1e-5)
