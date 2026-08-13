import numpy as np

from planning.pfcg import (
    fit_symmetric_probe_geometry,
    latent_l2,
    matched_random_geometry,
    pfcg_cost,
    projected_cost,
)


def test_pfcg_ignores_action_insensitive_nuisance_direction():
    plus = np.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
    minus = -plus
    geometry = fit_symmetric_probe_geometry(plus, minus)
    endpoints = np.array([[0.1, 100.0], [1.0, 0.0]])
    goal = np.zeros(2)

    assert np.argmin(latent_l2(endpoints, goal)) == 1
    assert geometry.rank == 1
    assert np.argmin(pfcg_cost(endpoints, goal, geometry)) == 0
    assert np.argmin(projected_cost(endpoints, goal, geometry)) == 0


def test_pfcg_inverse_weights_weaker_action_response_more():
    plus = np.array(
        [[10.0, 1.0], [-10.0, 1.0], [10.0, -1.0], [-10.0, -1.0]]
    )
    minus = -plus
    geometry = fit_symmetric_probe_geometry(plus, minus, ridge_fraction=0.01)
    costs = pfcg_cost(np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2), geometry)

    assert geometry.rank == 2
    assert costs[1] > costs[0]


def test_random_geometry_is_seed_deterministic_and_orthonormal():
    plus = np.eye(4)[:3]
    minus = -plus
    geometry = fit_symmetric_probe_geometry(plus, minus)
    first = matched_random_geometry(geometry, latent_dim=4, seed=7)
    second = matched_random_geometry(geometry, latent_dim=4, seed=7)

    np.testing.assert_allclose(first.basis, second.basis)
    np.testing.assert_allclose(first.basis.T @ first.basis, np.eye(first.rank), atol=1e-12)
