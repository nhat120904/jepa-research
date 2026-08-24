import numpy as np

from gfpr_h0.core import (
    build_feature_views,
    farthest_point_indices,
    select_with_gate,
)


def test_feature_views_are_finite_and_candidate_aligned():
    rng = np.random.default_rng(3)
    views = build_feature_views(
        rng.normal(size=(7, 2, 3)),
        rng.normal(size=(2, 3)),
        np.arange(7.0),
        np.arange(7.0)[::-1],
        rng.normal(size=(7, 5)),
        rng.normal(size=5),
        rng.normal(size=5),
        rng.normal(size=5),
    )
    assert set(views) == {"action_only", "proxy_action", "latent_context"}
    assert all(value.shape[0] == 7 for value in views.values())
    assert all(np.isfinite(value).all() for value in views.values())
    assert views["latent_context"].shape[1] > views["proxy_action"].shape[1]


def test_gate_falls_back_when_lower_bound_is_small():
    ungated, gated = select_with_gate(
        np.array([2.5, 1.0]), np.array([1.0, 0.1]), margin_cm=2.0
    )
    assert ungated == 0
    assert gated == -1


def test_farthest_points_are_unique():
    actions = np.arange(30, dtype=float).reshape(5, 2, 3)
    selected = farthest_point_indices(actions, np.zeros((2, 3)), 3)
    assert len(np.unique(selected)) == 3

