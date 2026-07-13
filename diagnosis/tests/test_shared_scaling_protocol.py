import numpy as np

from scripts._shared_scaling_protocol import (
    fixed_physical_neighbours,
    physical_effect_scores,
    physical_regimes,
    physical_state_features,
)


def test_effect_score_is_model_independent_and_monotone():
    p0 = np.zeros((3, 3))
    p1 = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    scores, scales = physical_effect_scores(p0, p1, scales=np.ones(3))
    assert np.array_equal(scores, [0, 1, 2])
    assert np.array_equal(scales, np.ones(3))


def test_euler_wrap_is_not_a_giant_effect_and_state_embedding_is_wrap_safe():
    p0 = np.asarray([[0, 0, 0, np.pi - 0.01, 0, 0, 0]], float)
    p1 = np.asarray([[0, 0, 0, -np.pi + 0.01, 0, 0, 0]], float)
    scores, _ = physical_effect_scores(p0, p1, scales=np.ones(7))
    assert scores[0] < 0.03
    features, _, _ = physical_state_features(np.concatenate([p0, p1]))
    assert np.linalg.norm(features[0] - features[1]) < 3.0


def test_sensor_only_regimes_have_expected_precedence():
    p0 = np.asarray([[0, 0], [0, 0.8], [0, 0.1], [0, 0.1]])
    p1 = np.asarray([[0, 0], [0, 0.8], [0, 0.5], [0, 0.1]])
    regimes = physical_regimes(
        p0, p1, gripper_dim=1, motion_scores=np.asarray([0, 0, 3, 3]),
        motion_threshold=1)
    assert regimes.tolist() == [
        "free_space", "contact_manipulation", "gripper_actuation", "pre_grasp_proxy"]


def test_fixed_neighbours_are_shared_and_exclude_factual_index():
    features = np.arange(6, dtype=float)[:, None]
    pool_indices = np.arange(6)
    chosen = fixed_physical_neighbours(
        features, features, K=2, anchor_global_indices=np.arange(6),
        pool_global_indices=pool_indices)
    for anchor, rows in enumerate(chosen):
        assert anchor not in rows
        assert set(rows).issubset(set(range(6)))
