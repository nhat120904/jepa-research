import numpy as np

from scripts._same_state_intervention import make_local_action_fan, spearman, summarize_fan


def test_action_fan_is_deterministic_local_and_nested():
    nominal = np.zeros((40, 4), dtype=np.float32)
    fan_a, labels_a = make_local_action_fan(nominal, n_candidates=17, seed=9)
    fan_b, labels_b = make_local_action_fan(nominal, n_candidates=17, seed=9)
    assert fan_a.shape == (17, 40, 4)
    assert np.array_equal(fan_a, fan_b)
    assert labels_a == labels_b
    assert np.array_equal(fan_a[0], nominal)
    assert np.max(np.abs(fan_a)) <= 1.0
    # Horizon endpoints are prefixes of one maximum action fan, not re-sampled.
    assert np.array_equal(fan_a[:, :5], fan_b[:, :5])
    assert np.array_equal(fan_a[:, :20], fan_b[:, :20])


def test_spearman_handles_ties_and_monotonicity():
    assert np.isclose(spearman([0, 1, 1, 3], [0, 2, 2, 9]), 1.0)
    assert np.isclose(spearman([0, 1, 2], [9, 3, 0]), -1.0)


def test_fan_summary_perfect_model_has_perfect_causal_ranking_and_geometry():
    # Candidate zero is factual. Physical object outcomes and latent outcomes
    # have the same 1-D geometry; a perfect model predicts each true successor.
    true = np.asarray([[0.0], [1.0], [2.0], [4.0]])
    obj = np.c_[true[:, 0], np.zeros((4, 2))]
    stats = summarize_fan(true.copy(), true, obj)
    assert stats.causal_top1 == 1.0
    assert stats.causal_mrr == 1.0
    assert stats.factual_prediction_error == 0.0
    assert np.isclose(stats.effect_spearman, 1.0)
    assert np.isclose(stats.pairwise_spearman, 1.0)


def test_fan_summary_action_ignoring_model_gets_tie_aware_chance():
    true = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    pred = np.zeros_like(true)
    obj = np.c_[true[:, 0], np.zeros((4, 2))]
    stats = summarize_fan(pred, true, obj)
    assert np.isclose(stats.causal_top1, 0.25)
    # Average rank of four tied candidates is 2.5.
    assert np.isclose(stats.causal_mrr, 1 / 2.5)
    assert np.isnan(stats.effect_spearman)
