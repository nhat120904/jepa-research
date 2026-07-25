"""Belief-update correctness: normalization + hand-checkable Bayes."""

import numpy as np

from belief_compression.core import Belief
from belief_compression.tasks import HEAVY, LIGHT, MassSort, OccluderPush


def test_prior_sums_to_one():
    for task in (MassSort(n_objects=3), OccluderPush(n_locations=8, fine=4)):
        b = Belief.prior(task)
        assert abs(b.weights.sum() - 1.0) < 1e-12
        assert np.all(b.weights >= 0)


def test_posterior_sums_to_one_after_updates():
    task = MassSort(n_objects=3, probe_noise=0.2)
    b = Belief.prior(task)
    for probe, o in [("tap0", HEAVY), ("tap1", LIGHT), ("tap2", HEAVY)]:
        b = b.updated(probe, o)
        assert abs(b.weights.sum() - 1.0) < 1e-12
        assert np.all(b.weights >= -1e-15)


def test_bayes_hand_checkable():
    # N=1, uniform prior, tap noise 0.2. Observe HEAVY.
    # P(HEAVY|light)=0.2, P(HEAVY|heavy)=0.8; prior 0.5/0.5.
    # posterior propto [0.1, 0.4] -> [0.2, 0.8].
    task = MassSort(n_objects=1, p_heavy=0.5, probe_noise=0.2)
    b = Belief.prior(task)
    b = b.updated("tap0", HEAVY)
    # hidden_states order = [(LIGHT,), (HEAVY,)]
    assert task.hidden_states[0] == (LIGHT,)
    assert task.hidden_states[1] == (HEAVY,)
    np.testing.assert_allclose(b.weights, [0.2, 0.8], atol=1e-12)


def test_bayes_consistency_two_taps_equals_joint():
    # Two independent taps on the same object should compose multiplicatively.
    task = MassSort(n_objects=1, p_heavy=0.5, probe_noise=0.25)
    b = Belief.prior(task).updated("tap0", HEAVY).updated("tap0", HEAVY)
    # posterior propto [0.25^2, 0.75^2] = [0.0625, 0.5625] -> [0.1, 0.9]
    np.testing.assert_allclose(b.weights, [0.1, 0.9], atol=1e-12)


def test_occluder_look_is_bayes_consistent():
    task = OccluderPush(n_locations=8, fine=4, coarse_noise=0.1)
    b = Belief.prior(task)
    b2 = b.updated("look_coarse", 0)  # observed bin 0
    # bin-0 locations (0..3) should gain weight, bin-1 (4..7) lose it
    assert b2.weights[:4].sum() > b.weights[:4].sum()
    assert abs(b2.weights.sum() - 1.0) < 1e-12
