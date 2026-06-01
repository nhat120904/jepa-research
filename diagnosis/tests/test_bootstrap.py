import numpy as np

from metrics.bootstrap import bootstrap_ci


def test_iid_ci_contains_point():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=2000)
    ci = bootstrap_ci(x, seed=1)
    assert ci.low < ci.point < ci.high
    assert abs(ci.point - 0.5) < 0.02


def test_cluster_ci_is_wider_than_iid_for_correlated_data():
    """Transitions within a trajectory are correlated; the cluster bootstrap
    must produce a WIDER CI than the (over-confident) iid bootstrap."""
    rng = np.random.default_rng(0)
    n_groups, per = 10, 50
    group_means = rng.normal(0.0, 1.0, size=n_groups)
    samples = np.repeat(group_means, per)                 # identical within group
    groups = np.repeat(np.arange(n_groups), per)

    iid = bootstrap_ci(samples, groups=None, seed=1)
    clu = bootstrap_ci(samples, groups=groups, seed=1)
    iid_w = iid.high - iid.low
    clu_w = clu.high - clu.low
    assert clu_w > 1.5 * iid_w


def test_empty_returns_nan():
    ci = bootstrap_ci(np.array([]), seed=1)
    assert np.isnan(ci.point)
