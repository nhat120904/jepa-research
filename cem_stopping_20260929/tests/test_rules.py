"""Plain-assert tests for rules and the offline tree evaluator (numpy only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cemstop.evaluate import TreeRoot, evaluate, kendall_with_zone, oracle, paired_bootstrap, path_of  # noqa: E402
from cemstop.rules import CHECKPOINTS, TOPK, Band, Converge, Fixed, Gap, choose, elite_mean_cost, grid  # noqa: E402


def shrinking_costs(seed: int = 0, n_iter: int = 30, n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([1.0 + np.exp(-0.3 * j) * rng.random(n) for j in range(n_iter)])


def test_fixed():
    c = shrinking_costs()
    for k in CHECKPOINTS:
        assert choose(Fixed(k), c) == k


def test_causal():
    """Changing iterations after the chosen checkpoint never changes the choice."""
    rng = np.random.default_rng(1)
    for rule_list in grid().values():
        for rule in rule_list:
            for seed in range(5):
                c = shrinking_costs(seed)
                k = choose(rule, c)
                garbage = c.copy()
                garbage[k:] = rng.random(garbage[k:].shape) * 100
                assert choose(rule, garbage) == k, (rule, k)


def test_converge():
    c = np.ones((30, 300))
    assert choose(Converge(1e-3), c) == CHECKPOINTS[1]  # first comparable checkpoint
    # Costs that keep dropping by 50% per checkpoint never trigger eps = 0.1.
    c = np.stack([np.full(300, 0.5 ** j) for j in range(30)])
    assert choose(Converge(0.1), c) == 30
    row = np.arange(300, dtype=float)[::-1]
    assert np.isclose(elite_mean_cost(row), np.mean(np.arange(TOPK)))


def test_gap_and_band_degenerate():
    c = np.ones((30, 300))
    assert choose(Gap(1e-3), c) == 1
    assert choose(Band(1e-3), c) == 1


def test_gap_monotone_in_tau():
    c = shrinking_costs(3)
    ks = [choose(Gap(t), c) for t in np.logspace(-4, 0, 9)]
    assert all(a >= b for a, b in zip(ks, ks[1:])), ks
    ks = [choose(Band(t), c) for t in np.logspace(-3, 0, 9)]
    assert all(a >= b for a, b in zip(ks, ks[1:])), ks


def synthetic_root(seed: int) -> TreeRoot:
    rng = np.random.default_rng(seed)
    n = len(CHECKPOINTS)
    p1_term = np.zeros(n, bool)
    p1_term[-1] = seed % 2 == 0
    leaf = rng.random((n, n)) > 0.5
    leaf[p1_term] = True
    costs2 = np.stack([shrinking_costs(seed * 10 + i) for i in range(n)])
    costs2[p1_term] = np.nan
    return TreeRoot(root=seed, costs1=shrinking_costs(seed), costs2=costs2, p1_term=p1_term,
                    leaf_success=leaf, leaf_dist=rng.random((n, n)), checkpoints=CHECKPOINTS,
                    extra={})


def test_evaluator():
    roots = [synthetic_root(s) for s in range(6)]
    for tr in roots:
        i1, i2 = path_of(Fixed(30), tr)
        assert i1 == len(CHECKPOINTS) - 1
        assert (i2 is None) == bool(tr.p1_term[i1])
    res = evaluate(Fixed(10), roots)
    idx = CHECKPOINTS.index(10)
    assert np.array_equal(res["success"], [tr.leaf_success[idx, idx] for tr in roots])
    assert np.all(res["iterations"] == 20)
    o = oracle(roots)
    assert np.all(o["any_leaf_success"] >= res["success"])
    b = paired_bootstrap(res["success"], res["success"])
    assert b["mean"] == 0 and b["lo"] == 0 and b["hi"] == 0


def test_kendall_zone():
    cost = np.array([0.0, 1.0, 2.0, 3.0])
    phys = np.array([0.0, 1.0, 2.0, 3.0])
    assert kendall_with_zone(cost, phys, 0.5)["accuracy"] == 1.0
    assert kendall_with_zone(-cost, phys, 0.5)["accuracy"] == 0.0
    assert kendall_with_zone(cost, phys, 10.0)["pairs"] == 0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"ALL {len(tests)} TESTS PASSED")
