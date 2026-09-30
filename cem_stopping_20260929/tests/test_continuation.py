"""Causality, training path consistency and serialized inference checks."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemstop.continuation import features, fit_policy, outcomes, choose_index
from cemstop.evaluate import TreeRoot
from cemstop.rules import CHECKPOINTS


def root(seed):
    rng = np.random.default_rng(seed)
    m = len(CHECKPOINTS)
    c1 = rng.uniform(.1, 1, (30, 300))
    c2 = rng.uniform(.1, 1, (m, 30, 300))
    return TreeRoot(seed, c1, c2, np.zeros(m, bool), rng.random((m, m)) > .5,
                    rng.random((m, m)), CHECKPOINTS,
                    {'means1': rng.normal(size=(m, 5, 10)), 'std1': np.ones(30),
                     'means2': rng.normal(size=(m, m, 5, 10)), 'std2': np.ones((m, 30))})


def main():
    import json
    roots = [root(i) for i in range(20)]
    policy = json.loads(json.dumps(fit_policy(roots[:15], .002)))
    for r in roots[15:]:
        for i, k in enumerate(CHECKPOINTS):
            x = features(r.costs1, r.extra['means1'], r.extra['std1'], i)
            c, m, s = r.costs1.copy(), r.extra['means1'].copy(), r.extra['std1'].copy()
            c[k:], m[i + 1:], s[k:] = np.nan, np.nan, np.nan
            assert np.array_equal(x, features(c, m, s, i))
        i = choose_index(policy, 0, r.costs1, r.extra['means1'], r.extra['std1'])
        c, m, s = r.costs1.copy(), r.extra['means1'].copy(), r.extra['std1'].copy()
        c[CHECKPOINTS[i]:], m[i + 1:], s[CHECKPOINTS[i]:] = np.nan, np.nan, np.nan
        assert choose_index(policy, 0, c, m, s) == i
    out = outcomes(policy, roots[15:])
    for r, (i, j), y, count in zip(roots[15:], out['paths'], out['success'], out['iterations']):
        assert y == r.leaf_success[i, j]
        assert count == CHECKPOINTS[i] + CHECKPOINTS[j]
    # No continuation is beneficial if all actions already succeed and cost > 0.
    for r in roots:
        r.leaf_success[:] = True
        r.p1_term[:] = True
    p = fit_policy(roots, .002)
    o = outcomes(p, roots)
    assert o['success'].all() and (o['iterations'] == 1).all()
    print('CONTINUATION_TESTS_OK')


if __name__ == '__main__':
    main()
