"""Small supervised stopping policies; inference uses only CEM prefixes.

This is fitted optimal stopping, not a guarantee about model ranking errors.
All physical outcome labels are confined to fit_policy().
"""
from __future__ import annotations

import numpy as np

from .rules import CHECKPOINTS, TOPK, elite_mean_cost

FEATURE_NAMES = (
    'log_initial_cost', 'log_current_elite_cost', 'log_carried_mean_cost',
    'elite_over_initial', 'carried_over_initial', 'elite_progress',
    'iqr_contraction', 'elite_band_over_initial_iqr', 'boundary_gap_over_initial_iqr',
    'proposal_std', 'mean_shift', 'shift_over_std',
)


def features(costs, means, std, index):
    """index is a checkpoint index; access no arrays after that checkpoint."""
    k = CHECKPOINTS[index]
    c0 = np.asarray(costs[0], dtype=float)
    c = np.asarray(costs[k - 1], dtype=float)
    s = np.sort(c)
    base = max(abs(float(np.median(c0))), 1e-12)
    scale = max(float(np.percentile(c0, 75) - np.percentile(c0, 25)), 1e-12)
    ec = float(s[:TOPK].mean())
    old = elite_mean_cost(costs[CHECKPOINTS[index - 1] - 1]) if index else ec
    move = float(np.sqrt(np.mean((means[index] - means[index - 1]) ** 2))) if index else 0.
    sigma = float(std[k - 1])
    x = np.array([
        np.log(max(base, 1e-12)), np.log(max(ec, 1e-12)),
        np.log(max(float(c[0]), 1e-12)), ec / base, float(c[0]) / base,
        (old - ec) / base, (np.percentile(c, 75) - np.percentile(c, 25)) / scale,
        (s[TOPK - 1] - s[0]) / scale, (s[TOPK] - s[TOPK - 1]) / scale,
        sigma, move, move / max(sigma, 1e-12),
    ], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite stopping features')
    return np.clip(x, -1e6, 1e6)


def trace_features(costs, means, std):
    return np.stack([features(costs, means, std, i) for i in range(len(CHECKPOINTS))])


def encode_tree(estimator):
    t = estimator.tree_
    return {'left': t.children_left.tolist(), 'right': t.children_right.tolist(),
            'feature': t.feature.tolist(), 'threshold': t.threshold.tolist(),
            'value': t.value[:, 0, 0].tolist()}


def predict(tree, x):
    node = 0
    while tree['left'][node] != -1:
        node = tree['left'][node] if x[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
    return tree['value'][node]


def fit_tree(x, y, weights=None):
    from sklearn.tree import DecisionTreeRegressor
    est = DecisionTreeRegressor(max_depth=2, min_weight_fraction_leaf=.15,
                                min_impurity_decrease=.0005, random_state=0)
    est.fit(x, y, sample_weight=weights)
    return encode_tree(est)


def choose_index(policy, stage, costs, means, std):
    for i in range(len(CHECKPOINTS) - 1):
        if predict(policy['models'][str(stage)][i], features(costs, means, std, i)) <= 0:
            return i
    return len(CHECKPOINTS) - 1


def fit_policy(roots, penalty):
    """Backward fitted policy evaluation on training roots only.

    Continue targets follow the learned future rule; not clairvoyant best leaves.
    All branches of a root receive total weight one in the plan-2 regression.
    """
    n, m = len(roots), len(CHECKPOINTS)
    cp = np.array(CHECKPOINTS)
    x1 = np.stack([trace_features(r.costs1, r.extra['means1'], r.extra['std1']) for r in roots])
    pairs = [(a, b) for a, r in enumerate(roots) for b in range(m) if not r.p1_term[b]]
    models2 = [None] * (m - 1)
    selected2 = np.zeros((n, m), int)
    if pairs:
        x2 = np.stack([trace_features(roots[a].costs2[b], roots[a].extra['means2'][b],
                                     roots[a].extra['std2'][b]) for a, b in pairs])
        outcomes = np.stack([roots[a].leaf_success[b] for a, b in pairs]).astype(float)
        counts = np.bincount([a for a, _ in pairs], minlength=n)
        weights = np.array([1. / counts[a] for a, _ in pairs])
        selected = np.full(len(pairs), m - 1, int)
        rows = np.arange(len(pairs))
        for i in reversed(range(m - 1)):
            y = outcomes[rows, selected] - outcomes[:, i] - penalty * (cp[selected] - cp[i])
            models2[i] = fit_tree(x2[:, i], y, weights)
            cont = np.array([predict(models2[i], x) > 0 for x in x2[:, i]])
            selected = np.where(cont, selected, i)
        for (a, b), j in zip(pairs, selected):
            selected2[a, b] = j
    else:
        constant = {'left': [-1], 'right': [-1], 'feature': [-2], 'threshold': [-2.], 'value': [0.]}
        models2 = [constant for _ in models2]
    stop_values = np.empty((n, m))
    for a, r in enumerate(roots):
        for i in range(m):
            j = selected2[a, i]
            steps = cp[i] + (0 if r.p1_term[i] else cp[j])
            stop_values[a, i] = float(r.leaf_success[i, j]) - penalty * steps
    models1 = [None] * (m - 1)
    selected = np.full(n, m - 1, int)
    for i in reversed(range(m - 1)):
        y = stop_values[np.arange(n), selected] - stop_values[:, i]
        models1[i] = fit_tree(x1[:, i], y)
        cont = np.array([predict(models1[i], x) > 0 for x in x1[:, i]])
        selected = np.where(cont, selected, i)
    return {'kind': 'continuation', 'penalty': float(penalty), 'feature_names': FEATURE_NAMES,
            'models': {'0': models1, '1': models2}, 'checkpoints': CHECKPOINTS}


def path(policy, root):
    i = choose_index(policy, 0, root.costs1, root.extra['means1'], root.extra['std1'])
    j = None if root.p1_term[i] else choose_index(policy, 1, root.costs2[i], root.extra['means2'][i], root.extra['std2'][i])
    return i, j


def outcomes(policy, roots):
    ps = [path(policy, r) for r in roots]
    return {'success': np.array([r.leaf_success[i, 0 if j is None else j] for r, (i, j) in zip(roots, ps)], bool),
            'iterations': np.array([CHECKPOINTS[i] + (0 if j is None else CHECKPOINTS[j]) for i, j in ps]),
            'paths': ps}
