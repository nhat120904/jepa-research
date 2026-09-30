"""A deliberately small contextual-budget alternative to many stopping models.

One split per plan. First-population statistics choose a fixed budget for that
plan. All training choices are evaluated on withheld source episodes.
"""
import numpy as np
from .continuation import features
from .rules import CHECKPOINTS

# Locked after v2: only first-population goal cost and relative dispersion.
INPUTS = (0, 2, 7)
QUANTILES = (.25, .5, .75)
MIN_GAIN = .02


def initial_features(costs, means, std):
    return features(costs, means, std, 0)[list(INPUTS)]


def pick(stump, x):
    if stump['feature'] is None:
        return stump['default']
    return stump['left'] if x[stump['feature']] <= stump['threshold'] else stump['right']


def fit_stump(x, rewards, weights, costs, group_ids):
    """Optimize a constant budget or one split; avoid tiny trajectory subsets."""
    def action(mask):
        w = weights[mask]
        u = np.average(rewards[mask], weights=w, axis=0)
        c = np.average(costs[mask], weights=w, axis=0)
        return max(range(len(CHECKPOINTS)), key=lambda a: (u[a], -c[a]))
    whole = np.ones(len(x), bool)
    default = action(whole)
    base = float(np.average(rewards[:, default], weights=weights))
    best = {'feature': None, 'default': int(default), 'training_gain': 0.}
    best_value = base + MIN_GAIN
    min_groups = max(10, int(np.ceil(.2 * len(np.unique(group_ids)))))
    for f in range(x.shape[1]):
        for threshold in np.unique(np.quantile(x[:, f], QUANTILES)):
            left = x[:, f] <= threshold
            right = ~left
            if min(len(np.unique(group_ids[left])), len(np.unique(group_ids[right]))) < min_groups:
                continue
            a, b = action(left), action(right)
            if a == b:
                continue
            value = float(np.average(np.where(left, rewards[:, a], rewards[:, b]), weights=weights))
            if value > best_value:
                best_value = value
                best = {'feature': f, 'threshold': float(threshold), 'left': int(a), 'right': int(b),
                        'default': int(default), 'training_gain': value - base}
    return best


def fit_budget_uniform(roots, penalty):
    m, n = len(CHECKPOINTS), len(roots)
    cp = np.array(CHECKPOINTS)
    branches = [(a, b) for a, r in enumerate(roots) for b in range(m) if not r.p1_term[b]]
    if branches:
        x2 = np.stack([initial_features(roots[a].costs2[b], roots[a].extra['means2'][b],
                                       roots[a].extra['std2'][b]) for a, b in branches])
        c2 = np.broadcast_to(cp, (len(branches), m))
        rewards2 = np.stack([roots[a].leaf_success[b] for a, b in branches]).astype(float) - penalty * c2
        counts = np.bincount([a for a, _ in branches], minlength=n)
        weights = np.array([1. / counts[a] for a, _ in branches])
        stage2 = fit_stump(x2, rewards2, weights, c2, np.array([a for a, _ in branches]))
    else:
        stage2 = {'feature': None, 'default': 0, 'training_gain': 0.}
    second = np.zeros((n, m), int)
    for t, (a, b) in enumerate(branches):
        second[a, b] = pick(stage2, x2[t])
    rewards1, costs1 = np.empty((n, m)), np.empty((n, m))
    for a, r in enumerate(roots):
        for i in range(m):
            j = second[a, i]
            costs1[a, i] = cp[i] + (0 if r.p1_term[i] else cp[j])
            rewards1[a, i] = r.leaf_success[i, j] - penalty * costs1[a, i]
    x1 = np.stack([initial_features(r.costs1, r.extra['means1'], r.extra['std1']) for r in roots])
    stage1 = fit_stump(x1, rewards1, np.ones(n), costs1, np.arange(n))
    return {'kind': 'budget_stump', 'penalty': penalty, 'stages': [stage1, stage2],
            'features': ['initial_log_median', 'initial_log_carried_cost', 'initial_elite_band_over_iqr']}


def fit_budget(roots, penalty):
    """Two policy updates on reached branches, anchored at best fixed pair.

    Uniformly training plan 2 on all nine hypothetical plan-1 budgets can
    optimize states that the controller never visits. Refit on the branches
    reached by the current plan-1 rule, then improve that rule. This is a
    fixed two-update procedure; iteration count is not selected on validation.
    """
    cp = np.asarray(CHECKPOINTS)
    m, n = len(cp), len(roots)
    def score(spec):
        ev = outcomes(spec, roots)
        return (float(np.mean(ev['success'] - penalty * ev['iterations'])),
                -float(np.mean(ev['iterations'])))
    constants = []
    for i in range(m):
        for j in range(m):
            constants.append({'kind': 'budget_stump', 'penalty': penalty,
                              'stages': [{'feature': None, 'default': i}, {'feature': None, 'default': j}]})
    spec = max(constants, key=score)
    x1 = np.stack([initial_features(r.costs1, r.extra['means1'], r.extra['std1']) for r in roots])
    for _ in range(2):
        branches = [(a, pick(spec['stages'][0], x1[a])) for a in range(n)]
        branches = [(a, i) for a, i in branches if not roots[a].p1_term[i]]
        if branches:
            x2 = np.stack([initial_features(roots[a].costs2[i], roots[a].extra['means2'][i],
                                           roots[a].extra['std2'][i]) for a, i in branches])
            costs2 = np.broadcast_to(cp, (len(branches), m))
            values2 = np.stack([roots[a].leaf_success[i] for a, i in branches]).astype(float) - penalty * costs2
            stage2 = fit_stump(x2, values2, np.ones(len(branches)), costs2, np.array([a for a, _ in branches]))
        else:
            stage2 = spec['stages'][1]
        values1, costs1 = np.empty((n, m)), np.empty((n, m))
        for a, r in enumerate(roots):
            for i in range(m):
                j = 0 if r.p1_term[i] else pick(stage2, initial_features(r.costs2[i], r.extra['means2'][i], r.extra['std2'][i]))
                costs1[a, i] = cp[i] + (0 if r.p1_term[i] else cp[j])
                values1[a, i] = r.leaf_success[i, j] - penalty * costs1[a, i]
        stage1 = fit_stump(x1, values1, np.ones(n), costs1, np.arange(n))
        candidate = {'kind': 'budget_stump', 'penalty': penalty, 'stages': [stage1, stage2]}
        if score(candidate) > score(spec):
            spec = candidate
    spec['training'] = 'fixed_pair_anchor_two_reached_branch_updates'
    spec['features'] = ['initial_log_median', 'initial_log_carried_cost', 'initial_elite_band_over_iqr']
    return spec


def path(spec, r):
    i = pick(spec['stages'][0], initial_features(r.costs1, r.extra['means1'], r.extra['std1']))
    j = None if r.p1_term[i] else pick(spec['stages'][1], initial_features(r.costs2[i], r.extra['means2'][i], r.extra['std2'][i]))
    return i, j


def outcomes(spec, roots):
    paths = [path(spec, r) for r in roots]
    return {'success': np.array([r.leaf_success[i, 0 if j is None else j] for r, (i, j) in zip(roots, paths)]),
            'iterations': np.array([CHECKPOINTS[i] + (0 if j is None else CHECKPOINTS[j]) for i, j in paths])}
