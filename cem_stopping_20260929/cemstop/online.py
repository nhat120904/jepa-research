"""Actually stop upstream CEM from its post-update callback."""
from __future__ import annotations

import time
import numpy as np

from .continuation import features, predict
from .recorder import PlanRecorder
from .rules import CHECKPOINTS, Fixed, Converge, Gap, Band
from .tree import plan_seed


class StopSearch(Exception):
    """Private control-flow signal; never catches model/simulator errors."""
    pass


class OnlineRecorder(PlanRecorder):
    def __init__(self, spec, stage):
        self.spec, self.stage = spec, stage
        super().__init__(CHECKPOINTS)
        self.chosen = None

    def __call__(self, **state):
        super().__call__(**state)
        k = int(state['step']) + 1
        if k not in CHECKPOINTS:
            return
        i = CHECKPOINTS.index(k)
        kind = self.spec['kind']
        stop = k == CHECKPOINTS[-1]
        if not stop and kind == 'budget_stump':
            from .budget_stump import initial_features, pick
            # Recomputing from iteration 1 is causal and returns the same budget.
            means = np.stack([self.means[c] for c in CHECKPOINTS[:i + 1]])
            target = pick(self.spec['stages'][self.stage], initial_features(self.cost_matrix(), means, self.elite_std))
            stop = k >= CHECKPOINTS[target]
        elif not stop and kind == 'continuation':
            means = np.stack([self.means[c] for c in CHECKPOINTS[:i + 1]])
            x = features(self.cost_matrix(), means, self.elite_std, i)
            stop = predict(self.spec['models'][str(self.stage)][i], x) <= 0
        elif not stop and kind == 'fixed_pair':
            stop = k >= self.spec['k1' if self.stage == 0 else 'k2']
        elif not stop:
            cls = {'fixed': Fixed, 'converge': Converge, 'gap': Gap, 'band': Band}[kind]
            rule = cls(self.spec['param'])
            stop = rule.stop(self.cost_matrix(), k, None if i == 0 else CHECKPOINTS[i - 1])
        if stop:
            self.chosen = k
            raise StopSearch()


def plan_online(task, frame, goal_image, seed, spec, stage):
    recorder = OnlineRecorder(spec, stage)
    t = time.perf_counter()
    prepared = task.prepared(frame, goal_image)
    try:
        task.plan(prepared, seed, recorder)
    except StopSearch:
        pass
    if recorder.chosen is None:
        raise RuntimeError('solver returned without a stopping decision')
    # Recorder copies to CPU every iteration, synchronising GPU work.
    elapsed = time.perf_counter() - t
    return recorder.means[recorder.chosen], recorder, elapsed


def run_online(task, root, base_seed, spec, plain_fixed=False):
    init, goal = task.rows(root)
    goal_image = np.asarray(goal['goal'])
    task.restore(root, init, goal)
    t0 = time.perf_counter()
    costs, means, ks, seconds = [], [], [], []
    total_steps = 0
    term = False
    for stage in range(2):
        frame = task.render()
        if plain_fixed and spec['kind'] in ('fixed', 'fixed_pair'):
            k = int(spec['param']) if spec['kind'] == 'fixed' else spec['k1' if stage == 0 else 'k2']
            old_steps = task.solver.n_steps
            t = time.perf_counter()
            try:
                task.solver.n_steps = k
                mean = task.plan(task.prepared(frame, goal_image),
                                 plan_seed(base_seed, root.root, stage), None).numpy()
            finally:
                task.solver.n_steps = old_steps
            elapsed = time.perf_counter() - t
            ks.append(k)
            costs.append(None)
        else:
            mean, rec, elapsed = plan_online(task, frame, goal_image,
                                             plan_seed(base_seed, root.root, stage), spec, stage)
            ks.append(rec.chosen)
            costs.append(rec.cost_matrix())
        means.append(mean)
        seconds.append(elapsed)
        term, steps = task.execute(task.to_raw(mean))
        total_steps += steps
        if term:
            break
    d = task.distance(goal)
    return {'success': task.success(term, d), 'distance': d, 'ks': ks,
            'iterations': sum(ks), 'planning_seconds': sum(seconds),
            'episode_seconds': time.perf_counter() - t0, 'steps': total_steps,
            'costs': costs, 'means': means}
