"""Two-plan stopping tree and the matching live closed-loop controller."""

from __future__ import annotations

import time

import numpy as np

from .recorder import PlanRecorder
from .rules import CHECKPOINTS, choose

LABEL_ITERS = (1, 3, 5, 10, 20, 30)
LABEL_CANDIDATES = 32


def plan_seed(base: int, root: int, plan_index: int) -> int:
    return int(base) + 1000 * int(root) + int(plan_index)


def label_index(base: int, root: int) -> np.ndarray:
    """Fixed candidate slots to label: the carried mean (slot 0) plus 31 others."""
    rng = np.random.default_rng(int(base) + 7_000_000 + int(root))
    others = rng.choice(np.arange(1, 300), size=LABEL_CANDIDATES - 1, replace=False)
    return np.concatenate([[0], np.sort(others)]).astype(np.int64)


def _image_diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return {"max_abs": int(d.max()), "mean_abs": float(d.mean())}


def run_tree(task, root, base_seed: int, label: bool = False,
             checkpoints: tuple[int, ...] = CHECKPOINTS) -> tuple[dict, dict]:
    """Return (arrays, summary) for one root."""
    n = len(checkpoints)
    t_start = time.perf_counter()
    init, goal = task.rows(root)
    goal_image = task.goal_image(root, init, goal)
    task.restore(root, init, goal)
    frame0 = task.render()
    task.restore(root, init, goal)
    if not np.array_equal(frame0, task.render()):
        raise RuntimeError("root restore is not pixel-exact")
    start_diff = _image_diff(frame0, np.asarray(init["pixels"]))
    d0 = task.distance(goal)

    lidx = label_index(base_seed, root.root) if label else None
    rec1 = PlanRecorder(checkpoints, LABEL_ITERS if label else (), lidx)
    t = time.perf_counter()
    final1 = task.plan(task.prepared(frame0, goal_image), plan_seed(base_seed, root.root, 0), rec1)
    plan_seconds = [time.perf_counter() - t]
    if not np.array_equal(final1.numpy(), rec1.means[checkpoints[-1]]):
        raise RuntimeError("solver output differs from the recorded final mean")
    means1 = rec1.checkpoint_means()
    raws1 = [task.to_raw(m) for m in means1]

    p1_term = np.zeros(n, bool)
    p1_steps = np.zeros(n, np.int16)
    p1_dist = np.full(n, np.nan)
    costs2 = np.full((n, checkpoints[-1], 300), np.nan, np.float32)
    means2 = np.full((n,) + means1.shape, np.nan, np.float32)
    std2 = np.full((n, checkpoints[-1]), np.nan, np.float32)
    iter_sec2 = np.full((n, checkpoints[-1]), np.nan, np.float32)
    leaf_success = np.zeros((n, n), bool)
    leaf_dist = np.full((n, n), np.nan)
    leaf_steps = np.zeros((n, n), np.int16)
    replay_max_abs = 0.0
    sim_seconds = 0.0

    for i in range(n):
        t = time.perf_counter()
        task.restore(root, init, goal)
        term1, s1 = task.execute(raws1[i])
        p1_term[i], p1_steps[i] = term1, s1
        p1_dist[i] = task.distance(goal)
        after1 = task.state()
        sim_seconds += time.perf_counter() - t
        if term1:
            leaf_success[i, :] = True
            leaf_dist[i, :] = p1_dist[i]
            leaf_steps[i, :] = s1
            continue
        frame1 = task.render()
        rec2 = PlanRecorder(checkpoints)
        t = time.perf_counter()
        task.plan(task.prepared(frame1, goal_image), plan_seed(base_seed, root.root, 1), rec2)
        plan_seconds.append(time.perf_counter() - t)
        costs2[i] = rec2.cost_matrix()
        means2[i] = rec2.checkpoint_means()
        std2[i] = np.asarray(rec2.elite_std, np.float32)
        iter_sec2[i] = np.asarray(rec2.iter_seconds, np.float32)
        t = time.perf_counter()
        for j in range(n):
            task.restore(root, init, goal)
            task.execute(raws1[i])
            replay_max_abs = max(replay_max_abs, float(np.max(np.abs(task.state() - after1))))
            term2, s2 = task.execute(task.to_raw(means2[i, j]))
            d = task.distance(goal)
            leaf_success[i, j] = task.success(term2, d)
            leaf_dist[i, j] = d
            leaf_steps[i, j] = s1 + s2
        sim_seconds += time.perf_counter() - t
    if replay_max_abs != 0.0:
        raise RuntimeError(f"branch replay is not exact (max abs {replay_max_abs})")

    arrays = {
        "costs1": rec1.cost_matrix(),
        "means1": means1,
        "std1": np.asarray(rec1.elite_std, np.float32),
        "iter_sec1": np.asarray(rec1.iter_seconds, np.float32),
        "p1_term": p1_term, "p1_steps": p1_steps, "p1_dist": p1_dist,
        "costs2": costs2, "means2": means2, "std2": std2, "iter_sec2": iter_sec2,
        "leaf_success": leaf_success, "leaf_dist": leaf_dist, "leaf_steps": leaf_steps,
        "checkpoints": np.asarray(checkpoints, np.int16),
    }
    if label:
        t = time.perf_counter()
        its = [k for k in LABEL_ITERS if k in rec1.label_actions]
        lab_cost = np.stack([rec1.label_costs[k] for k in its])
        lab_dist = np.full(lab_cost.shape, np.nan)
        lab_success = np.zeros(lab_cost.shape, bool)
        for a, k in enumerate(its):
            for b, cand in enumerate(rec1.label_actions[k]):
                task.restore(root, init, goal)
                term, _ = task.execute(task.to_raw(cand))
                lab_dist[a, b] = task.distance(goal)
                lab_success[a, b] = task.success(term, lab_dist[a, b])
        sim_seconds += time.perf_counter() - t
        arrays.update({"label_iters": np.asarray(its, np.int16), "label_index": lidx,
                       "label_cost": lab_cost, "label_dist": lab_dist,
                       "label_success": lab_success})

    diag = np.diag(leaf_success)
    summary = {
        "task": task.name, "root": root.root, "episode": root.episode,
        "start_step": root.start_step, "reset_seed": root.reset_seed,
        "seeds": [plan_seed(base_seed, root.root, 0), plan_seed(base_seed, root.root, 1)],
        "start_distance": d0, "start_image_vs_dataset": start_diff,
        "fixed_k_success": {int(k): bool(diag[i]) for i, k in enumerate(checkpoints)},
        "any_leaf_success": bool(leaf_success.any()),
        "plan_seconds": plan_seconds, "sim_seconds": sim_seconds,
        "total_seconds": time.perf_counter() - t_start,
        "replay_max_abs": replay_max_abs, "labelled": bool(label),
        "goal_source": task.goal_source,
    }
    return arrays, summary


def run_live(task, root, base_seed: int, rule,
             checkpoints: tuple[int, ...] = CHECKPOINTS) -> dict:
    """One continuous episode where ``rule`` picks the checkpoint of each plan."""
    init, goal = task.rows(root)
    goal_image = task.goal_image(root, init, goal)
    task.restore(root, init, goal)
    frame = task.render()
    rec1 = PlanRecorder(checkpoints)
    task.plan(task.prepared(frame, goal_image), plan_seed(base_seed, root.root, 0), rec1)
    costs1 = rec1.cost_matrix()
    k1 = choose(rule, costs1, checkpoints)
    term, steps = task.execute(task.to_raw(rec1.means[k1]))
    k2, costs2 = None, None
    if not term:
        frame = task.render()
        rec2 = PlanRecorder(checkpoints)
        task.plan(task.prepared(frame, goal_image), plan_seed(base_seed, root.root, 1), rec2)
        costs2 = rec2.cost_matrix()
        k2 = choose(rule, costs2, checkpoints)
        term, s2 = task.execute(task.to_raw(rec2.means[k2]))
        steps += s2
    d = task.distance(goal)
    return {"k1": int(k1), "k2": None if k2 is None else int(k2),
            "success": task.success(term, d), "final_distance": float(d), "steps": int(steps),
            "costs1": costs1, "costs2": costs2}
