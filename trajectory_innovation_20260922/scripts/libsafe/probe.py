"""LIBERO-Safety CPU probe (docs/CTA_LIBSAFE_PROTOCOL.md, setup): build each moving-obstacle task, count init states,
check twin branching and time a control step with and without cameras. No policy. Compute node only.

Twin check, per task: from the state after the wait plus 20 random steps, run 15 random steps in the real episode and
the same 15 steps in the camera-free twin loaded from the saved state; compare qpos, mocap positions and per-step
violation flags. The same 15 steps are then replayed in the twin from the same state (repeatability).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ti_wm import libsafe_runtime as ls  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from headroom import parse_tasks, root_id  # noqa: E402


def probe(suite, level, level_id, rng):
    lang, bddl, inits = ls.task_info(suite, level, level_id)
    root = root_id(suite, level, level_id, 0)
    env, twin = ls.make_env(bddl, root), ls.make_env(bddl, root, cameras=False)
    twin.seed(root)
    twin.reset()
    ls.start(env, inits[0], root)
    acts = rng.uniform(-0.5, 0.5, (35, 7))
    acts[:, -1] = np.sign(acts[:, -1])
    t0 = time.perf_counter()
    for a in acts[:20]:
        ls.step(env, a)
    render_ms = 1000 * (time.perf_counter() - t0) / 20
    state = ls.save_state(env)
    live = []
    for a in acts[20:]:
        _, _, c = ls.step(env, a)
        live.append(ls.violated(c))
    q_live, m_live = ls.qpos(env), ls.mocap(env)
    out = {"task": f"{suite}:L{level}:{level_id}", "language": lang, "n_init": len(inits), "horizon": ls.max_steps(lang),
           "n_mocap": int(m_live.shape[0]), "render_ms_per_step": render_ms}
    reps = []
    for _ in range(2):
        t0 = time.perf_counter()
        ls.load_state(twin, state)
        _, _, viol, _ = ls.run_chunk(twin, acts[20:])
        reps.append({"ms_per_step": 1000 * (time.perf_counter() - t0) / 15, "dqpos": float(np.max(np.abs(ls.qpos(twin) - q_live))),
                     "dmocap": float(np.max(np.abs(ls.mocap(twin) - m_live))) if m_live.size else 0.0,
                     "viol_match": viol == live})
    out["twin"] = reps
    out["live_violations"] = int(sum(live))
    env.close()
    twin.close()
    return out


def main(a):
    require_compute()
    ls.write_config(str(a.run / "libero_config"), a.repo)
    rng = np.random.default_rng(0)
    rep = [probe(s, l, i, rng) for s, l, i in parse_tasks(a.tasks)]
    for r in rep:
        print(json.dumps(r), flush=True)
    (a.run / "probe_report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--tasks", default="obstacle_avoidance:1:0-4,obstacle_avoidance_human:1:0-4,human_safety:0:1-1")
    main(p.parse_args())
