"""OGBench manipulation runtime for CTA (docs/CTA_OGBENCH_PLAN.md): environments, exact branching, oracle labels.

Branching restores the full MuJoCo integration state (qpos, qvel, act, ctrl, mocap targets, warm start, time) with
mj_setState + mj_forward. Every other piece of ManipSpaceEnv state used by a step (previous qpos, target effector pose,
success flag, target-geom colors) is recomputed from MjData inside the next step, so a restored env steps exactly like
the original; `check_restore` verifies this bit-for-bit before any evaluation.
Branch simulation steps physics only (no rendering): set_control, pre_step, mj_step, post_step, as ManipSpaceEnv.step.
Compute-node only (MuJoCo, EGL rendering).
"""
import numpy as np


def make_env(env_name):
    import ogbench

    return ogbench.make_env_and_datasets(env_name, env_only=True)


def reset(env, task_id, seed):
    ob, info = env.reset(seed=int(seed), options={"task_id": int(task_id), "render_goal": True})
    return ob, info["goal"]


def _mj(env):
    u = env.unwrapped
    return u, u._model, u._data


def save_state(env):
    import mujoco

    u, model, data = _mj(env)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    buf = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, buf, spec)
    return {"mj": buf, "success": bool(u._success), "elapsed": int(getattr(env, "_elapsed_steps", 0) or 0)}


def load_state(env, state):
    import mujoco

    u, model, data = _mj(env)
    mujoco.mj_setState(model, data, state["mj"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    u._success = state["success"]
    if hasattr(env, "_elapsed_steps"):
        env._elapsed_steps = state["elapsed"]


def physics_step(env, action):
    """ManipSpaceEnv.step without observation rendering; returns the post-step success flag."""
    import mujoco

    u, model, data = _mj(env)
    u.set_control(np.asarray(action))
    u.pre_step()
    mujoco.mj_step(model, data, nstep=u._n_steps)
    mujoco.mj_rnePostConstraint(model, data)
    u.post_step()
    return bool(u._success)


def cube_distances(env):
    """Privileged distance of each cube to its target (m). Oracle labels only."""
    u, _, data = _mj(env)
    return np.array([np.linalg.norm(data.joint(f"object_joint_{i}").qpos[:3] - data.mocap_pos[u._cube_target_mocap_ids[i]])
                     for i in range(u._num_cubes)])


def progress(env):
    """Oracle label: 1 if every cube is within the task's 4 cm success radius, else minus the summed distance (m)."""
    d = cube_distances(env)
    return 1.0 if bool(np.all(d <= 0.04)) else float(-d.sum())


def simulate_chunk(env, state, chunk):
    """Label of one candidate chunk from `state` (physics only); the env is restored to `state` afterwards."""
    load_state(env, state)
    for a in chunk:
        if physics_step(env, a):
            break
    y = progress(env)
    load_state(env, state)
    return y


def check_restore(env, actions, tol=0.0):
    """Stepping after a restore must repeat the original trajectory exactly (qpos per step)."""
    _, _, data = _mj(env)
    state = save_state(env)
    ref = []
    for a in actions:
        physics_step(env, a)
        ref.append(data.qpos.copy())
    load_state(env, state)
    dev = 0.0
    for a, q in zip(actions, ref):
        physics_step(env, a)
        dev = max(dev, float(np.abs(data.qpos - q).max()))
    load_state(env, state)
    if dev > tol:
        raise RuntimeError(f"restore is not exact: max |dqpos| = {dev}")
    return dev
