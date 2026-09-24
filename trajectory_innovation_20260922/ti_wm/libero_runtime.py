"""LIBERO-Goal runtime for CTA (docs/LIBERO_QUALIFICATION_PROTOCOL.md, L2-L3). LIBERO venv only (Python 3.12).

Policy, processors and environments are built from the SAME command-line configuration as the L1 lerobot-eval run
(scripts/libero/slurm_l1_eval.sh), so P0 here is the policy measured at L1.

Candidates: K seeded flow-matching noise draws per decision. SmolVLA integrates the flow deterministically given the
noise, so candidate k depends only on its seed and candidate 0 is a P0 draw.
Branching: the full MuJoCo integration state (mjSTATE_INTEGRATION, which includes the solver warm start, the
PushT lesson) plus robosuite's Python-side state (step counter, time, done flag, gripper action integrators),
copied into a second environment instance of the same task.
"""

import copy
import sys

import numpy as np
import torch

from ti_wm.contract import candidate_seed

N_ACTION_STEPS = 10


def l1_args(checkpoint, task_ids="[0,1,2,3,4,5,6,7,8,9]", seed=0):
    return [f"--policy.path={checkpoint}", f"--policy.n_action_steps={N_ACTION_STEPS}", "--policy.device=cuda",
            "--env.type=libero", "--env.task=libero_goal", f"--env.task_ids={task_ids}", "--eval.batch_size=1",
            "--eval.n_episodes=1", f"--seed={seed}", "--output_dir=/tmp/unused_cta_libero"]


def parse_eval_config(args):
    """EvalPipelineConfig exactly as lerobot-eval parses it (its __post_init__ reads sys.argv)."""
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig

    def _identity(cfg: EvalPipelineConfig):
        return cfg

    saved = sys.argv
    sys.argv = ["lerobot-eval", *args]
    try:
        return parser.wrap()(_identity)()
    finally:
        sys.argv = saved


class Policy:
    """SmolVLA with lerobot-eval's processors; draws a nested seeded bank of chunks from one observation."""

    def __init__(self, checkpoint, device="cuda"):
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        args = l1_args(checkpoint)
        if device != "cuda":
            args = [a.replace("--policy.device=cuda", f"--policy.device={device}") for a in args]
        self.cfg = parse_eval_config(args)
        self.policy = make_policy(cfg=self.cfg.policy, env_cfg=self.cfg.env, rename_map=self.cfg.rename_map).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.cfg.policy, pretrained_path=self.cfg.policy.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(self.cfg.policy.device)},
                                    "rename_observations_processor": {"rename_map": self.cfg.rename_map}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=self.cfg.env, policy_cfg=self.cfg.policy)
        self.device = torch.device(self.cfg.policy.device)
        pc = self.policy.config
        self.noise_shape = (pc.chunk_size, pc.max_action_dim)

    def _batch(self, obs, task, k):
        from lerobot.envs.utils import preprocess_observation

        batched = {"pixels": {c: np.repeat(img[None], k, 0) for c, img in obs["pixels"].items()},
                   "robot_state": _tree_repeat(obs["robot_state"], k)}
        batch = preprocess_observation(batched)
        batch["task"] = [task] * k
        return self.pre(self.env_pre(batch))

    @torch.inference_mode()
    def bank(self, obs, task, seeds):
        """(K, N_ACTION_STEPS, 7) env actions; row k uses flow noise from its own seeded generator."""
        k = len(seeds)
        noise = torch.stack([torch.randn(self.noise_shape, generator=torch.Generator().manual_seed(int(s)))
                             for s in seeds]).to(self.device)
        self.policy.reset()
        chunk = self.policy.predict_action_chunk(self._batch(obs, task, k), noise=noise)[:, :N_ACTION_STEPS]
        steps = []
        for t in range(N_ACTION_STEPS):
            act = self.env_post({"action": self.post(chunk[:, t])})["action"]
            steps.append(act.float().cpu().numpy())
        return np.stack(steps, axis=1)


def _tree_repeat(tree, k):
    if isinstance(tree, dict):
        return {key: _tree_repeat(v, k) for key, v in tree.items()}
    return np.repeat(np.asarray(tree)[None], k, 0)


def make_env(cfg, task_id, init_index):
    """One LiberoEnv exactly as lerobot's factory builds it for (task_id, episode_index=init_index)."""
    from lerobot.envs.libero import LiberoEnv, _get_suite

    env_cfg = cfg.env
    kwargs = {k: v for k, v in env_cfg.gym_kwargs.items() if k != "task_ids"}
    return LiberoEnv(task_suite=_get_suite(env_cfg.task), task_id=task_id, task_suite_name=env_cfg.task,
                     camera_name=env_cfg.camera_name, init_states=env_cfg.init_states,
                     episode_length=env_cfg.episode_length, episode_index=init_index, n_envs=1,
                     control_mode=env_cfg.control_mode, camera_name_mapping=env_cfg.camera_name_mapping,
                     is_libero_plus=env_cfg.is_libero_plus, **kwargs)


def _inner(env):
    return env._env.env          # LiberoEnv -> OffScreenRenderEnv -> robosuite problem env


def _raw(sim):
    return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)


GRIPPER_FIELDS = ("current_action",)


def save_state(env):
    import mujoco

    inner = _inner(env)
    model, data = _raw(inner.sim)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    buf = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, buf, spec)
    robots = []
    for r in inner.robots:
        grip = {f: copy.deepcopy(getattr(r.gripper, f)) for f in GRIPPER_FIELDS if hasattr(r.gripper, f)}
        ctrl = {f: copy.deepcopy(v) for f, v in vars(r.controller).items() if isinstance(v, (np.ndarray, float, int, bool))}
        robots.append({"gripper": grip, "controller": ctrl})
    return {"mj": buf, "timestep": inner.timestep, "cur_time": inner.cur_time, "done": inner.done, "robots": robots}


def load_state(env, state):
    """Restore into env (any instance of the same task) and return its observation, as LiberoEnv returns them."""
    import mujoco

    inner = _inner(env)
    model, data = _raw(inner.sim)
    mujoco.mj_setState(model, data, state["mj"], mujoco.mjtState.mjSTATE_INTEGRATION)
    inner.timestep, inner.cur_time, inner.done = state["timestep"], state["cur_time"], state["done"]
    for r, saved in zip(inner.robots, state["robots"]):
        for f, v in saved["gripper"].items():
            setattr(r.gripper, f, copy.deepcopy(v))
        for f, v in saved["controller"].items():
            setattr(r.controller, f, copy.deepcopy(v))
    inner.sim.forward()
    inner._update_observables(force=True)
    return env._format_raw_obs(inner._get_observations())


def qpos(env):
    return np.array(_raw(_inner(env).sim)[1].qpos)


def object_positions(env):
    """Privileged positions of the task's movable objects (for diversity and oracle diagnostics only)."""
    inner = _inner(env)
    return {name: np.array(inner.sim.data.body_xpos[inner.obj_body_id[name]]) for name in inner.objects_dict}


def scene_state(env):
    """Privileged non-robot state: every qpos entry outside the robot's arm and gripper joints (objects' free joints
    and articulated fixtures such as drawers and doors), plus the end-effector position. Diagnostics and oracle only."""
    inner = _inner(env)
    robot = set()
    for r in inner.robots:
        robot.update(np.atleast_1d(r._ref_joint_pos_indexes).tolist())
        robot.update(np.atleast_1d(getattr(r, "_ref_gripper_joint_pos_indexes", [])).tolist())
    q = qpos(env)
    keep = [i for i in range(len(q)) if i not in robot]
    return {"scene_qpos": q[keep], "eef": np.array(inner.robots[0].controller.ee_pos)}


def run_chunk(env, actions, keep=()):
    """Step the chunk; stops at success. Returns (obs, success, steps taken, {step: obs} for steps in keep, qpos list)."""
    frames, trace, success, obs = {}, [], False, None
    for t, action in enumerate(actions, start=1):
        obs, _, terminated, _, info = env.step(np.asarray(action, dtype=np.float32))
        trace.append(qpos(env))
        if t in keep:
            frames[t] = obs
        success = bool(info.get("is_success", False))
        if terminated:
            break
    return obs, success, len(trace), frames, trace


def candidate_seeds(root, d, k):
    return [candidate_seed(root, d, j) for j in range(k)]


# ----------------------------------------------------------------------------- goal progress (Amendment 2)

def _goal_state(env):
    return [[str(x) for x in s] for s in _inner(env).parsed_problem["goal_state"]]


def _joint_values(inner, name):
    if name in inner.object_sites_dict:
        joints = inner.object_sites_dict[name].joints
        parent = inner.get_object(inner.object_sites_dict[name].parent_name)
    else:
        joints, parent = inner.get_object(name).joints, inner.get_object(name)
    return [float(inner.sim.data.qpos[inner.sim.model.get_joint_qpos_addr(j)]) for j in joints], parent


def goal_reference(env):
    """Start-of-episode joint values and targets for the joint predicates of the task's goal."""
    from ti_wm.goal_progress import JOINT_KINDS, nearest_endpoint

    inner, ref = _inner(env), []
    for state in _goal_state(env):
        kind = state[0].lower()
        if kind in JOINT_KINDS:
            qs, parent = _joint_values(inner, state[1])
            ranges = parent.object_properties["articulation"][JOINT_KINDS[kind]]
            ref.append({"q0": qs, "target": [nearest_endpoint(ranges, q) for q in qs]})
        else:
            ref.append({})
    return ref


def goal_progress(env, ref):
    """Privileged dense progress of the current state toward the task's BDDL goal (oracle and reader labels only)."""
    from ti_wm.goal_progress import DISTANCE_KINDS, JOINT_KINDS, combine, distance_progress, joint_progress

    inner, values, satisfied = _inner(env), [], []
    for state, r in zip(_goal_state(env), ref):
        kind = state[0].lower()
        if kind in JOINT_KINDS:
            qs, _ = _joint_values(inner, state[1])
            values.append(max(joint_progress(q, q0, tg) for q, q0, tg in zip(qs, r["q0"], r["target"])))
        elif kind in DISTANCE_KINDS:
            pos = inner.sim.data.body_xpos[inner.obj_body_id[state[1]]]
            values.append(distance_progress(pos, inner.object_states_dict[state[2]].get_geom_state()["pos"]))
        else:
            raise ValueError(f"no progress function for predicate {state}")
        satisfied.append(bool(inner._eval_predicate(state)))
    return combine(values, satisfied)
