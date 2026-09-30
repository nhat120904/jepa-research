"""Task adapters: dataset roots, exact restore, execution and success.

Everything here imports the simulator/model stack lazily and must run under
Slurm. The Cube restore reuses the procedure verified by the corrected
OGBench true-endpoint audit (``diagnosis/scripts/76_*.py``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DIAG_SCRIPTS = REPO / "diagnosis" / "scripts"
STAGE0 = Path(os.environ.get("STABLEWM_HOME", "/mnt/data/nhatnc129/jepa/lewm_stage0"))
REACHER_H5 = STAGE0 / "downloads/lewm/lewm-reacher/extracted/reacher.h5"

GOAL_OFFSET = 25
HORIZON = 5
BLOCK = 5
BUDGET = 50
NUM_SAMPLES = 300
TOPK = 30
N_STEPS = 30
IMAGE = 224


@dataclass(frozen=True)
class Root:
    root: int
    episode: int
    start_step: int
    reset_seed: int


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_roots(dataset, n: int, seed: int) -> list[Root]:
    """Uniform over valid starts, then a seeded permutation (root id = rank)."""
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    counts = np.maximum(lengths - GOAL_OFFSET, 0)
    total = int(counts.sum())
    rng = np.random.default_rng(seed)
    ordinals = rng.choice(total, size=n, replace=False)
    cumulative = np.cumsum(counts)
    roots = []
    for rank, ordinal in enumerate(ordinals):
        episode = int(np.searchsorted(cumulative, ordinal, side="right"))
        previous = int(cumulative[episode - 1]) if episode else 0
        roots.append(Root(root=rank, episode=episode, start_step=int(ordinal - previous),
                          reset_seed=seed + 10_000 + rank))
    return roots


def image_transform():
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=IMAGE),
    ])


def resize(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    image = np.asarray(image)
    if image.shape[:2] == (IMAGE, IMAGE):
        return image.copy()
    return np.asarray(Image.fromarray(image).resize((IMAGE, IMAGE), Image.BILINEAR))


class Task:
    name: str
    repo: str
    raw_dim: int
    tolerance: float  # physical success tolerance (m or rad)

    def __init__(self, swm, device: str = "cuda") -> None:
        from sklearn.preprocessing import StandardScaler

        self.swm = swm
        self.device = device
        self.dataset = self._load_dataset()
        raw = np.asarray(self.dataset.get_col_data("action"))
        self.scaler = StandardScaler().fit(raw[~np.isnan(raw).any(axis=1)])
        self.model = swm.wm.utils.load_pretrained(self.repo).to(device).eval()
        self.model.requires_grad_(False)
        self.model.interpolate_pos_encoding = True
        self.transform = image_transform()
        self.world = None
        self.raw_env = None
        self._make_env()
        self.solver = swm.planning.CEMSolver(
            cost=swm.planning.ShootingCostEvaluator(self.model, swm.planning.GoalMSE()),
            batch_size=1, num_samples=NUM_SAMPLES, n_steps=N_STEPS, topk=TOPK,
            var_scale=1.0, device=device, seed=0, callbacks=[],
        )
        self.policy = swm.policy.WorldModelPolicy(
            solver=self.solver,
            config=swm.PlanConfig(horizon=HORIZON, receding_horizon=HORIZON,
                                  action_block=BLOCK, history_len=1, warm_start=True),
            process={"action": self.scaler},
            transform={"pixels": self.transform, "goal": self.transform},
        )
        self.policy.set_env(self.world.envs)

    # -- dataset -----------------------------------------------------------
    def _load_dataset(self):
        raise NotImplementedError

    def rows(self, root: Root) -> tuple[dict, dict]:
        from stable_worldmodel.world.world import _extract_init_goal

        init, goal, _ = _extract_init_goal(self.dataset, [root.episode], [root.start_step],
                                           GOAL_OFFSET)
        return init[0], goal[0]

    # -- simulator ---------------------------------------------------------
    def _make_env(self) -> None:
        raise NotImplementedError

    def restore(self, root: Root, init: dict, goal: dict) -> None:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        raise NotImplementedError

    def distance(self, goal: dict) -> float:
        raise NotImplementedError

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def execute(self, raw_actions: np.ndarray) -> tuple[bool, int]:
        """Step until the plan ends or the env terminates (goal reached)."""
        steps = 0
        for action in raw_actions:
            _, _, terminated, truncated, _ = self.raw_env.step(action)
            steps += 1
            if terminated:
                return True, steps
            if truncated:
                break
        return False, steps

    def success(self, terminated_any: bool, final_distance: float) -> bool:
        return bool(terminated_any) or final_distance <= self.tolerance

    # "dataset": the recorded goal frame, as upstream. "render": restore the
    # recorded goal state and render it with the evaluation renderer, so current
    # and goal frames share one renderer domain (see diag_render).
    goal_source = "dataset"

    def goal_image(self, root: Root, init: dict, goal: dict) -> np.ndarray:
        if self.goal_source == "dataset":
            return np.asarray(goal["goal"])
        if self.goal_source != "render":
            raise ValueError(f"unknown goal source {self.goal_source}")
        self.restore(root, {"qpos": goal["goal_qpos"], "qvel": goal["goal_qvel"]}, goal)
        return self.render()

    # -- planning ----------------------------------------------------------
    def prepared(self, frame: np.ndarray, goal_image: np.ndarray) -> dict:
        info = {
            "pixels": np.asarray(frame)[None, None],
            "goal": np.asarray(goal_image)[None, None],
            "action": np.full((1, 1, self.raw_dim), np.nan, np.float32),
        }
        return self.policy._prepare_info(info)

    @torch.inference_mode()
    def plan(self, prepared: dict, seed: int, recorder) -> torch.Tensor:
        self.solver.callbacks = [recorder] if recorder is not None else []
        self.solver.torch_gen.manual_seed(int(seed))
        out = self.solver.solve(dict(prepared))
        return out["actions"][0]

    def to_raw(self, mean) -> np.ndarray:
        """Blocked plan (H, BLOCK*raw_dim) -> raw env actions, as upstream."""
        plan = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        plan = plan.reshape(HORIZON * BLOCK, self.raw_dim).numpy()
        return self.scaler.inverse_transform(plan)

    def close(self) -> None:
        if self.world is not None:
            self.world.close()


class CubeTask(Task):
    name = "cube"
    repo = "quentinll/lewm-cube"
    raw_dim = 5
    tolerance = 0.04

    def _load_dataset(self):
        return self.swm.data.load_dataset("ogbench/cube_single_expert.h5", keys_to_cache=["action"])

    def _make_env(self) -> None:
        self.corrected = _load_module(DIAG_SCRIPTS / "76_ogb_true_endpoint_corrected.py",
                                      "cemstop_cube_corrected")
        self.audit = self.corrected.load_stage0_module()
        self.corrected.load_stage0_transform_images = self.audit.transform_images
        first = Root(root=-1, episode=0, start_step=0, reset_seed=20260929)
        self.world, self.raw_env, self.visual_signature, _ = self.corrected.make_world(self.swm, first)

    def restore(self, root: Root, init: dict, goal: dict) -> None:
        import mujoco

        self.raw_env.reset(seed=root.reset_seed, options={"variation": []})
        self.raw_env._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        self.corrected.restore_complete(self.raw_env, init["qpos"], init["qvel"], goal, self.audit)
        if not np.array_equal(self.raw_env._data.qpos, np.asarray(init["qpos"])):
            raise RuntimeError("exact Cube qpos restoration failed")
        if not np.array_equal(self.raw_env._data.qvel, np.asarray(init["qvel"])):
            raise RuntimeError("exact Cube qvel restoration failed")

    def render(self) -> np.ndarray:
        return resize(self.raw_env.render())

    def distance(self, goal: dict) -> float:
        return self.audit.cube_distance(self.raw_env, self.audit.goal_field(goal, "block_0_pos"))

    def state(self) -> np.ndarray:
        return np.concatenate([self.raw_env._data.qpos, self.raw_env._data.qvel]).copy()


class ReacherTask(Task):
    name = "reacher"
    repo = "quentinll/lewm-reacher"
    raw_dim = 2
    tolerance = 0.05  # per-joint rad; success is the env's qpos_match termination

    def _load_dataset(self):
        return self.swm.data.load_dataset(str(REACHER_H5), keys_to_cache=["action"])

    def _make_env(self) -> None:
        self.world = self.swm.World("swm/ReacherDMControl-v0", num_envs=1,
                                    image_shape=(IMAGE, IMAGE), max_episode_steps=2 * BUDGET,
                                    task="qpos_match")
        self.raw_env = self.world.envs.envs[0].unwrapped

    def restore(self, root: Root, init: dict, goal: dict) -> None:
        self.raw_env.reset(seed=root.reset_seed)
        self.raw_env.set_state(np.asarray(init["qpos"], np.float64), np.asarray(init["qvel"], np.float64))
        self.raw_env.set_target_qpos(np.asarray(goal["goal_qpos"], np.float64))
        physics = self.raw_env.env.physics
        if not np.array_equal(physics.data.qpos, np.asarray(init["qpos"])):
            raise RuntimeError("exact Reacher qpos restoration failed")

    def render(self) -> np.ndarray:
        return resize(self.raw_env.render(width=IMAGE, height=IMAGE))

    def distance(self, goal: dict) -> float:
        qpos = np.asarray(self.raw_env.env.physics.data.qpos, np.float64)
        return float(np.max(np.abs(qpos - np.asarray(goal["goal_qpos"], np.float64))))

    def success(self, terminated_any: bool, final_distance: float) -> bool:
        # The environment's own criterion is strict (< 0.05 on every joint).
        return bool(terminated_any) or final_distance < self.tolerance

    def state(self) -> np.ndarray:
        return np.asarray(self.raw_env.env.physics.get_state(), np.float64).copy()


TASKS = {"cube": CubeTask, "reacher": ReacherTask}
