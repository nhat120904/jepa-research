"""Thin RGB-only interface to original DINO-WM Wall; no substituted dynamics."""
import copy
import random
import sys
from pathlib import Path

import numpy as np

from .runtime import require_slurm


class WallRGB:
    def __init__(self, upstream_root, wall_x=32, door_y=30, seed=0):
        require_slurm()
        # Import Wall without original env/__init__.py (which imports unrelated MuJoCo).
        source = Path(upstream_root).resolve() / "env"
        if not (source / "wall/envs/wall.py").is_file():
            raise FileNotFoundError(source / "wall/envs/wall.py")
        sys.path.insert(0, str(source))
        import torch
        from wall.envs.wall import DotWall, WallDatasetConfig

        random.seed(seed)
        torch.manual_seed(seed)
        self.torch = torch
        config = WallDatasetConfig(
            device="cpu", img_size=65, dot_std=1.7, border_wall_loc=5,
            wall_width=6, door_space=4, fix_wall=True,
            fix_wall_location=int(wall_x), fix_door_location=int(door_y),
        )
        self.env = DotWall(wall_config=config, device="cpu")
        self.layout = (int(wall_x), int(door_y))

    @staticmethod
    def _rgb(obs):
        # Deliberately discard upstream obs['proprio'], info and rewards.
        rgb = obs["visual"].detach().cpu().permute(1, 2, 0).numpy()
        if not np.isfinite(rgb).all():
            raise RuntimeError("Nonfinite RGB in upstream environment")
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def reset(self, position):
        pos = self.torch.as_tensor(np.asarray(position).copy(), dtype=self.torch.float32)
        if pos.shape != (2,):
            raise ValueError("position must be [2]")
        obs, _ = self.env.reset(location=pos)
        return self._rgb(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must be finite [2]")
        obs, _, _, _ = self.env.step(self.torch.from_numpy(action.copy()))
        if not self.torch.isfinite(self.env.dot_position).all():
            raise RuntimeError("Nonfinite upstream physics state; do not silently filter")
        return self._rgb(obs)

    def evaluation_state(self):
        """Privileged diagnostic only. Never feed to the model or proposal."""
        return self.env.dot_position.detach().cpu().numpy().copy()

    def snapshot(self):
        return {
            "position": self.evaluation_state(), "layout": self.layout,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": self.torch.get_rng_state().clone(),
            "env_rng": copy.deepcopy(self.env.rng.bit_generator.state),
        }

    def restore(self, snapshot):
        if tuple(snapshot["layout"]) != self.layout:
            raise ValueError("Cannot restore across different layouts")
        obs = self.reset(snapshot["position"])
        random.setstate(snapshot["python_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        self.torch.set_rng_state(snapshot["torch_rng"])
        self.env.rng.bit_generator.state = copy.deepcopy(snapshot["env_rng"])
        return obs

    def rollout(self, snapshot, actions):
        images = [self.restore(snapshot)]
        states = [self.evaluation_state()]
        for action in actions:
            images.append(self.step(action))
            states.append(self.evaluation_state())
        return np.stack(images), np.stack(states)

    def goal_image(self, position):
        snapshot = self.snapshot()
        image = self.reset(position)
        self.restore(snapshot)
        return image

    def sample_position(self, rng):
        # Valid environment initial/goal-state sampling, not a deployment input.
        wall_x, _ = self.layout
        if rng.integers(2) == 0:
            x = rng.uniform(8, wall_x - 7)
        else:
            x = rng.uniform(wall_x + 7, 57)
        return np.array([x, rng.uniform(8, 57)], dtype=np.float32)
