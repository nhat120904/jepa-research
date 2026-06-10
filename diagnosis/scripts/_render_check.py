"""Save rendered frames (raw vs flipped) + list cameras, to verify orientation
and camera setup against the training data."""
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE  # noqa: E402

env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE["reach-v3-goal-observable"](seed=0)
for i in range(env.model.ncam):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    print(f"cam {i}: {name} pos={env.model.cam_pos[i]}")

env.model.cam_pos[2] = [0.75, 0.075, 0.7]
env.render_mode = "rgb_array"
env.camera_name = "corner2"
env.width = env.height = 224
env.reset()
frame = env.render()
print("frame shape:", frame.shape)
out = Path("results/logs")
imageio.imwrite(out / "render_raw.png", frame)
imageio.imwrite(out / "render_flipped.png", frame[::-1].copy())
print("saved render_raw.png / render_flipped.png")
