"""Smoke: can we create a Metaworld V3 env, render offscreen at 224px from the
upstream camera, and roll the scripted expert? (No world model involved.)"""
import sys

import numpy as np

from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld import policies

task = sys.argv[1] if len(sys.argv) > 1 else "reach-v3-goal-observable"
env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[task](seed=0)
env.seeded_rand_vec = False
env.model.cam_pos[2] = [0.75, 0.075, 0.7]          # upstream corner2 tweak
env.render_mode = "rgb_array"
env.camera_name = "corner2"
env.width = env.height = 224

obs, info = env.reset()
print("obs shape:", obs.shape, "action space:", env.action_space.shape)

frame = env.render()
print("frame:", None if frame is None else (frame.shape, frame.dtype,
                                            int(frame.sum())))

pol = policies.SawyerReachV3Policy()
succ = False
for t in range(150):
    a = pol.get_action(obs)
    obs, r, terminated, truncated, info = env.step(a)
    if info.get("success", 0) > 0.5:
        succ = True
        break
print(f"expert success={succ} at t={t}")
frame = env.render()
print("frame after rollout:", frame.shape, int(frame.sum()))
