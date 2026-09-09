"""State-only ManiSkill strike--slide environment for ER-WM Gate E0.

Importing this module registers ``ERStrikeSlide-v0``. Simulation must run under
Slurm; the module itself is safe to import for syntax checking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, DefaultMaterialsConfig, SimConfig


@register_env("ERStrikeSlide-v0", max_episode_steps=5000)
class ERStrikeSlideEnv(BaseEnv):
    """Panda strikes a rectangular puck toward a distant table target."""

    SUPPORTED_ROBOTS = ["panda"]
    goal_radius = 0.05
    puck_half_sizes = (0.050, 0.035, 0.020)
    terminal_speed = 0.10
    stable_seconds = 0.20

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.0,
        physics_hz: int = 500,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.physics_hz = int(physics_hz)
        if self.physics_hz % 20 != 0:
            raise ValueError("physics_hz must be divisible by the 20 Hz action rate")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=self.physics_hz,
            # One env.step is one physics step. The runner changes the target only
            # every physics_hz/20 calls, which preserves a 20 Hz action interface.
            control_freq=self.physics_hz,
            default_materials_config=DefaultMaterialsConfig(
                static_friction=0.35,
                dynamic_friction=0.25,
                restitution=0.05,
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.1, 0.9, 0.5], target=[0.0, -0.1, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.8, 1.3, 1.0], [0.0, -0.1, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.puck = actors.build_box(
            self.scene,
            half_sizes=self.puck_half_sizes,
            color=[0.05, 0.35, 0.9, 1.0],
            name="puck",
            initial_pose=sapien.Pose(p=[0, 0.45, self.puck_half_sizes[2]]),
        )
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, -0.60, 1e-3]),
        )
        self.stable_count = torch.zeros(self.num_envs, dtype=torch.int64)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.stable_count = self.stable_count.to(self.device)
        with torch.device(self.device):
            batch = len(env_idx)
            self.table_scene.initialize(env_idx)
            robot_pose = Pose.create_from_pq(
                p=[-0.1, 1.0, 0], q=[0.7071, 0, 0, -0.7072]
            )
            self.agent.robot.set_pose(robot_pose)

            puck_xy = torch.zeros((batch, 3))
            puck_xy[:, 0] = (torch.rand(batch) * 2 - 1) * 0.055
            puck_xy[:, 1] = 0.43 + torch.rand(batch) * 0.04
            puck_xy[:, 2] = self.puck_half_sizes[2]
            yaw = (torch.rand(batch) * 2 - 1) * 0.20
            quat = torch.zeros((batch, 4))
            quat[:, 0] = torch.cos(yaw / 2)
            quat[:, 3] = torch.sin(yaw / 2)
            self.puck.set_pose(Pose.create_from_pq(p=puck_xy, q=quat))
            self.puck.set_linear_velocity(torch.zeros((batch, 3)))
            self.puck.set_angular_velocity(torch.zeros((batch, 3)))

            goal = torch.zeros((batch, 3))
            goal[:, 0] = (torch.rand(batch) * 2 - 1) * 0.16
            goal[:, 1] = -0.58 - torch.rand(batch) * 0.08
            goal[:, 2] = 1e-3
            goal_quat = torch.as_tensor(
                euler2quat(0, np.pi / 2, 0), dtype=torch.float32
            ).repeat(batch, 1)
            self.goal_region.set_pose(Pose.create_from_pq(p=goal, q=goal_quat))
        self.stable_count[env_idx] = 0

    def evaluate(self):
        distance = torch.linalg.norm(
            self.puck.pose.p[..., :2] - self.goal_region.pose.p[..., :2], dim=1
        )
        speed = torch.linalg.norm(self.puck.linear_velocity, dim=1)
        currently_stable = (distance < self.goal_radius) & (speed < self.terminal_speed)
        self.stable_count = torch.where(
            currently_stable, self.stable_count + 1, torch.zeros_like(self.stable_count)
        )
        required = int(round(self.stable_seconds * self.physics_hz))
        return {
            "success": self.stable_count >= required,
            "puck_goal_distance": distance,
            "puck_speed": speed,
        }

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                puck_pose=self.puck.pose.raw_pose,
                puck_vel=self.puck.linear_velocity,
                puck_ang_vel=self.puck.angular_velocity,
                tcp_to_puck_pos=self.puck.pose.p - self.agent.tcp.pose.p,
                puck_to_goal_pos=self.goal_region.pose.p - self.puck.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        distance = info["puck_goal_distance"]
        speed = info["puck_speed"]
        return -(distance.square() / self.goal_radius**2) - 0.1 * (
            speed.square() / self.terminal_speed**2
        )

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info)

