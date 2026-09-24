"""PushT runtime for gates A-C: pinned policy, seeded candidate banks, branch cloning, scorers.

Compute-node only. Every function mirrors the pinned LeRobot/gym-pusht code paths; the smoke
job checks bitwise equivalence against the official select_action loop.
"""

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from ti_wm.contract import compatible_config, native_action_slice

ENV_ID = "gym_pusht/PushT-v0"
ENV_KWARGS = {"obs_type": "pixels_agent_pos", "render_mode": "rgb_array",
              "visualization_width": 384, "visualization_height": 384}
MAX_STEPS = 300
BANK = 32
DINO_HUB = "/mnt/data/nhatnc129/jepa/cache/torch/hub/facebookresearch_dinov2_main"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_env(max_episode_steps=None):
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the environment)

    kwargs = dict(ENV_KWARGS)
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    return gym.make(ENV_ID, **kwargs)


# ----------------------------------------------------------------------------- branching

def clone_deepcopy(env):
    """Full copy of an unwrapped PushT env; pymunk copies bodies, handlers and arbiters.

    The CollisionHandler wrapper holds raw cffi pointers, so it is detached for the copy and
    re-registered on the copied space (pymunk restores the handler callbacks themselves).
    """
    handler = env.__dict__.pop("collision_handeler")
    try:
        new = copy.deepcopy(env)
    finally:
        env.collision_handeler = handler
    new.collision_handeler = new.space.add_collision_handler(0, 0)
    new.collision_handeler.post_solve = new._handle_collision
    return new


def _copy_kinematics(src_env, dst_env):
    for src, dst in ((src_env.agent, dst_env.agent), (src_env.block, dst_env.block)):
        dst.center_of_gravity = src.center_of_gravity
        dst.angle = src.angle
        dst.position = src.position
        dst.velocity = src.velocity
        dst.angular_velocity = src.angular_velocity


def clone_deepcopy_fixed(env):
    """deepcopy (keeps arbiters) with body kinematics re-asserted from the source.

    Smoke 53787: plain deepcopy repeated exactly but drifted up to 112 units from the live
    continuation, so restored body state cannot be trusted without re-assertion.
    """
    new = clone_deepcopy(env)
    _copy_kinematics(env, new)
    return new


def clone_manual(env):
    """Fallback: fresh space with copied body kinematics (loses the contact warm-start cache)."""
    new = copy.copy(env)
    new._setup()
    new.goal_pose = np.array(env.goal_pose, copy=True)
    _copy_kinematics(env, new)
    new._last_action = env._last_action
    new.n_contact_points = env.n_contact_points
    return new


CLONERS = {"deepcopy_fixed": clone_deepcopy_fixed, "manual": clone_manual, "deepcopy": clone_deepcopy}
# Amendment 1 (docs/GATE_ABC_PROTOCOL.md): a cloner must repeat exactly AND track the live
# continuation within this tolerance; the first qualifying entry in CLONERS order is used.
LIVE_TOLERANCE = 0.01


def clone_integrity(env, cloner):
    """State right after cloning: must equal the source, with bodies owned by the new space."""
    new = cloner(env)
    return {"pre_step_max_abs": float(np.abs(physical_state(new) - physical_state(env)).max()),
            "agent_in_space": new.agent in new.space.bodies,
            "block_in_space": new.block in new.space.bodies,
            "cog_equal": tuple(new.block.center_of_gravity) == tuple(env.block.center_of_gravity)}


@dataclasses.dataclass
class Branch:
    env: object
    hist: list            # last two observations, [previous, current]
    t: int = 0
    success: bool = False
    coverage: float = 0.0
    max_coverage: float = 0.0


def reset_branch(root):
    env = make_env().unwrapped
    obs, _ = env.reset(seed=root)
    coverage = float(env._get_coverage())
    return Branch(env=env, hist=[obs, obs], coverage=coverage, max_coverage=coverage)


def run_prefix(branch, actions, cloner=clone_deepcopy):
    """Execute a chunk on a CLONE of the branch; stop at success or the step limit."""
    out = Branch(cloner(branch.env), list(branch.hist), branch.t, branch.success,
                 branch.coverage, branch.max_coverage)
    for action in actions:
        if out.success or out.t >= MAX_STEPS:
            break
        obs, _, terminated, _, info = out.env.step(np.asarray(action, dtype=np.float32))
        out.t += 1
        out.hist = [out.hist[-1], obs]
        out.coverage = float(info["coverage"])
        out.max_coverage = max(out.max_coverage, out.coverage)
        out.success = bool(terminated)
    return out


def done(branch):
    return branch.success or branch.t >= MAX_STEPS


def physical_state(env):
    """Everything the next physics step depends on, for exactness checks."""
    return np.array([*env.agent.position, *env.agent.velocity, *env.block.position,
                     env.block.angle, *env.block.velocity, env.block.angular_velocity],
                    dtype=np.float64)


# ----------------------------------------------------------------------------- policy

class PolicyRunner:
    def __init__(self, checkpoint_dir, device="cuda"):
        import draccus
        import safetensors.torch
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        checkpoint_dir = Path(checkpoint_dir)
        raw = json.loads((checkpoint_dir / "config.json").read_text())
        fields = {f.name for f in dataclasses.fields(DiffusionConfig)}
        config, self.removed_fields = compatible_config(raw, fields)
        config = draccus.decode(DiffusionConfig, config)
        policy = DiffusionPolicy(config)
        missing, unexpected = safetensors.torch.load_model(
            policy, checkpoint_dir / "model.safetensors", strict=True)
        self.missing, self.unexpected = list(missing), list(unexpected)
        for name, buffer in policy.named_buffers():
            if not torch.isfinite(buffer).all():
                raise RuntimeError(f"Non-finite normalization buffer after load: {name}")
        self.device = torch.device(device)
        self.policy = policy.eval().to(self.device)
        self.config = config
        self.start, self.end = native_action_slice(raw)
        self.parameters = sum(p.numel() for p in policy.parameters())

    def batch(self, hists):
        """Stack [previous, current] observations exactly as select_action's queues do."""
        from lerobot.common.envs.utils import preprocess_observation

        steps = []
        for j in range(self.config.n_obs_steps):
            obs = {"pixels": np.stack([h[j]["pixels"] for h in hists]),
                   "agent_pos": np.stack([h[j]["agent_pos"] for h in hists])}
            step = preprocess_observation(obs)
            step = {k: v.to(self.device) for k, v in step.items()}
            step = dict(self.policy.normalize_inputs(step))
            step["observation.images"] = torch.stack(
                [step[key] for key in self.config.image_features], dim=-4)
            steps.append(step)
        return {k: torch.stack([s[k] for s in steps], dim=1)
                for k in ("observation.state", "observation.images")}

    @torch.inference_mode()
    def denoise(self, global_cond, generator):
        """conditional_sample with one generator per sample (diffusers randn_tensor semantics)."""
        from diffusers.utils.torch_utils import randn_tensor

        diffusion = self.policy.diffusion
        shape = (global_cond.shape[0], self.config.horizon, self.config.action_feature.shape[0])
        sample = randn_tensor(shape, generator=generator, device=self.device, dtype=global_cond.dtype)
        diffusion.noise_scheduler.set_timesteps(diffusion.num_inference_steps)
        for t in diffusion.noise_scheduler.timesteps:
            timestep = torch.full(sample.shape[:1], t, dtype=torch.long, device=self.device)
            output = diffusion.unet(sample, timestep, global_cond=global_cond)
            sample = diffusion.noise_scheduler.step(output, t, sample, generator=generator).prev_sample
        return sample

    @torch.inference_mode()
    def _actions(self, global_cond, generator):
        sample = self.denoise(global_cond, generator)[:, self.start:self.end]
        return self.policy.unnormalize_outputs({"action": sample})["action"]

    @torch.inference_mode()
    def bank(self, hist, seeds):
        """K candidate chunks from ONE state; candidate k depends only on seeds[k]."""
        cond = self.policy.diffusion._prepare_global_conditioning(self.batch([hist]))
        cond = cond.expand(len(seeds), -1).contiguous()
        generators = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]
        return self._actions(cond, generators).float().cpu().numpy()

    @torch.inference_mode()
    def draw(self, hists, seeds):
        """One chunk per state for a batch of different states (continuations)."""
        cond = self.policy.diffusion._prepare_global_conditioning(self.batch(hists))
        generators = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]
        return self._actions(cond, generators).float().cpu().numpy()


# ----------------------------------------------------------------------------- scorers

def physical_scores(branches):
    return [b.coverage for b in branches]


class VisualScorer:
    """Pinned kernel: -mean squared DINOv2 ViT-S/14 patch-feature distance to a goal map."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        model = torch.hub.load(DINO_HUB, "dinov2_vits14", source="local", pretrained=True)
        self.model = model.eval().to(self.device).requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.goal = None

    @torch.inference_mode()
    def features(self, frames):
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.model.forward_features((x - self.mean) / self.std)["x_norm_patchtokens"]

    def set_goal(self, goal_map):
        self.goal = goal_map.to(self.device)

    @torch.inference_mode()
    def scores(self, frames):
        f = self.features(frames)
        return (-(f - self.goal[None]).pow(2).mean(dim=(1, 2))).float().cpu().tolist()


def final_frames(branches):
    return np.stack([b.hist[-1]["pixels"] for b in branches])
