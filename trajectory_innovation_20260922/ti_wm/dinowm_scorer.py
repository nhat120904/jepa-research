"""DINO-WM (Zhou et al., ICML 2025) as a scorer of the Diffusion Policy's PushT candidates.

The official PushT checkpoint and planning objective, unchanged; only the interface is adapted:
- Observation. DINO-WM's PushT renderer and gym-pusht draw the same 512-px scene (white canvas, LightGreen goal polygon,
  pymunk debug draw of block and agent, no action marker) and cv2-resize it; the adapter resizes the live env's canvas
  to 224. Proprio = agent position and velocity (DINO-WM PushT, with_velocity=True), normalized with its constants.
- Actions. DINO-WM acts with relative targets rel_t = target_t - agent_position_t (verified on its pusht_noise data,
  max error 3e-5), frameskip 5 (10-D macro actions). The PushT agent is a KINEMATIC pymunk body driven by the PD law
  (k_p 100, k_v 20, ten 0.01 s substeps per action), so its positions under a candidate's absolute targets follow
  exactly from its current position and velocity; contacts cannot alter them. No simulated future is read.
- Horizon. The 8 executed actions = one macro step + 3 actions; the second macro step holds the last target twice.
  With longer executed chunks (docs/CTA_REPLAN_INTERVAL_PROTOCOL.md) macro = ceil(T / 5): 15 actions = 3 macro steps.
- Cost. DINO-WM's "last" objective: MSE between the final predicted visual latent and the goal's, plus alpha times the
  MSE of proprio embeddings. Score = minus the cost averaged over the goal states (the CTA readers average goal images).
Goal states: the goal pose of the block with the agent where it is in each of the planner's goal frames.
Compute node only (loads the model).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027, 2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])
FRAMESKIP, MACRO = 5, 2
DT, SUBSTEPS, KP, KV, ACTION_SCALE = 0.01, 10, 100.0, 20.0, 100.0
GOAL_POSE = (256.0, 256.0, np.pi / 4)
ROYAL_BLUE = np.array([65, 105, 225])


def pd_positions(pos, vel, targets):
    """Kinematic agent under the PushT PD law. pos, vel (N, 2); targets (N, T, 2) -> positions at the start of each
    action (N, T, 2). Semi-implicit Euler exactly as gym-pusht: v += a dt, then the space step moves p by v dt."""
    p, v = np.asarray(pos, np.float64).copy(), np.asarray(vel, np.float64).copy()
    targets = np.asarray(targets, np.float64)
    out = np.zeros(targets.shape)
    for t in range(targets.shape[1]):
        out[:, t] = p
        for _ in range(SUBSTEPS):
            v = v + (KP * (targets[:, t] - p) - KV * v) * DT
            p = p + v * DT
    return out


def macro_actions(pos, vel, chunks, macro=MACRO):
    """(N, 2), (N, 2), (N, T, 2) absolute targets -> (N, macro, 10) normalized DINO-WM macro actions."""
    chunks = np.asarray(chunks, np.float64)
    need = macro * FRAMESKIP - chunks.shape[1]
    targets = np.concatenate([chunks, np.repeat(chunks[:, -1:], need, 1)], 1) if need > 0 else chunks[:, :macro * FRAMESKIP]
    rel = targets - pd_positions(pos, vel, targets)
    a = (torch.as_tensor(rel, dtype=torch.float32) / ACTION_SCALE - ACTION_MEAN) / ACTION_STD
    return a.reshape(len(a), macro, FRAMESKIP * 2)


def agent_from_frame(frame):
    """Agent position (512-px world units) from a rendered frame: centroid of the RoyalBlue agent pixels."""
    f = np.asarray(frame).astype(np.int32)
    mask = np.abs(f - ROYAL_BLUE).sum(-1) < 60
    if not mask.any():
        raise ValueError("agent not found in goal frame")
    ys, xs = np.nonzero(mask)
    scale = 512.0 / f.shape[0]
    return np.array([(xs.mean() + 0.5) * scale, (ys.mean() + 0.5) * scale])


class DinoWMScorer:
    def __init__(self, dino_root, ckpt_dir, device, goal_frames, render_env, alpha=1.0,
                 extra_site="/mnt/data/nhatnc129/jepa/trajectory_innovation/dinowm_extra_site", macro=MACRO):
        root = str(Path(dino_root).resolve())
        if root not in sys.path:
            sys.path.append(root)          # appended: DINO-WM's generic top-level names must not shadow installed ones
        import models  # noqa: F401
        if not str(Path(models.__file__).resolve()).startswith(root):
            raise ImportError(f"'models' resolves to {models.__file__}, not DINO-WM")
        from models.visual_world_model import VWorldModel
        hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"       # pickled encoder classes live here
        # the checkpoint pickles accelerate objects: DINO-WM's pinned extra packages, appended last so that the
        # environment's own versions (pymunk, gym-pusht deps) keep precedence
        for extra in (hub, Path(extra_site)):
            if extra.is_dir() and str(extra) not in sys.path:
                sys.path.append(str(extra))
        payload = torch.load(Path(ckpt_dir) / "checkpoints" / "model_latest.pth", map_location=device, weights_only=False)
        if "encoder" not in payload:                                          # frozen encoder not saved: rebuild it
            from models.dino import DinoV2Encoder
            payload["encoder"] = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens").to(device)
        self.model = VWorldModel(image_size=224, num_hist=3, num_pred=1, encoder=payload["encoder"],
                                 proprio_encoder=payload["proprio_encoder"], action_encoder=payload["action_encoder"],
                                 decoder=payload.get("decoder"), predictor=payload["predictor"], proprio_dim=10,
                                 action_dim=10, concat_dim=1, num_action_repeat=1, num_proprio_repeat=1,
                                 train_encoder=False, train_predictor=True, train_decoder=True).to(device)
        self.model.eval()                      # VWorldModel.eval() sets its parts to eval but returns None
        self.device, self.alpha, self.macro = device, alpha, macro
        from torchvision import transforms
        self.transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
        # goal states: block at the goal pose, agent where it is in each goal frame; rendered by the live renderer
        goals, props = [], []
        for frame in goal_frames:
            agent = agent_from_frame(frame)
            goals.append(self.render_state(render_env, agent, GOAL_POSE))
            props.append(np.r_[agent, 0.0, 0.0])
        with torch.inference_mode():
            self.z_goal = self.model.encode_obs(self.obs(np.stack(goals), np.stack(props)))

    @staticmethod
    def render_state(env, agent, block):
        u = getattr(env, "unwrapped", env)
        u.agent.position = tuple(agent)
        u.agent.velocity = (0.0, 0.0)
        u.block.position = tuple(block[:2])
        u.block.angle = float(block[2])
        u.space.step(1e-6)                 # refresh shape caches without moving anything measurably
        return canvas224(u)

    def obs(self, frames224, proprio):
        v = torch.as_tensor(np.asarray(frames224), device=self.device).permute(0, 3, 1, 2).float() / 255.0
        v = self.transform(v)[:, None]                                                      # (N, 1, 3, 224, 224)
        p = ((torch.as_tensor(np.asarray(proprio), dtype=torch.float32) - PROPRIO_MEAN) / PROPRIO_STD).to(self.device)
        return {"visual": v, "proprio": p[:, None]}

    @torch.inference_mode()
    def score(self, envs, chunks):
        """envs: A live gym-pusht envs (current states); chunks (A, B, T, 2) absolute targets -> (A, B) float32."""
        a, b = len(envs), np.asarray(chunks).shape[1]
        us = [getattr(e, "unwrapped", e) for e in envs]
        frames = np.stack([canvas224(u) for u in us])
        pos = np.stack([np.array(u.agent.position) for u in us])
        vel = np.stack([np.array(u.agent.velocity) for u in us])
        obs0 = self.obs(frames, np.concatenate([pos, vel], 1))
        obs0 = {k: v.repeat_interleave(b, 0) for k, v in obs0.items()}
        act = macro_actions(np.repeat(pos, b, 0), np.repeat(vel, b, 0),
                            np.asarray(chunks).reshape(a * b, *np.asarray(chunks).shape[2:]), self.macro).to(self.device)
        z_obs, _ = self.model.rollout(obs_0=obs0, act=act)
        g = self.z_goal["visual"].shape[0]
        vis = z_obs["visual"][:, -1]                                                         # (A*B, P, D)
        pro = z_obs["proprio"][:, -1]
        cost_v = ((vis[:, None] - self.z_goal["visual"][None, :, 0]) ** 2).mean(dim=(2, 3))  # (A*B, G)
        cost_p = ((pro[:, None] - self.z_goal["proprio"][None, :, 0]) ** 2).mean(dim=2)
        cost = (cost_v + self.alpha * cost_p).mean(1)
        return (-cost).float().view(a, b).cpu().numpy()


def canvas224(u):
    """The env's 512-px canvas resized to 224 (as DINO-WM renders)."""
    return u._get_img(u._draw(), 224, 224)
