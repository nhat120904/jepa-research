"""World-model arms for the offline composition pilot (see docs/COMP_PILOT_PROTOCOL.md).

Window convention: current frame t, actions t..t+L-1, future frames t+1..t+L.
Memory tokens live on a stride-4 grid of frames.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4
SUMMARY_TOKENS = 4
MAP_SIZE = 144


def sinusoid(positions: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device) / half)
    angles = positions.float()[..., None] * freq
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def transformer(width: int, depth: int, heads: int = 8, dropout: float = 0.0) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class FrameFuser(nn.Module):
    """[..., 3 cams, 16 patches, 384] visual + [..., 16] proprio -> [..., D]."""

    def __init__(self, dim: int, feat: int = 384, cams: int = 3, patches: int = 16, proprio: int = 16) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, dim))
        self.cam = nn.Parameter(torch.zeros(1, cams, 1, dim))
        self.patch = nn.Parameter(torch.zeros(1, 1, patches, dim))
        nn.init.normal_(self.cam, std=0.02)
        nn.init.normal_(self.patch, std=0.02)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.proprio = nn.Linear(proprio, dim)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        lead = visual.shape[:-3]
        v = visual.reshape(-1, *visual.shape[-3:]).float()
        x = (self.inp(v) + self.cam + self.patch).flatten(1, 2)
        pooled, _ = self.attn(self.query.expand(len(x), -1, -1), x, x, need_weights=False)
        z = pooled[:, 0] + self.proprio(proprio.reshape(-1, proprio.shape[-1]).float())
        return self.out(z).reshape(*lead, -1)


class Memory(nn.Module):
    """Causal GRU over stride-4 frame tokens (fused frame + action)."""

    def __init__(self, dim: int, action: int = 12) -> None:
        super().__init__()
        self.action = nn.Linear(action, dim)
        self.pad = nn.Parameter(torch.zeros(dim))
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def tokens(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return frames + self.action(actions)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, h0: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            tokens = torch.where(mask[..., None], tokens, self.pad.expand_as(tokens))
        with torch.autocast("cuda", enabled=False):  # keep GRU input and hidden state in one dtype
            _, h = self.gru(tokens.float(), None if h0 is None else h0.float()[None].contiguous())
        return h[0]


class SegmentPredictor(nn.Module):
    """(context h, current z, actions [B, L, 12]) -> summary [B, 4, D], endpoint [B, D]."""

    def __init__(self, dim: int, depth: int, action: int = 12, outputs: int = SUMMARY_TOKENS + 1) -> None:
        super().__init__()
        self.dim = dim
        self.action = nn.Linear(action, dim)
        self.ctx = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, outputs, dim) * 0.02)
        self.body = transformer(dim, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        b, length, _ = actions.shape
        steps = torch.arange(length, device=actions.device)
        act = self.action(actions) + sinusoid(steps, self.dim)[None]
        duration = sinusoid(torch.full((b, 1), float(length), device=actions.device), self.dim)
        ctx = torch.stack([h, z], 1) + self.ctx
        queries = self.queries.expand(b, -1, -1) + duration
        out = self.body(torch.cat([ctx, act, queries], 1))
        return self.norm(out[:, -queries.shape[1] :])


class TrajectoryEncoder(nn.Module):
    """Future fused frames [B, L, D] -> summary [B, 4, D]."""

    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.dim = dim
        self.queries = nn.Parameter(torch.randn(1, SUMMARY_TOKENS, dim) * 0.02)
        self.body = transformer(dim, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, length, _ = frames.shape
        x = frames + sinusoid(torch.arange(length, device=frames.device), self.dim)[None]
        duration = sinusoid(torch.full((b, 1), float(length), device=frames.device), self.dim)
        out = self.body(torch.cat([self.queries.expand(b, -1, -1) + duration, x], 1))
        return self.norm(out[:, :SUMMARY_TOKENS])


class ReconstructionDecoder(nn.Module):
    """Summary [B, 4, D] -> per-frame mean DINO feature [B, L, 384] (target-encoder anchor)."""

    def __init__(self, dim: int, feat: int = 384) -> None:
        super().__init__()
        self.dim = dim
        layer = nn.TransformerDecoderLayer(dim, 8, 4 * dim, 0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerDecoder(layer, 2)
        self.out = nn.Linear(dim, feat)

    def forward(self, summary: torch.Tensor, length: int) -> torch.Tensor:
        q = sinusoid(torch.arange(length, device=summary.device), self.dim)[None].expand(len(summary), -1, -1)
        return self.out(self.body(q, summary))


class MemoryUpdate(nn.Module):
    """Imagined memory after a segment: (h, summary tokens, endpoint) -> h'."""

    def __init__(self, dim: int, tokens_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * (tokens_in + 2), 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, h: torch.Tensor, tokens: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
        return h + self.net(torch.cat([h, tokens.flatten(1), endpoint], 1))


class Composer(nn.Module):
    """Ordered merge of two summaries with segment identity and durations."""

    def __init__(self, dim: int, depth: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.side = nn.Parameter(torch.randn(1, 2, 1, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, SUMMARY_TOKENS, dim) * 0.02)
        self.body = transformer(dim, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor, left_len: int, right_len: int) -> torch.Tensor:
        b = len(left)
        dev = left.device
        dl = sinusoid(torch.tensor([float(left_len)], device=dev), self.dim)
        dr = sinusoid(torch.tensor([float(right_len)], device=dev), self.dim)
        total = sinusoid(torch.tensor([float(left_len + right_len)], device=dev), self.dim)
        x = torch.cat([left + self.side[:, 0] + dl, right + self.side[:, 1] + dr, self.queries.expand(b, -1, -1) + total], 1)
        return self.norm(self.body(x)[:, -SUMMARY_TOKENS:])


def variance_penalty(x: torch.Tensor) -> torch.Tensor:
    x = x.float().reshape(len(x), -1)
    return F.relu(1.0 - torch.sqrt(x.var(0) + 1e-4)).mean()


class Arm(nn.Module):
    """Shared encoder, memory and EMA target fuser."""

    name = "base"
    uses_labels = False

    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.dim = dim
        self.fuser = FrameFuser(dim)
        self.memory = Memory(dim)
        self.target_fuser = copy.deepcopy(self.fuser).requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, decay: float) -> None:
        for online, target in self._ema_pairs():
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(decay).add_(po.detach(), alpha=1 - decay)

    def _ema_pairs(self):
        return [(self.fuser, self.target_fuser)]

    def context(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self.fuser(batch["hist_visual"], batch["hist_proprio"])
        h = self.memory(self.memory.tokens(hist, batch["hist_actions"]), batch["hist_mask"])
        z = self.fuser(batch["cur_visual"], batch["cur_proprio"])
        return h, z

    def future_targets(self, batch: dict) -> torch.Tensor:
        with torch.no_grad():
            return self.target_fuser(batch["fut_visual"], batch["fut_proprio"])

    def true_memory(self, h: torch.Tensor, batch: dict, fut_online: torch.Tensor, upto: int) -> torch.Tensor:
        """Continue memory over observed stride-4 future frames (t+4, ..., t+upto)."""
        pos = torch.arange(STRIDE - 1, upto, STRIDE, device=h.device)
        tokens = self.memory.tokens(fut_online[:, pos], batch["actions"][:, pos + 1])
        return self.memory(tokens, None, h)


class SegmentArm(Arm):
    name = "segment_nocomp"
    compose = False

    def __init__(self, dim: int, depth: int) -> None:
        super().__init__(dim, depth)
        self.predictor = SegmentPredictor(dim, depth)
        self.traj = TrajectoryEncoder(dim, 4)
        self.target_traj = copy.deepcopy(self.traj).requires_grad_(False)
        self.decoder = ReconstructionDecoder(dim)
        self.update = MemoryUpdate(dim, SUMMARY_TOKENS)
        self.composer = Composer(dim) if self.compose else None

    def _ema_pairs(self):
        return [(self.fuser, self.target_fuser), (self.traj, self.target_traj)]

    def predict(self, h, z, actions):
        out = self.predictor(h, z, actions)
        return out[:, :SUMMARY_TOKENS], out[:, SUMMARY_TOKENS]

    def represent(self, summary, endpoint):
        return torch.cat([summary, endpoint[:, None]], 1)

    def loss(self, batch: dict, weights: dict) -> tuple[torch.Tensor, dict]:
        length = batch["actions"].shape[1]
        m = length // 2
        h, z = self.context(batch)
        fut_target = self.future_targets(batch)
        fut_online = self.fuser(batch["fut_visual"], batch["fut_proprio"])
        with torch.no_grad():
            s_full_t = self.target_traj(fut_target)
            s_left_t = self.target_traj(fut_target[:, :m])
            s_right_t = self.target_traj(fut_target[:, m:])
        e_full_t, e_left_t, e_right_t = fut_target[:, -1], fut_target[:, m - 1], fut_target[:, -1]

        s_full, e_full = self.predict(h, z, batch["actions"])
        s_left, e_left = self.predict(h, z, batch["actions"][:, :m])
        h_mid = self.true_memory(h, batch, fut_online, m)
        s_right_tf, e_right_tf = self.predict(h_mid, fut_online[:, m - 1], batch["actions"][:, m:])
        h_roll = self.update(h, s_left, e_left)
        s_right_roll, e_right_roll = self.predict(h_roll, e_left, batch["actions"][:, m:])

        def pair(s, e, st, et):
            return F.smooth_l1_loss(s.float(), st.float()) + F.smooth_l1_loss(e.float(), et.float())

        terms = {
            "pred_full": pair(s_full, e_full, s_full_t, e_full_t),
            "pred_left": pair(s_left, e_left, s_left_t, e_left_t),
            "pred_right_tf": pair(s_right_tf, e_right_tf, s_right_t, e_right_t),
            "pred_right_roll": pair(s_right_roll, e_right_roll, s_right_t, e_right_t),
            "memory": F.mse_loss(self.update(h, s_left_t, e_left_t).float(), h_mid.detach().float()),
        }
        s_online = self.traj(fut_online)
        terms.update(self.anchor_terms(s_online, batch, length))
        terms["variance"] = variance_penalty(e_full) + variance_penalty(s_full.mean(1)) + variance_penalty(s_online.mean(1))
        if self.compose:
            left_online = self.traj(fut_online[:, :m])
            right_online = self.traj(fut_online[:, m:])
            terms["comp_observed"] = F.smooth_l1_loss(self.composer(left_online, right_online, m, length - m).float(), s_full_t.float())
            terms["comp_predicted"] = F.smooth_l1_loss(self.composer(s_left, s_right_roll, m, length - m).float(), s_full_t.float())
        total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}

    def anchor_terms(self, s_online: torch.Tensor, batch: dict, length: int) -> dict:
        """52634 anchor: per-frame DINO feature averaged over cameras and patches."""
        recon_target = F.layer_norm(batch["fut_visual"].float().mean(dim=(2, 3)), (384,))
        return {"recon": F.mse_loss(self.decoder(s_online, length).float(), recon_target)}

    @torch.no_grad()
    def true_representation(self, batch: dict) -> torch.Tensor:
        fut_target = self.future_targets(batch)
        return self.represent(self.target_traj(fut_target), fut_target[:, -1])

    @torch.no_grad()
    def rollout(self, batch: dict, parts: list[int], composed: bool) -> dict:
        """Sequential parts from the predicted context. Returns part representations and composed one."""
        h, z = self.context(batch)
        reps, summary, start = [], None, 0
        for length in parts:
            s, e = self.predict(h, z, batch["actions"][:, start : start + length])
            reps.append(self.represent(s, e))
            if composed and self.composer is not None:
                summary = s if summary is None else self.composer(summary, s, start, length)
            h, z, start = self.update(h, s, e), e, start + length
        out = {"parts": reps}
        if composed and self.composer is not None:
            out["composed"] = self.represent(summary, reps[-1][:, -1])
        return out


class CompositionalSegmentArm(SegmentArm):
    name = "segment_comp"
    compose = True


class SpatialReconstructionDecoder(nn.Module):
    """Target correction (COMP_PILOT_TARGET_FIX_PROTOCOL.md): summary [B, 4, D] ->
    every camera x patch token of every 4th window frame, plus proprio of every frame."""

    def __init__(self, dim: int, feat: int = 384, cams: int = 3, patches: int = 16, proprio: int = 16, frame_stride: int = STRIDE) -> None:
        super().__init__()
        self.dim, self.cams, self.patches, self.frame_stride, self.feat = dim, cams, patches, frame_stride, feat
        self.cam = nn.Parameter(torch.randn(1, 1, cams, 1, dim) * 0.02)
        self.patch = nn.Parameter(torch.randn(1, 1, 1, patches, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, feat))
        self.proprio_norm = nn.LayerNorm(dim)
        self.proprio_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.proprio_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proprio))

    def frames(self, length: int, device) -> torch.Tensor:
        return torch.arange(self.frame_stride - 1, length, self.frame_stride, device=device)

    def forward(self, summary: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        b = len(summary)
        frames = self.frames(length, summary.device)
        q = sinusoid(frames, self.dim)[None, :, None, None, :] + self.cam + self.patch
        q = q.expand(b, -1, -1, -1, -1).reshape(b, -1, self.dim)
        x = q + self.attn(self.query_norm(q), summary, summary, need_weights=False)[0]
        x = x + self.mlp(x)
        visual = self.out(x).reshape(b, len(frames), self.cams, self.patches, self.feat)
        pq = sinusoid(torch.arange(length, device=summary.device), self.dim)[None].expand(b, -1, -1)
        p = pq + self.proprio_attn(self.proprio_norm(pq), summary, summary, need_weights=False)[0]
        return visual, self.proprio_out(p)


class SpatialSegmentArm(SegmentArm):
    name = "segment_nocomp_spatial"
    compose = False

    def __init__(self, dim: int, depth: int) -> None:
        super().__init__(dim, depth)
        self.decoder = SpatialReconstructionDecoder(dim)

    def anchor_terms(self, s_online: torch.Tensor, batch: dict, length: int) -> dict:
        visual, proprio = self.decoder(s_online, length)
        frames = self.decoder.frames(length, s_online.device)
        target = F.layer_norm(batch["fut_visual"][:, frames].float(), (384,))
        return {
            "recon": F.mse_loss(visual.float(), target),
            "recon_proprio": F.mse_loss(proprio.float(), batch["fut_proprio"].float()),
        }


class SpatialCompositionalSegmentArm(SpatialSegmentArm):
    name = "segment_comp_spatial"
    compose = True


class FrameRolloutArm(Arm):
    name = "frame_rollout"

    def __init__(self, dim: int, depth: int) -> None:
        super().__init__(dim, depth)
        self.predictor = FramePredictor(dim, depth)

    def roll_memory(self, h, frames, actions):
        pos = torch.arange(STRIDE - 1, frames.shape[1], STRIDE, device=h.device)
        if len(pos) == 0:
            return h
        return self.memory(self.memory.tokens(frames[:, pos], actions[:, (pos + 1).clamp(max=actions.shape[1] - 1)]), None, h)

    def loss(self, batch: dict, weights: dict) -> tuple[torch.Tensor, dict]:
        length = batch["actions"].shape[1]
        m = length // 2
        h, z = self.context(batch)
        fut_target = self.future_targets(batch)
        fut_online = self.fuser(batch["fut_visual"], batch["fut_proprio"])
        full = self.predictor(h, z, batch["actions"])
        left = self.predictor(h, z, batch["actions"][:, :m])
        h_mid = self.true_memory(h, batch, fut_online, m)
        right_tf = self.predictor(h_mid, fut_online[:, m - 1], batch["actions"][:, m:])
        h_roll = self.roll_memory(h, left, batch["actions"][:, :m])
        right_roll = self.predictor(h_roll, left[:, -1], batch["actions"][:, m:])
        tgt = fut_target.float()
        terms = {
            "pred_full": F.smooth_l1_loss(full.float(), tgt),
            "pred_left": F.smooth_l1_loss(left.float(), tgt[:, :m]),
            "pred_right_tf": F.smooth_l1_loss(right_tf.float(), tgt[:, m:]),
            "pred_right_roll": F.smooth_l1_loss(right_roll.float(), tgt[:, m:]),
            "variance": variance_penalty(full.mean(1)),
        }
        total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}

    @torch.no_grad()
    def rollout(self, batch: dict, parts: list[int], composed: bool) -> dict:
        h, z = self.context(batch)
        frames, reps, start = [], [], 0
        for length in parts:
            actions = batch["actions"][:, start : start + length]
            pred = self.predictor(h, z, actions)
            frames.append(pred)
            reps.append(pred)
            h, z, start = self.roll_memory(h, pred, actions), pred[:, -1], start + length
        out = {"parts": reps}
        if composed:
            out["composed"] = torch.cat(frames, 1)
        return out


class FramePredictor(nn.Module):
    """(h, z, actions [B, L, 12]) -> predicted fused latent of frames t+1..t+L [B, L, D]."""

    def __init__(self, dim: int, depth: int, action: int = 12) -> None:
        super().__init__()
        self.dim = dim
        self.action = nn.Linear(action, dim)
        self.ctx = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.body = transformer(dim, depth)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, h, z, actions):
        length = actions.shape[1]
        act = self.action(actions) + sinusoid(torch.arange(length, device=actions.device), self.dim)[None]
        ctx = torch.stack([h, z], 1) + self.ctx
        out = self.body(torch.cat([ctx, act], 1))[:, 2:]
        return self.out(self.norm(out))


class MapUnionArm(Arm):
    """Supervised control: predicts contact map and scalars directly; maps compose by OR."""

    name = "map_union"
    uses_labels = True

    def __init__(self, dim: int, depth: int) -> None:
        super().__init__(dim, depth)
        self.predictor = SegmentPredictor(dim, depth, outputs=SUMMARY_TOKENS + 1)
        self.map_head = nn.Linear(dim * SUMMARY_TOKENS, MAP_SIZE)
        self.scalar_head = nn.Linear(dim * SUMMARY_TOKENS, 3)  # count_increment, count_empty, area_empty
        self.update = MemoryUpdate(dim, SUMMARY_TOKENS)

    def heads(self, out):
        flat = out[:, :SUMMARY_TOKENS].flatten(1)
        return self.map_head(flat), self.scalar_head(flat), out[:, SUMMARY_TOKENS]

    def supervised(self, map_logits, scalars, labels):
        target = torch.stack([labels["count_increment"], labels["count_empty"], labels["area_empty"] / 10.0], 1)
        return F.binary_cross_entropy_with_logits(map_logits.float(), labels["map"].float()) + F.smooth_l1_loss(scalars.float(), target.float())

    def loss(self, batch: dict, weights: dict) -> tuple[torch.Tensor, dict]:
        length = batch["actions"].shape[1]
        m = length // 2
        h, z = self.context(batch)
        fut_target = self.future_targets(batch)
        fut_online = self.fuser(batch["fut_visual"], batch["fut_proprio"])
        full = self.predictor(h, z, batch["actions"])
        left = self.predictor(h, z, batch["actions"][:, :m])
        h_mid = self.true_memory(h, batch, fut_online, m)
        right_tf = self.predictor(h_mid, fut_online[:, m - 1], batch["actions"][:, m:])
        lm, ls, le = self.heads(left)
        h_roll = self.update(h, left[:, :SUMMARY_TOKENS], le)
        right_roll = self.predictor(h_roll, le, batch["actions"][:, m:])
        terms = {}
        for key, out, labels, endpoint in (
            ("full", full, batch["labels_full"], fut_target[:, -1]),
            ("left", left, batch["labels_left"], fut_target[:, m - 1]),
            ("right_tf", right_tf, batch["labels_right"], fut_target[:, -1]),
            ("right_roll", right_roll, batch["labels_right"], fut_target[:, -1]),
        ):
            mp, sc, e = self.heads(out)
            terms[f"sup_{key}"] = self.supervised(mp, sc, labels) + F.smooth_l1_loss(e.float(), endpoint.float())
        terms["memory"] = F.mse_loss(self.update(h, left[:, :SUMMARY_TOKENS].detach(), fut_target[:, m - 1]).float(), h_mid.detach().float())
        terms["variance"] = variance_penalty(full[:, SUMMARY_TOKENS])
        total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}

    @torch.no_grad()
    def rollout(self, batch: dict, parts: list[int], composed: bool) -> dict:
        h, z = self.context(batch)
        maps, start = [], 0
        for length in parts:
            out = self.predictor(h, z, batch["actions"][:, start : start + length])
            mp, _, e = self.heads(out)
            maps.append(torch.sigmoid(mp)[..., None])
            h, z, start = self.update(h, out[:, :SUMMARY_TOKENS], e), e, start + length
        result = {"parts": maps}
        if composed:
            result["composed"] = torch.stack(maps, 0).amax(0)
        return result


ARMS = {cls.name: cls for cls in (SegmentArm, CompositionalSegmentArm, FrameRolloutArm, MapUnionArm, SpatialSegmentArm, SpatialCompositionalSegmentArm)}


class Readout(nn.Module):
    """Identical architecture for every arm: token set -> scalars and 144-cell map logits."""

    def __init__(self, token_dim: int, width: int = 128, max_tokens: int = 256) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, width))
        self.pos = nn.Parameter(torch.randn(1, max_tokens, width) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.body = transformer(width, 1, heads=4)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3 + MAP_SIZE))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inp(tokens.float()) + self.pos[:, : tokens.shape[1]]
        x = self.body(torch.cat([self.query.expand(len(x), -1, -1), x], 1))[:, 0]
        out = self.head(x)
        return out[:, :3], out[:, 3:]
