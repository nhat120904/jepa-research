"""CTA modules for OGBench visual-cube (docs/CTA_OGBENCH_PROTOCOL.md): the PushT architecture with OGBench actions.

The source encoder, code reader, FULL reader and distortion decoder are ti_wm.cta's classes unchanged: C = decision
frame (256 tokens) + previous frame (8x8), tau = end frame (256) + steps 2-4 (8x8 each). OGBench observations carry
no separate proprioception, so the proprio slot of C and tau is a constant zero for every model.
Only the action inputs differ: 5 steps x 5-D actions (PushT: 8 x 4). Matched world models:
  ActParallelFSQWM   CTA world model p(S | C, A) (Round-3/4 parallel FSQ recipe)
  ActEndpointWM      one pass, predicts the end frame's 256 tokens (ENDPOINT)
  FrameWM            per-frame latent world model: autoregressive over steps 2, 3, 4, 5, 256 tokens per step (FRAME)
  action_scorer      D_direct(C, A, g) (DIRECT)
"""
import torch
from torch import nn

from ti_wm.codec import pool_grid
from ti_wm.cta import Scorer, _param
from ti_wm.cta_parallel import EndpointWM, ParallelFSQWM
from ti_wm.sibling import TOKENS

ACT_DIM, H = 5, 5
KEEP = (2, 3, 4, 5)


def zeros_prop(n, device):
    return torch.zeros(n, 4, device=device)


def _swap_actions(module, width, chunk=H):
    module.act = nn.Linear(ACT_DIM, width)
    module.act_pos = _param(chunk, width)


class ActParallelFSQWM(ParallelFSQWM):
    def __init__(self, m=16, **kwargs):
        super().__init__(m=m, **kwargs)
        _swap_actions(self, self.queries.shape[-1])


class ActEndpointWM(EndpointWM):
    def __init__(self, chunk=H, **kwargs):
        super().__init__(**kwargs)
        _swap_actions(self, self.queries.shape[-1], chunk)


def action_scorer(layers=8, dropout=0.0):
    s = Scorer("action", layers=layers, dropout=dropout)
    width = s.ev.out_features
    s.ev, s.ev_pos = nn.Linear(ACT_DIM, width), _param(H, width)
    return s


class FrameWM(nn.Module):
    """Autoregressive frame-latent world model (DINO-WM-style): step 2 from (frame 0, actions 1-2), then one action per
    step. Each transition is an ActEndpointWM over (current frame, previous frame) and its actions (zero-padded to 2)."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.step = ActEndpointWM(chunk=2, dropout=dropout)

    def transition(self, cur, prev, acts):
        if acts.shape[1] == 1:
            acts = torch.cat([acts, torch.zeros_like(acts)], 1)
        ctx = {"cur": cur, "prev": pool_grid(prev.float(), 8).to(cur.dtype), "prop": zeros_prop(len(cur), cur.device)}
        return self.step(ctx, acts)["end"]

    def rollout(self, ctx, actions, teacher=None):
        """actions (B, 5, 5). teacher (B, 4, 256, D) actual frames at steps 2-5 for teacher forcing, or None for an
        autoregressive rollout. Returns the predicted frames at steps 2-5 (B, 4, 256, D)."""
        cur = ctx["cur"].float()
        prev = cur
        outs = []
        spans = [(0, 2), (2, 3), (3, 4), (4, 5)]
        for k, (a, b) in enumerate(spans):
            nxt = self.transition(cur, prev, actions[:, a:b])
            outs.append(nxt)
            prev, cur = cur, (teacher[:, k].float() if teacher is not None else nxt)
        return torch.stack(outs, 1)


def fut_from_frames(frames, device=None):
    """(B, 4, 256, D) frames at steps 2-5 -> the tau dict of ti_wm.cta: end = step 5, seg = steps 2-4 pooled 8x8."""
    b = frames.shape[0]
    seg = pool_grid(frames[:, :3].flatten(0, 1).float(), 8).view(b, 3, 64, frames.shape[-1])
    return {"end": frames[:, 3], "seg": seg, "prop": zeros_prop(b, frames.device if device is None else device)}


assert TOKENS == 256
