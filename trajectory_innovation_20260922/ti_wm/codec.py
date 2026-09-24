"""Gate D: conditional compact codes of a candidate's actual future, and the reader D(C, S, q).

docs/GATE_D_CODEC_PROTOCOL.md. All image inputs are PCA-reduced DINOv2 16x16 tokens (dim 128).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ti_wm.sibling import PCA_DIM, TOKENS

LEVELS = (8, 8, 8, 5, 5, 5)          # FSQ levels per code token: sum(log2) = 15.97 bits
SEG_TOKENS = 64                       # intermediate segment frames are 2x2-pooled to 8x8


def bits_per_token(levels=LEVELS):
    return sum(math.log2(n) for n in levels)


class FSQ(nn.Module):
    """Finite scalar quantization (Mentzer et al., 2023) with a straight-through estimator."""

    def __init__(self, levels=LEVELS, eps=1e-3):
        super().__init__()
        lv = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("half_l", (lv - 1) * (1 - eps) / 2)
        self.register_buffer("offset", torch.where(lv % 2 == 0, 0.5, 0.0))
        self.register_buffer("half_w", torch.floor(lv / 2))

    def forward(self, z):
        with torch.autocast(z.device.type, enabled=False):
            return self._quantize(z.float())

    def _quantize(self, z):
        shift = torch.atanh(self.offset / self.half_l)
        bounded = torch.tanh(z + shift) * self.half_l - self.offset
        quantized = bounded + (torch.round(bounded) - bounded).detach()
        return quantized / self.half_w


def pool_grid(tokens, out_side):
    """(B, 256, D) tokens on a 16x16 grid -> (B, out_side^2, D) by average pooling."""
    b, n, d = tokens.shape
    side = int(math.isqrt(n))
    grid = tokens.reshape(b, side, side, d).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(grid, out_side).flatten(2).transpose(1, 2)


class CodeEncoder(nn.Module):
    """q_phi(S | C, tau): M learned queries cross-attend to the future tokens (and C if conditional), then FSQ.

    mode: "cond" (sees C), "uncond" (future only), "traj" (sees C, the end frame and 3 intermediate frames),
    "pool" (per-frame control: end tokens pooled to M tokens, linear projection, FSQ; no attention, no C).
    """

    def __init__(self, m, mode, dim=PCA_DIM, width=256, layers=2, heads=4, levels=LEVELS):
        super().__init__()
        self.m, self.mode = m, mode
        self.fsq = FSQ(levels)
        if mode == "pool":
            self.side = int(math.isqrt(m))
            assert self.side ** 2 == m
            self.to_code = nn.Linear(dim, len(levels))
            return
        self.queries = nn.Parameter(torch.randn(m, width) * 0.02)
        self.proj = nn.Linear(dim, width)
        self.type_embed = nn.Parameter(torch.randn(3, width) * 0.02)       # ctx, end, intermediate
        self.pos_embed = nn.Parameter(torch.randn(TOKENS, width) * 0.02)
        self.seg_pos = nn.Parameter(torch.randn(3, SEG_TOKENS, width) * 0.02)
        layer = nn.TransformerDecoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.to_code = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(levels)))

    def forward(self, ctx, end, seg=None):
        """ctx/end: (B, 256, dim); seg: (B, 3, 64, dim) for mode "traj". Returns the code (B, M, len(levels))."""
        if self.mode == "pool":
            return self.fsq(self.to_code(pool_grid(end.float(), self.side)))
        mem = [self.proj(end.float()) + self.pos_embed + self.type_embed[1]]
        if self.mode in ("cond", "traj"):
            mem.append(self.proj(ctx.float()) + self.pos_embed + self.type_embed[0])
        if self.mode == "traj":
            mem.append((self.proj(seg.float()) + self.seg_pos + self.type_embed[2]).flatten(1, 2))
        queries = self.queries.expand(end.shape[0], -1, -1)
        return self.fsq(self.to_code(self.decoder(queries, torch.cat(mem, dim=1))))


class CodeReader(nn.Module):
    """D(C, S, q): scalar score from context tokens, code tokens, goal tokens and end proprio."""

    def __init__(self, code_dim, n_code, dim=PCA_DIM, width=256, layers=4, heads=4):
        super().__init__()
        self.ctx_proj, self.goal_proj = nn.Linear(dim, width), nn.Linear(dim, width)
        self.code_proj = nn.Linear(code_dim, width)
        self.pos_embed = nn.Parameter(torch.randn(TOKENS, width) * 0.02)
        self.code_pos = nn.Parameter(torch.randn(n_code, width) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(4, width) * 0.02)
        self.proprio = nn.Linear(4, width)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, ctx, code, goal, proprio):
        parts = [self.ctx_proj(ctx.float()) + self.pos_embed + self.type_embed[0],
                 self.code_proj(code.float()) + self.code_pos + self.type_embed[1],
                 self.goal_proj(goal.float()) + self.pos_embed + self.type_embed[2],
                 (self.proprio(proprio.float()) + self.type_embed[3])[:, None]]
        x = torch.cat([self.cls.expand(ctx.shape[0], -1, -1), *parts], dim=1)
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


class CodeDecoder(nn.Module):
    """Auxiliary distortion head: predict the end-frame tokens from (C, S). Keeps S from becoming rank-only."""

    def __init__(self, code_dim, n_code, dim=PCA_DIM, width=256, layers=2, heads=4):
        super().__init__()
        self.ctx_proj, self.code_proj = nn.Linear(dim, width), nn.Linear(code_dim, width)
        self.pos_embed = nn.Parameter(torch.randn(TOKENS, width) * 0.02)
        self.code_pos = nn.Parameter(torch.randn(n_code, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, dim)

    def forward(self, ctx, code):
        x = torch.cat([self.ctx_proj(ctx.float()) + self.pos_embed,
                       self.code_proj(code.float()) + self.code_pos], dim=1)
        return self.out(self.encoder(x)[:, :TOKENS])
