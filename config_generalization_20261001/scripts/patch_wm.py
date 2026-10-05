"""Patch-token world model on a frozen encoder (DINO-WM-style predictor).

The 56657 diagnosis showed that the LeWM CLS latent drops the puzzle button states
(linear probe 68% bits) while the same encoder's patch tokens keep them (99.99%).
This model keeps the encoder frozen and predicts the next frame's 8x8 grid of patch
tokens from three context frames and their action blocks. Frames attend
block-causally (a frame sees itself and earlier frames); each frame's tokens are
conditioned on its action block by AdaLN, as in the LeWM predictor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stable_worldmodel.wm.lewm.module import ConditionalBlock, Embedder


class PatchPredictor(nn.Module):
    def __init__(self, dim=192, tokens=64, frames=3, action_dim=25, depth=6, heads=8, mlp=1024):
        super().__init__()
        self.tokens, self.frames = tokens, frames
        self.pos = nn.Parameter(torch.randn(1, 1, tokens, dim) * 0.02)
        self.time = nn.Parameter(torch.randn(1, frames, 1, dim) * 0.02)
        self.act = Embedder(input_dim=action_dim, emb_dim=dim)
        self.blocks = nn.ModuleList([ConditionalBlock(dim, heads, dim // heads, mlp) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        mask = torch.ones(frames * tokens, frames * tokens, dtype=torch.bool)
        for f in range(frames):
            mask[f * tokens:(f + 1) * tokens, (f + 1) * tokens:] = False   # no attention to later frames
        self.register_buffer("mask", mask)

    def forward(self, z, a):
        """z (B, F, T, D) patch tokens, a (B, F, action_dim) -> (B, F, T, D): frame f predicts f + 1."""
        B, F, T, D = z.shape
        x = (z + self.pos + self.time[:, :F]).reshape(B, F * T, D)
        c = self.act(a).unsqueeze(2).expand(B, F, T, D).reshape(B, F * T, D)
        m = self.mask[: F * T, : F * T]
        for blk in self.blocks:
            x = self._block(blk, x, c, m)
        return self.out(self.norm(x)).reshape(B, F, T, D)

    @staticmethod
    def _block(blk, x, c, mask):
        # ConditionalBlock with an explicit attention mask (its Attention uses is_causal by default).
        import torch.nn.functional as Fn
        from einops import rearrange

        from stable_worldmodel.wm.lewm.module import modulate

        sh_a, sc_a, g_a, sh_m, sc_m, g_m = blk.adaLN_modulation(c).chunk(6, dim=-1)
        h = modulate(blk.norm1(x), sh_a, sc_a)
        att = blk.attn
        h = att.norm(h)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=att.heads) for t in att.to_qkv(h).chunk(3, dim=-1))
        o = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + g_a * att.to_out(rearrange(o, "b h n d -> b n (h d)"))
        x = x + g_m * blk.mlp(modulate(blk.norm2(x), sh_m, sc_m))
        return x


def patch_tokens(encoder, pixels):
    """pixels (N, 3, 64, 64) -> (N, 64, 192) last-layer patch tokens (no CLS)."""
    return encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
