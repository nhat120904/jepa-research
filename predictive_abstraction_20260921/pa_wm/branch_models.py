"""Prospective pilot v2: spatial reconstruction, no arbitrary feature truncation."""
import torch
from torch import nn
from .models import (SpatialActionSummary, SpatialObservedCodec, QueryReader, positions)


class SpatialDecoder(nn.Module):
    def __init__(self, patches, dim, width):
        super().__init__()
        self.spatial = nn.Parameter(torch.randn(patches, width) * .02)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)

    def tokens(self, context, horizon, endpoint=False):
        time = positions(horizon, context.shape[-1], context.device, context.dtype)
        if endpoint:
            time = time[-1:]
        q = (time[:, None] + self.spatial[None]).flatten(0, 1)
        q = q[None].expand(len(context), -1, -1)
        x, _ = self.attn(q, context, context, need_weights=False)
        return self.norm(q + x)

    def reconstruct(self, tokens):
        return self.out(tokens).unflatten(1, (-1, len(self.spatial)))


class ReconstructionCodec(nn.Module):
    def __init__(self, patches, dim, width, tokens):
        super().__init__()
        self.encoder = SpatialObservedCodec(patches, dim, width, tokens)
        self.decoder = SpatialDecoder(patches, dim, width)

    def forward(self, future):
        return self.encoder(future)

    def reconstruct(self, summary, horizon):
        return self.decoder.reconstruct(self.decoder.tokens(summary, horizon))


class SpatialForecast(nn.Module):
    def __init__(self, patches, dim, width, endpoint=False):
        super().__init__()
        self.context = SpatialActionSummary(patches, dim, width=width)
        self.decoder = SpatialDecoder(patches, dim, width)
        self.endpoint = endpoint

    def forward(self, history, actions):
        return self.decoder.tokens(self.context.context(history, actions),
                                   actions.shape[1], self.endpoint)

    def reconstruct(self, tokens):
        return self.decoder.reconstruct(tokens)


class QueryOnly(nn.Module):
    def __init__(self, query_dim, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(query_dim, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
                                 nn.Sigmoid())

    def forward(self, queries):
        return self.net(queries).squeeze(-1)
