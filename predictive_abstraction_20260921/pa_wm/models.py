"""Initial learned-summary architecture; not a trained/reproduced WM baseline.

All inputs are cached spatial observation features or proposed actions. No future
observation argument exists in the forecasting interface. Query vectors are constructed
separately from observation-derived anchors/operator/time parameters.
"""
import math

import torch
from torch import nn


def positions(length, width, device, dtype):
    if width % 2:
        raise ValueError("Sinusoidal positions require even width")
    t = torch.arange(length, device=device, dtype=dtype)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                      * (-math.log(10000.0) / width))
    return torch.stack([(t * scale).sin(), (t * scale).cos()], dim=-1).flatten(1)


class TokenPool(nn.Module):
    def __init__(self, width, tokens, heads=4):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, sequence):
        q = self.tokens[None].expand(sequence.shape[0], -1, -1)
        output, _ = self.attn(q, sequence, sequence, need_weights=False)
        return self.norm(q + output)


def temporal_encoder(width, layers=2):
    block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.0,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(block, layers, enable_nested_tensor=False)


class ObservedCodec(nn.Module):
    def __init__(self, feature_dim, width=128, tokens=4):
        super().__init__()
        self.project = nn.Linear(feature_dim, width)
        self.encoder = temporal_encoder(width)
        self.pool = TokenPool(width, tokens)

    def forward(self, future_features):
        x = self.project(future_features)
        x = x + positions(x.shape[1], x.shape[2], x.device, x.dtype)
        return self.pool(self.encoder(x))


class ActionSummaryPredictor(nn.Module):
    def __init__(self, feature_dim, action_dim=2, width=128, tokens=4):
        super().__init__()
        self.obs = nn.Linear(feature_dim, width)
        self.action = nn.Linear(action_dim, width)
        self.modality = nn.Parameter(torch.randn(2, width) * 0.02)
        self.encoder = temporal_encoder(width)
        self.pool = TokenPool(width, tokens)

    def forward(self, history_features, proposed_actions):
        # All proposed actions are known now: noncausal attention over them is valid.
        # No realized future observations, goals or queries enter this summary.
        h = self.obs(history_features) + self.modality[0]
        a = self.action(proposed_actions) + self.modality[1]
        x = torch.cat([h, a], dim=1)
        x = x + positions(x.shape[1], x.shape[2], x.device, x.dtype)
        return self.pool(self.encoder(x))


class QueryReader(nn.Module):
    def __init__(self, query_dim, width=128):
        super().__init__()
        self.query = nn.Linear(query_dim, width)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                     nn.GELU(), nn.Linear(width, 1), nn.Sigmoid())

    def forward(self, summary, queries):
        # queries [batch, number_of_queries, query_dim]; no query-specific re-forecast.
        q = self.query(queries)
        x, _ = self.attn(q, summary, summary, need_weights=False)
        return self.readout(q + x).squeeze(-1)


class SequenceProjector(nn.Module):
    """Project per-frame pooled patch grids without discarding spatial arrangement first."""
    def __init__(self, patch_count, patch_dim, width):
        super().__init__()
        self.patch_count, self.patch_dim = patch_count, patch_dim
        self.patch = nn.Linear(patch_dim, width)
        self.spatial = nn.Parameter(torch.randn(patch_count, width) * 0.02)
        self.merge = nn.Linear(patch_count * width, width)

    def forward(self, features):
        if features.shape[-2:] != (self.patch_count, self.patch_dim):
            raise ValueError(f"Expected [...,{self.patch_count},{self.patch_dim}]")
        x = self.patch(features) + self.spatial
        return self.merge(x.flatten(-2))


class SpatialObservedCodec(nn.Module):
    def __init__(self, patch_count=16, patch_dim=1024, width=128, tokens=4):
        super().__init__()
        self.project = SequenceProjector(patch_count, patch_dim, width)
        self.encoder = temporal_encoder(width)
        self.pool = TokenPool(width, tokens)

    def forward(self, future_features):
        x = self.project(future_features)
        x = x + positions(x.shape[1], x.shape[2], x.device, x.dtype)
        return self.pool(self.encoder(x))


class SpatialActionSummary(nn.Module):
    def __init__(self, patch_count=16, patch_dim=1024, action_dim=2, width=128, tokens=4):
        super().__init__()
        self.obs = SequenceProjector(patch_count, patch_dim, width)
        self.action = nn.Linear(action_dim, width)
        self.modality = nn.Parameter(torch.randn(2, width) * 0.02)
        self.encoder = temporal_encoder(width)
        self.pool = TokenPool(width, tokens)

    def context(self, history, actions):
        h = self.obs(history) + self.modality[0]
        a = self.action(actions) + self.modality[1]
        x = torch.cat([h, a], dim=1)
        x = x + positions(x.shape[1], x.shape[2], x.device, x.dtype)
        return self.encoder(x)

    def forward(self, history, actions):
        return self.pool(self.context(history, actions))


class DirectQueryPredictor(nn.Module):
    """Strong scalar baseline: query attends directly to history+all proposed actions."""
    def __init__(self, query_dim, patch_count=16, patch_dim=1024, width=128):
        super().__init__()
        self.context_model = SpatialActionSummary(patch_count, patch_dim, width=width)
        self.reader = QueryReader(query_dim, width)

    def forward(self, history, actions, queries):
        return self.reader(self.context_model.context(history, actions), queries)


class FrameSequencePredictor(nn.Module):
    """Ordered frame-token WM with full horizon tokens and answer auxiliary loss."""
    def __init__(self, patch_count=16, patch_dim=1024, width=128):
        super().__init__()
        self.context_model = SpatialActionSummary(patch_count, patch_dim, width=width)
        self.future_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, history, actions):
        context = self.context_model.context(history, actions)
        horizon = actions.shape[1]
        q = positions(horizon, context.shape[-1], context.device, context.dtype)
        q = q[None].expand(context.shape[0], -1, -1)
        output, _ = self.future_attn(q, context, context, need_weights=False)
        return self.norm(q + output)


class EndpointPredictor(FrameSequencePredictor):
    def forward(self, history, actions):
        return super().forward(history, actions)[:, -1:]


class GenericSummaryCodec(nn.Module):
    """Matched token bottleneck trained to reconstruct ordered global future features."""
    def __init__(self, patch_count=16, patch_dim=1024, width=128, tokens=4):
        super().__init__()
        self.encoder = SpatialObservedCodec(patch_count, patch_dim, width, tokens)
        self.decode_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, patch_dim))

    def forward(self, future):
        return self.encoder(future)

    def reconstruct(self, summary, horizon):
        q = positions(horizon, summary.shape[-1], summary.device, summary.dtype)
        q = q[None].expand(summary.shape[0], -1, -1)
        x, _ = self.decode_attn(q, summary, summary, need_weights=False)
        return self.out(q + x)


def forecast_loss(predictor, frozen_codec, frozen_reader, history, actions, future,
                  queries, answers, latent_weight=0.0):
    """Answer loss differentiates THROUGH frozen reader to predicted summary.

    Freezing teacher parameters is the caller's responsibility. Do not wrap the
    reader(prediction, query) call in no_grad: that would silently break learning.
    """
    predicted = predictor(history, actions)
    predicted_answers = frozen_reader(predicted, queries)
    loss = torch.nn.functional.mse_loss(predicted_answers, answers)
    if latent_weight:
        with torch.no_grad():
            target = frozen_codec(future)
        loss = loss + latent_weight * torch.nn.functional.mse_loss(predicted, target)
    return loss
