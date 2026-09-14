"""Matched learned model arms for the Stage-C screening experiment."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelDimensions:
    cameras: int = 3
    camera_feature_dim: int = 384
    proprio_dim: int = 16
    action_dim: int = 12
    latent_dim: int = 384
    context_dim: int = 256
    summary_width: int = 256
    summary_tokens: int = 4
    predictor_width: int = 384
    task_embed_dim: int = 32
    num_tasks: int = 2
    progress_dim: int = 7
    max_segment_steps: int = 8


def make_transformer(width: int, depth: int, heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=4 * width,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(width))


class ObservationFusion(nn.Module):
    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__()
        self.dims = dims
        self.visual = nn.Sequential(
            nn.LayerNorm(dims.cameras * dims.camera_feature_dim),
            nn.Linear(dims.cameras * dims.camera_feature_dim, dims.latent_dim),
            nn.GELU(),
        )
        self.proprio = nn.Sequential(
            nn.LayerNorm(dims.proprio_dim),
            nn.Linear(dims.proprio_dim, dims.latent_dim),
        )
        self.output = nn.LayerNorm(dims.latent_dim)

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if visual.ndim != 4:
            raise ValueError(f"visual must be [B,T,C,D], got {tuple(visual.shape)}")
        if visual.shape[-2:] != (self.dims.cameras, self.dims.camera_feature_dim):
            raise ValueError(f"Unexpected camera features: {tuple(visual.shape)}")
        if proprio.shape[:-1] != visual.shape[:-2] or proprio.shape[-1] != self.dims.proprio_dim:
            raise ValueError(f"Unexpected proprio shape: {tuple(proprio.shape)}")
        return self.output(self.visual(visual.flatten(-2)) + self.proprio(proprio))


class CausalContext(nn.Module):
    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__()
        self.gru = nn.GRU(dims.latent_dim, dims.context_dim, batch_first=True)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.gru(latents)[1][-1]

    def advance(self, context: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
        """Shared update for an imagined macro-step, not a claim of full-history rollout."""
        return self.gru(endpoint.unsqueeze(1), context.unsqueeze(0))[1][-1]


class ActionConditionedPredictor(nn.Module):
    def __init__(self, dims: ModelDimensions, depth: int = 6) -> None:
        super().__init__()
        self.dims = dims
        width = dims.predictor_width
        self.context = nn.Linear(dims.context_dim, width)
        self.action = nn.Linear(dims.action_dim, width)
        self.summary_queries = nn.Parameter(torch.randn(dims.summary_tokens, width) * 0.02)
        self.position = nn.Parameter(
            torch.randn(1, 1 + dims.summary_tokens + dims.max_segment_steps, width) * 0.01
        )
        self.body = make_transformer(width, depth, heads=8)
        self.endpoint = nn.Linear(width, dims.latent_dim)
        self.summary = nn.Linear(width, dims.summary_width)

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = actions.shape
        if steps > self.dims.max_segment_steps:
            raise ValueError(f"Segment {steps} exceeds {self.dims.max_segment_steps}")
        queries = self.summary_queries.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat(
            [self.context(context).unsqueeze(1), queries, self.action(actions)], dim=1
        )
        tokens = self.body(tokens + self.position[:, : tokens.shape[1]])
        summary = self.summary(tokens[:, 1 : 1 + self.dims.summary_tokens])
        endpoint = self.endpoint(tokens[:, -1])
        return endpoint, summary


class SequencePredictor(nn.Module):
    def __init__(self, dims: ModelDimensions, output_dim: int, depth: int = 6) -> None:
        super().__init__()
        width = dims.predictor_width
        self.context = nn.Linear(dims.context_dim, width)
        self.action = nn.Linear(dims.action_dim, width)
        self.position = nn.Parameter(torch.randn(1, 1 + dims.max_segment_steps, width) * 0.01)
        self.body = make_transformer(width, depth, heads=8)
        self.output = nn.Linear(width, output_dim)

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat(
            [self.context(context).unsqueeze(1), self.action(actions)], dim=1
        )
        tokens = self.body(tokens + self.position[:, : tokens.shape[1]])
        return self.output(tokens[:, 1:])


class TrajectoryTargetEncoder(nn.Module):
    """Training-only visual trajectory target; it deliberately receives no actions."""

    def __init__(self, dims: ModelDimensions, depth: int = 4) -> None:
        super().__init__()
        width = dims.summary_width
        self.input = nn.Linear(dims.latent_dim, width)
        self.queries = nn.Parameter(torch.randn(dims.summary_tokens, width) * 0.02)
        self.position = nn.Parameter(
            torch.randn(1, dims.summary_tokens + dims.max_segment_steps, width) * 0.01
        )
        self.body = make_transformer(width, depth, heads=8)
        self.decode_queries = nn.Parameter(torch.randn(1, dims.max_segment_steps, width) * 0.02)
        self.decoder = make_transformer(width, depth=2, heads=8)
        self.reconstruct = nn.Linear(width, dims.cameras * dims.camera_feature_dim + dims.proprio_dim)
        self.summary_tokens = dims.summary_tokens

    def forward(self, future_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.queries.unsqueeze(0).expand(future_latents.shape[0], -1, -1)
        tokens = torch.cat([queries, self.input(future_latents)], dim=1)
        tokens = self.body(tokens + self.position[:, : tokens.shape[1]])
        summary = tokens[:, : self.summary_tokens]
        queries = self.decode_queries[:, :future_latents.shape[1]].expand(future_latents.shape[0], -1, -1)
        decoded = self.decoder(torch.cat([summary, queries], dim=1))[:, self.summary_tokens:]
        reconstruction = self.reconstruct(decoded)
        return summary, reconstruction


class Composer(nn.Module):
    def __init__(self, dims: ModelDimensions, depth: int = 2) -> None:
        super().__init__()
        width = dims.summary_width
        self.queries = nn.Parameter(torch.randn(dims.summary_tokens, width) * 0.02)
        self.boundary = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.position = nn.Parameter(torch.randn(1, 3 * dims.summary_tokens + 1, width) * 0.02)
        self.body = make_transformer(width, depth, heads=8)
        self.tokens = dims.summary_tokens

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(left.shape[0], -1, -1)
        boundary = self.boundary.expand(left.shape[0], -1, -1)
        output = self.body(torch.cat([queries, left, boundary, right], dim=1) + self.position)
        return output[:, : self.tokens]


class ValueHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class StageCArm(nn.Module):
    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__()
        self.dims = dims
        self.fusion = ObservationFusion(dims)
        self.target_fusion = copy.deepcopy(self.fusion)
        self.target_fusion.requires_grad_(False)
        self.memory = CausalContext(dims)
        self.task_embedding = nn.Embedding(dims.num_tasks, dims.task_embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_fusion.eval()
        if hasattr(self, "target_encoder"):
            self.target_encoder.eval()
        return self

    def context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.fusion(batch["visual_history"], batch["proprio_history"])
        return self.memory(history), history

    def future(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.target_fusion(batch["visual_future"], batch["proprio_future"])

    def task(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.task_embedding(batch["task_id"].long())

    @torch.no_grad()
    def update_ema(self, decay: float = 0.996) -> None:
        for target, online in zip(self.target_fusion.parameters(), self.fusion.parameters()):
            target.mul_(decay).add_(online, alpha=1.0 - decay)


class FrameRolloutArm(StageCArm):
    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__(dims)
        self.predictor = SequencePredictor(dims, dims.latent_dim)
        self.sequence_readout = nn.GRU(dims.latent_dim, dims.latent_dim, batch_first=True)
        self.value = ValueHead(
            dims.context_dim + 2 * dims.latent_dim + dims.task_embed_dim
        )

    def loss(self, batch, weights):
        context, _ = self.context(batch)
        target = self.future(batch).detach()
        predicted = self.predictor(context, batch["actions"])
        logits = self.value(
            torch.cat([context, predicted[:, -1], self.sequence_readout(predicted)[1][-1], self.task(batch)], dim=-1)
        )
        losses = {
            "trajectory": F.mse_loss(predicted, target),
            "value": F.binary_cross_entropy_with_logits(logits, batch["eventual_success"].float()),
        }
        total = weights["frame"] * losses["trajectory"] + weights["value"] * losses["value"]
        return total, losses, logits


class EndpointOnlyArm(StageCArm):
    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__(dims)
        self.predictor = SequencePredictor(dims, dims.latent_dim)
        self.value = ValueHead(dims.context_dim + dims.latent_dim + dims.task_embed_dim)

    def loss(self, batch, weights):
        context, _ = self.context(batch)
        target = self.future(batch).detach()[:, -1]
        endpoint = self.predictor(context, batch["actions"])[:, -1]
        logits = self.value(torch.cat([context, endpoint, self.task(batch)], dim=-1))
        losses = {
            "endpoint": F.mse_loss(endpoint, target),
            "value": F.binary_cross_entropy_with_logits(logits, batch["eventual_success"].float()),
        }
        total = weights["endpoint"] * losses["endpoint"] + weights["value"] * losses["value"]
        return total, losses, logits


class SimpleProgressArm(StageCArm):
    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__(dims)
        self.predictor = SequencePredictor(dims, dims.progress_dim)
        self.value = ValueHead(
            dims.context_dim + 2 * dims.progress_dim + dims.task_embed_dim
        )

    def loss(self, batch, weights):
        if not bool(batch["has_progress"].all()):
            raise ValueError("simple_progress requires progress labels for every batch item")
        context, _ = self.context(batch)
        predicted = self.predictor(context, batch["actions"])
        target = batch["progress_future"].float()
        logits = self.value(
            torch.cat([context, predicted[:, -1], predicted.mean(dim=1), self.task(batch)], dim=-1)
        )
        losses = {
            "progress": F.mse_loss(predicted, target),
            "value": F.binary_cross_entropy_with_logits(logits, batch["eventual_success"].float()),
        }
        total = weights["progress"] * losses["progress"] + weights["value"] * losses["value"]
        return total, losses, logits


class SegmentTargetArm(StageCArm):
    def __init__(self, dims: ModelDimensions, composition_weight: float) -> None:
        super().__init__(dims)
        self.composition_weight = float(composition_weight)
        self.predictor = ActionConditionedPredictor(dims)
        self.online_target = TrajectoryTargetEncoder(dims)
        self.target_encoder = copy.deepcopy(self.online_target)
        self.target_encoder.requires_grad_(False)
        self.composer = Composer(dims)
        self.value = ValueHead(
            dims.context_dim + dims.latent_dim + dims.summary_width + dims.task_embed_dim
        )

    @torch.no_grad()
    def update_ema(self, decay: float = 0.996) -> None:
        super().update_ema(decay)
        for target, online in zip(self.target_encoder.parameters(), self.online_target.parameters()):
            target.mul_(decay).add_(online, alpha=1.0 - decay)

    def loss(self, batch, weights):
        context, _ = self.context(batch)
        future = self.future(batch).detach()
        actions = batch["actions"]
        endpoint, summary = self.predictor(context, actions)
        with torch.no_grad():
            target_summary, _ = self.target_encoder(future)
        online_full, reconstruction = self.online_target(future)
        raw_future = torch.cat([batch["visual_future"].flatten(-2), batch["proprio_future"]], dim=-1).detach()
        logits = self.value(
            torch.cat([context, endpoint, summary.mean(dim=1), self.task(batch)], dim=-1)
        )
        losses = {
            "endpoint": F.mse_loss(endpoint, future[:, -1]),
            "segment": F.mse_loss(summary, target_summary),
            "target_anchor": F.mse_loss(reconstruction, raw_future),
            "value": F.binary_cross_entropy_with_logits(logits, batch["eventual_success"].float()),
        }

        midpoint = actions.shape[1] // 2
        if midpoint < 1:
            raise ValueError("Segment training requires at least two action steps")
        first_endpoint, first_summary = self.predictor(context, actions[:, :midpoint])
        second_context = self.memory.advance(context, first_endpoint)
        second_endpoint, second_summary = self.predictor(second_context, actions[:, midpoint:])
        predicted_composed = self.composer(first_summary, second_summary)
        with torch.no_grad():
            target_left, _ = self.target_encoder(future[:, :midpoint])
            target_right, _ = self.target_encoder(future[:, midpoint:])
        online_left, rec_left = self.online_target(future[:, :midpoint])
        online_right, rec_right = self.online_target(future[:, midpoint:])
        target_composed = self.composer(online_left, online_right)
        losses.update(
            {
                "observed_composition": 0.5 * (F.mse_loss(target_composed, online_full.detach())
                                              + F.mse_loss(online_full, target_composed.detach())),
                "predicted_composition": F.mse_loss(predicted_composed, summary.detach()),
                "partition_endpoint": F.mse_loss(second_endpoint, endpoint.detach()),
                "first_endpoint_anchor": F.mse_loss(first_endpoint, future[:, midpoint - 1]),
                "second_endpoint_anchor": F.mse_loss(second_endpoint, future[:, -1]),
                "subsegment_anchor": 0.5 * (F.mse_loss(first_summary, target_left)
                                           + F.mse_loss(second_summary, target_right)),
                "subsegment_reconstruction": 0.5 * (F.mse_loss(rec_left, raw_future[:, :midpoint])
                                                    + F.mse_loss(rec_right, raw_future[:, midpoint:])),
                "target_variance": torch.relu(0.1 - torch.sqrt(online_full.var(dim=0, unbiased=False) + 1e-4)).mean(),
            }
        )
        composition = losses["observed_composition"] + losses["predicted_composition"]
        total = (
            weights["endpoint"] * losses["endpoint"]
            + weights["segment"] * losses["segment"]
            + weights["reconstruction"] * losses["target_anchor"]
            + weights["value"] * losses["value"]
            + weights["endpoint"] * 0.5 * (losses["first_endpoint_anchor"] + losses["second_endpoint_anchor"])
            + weights["segment"] * losses["subsegment_anchor"]
            + weights["reconstruction"] * (losses["subsegment_reconstruction"] + 0.01 * losses["target_variance"])
            + self.composition_weight * weights["composition"] * composition
            + self.composition_weight * weights["split_consistency"] * losses["partition_endpoint"]
        )
        return total, losses, logits

    def predict_partition(self, batch, split: int):
        """Explicit composed inference for held-out partition evaluation (no future inputs)."""
        context, _ = self.context(batch)
        actions = batch["actions"]
        if not 0 < split < actions.shape[1]:
            raise ValueError("Partition must have two nonempty segments")
        first_endpoint, left = self.predictor(context, actions[:, :split])
        second_context = self.memory.advance(context, first_endpoint)
        endpoint, right = self.predictor(second_context, actions[:, split:])
        summary = self.composer(left, right)
        logits = self.value(torch.cat([context, endpoint, summary.mean(dim=1), self.task(batch)], dim=-1))
        return endpoint, summary, logits


class DirectValueArm(StageCArm):
    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__(dims)
        self.action = nn.Sequential(
            nn.LayerNorm(dims.action_dim),
            nn.Linear(dims.action_dim, dims.predictor_width),
            nn.GELU(),
        )
        self.action_body = make_transformer(dims.predictor_width, depth=6, heads=8)
        self.action_position = nn.Parameter(torch.randn(1, dims.max_segment_steps, dims.predictor_width) * 0.02)
        self.value = ValueHead(
            dims.context_dim + dims.predictor_width + dims.task_embed_dim
        )

    def loss(self, batch, weights):
        context, _ = self.context(batch)
        steps = batch["actions"].shape[1]
        action_features = self.action_body(self.action(batch["actions"]) + self.action_position[:, :steps]).mean(dim=1)
        logits = self.value(torch.cat([context, action_features, self.task(batch)], dim=-1))
        losses = {
            "value": F.binary_cross_entropy_with_logits(logits, batch["eventual_success"].float())
        }
        return weights["value"] * losses["value"], losses, logits


def build_arm(name: str, dims: ModelDimensions) -> StageCArm:
    arms = {
        "endpoint_only": lambda: EndpointOnlyArm(dims),
        "frame_rollout": lambda: FrameRolloutArm(dims),
        "simple_progress": lambda: SimpleProgressArm(dims),
        "unstructured_segment": lambda: SegmentTargetArm(dims, composition_weight=0.0),
        "compositional_segment": lambda: SegmentTargetArm(dims, composition_weight=1.0),
        "direct_value": lambda: DirectValueArm(dims),
    }
    if name not in arms:
        raise ValueError(f"Unknown Stage-C arm {name}; choose from {sorted(arms)}")
    return arms[name]()
