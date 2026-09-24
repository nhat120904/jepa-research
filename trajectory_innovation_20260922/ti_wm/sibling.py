"""Gate S1 sibling readers (docs/SIBLING_READER_OFFLINE_PROTOCOL.md).

A reader scores one candidate's end state against a goal. It sees the full 16x16 DINOv2 grid
(PCA-reduced channels), the change from the decision-time state, the goal, and agent positions.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = 256
PCA_DIM = 128
EQUAL_STATE = 1e-3
RANK_MARGIN = 1e-3


def fit_pca(tokens, dim=PCA_DIM):
    """tokens: (N, D) float. Returns (mean, basis (D, dim)) from the top eigenvectors of the covariance."""
    x = tokens.double()
    mean = x.mean(0)
    cov = (x - mean).T @ (x - mean) / (len(x) - 1)
    _, vecs = torch.linalg.eigh(cov)
    return mean.float(), vecs[:, -dim:].flip(1).float().contiguous()


def project(tokens, mean, basis):
    return (tokens.float() - mean) @ basis


class SiblingReader(nn.Module):
    def __init__(self, dim=PCA_DIM, width=256, layers=4, heads=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, width) for _ in range(3)])  # end, end - ctx, goal
        self.type_embed = nn.Parameter(torch.randn(4, width) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(TOKENS, width) * 0.02)
        self.proprio = nn.Linear(4, width)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, end, ctx, goal, proprio):
        """end/ctx/goal: (B, 256, dim); proprio: (B, 4). Returns one scalar score per row."""
        parts = (end.float(), end.float() - ctx.float(), goal.float())
        tokens = [p(x) + self.pos_embed + self.type_embed[i] for i, (p, x) in enumerate(zip(self.proj, parts))]
        prop = (self.proprio(proprio.float()) + self.type_embed[3])[:, None]
        x = torch.cat([self.cls.expand(end.shape[0], -1, -1), *tokens, prop], dim=1)
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


def contrast_targets(phys, k):
    """phys: (B, K, P) sibling end states; k: (B,) chosen sibling. Siblings equal to k share the target."""
    same = (phys - phys[torch.arange(len(k)), k][:, None]).abs().amax(-1) < EQUAL_STATE
    return same.float() / same.float().sum(-1, keepdim=True)


def contrast_loss(scores, targets):
    return -(targets * F.log_softmax(scores, dim=-1)).sum(-1).mean()


def rank_loss(scores, labels, margin=RANK_MARGIN):
    """Pairwise logistic loss over sibling pairs whose labels differ by more than margin."""
    better = (labels[:, :, None] - labels[:, None, :]) > margin
    if not better.any():
        return scores.sum() * 0.0
    diff = scores[:, :, None] - scores[:, None, :]
    return F.softplus(-diff)[better].mean()


def retained_gap(scores, labels, select):
    """Per-decision numerator and denominator of the offline retained gap (candidate 0 is the default)."""
    chosen = select(list(scores))
    return float(labels[chosen] - labels[0]), float(np.max(labels) - labels[0])


class SiblingScorer:
    """Test-time scorer for a trained SiblingReader, identical to the S1 held-out evaluation.

    score(k) = mean over goal images of reader(end_k, ctx, goal, proprio_k), with ctx the decision-time frame.
    """

    def __init__(self, checkpoint, visual, goal_frames):
        blob = torch.load(checkpoint, map_location=visual.device)
        self.model = SiblingReader().to(visual.device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.visual = visual
        self.device = visual.device
        self.mean, self.basis = blob["pca_mean"].to(self.device), blob["pca_basis"].to(self.device)
        self.goals = self._pca(goal_frames)

    @torch.inference_mode()
    def _pca(self, frames):
        return project(self.visual.features(np.ascontiguousarray(frames)), self.mean, self.basis).half()

    @torch.inference_mode()
    def scores(self, state, branches):
        from ti_wm.progress import proprio_vector

        k, g = len(branches), len(self.goals)
        end = self._pca(np.stack([b.hist[-1]["pixels"] for b in branches]))
        ctx = self._pca(state.hist[-1]["pixels"][None]).expand(k, -1, -1)
        prop = torch.from_numpy(np.stack([proprio_vector(b.hist[-1]["agent_pos"], b.hist[-2]["agent_pos"])
                                          for b in branches])).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.model(end.repeat_interleave(g, 0), ctx.repeat_interleave(g, 0),
                             self.goals.repeat(k, 1, 1), prop.repeat_interleave(g, 0))
        return out.float().view(k, g).mean(-1).cpu().tolist()
