"""Self-supervised goal-conditioned progress reader for gate C2 (docs/GATE_C2_PROGRESS_READER_PROTOCOL.md).

One reader for every goal: the goal image is an input. Trained by hindsight relabeling only:
target = log(1 + steps from a state to a later frame of the same episode).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GRID, POOLED, FEAT = 16, 8, 384
TOKENS = POOLED * POOLED


def pool_tokens(patch):
    """(N, 256, 384) DINOv2 patch tokens -> (N, 64, 384) by 2x2 average pooling of the grid."""
    n = patch.shape[0]
    grid = patch.reshape(n, GRID, GRID, FEAT).permute(0, 3, 1, 2)
    return F.avg_pool2d(grid, 2).flatten(2).transpose(1, 2).contiguous()


def proprio_vector(cur_pos, prev_pos):
    return np.concatenate([np.asarray(cur_pos), np.asarray(prev_pos)], axis=-1).astype(np.float32) / 512.0


class ProgressReader(nn.Module):
    def __init__(self, width=256, layers=4, heads=4):
        super().__init__()
        self.proj = nn.Linear(FEAT, width)
        self.proprio = nn.Linear(4, width)
        self.type_embed = nn.Parameter(torch.randn(4, width) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(TOKENS, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, cur, prev, goal, proprio):
        """cur/prev/goal: (B, 64, 384); proprio: (B, 4). Returns predicted log(1 + steps to goal)."""
        tokens = [self.proj(x.float()) + self.pos_embed + self.type_embed[i] for i, x in enumerate((cur, prev, goal))]
        p = (self.proprio(proprio.float()) + self.type_embed[3])[:, None]
        x = torch.cat([self.cls.expand(cur.shape[0], -1, -1), *tokens, p], dim=1)
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


def episode_bounds(episode_ids):
    """For a flat array of per-state episode ids (contiguous), the end index (exclusive) of each state's episode."""
    ids = np.asarray(episode_ids)
    change = np.flatnonzero(np.diff(ids)) + 1
    ends = np.append(change, len(ids))
    starts = np.insert(change, 0, 0)
    end_of = np.empty(len(ids), dtype=np.int64)
    for s, e in zip(starts, ends):
        end_of[s:e] = e
    return end_of


def sample_pairs(times, end_of, rng, n, candidates=None):
    """Hindsight pairs: state i, goal j uniform in [i, end of i's episode); target log1p(t_j - t_i)."""
    pool = np.arange(len(times)) if candidates is None else np.asarray(candidates)
    i = pool[rng.integers(0, len(pool), size=n)]
    j = i + np.floor(rng.random(n) * (end_of[i] - i)).astype(np.int64)
    return i, j, np.log1p(times[j] - times[i]).astype(np.float32)


class ProgressScorer:
    """Score = -mean over goal images of the predicted log-distance, from a candidate's ACTUAL segment."""

    def __init__(self, reader, visual, goal_frames):
        self.reader = reader.eval()
        self.visual = visual
        self.device = visual.device
        self.goals = pool_tokens(visual.features(goal_frames)).half()

    @torch.inference_mode()
    def scores(self, branches):
        cur = np.stack([b.hist[-1]["pixels"] for b in branches])
        prev = np.stack([b.hist[-2]["pixels"] for b in branches])
        fc = pool_tokens(self.visual.features(cur))
        fp = pool_tokens(self.visual.features(prev))
        prop = torch.from_numpy(np.stack([proprio_vector(b.hist[-1]["agent_pos"], b.hist[-2]["agent_pos"])
                                          for b in branches])).to(self.device)
        k, g = len(branches), self.goals.shape[0]
        pred = self.reader(fc.repeat_interleave(g, 0), fp.repeat_interleave(g, 0),
                           self.goals.repeat(k, 1, 1), prop.repeat_interleave(g, 0))
        return (-pred.view(k, g).mean(dim=1)).float().cpu().tolist()
