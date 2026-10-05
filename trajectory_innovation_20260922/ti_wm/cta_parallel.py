"""Parallel coordinate-FSQ and endpoint predictors; fixed-source Round 3."""
import torch
from torch import nn
from torch.nn import functional as F

from ti_wm.cta import CHUNK, LEVELS, ContextTokens, _decoder, _encoder, _param
from ti_wm.sibling import PCA_DIM, TOKENS


class ParallelBackbone(nn.Module):
    def __init__(self, queries, width=256, layers=4, heads=8, dim=PCA_DIM, dropout=0.0, chunk=CHUNK):
        super().__init__()
        self.ctx = ContextTokens(width, dim)
        self.act, self.act_pos = nn.Linear(4, width), _param(chunk, width)
        self.memory = _encoder(width, layers, heads, dropout)
        self.queries = _param(queries, width)
        self.decoder = _decoder(width, layers, heads, dropout)

    def hidden(self, ctx, actions):
        memory = self.memory(torch.cat([self.ctx(ctx), self.act(actions.float()) + self.act_pos], dim=1))
        return self.decoder(self.queries.expand(actions.shape[0], -1, -1), memory)

    def warm_start(self, old):
        own = self.state_dict()
        copied = {key: value for key, value in old.items()
                  if key in own and key.startswith(('ctx.', 'act.', 'memory.', 'decoder.'))}
        copied['act_pos'] = old['act_pos']
        if old['pos'].shape == self.queries.shape:
            copied['queries'] = old['pos']
        self.load_state_dict(copied, strict=False)
        return sorted(copied)


class ParallelFSQWM(ParallelBackbone):
    """Jointly computed token features; factorized categorical coordinates, no greedy prefix."""
    def __init__(self, m=16, levels=LEVELS, **kwargs):
        super().__init__(m, **kwargs)
        self.levels = tuple(levels)
        width = self.queries.shape[-1]
        self.head = nn.Linear(width, sum(levels))
        for j, count in enumerate(levels):
            self.register_buffer(f'values_{j}', (torch.arange(count).float() - count // 2) / (count // 2))

    def forward(self, ctx, actions):
        logits = self.head(self.hidden(ctx, actions)).float().split(self.levels, dim=-1)
        expected = torch.stack([(p.softmax(-1) * getattr(self, f'values_{j}')).sum(-1)
                                for j, p in enumerate(logits)], dim=-1)
        return expected, logits

    def nll(self, logits, source):
        losses = []
        for j, (p, count) in enumerate(zip(logits, self.levels)):
            digits = (source[..., j].float() * (count // 2) + count // 2).round().long()
            if not ((digits >= 0) & (digits < count)).all():
                raise ValueError('Source lies outside the FSQ grid')
            losses.append(F.cross_entropy(p.flatten(0, 1), digits.flatten()))
        # Optimization uses mean coordinate NLL; sum is reported as nats/token.
        return torch.stack(losses).mean()


class EndpointWM(ParallelBackbone):
    """Predict PCA frame tokens AND future proprio. No future input at inference."""
    def __init__(self, dim=PCA_DIM, **kwargs):
        super().__init__(TOKENS + 1, dim=dim, **kwargs)
        width = self.queries.shape[-1]
        self.image = nn.Linear(width, dim)
        self.proprio = nn.Linear(width, 4)
        nn.init.zeros_(self.image.weight)
        nn.init.zeros_(self.image.bias)
        nn.init.zeros_(self.proprio.weight)
        nn.init.zeros_(self.proprio.bias)

    def forward(self, ctx, actions):
        h = self.hidden(ctx, actions)
        return {'end': ctx['cur'].float() + self.image(h[:, :TOKENS]).float(),
                'prop': ctx['prop'].float() + self.proprio(h[:, TOKENS]).float()}


def weighted_rank(scores, labels, scale=.01, margin=1e-3, pair_count=None):
    """Weight physically small advantages softly; do not normalize each bank separately."""
    delta = labels[:, :, None] - labels[:, None, :]
    mask = delta > margin
    weight = (delta / scale).clamp(0, 1) * mask
    diff = scores[:, :, None] - scores[:, None, :]
    denominator = mask.sum() if pair_count is None else pair_count
    return (weight * F.softplus(-diff)).sum() / denominator.clamp_min(1)


def score_consistency(predicted, teacher, scale):
    """Match within-bank score differences, invariant to an arbitrary bank offset."""
    a = predicted - predicted.mean(-1, keepdim=True)
    b = teacher.detach() - teacher.detach().mean(-1, keepdim=True)
    return F.smooth_l1_loss(a / scale, b / scale)


def normalized_score(max_coverage, threshold=.95):
    """Maximum native reward, given coverage accumulated ONLY AFTER env.step."""
    return max(0., min(float(max_coverage) / float(threshold), 1.))
