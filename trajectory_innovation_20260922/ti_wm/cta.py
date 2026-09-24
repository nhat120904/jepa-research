"""Conditional Trajectory Abstraction (CTA): source encoder, reader, code world model and matched baselines.

RESEARCH_DESIGN.md §3-4 with the §5a amendment (the reader's distortion term is the task's ranking label), and
docs/CTA_E2E_PROTOCOL.md. Image inputs are PCA-reduced frozen DINOv2 ViT-S/14 patch tokens: the decision frame and
the segment's end frame on the 16x16 grid, the previous frame and the intermediate segment frames pooled to 8x8.

Context C = (decision frame, previous frame, agent positions now and one step before).
Future tau = (frames after steps 2, 4, 6, end frame, end agent positions).
The reader never receives actions; the world model never receives tau, the goal, or a label.
"""

import math

import torch
from torch import nn

from ti_wm.codec import FSQ, pool_grid
from ti_wm.sibling import PCA_DIM, TOKENS

LEVELS = (8, 8, 4)      # FSQ levels per code token: implicit vocabulary V = 256 (8 bits per token)
SMALL = 64              # 8x8 grid for the previous frame and the intermediate segment frames
N_SEG = 3               # intermediate frames after steps 2, 4, 6 of the executed chunk
CHUNK = 8               # executed actions per candidate
POS_SCALE, REL_SCALE = 512.0, 64.0


def vocab_size(levels=LEVELS):
    return math.prod(levels)


def proprio(cur, prev):
    """Agent positions (..., 2) now and one step before -> (..., 4), as ti_wm.progress.proprio_vector."""
    return torch.cat([cur.float(), prev.float()], dim=-1) / POS_SCALE


def action_features(chunk, agent_pos):
    """chunk (..., 8, 2) absolute agent targets in pixels; agent_pos (..., 2) -> (..., 8, 4) absolute and relative."""
    chunk, agent_pos = chunk.float(), agent_pos.float()
    return torch.cat([chunk / POS_SCALE - 0.5, (chunk - agent_pos[..., None, :]) / REL_SCALE], dim=-1)


def repeat_rows(x, n):
    """repeat_interleave(n) on dim 0, for tensors and dicts of tensors."""
    if isinstance(x, dict):
        return {k: repeat_rows(v, n) for k, v in x.items()}
    return x.repeat_interleave(n, 0)


class IndexedFSQ(FSQ):
    """FSQ with the bijection between quantized code vectors and indices in [0, V)."""

    def __init__(self, levels=LEVELS):
        super().__init__(levels)
        lv = torch.tensor(levels, dtype=torch.long)
        self.register_buffer("levels", lv)
        self.register_buffer("basis", torch.cumprod(torch.cat([torch.ones(1, dtype=torch.long), lv[:-1]]), 0))
        self.register_buffer("codebook", self.indices_to_codes(torch.arange(math.prod(levels))))

    def codes_to_indices(self, codes):
        digits = torch.round(codes.float() * self.half_w + self.half_w).long()
        return (digits * self.basis).sum(-1)

    def indices_to_codes(self, idx):
        digits = torch.div(idx[..., None], self.basis, rounding_mode="floor") % self.levels
        return (digits - self.half_w) / self.half_w


def _param(*shape):
    return nn.Parameter(torch.randn(*shape) * 0.02)


def _encoder(width, layers, heads):
    layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)


def _decoder(width, layers, heads):
    layer = nn.TransformerDecoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))


class ContextTokens(nn.Module):
    """C -> 256 + 64 + 1 tokens: decision frame, previous frame, agent positions."""

    def __init__(self, width, dim=PCA_DIM):
        super().__init__()
        self.cur, self.prev, self.prop = nn.Linear(dim, width), nn.Linear(dim, width), nn.Linear(4, width)
        self.pos, self.pos_small, self.type_embed = _param(TOKENS, width), _param(SMALL, width), _param(3, width)

    def forward(self, ctx):
        return torch.cat([self.cur(ctx["cur"].float()) + self.pos + self.type_embed[0],
                          self.prev(ctx["prev"].float()) + self.pos_small + self.type_embed[1],
                          (self.prop(ctx["prop"].float()) + self.type_embed[2])[:, None]], dim=1)


class FutureTokens(nn.Module):
    """tau -> end frame (256), end agent positions (1) and, with path=True, intermediate frames (3 x 64, timed)."""

    def __init__(self, width, path=True, dim=PCA_DIM):
        super().__init__()
        self.path = path
        self.end, self.prop = nn.Linear(dim, width), nn.Linear(4, width)
        self.pos, self.type_embed = _param(TOKENS, width), _param(3, width)
        if path:
            self.seg, self.seg_pos = nn.Linear(dim, width), _param(N_SEG, SMALL, width)

    def forward(self, fut):
        parts = [self.end(fut["end"].float()) + self.pos + self.type_embed[0],
                 (self.prop(fut["prop"].float()) + self.type_embed[1])[:, None]]
        if self.path:
            parts.append((self.seg(fut["seg"].float()) + self.seg_pos + self.type_embed[2]).flatten(1, 2))
        return torch.cat(parts, dim=1)


class SourceEncoder(nn.Module):
    """q_phi(S | C, tau), training only: M segment queries cross-attend to the actual future and to C, then FSQ."""

    def __init__(self, m=16, levels=LEVELS, width=256, layers=3, heads=8, conditional=True, path=True, dim=PCA_DIM):
        super().__init__()
        self.fut = FutureTokens(width, path, dim)
        self.ctx = ContextTokens(width, dim) if conditional else None
        self.queries = _param(m, width)
        self.decoder = _decoder(width, layers, heads)
        self.to_code = nn.Linear(width, len(levels))
        self.fsq = IndexedFSQ(levels)

    def forward(self, ctx, fut, return_pre=False):
        """Code (B, M, len(levels)); with return_pre also the pre-tanh activation, for the saturation penalty."""
        mem = [self.fut(fut)] + ([self.ctx(ctx)] if self.ctx is not None else [])
        queries = self.queries.expand(fut["end"].shape[0], -1, -1)
        z = self.to_code(self.decoder(queries, torch.cat(mem, dim=1)))
        code = self.fsq(z)
        if return_pre:
            return code, z.float() + torch.atanh(self.fsq.offset / self.fsq.half_l)
        return code


def saturation_penalty(pre, bound=1.5):
    """Squared excess of the pre-tanh FSQ activation beyond +-bound. tanh saturates beyond it and the straight-through
    gradient dies (round 0: 2 of 3 dimensions stuck at one level). The extreme levels stay reachable (|pre| >= 1.29)."""
    return torch.relu(pre.abs() - bound).pow(2).mean()


class Scorer(nn.Module):
    """Goal-conditioned score from C and one evidence channel about a candidate.

    evidence "code":   the CTA reader D(C, S, q); the code S is the only channel from the action.
             "future": tier-1 reader on the uncompressed actual future (end frame, end agent positions).
             "action": the direct scorer D_direct(C, A, q), the required strong baseline (RESEARCH_DESIGN §4).
    """

    def __init__(self, evidence, width=256, layers=4, heads=8, m=16, levels=LEVELS, dim=PCA_DIM):
        super().__init__()
        self.evidence = evidence
        self.ctx = ContextTokens(width, dim)
        if evidence == "code":
            self.ev, self.ev_pos = nn.Linear(len(levels), width), _param(m, width)
        elif evidence == "action":
            self.ev, self.ev_pos = nn.Linear(4, width), _param(CHUNK, width)
        elif evidence == "future":
            self.ev = FutureTokens(width, path=False, dim=dim)
        else:
            raise ValueError(evidence)
        self.goal, self.goal_pos = nn.Linear(dim, width), _param(TOKENS, width)
        self.type_embed, self.cls = _param(3, width), _param(1, 1, width)
        self.encoder = _encoder(width, layers, heads)
        self.head = nn.Linear(width, 1)

    def forward(self, ctx, x, goal):
        ev = self.ev(x) if self.evidence == "future" else self.ev(x.float()) + self.ev_pos
        tokens = torch.cat([self.cls.expand(goal.shape[0], -1, -1), self.ctx(ctx) + self.type_embed[0], ev + self.type_embed[1],
                            self.goal(goal.float()) + self.goal_pos + self.type_embed[2]], dim=1)
        return self.head(self.encoder(tokens)[:, 0]).squeeze(-1)


def goal_scores(scorer, ctx, x, goals):
    """Planning score of each row: mean over the goal images of scorer(ctx, x, goal). Returns (B,) float."""
    b, g = ctx["cur"].shape[0], goals.shape[0]
    out = scorer(repeat_rows(ctx, g), repeat_rows(x, g), goals.repeat(b, 1, 1))
    return out.float().view(b, g).mean(-1)


class FutureDecoder(nn.Module):
    """Distortion head: reconstruct the future tokens (end 16x16, intermediate 3 x 8x8) from (C, S) only.

    Predicts the change from the decision frame, so a code that carries nothing scores the copy baseline.
    """

    def __init__(self, m=16, levels=LEVELS, width=256, layers=2, heads=8, dim=PCA_DIM):
        super().__init__()
        self.ctx = ContextTokens(width, dim)
        self.code, self.code_pos = nn.Linear(len(levels), width), _param(m, width)
        self.queries = _param(TOKENS + N_SEG * SMALL, width)
        self.decoder = _decoder(width, layers, heads)
        self.out = nn.Linear(width, dim)

    def forward(self, ctx, code):
        mem = torch.cat([self.ctx(ctx), self.code(code.float()) + self.code_pos], dim=1)
        y = self.out(self.decoder(self.queries.expand(code.shape[0], -1, -1), mem)).float()
        cur = ctx["cur"].float()
        end = cur + y[:, :TOKENS]
        seg = pool_grid(cur, int(math.isqrt(SMALL)))[:, None] + y[:, TOKENS:].reshape(-1, N_SEG, SMALL, y.shape[-1])
        return end, seg


class CodeWM(nn.Module):
    """p_theta(S | C, A): Transformer over C and the action tokens, then autoregressive categorical decoding of the M
    code tokens. use_actions=False gives the context-only model r_omega(S | C): the rate estimate and the
    history-only diagnostic (its gap to the WM is the information the actions add about the code)."""

    def __init__(self, m=16, levels=LEVELS, width=256, enc_layers=4, dec_layers=4, heads=8, use_actions=True,
                 dim=PCA_DIM):
        super().__init__()
        self.m, self.v, self.use_actions = m, vocab_size(levels), use_actions
        self.ctx = ContextTokens(width, dim)
        if use_actions:
            self.act, self.act_pos = nn.Linear(4, width), _param(CHUNK, width)
        self.memory = _encoder(width, enc_layers, heads)
        self.tok = nn.Embedding(self.v + 1, width)          # index V is BOS
        self.pos = _param(m, width)
        self.decoder = _decoder(width, dec_layers, heads)
        self.head = nn.Linear(width, self.v)
        self.register_buffer("causal", torch.triu(torch.full((m, m), float("-inf")), 1), persistent=False)

    def encode(self, ctx, actions=None):
        parts = [self.ctx(ctx)]
        if self.use_actions:
            parts.append(self.act(actions.float()) + self.act_pos)
        return self.memory(torch.cat(parts, dim=1))

    def _run(self, mem, prefix):
        n = prefix.shape[1]
        return self.head(self.decoder(self.tok(prefix) + self.pos[:n], mem, tgt_mask=self.causal[:n, :n]))

    def logits(self, mem, idx):
        """Teacher forcing: (B, M) target indices -> (B, M, V) logits of each position given the true prefix."""
        bos = torch.full_like(idx[:, :1], self.v)
        return self._run(mem, torch.cat([bos, idx[:, :-1]], dim=1))

    @torch.no_grad()
    def decode(self, mem, sample=False):
        seq = torch.full((mem.shape[0], 1), self.v, dtype=torch.long, device=mem.device)
        for _ in range(self.m):
            logit = self._run(mem, seq)[:, -1].float()
            nxt = torch.multinomial(logit.softmax(-1), 1) if sample else logit.argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, 1:]


def sibling_contrast(wm, mem, idx, k):
    """Action InfoNCE within banks: the code of sibling j must be most likely under sibling j's own actions.

    mem: (B*K, T, W) WM memories of (C, A_k); idx: (B*K, M) source code indices. Siblings with identical codes share the
    target. Returns (loss, accuracy over banks' siblings whose code is unique in the bank).
    """
    b = mem.shape[0] // k
    t, w = mem.shape[1:]
    m = idx.shape[1]
    mem_rep = mem.view(b, 1, k, t, w).expand(b, k, k, t, w).reshape(b * k * k, t, w)     # [b, j, a] -> memory of A_a
    idx_rep = idx.view(b, k, 1, m).expand(b, k, k, m).reshape(b * k * k, m)              # [b, j, a] -> code S_j
    logp = torch.log_softmax(wm.logits(mem_rep, idx_rep).float(), -1)
    ll = logp.gather(-1, idx_rep[..., None]).squeeze(-1).sum(-1).view(b, k, k)          # log p(S_j | C, A_a)
    codes = idx.view(b, k, m)
    same = (codes[:, :, None] == codes[:, None, :]).all(-1).float()                      # [b, j, a]: S_a == S_j
    target = same / same.sum(-1, keepdim=True)
    loss = -(target * torch.log_softmax(ll, -1)).sum(-1).mean()
    unique = same.sum(-1) == 1
    hit = ll.argmax(-1) == torch.arange(k, device=idx.device)
    acc = hit[unique].float().mean() if unique.any() else ll.new_tensor(float("nan"))
    return loss, acc
