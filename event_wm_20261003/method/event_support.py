"""Component 13 (method/README.md): generic contrastive support of (state, acted entity, intended target), learned from
play events. Port of docs/generic_state_20261007/base_source/event_support.py with the state length D as a parameter
(u_wm: D = 6). Corrupted pairs are density-estimation negatives, not verified physical failure labels. Low support
abstains from predicting progress; no task names, lock relations or action order here."""
import math

import numpy as np


def make_support(K, D, width=256):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(K * D + K + 4 * D, width), nn.GELU(),
                         nn.Linear(width, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))


def features(torch, S, e, x):
    cur = S[torch.arange(len(S), device=S.device), e]
    return torch.cat([S.flatten(1), torch.nn.functional.one_hot(e, S.shape[1]).float(),
                      x, cur, x - cur, (x - cur).abs()], dim=1)


def apply_event_support(S, predicted, scores, thresholds):
    """Abstain on low-support requests rather than create imagined progress."""
    return np.where((scores >= thresholds)[:, None, None], predicted, S).astype(np.float32)


def make_pair_support(K, D, width=256):
    """PAIRWISE support (2026-10-11): one logit per (event, other entity k) = is k's state compatible with the event? Trained
    as NCE per pair: the genuine k state of a play event against k's state taken from another event. An entity the event
    does not depend on gets logit ~0 (indistinguishable), an incompatible one (a red button for a moving drawer) a large
    negative logit. The joint model also learned context TYPICALITY, which gave legal button presses support ~.1."""
    import torch.nn as nn
    return nn.Sequential(nn.Linear(2 * K + 7 * D, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))


def pair_features(torch, S, e, x, k):
    """S (B, K, D) normalized states, e / k (B,) acted / other entity, x (B, D) normalized target -> (B, 2K + 7D). The
    event's displacement also enters magnified (half pixels, .05 colour steps): the scene drawer travels 3 px, 0.1 in
    normalized units, and a pair model on unscaled inputs let 'close the locked drawer' pass."""
    r = torch.arange(len(S), device=S.device)
    cur, sk = S[r, e], S[r, k]
    oh = torch.nn.functional.one_hot
    mag = torch.ones(S.shape[-1], device=S.device); mag[:2] = 16.0; mag[2:-1] = 10.0
    return torch.cat([oh(e, S.shape[1]).float(), oh(k, S.shape[1]).float(), x, cur, x - cur, (x - cur).abs() * mag, sk, sk - cur, sk - x], dim=1)


def pair_logits(model, torch, S, e, x):
    """(B, K) logits for every other entity; the acted entity gets +inf"""
    B, K, _ = S.shape
    kk = torch.arange(K, device=S.device).repeat(B)
    lg = model(pair_features(torch, S.repeat_interleave(K, 0), e.repeat_interleave(K), x.repeat_interleave(K, 0), kk)).view(B, K)
    return lg.masked_fill(torch.nn.functional.one_hot(e, K).bool(), float("inf"))


def pair_cost(torch, lg, known):
    """event cost in nats: per other KNOWN entity the evidence against compatibility beyond chance,
    max(0, softplus(-logit) - log 2) (an irrelevant entity, logit ~0, costs nothing); summed over entities."""
    c = (torch.nn.functional.softplus(-lg) - math.log(2)).clamp(min=0)
    c = torch.where(torch.isfinite(lg) & known, c, torch.zeros_like(c))
    return c.sum(-1)


def known_entities(S_raw):
    """an entity whose state was never read sits at (0, 0) uncovered"""
    return (np.abs(S_raw[..., :2]).sum(-1) > 0) | (S_raw[..., -1] > 0.5)
