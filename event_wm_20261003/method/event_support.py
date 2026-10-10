"""Component 13 (method/README.md): generic contrastive support of (state, acted entity, intended target), learned from
play events. Port of docs/generic_state_20261007/base_source/event_support.py with the state length D as a parameter
(u_wm: D = 6). Corrupted pairs are density-estimation negatives, not verified physical failure labels. Low support
abstains from predicting progress; no task names, lock relations or action order here."""
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
