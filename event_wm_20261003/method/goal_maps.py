"""Delta-map goals for the low level (method/V2_PLAN.md Sec. 4.1): the executor is told WHAT must change, not which
entity to act on. No acted-entity label reaches execution, so attribution errors of the event layer do not train it.

Delta = entities known in both states whose position moves beyond tol_pos, whose appearance changes beyond its per-entity
threshold thr_app_id (the events' change rule), or whose covered bit flips. Three 64 x 64 channels, Gaussian blobs
(sigma = SIGMA px, the old skill's spatial maps): c0 at the current positions of Delta members ("from"), c1 at their goal
positions ("to"), c2 at the goal positions scaled by the appearance change in units of the entity's appearance unit
(clipped at 1). An empty Delta gives zero maps. numpy for the loop, torch for training (same formula)."""

from __future__ import annotations

import numpy as np

SIGMA = 2.0


def delta_mask(S, G, known_S, known_G, tol_pos, thr_app_id):
    """S, G (..., K, D) states (pos 2, app A, covered 1) -> (..., K) bool."""
    dpos = np.linalg.norm(G[..., :2] - S[..., :2], axis=-1) > tol_pos
    dapp = (np.abs(G[..., 2:-1] - S[..., 2:-1]) > np.asarray(thr_app_id)[..., :, None]).any(-1)
    dcov = (G[..., -1] > 0.5) != (S[..., -1] > 0.5)
    return known_S & known_G & (dpos | dapp | dcov)


def render(S, G, delta, app_unit, shift=(0.0, 0.0)):
    """one state pair -> maps (3, 64, 64) float32 (numpy). shift: (du, dv) px added to every position (augmentation)."""
    g = np.arange(64, dtype=np.float32)
    out = np.zeros((3, 64, 64), np.float32)
    for k in np.flatnonzero(delta):
        for c, p in ((0, S[k, :2]), (1, G[k, :2])):
            u, v = float(p[0]) + shift[0], float(p[1]) + shift[1]
            blob = np.exp(-((g[None, :] - u) ** 2 + (g[:, None] - v) ** 2) / (2 * SIGMA ** 2))
            out[c] = np.maximum(out[c], blob)
            if c == 1:
                amp = min(1.0, float(np.abs(G[k, 2:-1] - S[k, 2:-1]).max() / max(float(app_unit[k]), 1e-6)))
                out[2] = np.maximum(out[2], amp * blob)
    return out


def render_torch(S, G, delta, app_unit, shift=None):
    """batched: S, G (B, K, D), delta (B, K) bool, app_unit (K,), shift (B, 2) or None -> (B, 3, 64, 64)."""
    import torch
    g = torch.arange(64, device=S.device, dtype=torch.float32)
    sh = shift[:, None, :] if shift is not None else 0.0
    ps, pg = S[..., :2] + sh, G[..., :2] + sh                                    # (B, K, 2)

    def blobs(p):
        return torch.exp(-((g[None, None, None, :] - p[..., 0, None, None]) ** 2 + (g[None, None, :, None] - p[..., 1, None, None]) ** 2)
                         / (2 * SIGMA ** 2))                                      # (B, K, 64, 64)

    m = delta.float()[..., None, None]
    bs, bg = blobs(ps) * m, blobs(pg) * m
    amp = ((G[..., 2:-1] - S[..., 2:-1]).abs().amax(-1) / app_unit.clamp_min(1e-6)).clamp(max=1.0)[..., None, None]
    return torch.stack([bs.amax(1), bg.amax(1), (bg * amp).amax(1)], 1)
