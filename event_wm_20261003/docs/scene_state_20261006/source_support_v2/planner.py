"""Event world model, cost-to-go and batch weighted A* over event codes."""

from __future__ import annotations

import heapq
import itertools

import numpy as np


def make_event_wm(bits: int, events: int, hidden: int = 512):
    import torch
    import torch.nn as nn

    class EventWM(nn.Module):
        """(code b, event e) -> logits of the code after the event."""

        def __init__(self):
            super().__init__()
            self.events = events
            self.net = nn.Sequential(nn.Linear(bits + events, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, bits))

        def forward(self, b, e):
            return self.net(torch.cat([b.float(), nn.functional.one_hot(e.long(), self.events).float()], -1))

        @torch.no_grad()
        def successors(self, b):
            """b (N, K) {0,1} -> (N, E, K) rounded codes after every event."""
            N, K = b.shape
            bb = b.unsqueeze(1).expand(N, self.events, K).reshape(-1, K)
            ee = torch.arange(self.events, device=b.device).repeat(N)
            return (self.forward(bb, ee) > 0).to(torch.uint8).view(N, self.events, K)

    return EventWM()


def make_costtogo(bits: int, hidden: int = 512, xor: bool = False):
    import torch
    import torch.nn as nn

    class CostToGo(nn.Module):
        """h(b, g) >= 0: estimated number of events from code b to goal code g.
        xor=True adds bitwise goal-difference features b xor g (domain-agnostic for binary codes)."""

        def __init__(self):
            super().__init__()
            inp = (3 if xor else 2) * bits
            self.net = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

        def forward(self, b, g):
            b, g = b.float(), g.float()
            z = torch.cat([b, g, (b - g).abs()], -1) if xor else torch.cat([b, g], -1)
            return nn.functional.softplus(self.net(z).float()).squeeze(-1)

    return CostToGo()


def keys_of(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.int64) << np.arange(b.shape[1], dtype=np.int64)).sum(1)


def bwas(start, goal, wm, h, device, lam=0.6, batch=128, max_expansions=200_000):
    """Batch weighted A* (DeepCubeA): f = lam * g + h; pop `batch` nodes per iteration.

    start, goal: (K,) uint8 codes. Returns (event list or None, info dict).
    """
    import torch

    start, goal = np.asarray(start, np.uint8), np.asarray(goal, np.uint8)
    gkey = int(keys_of(goal[None])[0])
    skey = int(keys_of(start[None])[0])
    gt = torch.as_tensor(goal, device=device)[None]
    nodes = {skey: (0, None, None, start)}           # key -> (path cost, parent key, event, code)
    tie = itertools.count()
    with torch.no_grad():
        h0 = float(h(torch.as_tensor(start, device=device)[None], gt)[0])
    open_ = [(h0, next(tie), skey)]
    closed, expanded = set(), 0
    while open_ and expanded < max_expansions:
        pop = []
        while open_ and len(pop) < batch:
            _, _, k = heapq.heappop(open_)
            if k in closed:
                continue
            closed.add(k)
            if k == gkey:
                return _path(nodes, k), {"expanded": expanded, "found": True, "h0": h0}
            pop.append(k)
        if not pop:
            break
        expanded += len(pop)
        codes = torch.as_tensor(np.stack([nodes[k][3] for k in pop]), device=device)
        ch = wm.successors(codes)                                       # (P, E, K)
        P, E, K = ch.shape
        flat = ch.view(-1, K)
        with torch.no_grad():
            hv = h(flat, gt.expand(len(flat), K)).float().cpu().numpy()
        flat_np = flat.cpu().numpy()
        ck = keys_of(flat_np)
        for i in range(P * E):
            pk = pop[i // E]
            g_new = nodes[pk][0] + 1
            k = int(ck[i])
            if k in closed:
                continue
            if k not in nodes or g_new < nodes[k][0]:
                nodes[k] = (g_new, pk, i % E, flat_np[i])
                heapq.heappush(open_, (lam * g_new + (0.0 if k == gkey else float(hv[i])), next(tie), k))
    return None, {"expanded": expanded, "found": False, "h0": h0}


def _path(nodes, k):
    events = []
    while nodes[k][1] is not None:
        events.append(int(nodes[k][2]))
        k = nodes[k][1]
    return events[::-1]
