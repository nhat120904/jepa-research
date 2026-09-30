"""World-model policies that change only the planning start state.

``PrefillPolicy`` seeds the released ``WorldModelPolicy`` history buffer with
the dataset frames and actions that preceded each evaluation start, so the
first plan sees the same context as later plans (the released buffer starts
empty, so the first plan of every episode sees a single frame). Envs whose
start has fewer preceding steps get the ones that exist; the buffer pads the
rest exactly as it does during its own warm-up.
"""

from __future__ import annotations

import numpy as np


def make_policy_class(swm):
    base = swm.policy.WorldModelPolicy

    class PrefillPolicy(base):
        def __init__(self, *args, prefill=None, **kwargs):
            super().__init__(*args, **kwargs)
            # prefill: per-env lists; pixels[i] (k_i, H, W, C) uint8 for steps
            # s-k_i..s-1, action[i] (k_i + 1, A) raw actions for s-k_i-1..s-1.
            self.prefill = prefill
            self._prefilled = False

        def get_action(self, info_dict, **kwargs):
            if (self.prefill is not None and not self._prefilled
                    and self._history_buffer is not None):
                self._prefilled = True
                # Apply the reset flush first so it cannot wipe the prefill.
                info_dict = dict(info_dict)
                flush = info_dict.pop("_needs_flush", None)
                if flush is not None:
                    ids = [i for i in range(len(flush)) if bool(flush[i])]
                    for i in ids:
                        self._action_buffer[i].clear()
                    if ids:
                        self._history_buffer.reset(ids)
                keys = (*self.history_keys, "action")
                cur = np.array(info_dict["action"], dtype=np.float32, copy=True)
                for i, (pix, act) in enumerate(zip(self.prefill["pixels"], self.prefill["action"])):
                    for j in range(len(pix)):
                        entry = self._prepare_info({"pixels": pix[None, j:j + 1], "action": act[None, j:j + 1]})
                        self._history_buffer._buffers[i].append({k: entry[k][0] for k in keys})
                    if len(pix):
                        # The first real entry records the action that led to it.
                        cur[i] = act[-1:]
                info_dict["action"] = cur
            return super().get_action(info_dict, **kwargs)

    return PrefillPolicy


def load_prefill(dataset, episodes, starts, steps: int):
    """Per env: dataset pixels for the (up to) ``steps`` steps before the start
    and the actions that led into each of them plus into the start frame.
    The action before the oldest prefilled step only fills that entry's
    'action', which no requested block uses; it is NaN when it does not exist."""
    pix_out, act_out = [], []
    for e, s in zip(episodes, starts):
        k = min(int(s), steps)
        if k == 0:
            pix_out.append(np.zeros((0,), np.uint8))
            act_out.append(np.zeros((0,), np.float32))
            continue
        lo = s - k - 1
        c = dataset.load_chunk(np.array([e]), np.array([max(lo, 0)]), np.array([s]))[0]
        p = c["pixels"]
        a = c["action"]
        p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
        a = np.asarray(a.numpy() if hasattr(a, "numpy") else a, np.float32)
        if p.shape[-1] != 3:  # stored channel-first
            p = np.moveaxis(p, 1, -1)
        if lo < 0:
            p = np.concatenate([p[:1], p])
            a = np.concatenate([np.full_like(a[:1], np.nan), a])
        pix_out.append(p[1:].astype(np.uint8))   # steps s-k..s-1
        act_out.append(a)                        # steps s-k-1..s-1
    return {"pixels": pix_out, "action": act_out}
