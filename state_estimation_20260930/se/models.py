"""Planning-time corrections that leave the world-model weights untouched.

``enable_innovation`` makes a LeWM rollout add ``gamma * nu`` to its
predictions, where ``nu`` is the one-step innovation of the newest context
frame: its encoding minus the prediction from the ``num_frames`` frames and
executed blocks before it. It needs one more context frame than the predictor
uses (history_len = num_frames + 1); with fewer frames the rollout is unchanged.
  mode 'dist': disturbance model, bias added inside the autoregressive loop;
  mode 'post': bias added to the predicted latents after the rollout.
"""

from __future__ import annotations

import torch
from einops import rearrange


def enable_innovation(model, gamma: float, mode: str = "dist"):
    if mode not in ("dist", "post"):
        raise ValueError(mode)
    base = type(model)

    class InnovationLeWM(base):
        def rollout(self, info, action_sequence, history_size=None):
            if history_size is None:
                history_size = getattr(self.predictor, "num_frames", 3)
            H = info["pixels"].size(2)
            B, S, T = action_sequence.shape[:3]
            act_past = info.get("action_history")
            if act_past is None:
                act_past = action_sequence.new_zeros(B, S, 0, action_sequence.size(-1))
            assert act_past.size(2) == H - 1
            info["action"] = torch.cat([act_past, action_sequence[:, :, :1]], dim=2)
            if "emb" not in info:
                _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
                _init = self.encode(_init)
                info["emb"] = _init["emb"].detach().unsqueeze(1).expand(B, S, -1, -1)
            emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
            act_past_flat = rearrange(act_past, "b s ... -> (b s) ...")
            act_cand_flat = rearrange(action_sequence, "b s ... -> (b s) ...")
            all_act_emb = self.action_encoder(torch.cat([act_past_flat, act_cand_flat], dim=1))
            HS = history_size
            bias = None
            if self.innov_gamma and H > HS:
                ctx = emb_init[:, H - 1 - HS:H - 1]
                act = all_act_emb[:, H - 1 - HS:H - 1]
                nu = emb_init[:, H - 1] - self.predict(ctx, act)[:, -1]
                bias = self.innov_gamma * nu
            emb_list = list(emb_init.unbind(dim=1))
            for t in range(T):
                lo = max(0, H + t - HS)
                nxt = self.predict(torch.stack(emb_list[lo:], dim=1), all_act_emb[:, lo:H + t])[:, -1]
                if bias is not None and self.innov_mode == "dist":
                    nxt = nxt + bias
                emb_list.append(nxt)
            emb = torch.stack(emb_list, dim=1)
            if bias is not None and self.innov_mode == "post":
                emb = torch.cat([emb[:, :H], emb[:, H:] + bias[:, None]], dim=1)
            info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
            return info

    model.__class__ = InnovationLeWM
    model.innov_gamma = float(gamma)
    model.innov_mode = mode
    return model
