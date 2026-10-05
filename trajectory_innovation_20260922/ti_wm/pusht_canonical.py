"""PushT proposal sampling with tensor shapes independent of active roots and candidate count.

Use explicitly for new runs. Historical batched sampling remains in pusht_runtime.PolicyRunner.
The fixed microbatch is part of the numerical policy configuration and must be logged.
"""
import numpy as np
import torch

from ti_wm.pusht_runtime import PolicyRunner


class CanonicalPolicyRunner(PolicyRunner):
    def __init__(self, *args, proposal_microbatch=8, **kwargs):
        if proposal_microbatch < 1:
            raise ValueError("proposal_microbatch must be positive")
        super().__init__(*args, **kwargs)
        self.proposal_microbatch = int(proposal_microbatch)

    @torch.inference_mode()
    def bank(self, hist, seeds):
        seeds = list(seeds)
        if not seeds:
            raise ValueError("An empty proposal bank is not valid")
        cond = self.policy.diffusion._prepare_global_conditioning(self.batch([hist]))
        width = self.proposal_microbatch
        cond = cond.expand(width, -1).contiguous()
        chunks = []
        for first in range(0, len(seeds), width):
            part = seeds[first:first + width]
            padded = part + [part[-1]] * (width - len(part))
            generators = [torch.Generator("cpu").manual_seed(int(seed)) for seed in padded]
            chunks.append(self._actions(cond, generators)[:len(part)].float().cpu().numpy())
        return np.concatenate(chunks)

    @torch.inference_mode()
    def draw(self, hists, seeds):
        """Group consecutive references to the same history, as the batched CTA loops construct their banks.

        Distinct history objects are sampled independently; content comparison is unnecessary. Every call to bank
        uses the same conditioning and denoising shapes, including when only one candidate is requested.
        """
        if len(hists) != len(seeds) or not hists:
            raise ValueError("Expected equally sized, nonempty histories and seeds")
        chunks, first = [], 0
        while first < len(hists):
            last = first + 1
            while last < len(hists) and hists[last] is hists[first]:
                last += 1
            chunks.append(self.bank(hists[first], seeds[first:last]))
            first = last
        return np.concatenate(chunks)
