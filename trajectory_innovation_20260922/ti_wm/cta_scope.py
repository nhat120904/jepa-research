"""Matched endpoint/trajectory codec components; no predictor or action input."""
import hashlib

import torch
from torch import nn

from ti_wm.cta import FutureDecoder, Scorer, SourceEncoder, TOKENS


class EndpointDecoder(FutureDecoder):
    """Common auxiliary target: endpoint features only, for both input arms."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queries = nn.Parameter(self.queries[:TOKENS].detach().clone())

    def forward(self, ctx, code):
        mem = torch.cat([self.ctx(ctx), self.code(code.float()) + self.code_pos], dim=1)
        y = self.out(self.decoder(self.queries.expand(code.shape[0], -1, -1), mem)).float()
        return ctx['cur'].float() + y


def future_evidence(future, path):
    keys = ('end', 'prop', 'seg') if path else ('end', 'prop')
    return {k: future[k] for k in keys}


def build(seed, path, m=16, width=256, heads=8, dim=128,
          encoder_layers=3, reader_layers=4, decoder_layers=2):
    """Exactly equal common initial weights despite the extra path projection."""
    common = dict(m=m, width=width, heads=heads, dim=dim)
    torch.manual_seed(seed)
    template = SourceEncoder(path=True, layers=encoder_layers, **common)
    if path:
        enc = template
    else:
        enc = SourceEncoder(path=False, layers=encoder_layers, **common)
        keys = enc.state_dict()
        enc.load_state_dict({k: v for k, v in template.state_dict().items() if k in keys}, strict=True)
    torch.manual_seed(seed + 1_000_000)
    reader = Scorer('code', layers=reader_layers, **common)
    torch.manual_seed(seed + 2_000_000)
    decoder = EndpointDecoder(layers=decoder_layers, **common)
    return {'enc': enc, 'reader': reader, 'dec': decoder}


def shared_initial_hash(models):
    """Hash only common weights; path-only projection/positions are reported separately."""
    digest = hashlib.sha256()
    for name, model in sorted(models.items()):
        for key, value in sorted(model.state_dict().items()):
            if name == 'enc' and (key.startswith('fut.seg.') or key == 'fut.seg_pos'):
                continue
            digest.update(f'{name}.{key}'.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
