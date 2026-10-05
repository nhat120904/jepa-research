"""Fresh task-only action bottleneck with CTA's deployment architecture.

The predictor has the same 16x3 expected-coordinate channel as CTA and is goal
agnostic. Both it and the reader start from random weights and learn task losses
jointly. No code targets, NLL, reconstruction, future inputs, or CTA warm-starts
are part of this baseline. Frozen visual features remain shared with other arms.
"""
import copy

from torch import nn

from ti_wm.cta import CHUNK, LEVELS, Scorer
from ti_wm.cta_parallel import ParallelFSQWM
from ti_wm.sibling import PCA_DIM


class FactorizedDirect(nn.Module):
    def __init__(self, m=16, levels=LEVELS, width=256, predictor_layers=4, reader_layers=4,
                 heads=8, dim=PCA_DIM, dropout=0.0, chunk=CHUNK):
        super().__init__()
        self.predictor = ParallelFSQWM(m=m, levels=levels, width=width, layers=predictor_layers,
                                       heads=heads, dim=dim, dropout=dropout, chunk=chunk)
        self.reader = Scorer("code", m=m, levels=levels, width=width, layers=reader_layers,
                             heads=heads, dim=dim, dropout=dropout, chunk=chunk)
        self.architecture = {"m": m, "levels": list(levels), "width": width,
                             "predictor_layers": predictor_layers, "reader_layers": reader_layers,
                             "heads": heads, "dim": dim, "dropout": dropout, "chunk": chunk}

    def encode(self, ctx, actions):
        """Goal-agnostic z=f(C,A); logits are not supervised as code targets."""
        return self.predictor(ctx, actions)[0]

    def read(self, ctx, code, goal):
        """h(C,z,g); the reader has no action argument or direct action channel."""
        return self.reader(ctx, code, goal)

    def forward(self, ctx, actions, goal):
        return self.read(ctx, self.encode(ctx, actions), goal)

    def deployable_checkpoint(self, config):
        """v2-compatible minimal bundle for ``--arm-spec FDIRECT=FD:cta``.

        This channel name describes the runtime architecture. Config identifies
        the model as task-only Factorized DIRECT rather than future-trained CTA.
        """
        required = {"width": 256, "predictor_layers": 4, "reader_layers": 4,
                    "heads": 8, "dim": PCA_DIM, "levels": list(LEVELS)}
        if any(self.architecture[key] != value for key, value in required.items()):
            raise ValueError("Existing v2 runtime requires standard CTA width/layers/heads/dimension/levels")
        own = copy.deepcopy(config)
        own["m"], own["chunk"] = self.architecture["m"], self.architecture["chunk"]
        own["method"] = "task_only_factorized_direct"
        own["architecture"] = dict(self.architecture)
        return {"config": own,
                "stage1": {"reader": {key: value.detach().cpu().clone() for key, value in self.reader.state_dict().items()}},
                "wms": {"cta": {key: value.detach().cpu().clone() for key, value in self.predictor.state_dict().items()}}}
