import torch.nn as nn

from models.heads.encoder_adaptation import unfreeze_last_encoder_blocks


class DummyAdapter:
    def __init__(self):
        encoder = nn.Module()
        encoder.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(6)])
        encoder.norm = nn.LayerNorm(3)
        self.wm = nn.Module()
        self.wm.encoder = encoder


def test_only_last_blocks_and_final_norm_are_unfrozen():
    adapter = DummyAdapter()
    named, meta = unfreeze_last_encoder_blocks(adapter, n_blocks=2)
    names = {name for name, _ in named}
    assert meta == {"block_list": "blocks", "n_total_blocks": 6, "start": 4}
    assert any(name.startswith("blocks.4") for name in names)
    assert any(name.startswith("blocks.5") for name in names)
    assert any(name.startswith("norm") for name in names)
    assert not any(name.startswith("blocks.3") for name in names)
