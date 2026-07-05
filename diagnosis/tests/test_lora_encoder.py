"""Offline unit tests for the encoder-LoRA injection (Phase D,
docs/plans/2026-07-02-encoder-lora-action-grounding-design.md).

Pure CPU tensors on a synthetic ViT-like encoder — no upstream repo, no GPU, no
checkpoints. Locks the three contracts scripts/30 --encoder-lora and scripts/38
rely on:

  1. zero-init identity — an injected-but-untrained (or toggled-off) encoder is
     numerically IDENTICAL to the frozen one (the check_normalization gate on the
     server depends on this);
  2. gradients flow only into LoRA params, never the base weights;
  3. state-dict save/load round-trips by module name onto a fresh injection.
"""

import torch
import torch.nn as nn

from models.heads.lora_encoder import (
    _encoder_of, encoder_lora_state_dict, inject_encoder_lora, load_encoder_lora)
from models.heads.lora_predictor import LoRALinear, set_lora_enabled


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(3 * d, d)
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)

    def forward(self, x):
        x = x + self.proj(torch.relu(self.qkv(x)))
        return x + self.fc2(torch.relu(self.fc1(x)))


class _FakeEncoder(nn.Module):
    def __init__(self, d=16, n_blocks=2):
        super().__init__()
        self.patch_embed = nn.Linear(8, d)     # outside "blocks" — must NOT be injected
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n_blocks)])

    def forward(self, x):
        h = self.patch_embed(x)
        for b in self.blocks:
            h = b(h)
        return h


class _FakeWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()


class _FakeAdapter:
    def __init__(self):
        self.wm = _FakeWM()
        self.encpred = self.wm


def _fresh():
    torch.manual_seed(0)
    return _FakeAdapter()


def test_inject_targets_only_blocks():
    ad = _fresh()
    injected = inject_encoder_lora(ad, r=4, alpha=8)
    # 2 blocks x 4 Linears each; patch_embed untouched.
    assert len(injected) == 8
    assert isinstance(ad.wm.encoder.patch_embed, nn.Linear)
    assert not isinstance(ad.wm.encoder.patch_embed, LoRALinear)
    assert all(isinstance(m, LoRALinear) for m in injected)


def test_zero_init_identity_and_toggle():
    torch.manual_seed(1)
    x = torch.randn(5, 3, 8)
    ad = _fresh()
    y_frozen = ad.wm.encoder(x)
    injected = inject_encoder_lora(ad, r=4, alpha=8)
    # B is zero-init -> injected output is exactly the frozen output.
    assert torch.allclose(ad.wm.encoder(x), y_frozen, atol=0)
    # Perturb LoRA -> output moves; toggle off -> exact frozen output again.
    with torch.no_grad():
        for m in injected:
            m.B.add_(0.1)
    assert not torch.allclose(ad.wm.encoder(x), y_frozen)
    set_lora_enabled(injected, False)
    assert torch.allclose(ad.wm.encoder(x), y_frozen, atol=0)
    set_lora_enabled(injected, True)


def test_gradients_only_reach_lora():
    ad = _fresh()
    injected = inject_encoder_lora(ad, r=4, alpha=8)
    out = ad.wm.encoder(torch.randn(2, 3, 8)).sum()
    out.backward()
    for m in injected:
        assert m.A.grad is not None and m.B.grad is not None
        assert m.base.weight.grad is None and not m.base.weight.requires_grad
    assert ad.wm.encoder.patch_embed.weight.grad is not None  # not frozen, but not LoRA'd


def test_state_dict_roundtrip(tmp_path):
    ad = _fresh()
    injected = inject_encoder_lora(ad, r=4, alpha=8)
    with torch.no_grad():
        for i, m in enumerate(injected):
            m.A.copy_(torch.full_like(m.A, 0.01 * (i + 1)))
            m.B.copy_(torch.full_like(m.B, 0.02 * (i + 1)))
    x = torch.randn(3, 2, 8)
    y_trained = ad.wm.encoder(x)
    ckpt_path = tmp_path / "enc_lora.pt"
    torch.save({"r": 4, "alpha": 8, "target_substrs": ("blocks",),
                "lora": encoder_lora_state_dict(injected, ad)}, ckpt_path)

    ad2 = _fresh()
    injected2, ckpt = load_encoder_lora(ad2, ckpt_path, "cpu")
    assert len(injected2) == len(injected) and ckpt["r"] == 4
    assert torch.allclose(ad2.wm.encoder(x), y_trained, atol=1e-6)


def test_encoder_of_errors_without_encoder():
    class _NoEnc:
        wm = nn.Module()
        encpred = wm
    try:
        _encoder_of(_NoEnc())
        raised = False
    except RuntimeError:
        raised = True
    assert raised
