"""Offline unit tests for the latent counterfactual objective in
scripts/40_train_predictor_cf.py (cf_infonce_loss). Pure-tensor, no GPU/cache/model."""

from __future__ import annotations

import importlib.util as _ilu
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_cf():
    spec = _ilu.spec_from_file_location(
        "predictor_cf", str(ROOT / "scripts" / "40_train_predictor_cf.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_perfect_grounding_low_loss_full_rank_acc():
    """Factual distance 0, all counterfactuals far -> factual is always ranked first
    and the CE loss is ~0 (softmax puts all mass on the factual column)."""
    m = _load_cf()
    B, K = 8, 4
    d_fac = torch.zeros(B)
    d_neg = torch.full((B, K), 5.0)
    loss, rank_acc, temp = m.cf_infonce_loss(d_fac, d_neg, temp=0.1)
    assert rank_acc == 1.0
    assert loss < 1e-2
    assert temp == 0.1


def test_action_blind_is_chance():
    """Action-blind predictor: factual and counterfactual distances identical ->
    uniform softmax -> loss = log(1+K) and rank_acc at chance (argmax ties to col 0
    under a stable argmax, but the LOSS is the chance signal we gate on)."""
    m = _load_cf()
    B, K = 16, 3
    d_fac = torch.full((B,), 2.0)
    d_neg = torch.full((B, K), 2.0)
    loss, rank_acc, _ = m.cf_infonce_loss(d_fac, d_neg)
    assert abs(float(loss) - math.log(1 + K)) < 1e-5


def test_auto_temp_is_mean_factual():
    """Default temperature auto-scales to mean(d_fac) so logits stay O(1)."""
    m = _load_cf()
    d_fac = torch.tensor([1.0, 3.0])          # mean 2.0
    d_neg = torch.tensor([[4.0], [5.0]])
    _, _, temp = m.cf_infonce_loss(d_fac, d_neg, temp=None)
    assert abs(temp - 2.0) < 1e-6


def test_loss_decreases_as_factual_gets_closer():
    """Monotonicity: shrinking the factual distance (better grounding) must lower the
    InfoNCE loss when counterfactuals are held fixed and far."""
    m = _load_cf()
    B, K = 8, 4
    d_neg = torch.full((B, K), 4.0)
    loss_far, _, _ = m.cf_infonce_loss(torch.full((B,), 3.0), d_neg, temp=1.0)
    loss_near, _, _ = m.cf_infonce_loss(torch.full((B,), 0.5), d_neg, temp=1.0)
    assert loss_near < loss_far


def test_gradient_flows_to_factual_prediction():
    """The loss must pull F(z,a) TOWARD the true next latent: a POSITIVE grad wrt
    d_fac means gradient descent shrinks d_fac (loss rises as the factual prediction
    drifts away). Checked at the tied action-blind point, where the CF term is the
    only active signal."""
    m = _load_cf()
    d_fac = torch.tensor([2.0, 2.0], requires_grad=True)
    d_neg = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    loss, _, _ = m.cf_infonce_loss(d_fac, d_neg, temp=1.0)
    loss.backward()
    assert (d_fac.grad > 0).all()
