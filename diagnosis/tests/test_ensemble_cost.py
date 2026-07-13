"""Offline unit tests for the ensemble/disagreement cost in
scripts/39_latent_oracle_ensemble.py. Pure-tensor, no env/GPU/model needed.

The cost scores each seed against its OWN goal readout (per-seed distance d_k),
so cost = mean_k d_k² + λ·Var_k[d_k]."""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_ensemble():
    spec = _ilu.spec_from_file_location(
        "latent_oracle_ensemble", str(ROOT / "scripts" / "39_latent_oracle_ensemble.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_zero_at_goal_each_seed_its_own():
    """A candidate that matches EACH seed's own goal readout -> every d_k=0 -> cost 0,
    even though the seeds' goal estimates differ from each other (bias cancels)."""
    m = _load_ensemble()
    K, B, obj = 5, 4, 3
    E_goal = torch.randn(K, obj)                       # seeds disagree at the goal
    E = E_goal[:, None, :].expand(K, B, obj).contiguous()
    c = m.ensemble_cost(E, E_goal, s_g=0.1276, lam=1.0)
    assert torch.allclose(c, torch.zeros(B), atol=1e-5)


def test_lambda_zero_is_pure_consensus():
    """lam=0 must drop the disagreement term: cost = mean_k d_k²."""
    m = _load_ensemble()
    torch.manual_seed(0)
    K, B, obj = 5, 3, 3
    E = torch.randn(K, B, obj)
    E_goal = torch.randn(K, obj)
    s_g = 0.2
    c0 = m.ensemble_cost(E, E_goal, s_g=s_g, lam=0.0)
    d = (E - E_goal[:, None, :]).norm(dim=-1) / s_g
    assert torch.allclose(c0, (d ** 2).mean(0), atol=1e-6)


def test_disagreement_penalises_exploit_pocket():
    """Two candidates with the SAME mean per-seed distance but different spread: for
    lam>0 the higher-spread one (an exploit pocket: one seed fooled to ~0, others far)
    costs more; for lam=0 they tie (spread invisible)."""
    m = _load_ensemble()
    K, obj = 4, 3
    E_goal = torch.zeros(K, obj)
    # candidate 0: every seed reads distance √2 -> d²=2 each, mean 2, var 0
    tight = torch.tensor([[2.0 ** 0.5, 0, 0]] * K)
    # candidate 1: distances {0,0,2,2} -> d²={0,0,4,4} mean 2 (same consensus), var>0
    loose = torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [2.0, 0, 0], [2.0, 0, 0]])
    E = torch.stack([tight, loose], dim=1)                       # (K, 2, obj)

    d_tight = tight.norm(dim=-1); d_loose = loose.norm(dim=-1)
    assert torch.allclose((d_tight ** 2).mean(), (d_loose ** 2).mean())   # equal consensus
    assert d_loose.var(unbiased=False) > d_tight.var(unbiased=False)

    c_pos = m.ensemble_cost(E, E_goal, s_g=0.1276, lam=1.0)
    assert c_pos[1] > c_pos[0]                                   # spread punished
    c_zero = m.ensemble_cost(E, E_goal, s_g=0.1276, lam=0.0)
    assert torch.allclose(c_zero[0], c_zero[1], atol=1e-6)       # spread invisible


def test_larger_lambda_monotone_in_spread():
    """Increasing lam must not decrease the cost of any candidate (Var ≥ 0)."""
    m = _load_ensemble()
    torch.manual_seed(1)
    K, B, obj = 5, 6, 3
    E = torch.randn(K, B, obj)
    E_goal = torch.randn(K, obj)
    c_lo = m.ensemble_cost(E, E_goal, s_g=0.1276, lam=0.5)
    c_hi = m.ensemble_cost(E, E_goal, s_g=0.1276, lam=2.0)
    assert (c_hi >= c_lo - 1e-6).all()
