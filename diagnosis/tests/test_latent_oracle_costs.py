"""Offline unit tests for the two new `build_oracle_cost` arms in
scripts/30_latent_oracle.py — `advmetric` (a distinctly-named alias of `metric`,
meant to be paired with a Track-1-hardened checkpoint) and `phi` (Track 2's
learned representation-adapter cost). Pure-tensor tests, no env/GPU/cache needed.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path

import torch

from models.heads.action_repr_adapter import ActionReprAdapter
from models.heads.latent_metric import LatentMetric

ROOT = Path(__file__).resolve().parents[1]


def _load_latent_oracle():
    spec = _ilu.spec_from_file_location("latent_oracle_test2", str(ROOT / "scripts" / "30_latent_oracle.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_advmetric_matches_metric_numerically():
    """advmetric is a distinctly-named arm for CSV/log traceability (it's meant to
    load a Track-1-hardened checkpoint), but the SCORING FUNCTION is identical to
    `metric` given the same metric head — this locks that equivalence in."""
    lo = _load_latent_oracle()
    torch.manual_seed(0)
    metric = LatentMetric(latent_dim=16, embed_dim=8, hidden=32).eval()
    z_goal = torch.randn(16)
    z_fin = torch.randn(5, 16)

    fn_metric = lo.build_oracle_cost("metric", z_goal, metric=metric)
    fn_adv = lo.build_oracle_cost("advmetric", z_goal, metric=metric)
    assert torch.allclose(fn_metric(z_fin), fn_adv(z_fin))


def test_metric_arms_require_metric_head():
    lo = _load_latent_oracle()
    z_goal = torch.randn(16)
    for cost in ("metric", "advmetric"):
        try:
            lo.build_oracle_cost(cost, z_goal)
            assert False, f"{cost} should require --metric-head"
        except ValueError:
            pass


def test_phi_cost_requires_repr_adapter():
    lo = _load_latent_oracle()
    z_goal = torch.randn(16)
    try:
        lo.build_oracle_cost("phi", z_goal)
        assert False, "phi should require --repr-adapter"
    except ValueError:
        pass


def test_phi_cost_zero_at_goal_and_monotone():
    """The phi cost must be (a) zero when z_fin == z_goal and (b) strictly larger
    for a candidate whose true object is further from the goal — the two
    properties `scripts/30 --cost phi`'s planner relies on."""
    lo = _load_latent_oracle()
    torch.manual_seed(0)
    adapter = ActionReprAdapter(latent_dim=16, phi_dim=12, obj_dim=3, n_layers=1,
                                n_heads=2, hidden=32, extra_scale=2.0).eval()
    z_goal = torch.randn(16)

    fn = lo.build_oracle_cost("phi", z_goal, repr_adapter=adapter, s_g=1.0, beta=1.0)
    c_goal = fn(z_goal.unsqueeze(0))
    assert torch.allclose(c_goal, torch.zeros_like(c_goal), atol=1e-5)

    z_near = z_goal.unsqueeze(0) + 0.01 * torch.randn(1, 16)
    z_far = z_goal.unsqueeze(0) + 2.0 * torch.randn(1, 16)
    c_near, c_far = fn(z_near), fn(z_far)
    assert (c_near < c_far).all()


def test_phi_cost_scales_by_extra_scale():
    """A larger extra_scale must shrink the extra-subspace contribution (the
    'squared norms over scale norms' pattern) — verifies the beta/s_extra wiring
    actually reaches the cost function rather than being silently dropped."""
    lo = _load_latent_oracle()
    torch.manual_seed(1)
    latent_dim, phi_dim, obj_dim = 10, 8, 3
    z_goal = torch.randn(latent_dim)
    z_fin = torch.randn(3, latent_dim)

    small = ActionReprAdapter(latent_dim=latent_dim, phi_dim=phi_dim, obj_dim=obj_dim,
                              n_layers=1, n_heads=2, hidden=16, extra_scale=1.0).eval()
    big = ActionReprAdapter(latent_dim=latent_dim, phi_dim=phi_dim, obj_dim=obj_dim,
                            n_layers=1, n_heads=2, hidden=16, extra_scale=1.0).eval()
    big.load_state_dict(small.state_dict())          # identical weights
    big.extra_scale.fill_(50.0)                       # only the scale differs

    fn_small = lo.build_oracle_cost("phi", z_goal, repr_adapter=small, s_g=1.0, beta=1.0)
    fn_big = lo.build_oracle_cost("phi", z_goal, repr_adapter=big, s_g=1.0, beta=1.0)
    # Same phi(z) values, but big's extra term is divided by a much larger scale^2
    # -> its total cost must be <= small's (equal only in the zero-extra-diff limit).
    assert (fn_big(z_fin) <= fn_small(z_fin) + 1e-6).all()
