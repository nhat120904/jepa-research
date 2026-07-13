import importlib.util as _ilu
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts._coverage_selection_metrics import (
    coverage_selection_summary,
    spearman_costs,
    topk_overlap,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_latent_oracle():
    spec = _ilu.spec_from_file_location(
        "latent_oracle_candidate_hook_test", ROOT / "scripts" / "30_latent_oracle.py")
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perfect_order_has_no_selection_regret():
    proxy = np.array([0.1, 0.2, 0.3, 0.4])
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    result = coverage_selection_summary(
        proxy, truth, [True, False, False, False], [True, False, False, False],
        topk_frac=0.5,
    )
    assert result["selected_physical_regret"] == 0.0
    assert result["proxy_true_spearman"] == pytest.approx(1.0)
    assert result["proxy_true_topk_overlap"] == 1.0
    assert result["coverage_success_end"] == 1
    assert result["selected_success_end"] == 1


def test_coverage_can_coexist_with_selection_failure():
    proxy = np.array([0.0, 1.0, 2.0, 3.0])
    truth = np.array([4.0, 3.0, 2.0, 1.0])
    success = np.array([False, False, False, True])
    result = coverage_selection_summary(proxy, truth, success, success, topk_frac=0.25)
    assert result["coverage_success_any"] == 1
    assert result["selected_success_any"] == 0
    assert result["selected_physical_regret"] == 3.0
    assert result["proxy_true_spearman"] == pytest.approx(-1.0)
    assert result["proxy_true_topk_overlap"] == 0.0


def test_spearman_uses_average_tie_ranks_and_topk_is_reproducible():
    assert spearman_costs([0, 0, 1], [0, 0, 1]) == pytest.approx(1.0)
    assert np.isnan(spearman_costs([1, 1], [0, 1]))
    assert topk_overlap([0, 0, 2], [0, 1, 2], 1) == 1.0


def test_rejects_misaligned_candidates():
    with pytest.raises(ValueError):
        coverage_selection_summary([0, 1], [0], [False], [False])


def test_full_candidate_hook_preserves_alignment(monkeypatch):
    lo = _load_latent_oracle()
    dim, population, iterations = 4, 12, 3

    def fake_roll(env, snap, plan_raw, *, return_success=False):
        action = np.asarray(plan_raw, dtype=np.float32).reshape(-1)
        raw = np.zeros(39, dtype=np.float32)
        raw[:dim] = action[:dim]
        base = (action, np.zeros(4, dtype=np.float32), raw)
        success = bool(action[0] < 0)
        return (*base, success, success) if return_success else base

    def fake_encode(adapter, frames, proprios, device):
        return torch.tensor(np.stack([frame[:dim] for frame in frames]))

    monkeypatch.setattr(lo, "snapshot", lambda env: {})
    monkeypatch.setattr(lo, "restore", lambda env, snap: None)
    monkeypatch.setattr(lo, "roll_final_frame", fake_roll)
    monkeypatch.setattr(lo, "encode_batch", fake_encode)
    calls = []

    def on_candidates(z, raw, proxy, actions, iteration, *, success_any, success_end):
        raw_arr = np.asarray(raw)
        # Candidate index alignment holds across latent, raw state, action, cost,
        # and exact success arrays before any elite sorting is applied.
        np.testing.assert_allclose(z[:, 0].numpy(), raw_arr[:, 0])
        np.testing.assert_allclose(actions[:, 0, 0], raw_arr[:, 0])
        np.testing.assert_allclose(proxy, (z ** 2).sum(-1).numpy())
        np.testing.assert_array_equal(success_any, raw_arr[:, 0] < 0)
        np.testing.assert_array_equal(success_end, success_any)
        calls.append((iteration, len(proxy)))

    lo.cem_plan_latent(
        env=None, adapter=None, z_goal=torch.zeros(dim), device=torch.device("cpu"),
        plan_h=1, num_samples=population, iterations=iterations, elite_frac=0.25,
        var0=1.0, rng=np.random.default_rng(7), cost_fn=lambda z: (z ** 2).sum(-1),
        on_candidates=on_candidates,
    )
    assert calls == [(i, population) for i in range(iterations)]
