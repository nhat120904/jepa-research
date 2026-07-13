"""Focused CPU-only tests for the TRM-style baseline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.heads.trajectory_reachability import (
    TrajectoryReachabilityMetric,
    standardize_candidate_cost,
    trm_terminal_cost,
)
from trm_protocol import (materialize_temporal_pairs,
                          sample_balanced_temporal_pairs,
                          spearman_rank_correlation)


def test_pair_head_is_symmetric_and_nonnegative():
    torch.manual_seed(3)
    metric = TrajectoryReachabilityMetric(latent_dim=5, hidden=16).eval()
    z_i = torch.randn(7, 2, 3, 5)
    z_j = torch.randn(7, 2, 3, 5)
    forward = metric(z_i, z_j)
    reverse = metric(z_j, z_i)
    assert torch.allclose(forward, reverse, atol=1e-6)
    assert torch.all(forward >= 0)


class _FixedMetric(torch.nn.Module):
    def forward(self, z_i, z_j):
        # Deliberately differs from raw MSE while remaining symmetric.
        return (z_i.reshape(len(z_i), -1) - z_j.reshape(len(z_j), -1)).abs().sum(-1)


def test_hybrid_is_per_population_standardized_sum():
    z_goal = torch.zeros(3)
    z_final = torch.tensor([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 1.0, 1.0]])
    metric = _FixedMetric()
    got = trm_terminal_cost(metric, z_final, z_goal, mode="hybrid", hybrid_weight=0.5)
    raw = z_final.square().mean(-1)
    learned = z_final.abs().sum(-1)
    expected = standardize_candidate_cost(raw) + 0.5 * standardize_candidate_cost(learned)
    assert torch.allclose(got, expected)


def test_replacement_uses_only_pair_head():
    z_goal = torch.zeros(2)
    z_final = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metric = _FixedMetric()
    assert torch.equal(
        trm_terminal_cost(metric, z_final, z_goal, mode="replacement"),
        torch.tensor([3.0, 7.0]),
    )


def test_balanced_sampler_respects_horizon_and_supplied_split():
    rng = np.random.default_rng(17)
    pairs = sample_balanced_temporal_pairs(
        {"train/a": 8, "train/b": 5}, 20000, max_gap=6, rng=rng
    )
    assert {p.trajectory_id for p in pairs} <= {"train/a", "train/b"}
    assert all(1 <= p.gap <= 6 and abs(p.j - p.i) == p.gap for p in pairs)
    counts = np.bincount([p.gap for p in pairs], minlength=7)[1:]
    assert (counts.max() - counts.min()) / counts.mean() < 0.08
    assert any(p.i > p.j for p in pairs) and any(p.i < p.j for p in pairs)


class _TinyCache:
    def __init__(self):
        self.rows = {
            "a": {"z": np.arange(12, dtype=np.float32).reshape(4, 3), "action": np.zeros((3, 1))},
            "b": {"z": (100 + np.arange(9, dtype=np.float32)).reshape(3, 3), "action": np.zeros((2, 1))},
        }
        self.reads = []

    def read_trajectory(self, tid):
        self.reads.append(tid)
        return self.rows[tid]


def test_materialization_reads_each_trajectory_once():
    cache = _TinyCache()
    rng = np.random.default_rng(1)
    pairs = sample_balanced_temporal_pairs({"a": 4, "b": 3}, 20, 2, rng)
    z_i, z_j, gap = materialize_temporal_pairs(cache, pairs)
    assert z_i.shape == z_j.shape == (20, 3)
    assert gap.shape == (20,)
    assert sorted(cache.reads) == sorted({p.trajectory_id for p in pairs})


def test_tie_aware_spearman_and_invalid_shape():
    assert spearman_rank_correlation(
        np.asarray([0, 0, 1, 2]), np.asarray([0, 0, 2, 4])
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        spearman_rank_correlation(np.asarray([1, 2]), np.asarray([1, 2]))
