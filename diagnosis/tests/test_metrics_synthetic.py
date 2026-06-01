"""Synthetic-model validation of the metric code (the exact per-transition path
the runner uses). Mirrors scripts/07_validate_synthetic.py as assertions."""

import numpy as np
import torch

from metrics import cra_per_transition, aug_per_transition, sample_negatives, effect_mask
from models.adapters.synthetic import PerfectModel, ActionIgnoringModel, RandomModel

K = 16


def _dataset(N=512, D=64, A=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    z_t = torch.randn(N, D, generator=g)
    a_t = torch.randn(N, A, generator=g).clamp(-1, 1)
    W = torch.randn(D, A, generator=g) * 0.3
    z_t1 = z_t + a_t @ W.T
    return z_t, a_t, z_t1, W


def test_perfect_model_scores_top1_one_and_positive_aug():
    z_t, a_t, z_t1, W = _dataset()
    m = PerfectModel(lookup_fn=lambda z, a: z + a @ W.T, action_dim=4)
    a_neg = sample_negatives("random", a_t=a_t, K=K, action_bounds=(-1.0, 1.0))
    correct, recip, _, _ = cra_per_transition(m, z_t, a_t, z_t1, a_neg)
    aug = aug_per_transition(m, z_t, a_t, z_t1)
    assert correct.mean() > 0.98
    assert recip.mean() > 0.98
    assert aug.mean() > 0.0


def test_action_ignoring_is_chance_and_zero_aug():
    z_t, a_t, z_t1, _ = _dataset()
    m = ActionIgnoringModel(action_dim=4)
    a_neg = sample_negatives("random", a_t=a_t, K=K, action_bounds=(-1.0, 1.0))
    correct, _, _, _ = cra_per_transition(m, z_t, a_t, z_t1, a_neg)
    aug = aug_per_transition(m, z_t, a_t, z_t1)
    chance = 1.0 / (K + 1)
    assert abs(correct.mean() - chance) < 0.02   # tie-fair top-1 == chance
    assert abs(aug.mean()) < 1e-6                 # ignores actions → AUG 0


def test_random_model_near_chance():
    z_t, a_t, z_t1, _ = _dataset()
    m = RandomModel(action_dim=4, sigma=0.1, seed=7)
    a_neg = sample_negatives("random", a_t=a_t, K=K, action_bounds=(-1.0, 1.0))
    correct, _, _, _ = cra_per_transition(m, z_t, a_t, z_t1, a_neg)
    assert abs(correct.mean() - 1.0 / (K + 1)) < 0.10


def test_effect_mask_selects_changed_transitions():
    z_t = torch.zeros(10, 8)
    z_t1 = z_t.clone()
    z_t1[5:] += 1.0  # half the transitions "change"
    mask = effect_mask(z_t, z_t1, threshold=0.5)
    assert mask.sum() == 5 and mask[5:].all() and not mask[:5].any()


def test_cra_handles_6d_latent_shape():
    # Real models return (B, V, H, W, D); metrics must flatten transparently.
    z_t = torch.randn(32, 1, 4, 4, 8)
    a_t = torch.randn(32, 4)
    W = torch.randn(1 * 4 * 4 * 8, 4) * 0.2

    def lookup(z, a):
        delta = (a @ W.T).reshape(z.shape)
        return z + delta

    m = PerfectModel(lookup_fn=lookup, action_dim=4)
    z_t1 = lookup(z_t, a_t)
    a_neg = sample_negatives("random", a_t=a_t, K=8, action_bounds=(-1.0, 1.0))
    correct, _, _, _ = cra_per_transition(m, z_t, a_t, z_t1, a_neg)
    assert correct.mean() > 0.9
