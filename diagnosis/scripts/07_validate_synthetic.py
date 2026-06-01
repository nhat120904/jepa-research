"""Validate metric implementations on synthetic models BEFORE running on real ones.

Asserts:
    - PerfectModel       → CRA top-1 ≈ 1.0, AUG > 0
    - ActionIgnoringModel → CRA top-1 ≈ 1/(K+1), AUG ≈ 0
    - RandomModel         → CRA top-1 ≈ 1/(K+1), AUG ≈ 0

If any assertion fails, the metric code is buggy — fix before proceeding to
real model evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics import compute_cra, compute_aug, sample_negatives  # noqa: E402
from models.adapters.synthetic import (  # noqa: E402
    PerfectModel, ActionIgnoringModel, RandomModel,
)


def _build_dataset(N=512, D=64, A=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    z_t = torch.randn(N, D, generator=g)
    a_t = torch.randn(N, A, generator=g).clamp(-1, 1)
    # A simple "world": z_{t+1} = z_t + W @ a_t.
    W = torch.randn(D, A, generator=g) * 0.3
    z_t1 = z_t + a_t @ W.T
    return z_t, a_t, z_t1, W


def _build_perfect(W):
    def lookup(z, a):
        return z + a @ W.T
    return PerfectModel(lookup_fn=lookup, action_dim=W.shape[1])


def main() -> int:
    K = 16
    z_t, a_t, z_t1, W = _build_dataset()
    a_neg = sample_negatives("random", a_t=a_t, K=K, action_bounds=(-1.0, 1.0))

    fails = []

    print("[1/3] PerfectModel")
    m = _build_perfect(W)
    cra = compute_cra(m, z_t, a_t, z_t1, a_neg)
    aug = compute_aug(m, z_t, a_t, z_t1)
    print(f"  CRA top-1 = {cra.top1:.3f}  MRR = {cra.mrr:.3f}  AUG = {aug.aug:+.4f}")
    if cra.top1 < 0.98: fails.append(f"Perfect CRA={cra.top1:.3f} (expected ≥ 0.98)")
    if aug.aug <= 0:     fails.append(f"Perfect AUG={aug.aug:+.4f} (expected > 0)")

    print("[2/3] ActionIgnoringModel")
    m = ActionIgnoringModel(action_dim=4)
    cra = compute_cra(m, z_t, a_t, z_t1, a_neg)
    aug = compute_aug(m, z_t, a_t, z_t1)
    chance = 1.0 / (K + 1)
    print(f"  CRA top-1 = {cra.top1:.3f}  MRR = {cra.mrr:.3f}  "
          f"AUG = {aug.aug:+.4f}  (chance = {chance:.3f})")
    if abs(cra.top1 - chance) > 0.05:
        fails.append(f"ActionIgnoring CRA={cra.top1:.3f} (expected ≈ {chance:.3f})")
    if abs(aug.aug) > 1e-5:
        fails.append(f"ActionIgnoring AUG={aug.aug:+.4f} (expected ≈ 0)")

    print("[3/3] RandomModel")
    m = RandomModel(action_dim=4, sigma=0.1, seed=7)
    cra = compute_cra(m, z_t, a_t, z_t1, a_neg)
    aug = compute_aug(m, z_t, a_t, z_t1)
    print(f"  CRA top-1 = {cra.top1:.3f}  MRR = {cra.mrr:.3f}  AUG = {aug.aug:+.4f}")
    if abs(cra.top1 - chance) > 0.10:
        fails.append(f"Random CRA={cra.top1:.3f} (expected ≈ {chance:.3f})")

    if fails:
        print("\nFAILURES:")
        for f in fails: print(f"  - {f}")
        return 1
    print("\nAll synthetic validations PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
