"""Statistics and pre-registered verdict rules for PushT gates A-C (docs/GATE_ABC_PROTOCOL.md)."""

import math

import numpy as np

PUBLISHED_SUCCESS = 0.654
A_MAX_DEVIATION = 0.10
B_PASS = 0.10
B_EXTEND = 0.05
C_RETENTION = 0.80
BOOTSTRAP = 10_000


def wilson(successes, n, z=1.959963984540054):
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _root_indices(n, resamples, seed):
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def paired_diff(a, b, resamples=BOOTSTRAP, seed=0):
    """Mean of a-b over roots with a percentile bootstrap CI (roots resampled)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Paired arrays must be equal-length vectors")
    diff = a - b
    boot = diff[_root_indices(len(diff), resamples, seed)].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(diff))}


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    from scipy.stats import binomtest

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    if only_a + only_b == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    return {"only_a": only_a, "only_b": only_b, "p": float(binomtest(only_a, only_a + only_b, 0.5).pvalue)}


def cluster_ratio(numerators, denominators, resamples=BOOTSTRAP, seed=0):
    """Ratio of sums with roots as clusters; resamples with a nonpositive denominator are counted."""
    num, den = np.asarray(numerators, float), np.asarray(denominators, float)
    if num.shape != den.shape or not len(num):
        raise ValueError("Cluster arrays must match")
    point = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    idx = _root_indices(len(num), resamples, seed)
    bn, bd = num[idx].sum(axis=1), den[idx].sum(axis=1)
    valid = bd > 0
    ratios = bn[valid] / bd[valid]
    lo, hi = (np.percentile(ratios, [2.5, 97.5]) if valid.any() else (float("nan"),) * 2)
    return {"ratio": point, "lo": float(lo), "hi": float(hi), "undefined_resamples": int((~valid).sum())}


def retention(vis, phys, p0, resamples=BOOTSTRAP, seed=0):
    """(VIS-P0)/(PHYS-P0) on paired root outcomes, bootstrapped by root."""
    vis, phys, p0 = (np.asarray(x, float) for x in (vis, phys, p0))
    return cluster_ratio(vis - p0, phys - p0, resamples, seed)


def verdict_a(official_success):
    k, n = int(np.sum(official_success)), len(official_success)
    rate = k / n
    return {
        "successes": k, "n": n, "rate": rate, "wilson": wilson(k, n),
        "deviation": rate - PUBLISHED_SUCCESS,
        "verdict": "PASS" if abs(rate - PUBLISHED_SUCCESS) <= A_MAX_DEVIATION + 1e-12 else "FAIL",
    }


def runtime_blocker(p0_minus_official):
    d = p0_minus_official
    return abs(d["mean"]) >= A_MAX_DEVIATION and (d["lo"] > 0 or d["hi"] < 0)


def verdict_b(phys_minus_p0):
    d = phys_minus_p0
    if d["mean"] >= B_PASS - 1e-12 and d["lo"] > 0:
        return "PASS"
    if d["mean"] >= B_EXTEND - 1e-12:
        return "EXTEND"
    return "FAIL"


def verdict_c(b_verdict, retention_estimate):
    if b_verdict != "PASS":
        return "NOT_INTERPRETED"
    r = retention_estimate["ratio"]
    return "PASS" if math.isfinite(r) and r >= C_RETENTION - 1e-12 else "FAIL"


def verdict_confirm(diff):
    """C2 confirmation: paired PROG8 - P0 difference with CI lower bound > 0."""
    return "CONFIRMED" if diff["lo"] > 0 else "NOT_CONFIRMED"
