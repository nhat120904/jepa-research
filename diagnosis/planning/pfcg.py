"""Probe-frozen controllability geometry for latent terminal costs.

The geometry is estimated from paired ``+u``/``-u`` model rollouts before a
candidate population is scored.  It is then held fixed while candidates are
ranked.  No environment outcome, reward, or simulator state enters these
functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PFCGGeometry:
    """Low-rank empirical finite-horizon controllability geometry."""

    basis: np.ndarray
    eigenvalues: np.ndarray
    ridge: float
    relative_eigen_floor: float
    response_energy_retained: float

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])


def fit_symmetric_probe_geometry(
    plus_endpoints: np.ndarray,
    minus_endpoints: np.ndarray,
    *,
    relative_eigen_floor: float = 1e-6,
    ridge_fraction: float = 0.1,
) -> PFCGGeometry:
    """Fit a low-rank Gramian from paired endpoint responses.

    ``0.5 * (f(+u) - f(-u))`` cancels the common autonomous drift to first
    order.  The response mean is removed before the SVD.  All modes above a
    relative numerical floor are retained; a median-eigenvalue ridge prevents
    weak retained modes from dominating through an unstable inverse.
    """

    plus = np.asarray(plus_endpoints, dtype=np.float64)
    minus = np.asarray(minus_endpoints, dtype=np.float64)
    if plus.shape != minus.shape or plus.ndim != 2:
        raise ValueError(
            "plus/minus endpoints must have identical shape (pairs, latent_dim)"
        )
    if plus.shape[0] < 2:
        raise ValueError("at least two symmetric probe pairs are required")
    if not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise ValueError("probe endpoints must be finite")
    if not 0 < relative_eigen_floor < 1:
        raise ValueError("relative_eigen_floor must lie in (0, 1)")
    if ridge_fraction <= 0:
        raise ValueError("ridge_fraction must be positive")

    response = 0.5 * (plus - minus)
    response -= response.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(response, full_matrices=False)
    eigenvalues = np.square(singular) / (len(response) - 1)
    if not len(eigenvalues) or eigenvalues[0] <= 0:
        raise ValueError("symmetric probes produced no nonzero latent response")

    keep = eigenvalues >= relative_eigen_floor * eigenvalues[0]
    if not np.any(keep):
        raise ValueError("no response mode survived the eigenvalue floor")
    retained = eigenvalues[keep]
    basis = vh[keep].T
    total_energy = float(eigenvalues.sum())
    retained_fraction = float(retained.sum() / total_energy)
    ridge = float(ridge_fraction * np.median(retained))
    return PFCGGeometry(
        basis=basis,
        eigenvalues=retained,
        ridge=ridge,
        relative_eigen_floor=relative_eigen_floor,
        response_energy_retained=retained_fraction,
    )


def latent_l2(endpoints: np.ndarray, goal: np.ndarray) -> np.ndarray:
    residual = np.asarray(endpoints, dtype=np.float64) - np.asarray(
        goal, dtype=np.float64
    )
    return np.square(residual).sum(axis=-1)


def projected_cost(
    endpoints: np.ndarray, goal: np.ndarray, geometry: PFCGGeometry
) -> np.ndarray:
    """Unwhitened PCA/control-subspace baseline."""

    residual = np.asarray(endpoints, dtype=np.float64) - np.asarray(
        goal, dtype=np.float64
    )
    coordinates = residual @ geometry.basis
    return np.square(coordinates).sum(axis=-1)


def pfcg_cost(
    endpoints: np.ndarray, goal: np.ndarray, geometry: PFCGGeometry
) -> np.ndarray:
    """Minimum-energy-style terminal cost in the retained response subspace."""

    residual = np.asarray(endpoints, dtype=np.float64) - np.asarray(
        goal, dtype=np.float64
    )
    coordinates = residual @ geometry.basis
    return (np.square(coordinates) / (geometry.eigenvalues + geometry.ridge)).sum(
        axis=-1
    )


def diagonal_response_cost(
    endpoints: np.ndarray,
    goal: np.ndarray,
    plus_endpoints: np.ndarray,
    minus_endpoints: np.ndarray,
    *,
    relative_variance_floor: float = 1e-6,
    ridge_fraction: float = 0.1,
) -> np.ndarray:
    """Coordinate-wise response-whitening baseline.

    Only coordinates with measurable action response are used.  Unlike PFCG,
    this baseline is not invariant to a rotation of the latent coordinates.
    """

    response = 0.5 * (
        np.asarray(plus_endpoints, dtype=np.float64)
        - np.asarray(minus_endpoints, dtype=np.float64)
    )
    variance = response.var(axis=0, ddof=1)
    if variance.max(initial=0.0) <= 0:
        raise ValueError("symmetric probes produced no diagonal response variance")
    active = variance >= relative_variance_floor * variance.max()
    retained = variance[active]
    ridge = ridge_fraction * np.median(retained)
    residual = np.asarray(endpoints, dtype=np.float64) - np.asarray(
        goal, dtype=np.float64
    )
    return (np.square(residual[:, active]) / (retained + ridge)).sum(axis=-1)


def matched_random_geometry(
    geometry: PFCGGeometry, latent_dim: int, seed: int
) -> PFCGGeometry:
    """Random-subspace negative control with matched rank/eigen spectrum."""

    if geometry.rank > latent_dim:
        raise ValueError("geometry rank cannot exceed latent_dim")
    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal((latent_dim, geometry.rank))
    basis, _ = np.linalg.qr(random_matrix, mode="reduced")
    return PFCGGeometry(
        basis=basis,
        eigenvalues=geometry.eigenvalues.copy(),
        ridge=geometry.ridge,
        relative_eigen_floor=geometry.relative_eigen_floor,
        response_energy_retained=geometry.response_energy_retained,
    )
