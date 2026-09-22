"""Differentiable truncated tensor signatures of piecewise-linear feature paths.

No simulator/task labels, learned merger, logsignature, or external signature library.
Tensor levels are flattened in lexicographic coordinate order; level zero is one.
The exact algebra applies to the chosen sampled/interpolated path, NOT to an
unobserved continuous physical path, and does not guarantee useful task information.
"""

from __future__ import annotations

import torch


def identity(channels: int, degree: int, *, like: torch.Tensor) -> tuple:
    if channels < 1 or degree < 1:
        raise ValueError("channels and degree must be positive")
    batch = like.shape[:-1]
    return (like.new_ones((*batch, 1)),) + tuple(
        like.new_zeros((*batch, channels ** k)) for k in range(1, degree + 1))


def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(-2)


def chen(left: tuple, right: tuple) -> tuple:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("matching positive truncation degrees required")
    d = left[1].shape[-1]
    for k, (a, b) in enumerate(zip(left, right)):
        if a.shape != b.shape or a.shape[-1] != d ** k:
            raise ValueError("incompatible signature shapes/batches")
    return tuple(sum((outer(left[j], right[k-j]) for j in range(k+1)),
                     torch.zeros_like(left[k])) for k in range(len(left)))


def line_signature(delta: torch.Tensor, degree: int) -> tuple:
    """Tensor exponential: level k = delta**tensor(k) / k!."""
    if degree < 1:
        raise ValueError("positive degree required")
    levels = [torch.ones_like(delta[..., :1])]
    for k in range(1, degree + 1):
        levels.append(outer(levels[-1], delta) / k)
    return tuple(levels)


def signature(path: torch.Tensor, degree: int = 3, *, stream: bool = False) -> tuple:
    """Input [..., observations, channels], INCLUDING start/end boundaries.

    One observation represents zero duration and returns identity. With stream=True
    return a prefix signature at every observation, including the initial identity.
    Use fp64 for audits/inversion; half-precision accumulation is rejected.
    """
    if path.ndim < 2 or path.shape[-2] < 1 or path.dtype not in (torch.float32, torch.float64):
        raise ValueError("nonempty float32/64 path required")
    state = identity(path.shape[-1], degree, like=path[..., 0, :])
    prefixes = [state] if stream else None
    for delta in (path[..., 1:, :] - path[..., :-1, :]).unbind(-2):
        state = chen(state, line_signature(delta, degree))
        if stream:
            prefixes.append(state)
    if stream:
        return tuple(torch.stack([s[k] for s in prefixes], dim=-2) for k in range(degree + 1))
    return state


def inverse(sig: tuple) -> tuple:
    """Inverse in truncated tensor algebra for unit level zero."""
    if not torch.allclose(sig[0], torch.ones_like(sig[0])):
        raise ValueError("inverse expects unit level zero")
    out = [sig[0]]
    for k in range(1, len(sig)):
        out.append(-sum((outer(out[j], sig[k-j]) for j in range(k)),
                        torch.zeros_like(sig[k])))
    return tuple(out)


def interval(prefix: tuple, start: int, end: int) -> tuple:
    """Signature on [start,end] from a streamed prefix; bounds are observations."""
    if not 0 <= start <= end < prefix[0].shape[-2]:
        raise ValueError("invalid interval")
    return chen(inverse(tuple(x[..., start, :] for x in prefix)),
                tuple(x[..., end, :] for x in prefix))


def flatten(sig: tuple) -> torch.Tensor:
    """Exclude the constant level zero from prediction targets."""
    return torch.cat(sig[1:], dim=-1)


def integrated_lift(features: torch.Tensor, dt: float = 0.05,
                    initial: torch.Tensor | None = None) -> torch.Tensor:
    """X=(physical time, integral features dt), trapezoidal integration.

    No actions enter this target, no per-window time normalization. Splitting an
    already constructed path preserves the exact sampled path. An independently
    constructed right segment differs only by a constant translation (irrelevant
    to signatures) if the same boundary feature and dt are supplied.
    """
    if features.ndim < 2 or features.shape[-2] < 1 or dt <= 0:
        raise ValueError("nonempty features and positive dt required")
    delta_features = dt * (features[..., 1:, :] + features[..., :-1, :]) / 2
    delta_time = torch.full_like(delta_features[..., :1], dt)
    increments = torch.cat([delta_time, delta_features], -1)
    origin = features.new_zeros((*features.shape[:-2], 1, features.shape[-1] + 1))
    path = torch.cat([origin, increments.cumsum(-2)], -2)
    if initial is not None:
        path = path + initial.unsqueeze(-2)
    return path
