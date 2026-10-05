"""Lights Out over GF(2): minimal number of presses between two configurations.

A press toggles the button and its four grid neighbours (OGBench PuzzleEnv.post_step).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def solver(rows: int, cols: int):
    n = rows * cols
    A = np.zeros((n, n), np.uint8)
    for i in range(n):
        x, y = divmod(i, cols)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                A[nx * cols + ny, i] = 1          # pressing i toggles light (nx, ny)
    # Row-reduce [A | I] to get a left inverse on the pivot rows and the nullspace.
    M = np.concatenate([A.copy(), np.eye(n, dtype=np.uint8)], 1)
    piv, r = [], 0
    for c in range(n):
        p = next((k for k in range(r, n) if M[k, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for k in range(n):
            if k != r and M[k, c]:
                M[k] ^= M[r]
        piv.append(c)
        r += 1
    free = [c for c in range(n) if c not in piv]
    null = []
    for f in free:                                # nullspace vector for each free column
        v = np.zeros(n, np.uint8)
        v[f] = 1
        for row, c in enumerate(piv):
            v[c] = M[row, f]
        null.append(v)
    null = np.array(null, np.uint8).reshape(len(free), n)
    combos = np.array([[(m >> j) & 1 for j in range(len(free))] for m in range(2 ** len(free))], np.uint8)
    null_span = (combos @ null) % 2 if len(free) else np.zeros((1, n), np.uint8)
    return A, M, piv, null_span


def min_presses(a: np.ndarray, b: np.ndarray, rows: int, cols: int) -> int:
    """Fewest presses turning configuration a into b; -1 if unreachable."""
    A, M, piv, span = solver(rows, cols)
    n = rows * cols
    t = (a.astype(np.uint8) ^ b.astype(np.uint8))
    # Solve A x = t with the reduced system: E A = R where E = M[:, n:].
    E, R = M[:, n:], M[:, :n]
    et = (E @ t) % 2
    r = len(piv)
    if et[r:].any():
        return -1
    x = np.zeros(n, np.uint8)
    for row, c in enumerate(piv):
        x[c] = et[row]
    assert ((A @ x) % 2 == t).all()
    return int(((x[None] ^ span).sum(1)).min())
