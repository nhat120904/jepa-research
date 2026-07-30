"""Belief models: the cheapest things that yield K hypotheses over a hidden grid.

P0 asks whether decision-diversity EXISTS in a learned belief on pixels, not
whether the belief model is good.  So the model is chosen to be the cheapest
thing that can emit multiple hypotheses at all, and the SAME classes serve as
(a) the offline synthetic stand-in the unit tests exercise and (b) the decode
head of the trained network.  Only the source of the per-cell probabilities
differs: a fixed array in the tests, a CNN+GRU forward pass in production.
This is the repo's standing convention -- synthetic validation must test the
production path (`diagnosis/scripts/07_validate_synthetic.py`).

    FactoredBernoulliBelief   K hypotheses from per-cell marginals.  This is the
                              production head: the learned net emits n_cells
                              logits, this class turns them into hypotheses.
    EnsembleBelief            K hypotheses pooled from E member models.  The
                              robustness check that the diversity is not one
                              network's calibration artefact.
    ExactEnumerationBelief    the exact Bayes posterior over an ENUMERATED
                              hidden space.  No training at all.  Available
                              because MineSweeperEasy has only C(16,2) = 120
                              hidden states (verified empirically), so it is
                              the ground-truth reference every learned number
                              is read against.

Both extreme failure modes are reachable by construction, which is what makes
the P0 thresholds testable offline:
  probs -> {0,1}  =>  one hypothesis repeated  =>  M = 1   (VACUOUS)
  probs -> 0.5    =>  K near-independent draws =>  M -> min(K, bound)
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence

import numpy as np

from .hypotheses import HypothesisSet


class BeliefModel:
    """Anything that turns evidence into K weighted hypotheses."""

    name = "belief_model"

    def hypotheses(self, K: int, rng: np.random.Generator) -> HypothesisSet:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Factored Bernoulli -- the production decode head
# --------------------------------------------------------------------------- #
def _log_prob(params: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """log p(z_k) under independent per-cell Bernoulli(probs)."""
    p = np.clip(probs, 1e-9, 1 - 1e-9)
    return (params * np.log(p) + (1 - params) * np.log1p(-p)).sum(axis=1)


class FactoredBernoulliBelief(BeliefModel):
    """K hypotheses drawn from independent per-cell Bernoulli marginals.

    `weighting`:
      "model"    w_k proportional to p(z_k).  PRE-REGISTERED for the ESS check.
                 A peaked posterior then reports a low ESS even when K distinct
                 samples were drawn, which is the conservative reading: it can
                 only make the VACUOUS verdict fire more easily, never less.
      "uniform"  w_k = 1/K (plain Monte-Carlo weighting).  Reported alongside
                 so the ESS number is not an artefact of the weighting choice.

    `mode`:
      "sample"   K ancestral draws (duplicates kept -- see HypothesisSet).
      "beam"     the K highest-probability configurations, found exactly by a
                 K-wide beam over cells (valid because the cells are
                 independent).  Deterministic, so it isolates "K is a knob of
                 the filter" from sampling noise.
    """

    def __init__(self, probs: Sequence[float], weighting: str = "model",
                 mode: str = "sample", name: str = "factored_bernoulli"):
        self.probs = np.clip(np.asarray(probs, dtype=float).reshape(-1), 0.0, 1.0)
        if weighting not in ("model", "uniform"):
            raise ValueError("weighting must be 'model' or 'uniform'")
        if mode not in ("sample", "beam"):
            raise ValueError("mode must be 'sample' or 'beam'")
        self.weighting = weighting
        self.mode = mode
        self.name = name

    @property
    def n_cells(self) -> int:
        return int(self.probs.shape[0])

    # ---------------------------------------------------------------- decode -
    def _beam(self, K: int) -> np.ndarray:
        """Exact top-K configurations of a product of independent Bernoullis."""
        beam = np.zeros((1, 0), dtype=np.int8)
        logp = np.zeros(1)
        p = np.clip(self.probs, 1e-9, 1 - 1e-9)
        for j in range(self.n_cells):
            l0, l1 = np.log1p(-p[j]), np.log(p[j])
            cand = np.concatenate(
                [np.hstack([beam, np.zeros((len(beam), 1), np.int8)]),
                 np.hstack([beam, np.ones((len(beam), 1), np.int8)])]
            )
            cl = np.concatenate([logp + l0, logp + l1])
            keep = np.argsort(-cl)[:K]
            beam, logp = cand[keep], cl[keep]
        return beam

    def hypotheses(self, K: int, rng: Optional[np.random.Generator] = None) -> HypothesisSet:
        if K < 1:
            raise ValueError("K must be >= 1")
        if self.mode == "beam":
            params = self._beam(K)
        else:
            if rng is None:
                rng = np.random.default_rng(0)
            params = (rng.random((K, self.n_cells)) < self.probs).astype(np.int8)

        if self.weighting == "uniform":
            w = np.full(len(params), 1.0 / len(params))
        else:
            lp = _log_prob(params, self.probs)
            lp -= lp.max()
            w = np.exp(lp)
        return HypothesisSet(params=params, weights=w, source=f"{self.name}:{self.mode}")


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
class EnsembleBelief(BeliefModel):
    """Pool K hypotheses evenly across E member models.

    The robustness check for Gate P0: if hypothesis diversity is real it should
    survive when the K draws come from E independently-seeded networks rather
    than one.  (Phase G in this repo's memory is the cautionary case -- an
    ensemble of LoRA seeds over a frozen base shared its blind spot, so the
    disagreement signal was flat.  P0 reports the ensemble number next to the
    single-model one rather than assuming either.)
    """

    def __init__(self, members: Sequence[BeliefModel], name: str = "ensemble"):
        if not members:
            raise ValueError("need at least one member")
        self.members = list(members)
        self.name = name

    def hypotheses(self, K: int, rng: Optional[np.random.Generator] = None) -> HypothesisSet:
        if rng is None:
            rng = np.random.default_rng(0)
        E = len(self.members)
        per = [K // E + (1 if i < K % E else 0) for i in range(E)]
        parts, ws = [], []
        for m, k in zip(self.members, per):
            if k == 0:
                continue
            hs = m.hypotheses(k, rng)
            parts.append(hs.params)
            ws.append(hs.weights * (k / K))     # member-weighted pooling
        return HypothesisSet(
            params=np.concatenate(parts, axis=0),
            weights=np.concatenate(ws),
            source=f"{self.name}[{E}]",
        )


# --------------------------------------------------------------------------- #
# Exact Bayes over an enumerated hidden space -- the reference, no training
# --------------------------------------------------------------------------- #
class ExactEnumerationBelief(BeliefModel):
    """Uniform posterior over every hidden configuration consistent with the
    revealed cells.

    Usable because MineSweeperEasy's hidden space is exactly C(16, 2) = 120
    configurations (4x4 grid, 2 mines; verified: 2000 resets produced exactly
    120 distinct `mine_grid`s).  This gives P0 a ground truth that costs zero
    GPU-hours: the learned model's M/K is only interpretable next to the M/K
    the EXACT posterior produces on the same evidence and the same goal family.

    `revealed` is a dict cell -> bit.  Configurations contradicting it get
    weight 0 and are dropped.
    """

    def __init__(self, n_cells: int, n_positive: int, revealed: Optional[dict] = None,
                 name: str = "exact_bayes"):
        self.n_cells = int(n_cells)
        self.n_positive = int(n_positive)
        self.revealed = dict(revealed or {})
        self.name = name

    def support(self) -> np.ndarray:
        """Every consistent configuration, as an (S, n_cells) int8 array."""
        rows = []
        for combo in itertools.combinations(range(self.n_cells), self.n_positive):
            z = np.zeros(self.n_cells, dtype=np.int8)
            z[list(combo)] = 1
            if all(int(z[c]) == int(b) for c, b in self.revealed.items()):
                rows.append(z)
        if not rows:
            raise ValueError("no hidden configuration is consistent with `revealed`")
        return np.stack(rows)

    def hypotheses(self, K: int, rng: Optional[np.random.Generator] = None) -> HypothesisSet:
        """The first `K` support points, uniformly weighted.

        Truncating rather than sub-sampling is deliberate: when K >= |support|
        this returns the EXACT posterior, which is the reference condition; when
        K < |support| it is an honest "the filter cannot carry the whole
        posterior" condition, which is the situation at visual scale.
        """
        sup = self.support()
        if K < len(sup):
            sup = sup[:K]
        return HypothesisSet(params=sup, weights=np.ones(len(sup)),
                             source=f"{self.name}(|support|={len(self.support())})")


# --------------------------------------------------------------------------- #
# Diagnostic constructors used by the tests and the --dry-run entry point
# --------------------------------------------------------------------------- #
def collapsed(n_cells: int, n_positive: int = 2, seed: int = 0) -> FactoredBernoulliBelief:
    """A maximally CONFIDENT (hence vacuous) belief: marginals pinned to 0/1.

    Every draw returns the same hypothesis, so M = 1 for any goal family.  This
    is the synthetic stand-in for Gate C1 STOP S5 / the VACUOUS failure mode.
    """
    rng = np.random.default_rng(seed)
    p = np.full(n_cells, 1e-6)
    p[rng.choice(n_cells, n_positive, replace=False)] = 1 - 1e-6
    return FactoredBernoulliBelief(p, name="collapsed")


def diffuse(n_cells: int, p: float = 0.5) -> FactoredBernoulliBelief:
    """A maximally UNCERTAIN belief: every cell independent at `p`.

    Nearly every draw is a distinct hypothesis, so M rises to the analytic
    bound.  The synthetic stand-in for the NO-COMPRESSION failure mode when the
    goal family is rich, and for a healthy compressible belief when it is not.
    """
    return FactoredBernoulliBelief(np.full(n_cells, float(p)), name="diffuse")


def partially_informed(n_cells: int, n_positive: int, n_known: int,
                       seed: int = 0) -> FactoredBernoulliBelief:
    """The realistic middle: `n_known` cells resolved safe, the rest sharing the
    remaining probability mass.

    This is what a belief model looks like part-way through an episode, and it
    is the regime P0 actually cares about.

    NOTE the renormalisation, which is not cosmetic.  Leaving the unresolved
    cells at the *unconditional* prior `n_positive / n_cells` makes the
    marginals sum to less than `n_positive`, so nearly every ancestral draw
    comes back all-zero and the hypothesis set looks collapsed for reasons of
    MISCALIBRATION rather than genuine confidence -- which reads as a spurious
    VACUOUS verdict.  `calibration_error` exists to catch exactly this in a
    trained model, and the P0 protocol reports it alongside every M.
    """
    rng = np.random.default_rng(seed)
    p = np.zeros(n_cells)
    known = rng.choice(n_cells, min(n_known, n_cells), replace=False)
    unknown = np.setdiff1d(np.arange(n_cells), known)
    if len(unknown):
        p[unknown] = min(1.0, n_positive / len(unknown))
    p[known] = 1e-6
    return FactoredBernoulliBelief(p, name=f"partial(known={n_known})")


def calibration_error(model: FactoredBernoulliBelief, n_positive: int) -> float:
    """|sum(marginals) - n_positive| -- a mandatory reported diagnostic.

    A factored head whose marginals do not sum to the true number of positives
    produces a degenerate hypothesis set (see `partially_informed`).  P0 must
    never report a VACUOUS verdict without this number next to it: a large
    calibration error means "the head is broken", not "the belief has no
    decision diversity", and those demand completely different responses.
    """
    return float(abs(model.probs.sum() - n_positive))
