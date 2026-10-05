"""Native-success training labels and a reproducible CTA continuation sampler.

Future termination flags are supervision only. ``HitBank.batch`` preserves the
existing feature-bank interface; only ``StratifiedHitPool.sample`` returns those
labels to the training loss. No deployment module imports this helper.
"""

import math
from pathlib import Path

import numpy as np
import torch


class HitBank:
    """Wrap a feature bank with aligned native termination labels from its cache.

    ``done8`` is the simulator's termination flag for the executed candidate
    segment, including early successful termination. The historical field name
    also applies to L=15 caches. Coverage thresholds are never substituted for it.
    """

    def __init__(self, base_bank, feature_path, standard=True):
        self.base = base_bank
        self.feature_path = Path(feature_path)
        self.standard = bool(standard)
        with np.load(self.feature_path / "meta.npz") as meta:
            for key in ("root", "decision", "done8"):
                if key not in meta.files:
                    raise ValueError(f"{base_bank.name}: missing native-label metadata {key}")
            root = np.array(meta["root"], copy=True)
            decision = np.array(meta["decision"], copy=True)
            flags = np.array(meta["done8"], copy=True)
        if not np.array_equal(root, np.asarray(base_bank.root)):
            raise ValueError(f"{base_bank.name}: native-label roots do not align with feature bank")
        if flags.ndim != 2 or len(flags) != len(root) or flags.shape[1] < 2:
            raise ValueError(f"{base_bank.name}: expected done8 shape (N,K>=2), got {flags.shape}")
        if decision.shape != root.shape or not np.isin(flags, (False, True)).all():
            raise ValueError(f"{base_bank.name}: invalid decision IDs or nonbinary done8 flags")
        if not (np.issubdtype(root.dtype, np.integer) and np.issubdtype(decision.dtype, np.integer)):
            raise ValueError(f"{base_bank.name}: root and decision IDs must be integers")
        if (root < 0).any() or (decision < 0).any():
            raise ValueError(f"{base_bank.name}: root and decision IDs must be nonnegative")
        keys = np.stack([root, decision], axis=1)
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError(f"{base_bank.name}: repeated (root, decision) rows")
        if flags.shape != np.asarray(base_bank.geom).shape:
            raise ValueError(f"{base_bank.name}: done8 and geometry bank shapes differ")
        if not 0 < int(base_bank.n) <= len(root):
            raise ValueError(f"{base_bank.name}: invalid active feature-bank size")
        self.decision = decision
        self.hits = flags.astype(np.bool_, copy=False)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def batch(self, indices, device):
        return self.base.batch(indices, device)


def native_hit_loss(scores, hits, temperature=1.0):
    """Negative log probability mass of successful candidates in mixed banks.

    All-success and all-failure banks provide no preference and contribute no
    loss. Reduction averages over mixed banks, so flat banks cannot dilute the
    gradient. Log-sum-exp avoids overflow even when every hit has a low score.
    """
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("native-hit temperature must be finite and positive")
    if scores.ndim != 2 or hits.shape != scores.shape or hits.dtype != torch.bool:
        raise ValueError("native-hit loss requires scores (B,K) and matching boolean hits")
    if not scores.is_floating_point():
        raise ValueError("native-hit scores must be floating point")
    if hits.device != scores.device:
        raise ValueError("native-hit scores and labels must share a device")
    mixed = hits.any(-1) & ~hits.all(-1)
    if not mixed.any():
        return scores.sum() * 0.0
    # Preserve doubles for numerical diagnostics; promote half precision for
    # stable accumulation in the mixed-precision training path.
    s = scores[mixed]
    if s.dtype in (torch.float16, torch.bfloat16):
        s = s.float()
    s = s / temperature
    hit_scores = s.masked_fill(~hits[mixed], -torch.inf)
    return (torch.logsumexp(s, -1) - torch.logsumexp(hit_scores, -1)).mean()


def native_hit_metrics(scores, hits):
    """Capture statistics for bank argmax selection; no privileged inference.

    Use on train/selection predictions. ``hits`` are labels read only after the
    scores have been produced. The first candidate wins exact score ties.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()
    if isinstance(hits, torch.Tensor):
        hits = hits.detach().cpu().numpy()
    scores, hits = np.asarray(scores), np.asarray(hits)
    if scores.ndim != 2 or hits.shape != scores.shape or hits.dtype != np.bool_:
        raise ValueError("native-hit metrics require scores (B,K) and matching boolean hits")
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite native-hit scores")
    eligible = hits.any(-1)
    mixed = eligible & ~hits.all(-1)
    picked = hits[np.arange(len(hits)), scores.argmax(-1)]
    eligible_count, mixed_count = int(eligible.sum()), int(mixed.sum())
    return {
        "banks": int(len(hits)),
        "eligible_banks": eligible_count,
        "mixed_banks": mixed_count,
        "eligible_hits_selected": int((eligible & picked).sum()),
        "mixed_hits_selected": int((mixed & picked).sum()),
        "mixed_misses": int((mixed & ~picked).sum()),
        "eligible_capture": float(picked[eligible].mean()) if eligible_count else None,
        "mixed_capture": float(picked[mixed].mean()) if mixed_count else None,
    }


class StratifiedHitPool:
    """Sample standard-heavy banks with explicit mixed-success replay mass.

    Sampling is with replacement. Each draw first chooses a standard/perturbed
    stratum, then mixed-success replay or the old informative/all mixture. Thus
    ``mixed_frac`` is the replay component mass, while the actual mixed exposure
    can be higher because ordinary components also include mixed banks. Exact
    per-row probabilities and fallback behavior are disclosed in ``report``.
    """

    def __init__(self, banks, standard_mass=0.8, mixed_frac=0.25, spread_frac=0.75):
        self.banks = list(banks)
        if not self.banks:
            raise ValueError("native-hit training pool is empty")
        if len({b.name for b in self.banks}) != len(self.banks):
            raise ValueError("native-hit bank names must be unique")
        self.standard_mass = self._fraction(standard_mass, "standard_mass")
        self.mixed_frac = self._fraction(mixed_frac, "mixed_frac")
        self.spread_frac = self._fraction(spread_frac, "spread_frac")
        sizes = {b.hits.shape[1] for b in self.banks}
        if len(sizes) != 1:
            raise ValueError("native-hit banks have different candidate counts")
        self.k = sizes.pop()
        self.all = self._rows()
        self.inf = self._rows(informative=True)
        self.components = []
        for standard, mass in ((True, self.standard_mass), (False, 1 - self.standard_mass)):
            if mass == 0:
                continue
            all_rows = self._rows(standard=standard)
            if not len(all_rows):
                raise ValueError(f"positive sampling mass for absent {'standard' if standard else 'perturbed'} banks")
            mixed = self._rows(standard=standard, mixed=True)
            informative = self._rows(standard=standard, informative=True)
            self._component(standard, "mixed", mass * self.mixed_frac, mixed, all_rows)
            self._component(standard, "informative", mass * (1 - self.mixed_frac) * self.spread_frac,
                            informative, all_rows)
            self._component(standard, "all", mass * (1 - self.mixed_frac) * (1 - self.spread_frac),
                            all_rows, all_rows)
        self.component_mass = np.array([x["mass"] for x in self.components], np.float64)
        self.component_mass /= self.component_mass.sum()
        self.row_probability = [np.zeros(b.n, np.float64) for b in self.banks]
        for component, mass in zip(self.components, self.component_mass):
            for j in np.unique(component["rows"][:, 0]):
                rows = component["rows"][component["rows"][:, 0] == j, 1]
                self.row_probability[j][rows] += mass / len(component["rows"])

    @staticmethod
    def _fraction(value, name):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0,1]")
        return value

    def _rows(self, standard=None, mixed=False, informative=False):
        chunks = []
        for j, bank in enumerate(self.banks):
            if standard is not None and bank.standard != standard:
                continue
            rows = np.arange(bank.n)
            if mixed:
                hits = bank.hits[:bank.n]
                rows = rows[hits.any(-1) & ~hits.all(-1)]
            if informative:
                spread = np.asarray(bank.spread)
                if ((spread < 0) | (spread >= bank.n)).any():
                    raise ValueError(f"{bank.name}: informative row outside active bank")
                rows = np.intersect1d(rows, spread)
            if len(rows):
                chunks.append(np.stack([np.full(len(rows), j, np.int64), rows], 1))
        return np.concatenate(chunks) if chunks else np.empty((0, 2), np.int64)

    def _component(self, standard, name, mass, rows, fallback):
        if mass <= 0:
            return
        self.components.append({"standard": standard, "kind": name, "mass": mass,
                                "fallback_to_all": not bool(len(rows)),
                                "rows": rows if len(rows) else fallback})

    def sample_rows(self, rng, decisions):
        if int(decisions) != decisions or decisions <= 0:
            raise ValueError("decisions must be a positive integer")
        decisions = int(decisions)
        component = rng.choice(len(self.components), size=decisions, p=self.component_mass)
        rows = np.empty((decisions, 2), np.int64)
        for j in np.unique(component):
            mask = component == j
            source = self.components[j]["rows"]
            rows[mask] = source[rng.integers(0, len(source), int(mask.sum()))]
        return rows[np.lexsort((rows[:, 1], rows[:, 0]))]

    def sample(self, rng, decisions, device, return_rows=False):
        rows = self.sample_rows(rng, decisions)
        parts, hits = [], []
        for j in np.unique(rows[:, 0]):
            indices = rows[rows[:, 0] == j, 1]
            parts.append(self.banks[j].batch(indices, device))
            hits.append(torch.as_tensor(self.banks[j].hits[indices], device=device, dtype=torch.bool))
        def concatenate(values):
            if isinstance(values[0], dict):
                return {key: torch.cat([v[key] for v in values]) for key in values[0]}
            return torch.cat(values)
        batch = tuple(concatenate([p[t] for p in parts]) for t in range(4))
        result = (*batch, torch.cat(hits))
        return (*result, rows) if return_rows else result

    def assert_disjoint(self, selection_banks, forbidden_roots=None):
        train = set(np.concatenate([np.asarray(b.root)[:b.n] for b in self.banks]).tolist())
        selection = set(np.concatenate([np.asarray(b.root)[:b.n] for b in selection_banks]).tolist()) if selection_banks else set()
        overlap = train & selection
        if overlap:
            raise ValueError(f"train/selection root overlap: {sorted(overlap)[:10]}")
        if forbidden_roots is not None:
            overlap = (train | selection) & set(forbidden_roots)
            if overlap:
                raise ValueError(f"evaluation roots in train/selection: {sorted(overlap)[:10]}")
        return {"train_roots": len(train), "selection_roots": len(selection), "disjoint": True}

    def report(self, decisions=None):
        if decisions is not None and (int(decisions) != decisions or decisions <= 0):
            raise ValueError("reported decisions must be a positive integer")
        probabilities = self.row_probability
        by_bank = {}
        for bank, p in zip(self.banks, probabilities):
            mixed = bank.hits[:bank.n].any(-1) & ~bank.hits[:bank.n].all(-1)
            by_bank[bank.name] = {
                "banks": int(bank.n), "roots": int(len(np.unique(np.asarray(bank.root)[:bank.n]))),
                "standard": bank.standard, "mass": float(p.sum()),
                "mixed_banks": int(mixed.sum()), "mixed_exposure": float(p[mixed].sum()),
                "rows_with_nonzero_probability": int(np.count_nonzero(p)),
                "row_probability_min": float(p.min()), "row_probability_max": float(p.max()),
            }
            if decisions is not None:
                # Probability a row is seen at least once in an iid batch.
                with np.errstate(divide="ignore"):
                    inclusion = -np.expm1(int(decisions) * np.log1p(-p))
                by_bank[bank.name]["expected_distinct_rows_per_batch"] = float(inclusion.sum())
        return {
            "standard_mass": self.standard_mass, "mixed_replay_mass": self.mixed_frac,
            "spread_frac_within_nonreplay": self.spread_frac, "replacement": True,
            "expected_standard_exposure": sum(d["mass"] for d in by_bank.values() if d["standard"]),
            "expected_mixed_exposure": sum(d["mixed_exposure"] for d in by_bank.values()),
            "banks": by_bank,
            "components": [{k: v for k, v in c.items() if k != "rows"} | {"rows": int(len(c["rows"]))}
                           for c in self.components],
        }
