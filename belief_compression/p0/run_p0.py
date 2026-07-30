"""Gate P0 entry point.

    # offline dry run -- no jax, no GPU, no trained model; exercises the whole
    # measurement + verdict path against synthetic stand-in belief models
    diagnosis/.venv/bin/python -m belief_compression.p0.run_p0 --dry-run

    # substrate check -- confirms POPGym Arcade installs, makes and steps
    <p0venv>/bin/python -m belief_compression.p0.run_p0 --check-substrate

    # the real measurement, once belief models are trained (Gate P0 §7)
    <p0venv>/bin/python -m belief_compression.p0.run_p0 \
        --task MineSweeperEasy --ckpt runs/p0/msE/belief.pt --out results/p0/

`--dry-run` deliberately runs the SAME `measure.sweep` / `measure.verdict` code
as the real run, with the belief model swapped for a synthetic stand-in.  That
is what makes the pre-registered thresholds testable before any training
starts, and it is the repo's standing convention (validate metrics on synthetic
models before trusting a real number).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

from . import belief as B
from . import measure as MZ
from .decision import RegionCommit
from .envs import TASKS

# Pre-registered sweep grid (frozen with the thresholds).
K_VALUES = (8, 16, 32, 64, 128, 256)
REGION_SIZE = 4
# Goal-family richness.  With |g| = 4 the analytic bound prod_g|g| = 4^|G|
# sweeps 4, 16, 64, 256, 4096 -- i.e. it crosses K_REF = 128 between |G| = 3
# and |G| = 4, so the sweep brackets the point at which compression MUST stop
# working.  That crossing is the content of deliverable (c).
GOAL_RICHNESS = (1, 2, 3, 4, 6)
# Fraction of the episode elapsed when the belief is read.  A belief model is
# diffuse early and nearly resolved late, and M moves with it, so a
# single-point measurement would pick an arbitrary operating point.  All phases
# are reported; the PASS gate needs a witness at ANY ONE of them, which is the
# honest reading -- the planner runs at every step, so the question is whether
# the regime exists at some operating point, not at all of them.
PHASES = (0.1, 0.25, 0.5, 0.75)
N_SEEDS = 5


def goal_specs(n_cells: int, region_size: int = REGION_SIZE):
    """(n_goals, region_size) pairs that fit as DISJOINT regions on this board.

    Regions are kept disjoint (see `decision.RegionCommit.goal_family`) because
    overlapping regions inflate signature agreement for free, so richness is
    capped at n_cells // region_size.  MineSweeperEasy (16 cells) therefore
    reaches |G| = 4 and BattleShipEasy (64 cells) reaches |G| = 6.
    """
    cap = n_cells // region_size
    return tuple((g, region_size) for g in GOAL_RICHNESS if g <= cap)


# --------------------------------------------------------------------------- #
def check_substrate() -> int:
    """Make and step every P0 task; print what actually works."""
    from . import envs
    if not envs.available():
        print("popgym_arcade NOT importable in this interpreter.")
        return 1
    import jax
    print("jax devices:", jax.devices())
    ok = 0
    for name, spec in TASKS.items():
        try:
            env, ps = envs.make(name, partial_obs=True)
            o, st = env.reset(jax.random.PRNGKey(0), ps)
            o2, st2, r, d, _ = env.step(jax.random.PRNGKey(1), st, 4, ps)
            p = envs.oracle_params(st2, name)
            print(f"  OK  {name:20s} obs={tuple(o.shape)} nA={env.action_space(ps).n} "
                  f"hidden={spec.grid} |Z|={spec.hidden_cardinality:,} "
                  f"enumerable={spec.enumerable} n_pos={int(p.sum())}")
            ok += 1
        except Exception as e:  # pragma: no cover - substrate-dependent
            print(f"  FAIL {name:20s} {type(e).__name__}: {e}")
    print(f"{ok}/{len(TASKS)} tasks made and stepped.")
    return 0 if ok == len(TASKS) else 1


# --------------------------------------------------------------------------- #
def dry_run(out_dir: str | None = None, n_seeds: int = N_SEEDS) -> tuple:
    """Full measurement + verdict on synthetic stand-in belief models.

    Three synthetic regimes are run so the operator can see, before committing
    any GPU time, that the pre-registered thresholds actually discriminate:

      collapsed   -> must return STOP-VACUOUS
      diffuse     -> a maximally uncertain filter
      partial     -> the realistic mid-episode belief
    """
    results = {}
    for label, spec_name, factory in (
        # A filter that has already collapsed onto one hypothesis, at any phase.
        ("collapsed", "MineSweeperEasy",
         lambda s, ph, rng, n=16, k=2: B.collapsed(n, k, seed=s)),
        # A filter that never resolves anything.
        ("diffuse", "MineSweeperEasy",
         lambda s, ph, rng, n=16: B.diffuse(n, p=0.5)),
        # The realistic case: information accumulates with the episode phase,
        # so early phases are diffuse and late ones nearly resolved.
        ("partial", "MineSweeperEasy",
         lambda s, ph, rng, n=16, k=2: B.partially_informed(
             n, k, n_known=int(round(ph * (n - k))), seed=s)),
    ):
        spec = TASKS[spec_name]
        dec = RegionCommit.build(spec.n_cells, target_bit=spec.target_bit, seed=0)
        rows = MZ.sweep(factory, dec, K_VALUES, goal_specs(spec.n_cells),
                        task=spec_name, n_seeds=n_seeds, phases=PHASES)
        # Oracle-side control (Gate C1 STOP S5): how many distinct decision
        # signatures the TRUE hidden state produces.  Needs no belief model, so
        # it runs first and costs nothing.
        exact = B.ExactEnumerationBelief(spec.n_cells, spec.n_positive)
        true_params = exact.support()
        osig = {
            (g, rs): MZ.oracle_signature_count(
                true_params, dec, dec.goal_family(g, rs, seed=0))
            for g, rs in goal_specs(spec.n_cells)
        }
        summ = MZ.summarise(rows, spec_name, oracle_sigs=osig)
        # Cost split stand-in: a small CNN belief forward vs the measured
        # compress() ops.  Replaced by wall-clock on GPU in the real run.
        c_search = float(np.median([r.ops for r in rows if r.K == MZ.PREREG["K_REF"]]))
        cost = MZ.CostSplit(c_belief=0.30 * c_search, c_search=c_search, unit="ops(stand-in)")
        v, why = MZ.verdict([summ], rows, cost)
        results[label] = dict(verdict=v, reasons=why, oracle_sigs=osig,
                              n_rows=len(rows),
                              cells=[r.as_dict() for r in rows])
        print(f"\n=== dry-run [{label}] -> {v}")
        for line in why:
            print("   ", line)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "p0_dry_run.json")
        with open(p, "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "cells"}
                       for k, v in results.items()}, f, indent=2)
        print(f"\nwrote {p}")
    return results


# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gate P0 diversity smoke test")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic stand-in belief models; no jax, no GPU")
    ap.add_argument("--check-substrate", action="store_true",
                    help="make and step every POPGym Arcade task")
    ap.add_argument("--out", default=None, help="output directory for JSON")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args(argv)

    if a.check_substrate:
        return check_substrate()
    if a.dry_run:
        dry_run(a.out, n_seeds=a.seeds)
        return 0
    ap.print_help()
    print("\nNo trained belief model is wired in yet -- that is Gate P0 §7 step 3. "
          "Use --dry-run to exercise the measurement and the pre-registered "
          "thresholds offline, or --check-substrate to verify the environment.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
