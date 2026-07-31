"""Gate P0 step 0: the ORACLE CEILING, at zero GPU-hours.

Runs the pre-registered P0 sweep with the EXACT Bayes posterior in place of a
learned belief model.  MineSweeperEasy's hidden space is exactly C(16,2)=120,
so this is not an approximation -- it is the true posterior given the revealed
evidence.

Why this matters: the exact posterior is the CEILING on decision diversity and
compressibility for any filter on this task.  A learned model cannot carry more
decision structure than the true posterior has.  So if exact Bayes fails the
pre-registered thresholds, Gate P0 is dead before a single GPU-hour is spent,
and the config must be retuned (or the task changed) instead.
"""
import numpy as np

from belief_compression.p0 import belief as B
from belief_compression.p0 import measure as MZ
from belief_compression.p0.decision import RegionCommit
from belief_compression.p0.envs import TASKS
from belief_compression.p0.run_p0 import (PHASES, N_SEEDS, goal_specs)
K_VALUES = (8, 16, 32, 64, 120)

TASK = "MineSweeperEasy"
spec = TASKS[TASK]
N, KPOS = spec.n_cells, spec.n_positive


def exact_factory(seed, phase, rng, n=N, k=KPOS):
    """The true posterior after revealing a phase-dependent amount of evidence."""
    rs = np.random.default_rng(1000 + seed)
    mines = rs.choice(n, size=k, replace=False)
    truth = np.zeros(n, dtype=int)
    truth[mines] = 1
    n_known = int(round(phase * (n - k)))
    order = rs.permutation(n)
    revealed = {int(c): int(truth[c]) for c in order[:n_known]}
    return B.ExactEnumerationBelief(n, k, revealed=revealed)


def main():
    dec = RegionCommit.build(spec.n_cells, target_bit=spec.target_bit, seed=0)
    gspecs = goal_specs(spec.n_cells)

    # ---- task-side control: how much decision diversity does the TRUE hidden
    # state carry at all?  No belief model involved.
    exact_full = B.ExactEnumerationBelief(spec.n_cells, spec.n_positive)
    true_params = exact_full.support()
    print(f"task={TASK}  |Z|={len(true_params)}  n_cells={N}  n_positive={KPOS}")
    print(f"goal_specs={gspecs}\n")

    osig = {}
    print("ORACLE SIGNATURE CONTROL (distinct decision signatures of the true state)")
    print(f"{'|G|':>4} {'region':>7} {'oracle_sigs':>12} {'bound':>8}")
    for g, rs_ in gspecs:
        fam = dec.goal_family(g, rs_, seed=0)
        n_sig = MZ.oracle_signature_count(true_params, dec, fam)
        osig[(g, rs_)] = n_sig
        print(f"{g:>4} {rs_:>7} {n_sig:>12} {rs_ ** g:>8}")

    # ---- the ceiling sweep: exact posterior instead of a learned model
    print("\nORACLE-CEILING SWEEP (exact Bayes posterior as the belief model)")
    rows = MZ.sweep(exact_factory, dec, K_VALUES, gspecs,
                    task=TASK, n_seeds=N_SEEDS, phases=PHASES)
    summ = MZ.summarise(rows, TASK, oracle_sigs=osig)

    c_search = float(np.median([r.ops for r in rows if r.K == 120]))
    cost = MZ.CostSplit(c_belief=0.30 * c_search, c_search=c_search,
                        unit="ops(stand-in)")
    v, why = MZ.verdict([summ], rows, cost)

    print(f"\n{'|G|':>4} {'phase':>6} {'K':>5} {'M':>5} {'M/K':>7} {'ESS':>7}")
    for r in sorted(rows, key=lambda r: (r.n_goals, r.phase, r.K)):
        if r.K in (64, 120):
            print(f"{r.n_goals:>4} {r.phase:>6} {r.K:>5} {r.M:>5.0f} "
                  f"{r.M / r.K:>7.4f} {r.ess:>7.1f}")

    print(f"\n=== ORACLE CEILING VERDICT -> {v}")
    for line in why:
        print("   ", line)
    print("\nNOTE: this is the CEILING. A learned belief model cannot exceed it.")
    return v


if __name__ == "__main__":
    main()
