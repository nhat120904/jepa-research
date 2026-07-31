# Gate P0 — Step 0 results (substrate + oracle ceiling), 0 GPU-hours

Step 0 is the day-0 check `gateP0_design.md` §6 describes as able to
"kill/retune the config for free". It ran. It did retune the config, and it
surfaced a design defect that would have made the main run uninterpretable.

Everything below is measured, on this login node, with **no GPU and no training**.

## 1. Substrate check — PASS

`belief_compression/.venv` (uv, python 3.11, `popgym-arcade==0.0.7`).

```
diagnosis/.venv/bin/python -m belief_compression.p0.run_p0 --check-substrate
```

6/6 tasks made and stepped. `obs=(128,128,3) uint8`, `nA=5`.

| task | hidden grid | \|Z\| | enumerable |
|---|---|---:|---|
| MineSweeperEasy | 4x4 | **120** | yes |
| MineSweeperMedium | 6x6 | 1,947,792 | no |
| MineSweeperHard | 8x8 | 1.51e11 | no |
| BattleShipEasy | 8x8 | 3.28e12 | no |
| BattleShipMedium | 10x10 | 1.05e15 | no |
| BattleShipHard | 12x12 | 1.04e17 | no |

`MineSweeperEasy`'s |Z| = C(16,2) = 120 is confirmed, so exact Bayes is available
on it for free.

## 2. Oracle signature control — PASS

Distinct decision signatures of the TRUE hidden state, over all 120
configurations. Needs no belief model. `region_size = 4`.

| \|G\| | oracle_sigs | analytic bound 4^\|G\| |
|---:|---:|---:|
| 1 | 3 | 4 |
| 2 | 6 | 16 |
| 3 | 10 | 64 |
| 4 | 15 | 256 |

All >= MIN_ORACLE_SIGS = 3. The task itself carries decision diversity, so a
STOP-VACUOUS verdict later could not be blamed on the task being degenerate.

## 3. DESIGN DEFECT FOUND: `K_REF = 128` is unreachable on the primary task

`measure.PREREG["K_REF"] = 128`, inherited from Gate C1's G1 ("M/K <= 0.25 at
K >= 128"), and the PASS rule requires a witness cell **at exactly that K**.

But `MineSweeperEasy` has **120 hidden states in total**. A belief that respects
the task's support therefore cannot supply 128 distinct hypotheses — measured:

```
K requested:   8    64   120   128   256
n distinct:    8    64   120   120   120
```

So no cell at K = 128 is ever produced, `median(ops at K_REF)` is a median of an
empty sequence, and the verdict function returns `STOP-S3-FILTER-DOMINATES` as a
NaN artefact rather than as a measurement. The full run would have burned ~24
GPU-hours and produced this.

**Why the dry-run did not catch it.** `--dry-run` uses
`FactoredBernoulliBelief`, which samples each cell independently and therefore
draws from 2^16 = 65,536 configurations — including configurations that are not
valid MineSweeper states (they do not have exactly 2 mines). The stand-in lives
in a hypothesis space 546x larger than the real task's. It demonstrated that the
thresholds discriminate, which was its job, but it could not reveal this.

**The tension is structural, not a typo.** The task was chosen *because* it is
exactly enumerable, and it is enumerable *because* |Z| is small. "Exactly
enumerable" and "K >= 128 distinct hypotheses" pull against each other. Any fix
must give up one of them: lower K_REF, or move the K_REF witness to
`BattleShipEasy` (|Z| = 3.28e12) and keep `MineSweeperEasy` only for exact-Bayes
calibration at K <= 120.

## 4. Oracle ceiling — the load-bearing result

The exact Bayes posterior is the **ceiling**: no learned filter on this task can
carry more decision structure than the true posterior does. Running the
pre-registered sweep with `ExactEnumerationBelief` in place of a learned model
therefore bounds P0 from above, at zero cost.

Evidence model: reveal `round(phase * (n - k))` cells drawn uniformly, consistent
with a sampled true configuration. 5 seeds x phases {0.1, 0.25, 0.5, 0.75}.
Criteria as frozen in `measure.PREREG` (A: median M >= 3 and ESS >= 8 and
oracle_sigs >= 3; B: M/K <= 0.25; C: K/bound >= 4 and M <= 1.25*bound).

**At K = 64 — no witness cell exists:**

| \|G\| | bound | median M | M/K | K/bound | A | B | C |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 4 | 2 | 0.031 | 16.00 | **False** | True | True |
| 2 | 16 | 2 | 0.039 | 4.00 | **False** | True | True |
| 3 | 64 | 4 | 0.055 | 1.00 | True | True | **False** |
| 4 | 64 | 4 | 0.070 | 1.00 | True | True | **False** |

**At K = 120 (105 effective after evidence) — a witness exists, barely:**

| \|G\| | bound | median M | M/K | K/bound | A | B | C |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 4 | 2 | 0.019 | 26.25 | **False** | True | True |
| 2 | 16 | **3** | 0.029 | 6.56 | **True** | True | **True** |
| 3 | 64 | 4 | 0.033 | 1.64 | True | True | **False** |
| 4 | 105 | 4 | 0.043 | 1.00 | True | True | **False** |

### The squeeze

The two criteria move in opposite directions in |G|, and the window between them
is one cell wide:

- **Small |G|** -> the bound `4^|G|` is small, so `K/bound` is comfortable (C
  passes) — but the belief only resolves into 2 decision modes, so A fails as
  **vacuous**.
- **Large |G|** -> M rises to 4 and A passes — but the bound grows as `4^|G|` and
  saturates at K, so `K/bound -> 1` and C fails: **the regime the whole method
  needs does not hold**.

Structurally, C requires `4^|G| <= K/4`, i.e. `|G| <= 2` at K = 120. And the
single surviving cell sits **exactly on** the A threshold (median M = 3 versus
MIN_MODES = 3), at the ceiling, where a learned model can only do worse.

### What this means, stated conservatively

This is **not** a STOP verdict for Gate P0. It is a Step-0 finding, and three
things keep it from being decisive:

1. The evidence model here reveals cells directly. Real MineSweeper returns
   neighbour counts, so the true posterior's shape under real observations will
   differ.
2. Only `MineSweeperEasy` was measured. `BattleShipEasy` has |Z| = 3.28e12 and a
   different decision geometry, and is untested.
3. `region_size = 4` and the `RegionCommit` utilities are config, not physics.
   The squeeze is partly a property of that config.

What it does establish is that **the margin at the ceiling is razor-thin and
confined to |G| = 2** on the primary task — and |G| = 2 is a thin goal family for
a programme whose first differentiator is *multi-goal* invariance. Before any
GPU time is spent, the config must be re-tuned so that a witness cell exists with
margin, on a task where the K_REF question is not decided by |Z| < K_REF.

## 5. Status and what is NOT done

- **Steps 1-2 of `gateP0_design.md` §6 are not implemented.** `run_p0.py` says so
  itself: *"No trained belief model is wired in yet."* There is no data
  collection and no amortized-posterior training code. Nothing was submitted to
  Slurm.
- Cluster is live (`main`: 4 x 8xH100-80GB; `mig`: 16 x 3g.40gb slices), login
  node has no GPU.
- `jax` here is CPU-only; `jax[cuda12]` resolves cleanly for the GPU path.

## Reproduce

```
uv venv belief_compression/.venv --python 3.11
uv pip install --python belief_compression/.venv/bin/python popgym-arcade numpy
belief_compression/.venv/bin/python -m belief_compression.p0.run_p0 --check-substrate
```

The ceiling sweep is `scratchpad/p0_oracle_ceiling.py`; it uses only
`belief_compression.p0` and `diagnosis/.venv`.
