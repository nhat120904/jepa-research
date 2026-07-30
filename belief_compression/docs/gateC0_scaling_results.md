# Gate C0 — Scaling Study

Does decision-equivalent belief compression buy an ASYMPTOTIC advantage, or only the constant factor Gate B measured?

Reproduce: `diagnosis/.venv/bin/python -m belief_compression.scaling` (from repo root). Everything below is exact (exact Bayes, exact expectimax, exact expected return by enumeration) and was produced by that command. Cells the exact enumerator could not afford are printed `n/a`, never estimated.

## The question

The value proposition is *"full-belief planning becomes intractable as the problem scales; decision-mode planning stays cheap while preserving the decision"*. That requires **M bounded while K grows**. If instead M grows with K, the win is a constant factor and the claim is weak.

### Compute-accounting fix (disclosed)

`Belief.updated` previously charged the compute counter `K` likelihood touches even for a belief whose support was only `M` particles, which would have masked exactly the effect this study measures. It now charges (and evaluates) the support only — a filter carrying M particles does not evaluate the K-M particles it does not carry, and zero-weight particles stay zero under Bayes, so the posterior is unchanged. Re-running Gate B after the fix leaves every return, regret and verdict identical (still GO); only two compute numbers move, both for `amortized_voi` (mass_sort 222 -> 198, occluder_push 110 -> 74). Full-belief numbers are unchanged because its support is all of K.

## S1 (headline) — does M saturate as belief resolution K grows?

`grid_param` has a CONTINUOUS hidden parameter; `K = resolution` is purely how finely the belief discretizes it. The goal family (controller centres) lives in continuous parameter space and is untouched by the grid, so refining K cannot create new optimal actions. `bound` is the analytic prediction `min(K, |G|*(|A|-1)+1)` = number of cells in the arrangement of all goals' decision boundaries. `regret_commit` is the exact commit-decision regret vs the full-belief oracle (budget 0), maximised over goals in the family.

```
                 config     K  M     M/K  bound  |G|  regret_commit
-----------------------  ----  -  ------  -----  ---  -------------
grid_param(|G|=1,|A|=3)     6  3  0.5000      3    1       0.00e+00
grid_param(|G|=1,|A|=3)    12  3  0.2500      3    1       0.00e+00
grid_param(|G|=1,|A|=3)    24  3  0.1250      3    1       0.00e+00
grid_param(|G|=1,|A|=3)    48  3  0.0625      3    1       0.00e+00
grid_param(|G|=1,|A|=3)    96  3  0.0312      3    1       0.00e+00
grid_param(|G|=1,|A|=3)   192  3  0.0156      3    1       0.00e+00
grid_param(|G|=1,|A|=3)   384  3  0.0078      3    1            n/a
grid_param(|G|=1,|A|=3)   768  3  0.0039      3    1            n/a
grid_param(|G|=1,|A|=3)  1536  3  0.0020      3    1            n/a
grid_param(|G|=2,|A|=3)     6  5  0.8333      5    2       0.00e+00
grid_param(|G|=2,|A|=3)    12  5  0.4167      5    2       0.00e+00
grid_param(|G|=2,|A|=3)    24  5  0.2083      5    2       0.00e+00
grid_param(|G|=2,|A|=3)    48  5  0.1042      5    2       0.00e+00
grid_param(|G|=2,|A|=3)    96  5  0.0521      5    2       0.00e+00
grid_param(|G|=2,|A|=3)   192  5  0.0260      5    2       0.00e+00
grid_param(|G|=2,|A|=3)   384  5  0.0130      5    2            n/a
grid_param(|G|=2,|A|=3)   768  5  0.0065      5    2            n/a
grid_param(|G|=2,|A|=3)  1536  5  0.0033      5    2            n/a
grid_param(|G|=4,|A|=3)     6  5  0.8333      6    4       0.00e+00
grid_param(|G|=4,|A|=3)    12  9  0.7500      9    4       0.00e+00
grid_param(|G|=4,|A|=3)    24  9  0.3750      9    4       0.00e+00
grid_param(|G|=4,|A|=3)    48  9  0.1875      9    4       0.00e+00
grid_param(|G|=4,|A|=3)    96  9  0.0938      9    4       0.00e+00
grid_param(|G|=4,|A|=3)   192  9  0.0469      9    4       0.00e+00
grid_param(|G|=4,|A|=3)   384  9  0.0234      9    4            n/a
grid_param(|G|=4,|A|=3)   768  9  0.0117      9    4            n/a
grid_param(|G|=4,|A|=3)  1536  9  0.0059      9    4            n/a
```

Same refinement story on the pre-existing occluder task (more locations, same fixed number of push targets):

```
               config    K  M     M/K  bound  |G|  regret_commit
---------------------  ---  -  ------  -----  ---  -------------
occluder_push(bins=2)    8  2  0.2500      2    1       0.00e+00
occluder_push(bins=2)   16  2  0.1250      2    1       0.00e+00
occluder_push(bins=2)   32  2  0.0625      2    1       0.00e+00
occluder_push(bins=2)   64  2  0.0312      2    1       0.00e+00
occluder_push(bins=2)  128  2  0.0156      2    1       0.00e+00
occluder_push(bins=2)  256  2  0.0078      2    1            n/a
```

**Control axis** (this is what stops S1 being a tautology): here K grows because new decision-RELEVANT latent dimensions are added, not because one parameter is resolved more finely. Every object matters to some goal in the family.

```
                      config    K    M     M/K  bound  |G|  regret_commit
----------------------------  ---  ---  ------  -----  ---  -------------
mass_sort(N=2, all-relevant)    4    4  1.0000      -    2       0.00e+00
mass_sort(N=3, all-relevant)    8    8  1.0000      -    3       0.00e+00
mass_sort(N=4, all-relevant)   16   16  1.0000      -    4       0.00e+00
mass_sort(N=5, all-relevant)   32   32  1.0000      -    5       0.00e+00
mass_sort(N=6, all-relevant)   64   64  1.0000      -    6       0.00e+00
mass_sort(N=7, all-relevant)  128  128  1.0000      -    7       0.00e+00
mass_sort(N=8, all-relevant)  256  256  1.0000      -    8            n/a
```

```
Log-log slope of M vs K (0 = perfectly saturated, 1 = M grows like K):

     axis  K range  M range  M/K @ Kmax  slope(logM/logK)  saturated(last 3)  max |regret|
---------  -------  -------  ----------  ----------------  -----------------  ------------
grid|G|=1   6-1536      3-3      0.0020            0.0000               True      0.00e+00
grid|G|=2   6-1536      5-5      0.0033            0.0000               True      0.00e+00
grid|G|=4   6-1536      5-9      0.0059            0.0565               True      0.00e+00
 occluder    8-256      2-2      0.0078            0.0000               True      0.00e+00
mass_sort    4-256    4-256      1.0000            1.0000              False      0.00e+00
```

## S2 — planning compute vs belief resolution K (horizon H=1)

Root-decision compute (total primitive ops from the existing `ComputeCounter`: reward evals + belief touches + expectimax nodes + VOI inner calls). `compression` builds the decision signatures at decision time (O(K|G||A|) reward evals); `compression_cached` reuses signatures precomputed once per task/goal family, so its per-decision cost is O(K) bucketing + O(M x tree). `same_root` = the compressed planner chose the same root action as the full-belief oracle. Exact regret is computed for K <= 192 (beyond that the exact enumerator over K hidden states x observation trees is too slow; reported n/a).

```
   K  M     M/K  C_full  C_comp  C_cached  C_voi  C_amort  full/cached  ret_full  regret_cached  rel_regret  same_root
----  -  ------  ------  ------  --------  -----  -------  -----------  --------  -------------  ----------  ---------
  12  3  0.2500     708     222       198    707      221       3.5758    0.7433       0.00e+00       0.00%       True
  24  3  0.1250    1404     258       210   1403      257       6.6857    0.7482       0.00e+00       0.00%       True
  48  3  0.0625    2796     330       234   2795      329      11.9487    0.7482       8.23e-03       1.10%       True
  96  3  0.0312    5580     474       282   5579      473      19.7872    0.7482       8.23e-03       1.10%       True
 192  3  0.0156   11148     762       378  11147      761      29.4921    0.7482       8.23e-03       1.10%       True
 384  3  0.0078   22284    1338       570  22283     1337      39.0947       n/a            n/a         n/a       True
 768  3  0.0039   44556    2490       954  44555     2489      46.7044       n/a            n/a         n/a       True
1536  3  0.0020   89100    4794      1722  89099     4793      51.7422       n/a            n/a         n/a       True
```

```
log-log slopes of root compute vs K (H=1):
  full_belief         0.997
  compression         0.644
  compression_cached  0.441
  amortized_voi       0.645
  full/cached ratio   3.58x at K=12  ->  51.74x at K=1536
```

## S3 — planning compute vs horizon H (K=192 fixed)

Full-belief expectimax expands the same tree as the compressed planner, but pays O(K) per node instead of O(M). Exact closed-loop regret by enumeration is only affordable for H <= 1; for deeper horizons we report whether the compressed planner picked the SAME root action as the full-belief oracle, which is a necessary condition for zero root regret (and is cheap).

```
H    K  M    C_full  C_cached  full/cached  wall_full_s  wall_cached_s  regret_cached  same_root
-  ---  -  --------  --------  -----------  -----------  -------------  -------------  ---------
0  192  3       577       202       2.8564       0.0003         0.0001       0.00e+00       True
1  192  3     11148       378      29.4921       0.0050         0.0010       8.23e-03       True
2  192  3    127429      2314      55.0687       0.0579         0.0084            n/a       True
3  192  3   1406520     23610      59.5731       0.6566         0.0929            n/a       True
4  192  3  15476521    257866      60.0177       7.0904         1.0024            n/a       True
```

### S3b — the compute ratio `C_full / C_cached` over the joint (K, H) grid

This is the honest two-sided picture, and it is the most load-bearing table in the study. Reading DOWN a column (fixed H, growing K) the ratio flattens: the compressed planner still has to READ the K-particle belief once per decision to bucket it, an unavoidable `O(K)` term, so at fixed depth the ratio is capped by the size of the search tree. Reading ACROSS a row (fixed K, growing H) the ratio climbs toward `K/M`, because the `O(K)` read is amortized over an exponentially growing tree in which every node costs `O(M)` instead of `O(K)`. `K/M` is the ceiling, and `K/M` is unbounded in K. Cells whose full-belief cost would exceed 4e7 primitive ops were skipped, not estimated.

```
   K  M       K/M     H=0      H=1       H=2       H=3       H=4
----  -  --------  ------  -------  --------  --------  --------
  24  3    8.0000  2.1471   6.6857    7.4767    7.5546    7.5618
  96  3   32.0000  2.7264  19.7872   28.7561   29.9393   30.0513
 384  3  128.0000  2.9264  39.0947  101.6460  118.1235  119.8837
1536  3  512.0000  2.9812  51.7422  278.4311  450.5054       n/a
```

## S4 — terminal action-set size |A| (K=384, H=1)

M is expected to track |A| (one mode per optimal controller), i.e. the bound `|G|*(|A|-1)+1`, and to stay independent of K.

```
|A|    K   M     M/K  bound  C_full  C_cached  full/cached  same_root
---  ---  --  ------  -----  ------  --------  -----------  ---------
  2  384   2  0.0052      2   17676       488      36.2213      False
  3  384   3  0.0078      3   22284       570      39.0947       True
  4  384   4  0.0104      4   26892       676      39.7811       True
  6  384   6  0.0156      6   36108       960      37.6125       True
  8  384   8  0.0208      8   45324      1340      33.8239       True
 12  384  12  0.0312     12   63756      2388      26.6985       True
```

## S5 — goal-family richness x belief resolution

M as a joint function of (|G|, K), |A|=3. The Gate-B finding was that M grows with goal richness; the Gate-C0 question is whether, at each fixed richness, M is flat in K.

```
|G|  bound  K=24  K=96  K=384  K=1536
---  -----  ----  ----  -----  ------
  1      3     3     3      3       3
  2      5     5     5      5       5
  4      9     9     9      9       9
  8     17    17    17     17      17
```

## S6 — the limitation this study found: decision fidelity != VALUE fidelity

S1/S2 show the collapse is **exactly lossless for the terminal commit** (regret 0.00e+00 at budget 0, every K). But a planner that can PROBE compares the value of committing now against the expected value of committing after an observation, and decision-mode compression preserves the *argmax* of the commit, not its *value*. Collapsing a mode onto one representative particle therefore biases the value estimate, and the bias direction depends on which particle you pick:

`V_maxweight` uses the current implementation (max-weight member, which under a flat prior is an arbitrary mode-edge particle) and systematically UNDER-values; the mode-conditional-mean particle systematically OVER-values. Neither is value-consistent, so this is structural, not a bad choice of representative. `root_full` / `root_maxw` are the root actions; where they disagree the compressed planner skipped a probe the oracle took.

```
|A|    K   M  V_full  V_maxweight  V_centroid     root_full     root_maxw  same_root  regret_H1    rel
---  ---  --  ------  -----------  ----------  ------------  ------------  ---------  ---------  -----
  2   96   2  0.6786       0.5000      0.9115  probe:sense8      commit:0      False   3.04e-01  44.7%
  2  384   2  0.6786       0.5000      0.9190  probe:sense8      commit:0      False        n/a    n/a
  3   96   3  0.7482       0.6108      0.9114  probe:sense8  probe:sense8       True   8.23e-03   1.1%
  3  384   3  0.7482       0.6048      0.9189  probe:sense8  probe:sense8       True        n/a    n/a
  4   96   4  0.7946       0.6903      0.9080  probe:sense8  probe:sense8       True   2.77e-02   3.5%
  4  384   4  0.7946       0.6843      0.9154  probe:sense8  probe:sense8       True        n/a    n/a
  6   96   6  0.8210       0.7666      0.9053  probe:sense8  probe:sense8       True   2.04e-02   2.5%
  6  384   6  0.8210       0.7607      0.9127  probe:sense8  probe:sense8       True        n/a    n/a
  8   96   8  0.8531       0.8036      0.9026  probe:sense8  probe:sense8       True   0.00e+00   0.0%
  8  384   8  0.8531       0.7977      0.9100  probe:sense8  probe:sense8       True        n/a    n/a
 12   96  12  0.8462       0.8363      0.8561  probe:sense8  probe:sense8       True   2.58e-02   3.0%
 12  384  12  0.8462       0.8345      0.8593  probe:sense8  probe:sense8       True        n/a    n/a
```

## S7 — is the S6 failure intrinsic, or an artefact of the representative?

S6 established that the collapsed belief mis-values and can therefore pick the wrong root action. That is only damning if it survives a better choice of representative particle. It largely does not. The mode PARTITION is identical under both rules — same modes, same M, and (verified in the last column) byte-identical compute — so the only thing that changes is which particle carries a mode's weight, and any improvement here is FREE.

`maxweight` is the original rule: the mode's highest-weight member, which under a flat prior ties across the whole mode and resolves to the FIRST index — an arbitrary particle at the mode's EDGE. `centroid` instead picks the particle nearest the mode's belief-weighted mean parameter.

```
|A|  |G|    K   M  rel_regret_maxweight  rel_regret_centroid  C_maxweight  C_centroid  same_compute
---  ---  ---  --  --------------------  -------------------  -----------  ----------  ------------
  2    1   24   2                44.74%                0.00%          128         128          True
  2    1   48   2                44.74%                0.00%          152         152          True
  2    1   96   2                44.74%                0.00%          200         200          True
  2    1  192   2                44.74%                0.00%          296         296          True
  2    2   24   3                10.83%                0.00%          174         174          True
  2    2   48   3                10.83%                0.00%          198         198          True
  2    2   96   3                10.83%                0.00%          246         246          True
  2    2  192   3                10.83%                0.00%          342         342          True
  3    1   24   3                 0.00%                0.00%          210         210          True
  3    1   48   3                 1.10%                0.00%          234         234          True
  3    1   96   3                 1.10%                0.00%          282         282          True
  3    1  192   3                 1.10%                0.00%          378         378          True
  3    2   24   5                 0.00%                0.00%          326         326          True
  3    2   48   5                 1.10%                0.00%          350         350          True
  3    2   96   5                 1.10%                0.00%          398         398          True
  3    2  192   5                 1.10%                0.00%          494         494          True
  4    1   24   4                 0.00%                0.00%          316         316          True
  4    1   48   4                 0.00%                0.00%          340         340          True
  4    1   96   4                 3.48%                0.00%          388         388          True
  4    1  192   4                 3.48%                0.00%          484         484          True
  4    2   24   7                 0.00%                0.00%          526         526          True
  4    2   48   7                 0.00%                0.00%          550         550          True
  4    2   96   7                 3.50%                0.00%          598         598          True
  4    2  192   7                 7.23%                0.00%          694         694          True
  6    1   24   6                 0.00%                0.00%          600         600          True
  6    1   48   6                 2.49%                0.00%          624         624          True
  6    1   96   6                 2.49%                0.00%          672         672          True
  6    1  192   6                 2.49%                0.00%          768         768          True
  6    2   24  11                 0.00%                0.00%         1070        1070          True
  6    2   48  11                 2.23%                0.00%         1094        1094          True
  6    2   96  11                 2.49%                0.00%         1142        1142          True
  6    2  192  11                 2.49%                0.00%         1238        1238          True
  8    1   24   8                 0.00%                0.00%          980         980          True
  8    1   48   8                 0.00%                0.00%         1004        1004          True
  8    1   96   8                 0.00%                0.00%         1052        1052          True
  8    1  192   8                 2.43%                0.00%         1148        1148          True
  8    2   24  15                 0.25%                0.56%         1806        1806          True
  8    2   48  15                 0.25%                0.00%         1830        1830          True
  8    2   96  15                 0.25%                0.22%         1878        1878          True
  8    2  192  15                 0.25%                0.25%         1974        1974          True
 12    1   24  12                 0.00%                0.00%         2028        2028          True
 12    1   48  12                 3.05%                0.00%         2052        2052          True
 12    1   96  12                 3.05%                0.00%         2100        2100          True
 12    1  192  12                 3.05%                0.00%         2196        2196          True
 12    2   24  23                 0.00%                0.00%         3854        3854          True
 12    2   48  23                 0.00%                0.00%         3878        3878          True
 12    2   96  23                 0.00%                0.00%         3926        3926          True
 12    2  192  23                 0.66%                0.00%         4022        4022          True
```

## Verdict

**STRONG-BUT-NARROW** — the scaling measurement passes cleanly (§1); the fidelity
component is lossy-on-probing and only empirically patched (§2); and the finding
itself follows from an elementary bound whose enabling regime may not exist at
visual scale (§3). Read all three before quoting the headline.

Three separable questions, reported separately rather than averaged into one word.

### (1) Scaling: **STRONG**

- M saturates on every resolution-refinement axis (grid_param |G|=1,2,4): **True**
- M saturates on the occluder refinement axis: **True**
- M/K at the largest K measured (grid_param, |G|=1): **0.00195** (threshold 0.05); M was 3 at K=6 and 3 at K=1536
- Control axis (mass_sort, all latent dimensions decision-relevant) has M growing like K, log-log slope 1.000: **True** — so the saturation above is a property of RESOLUTION refinement, not a tautology of the setup
- Exact COMMIT regret (budget 0) vs full belief, worst measured cell: **0.00e+00** (exactly lossless: **True**)
- Compute advantage over K at fixed H=1: 3.58x at K=12 -> 51.74x at K=1536 (widens: **True**)
- Compute advantage over H at fixed K=192: 2.86x at H=0 -> 60.02x at H=4 (widens or holds: **True**)
- Deepest-horizon ratio grows with K: K=24: 7.6x (ceiling K/M=8), K=96: 30.1x (ceiling K/M=32), K=384: 119.9x (ceiling K/M=128), K=1536: 450.5x (ceiling K/M=512) -> **True**

### (2) Decision fidelity: **LOSSY-ON-PROBING (recoverable: fixed by the mode representative)**

- Exact CLOSED-LOOP regret (H=1, planner may probe) vs full belief, worst cell in the S2 sweep: **1.10%** of the full-belief return (within 2%: **True**; grows with K: **False**)
- Worst closed-loop regret across the S6 |A| sweep: **44.74%** of return
- Root-action disagreements with the full-belief oracle in S6: **2 / 12** cells (|A|=2,K=96, |A|=2,K=384)
- Compressed planner matched the oracle root action in every measured cell across S2+S3+S3b+S4 (the False below is exactly the |A|=2 cell above): **False**

- Swapping the mode representative from `maxweight` to `centroid` (same partition, same M, **identical compute**: True) moves the worst closed-loop regret over the whole 48-cell (|A|, |G|, K) grid from **44.74%** to **0.56%**, nonzero in only 3/48 cells: recoverable = **True**

The compression is exactly lossless for the terminal commit at every resolution, but it is NOT value-preserving, and probe decisions are made on values. Collapsing a mode onto a representative particle biases the value estimate in a direction set by which particle is chosen (S6: max-weight under-values, mode-mean over-values), so the compressed planner can decline a probe the oracle takes. Crucially this error is CONSTANT in K, not growing — it does not erode the scaling result.

S7 then shows most of that error was an ARTEFACT rather than a property of decision-mode compression: the original `maxweight` rule degenerates under a flat prior (all members tie, the tie-break returns the first index, i.e. a mode-EDGE particle), and simply collapsing onto the mode-conditional mean instead removes the catastrophic |A|=2 root flip and brings the worst regret across the whole grid to 0.56% at exactly the same compute. The residual is not zero (3/48 cells), and neither rule is value-CONSISTENT (S6: max-weight under-values, centroid over-values the same belief), so the clean guarantee still needs a value-consistent mode summary — e.g. carrying each mode's belief-weighted Q-vector and its within-mode observation likelihood instead of any single representative particle. What S7 establishes is that the fidelity gap is a fixable implementation detail that costs no compute, not a wall in front of the scaling result.

### (3) Interpretation caveat: **the bound is elementary, and its regime is not guaranteed**

Three qualifications that the §1 numbers do not by themselves convey. They are the
reason this document's headline is STRONG-BUT-NARROW rather than STRONG, and they
reconcile it with `gateC1_design.md`, which reads the same data pessimistically.

1. **The saturation is a two-line counting argument, not a discovery.** The analytic
   bound used throughout S1 is `M <= min(K, |G|*(|A|-1)+1)`: the number of cells in
   the arrangement of all goals' decision boundaries. For a 1-D hidden parameter each
   goal contributes at most `|A|-1` cut points, so `|G|` goals cut the line into at
   most `|G|*(|A|-1)+1` intervals -- independent of how finely the belief discretizes
   it. Measured `M` matches this bound exactly at every cell. The empirical study
   confirms the bound and rules out tautology via the control axis; it does not
   discover a surprising law.

2. **Compression helps iff `K >> |G|*(|A|-1)+1`.** That is the operative regime
   condition, and it is a statement about the *filter's resolution* relative to the
   *decision structure*, not about the task being hard. It is satisfied trivially here
   (oracle particle filters at `K` up to 1536). Whether it holds for a *learned* belief
   model -- which typically carries a handful of mixture components or particles -- is
   unknown and is the load-bearing open question for Gate C1. If a learned filter
   carries `K ~ 10-20` while `|G|*(|A|-1)+1` is comparable, then `M = K` and the entire
   compute advantage measured here vanishes. Gate C1 front-loads this as its P0 gate.

3. **The win is in the planning tree, not in belief maintenance.** The compressed
   planner still reads the `K`-particle belief once per decision (unavoidable `O(K)`),
   which is why the ratio *flattens* along the fixed-`H` axis in S2 and only climbs
   toward the `K/M` ceiling as `H` grows. Honest characterization: tree cost drops from
   `O(K * tree)` to `O(M * tree)`; belief cost stays `O(K)`. At visual scale, where a
   per-particle encoder pass is the dominant cost, this cap binds far earlier than it
   does on this toy -- Gate C1 STOP S3 (belief-model cost >= 80% of total) exists
   precisely for that failure mode.

## Figures

- `figures/gateC0_s1_saturation.png`
- `figures/gateC0_s2s3_compute.png`

_Total runtime: 244.2s._

