# Gate C0 — Scaling Study

Does decision-equivalent belief compression buy an ASYMPTOTIC advantage, or only the constant factor Gate B measured?

Reproduce: `diagnosis/.venv/bin/python -m belief_compression.scaling` (from repo root). Everything below is exact (exact Bayes, exact expectimax, exact expected return by enumeration) and was produced by that command. Cells the exact enumerator could not afford are printed `n/a`, never estimated.

## The question

The value proposition is *"full-belief planning becomes intractable as the problem scales; decision-mode planning stays cheap while preserving the decision"*. That requires **M bounded while K grows**. If instead M grows with K, the win is a constant factor and the claim is weak.

### Compute-accounting fix (disclosed)

`Belief.updated` previously charged the compute counter `K` likelihood touches even for a belief whose support was only `M` particles, which would have masked exactly the effect this study measures. It now charges (and evaluates) the support only — a filter carrying M particles does not evaluate the K-M particles it does not carry, and zero-weight particles stay zero under Bayes, so the posterior is unchanged. Re-running Gate B after the fix leaves every return, regret and verdict identical (still GO); only two compute numbers move, both for `amortized_voi` (mass_sort 222 -> 198, occluder_push 110 -> 74). Full-belief numbers are unchanged because its support is all of K.

### Default mode representative changed (disclosed)

S7 (below) showed that the original `maxweight` representative rule is a degenerate choice under a flat prior and costs up to 44.74% closed-loop regret at byte-identical compute. **`centroid` is now the default `rep_rule`** for `collapse_modes`, `compress()` and both compression planners; `maxweight` remains selectable so the S6/S7 failure stays reproducible. Every table below that says `compression` / `compression_cached` (S2, S3, S3b, S4) is therefore now the `centroid` planner. The mode PARTITION is unchanged by the rule, so M and every compute number are identical to the previous run; the closed-loop regret columns improve. Two further rules — `summary` and `summary_exact` — carry a value-consistent per-mode summary instead of a representative particle; they are measured in S6, S7 and S8.

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
  48  3  0.0625    2796     330       234   2795      329      11.9487    0.7482       0.00e+00       0.00%       True
  96  3  0.0312    5580     474       282   5579      473      19.7872    0.7482       0.00e+00       0.00%       True
 192  3  0.0156   11148     762       378  11147      761      29.4921    0.7482       0.00e+00       0.00%       True
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
1  192  3     11148       378      29.4921       0.0049         0.0009       0.00e+00       True
2  192  3    127429      2314      55.0687       0.0576         0.0083            n/a       True
3  192  3   1406520     23610      59.5731       0.6283         0.0917            n/a       True
4  192  3  15476521    257866      60.0177       7.0063         1.0143            n/a       True
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
  2  384   2  0.0052      2   17676       488      36.2213       True
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

## S6 — decision fidelity != VALUE fidelity, and how to get value fidelity back

S1/S2 show the collapse is **exactly lossless for the terminal commit** (regret 0.00e+00 at budget 0, every K). But a planner that can PROBE compares the value of committing now against the expected value of committing after an observation, and decision-mode compression preserves the *argmax* of the commit, not its *value*. Collapsing a mode onto one representative particle therefore biases the value estimate, and the bias direction depends on which particle you pick: `maxweight` (the mode's highest-weight member, which under a flat prior is an arbitrary mode-EDGE particle) systematically UNDER-values; `centroid` (the particle nearest the mode-conditional mean parameter) systematically OVER-values. Neither is value-consistent, so for a representative particle this is structural, not a bad choice of representative.

The fix is to stop collapsing onto a particle at all. `summary` carries, per mode, the belief-weighted Q-vector `Q_m(a,g) = E[reward(z,a,g) | mode m]` and the within-mode observation likelihood `L_m(o|u) = E[P(o|z,u) | mode m]`; then the commit value `sum_m w_m Q_m`, the observation marginal `sum_m w_m L_m` and the posterior mode weights `w_m L_m(o) / P(o)` all equal their full-belief values identically. What that does NOT carry is the within-mode conditional AFTER an observation (an observation reweights particles inside a mode too), so `summary` is exact at the root and stale below it. `summary_exact` additionally refreshes the within-mode weights at every observation — exact at every depth, at a cost measured in S8.

```
|A|    K   M  V_full  V_maxweight  V_centroid  V_summary  V_summary_exact     root_full  root_maxweight  root_centroid  root_summary
---  ---  --  ------  -----------  ----------  ---------  ---------------  ------------  --------------  -------------  ------------
  2   96   2  0.6786       0.5000      0.9115     0.6786           0.6786  probe:sense8        commit:0   probe:sense8  probe:sense8
  2  384   2  0.6786       0.5000      0.9190     0.6786           0.6786  probe:sense8        commit:0   probe:sense8  probe:sense8
  3   96   3  0.7482       0.6108      0.9114     0.7155           0.7482  probe:sense8    probe:sense8   probe:sense8  probe:sense8
  3  384   3  0.7482       0.6048      0.9189     0.7155           0.7482  probe:sense8    probe:sense8   probe:sense8  probe:sense8
  4   96   4  0.7946       0.6903      0.9080     0.7946           0.7946  probe:sense8    probe:sense8   probe:sense8  probe:sense8
  4  384   4  0.7946       0.6843      0.9154     0.7946           0.7946  probe:sense8    probe:sense8   probe:sense8  probe:sense8
  6   96   6  0.8210       0.7666      0.9053     0.7948           0.8210  probe:sense8    probe:sense8   probe:sense8  probe:sense8
  6  384   6  0.8210       0.7607      0.9127     0.7948           0.8210  probe:sense8    probe:sense8   probe:sense8  probe:sense8
  8   96   8  0.8531       0.8036      0.9026     0.8531           0.8531  probe:sense8    probe:sense8   probe:sense8  probe:sense8
  8  384   8  0.8531       0.7977      0.9100     0.8531           0.8531  probe:sense8    probe:sense8   probe:sense8  probe:sense8
 12   96  12  0.8462       0.8363      0.8561     0.8331           0.8462  probe:sense8    probe:sense8   probe:sense8  probe:sense8
 12  384  12  0.8462       0.8345      0.8593     0.8331           0.8462  probe:sense8    probe:sense8   probe:sense8  probe:sense8
```

The residual itself, which is the actual question: `dV0_*` is the value error at budget 0 (commit only) and `dV_*` at budget 1 (probing allowed). A value-consistent summary must be 0 in both columns; a representative rule is 0 in neither.

```
|A|    K   M  dV0_maxweight  dV0_centroid  dV0_summary  dV0_summary_exact  dV_maxweight  dV_centroid  dV_summary  dV_summary_exact
---  ---  --  -------------  ------------  -----------  -----------------  ------------  -----------  ----------  ----------------
  2   96   2       1.25e-01      1.25e-01     0.00e+00           0.00e+00     -1.79e-01     2.33e-01    1.11e-16          2.22e-16
  2  384   2       1.25e-01      1.25e-01     0.00e+00           0.00e+00     -1.79e-01     2.40e-01   -6.00e-15         -6.00e-15
  3   96   3      -5.21e-02      5.21e-02     5.55e-17           5.55e-17     -1.37e-01     1.63e-01   -3.27e-02          1.11e-16
  3  384   3      -5.47e-02      5.47e-02     0.00e+00           0.00e+00     -1.43e-01     1.71e-01   -3.27e-02         -6.22e-15
  4   96   4       3.12e-02      3.12e-02     0.00e+00           0.00e+00     -1.04e-01     1.13e-01    4.44e-16          4.44e-16
  4  384   4       3.12e-02      3.12e-02     0.00e+00           0.00e+00     -1.10e-01     1.21e-01   -7.33e-15         -7.33e-15
  6   96   6       1.39e-02      1.74e-02     1.11e-16           1.11e-16     -5.44e-02     8.43e-02   -2.62e-02          2.22e-16
  6  384   6       1.39e-02      1.56e-02     0.00e+00           0.00e+00     -6.03e-02     9.16e-02   -2.62e-02         -7.22e-15
  8   96   8       7.81e-03      7.81e-03     0.00e+00           0.00e+00     -4.95e-02     4.95e-02    2.22e-16          4.44e-16
  8  384   8       7.81e-03      7.16e-03     0.00e+00           0.00e+00     -5.54e-02     5.69e-02   -7.66e-15         -7.66e-15
 12   96  12       3.47e-03      1.74e-03    -5.55e-17          -5.55e-17     -9.97e-03     9.91e-03   -1.31e-02          6.66e-16
 12  384  12       3.47e-03      3.04e-03    -5.55e-17          -5.55e-17     -1.17e-02     1.30e-02   -1.31e-02         -7.77e-15
```

Exact closed-loop regret at H=1 for each rule (relative to the full-belief return; exact enumeration is only affordable for K <= 192):

```
|A|    K   M  ret_full  rel_maxweight  rel_centroid  rel_summary  rel_summary_exact
---  ---  --  --------  -------------  ------------  -----------  -----------------
  2   96   2    0.6786         44.74%         0.00%        0.00%              0.00%
  2  384   2       n/a            n/a           n/a          n/a                n/a
  3   96   3    0.7482          1.10%         0.00%        0.00%              0.00%
  3  384   3       n/a            n/a           n/a          n/a                n/a
  4   96   4    0.7946          3.48%         0.00%        0.00%              0.00%
  4  384   4       n/a            n/a           n/a          n/a                n/a
  6   96   6    0.8210          2.49%         0.00%        0.00%              0.00%
  6  384   6       n/a            n/a           n/a          n/a                n/a
  8   96   8    0.8531          0.00%         0.00%        0.00%              0.00%
  8  384   8       n/a            n/a           n/a          n/a                n/a
 12   96  12    0.8462          3.05%         0.00%        0.00%              0.00%
 12  384  12       n/a            n/a           n/a          n/a                n/a
```

## S7 — is the S6 failure intrinsic, or an artefact of how a mode is carried?

S6 established that the collapsed belief mis-values and can therefore pick the wrong root action. That is only damning if it survives a better way of carrying a mode. It does not. The mode PARTITION is identical under all four rules — same modes, same M — so this table isolates exactly the cost of what each rule carries per mode. For the two representative rules the compute is byte-identical, so the `maxweight -> centroid` improvement is FREE; the summary rules carry more and the `C_*` columns are the price (S8 measures it as a function of K and H).

```
|A|  |G|    K   M  rel_regret_maxweight  rel_regret_centroid  rel_regret_summary  rel_regret_summary_exact  C_maxweight  C_centroid  C_summary  C_summary_exact
---  ---  ---  --  --------------------  -------------------  ------------------  ------------------------  -----------  ----------  ---------  ---------------
  2    1   24   2                44.74%                0.00%               0.00%                     0.00%          128         128        440             1210
  2    1   48   2                44.74%                0.00%               0.00%                     0.00%          152         152        776             2338
  2    1   96   2                44.74%                0.00%               0.00%                     0.00%          200         200       1448             4594
  2    1  192   2                44.74%                0.00%               0.00%                     0.00%          296         296       2792             9106
  2    2   24   3                10.83%                0.00%               0.00%                     0.00%          174         174        486             1245
  2    2   48   3                10.83%                0.00%               0.00%                     0.00%          198         198        822             2373
  2    2   96   3                10.83%                0.00%               0.00%                     0.00%          246         246       1494             4629
  2    2  192   3                10.83%                0.00%               0.00%                     0.00%          342         342       2838             9141
  3    1   24   3                 0.00%                0.00%               0.00%                     0.00%          210         210        546             1569
  3    1   48   3                 1.10%                0.00%               0.00%                     0.00%          234         234        906             2985
  3    1   96   3                 1.10%                0.00%               0.00%                     0.00%          282         282       1626             5817
  3    1  192   3                 1.10%                0.00%               0.00%                     0.00%          378         378       3066            11481
  3    2   24   5                 0.00%                0.00%               0.00%                     0.00%          326         326        662             1663
  3    2   48   5                 1.10%                0.00%               0.00%                     0.00%          350         350       1022             3079
  3    2   96   5                 1.10%                0.00%               0.00%                     0.00%          398         398       1742             5911
  3    2  192   5                 1.10%                0.00%               0.00%                     0.00%          494         494       3182            11575
  4    1   24   4                 0.00%                0.00%               0.00%                     0.00%          316         316        676             1952
  4    1   48   4                 0.00%                0.00%               0.00%                     0.00%          340         340       1060             3656
  4    1   96   4                 3.48%                0.00%               0.00%                     0.00%          388         388       1828             7064
  4    1  192   4                 3.48%                0.00%               0.00%                     0.00%          484         484       3364            13880
  4    2   24   7                 0.00%                0.00%               0.00%                     0.00%          526         526        886             2129
  4    2   48   7                 0.00%                0.00%               0.00%                     0.00%          550         550       1270             3833
  4    2   96   7                 3.50%                0.00%               0.00%                     0.00%          598         598       2038             7241
  4    2  192   7                 7.23%                0.00%               0.00%                     0.00%          694         694       3574            14057
  6    1   24   6                 0.00%                0.00%               0.00%                     0.00%          600         600       1008             2790
  6    1   48   6                 2.49%                0.00%               0.00%                     0.00%          624         624       1440             5070
  6    1   96   6                 2.49%                0.00%               0.00%                     0.00%          672         672       2304             9630
  6    1  192   6                 2.49%                0.00%               0.00%                     0.00%          768         768       4032            18750
  6    2   24  11                 0.00%                0.00%               0.00%                     0.00%         1070        1070       1478             3205
  6    2   48  11                 2.23%                0.00%               0.00%                     0.00%         1094        1094       1910             5485
  6    2   96  11                 2.49%                0.00%               0.00%                     0.00%         1142        1142       2774            10045
  6    2  192  11                 2.49%                0.00%               0.00%                     0.00%         1238        1238       4502            19165
  8    1   24   8                 0.00%                0.00%               0.00%                     0.00%          980         980       1436             3724
  8    1   48   8                 0.00%                0.00%               0.00%                     0.00%         1004        1004       1916             6580
  8    1   96   8                 0.00%                0.00%               0.00%                     0.00%         1052        1052       2876            12292
  8    1  192   8                 2.43%                0.00%               0.00%                     0.00%         1148        1148       4796            23716
  8    2   24  15                 0.25%                0.56%               0.00%                     0.00%         1806        1806       2262             4473
  8    2   48  15                 0.25%                0.00%               0.00%                     0.00%         1830        1830       2742             7329
  8    2   96  15                 0.25%                0.22%               0.00%                     0.00%         1878        1878       3702            13041
  8    2  192  15                 0.25%                0.25%               0.00%                     0.00%         1974        1974       5622            24465
 12    1   24  12                 0.00%                0.00%               0.00%                     0.00%         2028        2028       2580             5880
 12    1   48  12                 3.05%                0.00%               0.00%                     0.00%         2052        2052       3156             9888
 12    1   96  12                 3.05%                0.00%               0.00%                     0.00%         2100        2100       4308            17904
 12    1  192  12                 3.05%                0.00%               0.00%                     0.00%         2196        2196       6612            33936
 12    2   24  23                 0.00%                0.00%               0.00%                     0.00%         3854        3854       4406             7585
 12    2   48  23                 0.00%                0.00%               0.00%                     0.00%         3878        3878       4982            11593
 12    2   96  23                 0.00%                0.00%               0.00%                     0.00%         3926        3926       6134            19609
 12    2  192  23                 0.66%                0.00%               0.00%                     0.00%         4022        4022       8438            35641
```

## S8 — what the value-consistent summary COSTS

S7 shows the summary's regret; this shows its compute, against the same `C_full / C_compressed` ratio S2/S3b report. The representative rules pay `O(K)` once to bucket the belief and then `O(M)` per expectimax node. A summary must additionally turn the K particles into per-mode Q-vectors and likelihoods, which is `O(K(|A| + sum_u |O_u|))` — the same `O(K)` order, but with a much larger constant (here 3 controllers + 11 sensor bins = 14x the bucketing pass). That cost is paid ONCE PER DECISION, so whether it matters is entirely a question of how much tree it is amortized over — hence the (K, H) sweep. `summary_exact` instead pays `O(K)` at EVERY node, which is full-belief cost by construction. Same affordability cap as S3b (4e7 primitive ops).

```
   K  H  M       K/M    C_full  C_maxweight  C_centroid  C_summary  C_summary_exact  x_maxweight  x_centroid  x_summary  x_summary_exact
----  -  -  --------  --------  -----------  ----------  ---------  ---------------  -----------  ----------  ---------  ---------------
  24  0  3    8.0000        73           34          34        106              106       2.1471      2.1471     0.6887           0.6887
  24  1  3    8.0000      1404          210         210        546             1569       6.6857      6.6857     2.5714           0.8948
  24  2  3    8.0000     16045         2146        2146       2482            17662       7.4767      7.4767     6.4645           0.9084
  24  3  3    8.0000    177096        23442       23442      23778           194685       7.5546      7.5546     7.4479           0.9097
  24  4  3    8.0000   1948657       257698      257698     258034          2141938       7.5618      7.5618     7.5519           0.9098
  96  0  3   32.0000       289          106         106        394              394       2.7264      2.7264     0.7335           0.7335
  96  1  3   32.0000      5580          282         282       1626             5817      19.7872     19.7872     3.4317           0.9593
  96  2  3   32.0000     63781         2218        2218       3562            65470      28.7561     28.7561    17.9060           0.9742
  96  3  3   32.0000    703992        23514       23514      24858           721653      29.9393     29.9393    28.3205           0.9755
  96  4  3   32.0000   7746313       257770      257770     259114          7939666      30.0513     30.0513    29.8954           0.9756
 384  0  3  128.0000      1153          394         394       1546             1546       2.9264      2.9264     0.7458           0.7458
 384  1  3  128.0000     22284          570         570       5946            22809      39.0947     39.0947     3.7477           0.9770
 384  2  3  128.0000    254725         2506        2506       7882           256702     101.6460    101.6460    32.3173           0.9923
 384  3  3  128.0000   2811576        23802       23802      29178          2829525     118.1235    118.1235    96.3594           0.9937
 384  4  3  128.0000  30936937       258058      258058     263434         31130578     119.8837    119.8837   117.4371           0.9938
1536  0  3  512.0000      4609         1546        1546       6154             6154       2.9812      2.9812     0.7489           0.7489
1536  1  3  512.0000     89100         1722        1722      23226            90777      51.7422     51.7422     3.8362           0.9815
1536  2  3  512.0000   1018501         3658        3658      25162          1021630     278.4311    278.4311    40.4777           0.9969
1536  3  3  512.0000  11241912        24954       24954      46458         11261013     450.5054    450.5054   241.9801           0.9983
```

## Verdict

**STRONG-BUT-NARROW** — the scaling measurement passes cleanly (§1); the fidelity component is value-inconsistent for every representative particle and needs a value-consistent mode summary to be exact (§2); and the finding itself follows from an elementary bound whose enabling regime may not exist at visual scale (§3). Read all three before quoting the headline.

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

### (2) Decision fidelity: **LOSSY for any representative particle; EXACTLY RECOVERABLE by a value-consistent mode summary (at a compute cost, §S8)**

- Exact CLOSED-LOOP regret (H=1, planner may probe) vs full belief, worst cell in the S2 sweep: **0.00%** of the full-belief return (within 2%: **True**; grows with K: **False**)
- Worst closed-loop regret across the S6 |A| sweep, `maxweight` (the pre-2026-07-30 default): **44.74%** of return
- Root-action disagreements with the full-belief oracle in S6 under `maxweight`: **2 / 12** cells (|A|=2,K=96, |A|=2,K=384)
- Compressed planner (now `centroid` by default) matched the oracle root action in every measured cell across S2+S3+S3b+S4: **True**

**Value-consistency residual** (S6), worst over the sweep — `dV0` is the commit-only value error (budget 0), `dV` the planning value error (budget 1). Anything at or below 1e-09 is floating-point zero (the summaries land at ~1e-16 to ~1e-14, i.e. exactly 0 up to accumulation error, not approximately 0):

  - `maxweight`: dV0 = **1.25e-01**, dV = **1.79e-01**  <- value-inconsistent everywhere
  - `centroid`: dV0 = **1.25e-01**, dV = **2.40e-01**  <- value-inconsistent everywhere
  - `summary`: dV0 = **1.11e-16**, dV = **3.27e-02**  <- exactly value-consistent for the commit, drifts once probing is allowed
  - `summary_exact`: dV0 = **1.11e-16**, dV = **7.77e-15**  <- exactly value-consistent at every budget

**Worst closed-loop regret over the whole 48-cell (|A|, |G|, K) grid** (S7), by how the mode is carried:

  - `maxweight`: **44.74%**, nonzero in 33/48 cells
  - `centroid`: **0.56%**, nonzero in 3/48 cells
  - `summary`: **0.00%**, nonzero in 0/48 cells
  - `summary_exact`: **0.00%**, nonzero in 0/48 cells

- `maxweight -> centroid` is a pure win: same partition, same M, **identical compute** (True), worst regret 44.74% -> 0.56%. `centroid` is therefore now the DEFAULT `rep_rule`; `maxweight` stays selectable so the failure above remains reproducible.

**What the value-consistent summary costs** (S8), `C_full / C_compressed` at the shallowest and deepest affordable horizon:

```
  K=24: centroid 2.1x -> 7.6x, summary 0.7x -> 7.6x, summary_exact 0.9x  (H=0 -> H=4)
  K=96: centroid 2.7x -> 30.1x, summary 0.7x -> 29.9x, summary_exact 1.0x  (H=0 -> H=4)
  K=384: centroid 2.9x -> 119.9x, summary 0.7x -> 117.4x, summary_exact 1.0x  (H=0 -> H=4)
  K=1536: centroid 3.0x -> 450.5x, summary 0.7x -> 242.0x, summary_exact 1.0x  (H=0 -> H=3)
```

- The frozen `summary` keeps a wide compute advantage once the tree is deep enough to amortize its `O(K(|A|+sum_u|O_u|))` build: **True**
- `summary_exact` pays `O(K)` at every node and lands on full-belief cost (≈1x at every measured cell): **True**

The compression is exactly lossless for the terminal commit at every resolution, but collapsing a mode onto a representative PARTICLE is not value-preserving, and probe decisions are made on values. The bias direction is set by which particle is chosen (S6: max-weight under-values, mode-mean over-values), so the compressed planner can decline a probe the oracle takes. Crucially this error is CONSTANT in K, not growing — it does not erode the scaling result.

Most of that error is an ARTEFACT rather than a property of decision-mode compression: the old `maxweight` rule degenerates under a flat prior (all members tie, the tie-break returns the first index, i.e. a mode-EDGE particle). Switching the default to the mode-conditional-mean particle removes the catastrophic |A|=2 root flip and brings the worst regret across the grid to 0.56% at exactly the same compute.

The residual is not zero, and no representative particle can be value-consistent, so the clean guarantee needs a value-consistent mode SUMMARY: per mode, the belief-weighted Q-vector and the within-mode observation likelihood instead of a particle. Measured (S6/S7/S8), that summary does exactly what the theory says and no more:

  1. It is exactly value-consistent for the COMMIT at every measured cell (dV0 = 1.11e-16, floating-point zero), which no representative rule is anywhere.
  2. Frozen, it is NOT value-consistent once probing is allowed (dV = 3.27e-02): an observation reweights the particles WITHIN a mode, and a summary built from the prior does not see that. That residual is still smaller than either representative rule's (3.27e-02 vs 2.40e-01 for `centroid`), and on this grid it never flips an argmax: worst closed-loop regret 0.00% in 0/48 cells. The value error is real; the DECISION error it causes is not, here.
  3. Refreshing the within-mode conditional at every observation (`summary_exact`) IS exactly value-consistent at every budget (dV0 = 1.11e-16, dV = 7.77e-15) and exactly regret-free on the whole grid (0/48 nonzero cells) — and costs `O(K)` per node, i.e. it gives back the entire compute advantage (S8: ≈1x vs full belief).

So value-consistency and the compute win trade off, and where the trade sits depends on the horizon, not on the method:

  - At `H <= 1` the summary's `O(K)` build is not amortized over enough tree and costs more than the compression saves (S8 at H=0: 0.7x/0.7x/0.7x/0.7x for K=24/96/384/1536, i.e. WORSE than planning the full belief). Use `centroid`: free, and worst-case 0.56% over the grid.
  - At `H >= 2` the build is amortized away and the frozen `summary` is close to free (S8, deepest affordable cell: K=24: 8x vs 8x, K=96: 30x vs 30x, K=384: 117x vs 120x, K=1536: 242x vs 451x), for 0.00% regret and exact commit values. That is the configuration to prefer when the planner searches at all deeply.
  - `summary_exact` is the auditable zero-regret reference, not a production planner: it is the only rule proven value-consistent under probing, and it costs exactly what full-belief planning costs.

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
   (oracle particle filters at `K` up to 1536). Whether it holds for a *learned*
   belief model -- which typically carries a handful of mixture components or particles
   -- is unknown and is the load-bearing open question for Gate C1. If a learned filter
   carries `K ~ 10-20` while `|G|*(|A|-1)+1` is comparable, then `M = K` and the entire
   compute advantage measured here vanishes. Gate C1 front-loads this as its P0 gate.

3. **The win is in the planning tree, not in belief maintenance -- and the
   value-consistent summary makes that sharper, not softer.** The compressed planner
   still reads the `K`-particle belief once per decision (unavoidable `O(K)`), which is
   why the ratio *flattens* along the fixed-`H` axis in S2 and only climbs toward the
   `K/M` ceiling as `H` grows. Honest characterization: tree cost drops from
   `O(K * tree)` to `O(M * tree)`; belief cost stays `O(K)`. S8 shows how much that
   `O(K)` constant matters: making the mode summary value-consistent for the commit
   multiplies it by roughly `|A| + sum_u |O_u|` (here 3 controllers + 11 sensor
   bins), which at `H=0..1` costs more than the whole compression win and only pays for
   itself at `H>=2`; making it value-consistent under probing costs `O(K)` per node and
   erases the win entirely. At visual scale, where a per-particle encoder pass is the
   dominant cost, this cap binds far earlier than it does on this toy -- Gate C1 STOP S3
   (belief-model cost >= 80% of total) exists precisely for that failure mode.

## Figures

- `figures/gateC0_s1_saturation.png`
- `figures/gateC0_s2s3_compute.png`

_Total runtime: 246.2s._

