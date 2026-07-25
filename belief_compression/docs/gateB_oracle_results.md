# Gate B — Oracle Feasibility Results

Decision-Equivalent Belief Compression + amortized decision-regret VOI, validated on two tiny exact POMDPs (pure numpy, no NN / no physics).

**VERDICT: GO**

Reproduce: `diagnosis/.venv/bin/python -m belief_compression.run` (from repo root).

## Measurement (i): compression vs goal-richness

`richness` = amount of the hidden parameter made decision-relevant by the goal family. `M/K` = surviving decision modes / hypotheses. `regret_in` = mean planning regret vs full-belief when executing goals INSIDE the family; `regret_out` = executing a goal that needs the full parameter (out of family). Returns are exact expectations. NOTE: `regret_out` is ~0 in both tasks because this measurement isolates the terminal commit (budget=0), and the commit in these tasks decomposes over the hidden parameter, so mode representatives preserve the per-dimension MAP even out of family. The out-of-family cost of over-compression shows up in PROBE valuation (a narrow-family compression misjudges the value of probing an object it merged away), not in the commit; measurement (ii) exercises that probing axis directly.

```
### mass_sort: compression vs goal-richness (exact decision-equivalence)

richness  |G|   K   M     M/K  regret_in  regret_out
--------  ---  --  --  ------  ---------  ----------
       1    1  16   2  0.1250     0.0000      0.0000
       2    2  16   4  0.2500     0.0000      0.0000
       3    3  16   8  0.5000     0.0000      0.0000
       4    4  16  16  1.0000     0.0000      0.0000

### mass_sort: lossy-compression frontier (tolerance sweep, richness=max)

   tol   K   M     M/K  regret_in
------  --  --  ------  ---------
0.0000  16  16  1.0000     0.0000
2.0000  16  16  1.0000     0.0000
2.9000  16   8  0.5000     0.0000
3.5000  16   8  0.5000     0.0000
4.1000  16   2  0.1250     0.0000
5.0000  16   2  0.1250     0.0000

### occluder_push: compression vs goal-richness (exact decision-equivalence)

richness  |G|  K  M     M/K  regret_in  regret_out
--------  ---  -  -  ------  ---------  ----------
       2    2  8  2  0.2500     0.0000      0.0000
       4    3  8  4  0.5000     0.0000      0.0000
       8    4  8  8  1.0000     0.0000      0.0000

### occluder_push: lossy-compression frontier (tolerance sweep, richness=max)

   tol  K  M     M/K  regret_in
------  -  -  ------  ---------
0.0000  8  8  1.0000     0.0000
2.0000  8  8  1.0000     0.0000
2.9000  8  4  0.5000     0.0000
3.5000  8  4  0.5000     0.0000
5.0000  8  1  0.1250     0.0000

```

## Measurement (ii): probe-policy comparison

Exact expected return (net of probe cost), decision regret vs the full-belief oracle and vs the fully-observed upper bound, expected probes taken, and planning compute at the root decision (reward evals / VOI inner calls / total primitive ops).

```
### mass_sort: probe-policy comparison

         policy  return  reg_vs_full  reg_vs_obs  E[probes]  rew_evals  voi_inner  compute
---------------  ------  -----------  ----------  ---------  ---------  ---------  -------
certainty_equiv  0.0000       1.3000      2.0000     0.0000          4          0        4
   entropy_seek  0.6000       0.7000      1.4000     2.0000         32          0      128
            voi  1.3000       0.0000      0.7000     2.0000        224          6      326
  amortized_voi  1.3000       0.0000      0.7000     2.0000        144          6      222
    full_belief  1.3000       0.0000      0.7000     2.0000       1376          0     2091
 fully_observed  2.0000      -0.7000      0.0000     0.0000          0          0        0

### occluder_push: probe-policy comparison

         policy   return  reg_vs_full  reg_vs_obs  E[probes]  rew_evals  voi_inner  compute
---------------  -------  -----------  ----------  ---------  ---------  ---------  -------
certainty_equiv   0.0000       0.7000      1.0000     0.0000          2          0        2
   entropy_seek  -0.1000       0.8000      1.1000     1.0000         16          0      112
            voi   0.7000       0.0000      0.3000     1.0000        112          6      214
  amortized_voi   0.7000       0.0000      0.3000     1.0000         44          6      110
    full_belief   0.7000       0.0000      0.3000     1.0000        112          0      215
 fully_observed   1.0000      -0.3000      0.0000     0.0000          0          0        0

```

## Gate A — substantial compression at near-full-belief regret

PASS. Regimes with M/K <= 0.5 and regret <= 1e-06 (non-trivial goal family, >= 2 goals):

- mass_sort: richness=2, |G|=2, K=16, M=4, M/K=0.2500, regret_in=0.000000
- mass_sort: richness=3, |G|=3, K=16, M=8, M/K=0.5000, regret_in=0.000000
- occluder_push: richness=2, |G|=2, K=8, M=2, M/K=0.2500, regret_in=0.000000
- occluder_push: richness=4, |G|=3, K=8, M=4, M/K=0.5000, regret_in=0.000000

## Gate B — decision-regret VOI beats entropy-seeking at lower compute

**mass_sort**: VOI return=1.3000 vs entropy=0.6000 (beats_return=True, beats_regret=True); VOI compute=326 < full-belief compute=2091 (True); amortized compute=222 (cheaper_than_VOI=True, matches_VOI_return=True).
**occluder_push**: VOI return=0.7000 vs entropy=-0.1000 (beats_return=True, beats_regret=True); VOI compute=214 < full-belief compute=215 (True); amortized compute=110 (cheaper_than_VOI=True, matches_VOI_return=True).

Gate B overall: PASS.

## Honest GO/STOP read

- Gate A (compression exists, near-zero regret): PASS
- Gate B (VOI > entropy, cheaper than full-belief): PASS

**GO.** Both gates pass. Decision-equivalent compression yields substantial, regret-free mode reduction whenever the goal family leaves part of the hidden parameter decision-irrelevant (M/K = 0.125-0.5 at regret_in = 0), and decision-regret VOI (with its cheap amortized-via-compression form) produces correct probe-then-act behaviour, beating entropy-seeking at strictly lower compute than nested full-belief planning.

Honest caveat surfaced by the data (NOT a free lunch): the compression ratio is *exactly* the goal-irrelevant fraction of the hidden parameter. M/K rises monotonically to 1.0 as goal richness grows to cover every latent dimension (richness 1->4 gives M/K 0.125->0.25->0.5->1.0 on mass_sort; 2->4->8 gives 0.25->0.5->1.0 on occluder_push). Compression does not collapse the instant a second goal appears (so the method is not empty), but it buys nothing once every dimension matters to some goal in the family. The method's value is therefore bounded by how much of the hidden state any realistic decision ignores — a property of the task distribution, not of the algorithm.

## Figures

- `figures/measurement1_compression.png`
- `figures/measurement2_probes.png`
