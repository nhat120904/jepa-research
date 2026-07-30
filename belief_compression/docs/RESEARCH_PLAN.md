# Decision-Equivalent Belief Compression for Scalable Active World-Model Planning

**Research plan — standalone.**
Owner: nhatnc. Date: 2026-07-30. Branch: `exp/oracle-ladder-cost-localization`.
Status of the program: Gate A **conditional GO (narrow)**, Gate B **GO**, Gate C0 **STRONG-BUT-NARROW**, Gate C1 **designed, not started**.

> **Read this first.** This is not a pitch. The project's own gates have already
> concluded that (a) the compression bound is an elementary counting argument
> that is very likely a corollary of published work, (b) one of the two surviving
> novelty claims was *weakened*, not strengthened, by the scaling study, and (c)
> the enabling regime for the whole method (`K >> |G|(|A|-1)+1` in a *learned*
> belief) is unverified and cheap to falsify. The honest expected outcome is a
> careful empirical systems paper about *when* decision-structured belief
> compression pays for itself, with a pre-registered task on which it should
> fail. It is not a conceptual breakthrough. Sections 4, 8 and 9 are the ones
> that matter; sections 1-3 exist so those three can be read precisely.

**One line.** In a partially-observed task, a belief is `K` weighted hypotheses
over hidden state; hypotheses whose *preferred-action signature* agrees across a
goal family are merged into `M <= K` **decision modes**; planning runs over `M`
modes; and the compressed representation is used to cheaply decide whether a
probing action's observation would change the preferred action (amortized
value-of-information), instead of solving an inner planning problem per possible
observation.

---

## Table of contents

1. [Problem & motivation](#1-problem--motivation)
2. [Method](#2-method)
3. [What is already established](#3-what-is-already-established)
4. [Honest novelty position](#4-honest-novelty-position)
5. [Experimental plan](#5-experimental-plan)
6. [Baselines](#6-baselines)
7. [Gates & pre-registered GO/STOP](#7-gates--pre-registered-gostop)
8. [Risks & open questions](#8-risks--open-questions)
9. [Realistic outcome assessment](#9-realistic-outcome-assessment)
10. [Effort & compute budget, task breakdown](#10-effort--compute-budget-task-breakdown)
11. [How to reproduce what exists today](#11-how-to-reproduce-what-exists-today)
12. [References](#12-references)

---

## 1. Problem & motivation

### 1.1 The setting

Contact-rich manipulation is routinely partially observed in a way that vision
does not fix: the mass of a box, the friction of a surface, whether a drawer is
latched, which of three identical cups hides the target, the compliance of a
cable. These are *low-dimensional hidden parameters with high decision leverage*
— you cannot see them, they change which controller is correct, and the only way
to learn them is to **act**: tap the object, lift it slightly, nudge the lid.

Two standard responses, both unsatisfying:

**Certainty-equivalent planning.** Collapse the belief to its MAP hypothesis and
plan as if certain. This is what essentially every visual MPC / world-model
planner in this repository's orbit does (DINO-WM, V-JEPA-2-AC, and the
CAI-JEPA diagnostic's own CEM planner). It is fast and it is *structurally
incapable of valuing information*: a point estimate has zero uncertainty, hence
zero value of information, hence it never probes. In Gate B's exact testbeds a
certainty-equivalent planner returns **0.0000** on both tasks where the
full-belief planner returns **1.3000** (`mass_sort`) and **0.7000**
(`occluder_push`) — it commits to the MAP hypothesis and eats the error
(`gateB_oracle_results.md`, measurement ii).

**Belief-space planning.** Plan over the full posterior. This is correct and it
is expensive: exact expectimax over a `K`-particle belief costs `O(K)` per node
of a tree that branches on (probe x observation) and grows exponentially in the
horizon. Gate C0 measures this directly: at `K = 192`, horizon `H = 4`, one root
decision costs **15,476,521** primitive operations and **7.0 s** of wall clock
(`gateC0_scaling_results.md` §S3). At visual scale each "particle touch" is an
encoder forward pass, and this is simply not a per-control-step budget.

### 1.2 The observation the method rests on

Most of a belief's resolution is **decision-irrelevant**. If your controller
library has 3 push primitives, then a continuous mass belief carrying 192
particles is being used to answer a question with 3 possible answers. Refining
the belief from 192 to 1536 particles cannot create a fourth correct controller.
The planner should be paying for the *decision structure*, not for the *filter's
resolution*.

That is a statement about the *arrangement of decision boundaries in hidden-state
space*, and it has an elementary bound (§1.4). The entire method is an attempt to
turn that bound into a planner, and then to reuse the resulting structure to make
probe valuation cheap.

### 1.3 Worked example — hidden-mass push with a probing tap

Concrete instance, matching the `grid_param` task actually implemented in
`belief_compression/tasks.py`:

- **Hidden state.** A block of unknown mass `m ∈ [50 g, 250 g]`; write
  `θ = (m − 50)/200 ∈ [0,1)`. Vision cannot resolve `θ`.
- **Belief.** A particle filter over `θ` with **K = 192** particles.
  `K` is a property of the *filter*, not of the task — it is a resolution knob.
- **Actions.** `|A| = 3` push controllers calibrated for light / medium / firm,
  at `θ = 1/6, 1/2, 5/6`. Reward `R(θ,a,g) = 1 − 2|θ − c_a(g)|`: the nearest
  calibrated controller wins.
- **Goal family.** `G = {near pad, far pad}`, `|G| = 2`. The far pad needs more
  force at the same mass, so its controller centres are offset — goal 2's
  decision boundaries in `θ` sit at different places than goal 1's.
- **Probe.** A "tap": a short force/torque-instrumented nudge whose readout is
  quantized into 8 bins (`sense8` in the code), noisy, costing `c(u) = 0.05` of
  return.

**Certainty-equivalent** picks the controller for the MAP mass and never taps.
It is wrong exactly when the true mass is on the far side of a decision boundary
from the MAP, and it has no mechanism to find out.

**Full belief-space planning** with a one-step probe lookahead costs
**11,148** primitive ops per decision at `K = 192, H = 1`; at `H = 4` it is
**15.48 M** ops / **7.0 s** (`gateC0_scaling_results.md` §S2, §S3).

**The compression.** Goal 1's three controllers partition `θ` at
`|A| − 1 = 2` points. Goal 2 adds 2 more. Two goals therefore cut the unit
interval into at most **5** pieces, and on each piece *every* particle prefers
the same controller for *both* goals. So:

```
M  <=  min(K, |G|(|A|−1) + 1)  =  min(192, 2·(3−1) + 1)  =  5
```

**192 particles collapse to 5 decision modes; `M/K = 0.026`.** Gate C0 §S5
measures `M = 5` exactly — hitting the bound, not merely under it — at
`K = 24, 96, 384, 1536`, i.e. `M` does not move when the filter's resolution
moves by a factor of 64.

**What that buys.** On the `|G| = 1` version of exactly this task (`M = 3`,
bound `= 3`), Gate C0 measures root-decision compute of
**11,148 → 378 ops** at `K = 192, H = 1` (**29.5x**), and
**15,476,521 → 257,866 ops**, **7.0 s → 1.0 s**, at `K = 192, H = 4`
(**60.0x**) (§S2, §S3). At `K = 1536, H = 3` the ratio is **450.5x** against a
ceiling of `K/M = 512` (§S3b).

**And the probe.** With the belief compressed to 5 modes, "would a tap change
which controller I pick?" can be evaluated at *mode* granularity instead of
particle granularity. On Gate B's `occluder_push`, exact decision-regret VOI
costs **214** ops and the mode-amortized version costs **74** ops for the
**identical** return (0.7000) and the identical expected probe count (1.0)
(`gateB_oracle_results.md`, measurement ii). On `mass_sort`: **326 → 198** ops,
return 1.3000 either way.

### 1.4 The bound, stated honestly

For a **1-D** hidden parameter `θ`, if for each goal `g` the pairwise reward
differences `R(θ,a,g) − R(θ,a',g)` are single-crossing in `θ` (true for the
`1 − 2|θ − c_a|` family, and for any "nearest calibrated controller" reward),
each goal contributes at most `|A| − 1` boundary points, so:

```
M  <=  min( K ,  |G|·(|A| − 1) + 1 )                                  (B1)
```

This is a two-line counting argument, not a discovery, and Gate C0 says so in
its own verdict (§3.1 of that document). **It is important to state the
generalization, because it is a threat.** For a `d`-dimensional hidden
parameter the boundaries are hypersurfaces, and an arrangement of `n`
hypersurfaces in `R^d` has `O(n^d)` cells, so:

```
M  <=  min( K ,  O( (|G|·|A|²)^d ) )                                  (B2)
```

The compression is only interesting in the regime `K >> M`. (B2) says that
regime shrinks *exponentially in the intrinsic dimension of the decision-relevant
hidden state*. On a hidden mass (`d = 1`) it is enormous; on a hidden 6-DoF
object pose it may not exist. This is not currently measured anywhere, and it is
one of the three things most likely to kill the direction (§8).

---

## 2. Method

### 2.1 Belief

A belief is a weighted particle set over hidden states, `b = {(z_k, w_k)}_{k=1}^K`,
`Σ_k w_k = 1`. Implemented exactly in `core.py::Belief` with an exact Bayes
update `b^{u,o}(k) ∝ w_k P(o | z_k, u)` for probe `u` and observation `o`.

*Accounting note (already fixed and disclosed in `gateC0_scaling_results.md`):*
`Belief.updated` charges the compute counter only for the **support** of the
belief, not for all `K` slots. A filter carrying `M` particles does not evaluate
the likelihood of the `K − M` it does not carry, and zero-weight particles stay
zero under Bayes. Charging `K` regardless would have masked exactly the effect
the scaling study measures.

### 2.2 Decision signature

Fix a goal family `G` (e.g. all pad locations, all target colours). For a
hypothesis `z`:

**Exact signature** (`tol = 0`) — a tuple of preferred actions, one per goal:

```
σ_G(z)  =  ( a*_g(z) )_{g ∈ G} ,        a*_g(z) = argmax_{a ∈ A_g} R(z, a, g)   (S1)
```

**Tolerance signature** (`tol = ε > 0`) — the concatenated Q-vector:

```
q_G(z)  =  [ R(z, a, g) ]_{g ∈ G, a ∈ A_g}  ∈  R^{Σ_g |A_g|}                    (S2)
```

`compression.py::_preferred_signature` / `_q_signature` implement these.

### 2.3 Decision-equivalence and the compression

```
z ~_G z'   ⟺   σ_G(z) = σ_G(z')            (exact)
z ~_G^ε z' ⟺   ‖ q_G(z) − q_G(z') ‖₂ ≤ ε   (tolerance, greedy single-pass merge)
```

The modes `m_1, …, m_M` are the equivalence classes intersected with
`supp(b)`. Mode weight and within-mode conditional:

```
w_m = Σ_{k ∈ m} w_k ,        p(k | m) = w_k / w_m                               (S3)
```

`M ≤ K` by construction, and `M` is bounded by (B1)/(B2) independently of `K`.
Exact (`ε = 0`) merging is order-independent (it is a `groupby` on the signature);
tolerance merging is greedy and order-dependent, which is a known and disclosed
approximation.

### 2.4 How a mode is carried — four rules, and they are not equivalent

This is the part of the method that Gate C0 §S6-S8 turned from a detail into a
central trade-off.

**(a) Representative particle** — put the whole mode's weight on one member.

- `maxweight`: the highest-weight member. Under a flat prior every member ties
  and the tie-break returns the first index, i.e. an arbitrary **mode-edge**
  particle; this systematically **under**-values the mode.
- `centroid` (**current default**): the member nearest the mode's
  belief-weighted mean parameter, `argmin_{k ∈ m} |emb(k) − Σ_i p(i|m) emb(i)|`.
  This systematically **over**-values the mode.

Neither is value-consistent — that is structural, not a bad choice. The
partition (and therefore `M`, and therefore every compute number) is **identical**
under both, so `maxweight → centroid` is free. See §3.4.

**(b) Value-consistent mode summary** (`ModeSummary` in `compression.py`) — do
not collapse onto a particle at all. Carry, per mode:

```
Q_m(a, g)  =  Σ_{k ∈ m} p(k|m) · R(z_k, a, g)              (mode Q-vector)      (S4)
L_m(o | u) =  Σ_{k ∈ m} p(k|m) · P(o | z_k, u)             (mode likelihood)    (S5)
```

Then, **identically** (not approximately):

```
E_b[ R(·, a, g) ]  =  Σ_m w_m Q_m(a, g)                      (commit value)     (S6)
P(o | b, u)        =  Σ_m w_m L_m(o | u)                     (obs marginal)     (S7)
w'_m               =  w_m L_m(o|u) / P(o | b, u)             (mode posterior)   (S8)
```

- `summary`: freeze `p(k|m)` at construction. Root commit value, root observation
  marginals and posterior *mode* weights are exact; post-observation Q-vectors
  are stale. Cost: `O(K)` once per decision to build the tables, `O(M)` per node.
- `summary_exact`: refresh `p(k|m)` after every observation. Exact at every
  depth. Cost: `O(K)` **per node** — which is full-belief cost by construction.

### 2.5 What is exactly lossless, and what is fundamentally lossy

**Exactly lossless — the commit.** By construction, every member of a mode
prefers the same action for every `g ∈ G`. With the summary rules, (S6) makes the
mode-level commit *value* identical to the full-belief commit value, so the
argmax is identical. Gate C0 measures commit regret of exactly **0.00e+00** at
every cell of §S1, at every `K` from 6 to 1536, and the summary's commit-value
residual is `dV0 = 1.11e-16` (floating-point zero).

**Fundamentally lossy — probe valuation.** A probe's value is a comparison of
*values*, not of argmaxes: "commit now" vs "commit after observing". An
observation reweights particles **within** a mode:

```
p'(k | m)  ∝  p(k | m) · P(o | z_k, u)                                          (S9)
```

and the within-mode conditional `p(·|m)` is **precisely the structure the
compression discards by construction**. There is no way to recover it from mode
statistics. Hence:

- Any representative particle mis-values (mode-edge under-values, mode-mean
  over-values) at both budgets.
- A frozen `summary` is exact at the root and **drifts once probing is allowed**
  (measured residual `dV = 3.27e-02`, §3.4).
- `summary_exact` is exact — and costs `O(K)` per node, i.e. it gives back the
  entire compute advantage.

**This trade-off is the honest core of the method.** It is not a bug to be fixed;
it is the price of the abstraction. Where it sits depends on the horizon (§3.5).

### 2.6 Amortized decision-regret VOI

Exact myopic decision-regret VOI for probe `u` under goal `g`:

```
VOI(u)  =  Σ_o P(o|b,u) · max_a E_{b^{u,o}}[R(·,a,g)]  −  max_a E_b[R(·,a,g)]  −  c(u)   (V1)
probe iff  max_u VOI(u) > 0
```

Each inner `max_a E_{b^{u,o}}[·]` is a full best-commit over `K` particles, so
(V1) costs `O(|U| · |O| · K · |A|)` plus `O(|U| · |O| · K)` for the posteriors.
This is the "nested inner planning problem per possible observation" the method
wants to remove.

**Amortized form** — evaluate (V1) at mode granularity, substituting (S6)-(S8):

```
VOI_M(u) = Σ_o [Σ_m w_m L_m(o|u)] · max_a Σ_m w'_m(o) Q_m(a,g)
           − max_a Σ_m w_m Q_m(a,g) − c(u)                                       (V2)
```

Cost: one `O(K)` bucketing/summary pass per decision, then
`O(|U| · |O| · M · |A|)`. Implemented as `AmortizedVOI` in `planners.py`.

**(V2) equals (V1) exactly at the root and is biased below it**, for exactly the
reason in §2.5: `w'_m(o)` is exact, but the post-observation `Q_m` is stale
unless you refresh `p(k|m)`, which costs `O(K)`.

**The learned-scale form (Gate C1, not yet built).** Replace (V2) with a learned
flip-predictor over the *compressed* representation:

```
f_φ( {w_m, Q_m, L_m}_{m=1..M}, u )  →  P̂( argmax_a E_{b^{u,o}}[R] ≠ argmax_a E_b[R] )  (V3)
```

trained by supervision from exact (V1) computed offline on a subsample. This —
a cheap flip-predictor whose input is the `M`-mode decision-clustered structure —
is the second of the two surviving novelty claims (§4). Note the tension it
inherits: (V3)'s input is the representation §2.5 proves is value-inconsistent
for exactly this question.

---

## 3. What is already established

All numbers below are from the gate documents named, all produced by exact
computation (exact Bayes, exact expectimax, exact expected return by
enumeration — no Monte-Carlo). Cells that could not be afforded exactly are
printed `n/a` and never estimated.

### 3.1 Gate B — the exact-POMDP oracle testbed (`gateB_oracle_results.md`, GO)

Two tiny exact POMDPs in pure numpy: `mass_sort` (N objects, hidden binary mass
class, tap probes, force-assignment commit) and `occluder_push` (hidden location
behind an occluder, coarse/fine look probes, push-target commit).

**(i) Compression vs goal-richness.** `richness` = how much of the hidden
parameter the goal family makes decision-relevant.

| task | richness | \|G\| | K | M | M/K | regret_in |
|---|---|---|---|---|---|---|
| mass_sort | 1 | 1 | 16 | 2 | 0.125 | 0.0000 |
| mass_sort | 2 | 2 | 16 | 4 | 0.250 | 0.0000 |
| mass_sort | 3 | 3 | 16 | 8 | 0.500 | 0.0000 |
| mass_sort | 4 | 4 | 16 | 16 | **1.000** | 0.0000 |
| occluder_push | 2 | 2 | 8 | 2 | 0.250 | 0.0000 |
| occluder_push | 4 | 3 | 8 | 4 | 0.500 | 0.0000 |
| occluder_push | 8 | 4 | 8 | 8 | **1.000** | 0.0000 |

**The honest reading, which Gate B states itself:** the compression ratio *is*
the goal-irrelevant fraction of the hidden parameter. `M/K → 1` as the goal
family grows to cover every latent dimension. The method's value is bounded by
how much of the hidden state a realistic decision ignores — a property of the
task distribution, not of the algorithm.

Caveat on the `regret_out` column (≈0 everywhere): measurement (i) isolates the
terminal commit at budget 0, and the commit in these tasks decomposes over the
hidden parameter, so mode representatives preserve the per-dimension MAP even
out of family. The out-of-family cost shows up in *probe valuation*, not the
commit. Do not quote `regret_out ≈ 0` as evidence of out-of-family safety.

**(ii) Probe-policy comparison** (exact expected return net of probe cost; root
compute in primitive ops):

`mass_sort`:

| policy | return | reg_vs_full | E[probes] | compute |
|---|---|---|---|---|
| certainty_equiv | 0.0000 | 1.3000 | 0.0 | 4 |
| entropy_seek | 0.6000 | 0.7000 | 2.0 | 128 |
| voi (exact) | 1.3000 | 0.0000 | 2.0 | 326 |
| **amortized_voi** | **1.3000** | **0.0000** | 2.0 | **198** |
| full_belief | 1.3000 | 0.0000 | 2.0 | 2091 |
| fully_observed | 2.0000 | −0.7000 | 0.0 | 0 |

`occluder_push`:

| policy | return | reg_vs_full | E[probes] | compute |
|---|---|---|---|---|
| certainty_equiv | 0.0000 | 0.7000 | 0.0 | 2 |
| entropy_seek | −0.1000 | 0.8000 | 1.0 | 112 |
| voi (exact) | 0.7000 | 0.0000 | 1.0 | 214 |
| **amortized_voi** | **0.7000** | **0.0000** | 1.0 | **74** |
| full_belief | 0.7000 | 0.0000 | 1.0 | 215 |
| fully_observed | 1.0000 | −0.3000 | 0.0 | 0 |

Both Gate B criteria pass: substantial compression at zero regret, and
decision-regret VOI beats entropy-seeking (which walks into the deliberately
planted "informative but decision-irrelevant" probe) at strictly lower compute
than full-belief planning, with the amortized form matching exact VOI's return
at 61% (mass_sort) / 35% (occluder_push) of its compute.

### 3.2 Gate C0 §S1 — does `M` saturate as `K` grows? (**yes**)

`grid_param` has a genuinely continuous hidden parameter; `K = resolution` is
purely how finely the belief discretizes it, and the goal family lives in
continuous parameter space untouched by the grid.

| axis | K range | M range | M/K @ Kmax | slope(log M / log K) | max \|regret\| |
|---|---|---|---|---|---|
| grid \|G\|=1 | 6–1536 | 3–3 | **0.0020** | **0.0000** | 0.00e+00 |
| grid \|G\|=2 | 6–1536 | 5–5 | 0.0033 | 0.0000 | 0.00e+00 |
| grid \|G\|=4 | 6–1536 | 5–9 | 0.0059 | 0.0565 | 0.00e+00 |
| occluder | 8–256 | 2–2 | 0.0078 | 0.0000 | 0.00e+00 |
| **mass_sort (control)** | 4–256 | **4–256** | **1.0000** | **1.0000** | 0.00e+00 |

`M` matches the analytic bound (B1) **exactly** at every cell. The `mass_sort`
control axis — where `K` grows because new decision-*relevant* latent dimensions
are added, so every object matters to some goal — has `M` growing like `K` with
log-log slope 1.000. That control is what stops §S1 being a tautology, and it is
also the pre-registered adversarial case (§6, §7).

§S5 confirms the joint picture: `M` equals `min(K, |G|(|A|−1)+1)` for
`|G| ∈ {1,2,4,8}` at every `K ∈ {24, 96, 384, 1536}` — flat in `K`, linear in
`|G|`. §S4 confirms `M` tracks `|A|` exactly at fixed `K = 384`.

### 3.3 Gate C0 §S2/§S3/§S3b — compute

At `H = 1`, sweeping `K` (`|G| = 1`, `M = 3`):

| K | C_full | C_cached | ratio |
|---|---|---|---|
| 12 | 708 | 198 | 3.58x |
| 192 | 11,148 | 378 | 29.49x |
| 1536 | 89,100 | 1,722 | **51.74x** |

log-log slopes of root compute vs `K`: `full_belief` **0.997**,
`compression_cached` **0.441**, `compression` (signatures built at decision time)
0.644.

At `K = 192`, sweeping `H`: **2.86x** at `H = 0` → **60.02x** at `H = 4`
(wall clock 7.0063 s → 1.0143 s).

The joint `(K, H)` ratio grid (§S3b) is the most load-bearing table in the study:

| K | M | K/M | H=0 | H=1 | H=2 | H=3 | H=4 |
|---|---|---|---|---|---|---|---|
| 24 | 3 | 8 | 2.15 | 6.69 | 7.48 | 7.55 | 7.56 |
| 96 | 3 | 32 | 2.73 | 19.79 | 28.76 | 29.94 | 30.05 |
| 384 | 3 | 128 | 2.93 | 39.09 | 101.65 | 118.12 | 119.88 |
| 1536 | 3 | 512 | 2.98 | 51.74 | 278.43 | **450.51** | n/a |

**Read it both ways.** Down a column (fixed `H`, growing `K`) the ratio
*flattens*: the compressed planner still reads the `K`-particle belief once per
decision — an unavoidable `O(K)` term. Across a row (fixed `K`, growing `H`) it
climbs toward the ceiling `K/M`, because the `O(K)` read amortizes over an
exponentially growing tree whose every node costs `O(M)` instead of `O(K)`.
**Tree cost drops from `O(K · tree)` to `O(M · tree)`; belief cost stays `O(K)`.**

### 3.4 Gate C0 §S6/§S7 — decision fidelity is not value fidelity

Value-consistency residuals, worst over the §S6 sweep. `dV0` = commit-only
(budget 0), `dV` = probing allowed (budget 1). Anything ≤ 1e-9 is
floating-point zero:

| rule | dV0 | dV | verdict |
|---|---|---|---|
| `maxweight` | 1.25e-01 | 1.79e-01 | value-inconsistent everywhere |
| `centroid` | 1.25e-01 | 2.40e-01 | value-inconsistent everywhere |
| `summary` | **1.11e-16** | **3.27e-02** | exact for the commit, **drifts under probing** |
| `summary_exact` | 1.11e-16 | 7.77e-15 | exact at every budget |

Worst exact closed-loop regret over the whole 48-cell `(|A|, |G|, K)` grid (§S7):

| rule | worst regret | nonzero cells |
|---|---|---|
| `maxweight` | **44.74%** | 33 / 48 |
| `centroid` | 0.56% | 3 / 48 |
| `summary` | 0.00% | 0 / 48 |
| `summary_exact` | 0.00% | 0 / 48 |

The **44.74%** figure is the headline failure and it deserves its own sentence:
under a flat prior, `maxweight` ties across the whole mode and the tie-break
returns an arbitrary mode-*edge* particle, which under-values the mode enough to
flip the root action (2/12 cells in §S6 disagreed with the oracle). The mode
partition is identical under all four rules, so `maxweight → centroid` is a
**free** fix — same `M`, byte-identical compute, 44.74% → 0.56%. `centroid` is
therefore now the default `rep_rule`; `maxweight` stays selectable so the failure
remains reproducible.

But `centroid` is not value-consistent either (`dV = 2.40e-01`, *worse* than
`maxweight`'s `dV`), and no representative particle can be. That is the §2.5
argument, confirmed empirically.

### 3.5 Gate C0 §S8 — what value-consistency costs

`C_full / C_compressed`, from the shallowest to the deepest affordable horizon:

```
  K=24:   centroid 2.1x → 7.6x   summary 0.7x → 7.6x     summary_exact ~0.9x
  K=96:   centroid 2.7x → 30.1x  summary 0.7x → 29.9x    summary_exact ~1.0x
  K=384:  centroid 2.9x → 119.9x summary 0.7x → 117.4x   summary_exact ~1.0x
  K=1536: centroid 3.0x → 450.5x summary 0.7x → 242.0x   summary_exact ~1.0x
```

Three facts:

1. **At `H ≤ 1` the frozen `summary` is a net loss** — 0.69–0.75x, i.e. *worse
   than planning the full belief*. Its `O(K(|A| + Σ_u |O_u|))` build (here 3
   controllers + 11 sensor bins ≈ 14x the bucketing pass) is not amortized over
   enough tree. At `H ≤ 1` you should use `centroid`, which is free and worst-case
   0.56% over the grid.
2. **At `H ≥ 2` the frozen `summary` is close to free** and gives 0.00% regret
   with exact commit values. That is the configuration to prefer.
3. **`summary_exact` lands at ≈1.0x vs full belief at every measured cell** — it
   is the auditable zero-regret reference, not a production planner.

### 3.6 The verdict Gate C0 actually reached

**STRONG-BUT-NARROW**, split into three separately-reported questions:

- **Scaling: STRONG.** `M` saturates on every resolution-refinement axis, the
  control axis behaves as predicted, commit regret is exactly 0, and the compute
  ratio widens in both `K` and `H`.
- **Decision fidelity: LOSSY for any representative particle; exactly
  recoverable by a value-consistent summary, at a compute cost that erases the
  win if you want the guarantee under probing.**
- **Interpretation: the bound is elementary** (a two-line counting argument), its
  enabling regime `K >> |G|(|A|−1)+1` is *unverified for a learned belief*, and
  **the win is in the planning tree, not in belief maintenance** — the `O(K)`
  belief read is unavoidable and at visual scale it is the dominant cost.

### 3.7 A documentation inconsistency (resolved)

`gateC1_design.md` was written against the pre-`centroid` state of the study and
characterized Gate C0 as returning **WEAK** with "1.1% lossy closed-loop", in two
places (its prior-gates line and its "what Gate C1 has to buy" paragraph). The
current `gateC0_scaling_results.md` reads **STRONG-BUT-NARROW** with `centroid`
worst-case 0.56% and `summary` 0.00%. **Both passages have been corrected**; the
C1 GO/STOP criteria were never affected. Recorded here because the two documents
were mutually inconsistent for one commit, and because the corrected text is the
one that should be quoted: the C0 verdict is three-part (scaling STRONG, fidelity
lossy-but-recoverable-at-a-cost, interpretation narrow), never a single word.

---

## 4. Honest novelty position

Gate A's audit (`gateA_novelty_matrix.md`) returned **conditional GO, narrowly
scoped**. Restated and refined here.

### 4.1 The matrix, condensed

| Prior-art line | Compression with a regret bound? | Amortized VOI w/o nested planning? | Goal-family general? | Continuous visual POMDP? |
|---|---|---|---|---|
| Ferns et al. bisimulation metrics for continuous MDPs ([PDF](https://www.cs.mcgill.ca/~prakash/Pubs/siamFP11.pdf)); DeepMDP ([1906.02736](https://arxiv.org/abs/1906.02736)); causal-state POMDP reps ([1906.10437](https://arxiv.org/abs/1906.10437)); bisimulation-for-MPC ([2410.04553](https://arxiv.org/abs/2410.04553)) | **P** — bounds \|V*(s)−V*(s')\| by a state metric, single fixed reward, state- not decision-signature-based | N | N | P |
| **ANPL "Simplified POMDP" program** — Adaptive Information BSP ([2201.05673](https://arxiv.org/abs/2201.05673)); **Simplifying Complex Observation Models, AAAI'24 ([2311.07745](https://arxiv.org/abs/2311.07745))**; anytime deterministic guarantees ([2310.01791](https://arxiv.org/abs/2310.01791)); risk-averse simplification ([2406.03000](https://arxiv.org/abs/2406.03000)); action-consistency in Dec-POMDPs ([2403.05962](https://arxiv.org/abs/2403.05962)); particle-belief-MDP optimality ([2210.05015](https://arxiv.org/abs/2210.05015)) | **Y, for a single fixed reward** — including belief/particle aggregation, with *planner-realized* bounds and literal same-action guarantees, in continuous / camera-image observation spaces | N | **N** | **Y** |
| Value-directed compression: Poupart & Boutilier NeurIPS'02 ([page](https://proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html)); Roy & Gordon E-PCA ([1107.0053](https://arxiv.org/abs/1107.0053)); linear belief compression re-examined ([1508.00986](https://arxiv.org/abs/1508.00986)) | **P** — value-preserving belief-simplex compression, but linear, exact-model, tabular | N | N | N |
| Task-aware belief representation: Gangwani et al., PMLR v115 ([page](https://proceedings.mlr.press/v115/gangwani20a.html) / [1906.09510](https://arxiv.org/abs/1906.09510)) | N (no bound, no hypothesis clustering) | N | N | P |
| Value Equivalence Principle ([2011.03506](https://arxiv.org/abs/2011.03506)); Proper Value Equivalence ([2106.10316](https://arxiv.org/abs/2106.10316)) | **The general form of "equivalence by decision-relevant quantity"** — our criterion is an instantiation, not a new principle | N | P (defined over a *set* of (π,v), never instantiated as a goal family) | N |
| Ran Wei, VOI & reward specification in active inference / POMDPs ([2408.06542](https://arxiv.org/abs/2408.06542)) | N | **P — strongest threat**: EFE's epistemic term is already a closed-form non-nested surrogate for Bayes-optimal information value, with the optimality gap quantified, *in belief-MDP language* | N | N |
| Deep Adaptive Design ([2103.02438](https://arxiv.org/abs/2103.02438)), iDAD ([page](https://proceedings.neurips.cc/paper/2021/file/d811406316b669ad3d370d78b51b1d2e-Paper.pdf)) | N | **Y, for information gain** — amortized design network, no nested per-step optimization | N | P |
| Deep active inference on a real robot ([2512.01924](https://arxiv.org/abs/2512.01924)) | N | P — VQ "abstract actions" compress the *action* space for cheap selection | N | **Y** |
| Particle belief in online solvers: DESPOT ([JAIR](https://www.jair.org/index.php/jair/article/download/11043/26215/20559)); POMCGS graph merging ([2507.20951](https://arxiv.org/abs/2507.20951)) | N — POMCGS merges by observation-outcome proximity, efficiency-only, no bound; degrades above modest observation dimension | N | N | N |
| Learned latent belief filters: PlaNet/Dreamer RSSM ([1811.04551](https://arxiv.org/abs/1811.04551), [2010.02193](https://arxiv.org/abs/2010.02193), [2301.04104](https://arxiv.org/abs/2301.04104)); Wasserstein Believer ([2303.03284](https://arxiv.org/abs/2303.03284)); flow/Stein-variational belief ([2510.21107](https://arxiv.org/abs/2510.21107)) | P (Wasserstein Believer has a bisimulation-style value guarantee) | N | N | **Y — commodity technology; must not be re-claimed** |

### 4.2 What must be conceded, explicitly, in the paper

These are not optional citations. A knowledgeable POMDP-theory or
active-inference reviewer will raise every one of them.

1. **The compression bound is not ours.** The Technion **ANPL "Simplified
   POMDP"** program — chiefly **Lev-Yehudi, Barenboim & Indelman, AAAI 2024,
   [arXiv:2311.07745](https://arxiv.org/abs/2311.07745)**, with
   [2310.01791](https://arxiv.org/abs/2310.01791) and
   [2201.05673](https://arxiv.org/abs/2201.05673) — already proves formal,
   *planner-realized* performance bounds for replacing any POMDP component
   (belief/particle aggregation included) with a cheaper one, in continuous
   high-dimensional observation spaces, up to literal same-action guarantees. Our
   bound (B1) is very likely a two-line corollary of their machinery.
   **Position pillar 1 as a special case, not as a new theorem.**
2. **Bisimulation metrics are the secondary ancestor.** Ferns et al. bound
   `|V*(s) − V*(s')|` by a state metric; ours is a decision-signature relation
   on beliefs rather than a state metric, and it is defined jointly over a goal
   family — but the *shape* of the argument is theirs.
3. **Amortized VOI without nested planning is not ours.** **Deep Adaptive
   Design** ([2103.02438](https://arxiv.org/abs/2103.02438)) already trains an
   offline design network against a contrastive information-gain bound and
   deploys in milliseconds. **Ran Wei
   ([2408.06542](https://arxiv.org/abs/2408.06542))** already shows EFE's
   epistemic term is a closed-form non-nested surrogate for Bayes-optimal
   information value, *in belief-MDP language*, and quantifies its optimality gap.
   Our contribution is the *substrate* (`M`-mode compressed structure) and the
   *target* (argmax-flip, not generic information gain), not the amortization.
4. **Value-directed belief compression is not ours.** Poupart & Boutilier
   (NeurIPS 2002), Roy & Gordon ([1107.0053](https://arxiv.org/abs/1107.0053)),
   Wang et al. ([1508.00986](https://arxiv.org/abs/1508.00986)) established
   "compress the belief simplex so that value/policy is preserved" two decades
   ago.
5. **Task-aware belief representation is not ours.** Gangwani et al.
   (PMLR v115) already shape a belief module by a downstream task loss.
6. **"Equivalence by decision-relevant quantity" is the Value Equivalence
   Principle** ([2011.03506](https://arxiv.org/abs/2011.03506)). Ours is an
   instantiation.
7. **The belief filter is commodity.** RSSM / PlaNet / DreamerV3 and the
   flow/Stein-variational successors are off-the-shelf. Reinventing one is the
   fastest way to burn the budget and lose the paper.

### 4.3 The two surviving differentiators

**D1 — multi-goal (family-of-goals) invariance of the compression, with a bound
that must hold jointly across the family.** Every value/regret-preserving
compression result found (bisimulation, ANPL simplification, value-directed
compression, Value Equivalence) is proven or instantiated for *one reward at a
time*. Nothing surveyed states a compression bound that must hold simultaneously
across a goal distribution, with an `ε` accounting for goal-family diameter.
Value Equivalence is the closest relative — its equivalence is stated over an
arbitrary *set* of (policy, value-function) pairs, which could be instantiated
as one value function per goal — but the paper never does this. This is a real,
citable gap, and it is the headline claim.

*Caveat:* it is open largely because **nobody bothered to test multi-goal
generality**, not because there is evidence it is deep or hard. Goal-conditioned
value functions (UVFAs) are commodity, so `Q(z,a,g)` is not new; only "compress
so the ranking of `Q(z,·,g)` is preserved for *every* `g` in a family, with a
bound" appears unclaimed.

**D2 — using the compressed decision-mode structure itself as the substrate for
an amortized flip-prediction VOI.** DAD amortizes generic info-gain from full
posteriors; Ran Wei's EFE analysis is a scalar closed-form surrogate. Making the
amortized VOI cheap *because* it only reasons over `M << K` modes, and targeting
"would the argmax flip" rather than "how many nats would I learn", is a concrete
architectural combination no single retrieved paper makes.

### 4.4 **Gate C0 §S6-S8 weakened D2. This must be stated, not buried.**

D2's whole pitch is "the compressed representation makes probe valuation cheap."
Gate C0 proved that **the compressed representation is exactly the object that
mis-values probes**:

- Probe decisions are *value* comparisons; compression preserves the *argmax*,
  not the value (§2.5, §3.4).
- The cheap carriers (`centroid`, frozen `summary`) have **no guarantee** under
  probing: `dV = 2.40e-01` and `3.27e-02` respectively.
- The carrier with a guarantee (`summary_exact`, `dV = 7.77e-15`) costs
  **`O(K)` per node — i.e. exactly full-belief cost, ≈1.0x, no win at all** (§3.5).
- The frozen `summary` is a **net loss** at `H ≤ 1` (0.69–0.75x vs full belief).

So D2 is a **trade-off with an operating regime (`H ≥ 2`), not a clean win.** The
defensible claim is: *"on the compressed structure, probe valuation is
`|A|+Σ|O|` times cheaper per node and empirically regret-free once the search
tree is deep enough to amortize the summary build, but it is provably
value-inconsistent and we characterize exactly where that bites."* Anything
stronger is overclaiming and will be caught.

### 4.5 Framing that survives review, and framing that does not

**Survives:** *"Decision-signature clustering across a goal family, with an
amortized flip-predictor operating on the compressed modes — positioned as an
instantiation of Value Equivalence and a corollary of the Simplified-POMDP
framework, with a measured characterization of when the compressed
representation's value-inconsistency does and does not cost you a decision."*

**Does not survive:** *"We invented regret-bounded belief compression"* /
*"We invented amortized VOI"* / *"belief compression makes belief-space planning
tractable"* (it makes the *tree* cheaper; the `O(K)` filter read is untouched,
and at visual scale that read is the dominant cost).

---

## 5. Experimental plan

Full detail in `gateC1_design.md`; this is the operative summary plus the
substrate facts that must not drift.

### 5.1 Substrate

**PRIMARY — MIKASA-Robo** ([arXiv:2502.10550](https://arxiv.org/abs/2502.10550),
ICLR 2026, MIT licence, [github.com/CognitiveAISystems/MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo),
`pip install mikasa-robo-suite`), built on ManiSkill3
([arXiv:2410.00425](https://arxiv.org/abs/2410.00425),
[repo](https://github.com/mani-skill/ManiSkill), GPU-parallel rendering).

- **`BatteriesChecker{Easy,Hard}-{3,6}` — the only genuine probe task.** *"Find
  all working batteries by inserting each one into the socket, observing the
  lamp result, and then pressing the button to confirm."* Structurally
  isomorphic to Gate B's `mass_sort`: N objects with a hidden binary property
  observable only by an action, a terminal commit, irreversible cost for a wrong
  commit.
- **`ShellGameColorLampTouch` / `ShellGameShuffleColorLampTouch` — the goal-family
  task.** *"Touch the cup matching the lamp colour"*: the lamp colour is a goal
  variable over a **shared** hidden state (which cup hides what). A textbook goal
  family.
- **Do not misrepresent MIKASA-Robo as an information-gathering benchmark.** The
  other 9 memory types (`RememberColor*`, `RememberShape*`, `FindImposter*`,
  `BunchOfColors*`, `TraceShape*`, `BlinkCount*`, …) are *memory*: observe → wait
  → recall. The agent never chooses what to find out. Only the BatteriesChecker
  family has genuine active discovery.

**SECONDARY (the `K`/`H` scale axis) — POPGym Arcade**
([arXiv:2503.01450](https://arxiv.org/abs/2503.01450), MIT,
[repo](https://github.com/bolt-research/popgym-arcade)): `BattleShip`,
`MineSweeper`, `Navigator`, `CountRecall`, all difficulties. JAX-jitted at ~10M
FPS — the only place we can afford to sweep `K` over two orders of magnitude *and*
`H` to depth 5 *and* still get clustered CIs. Ships paired fully-observable
variants of every env, giving the `fully_observed` upper bound free. Pixel-based,
so the belief is genuinely learned from images.

**FALLBACK (bespoke, used only if needed) — `MassSortProbe-v0`**: IMBench task
T25 reimplemented to its **published spec** in ManiSkill3 — three visually
identical cubes at 50/100/200 g in randomized order, three labelled zones, wrist
force/torque as the only mass channel, wrong-zone placement = immediate failure.

> **IMBench cannot be the substrate and cannot be cited normally.** Verified
> 2026-07-30: `imbench.org` hosts a real paper PDF, but it is an **ANONYMOUS
> CoRL 2026 submission** ("Submitted to the 10th Conference on Robot Learning
> (CoRL 2026). Do not distribute."; BibTeX author literally `Anonymous`).
> **There is no arXiv posting** — the ID `2607.15641` that circulated internally
> does not resolve to IMBench and no arXiv version exists. **There is no
> environment code**: no GitHub repository, no MJCF assets, no reset/randomization
> implementation, no success predicates as code, no evaluation harness. Only
> **10 demonstration episodes per task** are public (Appendix B: *"We release a
> subset of the dataset (10 episodes) via Hugging Face for review purposes… The
> complete benchmark and full dataset will be released upon acceptance."*). Ten
> offline episodes cannot support a planner. IMBench is usable **as a
> specification only**, must be cited as an anonymous unpublished submission with
> its URL, and any reimplementation must say plainly that it is ours.

Mitigations against "you designed the task to win", all pre-committed:
(i) established suites are primary and no headline claim rests on the bespoke
task alone; (ii) implement to *someone else's* published spec; (iii) swap-in
clause — if IMBench releases before submission, rerun on their environment and
report both; (iv) ship the adversarial control (§6); (v) release the env code.

**Explicitly rejected:** IMBench itself (no code); robosuite/RoboCasa alone (no
POMDP tasks); Habitat (hidden state is map geometry — no compact hypothesis set,
no goal family); original POPGym (no pixels).
**Held in reserve:** Tactile MNIST
([2506.06361](https://arxiv.org/abs/2506.06361),
[repo](https://github.com/TimSchneider42/tactile-mnist)) + APPLE
([2505.06182](https://arxiv.org/abs/2505.06182), ICLR 2026) as a third domain if
reviewers push on generality.

**The gap no released benchmark closes.** Our two novelties need **(P)** a
probing action that reveals hidden state and **(G)** a goal family over the same
hidden state, *in the same task*:

| | (P) probing | (G) goal family |
|---|---|---|
| MIKASA BatteriesChecker | YES | NO |
| POPGym Arcade BattleShip / MineSweeper | YES | NO |
| Tactile MNIST | YES | PARTIAL |
| MIKASA ShellGame-ColorLamp | NO | YES |
| POPGym Arcade Navigator | PARTIAL | YES |
| IMBench mass-sort (spec only, unreleased) | YES | YES |

**No released benchmark gives (P) and (G) simultaneously.** The two pillars are
therefore evaluated on *different established tasks*, and only the clearly
labelled reimplementation carries both. This is a load-bearing limitation and
goes in the paper's limitations section verbatim.

### 5.2 What is learned vs borrowed

**Borrowed, unmodified:** DreamerV3 ([2301.04104](https://arxiv.org/abs/2301.04104),
[repo](https://github.com/danijar/dreamerv3), MIT) as the belief model — its
**categorical (32x32 discrete) latent** matters, because categorical posteriors
are genuinely multimodal, which is what makes `K` *diverse* hypotheses possible
at all. `K` hypotheses = `K` samples from the posterior/prior categorical, or `K`
particles in a bootstrap filter using the RSSM as proposal. One alternative
filter (`NM512/dreamerv3-torch` or `glambrechts/informed-dreamer`) as a
robustness check. Simulators unmodified. MIKASA-Robo's released oracle
trajectories (22.5k PPO / motion-planning, RLDS + LeRobot v3) so the belief model
is trained from data rather than solved by RL from scratch. `pymdp`
([repo](https://github.com/infer-actively/pymdp), MIT) to sanity-check the EFE
baseline on a tabular version before trusting the deep one. This repo's
`ComputeCounter`, extended to NN FLOPs.

**Built by us — the entire contribution surface:**
1. Goal-conditioned action-value head `Q(z, a, g)` over the frozen belief latent
   (UVFA-style), needed to *define* a decision signature at all.
2. Decision-signature extractor + clustering with the `ε`-tolerance sweep.
3. **Calibrated within-mode value estimator** — required, not optional, per §2.5
   and §3.4.
4. Amortized flip-predictor (V3).
5. The compute-accounting harness.

### 5.3 Primary metric — a compute-vs-quality frontier, not a success-rate table

**x-axis: planning compute per decision, in three registers, always all three.**

1. `C_prim` — hardware-independent primitive ops (this repo's counter) **plus NN
   cost in FLOPs** (encoder forwards x FLOPs/forward, predictor rollouts x
   FLOPs/rollout).
2. `C_wall` — median ms/decision on one fixed GPU, batched identically across
   methods.
3. **`C_split` — the decomposition `C_belief` (encoder + filter update) vs
   `C_search` (planning). Mandatory.** This is how the program most plausibly
   dies (§7, STOP S3; §8, R2).

**y-axis: quality, two curves per method.**

1. Normalized task return (benchmark-native success predicate: MIKASA-Robo's own
   protocol; POPGym Arcade's standardized returns).
2. **Decision regret** vs the full-particle-belief planner at maximum affordable
   compute, plus (POPGym Arcade only, free) regret vs the fully-observable MDP
   variant as an upper bound.

**Sweep grid.** `K ∈ {8, 16, 32, 64, 128, 256, 512}` (POPGym Arcade extends to
2048); planner budget (CEM iterations x population) at 4 settings;
`H ∈ {1, 2, 3, 4, 5}`; **5 seeds**; **≥100 evaluation episodes per cell**.

**CIs: trajectory-clustered bootstrap** (repo convention — see
`diagnosis/metrics/bootstrap.py`). Resample *trajectories*, not transitions;
episodes from the same task instance are one cluster.

**Headline statistics (these are the abstract):**

- **Compute at iso-quality:** total `C_prim` to reach 95% of the full-belief
  planner's best return; report the ratio.
- **Quality at iso-compute:** return at matched `C_prim`.
- **Frontier slope:** log-log slope of the iso-quality compute ratio vs `K`. **A
  flat slope means a constant factor and the scalability claim is dead.**
- **Horizon behaviour:** the `(K, H)` ratio grid, exactly as §S3b — that table is
  where the asymptotic story lives, and where the `H ≥ 2` operating regime for
  the summary must be shown.

**Reporting rule inherited from Gate C0:** never estimate a cell you cannot
afford to measure. Print `n/a`. And report both `C_comp` (signatures built at
decision time) and `C_cached` (precomputed) — **`C_comp` is the headline**,
because at visual scale the belief is new every step and caching may not apply.

---

## 6. Baselines

Twelve entries. Each names what it controls for; several are non-negotiable.

| # | Baseline | Why it is required / what it controls for |
|---|---|---|
| **B1** | **Full-particle-belief planner (ceiling)** — exact expectimax / CEM over all `K` particles; POMCPOW / PFT-DPW from `POMDPs.jl` on a tabularized version as a sanity anchor | The quality ceiling **and the compute denominator**. Every frontier claim is stated relative to this. Without it there is no "iso-quality". |
| **B2** | **Certainty-equivalent MPC** — MAP collapse + CEM, i.e. the standard visual-MPC setup this repo already runs | The floor, and the honest statement of what practitioners actually do. Establishes that partial observability *costs* something on these tasks — if certainty-equivalent already matches B1, the task has no belief content and the whole study is moot. |
| **B3** | **Entropy / info-gain probing** — greedy max expected posterior-entropy reduction | Gate B's planted trap: a probe that is highly informative but decision-irrelevant. Entropy-seeking walks into it (returns **0.6000** vs VOI's 1.3000 on `mass_sort`; **−0.1000** vs 0.7000 on `occluder_push`). Controls for "is decision-relevance doing anything, or would any information-seeking do?" |
| **B4** | **Active-inference EFE planner** — expected free energy (pragmatic + epistemic) over the *same* RSSM, CEM search; tabular cross-check with `pymdp` | **Non-negotiable.** Gate A names Ran Wei ([2408.06542](https://arxiv.org/abs/2408.06542)) as the single strongest threat to D2: EFE's epistemic term is *already* a closed-form non-nested VOI surrogate in belief-MDP language. This is the head-to-head that decides whether D2 survives. |
| **B5** | **DAD-style amortized info-gain** — a design network trained offline on a contrastive information-gain bound ([2103.02438](https://arxiv.org/abs/2103.02438)) | The *amortized* version of B3, and the prior-art threat to "VOI without nested planning". Isolates our claim to the *target* (argmax-flip) and *substrate* (M modes), not to amortization per se. |
| **B6** | **RSSM / Dreamer actor-critic** ([2301.04104](https://arxiv.org/abs/2301.04104)), unchanged | Can a strong commodity agent just solve these tasks with no explicit belief-hypothesis machinery? If yes everywhere, the entire framing is decoration. |
| **B7** | **Multimodal predictor without an explicit hypothesis set** — diffusion/flow or categorical-mixture next-latent predictor + CEM (UWM / mixture-of-predictions style) | Tests whether explicit `K` hypotheses are needed at all, or whether an implicitly multimodal predictor already captures the ambiguity. *Citation note: the repo's concurrent-work memo (`concurrent-work-uwm-jepa-atm`) tracks this cluster; verify the exact arXiv IDs before citing — they are not on file here.* |
| **B8** | **Compression WITHOUT active probing** (ablation A1) — plan over `M` modes, certainty-equivalent, no VOI | Separates the two pillars. Does compression alone buy anything, or is all the gain from probing? |
| **B9** | **Probing WITHOUT compression** (ablation A2) — exact decision-regret VOI over all `K` particles | The quality ceiling for the VOI pillar and the exact compute we claim to remove. Also the supervision source for the flip-predictor. |
| **B10** | **Random subsample `K → M`** at matched `M` and matched compute (ablation A3) | **The single most important control.** If random `M`-particle subsampling matches decision-signature clustering, the criterion contributes nothing and the paper is "use fewer particles". **Pre-registered STOP condition S4.** |
| **B11** | **Latent k-means `→ M`** at matched `M` (ablation A4) | Direct head-to-head against Gate A's largest prior-art threat: state-similarity abstraction (bisimulation / Ferns / DeepMDP / ANPL). Decision-signature clustering must beat latent k-means at matched `M`, or pillar 1 reduces to known state abstraction. |
| **B12** | **Oracle fully-observed upper bound** | Free on POPGym Arcade (paired MDP variants); logged from sim state elsewhere. Bounds how much of the residual gap is *belief* vs *control*. Gate B reports it (`fully_observed`: 2.0000 / 1.0000) and it is what reveals R1's failure signature (§8). |
| **B13** | **A published active-perception number** — **APPLE** ([2505.06182](https://arxiv.org/abs/2505.06182), ICLR 2026) on Tactile MNIST; MIKASA-Robo's own reported PPO+memory baselines on MIKASA tasks | Repo convention: *always report next to the published baseline.* Anchors against a real number rather than only our own reimplementations. |

**Further ablations:** A5 — per-goal recomputation vs one shared partition
(this *is* the D1 test, feeds gate G4). A6 — signature cost accounted
(`C_comp`) vs cached (`C_cached`), both columns always shown.

### The pre-registered ADVERSARIAL control

**`BatteriesChecker` measured against the *true* hidden state is a task where our
method should NOT win, and we say so before running it.** Its hidden state is an
N-bit vector in which *every* bit is decision-relevant to the native goal
("report all working batteries"). That is exactly Gate C0's control axis
(`mass_sort(N, all-relevant)`: log-log slope of `M` vs `K` = **1.000**,
`M/K` = **1.0**). Under that reading `M = K` and the method buys **nothing — by
design, not by failure.**

Publishing a task on which we *predict* failure, and then failing there, is the
strongest available answer to "you built the benchmark to win". This mirrors
exactly what made Gate C0 credible.

The learned-visual reframing that keeps BatteriesChecker informative is that at
visual scale **`K` is a property of the filter, not of the task** — `K` is how
many latent particles the belief model carries, a resolution knob exactly like
`grid_param`'s. **Both readings must be tested and both reported.** And if the
method *does* win under the all-bits-relevant reading, that is a **bug report,
not a result** (STOP S6).

---

## 7. Gates & pre-registered GO/STOP

Fixed before any run. All thresholds on trajectory-clustered bootstrap 95% CIs.

### P0 — diversity smoke test. Run this FIRST, before anything else is built.

**This is the load-bearing risk and it is cheap to falsify.** The premise needs
the learned belief to carry `K` genuinely *decision-diverse* hypotheses, with
`K >> |G|(|A|−1)+1`:

- If the learned hypotheses are near-identical → `M = 1`, "compression" is
  trivially lossless and completely **vacuous**.
- If `K` is small (a learned filter typically carries a handful of components)
  and `|G|(|A|−1)+1` is comparable → `M = K` and **there is no compression**.

**Procedure.** Train or borrow one DreamerV3 on `BatteriesCheckerEasy-3` and
`BattleShipEasy`. Sample `K` latents. Measure (a) effective sample size of the
particle weights, and (b) the number of *distinct decision signatures* computed
under the **oracle** hidden state. ~1 week, ~50 GPU-hours.

**This is STOP criterion S5, run before P1.**

### GO requires all four

- **G1 — compression is real, and it is *our* criterion doing the work.**
  On ≥2 of 3 task families: `M/K ≤ 0.25` at `K ≥ 128`, closed-loop return within
  **5%** of B1 (CI excluding a 10% drop). **And** decision-signature clustering
  beats **both** B10 (random subsample) **and** B11 (latent k-means) at matched
  `M` with **non-overlapping CIs**. Without the B10/B11 margin, G1 fails even if
  `M/K` is tiny.
- **G2 — the compute win is asymptotic, not a constant.** At the largest feasible
  `K`: ≥**5x** reduction in **total** `C_prim` (belief + search, using `C_comp`
  not `C_cached`) at iso-quality (≥95% of B1's best return), **and** log-log slope
  of the iso-quality compute ratio vs `K` ≥ **0.3**.
- **G3 — the amortized VOI pillar holds.** Amortized flip-predictor VOI within
  **3%** of exact decision-regret VOI (B9) return at ≥**5x** lower compute; **and**
  beating B3+B5 (both entropy variants) *and* **B4 (EFE)** by ≥**15% relative
  return** on the probing tasks, non-overlapping CIs. B4 is the discriminating
  comparison.
- **G4 — multi-goal invariance (the headline novelty).** One `M`-mode partition
  computed once for a goal family must hold on **held-out goals from that family**
  with ≤**10%** regret inflation vs per-goal recomputation (A5). Measured on
  ShellGame-ColorLamp (established) and, if used, `MassSortProbe-v0`.

### STOP if any

- **S1 — nothing to compress.** `M/K > 0.5` at `K ≥ 128` on all three families.
- **S2 — constant factor only.** Iso-quality compute ratio `< 2x`, or its slope
  vs `K` `≤ 0.1`.
- **S3 — the filter eats the win.** `C_belief / (C_belief + C_search) ≥ 0.8` at
  the operating point **and** still ≥0.8 at the largest `H`. Then the search
  saving is irrelevant to end-to-end cost. *(Pivot option, not a free pass: this
  becomes a paper about belief-model cost, which is not the paper Gate A cleared.)*
- **S4 — the criterion adds nothing.** B10 (random subsample → `M`) matches
  decision-signature clustering within CI on ≥2 families.
- **S5 — mode collapse in the learned belief.** ESS of the particle weights
  **and** the number of distinct oracle-labelled decision signatures `< 2` on the
  probing tasks. A **"fix the filter first"** stop — not necessarily a program
  kill, but a hard stop on reporting any Gate C1 number.
- **S6 — control task inverted.** If the method *wins* on `BatteriesChecker`
  under the all-bits-relevant reading, treat it as a **bug**: Gate C0's control
  axis predicts `M = K` there. An unexplained win means the compute accounting or
  the signature extraction is wrong.

### The conditional outcome — state it plainly, do not let it drift

**If G1-G3 pass but G4 fails, the paper loses one of its two surviving
novelties.** Gate A established that *single-goal* regret-preserving belief
simplification is already covered by the ANPL program
([2311.07745](https://arxiv.org/abs/2311.07745),
[2310.01791](https://arxiv.org/abs/2310.01791)) and by bisimulation metrics. A G4
failure downgrades the work to D2 alone — and §4.4 already showed D2 is a
trade-off, not a clean win. That combination is a **workshop-scale contribution,
not the paper.** Decide explicitly at that point rather than drifting.

---

## 8. Risks & open questions

Blunt, worst first.

**R1 — The elementary-bound problem.** (B1) is a two-line counting argument.
Gate C0 says so in its own verdict: *"The empirical study confirms the bound and
rules out tautology via the control axis; it does not discover a surprising
law."* The theoretical contribution is therefore near-zero, and pillar 1's
*bound* is very likely a corollary of [2311.07745](https://arxiv.org/abs/2311.07745).
The paper must be an **empirical/systems** contribution about *when* the regime
holds and what it costs — not a theory paper. If a reviewer wants theory, we lose.

**R2 — The `K`-at-visual-scale question is unresolved and load-bearing.** The
whole method needs `K >> |G|(|A|−1)+1`. That is trivially satisfied by an oracle
particle filter at `K = 1536`. It is **unknown** for a learned filter, which
typically carries 10–20 effective components. If `K ~ 16` and
`|G|(|A|−1)+1 ~ 13`, then `M ≈ K` and the entire compute advantage measured in
Gate C0 vanishes. **P0 exists to falsify this cheaply and must run first.**
Failure signature to watch for: regret vs B1 near 0 *and* regret vs the
fully-observed upper bound (B12) enormous — that means the belief never
represented the ambiguity, so "compression" was lossless because there was
nothing there.

**R3 — The `O(K)` belief read caps the win, and at visual scale it caps it
early.** Gate C0 §S3b shows the ratio flattening along the fixed-`H` axis: the
compressed planner still reads all `K` particles once per decision. On the toy, a
"particle touch" is a float multiply. At visual scale it is an **encoder forward
pass**, orders of magnitude more expensive than a tree node. This is STOP S3 and
it is the most likely honest way this gate fails. Honest characterization to
carry into the paper: **tree cost drops from `O(K·tree)` to `O(M·tree)`; belief
cost stays `O(K)`.**

**R4 — The `H ≥ 2` operating regime; the cheap summary has no guarantee.**
§S8: at `H ≤ 1` the value-consistent `summary` is a **net loss** (0.69–0.75x,
worse than planning the full belief). Its build cost `O(K(|A| + Σ_u |O_u|))`
only amortizes at `H ≥ 2`. And even at `H ≥ 2` the frozen summary is
value-inconsistent under probing (`dV = 3.27e-02`) **with no bound** — it
happened to never flip an argmax on the 48-cell toy grid (0/48), which is
evidence, not a guarantee. The version *with* a guarantee (`summary_exact`)
costs ≈1.0x of full belief. **A visual MPC planner at `H = 1` gets no benefit
from the value-consistent path at all.**

**R5 — Decision fidelity is not value fidelity, and this repo has a documented
history of planners exploiting learned-cost error.** §2.5/§3.4 prove the bias is
structural (mode-edge under-values, mode-mean over-values). With a *learned*
`Q(z,a,g)` the bias can be much larger and, worse, *adversarially exploitable*:
the CAI-JEPA program in this repo repeatedly found CEM reward-hacking a learned
cost (Phase A: 91.5% → 24% decode accuracy on mined elites; E0's three-mode
Goodhart taxonomy). A calibrated within-mode value estimator is a **required**
component, not a refinement, and elite-mining adversarial checks should be run
on the compressed cost.

**R6 — Hidden-state dimension.** (B2): `M ≤ min(K, O((|G||A|²)^d))`. The
compression's headroom shrinks exponentially in the intrinsic dimension of the
decision-relevant hidden state. Everything measured so far is `d = 1` (mass,
location) or a product of binary bits. Nothing is known for `d ≥ 3`. **This is
not currently in any gate.** It should be added as an axis to Gate C1's sweep
(e.g. hidden mass **and** hidden friction **and** hidden CoM offset).

**R7 — The goal family may be too thin on established tasks.**
ShellGame-ColorLamp has 3 cups, so `|G| ≤ 3`, and (B1) leaves very little room to
show *interesting* multi-goal invariance. If `|G|` cannot be made ≥4–8 without
modifying the environment, **G4 — the headline novelty — is untestable on
established tasks**, and the bespoke fallback is triggered, which weakens the
"we didn't build it to win" defence. **Check this in P1, not P4.**

**R8 — Task difficulty floor.** A planning-level contribution is invisible under
a control-level floor. IMBench's own table is the cautionary example: mass-sort
scores **0.00 for every baseline** (π0.5 ZS/FT, GR00T ZS/FT, Diffusion Policy)
while Stage-1 "understanding" accuracy is 90–100% for frontier VLMs, and Stage-3
action success is 0.0% *even with privileged object poses*. (Occluder-push is not
uniformly zero: Diffusion Policy reaches 0.50.) BatteriesCheckerHard-6 has a
2160-step horizon. Mitigation: use Easy-3/Easy-6 for the frontier; check against
MIKASA-Robo's own reported baselines *before* committing; use their
motion-planning oracle as the low-level controller so the comparison sits at the
planning layer.

**R9 — No real robot.** Everything stands on simulation. Sim-only is normal for a
planning-algorithms paper, but it caps the claim: we can say "planning compute at
matched decision quality", not "this works on hardware". Do not let the framing
drift toward the latter.

**R10 — IMBench may release mid-project.** Upside. Keep `MassSortProbe-v0`'s
observation/action interface aligned to their spec so their environment drops in;
if CoRL 2026 accepts, rerun and report both.

### Open questions worth a line each

- Is (B1) actually a corollary of [2311.07745](https://arxiv.org/abs/2311.07745)?
  **Someone should sit down and try to derive it.** If it takes two lines, say so
  in the paper and move the contribution entirely to the empirical axis. If it
  genuinely doesn't follow (because their bounds are scalar-value and ours is
  vector-ranking, jointly over a family), that is worth a proposition.
- Can the frozen `summary`'s drift be *bounded* rather than merely measured?
  A bound of the form `|dV| ≤ f(within-mode likelihood spread)` would convert
  §3.4's empirical 0/48 into a guarantee and would materially strengthen D2.
  This is the most valuable piece of theory available to this project.
- Does the compression compose with *temporal* abstraction (mode identity is
  stable across timesteps → reuse the partition across decisions)? Would attack
  R3 directly by amortizing the `O(K)` read over multiple steps.

---

## 9. Realistic outcome assessment

### If everything works

**What it supports:** a clean **empirical systems/algorithms paper** —
*"Decision-structured belief compression makes belief-space planning tractable
in the tree, and here is exactly when it pays for itself"* — with:

- a compute-vs-quality Pareto frontier on two established suites plus one
  spec-faithful reimplementation;
- the `(K, H)` ratio grid showing the win is asymptotic, not constant;
- the multi-goal held-out-goal result (D1) as the headline novelty;
- the amortized flip-VOI on compressed modes (D2) with its operating regime
  (`H ≥ 2`) and its value-inconsistency **stated as a characterized trade-off**;
- an adversarial control task on which we predicted and observed failure;
- B10/B11 (random subsample, latent k-means) beaten at matched `M` with clean CIs.

**Realistic venue:** CoRL / L4DC / RA-L, or ICLR/NeurIPS if the frontier is
striking and the multi-goal result is clean. **It is not a conceptual
breakthrough** — the compression bound belongs to the Simplified-POMDP and
bisimulation literatures, and amortized VOI belongs to DAD and the
active-inference literature. The paper's honest claim is *instantiation +
measurement + a characterized trade-off*, and that is a perfectly respectable
paper if the measurement is rigorous and the negative results are published
alongside.

### If it partly works

- **G1-G3 pass, G4 fails** → workshop paper on amortized flip-VOI over
  compressed modes. §7 says decide explicitly at that point. Given §4.4, this is
  a weak position.
- **S3 fires (filter eats the win)** → pivot to a paper about belief-model cost
  in visual POMDP planning. Legitimate, but it is *not* the paper Gate A cleared,
  and it should be a conscious decision, not a drift.
- **P0 fails (S5)** → "fix the filter first". The honest report is a short note:
  *"learned visual belief models do not carry decision-diverse hypotheses; here
  is the measurement; the compression premise is untestable with commodity
  filters."* That is publishable as a negative result and is genuinely useful to
  the field — this repository has published complete negative results before
  (`phase-d-encoder-lora-first-crossing`, `phase-g-ensemble-disagreement-cost-null`).

### What makes it fail

In order of likelihood: R2 (learned `K` too small / hypotheses not diverse) →
R3 (the `O(K)` encoder read dominates end-to-end cost) → the B10/B11 controls
matching us at matched `M` (STOP S4, which would mean the decision criterion
contributes nothing) → R7 (`|G| ≤ 3` makes G4 untestable on established tasks) →
R1 (a reviewer derives our bound from [2311.07745](https://arxiv.org/abs/2311.07745)
in the review and correctly downgrades the theoretical contribution to zero).

---

## 10. Effort & compute budget, task breakdown

One person, honest estimate, from `gateC1_design.md` §2.6.

| Phase | Work | Wall time | GPU (A100-class) |
|---|---|---|---|
| **P0** | **Diversity smoke test — DO FIRST.** DreamerV3 on `BatteriesCheckerEasy-3` + `BattleShipEasy`; sample `K` latents; measure ESS and the number of distinct oracle-labelled decision signatures. This is STOP S5 and it can kill the gate for ~50 GPU-hours. | 1 week | ~50 h |
| P1 | Env plumbing (ManiSkill3 + MIKASA-Robo + POPGym Arcade), oracle hidden-state logging, `ComputeCounter` → FLOPs, `C_split` accounting. **Also: check R7 (`|G|` reachable on ShellGame) here, not in P4.** | 2 weeks | ~20 h |
| P2 | Belief/world models: DreamerV3 on 4-6 MIKASA tasks (from released oracle data) + 4 POPGym Arcade envs (cheap, JAX) | 3-4 weeks | 300-600 h |
| P3 | `Q(z,a,g)` head, signature extraction, **calibrated mode-value estimator**, flip-predictor training | 2 weeks | ~150 h |
| P4 | **The frontier sweep** — 13 baselines/ablations x 7 `K` x 5 `H` x 5 seeds x 4 tasks x ≥100 episodes | 2-3 weeks | 300-500 h |
| P5 | Analysis, clustered-bootstrap CIs, figures, `gateC1_results.md` | 1 week | ~20 h |
| **Total** | | **~10-12 weeks** | **~850-1350 h; budget 1500 h with buffer** |

Add `MassSortProbe-v0` (IMBench-spec reimplementation) **only if triggered by
R7**: +2 weeks, +150 GPU-h, and it must include the scripted probing oracle and
the success predicate exactly as specified.

**Ordering discipline.** P0 gates everything. R7 is checked in P1. Do not build
the flip-predictor (P3) before P0 has shown there are diverse hypotheses to
compress and P1 has shown there is a goal family to compress across.

**Zero-cost work available now, before any GPU:**
1. Attempt to derive (B1) from [2311.07745](https://arxiv.org/abs/2311.07745).
   Cheap, and it determines how the paper must be framed.
2. Add the **hidden-state dimension axis** (R6) to `scaling.py` — a
   `GridParam`-style task with a 2-D or 3-D hidden parameter, measuring `M`
   against (B2). Pure numpy, hours of work, and it closes a real gap.
3. Attempt a bound on the frozen `summary`'s drift (§8, open questions).
4. Fix the stale Gate C0 characterization in `gateC1_design.md` (§3.7).

---

## 11. How to reproduce what exists today

All from the repository root, `/home/nhatnc129/nhat.nc/jepa-research`.
No GPU, no data, no checkpoints — everything here is pure numpy and exact.

```bash
# Unit tests for the whole belief_compression package (46 tests, ~8 s)
diagnosis/.venv/bin/python -m pytest belief_compression/tests/ -q

# A single test file / a single test
diagnosis/.venv/bin/python -m pytest belief_compression/tests/test_compression.py -q
diagnosis/.venv/bin/python -m pytest belief_compression/tests/test_scaling.py::test_name -q

# Gate B — the exact-POMDP oracle testbed.
# Regenerates belief_compression/docs/gateB_oracle_results.md
# and docs/figures/measurement{1,2}_*.png. Seconds.
diagnosis/.venv/bin/python -m belief_compression.run

# Gate C0 — the scaling study.
# Regenerates belief_compression/docs/gateC0_scaling_results.md
# and docs/figures/gateC0_s1_saturation.png, gateC0_s2s3_compute.png.
# ~246 s total runtime.
diagnosis/.venv/bin/python -m belief_compression.scaling
```

**Both documents are generated artifacts** — they are written by the commands
above, so every table in §3 of this plan is reproducible byte-for-byte. Cells the
exact enumerator cannot afford print `n/a` and are never estimated
(`MAX_K_EXACT_REGRET = 192`, `MAX_H_EXACT_REGRET = 1`, and a 4e7-primitive-op cap
on §S3b/§S8).

**Source map:**

| File | Contents |
|---|---|
| `belief_compression/core.py` | `Task` ABC, `Belief` (exact Bayes, support-only compute accounting), `ComputeCounter` |
| `belief_compression/tasks.py` | `MassSort`, `OccluderPush`, `GridParam` (the resolution-knob task, with `decision_bound()` implementing (B1)) |
| `belief_compression/compression.py` | `compress()` (exact + `ε`-tolerance), `pick_rep` (`centroid` / `maxweight`), `ModeSummary` (`summary` / `summary_exact`), `ParticleTables` |
| `belief_compression/planners.py` | `expectimax`, `FullBeliefPlanner`, `CertaintyEquivalent`, `CompressionPlanner`, `PrecomputedCompressionPlanner`, `EntropySeeking`, `DecisionRegretVOI`, `AmortizedVOI` |
| `belief_compression/evaluate.py` | exact expected return by enumeration; `fully_observed_return` upper bound |
| `belief_compression/experiments.py` | the two Gate-B measurements |
| `belief_compression/run.py` | Gate B driver → `docs/gateB_oracle_results.md` |
| `belief_compression/scaling.py` | Gate C0 driver, sections S1-S8 → `docs/gateC0_scaling_results.md` |

**Changing the mode-carrying rule** (to reproduce the §3.4 failures):
`DEFAULT_REP_RULE = "centroid"` in `compression.py`; `REP_RULES = ("centroid",
"maxweight")`, `SUMMARY_RULES = ("summary", "summary_exact")`. All four rules
produce the **same partition and the same `M`**, so compute is identical for the
two representative rules; only the reported values differ.

---

## 12. References

Sorted by role. Every URL verified as present in the gate documents that cite it;
the IMBench and substrate rows were live-checked on 2026-07-30.

**Prior art that must be conceded (§4.2)**

- Lev-Yehudi, Barenboim & Indelman, *Simplifying Complex Observation Models in Continuous POMDP Planning with Probabilistic Guarantees and Practice*, AAAI 2024 — https://arxiv.org/abs/2311.07745
- Barenboim & Indelman, *Online POMDP Planning with Anytime Deterministic Optimality Guarantees*, NeurIPS 2023 — https://arxiv.org/abs/2310.01791
- Barenboim & Indelman, *Adaptive Information Belief Space Planning*, IJCAI 2022 — https://arxiv.org/abs/2201.05673
- *Simplification of Risk-Averse POMDPs with Performance Guarantees* — https://arxiv.org/abs/2406.03000
- Kundu, Rafaeli, Gulyaev & Indelman, *Action-Consistent Decentralized Belief Space Planning* — https://arxiv.org/abs/2403.05962
- *Optimality Guarantees for Particle Belief Approximation of POMDPs* — https://arxiv.org/abs/2210.05015
- Ferns, Panangaden & Precup, *Bisimulation Metrics for Continuous Markov Decision Processes* — https://www.cs.mcgill.ca/~prakash/Pubs/siamFP11.pdf
- Gelada et al., *DeepMDP* — https://arxiv.org/abs/1906.02736
- *Learning Causal State Representations of POMDPs* — https://arxiv.org/abs/1906.10437
- *Bisimulation Metric for Model Predictive Control* — https://arxiv.org/abs/2410.04553
- Poupart & Boutilier, *Value-Directed Compression of POMDPs*, NeurIPS 2002 — https://proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html
- Roy & Gordon, *Finding Approximate POMDP Solutions Through Belief Compression* — https://arxiv.org/abs/1107.0053
- Wang, Crook, Tang & Lemon, *On the Linear Belief Compression of POMDPs: a re-examination* — https://arxiv.org/abs/1508.00986
- Gangwani, Lehman, Liu & Peng, *Learning Belief Representations for Imitation Learning in POMDPs*, PMLR v115 — https://proceedings.mlr.press/v115/gangwani20a.html (also https://arxiv.org/abs/1906.09510)
- Grimm, Barreto, Singh & Silver, *The Value Equivalence Principle for Model-Based RL* — https://arxiv.org/abs/2011.03506
- Grimm et al., *Proper Value Equivalence* — https://arxiv.org/abs/2106.10316
- Ran Wei, *Value of Information and Reward Specification in Active Inference and POMDPs* — https://arxiv.org/abs/2408.06542
- Foster et al., *Deep Adaptive Design*, ICML 2021 — https://arxiv.org/abs/2103.02438 ; iDAD, NeurIPS 2021 — https://proceedings.neurips.cc/paper/2021/file/d811406316b669ad3d370d78b51b1d2e-Paper.pdf
- Fujii & Murata, *Real-World Robot Control by Deep Active Inference with a Temporally Hierarchical World Model* — https://arxiv.org/abs/2512.01924
- Somani et al., *DESPOT*, JAIR — https://www.jair.org/index.php/jair/article/download/11043/26215/20559
- *POMCGS: Partially Observable Monte-Carlo Graph Search* — https://arxiv.org/abs/2507.20951
- Li, Walsh & Littman, *Towards a Unified Theory of State Abstraction for MDPs* — http://rbr.cs.umass.edu/aimath06/proceedings/P21.pdf

**Belief filters (borrowed, not claimed)**

- Hafner et al., *PlaNet* — https://arxiv.org/abs/1811.04551
- Hafner et al., *DreamerV2* — https://arxiv.org/abs/2010.02193
- Hafner et al., *DreamerV3* — https://arxiv.org/abs/2301.04104 ; code https://github.com/danijar/dreamerv3
- Delgrange et al., *The Wasserstein Believer* — https://arxiv.org/abs/2303.03284
- Flow-based / Stein-variational belief-state learning (FORBES / ESCORT) — https://arxiv.org/abs/2510.21107
- `pymdp` (discrete active-inference reference) — https://github.com/infer-actively/pymdp

**Substrates and evaluation**

- MIKASA-Robo, ICLR 2026 — https://arxiv.org/abs/2502.10550 ; https://github.com/CognitiveAISystems/MIKASA-Robo ; docs https://mikasarobo.github.io/ ; PyPI `mikasa-robo-suite`
- ManiSkill3 — https://arxiv.org/abs/2410.00425 ; https://github.com/mani-skill/ManiSkill
- POPGym Arcade — https://arxiv.org/abs/2503.01450 ; https://github.com/bolt-research/popgym-arcade
- POPGym (original, non-visual), ICLR 2023 — https://arxiv.org/abs/2303.01859
- Tactile MNIST — https://arxiv.org/abs/2506.06361 ; https://github.com/TimSchneider42/tactile-mnist
- APPLE (active perception), ICLR 2026 — https://arxiv.org/abs/2505.06182 ; https://timschneider42.github.io/apple
- robosuite — https://github.com/ARISE-Initiative/robosuite
- **IMBench** — *anonymous CoRL 2026 submission; **no arXiv posting**; **no released environment code**; 10 demo episodes per task only.* Site https://imbench.org/ ; paper https://imbench.org/paper.pdf ; data https://huggingface.co/imbench . **Cite as an anonymous unpublished submission with URL; do not cite an arXiv ID (the circulated `2607.15641` is wrong and does not resolve to IMBench).**

**Internal gate documents (this repository)**

- `belief_compression/docs/gateA_novelty_matrix.md` — prior-art audit, conditional GO, the two surviving differentiators
- `belief_compression/docs/gateB_oracle_results.md` — exact-POMDP oracle results (GO)
- `belief_compression/docs/gateC0_scaling_results.md` — scaling study, S1-S8 (STRONG-BUT-NARROW)
- `belief_compression/docs/gateC1_design.md` — learned-visual design, substrate verification, GO/STOP (note §3.7: its Gate-C0 characterization is stale)
