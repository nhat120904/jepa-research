# Elite-optimism instrument gate

Date locked: 2026-09-04 UTC, after the five progress arms trained and before any
arm is compared on candidate ranking.

## Why this exists before anything else

The five progress arms are separated by 0.005 in validation expectile loss and
0.003 in MAE. Against the best constant predictor for this target distribution
(expectile loss 0.02924 at `tau = 0.2`), `prog_ssl` reaches 0.02105, so the whole
state-dependent signal the head learned is 0.00819. Of that, severing the
recurrence (`prog_frame`, which gives up history *and* latching together) costs
0.00052 and severing latching alone (`prog_nomono`) costs 0.00008, while zeroing
the observations (`prog_action_only`) costs 0.00674.

Those are calibration numbers. Planning does not consume a calibrated cost, it
consumes an *ordering* over candidate action chunks, and no metric in this
program measures one. Every question queued behind this — whether an expectile
with `tau > 0.5` helps, whether an observation-history belief ranks better than a
frame, whether a predictive-state objective beats a recurrent LeWM at equal
capacity — is a ranking question. This gate builds the instrument that can answer
them, and answers one question with it.

## The question

Does CEM preferentially select candidates the cost is wrong about?

If it does, model exploitation is a live mechanism on this arena and a
conservative objective has something to remove. If it does not, the cost's errors
are spread evenly over the proposal distribution, optimisation is not seeking
them out, and the conservative-objective arm should not be built.

## Ground truth

`c_true` for a candidate is the environment steps to satisfy the goal predicate:
the chunk's own executed steps, plus what OGBench's own Markov controllers then
need. The controllers are parameterised by the goal row (cube oracle at the goal
cube position, drawer and window oracles at the goal joint values) so an
arbitrary dataset goal is reachable, not only the five scripted tasks.

Two properties are declared, not discovered:

- The repair **order** is fixed — cube, drawer, window, then the buttons that
  lock them — and not searched. `c_true` is therefore an *upper bound* on the
  minimum remaining cost, and every absolute inversion rate is inflated by that.
  The gate turns on a **ratio between strata**, where the bias is common.
- A repair that has not reached the goal within `--max-repair-skills` is recorded
  as **censored** and dropped from the metrics, so an oracle failure cannot be
  read as a hard candidate. The censoring rate is reported.

## Strata

Equal n per instrumented replan, all four from the same frozen snapshot, same
goal, same horizon:

| stratum | source |
|---|---|
| `random_init` | candidates evaluated at CEM iteration 0 — the un-optimised proposal |
| `gen_mid` | candidates at iteration `n_steps // 2` |
| `gen_last` | candidates at the final iteration |
| `elite` | the final iteration's top-k |

`selected` (the committed plan) is one chunk per replan and is used only for
regret, never in the equal-n contrasts.

When a learned behaviour prior lands, `random_init` is the stratum it replaces;
nothing else in the instrument changes.

## Metrics

Primary is a **pairwise inversion rate**: over candidate pairs whose true costs
differ by more than `delta` environment steps, the fraction the cost orders
backwards. Only the model's order is used, so a latent distance and a normalised
remaining time are directly comparable. `delta` is **swept** over
`{0, 10, 25, 50}` and reported as a curve; the same candidate is re-executed
`--delta-probe-repeats` times from the same snapshot once per episode to record
the noise floor the sweep is read against.

Secondary: Kendall tau, top-k recall, selected regret in environment steps, and
the fraction of each candidate clipped at the action bounds — upstream CEM
samples an unbounded Gaussian, so a candidate can be *scored* outside the bounds
and *executed* inside them, and that gap belongs in the record rather than in the
noise.

Bootstrap is over instrumented replans, the unit the strata are paired within.

## Locked verdicts

Read at `delta = 25` (one committed chunk), on `latent_l2` at goal offset 100:

- `EXPLOITATION_CONFIRMED`: elite inversion rate exceeds `random_init` by a
  factor of at least 1.5 with a replan-bootstrap 95% CI on the paired difference
  excluding zero. Licenses the conservative-objective arm.
- `EXPLOITATION_ABSENT`: the paired difference CI contains zero and its upper
  bound is below 1.25x. The conservative-objective arm is **not** built, and the
  τ sweep is not run.
- `INSTRUMENT_UNUSABLE`: censoring above 30%, or too few decisive pairs (see the
  amendment below), or a repeat-probe spread above `delta = 10`. Fix the
  instrument before reading any contrast.
- `INCONCLUSIVE`: neither of the two above. The conditions for `CONFIRMED` and
  `ABSENT` do not partition the space — a ratio of 1.3 with a CI covering zero
  satisfies neither — and that region is reported as itself rather than rounded
  into a verdict.

### Amendment, 2026-09-04, before any result was read

The original `INSTRUMENT_UNUSABLE` condition read "fewer than 20 decisive pairs
per stratum" and was implemented as a per-*replan* minimum. That bar is
unreachable by construction: at `--n-per-stratum 6` a replan holds at most
`C(6, 2) = 15` pairs, so every run would have been declared unusable regardless
of what it measured. The error was found by running the analyser against
synthetic artifacts with known ground truth, while jobs 49557/49558 were still
executing and before any real artifact existed.

The bar moved to the two quantities that actually carry the estimate:

- at least **200 decisive pairs in total** in every stratum (the expected count
  at the default budget is roughly 470-540, so this fails only on a degenerate
  run);
- at least **30 replans contributing a defined paired rate**, which is also what
  the power table above is computed at.

This is a correction of an impossible threshold, not a relaxation of a bar that a
result failed. No result had been read when it was made, and the original wording
is preserved above this note.

A verdict token is not rewritten after the fact. If a threshold turns out to be
mis-specified, amend here in the open with the reason, as the harness gate did.

## Calibration of the verdict, measured before the run

The thresholds above were checked against synthetic candidate sets where the
truth is known, so that a verdict cannot be an artefact of the estimator. Under
the null — all four strata drawn independently at the same cost-noise level, 6
candidates per stratum — the four mean inversion rates agree to three decimals
(0.0678 / 0.0677 / 0.0667 / 0.0681 over 200 runs), so the instrument does not
manufacture a gradient across the strata by construction.

Detection rate for the locked conjunction (paired-difference CI above zero **and**
ratio at least 1.5), 150 runs per cell, by how much noisier the cost is on elites
than on the initial proposal:

| elite cost noise | n = 18 replans | n = 36 | n = 72 |
|---|---:|---:|---:|
| 1.0x (null) | 0.04 | 0.02 | 0.01 |
| 1.5x | 0.57 | 0.74 | 0.89 |
| 2.0x | 0.87 | 0.99 | 1.00 |
| 3.0x | 0.99 | 1.00 | 1.00 |

Two things follow and are locked here. The conjunction is what controls the false
positive rate: the ratio alone reaches 1.38 at the 90th percentile of the null at
n = 36, so it is a descriptor, not a test. And the default budget — 12 episodes x
3 instrumented replans = 36 — buys 0.74 power against a 1.5x effect and 0.99
against 2x; an `EXPLOITATION_ABSENT` verdict at that budget therefore rules out a
2x effect but not a 1.5x one, and must be reported that way. Ruling out 1.5x
needs 24 episodes.

## Result, jobs 49557 / 49558 / 49560 (2026-09-04)

Verdict: **`INSTRUMENT_UNUSABLE`** on both arms. No contrast is read from this
run.

| arm | base | elite | ratio | paired diff [95% CI] | n | verdict |
|---|---:|---:|---:|---|---:|---|
| `latent_l2` | 0.606 | 0.301 | 0.50 | -0.384 [-0.786, +0.071] | 5 | `INSTRUMENT_UNUSABLE` |
| `prog_ssl` | 0.467 | 0.469 | 1.01 | +0.051 [-0.231, +0.359] | 11 | `INSTRUMENT_UNUSABLE` |

The binding failure is the paired count: 5 and 11 replans contribute a defined
paired rate, against a bar of 30, out of 31 and 32 instrumented. Decisive pairs
per stratum are 40-106 against a bar of 200.

The cause is measured, not guessed. Within a replan the candidates' true costs
have a median spread of only 40-42 environment steps (sd 10.5-13.5), so at
`delta = 25` just 68-72% of replans contain even one separated pair, and a pair
must be separated in *both* strata for that replan to enter the paired
difference. `c_true` is dominated by the episode's distance to goal, which the
oracle repair re-drives identically whatever the chunk did, rather than by what
the chunk itself achieved -- and only 3.4-3.6% of chunks reach the goal on their
own.

Two readings of that remain open and this run cannot separate them: the 25-step
chunks may be genuinely near-equivalent in outcome (a proposal-coverage
bottleneck, which would itself be the finding), or `c_true` may be too coarse to
resolve differences that exist. The instrument as built cannot tell them apart
because it records no per-candidate outcome between "the chunk solved the task"
and "the oracle finished it".

What the run does establish, and what carries forward:

- The repeat probe returned a spread of exactly 0.0 at every instrumented
  replan on both arms. The arena is deterministic given state and actions, as
  predicted, so the margin sweep is the right treatment of `delta` and no
  noise-floor estimate is available from repetition.
- Censoring is 4.8% and 6.0%, far under the 30% bar: the parameterised oracle
  repair reaches arbitrary dataset goals reliably. That part of the instrument
  works.
- **31% of action components are scored outside `[-1, 1]` and executed clipped,
  and that fraction does not fall across CEM generations** (0.321 / 0.316 /
  0.311 / 0.309 from the initial proposal to the final elites). The planner is
  therefore ranking action sequences the environment cannot execute, and
  optimisation does not correct it. This is a defect in the planning loop that
  applies to every ladder result already collected, not only to this gate.

Required before the gate is re-read, in this order:

1. Record the goal-component match vector for every candidate immediately after
   its chunk and before any repair. That measures "did this chunk change
   anything" without the oracle, and separates the two readings above.
2. Raise `--n-per-stratum` from 6 to 16: `C(16, 2) = 120` pairs per replan
   against 15 today. Episodes ran at roughly 35s, so the cost is minutes.
3. Re-read the margin sweep against the observed within-replan spread. A bar of
   `delta = 25` against a median spread of 40 was set before that spread was
   known; the decision of which `delta` the verdict is read at must be made from
   the spread and fixed before the re-run, not chosen from its results.

### Re-run decision, 2026-09-05, before the re-run

`delta` **stays at 25**. Retuning it from the observed spread would let a
threshold be informed by a run that has already been read, which is the failure
mode this document exists to prevent; the cost of keeping it is power, and power
is bought instead by item 2, which is neutral with respect to the result. Items 1
and 2 are applied:

- every candidate now records the goal-component match its chunk leaves behind,
  before any repair (`component_delta`: which components were gained, which lost,
  and the net), so "the chunk did nothing" is measurable without the oracle;
- `--n-per-stratum` is 16 by default, giving `C(16, 2) = 120` pairs per replan
  against 15. A four-replan synthetic at n = 16 already yields 408 decisive
  pairs, against 40-106 over 31 replans at n = 6.

The power table above was computed at 6 candidates per stratum and is therefore
a **lower bound** for the re-run: more candidates per replan reduces the variance
of each replan's rate, which can only help. It is not recomputed, so that the
bars it justifies stay the ones that were locked.

## Result, jobs 49571 / 49572 / 49575 (2026-09-05), `--n-per-stratum 16`

Verdict: **`INSTRUMENT_UNUSABLE`** on both arms again. No contrast is read.

| arm | base | elite | ratio | paired diff [95% CI] | paired n | verdict |
|---|---:|---:|---:|---|---:|---|
| `latent_l2` | 0.376 | 0.585 | 1.56 | +0.221 [+0.045, +0.424] | 12 | `INSTRUMENT_UNUSABLE` |
| `prog_ssl` | 0.455 | 0.462 | 1.01 | -0.052 [-0.176, +0.067] | 15 | `INSTRUMENT_UNUSABLE` |

Item 2 did what it was for: decisive pairs per stratum went from 40-106 to
415-905, clearing the 200 bar with room. The binding constraint moved to the
**paired** count, which is 12 and 15 against a bar of 30. A replan enters the
paired difference only when *both* `random_init` and `elite` contain a pair
separated by more than `delta = 25`; individually those strata clear 17-23
replans out of 31-32, and their intersection is what collapses.

Item 1 answered the question v1 could not, and answered it against the simpler
explanation:

| stratum | changed any component | net components gained |
|---|---:|---:|
| `random_init` | 7.7% / 9.4% | -0.028 / -0.051 |
| `elite` | 18.1% / 11.3% | +0.050 / +0.049 |

(`latent_l2` / `prog_ssl`.) Candidates are **not** degenerate: CEM optimisation
makes chunks markedly more consequential, and flips the net component change from
negative to positive on both arms. "Nothing any candidate does matters" is
refuted, so the low decisive-pair yield is a property of `c_true`'s resolution,
not of the candidate set.

Not licensed by this run, recorded so it is not lost: the two arms move in
opposite directions. `latent_l2`'s rank correlation falls from +0.153 on the raw
proposal to +0.001 on elites while its inversion rate rises 0.376 -> 0.585;
`prog_ssl`'s rises from +0.003 to +0.144 with a flat inversion rate. That would
be a real contrast between the two costs if it survived, and at paired n of 12
and 15 it may equally be noise. It is not to be quoted until the gate reads.

Required before the gate is re-read: `--num-episodes` 12 -> 36. The binding
constraint is now the number of replans, so the fix is replans, and nothing else
changes -- not `delta`, not `--n-per-stratum`, not a threshold. Reaching the
pre-registered `n >= 30` is what the protocol already requires, not an additional
analysis; 31-32 instrumented replans yielded 12-15 paired, so roughly 90
instrumented replans are needed. Runtime was 15-17 minutes per arm at 12
episodes.

## Result, jobs 49578 / 49579 (2026-09-05), 36 episodes

| arm | base | elite | ratio | paired diff [95% CI] | paired n | verdict |
|---|---:|---:|---:|---|---:|---|
| `latent_l2` | 0.492 | 0.614 | 1.25 | +0.115 [-0.005, +0.245] | 27 | `INSTRUMENT_UNUSABLE` |
| `prog_ssl` | 0.433 | 0.488 | 1.13 | +0.045 [-0.068, +0.158] | 44 | **`EXPLOITATION_ABSENT`** |

`prog_ssl` is the first arm to clear every bar and produce a licensed verdict:
optimisation does **not** preferentially select the candidates its cost is wrong
about. Read with the power the budget bought, that rules out a 2.0x
amplification and does **not** rule out 1.5x. `latent_l2` misses the paired-count
bar at 27 and stays unread.

The trajectory across the three runs is the substantive finding and is more
informative than any single row. For `latent_l2` the ratio went 0.50 (paired
n = 5) -> 1.56 (n = 12) -> 1.25 (n = 27), and the paired interval closed onto
zero as it went. An effect that shrinks while power grows is the signature of a
null being estimated badly at small n, not of a real effect. The 1.56 at n = 12
was never quoted as a result, and this is why.

Censoring rose with the episode count -- 4.9% at 12 episodes, 20.6% and 24.2% at
36 -- because the wider episode draw includes goals the fixed-order oracle repair
cannot reach. It stays under the 30% bar, but it is what consumes paired replans,
and it is the reason `latent_l2`'s paired count grew only 12 -> 27 while episodes
grew 12 -> 36.

Recorded as description, not as a tested contrast (the component measurement
carries no verdict bar and no interval): `latent_l2`'s elites net-gain goal
components (+0.086) while `prog_ssl`'s do not (-0.004), and every `prog_ssl`
stratum is net-negative. Both arms solved 3 of 36 episodes.

### Gate closed

This gate licensed the conservative-objective arm, which the method had already
demoted to secondary. One arm answers `EXPLOITATION_ABSENT` and the other trends
the same way without clearing its bar, so the arm is **not built**. Reaching a
licensed verdict on `latent_l2` needs roughly 48 episodes and about three hours;
that is not spent, because it would buy the same decision.

The honest statement of the outcome is "no evidence of exploitation amplification
on this arena at this budget", not "exploitation does not occur". The 1.5x band
remains untested.

This measures one arm's cost against one world model on the self-rendered arena;
Stage 0 found the released pixels are not reproducible by this renderer, so no
number here is comparable to a published `visual-scene-play` result. The
`event_smdp_h0` finding that motivates the conservative arm — over-reads are near
fatal (1.92% episode success) while under-reads are partly recoverable (36.21%) —
was measured under a hand-specified automaton with oracle skill execution, and
job 49415 showed the direction is feedback-dependent: under `automaton_potential`
the same observer's under-reads rise 163 → 1034 and 188/192 task-5 episodes
exhaust the budget. Whether the asymmetry survives into learned-WM CEM planning
is exactly what this instrument is built to find out, and must not be assumed.
