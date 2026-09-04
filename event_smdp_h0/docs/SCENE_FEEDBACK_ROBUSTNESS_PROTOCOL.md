# Scene feedback-robustness sweep: is `event_progress` a lucky point?

Locked 2026-09-04, before any sweep cell was evaluated.

## Question

The 3x2 grid showed that two dense automaton feedbacks are statistically
indistinguishable under the simulator's own event state (76.56% against 75.00%
on task 5, CI covering zero) yet differ by 87.50 points under a learned
observation-history observer.  That leaves one question open: is the surviving
feedback a single lucky point, or does a class of feedbacks tolerate
state-estimation error?

## A correction to the framing: the two feedbacks are one family

`scene_core.feedback_reward` and `scene_feedback.automaton_potential` are not
two designs.  Both equal

    0.90 * ( w * cube/5 + (1-w) * window/3 )

at `w = 0.500` and `w = 0.625` respectively, verified exactly on every event
state of both tasks.  The 87.50-point collapse is therefore produced by moving a
single branch-weight parameter by 0.125.

The sweep accordingly walks that parameter rather than inventing unrelated
scalars, which turns the question into a one-dimensional sensitivity curve with
two already-measured anchors, and adds two designs from outside the family.

## Feedbacks

| Name | Definition |
|---|---|
| `branch_w030` | branch-weighted, w = 0.300 |
| `branch_w040` | branch-weighted, w = 0.400 |
| `branch_w050` | w = 0.500; identical to `event_progress`, an anchor |
| `branch_w056` | w = 0.5625, midpoint placed to locate the cliff |
| `branch_w062` | w = 0.625; identical to `automaton_potential`, an anchor |
| `branch_w070` | branch-weighted, w = 0.700 |
| `anti_livelock` | `branch_w050` minus 0.25 when a candidate's first skill is already known not to move the believed event state |
| `shaped_gamma09` | discounted potential shaping, `sum_k g^k (Phi(q_{k+1}) - Phi(q_k))` with `Phi = branch_w050`, `g = 0.9`, clipped to [0, 1] |

`anti_livelock` is the deliberate design for under-read tolerance.  Its blocked
set is episode memory: after deploying skill `s` from believed state `q`, if the
next believed state equals `q`, the pair is blocked.  The penalty 0.25 and the
discount 0.9 are fixed in advance and are **not** tuned after seeing results.

A note on `shaped_gamma09`: with `g = 1` and terminal-only scoring, potential
shaping telescopes to `Phi(q_H) - Phi(q_0)`, and `Phi(q_0)` is constant across
candidates at a replan, so it would collapse to `branch_w050` and test nothing.
The discount is what makes it a distinct scalar - it prefers progress that
arrives earlier in the rollout.

## Design

Three state sources as before: `frame_full` (3 seeds), `obs_history_full`
(3 seeds), simulator q.  8 feedbacks x 7 rows = 56 arms per reset, all run from
one restored snapshot per reset inside one job.

**Task 5 only**, 64 resets, seeds `88500-88563`.  Task 4 is dropped by
preregistration: several members of the family coincide there (the family
reduces to `0.90 * cube / 4` independently of `w`), so it cannot discriminate.
The `branch_w050` and `branch_w062` cells must reproduce the task-5 half of
`grid_20260904` exactly; that is the determinism check.

Everything else is held fixed: same H1b transition checkpoint, same observer
checkpoints, same skills, same Skill-UCT at budget 112, horizon 4, exploration
0.55, same search-seed formula.

## Primary endpoint

Not "which feedback scores highest".  For each feedback `f`:

    GAP(f) = obs_history_full(f) - oracle(f)

paired by reset, model seeds averaged within a reset, bootstrapped over reset
clusters.  A feedback **preserves** performance if `GAP(f)` has a 95% CI whose
lower bound is above -5 points.

Secondary, reported per feedback and state source: timeout/livelock rate
(episodes exhausting all 10 decisions), repeated-skill rate, exact-q rate, and
over-read/under-read counts.

## Locked decision rules

- `NONDETERMINISTIC_EVAL`: either anchor fails to reproduce `grid_20260904`.
  No other verdict may be issued.
- `FEEDBACK_TOLERANCE_IS_GENERAL`: at least 5 of the 8 feedbacks preserve.
  Claim becomes "observation-history reconstruction suffices for event planning
  provided the feedback tolerates state-estimation error", with the tolerant
  class characterised by the sweep.
- `EVENT_PROGRESS_IS_A_KNIFE_EDGE`: at most 2 of the 8 preserve.  Do not attempt
  to rescue generality.  The headline becomes the methodological finding:
  oracle-state evaluation hides severe feedback-estimator interactions.
- `PARTIAL`: anything else; report the tolerant and intolerant sets as measured.

Whether the skill-failure sweep is worth running is decided after this.

## Declared limitations

- One automaton, one task, one planner, one transition checkpoint.
- `anti_livelock` conditions on episode history, so it is not a pure function of
  the event state like the rest of the family; that is intrinsic to the design
  being tested, not an oversight.
- The sweep varies a scalar the planner consumes.  It does not vary the
  automaton itself, which remains hand specified.

## Outcome (jobs 49434-49451, 2026-09-04)

Verdict: **`PARTIAL`** - 4 of 8 feedbacks pass the strict preserve gate, between
the `>= 5` needed for `FEEDBACK_TOLERANCE_IS_GENERAL` and the `<= 2` that would
have declared a knife edge.

Anchor reproduction against `grid_20260904`: 896 rows, **0 mismatches**.

| Feedback | oracle | `obs_history_full` | `frame_full` | gap (obs - oracle) | preserves |
|---|---:|---:|---:|---|:--:|
| `branch_w030` | 93.75% | 92.71% | 82.29% | -1.04 [-3.12, +0.00] | yes |
| `branch_w040` | 95.31% | 92.71% | 81.77% | -2.60 [-6.77, +0.00] | no |
| `branch_w050` | 76.56% | 89.58% | 54.17% | +13.02 [+4.17, +22.40] | yes |
| `branch_w056` | 40.62% | 25.00% | 10.42% | -15.62 [-28.12, -3.12] | no |
| `branch_w062` | 75.00% | 2.08% | 10.94% | -72.92 [-83.33, -61.98] | no |
| `branch_w070` | 98.44% | 8.85% | 9.38% | -89.58 [-95.85, -81.77] | no |
| `anti_livelock` | 71.88% | 85.94% | 51.04% | +14.06 [+6.25, +23.44] | yes |
| `shaped_gamma09` | 92.19% | 92.19% | 74.48% | +0.00 [-3.65, +4.17] | yes |

### `event_progress` is not a lucky point

That was the question the sweep was locked to answer, and the answer is no.
Under a learned observation-history observer, `w = 0.300`, `0.400` and `0.500`
all land between 89.58% and 92.71%, so `w = 0.500` sits on a **plateau spanning
at least `[0.30, 0.50]`** rather than on a knife edge.  Two designs from outside
the family also hold up: `shaped_gamma09` at 92.19% and `anti_livelock` at
85.94%.  Five of eight feedbacks put `obs_history_full` at or above 85%.

The collapse is real but one-sided and abrupt, falling off a cliff between
`w = 0.500` (89.58%) and `w = 0.5625` (25.00%).

`branch_w040` is scored `no` only because its CI lower bound is -6.77 against a
-5.00 floor, while its absolute success is 92.71%.  The gate is not moved after
the fact; this is recorded as an observation about the gate's width, not a
reclassification.

### The methodological finding is now much sharper

The feedback that is **best** under the simulator's own event state is the
**second worst** under a learned one:

- `branch_w070`: oracle 98.44%, learned 8.85%, a gap of -89.58 points.
- Best under learned state is `branch_w030` at 92.71%, which is only fourth
  under oracle.
- Across the eight feedbacks the oracle and learned rankings correlate at
  Spearman 0.357 (Pearson 0.317).  Exploratory, n=8, not preregistered.

So oracle-state evaluation does not merely add noise to a feedback comparison;
in this setting it actively inverts the top of the ranking.  Any feedback or
reward-shaping study that selects its design under privileged symbolic state can
select the worst available option for deployment.

### Secondary metrics behave as the mechanism predicts

Under the collapsed feedbacks the same observer's exact-q rate falls from
91.8-98.6% to 40.5-54.7%, the repeated-skill rate rises from ~28% to 48-56%, and
the timeout rate rises to 75-98%.  The failure is livelock, not misplanning,
exactly as the 3x2 diagnostic found.

### The deliberate anti-livelock design did not win

`anti_livelock` was written specifically to tolerate under-reads, and it does
preserve (+14.06, CI excluding zero) and cuts its own timeout rate from 28.1% to
14.1%.  But it does not beat plain `branch_w050` in absolute terms (85.94%
against 89.58%), and it lowered the oracle arm (71.88% against 76.56%).  The
design principle is therefore **not** established; the honest statement is that
robustness came from the branch weighting, not from the explicit repetition
penalty.

`shaped_gamma09` is the most robust feedback measured: a gap of exactly 0.00
points with 92.19% under both state sources.  That is a single observation on
one automaton and is not a general claim.
