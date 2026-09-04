# Scene 3x2: is the binding constraint state estimation or feedback design?

Locked 2026-09-04, before any cell of this grid was evaluated.

## Question

Everything established so far varies the *state source* while holding the
planner's scalar feedback fixed at one hand-written event-progress function.  A
reviewer can therefore ask whether the result is about recovering task progress
at all, or merely about one convenient reward shaping.

This grid separates the two axes.

| Feedback / State source | `frame_full` | `obs_history_full` | simulator q |
|---|---|---|---|
| `event_progress` | measured (69.01%) | measured (93.49%) | measured (87.50%) |
| `automaton_potential` | new | new | new |

The `frame_full` row is essential and is not an afterthought.  Without a state
source that *fails*, all cells sit near ceiling and the grid cannot show that
state estimation was ever the constraint.  Its three cells under
`event_progress` also re-run existing configurations, giving a third exact
reproduction check.

## The two feedbacks

Both are capped at 0.90 below acceptance and return 1.0 on stable success, so
they differ only in shaping.

- `event_progress` (`scene_core.feedback_reward`) balances the two task-5
  branches against each other: `0.90 * (cube/5 + window/3) / 2`.
- `automaton_potential` (`scene_feedback.automaton_potential`) is the hint^2
  style potential: uniform credit per remaining automaton milestone,
  `0.90 * (1 - ((5-cube) + (3-window)) / 8)`.

They disagree in both directions, e.g. at `(cube=5, window=0)` 0.450 against
0.563, and at `(cube=0, window=3)` 0.450 against 0.338.

**On task 4 the two functions are identical** (`0.90 * cube / 4`).  The feedback
contrast therefore exists only on task 5 (n=64), and every task-4 pair must
match exactly; a task-4 disagreement means the harness is not holding search
randomness fixed and invalidates the run.

The feedback axis is not otherwise untested: `abstract_terminal` already
contrasts a sparse terminal feedback at 2.34% against 90.62%.  What this grid
adds is dense against dense.

## Held fixed

Same H1b transition checkpoint, same seven skills, same Skill-UCT with budget
112, horizon 4, exploration 0.55, the same search-seed formula, and the same
reset band as the ablation run: `88400-88463` (task 4) and `88500-88563`
(task 5).  All 14 arms run from one restored snapshot per reset inside one job.

## Locked decision rules

Contrasts paired by reset, model seeds averaged within a reset, bootstrapped
over reset clusters.

State-source gaps, evaluated **under each feedback** `f`:

- `STATE_GAP_FRAME(f)      = frame_full(f) - oracle(f)`
- `STATE_GAP_OBS_HISTORY(f) = obs_history_full(f) - oracle(f)`

Feedback gaps, evaluated **under each state source** `s`:

- `FEEDBACK_GAP(s) = event_progress(s) - automaton_potential(s)`

Verdicts:

- `NONDETERMINISTIC_EVAL`: any reproduction mismatch on the three
  `event_progress` cells, or any task-4 disagreement between the two feedbacks.
  No other verdict may be issued.
- `STATE_ESTIMATION_IS_THE_CONSTRAINT`: under **both** feedbacks,
  `STATE_GAP_FRAME` is negative with a CI excluding zero, **and**
  `STATE_GAP_OBS_HISTORY` has a CI covering zero.  The conclusion then holds
  regardless of which dense automaton feedback is used.
- `FEEDBACK_ALSO_MATTERS`: `FEEDBACK_GAP` has a CI excluding zero under both
  state sources.  This is reported as a weakening, not a strengthening: it
  would mean the result depends on the particular shaping we wrote.
- `BOTH_MATTER`: both of the above hold.
- `INCONCLUSIVE`: anything else.

## Declared limitations

- The task-5-only feedback contrast has n=64 and correspondingly less power than
  the pooled state-source contrasts.
- `automaton_potential` is a faithful reconstruction of the hint^2 *scoring
  principle* on our hand-specified automaton, not a reimplementation of that
  paper's model; it shares our transition model and planner by design, since the
  point is to isolate the scalar.
- The reset band is shared with the ablation run.  The `automaton_potential`
  cells have never been evaluated on it, but the `event_progress` cells are
  reproductions rather than fresh measurements.

## Outcome (jobs 49395-49415, 2026-09-04)

Verdict: **`INCONCLUSIVE`** under the locked rules - not for want of signal, but
because neither preregistered hypothesis is what the data shows.

Reproduction: 896 rows of the `event_progress` column against the ablation run,
**0 mismatches**.  Task-4 feedback identity: **0 mismatches**, as required by
construction.

| Feedback | `frame_full` | `obs_history_full` | simulator q |
|---|---:|---:|---:|
| `event_progress` | 69.01% | **93.49%** | 87.50% |
| `automaton_potential` | 47.40% | **49.74%** | 86.72% |

Task 5 only, where the two feedbacks actually differ:

| Feedback | `frame_full` | `obs_history_full` | simulator q |
|---|---:|---:|---:|
| `event_progress` | 54.17% | **89.58%** | 76.56% |
| `automaton_potential` | 10.94% | **2.08%** | 75.00% |

| Contrast | points | 95% CI |
|---|---:|---|
| `FEEDBACK_GAP_ORACLE__task5` | +1.56 | [-9.38, +12.50] |
| `FEEDBACK_GAP_FRAME_FULL__task5` | +43.23 | [+31.77, +54.69] |
| `FEEDBACK_GAP_OBS_HISTORY_FULL__task5` | **+87.50** | [+79.17, +94.79] |
| `STATE_GAP_OBS_HISTORY_FULL__event_progress` | +5.99 | [+1.30, +11.20] |
| `STATE_GAP_OBS_HISTORY_FULL__automaton_potential` | -36.98 | [-45.57, -28.65] |

### The finding is an interaction, and it refutes the convenient story

With the simulator's own event state the two dense feedbacks are
interchangeable: 76.56% against 75.00%, CI covering zero.  A reader given only
that row would conclude the feedback function does not matter.

With a learned observer they differ by **87.50 points**.  The observer
checkpoints are byte-identical across the two columns; only the scalar changed.
So the proposition "once the state is recovered, multiple automaton-based
feedback designs work similarly" is **false**, and it is false in a way that is
invisible to oracle-state evaluation.

The honest statement is the interaction:

> Robustness to state-estimation error is a property of the feedback function,
> and it cannot be measured with a privileged state estimator.  Two dense
> automaton feedbacks that are statistically indistinguishable under exact state
> differ by 87 points under a learned one.

### Mechanism: the cost of an under-read is set by the feedback

The observer does not get worse in isolation; the feedback changes which states
the planner visits, and there the same observer errs far more.  Under
`automaton_potential`, `obs_history_full` falls from 93.2% to 57.0% exact, with
under-reads rising 163 -> 1034 while over-reads stay near zero (1 -> 10, plus
115 mixed).  Its top confusions are `[2,2] -> [1,2]` 369 times and
`[2,1] -> [1,1]` 225 times: the drawer was opened, latching `cube_stage` at 2,
then closed, so the scene reads as stage 1 again.

The consequence is livelock rather than a wrong plan.  Task-5 episodes that
exhausted all ten decisions without success:

| Arm | exhausted |
|---|---:|
| `obs_history_full__event_progress` | 20/192 = 10.4% |
| `oracle__event_progress` | 15/64 = 23.4% |
| `oracle__automaton_potential` | 16/64 = 25.0% |
| `frame_full__event_progress` | 88/192 = 45.8% |
| `frame_full__automaton_potential` | 171/192 = 89.1% |
| `obs_history_full__automaton_potential` | **188/192 = 97.9%** |

`automaton_potential` credits every milestone equally, so the longer drawer
branch and the shorter window branch are worth the same per step.  When the
planner under-reads the drawer branch it re-runs `drawer_open`, under-reads
again, and has no incentive to switch to the window branch that would still make
progress.  `event_progress` normalises each branch to one before averaging, so a
window milestone is worth more (0.15 against 0.1125) and a stuck branch is
abandoned.

### Correction to a previously stated generalisation

The ablation write-up reported that under-reads are cheap - 60.51% of
under-read-only episodes still succeeded - and treated that as a property of the
error direction.  It is not: it was measured under `event_progress` only.  Under
`automaton_potential`, `obs_history_full` under-read-only episodes succeed at
**2%** (141 episodes).  Over-reads remain uniformly fatal in both feedbacks; it
is the *recoverability of under-reads* that is feedback-dependent.

### Caveat

`automaton_potential` is not broken: it reaches 75.00% with exact state, tied
with `event_progress`.  Its point estimate there is 1.56 points lower, so the
learned observer may be amplifying a small difference rather than creating one.
Either way the amplification is the result, and it is large.
