# Scene skill-failure sweep: a falsifiable prediction of the latching mechanism

Locked 2026-09-04, before any cell of this sweep was evaluated.

## The prediction being tested

The ablation established that a learned action-only observer is dead reckoning:
statistically identical to replaying the transition model over executed skills
(+0.26, CI [-1.30, +1.82]), and the arm with the most over-reads (368 in 2458
decisions), its dominant confusions being window relocks it credited without
evidence.  `obs_history_full` reads the scene instead and has one over-read in
2411 decisions.

That mechanism makes a directional prediction that has not yet been tested:

> Injecting skill failures - attempts that achieve nothing - should hurt
> dead-reckoning arms far more than the observation-history arm, because only
> the dead-reckoning arms infer progress from the attempt rather than from the
> scene.

If instead `obs_history_full` degrades as fast as `action_only_full`, the
mechanism as written is wrong: the observation history would not be supplying
independent evidence about whether a skill worked.

## Failure injection

With probability `p`, a deployed skill is executed and then the pre-skill
snapshot is restored.  The robot moves and returns; physical time passes and
nothing changes, which is exactly "attempted but achieved nothing".  The
believed and true event states both stay where they were.

The draw is deterministic in `(reset_seed, decision index, skill index, p)`, so
the same skill attempted at the same decision fails identically across arms, and
`p = 0` injects nothing.

`p` in `{0.00, 0.10, 0.20, 0.30}`.

## Arms

Feedback fixed at `branch_w050`, the centre of the tolerant plateau.  Five state
sources, 14 arms, times 4 failure rates = 56 arms per reset, all from one
restored snapshot per reset inside one job.

| Arm | Observation | Action | Trained |
|---|---|---|---|
| `oracle_event` | simulator q | simulator q | no |
| `openloop_transition` | none | executed skills | no |
| `frame_full` | current frame | none | 3 seeds |
| `action_only_full` | none | full prefix | 3 seeds |
| `obs_history_full` | full prefix | none | 3 seeds |
| `history_full` | full prefix | full prefix | 3 seeds |

Task 5 only, 64 resets, seeds `88500-88563`.  The `p = 0.00` cells must
reproduce the task-5 half of `ablation_20260904` exactly; that is the
determinism check.

Everything else is held fixed: same H1b transition checkpoint, same observer
checkpoints, same skills, Skill-UCT at budget 112, horizon 4, exploration 0.55,
same search-seed formula.

## Primary endpoint

For each state source `s`:

    DEGRADATION(s) = success(s, p=0.00) - success(s, p=0.30)

paired by reset, model seeds averaged within a reset, bootstrapped over reset
clusters.

## Locked decision rules

- `NONDETERMINISTIC_EVAL`: any `p = 0.00` cell fails to reproduce
  `ablation_20260904`.  No other verdict may be issued.
- `MECHANISM_CONFIRMED`: both
  `DEGRADATION(action_only_full) - DEGRADATION(obs_history_full)` and
  `DEGRADATION(openloop_transition) - DEGRADATION(obs_history_full)` have 95% CI
  lower bounds above zero.
- `MECHANISM_REFUTED`: either difference has a CI upper bound at or below zero,
  i.e. the observation-history arm degrades at least as fast as a dead-reckoning
  arm.
- `INCONCLUSIVE`: anything else.

Secondary, reported per source and rate: over-read and under-read counts,
exact-q rate, timeout rate.  The mechanism additionally predicts that the rise
in over-reads with `p` is concentrated in the dead-reckoning arms; that is a
descriptive check, not part of the gate.

## Declared limitations

- A restored snapshot is a *silent* failure: nothing in the scene marks the
  attempt as having happened.  Real skill failures often leave traces (a
  displaced object, a different arm pose), which would make the observation
  history strictly more informative.  This injection is therefore the harder
  case for `obs_history_full`, not the easier one.
- One automaton, one task, one planner, one feedback.
- `p` is applied uniformly across skills; failure probability is not made to
  depend on the contact difficulty of the skill.

## Amendment before the locked run (2026-09-04, after plumbing smoke `49453`)

The plumbing smoke revealed a design confound and it is fixed here, before any
evaluation data was collected: the smoke is a plumbing check on a single reset
and no cell of it is used as evidence.

Task 5 needs eight skills and the budget was ten decisions.  A failed attempt
still consumes a decision, so at `p >= 0.20` every arm including `oracle_event`
simply ran out of turns.  `DEGRADATION` would then have measured budget
exhaustion, which all arms share, rather than mis-inference after a failure,
which is the prediction under test.

The decision budget is therefore scaled as `ceil(10 / (1 - p))`, giving 10, 12,
13 and 15 decisions at `p = 0.00, 0.10, 0.20, 0.30`, so the expected number of
*effective* decisions is held roughly constant.  `p = 0.00` is unchanged at 10,
so the reproduction anchors still hold.

This has a side effect that is worth stating because it cuts against the
prediction: the observers were trained on histories of at most ten steps, so at
`p = 0.30` the history arms can be asked for up to fifteen, off their training
distribution, while the frame arm is unaffected.  The compensation therefore
handicaps `obs_history_full` and `history_full` specifically.  Occurrences are
recorded per decision as `beyond_trained_history`.

## Outcome (jobs 49453-49468, 2026-09-04)

Verdict: **`MECHANISM_CONFIRMED`**.

Reproduction of the `p = 0.00` column against `ablation_20260904`: 896 rows,
**0 mismatches**.

| State source | p=0.00 | p=0.10 | p=0.20 | p=0.30 | degradation |
|---|---:|---:|---:|---:|---|
| `oracle_event` | 76.56% | 75.00% | 76.56% | 75.00% | +1.56 [-10.94, +15.62] |
| `frame_full` | 54.17% | 57.81% | 61.98% | 54.69% | -0.52 [-12.50, +11.98] |
| `obs_history_full` | 89.58% | 89.58% | 88.02% | **78.65%** | +10.94 [+2.60, +19.79] |
| `history_full` | 83.33% | 68.75% | 61.46% | 37.50% | +45.83 [+34.38, +57.29] |
| `openloop_transition` | 62.50% | 28.12% | 10.94% | **3.12%** | +59.38 [+46.88, +71.88] |
| `action_only_full` | 63.02% | 29.69% | 14.06% | **3.12%** | +59.90 [+47.92, +71.35] |

| Contrast | points | 95% CI |
|---|---:|---|
| `EXTRA_DEGRADATION_ACTION_ONLY_FULL` | **+48.96** | [+34.38, +63.02] |
| `EXTRA_DEGRADATION_OPENLOOP_TRANSITION` | **+48.44** | [+33.33, +63.02] |

### The mediator moves only where predicted

Mean over-reads per episode:

| State source | p=0.00 | p=0.10 | p=0.20 | p=0.30 |
|---|---:|---:|---:|---:|
| `obs_history_full` | 0.01 | 0.03 | 0.06 | **0.20** |
| `frame_full` | 0.98 | 1.26 | 1.20 | **1.36** |
| `history_full` | 0.01 | 0.78 | 1.26 | **3.54** |
| `openloop_transition` | 1.62 | 5.77 | 8.45 | **11.66** |
| `action_only_full` | 1.89 | 5.74 | 8.32 | **11.66** |

Under-reads stay flat in every arm.  Silent skill failures therefore create
over-reads *only* in arms that infer progress from the attempt, which is the
mediator the mechanism named in advance.  `frame_full` is the clean control: it
never sees the attempt, so its error profile is unchanged by `p` even though its
absolute performance is poor.

### Action tokens are a liability, now causally

`history_full` has the observation prefix *and* the action prefix.  It degrades
by +45.83 points and its over-reads rise from 0.01 to 3.54, while
`obs_history_full` - the same architecture minus the action tokens - holds at
0.20.  The earlier correlational finding that action tokens add nothing and mildly
hurt now has a dose-response behind it: they are what converts a silent skill
failure into a fatal over-read.

### What `obs_history_full` actually loses

Its degradation of +10.94 points is real and CI-clean, but it is not
mis-inference: exact-q stays at 91.2% and over-reads at 0.20 across the sweep.
What rises is the timeout rate, 10.4% to 21.4%.  It loses episodes to wasted
attempts, not to wrong beliefs.

The budget compensation handicapped it as predicted: at `p = 0.30` it was asked
for 2.24 decisions per episode beyond its ten-step trained history length, and
still held 78.65%.  The test was run in the direction unfavourable to the
hypothesis.
