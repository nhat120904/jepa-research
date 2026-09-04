# Scene event-observer input ablation: is the history advantage dead reckoning?

Locked 2026-09-04, before any ablation arm was evaluated in closed loop.

## The threat this tests

The confirmatory factorial showed `history_full` at 90.62% against `frame_full`
at 69.01% (`HISTORY` +21.61, CI [+15.36, +28.39]).  The claim attached to that
number is that the event state must be inferred from the observation *and*
action history.

But the automaton advances almost deterministically with the executed skill
sequence whenever skills succeed.  A history observer could therefore ignore the
image entirely and replay the transition dynamics it was trained alongside.  If
so the result is a restatement of the H1b abstract transition model, not a
perception claim, and the honest response is to reframe rather than to publish
"history-conditioned perception".

Static accuracy cannot settle this.  On the held-out validation distribution all
three history variants are saturated and indistinguishable: `obs_history_full`
99.70%, `action_only_full` 99.70%, `history_full` 99.85%.  `action_only_full`
reaches that without seeing a single pixel, because on canonical-prefix-plus-one
-deviation sequences the action prefix determines the event state.  Only closed
loop deployment, where skills sometimes fail and the action-to-state map breaks,
can separate them.

## Arms

Run on one restored snapshot per reset, same frozen H1b transition checkpoint,
same Skill-UCT budget 112, horizon 4, exploration 0.55 and the same search seed
formula as the confirmatory run.

| Arm | Observation input | Action input | Trained? |
|---|---|---|---|
| `oracle_event` | simulator q | simulator q | no |
| `abstract_terminal` | simulator q | simulator q | no |
| `openloop_transition` | **none** | executed skills | **no** |
| `frame_full` | current frame | none | reused |
| `obs_history_full` | full prefix | **none** | 3 seeds |
| `action_only_full` | **none** | full prefix | 3 seeds |
| `history_full` | full prefix | full prefix | reused |

`openloop_transition` needs no training: the planning state starts at
`initial_milestones` and is advanced by applying the frozen abstract transition
model to each executed skill.  That model's update is monotone by construction,
so this arm cannot revise a belief downwards - which is precisely the dead
reckoning failure mode the mechanism predicts.

The ablated arms share the architecture, parameter count, optimiser, data and
coverage of `history_full`; `action_only_full` zeroes the visual features after
standardisation and `obs_history_full` forces every action token to `NO_SKILL`.
Each evaluation job asserts the corresponding invariance on random probes before
running, so an ablation that silently failed to apply aborts the job instead of
producing a flattering number.

## Reset band and reproduction check

The confirmatory band is reused: 128 resets, seeds `88400-88463` (task 4) and
`88500-88563` (task 5).  `frame_full` and `history_full` are re-run here with the
same checkpoints and search seeds, so their per-reset outcomes must reproduce
jobs `49328`/`49329` exactly.  Any mismatch means the evaluation is not
deterministic and every paired contrast in this project is weaker than claimed;
the analysis reports the mismatch count and refuses to issue any other verdict
if it is non-zero.

Reusing the band is deliberate: it makes every contrast paired within one job
and costs no fresh resets, and the ablation arms have never been evaluated on
any reset.

## Locked decision rules

All contrasts paired by reset, model seeds averaged within a reset, bootstrapped
over reset clusters.

- `VISION_GIVEN_ACTIONS = history_full - action_only_full`
- `ACTIONS_GIVEN_VISION = history_full - obs_history_full`
- `ACTION_ONLY_VS_OPENLOOP = action_only_full - openloop_transition`
- `OBS_HISTORY_VS_FRAME = obs_history_full - frame_full`
- `OPENLOOP_VS_HISTORY_FULL = openloop_transition - history_full`

Verdicts:

- `NONDETERMINISTIC_EVAL`: any reproduction mismatch on `frame_full` or
  `history_full`.  No other verdict may be issued.
- `DEAD_RECKONING_REFUTED`: `VISION_GIVEN_ACTIONS` has a 95% CI lower bound
  above zero **and** `history_full` beats `openloop_transition` with a CI lower
  bound above zero.  The observer genuinely uses vision and is not replaying the
  transition model.
- `DEAD_RECKONING_SUFFICIENT`: `action_only_full` or `openloop_transition` is
  within 5 points of `history_full` with a CI covering zero.  The perception
  framing is withdrawn and the result is rewritten as open-loop skill-sequence
  tracking.
- `PARTIAL`: anything else, reported as such without choosing a framing.

Whatever the verdict, the error-direction audit is re-run on these arms.  The
mechanism predicts that `openloop_transition` and `action_only_full` produce
over-reads on skill failure while `obs_history_full` does not, and that success
tracks over-read count rather than exact-q accuracy.

## Declared limitations

- Reusing the confirmatory band means these contrasts are not independent of the
  band that produced the headline number.  The ablation arms are new to it, but a
  fully fresh replication would be stronger.
- `openloop_transition` is charitable to dead reckoning in one way and harsh in
  another: it uses the transition model directly rather than a network trained to
  imitate it, and it has no mechanism to detect skill failure at all.
  `action_only_full` is the fairer dead-reckoning arm.

## Outcome (jobs 49360-49390, 2026-09-04)

Verdict: **`DEAD_RECKONING_REFUTED`**.

Reproduction check: 768 rows of `frame_full` and `history_full` compared against
jobs `49328`/`49329`, **0 mismatches** in both success and deployed skill
sequence.  The evaluation is deterministic, so every paired contrast in this
project rests on exact re-execution.  Ablation invariance was verified on all
128 shards.

| Arm | Observation | Action | success | 95% CI | task 4 | task 5 |
|---|---|---|---:|---|---:|---:|
| `oracle_event` | simulator q | simulator q | 87.50% | [81.25, 92.97] | 98.44% | 76.56% |
| `abstract_terminal` | simulator q | simulator q | 2.34% | [0.00, 5.47] | 0.00% | 4.69% |
| `openloop_transition` | none | executed skills | 80.47% | [73.44, 86.72] | 98.44% | 62.50% |
| `frame_full` | current frame | none | 69.01% | [61.20, 76.04] | 83.85% | 54.17% |
| `action_only_full` | none | full prefix | 80.73% | [73.95, 86.98] | 98.44% | 63.02% |
| `history_full` | full prefix | full prefix | 90.62% | [85.68, 95.05] | 97.92% | 83.33% |
| `obs_history_full` | full prefix | **none** | **93.49%** | [89.06, 97.14] | 97.40% | **89.58%** |

| Contrast | points | 95% CI |
|---|---:|---|
| `VISION_GIVEN_ACTIONS` | +9.90 | [+4.69, +15.62] |
| `ACTIONS_GIVEN_VISION` | -2.86 | [-6.25, +0.26] |
| `ACTION_ONLY_VS_OPENLOOP` | +0.26 | [-1.30, +1.82] |
| `OBS_HISTORY_VS_FRAME` | +24.48 | [+17.45, +32.03] |
| `OPENLOOP_VS_HISTORY_FULL` | -10.16 | [-15.89, -4.69] |

### What this changes

Dead reckoning is refuted, but the decomposition also **overturns the mechanism
previously written up**, and the correction runs the other way:

1. `action_only_full` is dead reckoning.  It is statistically identical to
   simply replaying the frozen transition model (`+0.26`, CI [-1.30, +1.82]),
   so the learned action-only observer contributes nothing beyond the transition
   dynamics we already had.
2. Dead reckoning is nonetheless **strong** - 80.47%, well above the 69.01% of a
   single frame.  Had the ablation not been run, the history result would have
   been substantially attributable to it.
3. The actual fix is the **observation** prefix, not the action prefix.
   `OBS_HISTORY_VS_FRAME` is +24.48 and accounts for the entire history effect,
   while `ACTIONS_GIVEN_VISION` is -2.86 with a CI that barely reaches zero:
   action tokens add nothing once observation history is present, and may cost a
   little.

The error-direction audit (`49389`) shows why, and it is the reverse of the
earlier reading:

| Arm | decisions | exact | behind | ahead + mixed |
|---|---:|---:|---:|---:|
| `frame_full` | 2555 | 76.5% | 353 | 247 |
| `openloop_transition` | 818 | 87.0% | **0** | 106 |
| `action_only_full` | 2458 | 83.9% | 28 | **368** |
| `obs_history_full` | 2411 | 93.2% | 163 | **1** |
| `history_full` | 2408 | 94.3% | 134 | 3 |

Action information is what **causes** over-reading, not what prevents it.
Knowing which skill was attempted tempts the model to assume it succeeded;
`openloop_transition` cannot under-read at all by construction (0 behind, 106
ahead), and `action_only_full` produces the most over-reads of any arm.  Its top
confusions are exactly assumed successes - `[5,2] -> [5,3]`, `[3,2] -> [3,3]`,
`[4,2] -> [4,3]`, all of them window relocks that did not happen.  The
observation prefix is what grounds the estimate: it drives over-reads to 1 in
2411 decisions.

The asymmetry itself replicates.  Pooled over every arm with an inferred state:
over-read episodes succeed 4/201 = 1.99%, under-read-only 95/157 = 60.51%,
error-free 1286/1306 = 98.47%.

**Correction (`docs/SCENE_STATE_VS_FEEDBACK_PROTOCOL.md`).**  The 60.51% figure
is not a property of the error direction; every arm here uses the same
`event_progress` feedback.  Under a hint^2 style `automaton_potential`,
`obs_history_full` under-read-only episodes succeed at 2% rather than 76%,
because the planner livelocks on the branch it keeps under-reading instead of
switching branches.  Over-reads are fatal under both feedbacks; the
recoverability of *under*-reads is feedback-dependent.

### Exploratory, not preregistered

`obs_history_full` beats planning with the simulator's own event state:
`+5.99` points CI [+1.04, +11.46] pooled, and `+13.02` CI [+4.17, +22.40] on
task 5, where task 4 is saturated for every arm.  A plausible reading is that
its under-reads act as conservatism, making the planner re-verify a milestone
the oracle commits past, and under-reads are cheap (60.51%) while over-reads are
fatal (1.99%).  That mechanism is **untested**; the contrast was not
preregistered and is reported as exploratory only.

## Why the current frame is insufficient (job 49393, exploratory)

`advance_milestones` only ever raises a stage, but the scene itself can regress:
opening the window reaches `window_stage=2`, and closing it again leaves the
stage at 2 while the image now looks like `window_stage=1`.  The same holds for
`toggle_button_0` re-pressed after `cube_stage` has latched at 1.  From that
moment the event state stops being a function of the current image.

Every error, in every arm, in both directions, occurs after such a regressing
skill has been deployed - 100.0% against a 52-63% base rate on correct
decisions:

| Arm | bucket | n | after a regressing skill |
|---|---|---:|---:|
| `frame_full` | window under-read | 324 | 100.0% |
| `frame_full` | window over-read | 97 | 100.0% |
| `frame_full` | cube under-read | 154 | 100.0% |
| `frame_full` | cube over-read | 155 | 100.0% |
| `frame_full` | correct (control) | 2134 / 2246 | 52.1% / 62.6% |
| `action_only_full` | window over-read | 255 | 100.0% |
| `action_only_full` | cube over-read | 174 | 100.0% |
| `obs_history_full` | window under-read | 28 | 100.0% |
| `obs_history_full` | cube under-read | 138 | 100.0% |
| `obs_history_full` | correct (control) | 2383 / 2272 | 59.2% / 61.4% |

This pins the mechanism, and it is **not** "the observation history verifies
whether the attempted skill succeeded".  The best arm, `obs_history_full`, has
no action tokens at all and cannot compare intent against outcome.  What it does
is reconstruct latched progress from what it has seen: having observed the
window open, stage 2 holds no matter how the scene looks now.

The verification framing also makes a prediction the data refutes.  If the
mechanism were attempt-versus-outcome verification, knowing the attempt should
help, and `history_full` should beat `obs_history_full`.  It does not
(`ACTIONS_GIVEN_VISION` -2.86, CI [-6.25, +0.26]).

`action_only_full` fails for the complementary reason: it knows a skill was
attempted but has never seen whether the predicate held, so it assumes the latch
advanced.  Its top confusions are exactly that - `[5,2] -> [5,3]`,
`[3,2] -> [3,3]`, `[4,2] -> [4,3]`, all window relocks it credited without
evidence.

Both the regressing-skill audit and the oracle comparison are exploratory and
were not preregistered.
