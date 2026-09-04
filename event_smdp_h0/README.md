# Event-SMDP / Hazard-JEPA H0

This directory implements the corrected first gate for an action-conditioned
multi-state event-time world model.  It does **not** train Hazard-JEPA yet.

## Question

Under exact dynamics, fixed skill proposals, and a matched finite search budget,
does backing up intermediate event state produce higher stable task success than
backing up only terminal success?

Both arms use the same deterministic UCT implementation and the same
nominal-plus-noise lattice of 5-step skills.  Only the backed-up feedback differs:

- `terminal_only`: stable success after three settle steps;
- `event_state`: cyclic ordinal states `free -> contact -> manipulated -> near_goal
  -> goal -> stable_success`, with a post-lift drop branch and small event-time
  tie-breakers.

The proposal lattice includes the future demonstration chunk.  This is deliberately
privileged support for an H0 causal-room test and must not be described as a learned
action prior or as deployable policy evidence.

## Locked protocol

- OGBench Cube-single, goal offset 25, chunk size 5, depth 5.
- Fresh episode-disjoint windows with start-to-goal cube displacement at least 8 cm.
- Snapshot `0` is plumbing-only smoke; snapshots `1..64` are locked evaluation.
- Branching factor 5; variant 0 is nominal and variants 1..4 use fixed correlated
  action noise `[.24,.24,.18,.16,.10]`, rho `.60`.
- UCT exploration `.65`; locked budgets `16,32,64` simulations per replan.
- Full root-to-leaf rollouts and exact snapshot restoration make transition-call
  accounting identical between arms.
- Primary endpoint: paired stable-success difference at budget 64.
- `GO_CAUSAL_ROOM` requires difference at least 10 percentage points, bootstrap
  lower 95% bound above zero, and one-sided exact McNemar p below .05.
- If the upper 95% bound is below 10 points: `STOP_NO_MATERIAL_CAUSAL_ROOM`;
  otherwise the gate is inconclusive.

A GO only licenses the next experiment: replace privileged event labels and the
future-action support with learned event-time heads and an offline learned skill
prior, then audit calibration as search intensity rises.

All simulator and analysis execution is submitted through Slurm.  See
`docs/JOB_LEDGER.md` for exact submissions and states.

## Locked result (2026-09-03)

The H0 instantiation stops.  Jobs `48420_[1-64]` and `48436` completed all 64
paired snapshots without a failed repeatability or budget-accounting check.
The nominal support succeeded on 64/64, and both `terminal_only` and
`event_state` succeeded on 64/64 at every locked budget (16, 32, and 64).
Thus the primary paired difference is exactly `0.000 [0.000, 0.000]`, with no
discordant pairs and one-sided exact McNemar `p=1`.

This is a **proposal/search ceiling**, not evidence that event-time learning is
generally useless.  The privileged future-demonstration proposal lattice makes
terminal feedback sufficient after only 16 simulations per replan.  The smoke
snapshot's event-only success at budget 10 is plumbing-only and cannot override
the preregistered locked result; deliberately weakening the terminal baseline to
that budget would not be a compelling success-rate comparison.  Do not train a
Hazard-JEPA head on this OGBench protocol without first moving to a task/proposal
regime in which a strong terminal-head + same-planner baseline is below ceiling.

## Active replacement gate: OGBench-Scene (2026-09-03)

The replacement removes future-demonstration proposals and moves to Scene tasks
4 and 5. Both arms now share seven reusable closed-loop oracle skills and
receding-horizon Skill-UCT. Task 4 requires unlock/open/place/close; task 5
additionally requires relocking the drawer and unlock/open/relock of the window.
The final goal therefore cannot reveal a valid action ordering by itself.

The treatment is a history-bearing milestone automaton; the baseline receives
only three-step-stable native success. MuJoCo remains the oracle world model,
so this is still an H0 causal-room experiment rather than a learned-model
claim. See `docs/SCENE_GATE0_PROTOCOL.md`, `docs/SCENE_RESEARCH_POSITIONING.md`,
and `scripts/run_scene_gate0.py`.

The Scene H0 pilot passed strongly, but the locked contextual H1 learnability
pilot did not.  Recursive latent and privileged skill-dynamics models both
collapsed in closed-loop planning despite strong one-step event metrics.  The
bounded H1b audit then closed the learned SMDP directly in automaton state and
recovered 12/16 physical successes at budget 28 versus 0/16 for the terminal
baseline.  This isolates recursive feature drift as a material failure mode
and makes the event-state-closed model the active architecture.  The negative
H1 result remains preserved in `docs/SCENE_H1_PROTOCOL.md`; H1b details and
limits are in `docs/SCENE_H1B_PROTOCOL.md`.  The active next gate is H2:
determine whether selected error or overconfidence grows with search width.
The completed H2 audit did not find a significant increase from K=14 to K=112,
while Event-SMDP physical success rose monotonically to 16/16.  SearchCal is
therefore not licensed for this setup.  Both the recursive-latent terminal
baseline and a stronger shared-checkpoint abstract terminal-probability arm
remained at 0/16 through K=112.  The shared-model factorial thus isolates a
causal event-feedback gain: event success rises from 7/16 at K=14 to 16/16 at
K=112 while terminal-only stays at zero.

The subsequent deployment-facing perception pilot passed.  A latent
observer trained on deduplicated canonical roots inferred current `q` from a
fresh rendered observation at every replan; at K=112 it achieved 15/16 stable
successes and 94% exact-q accuracy on visited states, versus 0/16 for the
shared-checkpoint terminal arm and 16/16 with simulator-monitored `q`.

The licensed 3-observer-seed, 64-fresh-reset-per-task replication was a strict
near-miss FAIL.  Mean learned success was 84.11%, versus 1.56% terminal and
88.28% simulator-q event planning, but seed 1 achieved 95/128 = 74.22%, one
episode below the locked 75% per-seed requirement.  We do not relax that gate.
The failure audit instead reveals history aliasing: on task 5, seed 1 repeatedly
maps true event state `(1,2)` to `(0,1)` after initially correct decisions.  The
active follow-up is therefore a separately preregistered prediction--correction
event-belief filter on new resets, not SearchCal and not a retroactive relabeling
of the replication.  That hard-filter pilot also failed its locked aggregate
gate: task-5 mean success moved only from 62.50% to 66.67%, although it repaired
the unstable seed 1 from 43.75% to 68.75%.  The irreversible MAP/lattice join
hurt the other two seeds, so it must not be scaled.  A future filter would need
a calibrated distribution over event states and reversible evidence weighting;
the strong dense event/value baselines remain mandatory.  The automaton and
labels remain hand specified throughout.

The posterior-support gate then closed the belief-filter line for the right
reason: on the 197 hard misreads the single-frame observer gives the true event
state ~1e-07 probability at rank 20/48, and 0.00% top-3 coverage on the
catastrophic `(1,2)` state, so no Bayesian wrapper over that posterior can
recover it.  A coverage audit changed the diagnosis: the deployed observer
scores 95.23% exact q on the 2138 visits inside its canonical training support
and 0.00% on the 311 outside it, and every one of those 311 visits is a state
the collector had already recorded as a counterfactual endpoint.  The original
training script had deduplicated 2624 labelled states down to 368 canonical
roots.

That produced two competing explanations - coverage and history - so the active
experiment is a matched 2x2 factorial (`docs/SCENE_EVENT_HISTORY_PROTOCOL.md`),
locked before training.  One GRU architecture serves both input regimes at
identical parameter count; arms differ only in current-frame versus full
`(previous skill, observation)` prefix, and in canonical-roots-only versus
roots-plus-endpoints training states.  Static held-out accuracy predicted
coverage would be the fix, but planning reversed it: on 32 pilot resets
`frame_full` fell to 68.75% while `history_full` reached 87.50%, tying
simulator-q oracle planning exactly, with `HISTORY` +18.75 points CI
[+8.33, +31.25] and `COVERAGE` -6.25 points CI [-20.83, +8.33].  The locked
128-fresh-reset confirmatory run reproduced this: `history_full` 90.62% CI
[85.68, 95.05] with every model seed above the 75% bar, against `frame_full`
69.01%, `frame_canonical` 74.22% and terminal 2.34%.  `HISTORY` is +21.61 points
CI [+15.36, +28.39], `COVERAGE` -5.21 CI [-12.24, +1.82], and `history_full` is
indistinguishable from planning with the simulator's own event state (+3.12 CI
[-1.82, +8.33]).

The error-direction audit explains the reversal.  Because milestones are
causally ordered, under-reading progress is recoverable but over-reading it is
not: every episode containing an `ahead` or `mixed` event-state error failed,
1.92% across 156 such episodes at confirmatory scale, against 98.77% for
episodes with no misread and 36.21% for episodes whose misreads only under-read
progress.  Coverage training raises the raw exact rate (75.0% -> 76.5%) while
turning 25 over-reads into 247, which is why the more accurate frame observer
plans worse.  Conditioning on history removes over-reading almost entirely
(94.3% exact, 3 over-reads in 2408 decisions).  Exact-q accuracy is therefore a
misleading selection metric here and error direction is the quantity that
matters.  Coverage and history also interact: coverage alone
is worthless, but +8.33 points CI [+2.34, +14.58] once history is present.  No
deployment sequence exceeded the trained history length (0 of 9978 decisions).
The automaton, skill library and labels remain hand specified, and the oracle
arm still reads simulator q, so this result concerns the observer only.


A locked input ablation (`docs/SCENE_EVENT_ABLATION_PROTOCOL.md`) then tested
whether that history advantage was really perception.  The event automaton
advances almost deterministically with the executed skill sequence, so a history
observer could have been replaying the H1b transition model rather than reading
the scene.  Static accuracy could not settle it: `obs_history_full`,
`action_only_full` and `history_full` all sit at 99.7-99.85% on held-out data,
and `action_only_full` reaches that without seeing a pixel.

Closed-loop planning separates them, and the verdict is
`DEAD_RECKONING_REFUTED` with an exact 768-row reproduction of the confirmatory
arms.  Dead reckoning is real and strong - an untrained `openloop_transition`
arm reaches 80.47% and a learned action-only observer is statistically identical
to it (+0.26, CI [-1.30, +1.82]) - but it does not explain the result: vision
given actions is worth +9.90 points, CI [+4.69, +15.62].

The ablation also **corrects the mechanism**.  The earlier write-up credited the
action prefix with suppressing over-reads; the opposite is true.  The
observation prefix carries the whole effect (`OBS_HISTORY_VS_FRAME` +24.48, CI
[+17.45, +32.03]) while action tokens add nothing once it is present
(`ACTIONS_GIVEN_VISION` -2.86, CI [-6.25, +0.26]).  Action information *causes*
over-reading: `openloop_transition` cannot under-read at all by construction (0
behind, 106 ahead) and `action_only_full` has the most over-reads of any arm
(368), its top confusions being window relocks it assumed had succeeded.  The
best arm is `obs_history_full` at 93.49%, with a single over-read in 2411
decisions.  Exploratory and not preregistered: it exceeds planning with the
simulator's own event state by +5.99 points CI [+1.04, +11.46], and by +13.02 CI
[+4.17, +22.40] on task 5.

A locked 3x2 grid (`docs/SCENE_STATE_VS_FEEDBACK_PROTOCOL.md`) then crossed the
event-state source with the planner's scalar feedback, adding a hint^2 style
`automaton_potential` that credits every remaining automaton milestone equally.
On task 4 the two feedbacks are identical by construction and matched exactly;
the contrast lives on task 5.

The locked verdict is `INCONCLUSIVE`, because neither preregistered hypothesis
holds.  With the simulator's own event state the two dense feedbacks are
interchangeable (76.56% against 75.00%, CI covering zero).  With
`obs_history_full` they differ by 87.50 points (89.58% against 2.08%), on
byte-identical observer checkpoints.  The result is therefore an interaction:
robustness to state-estimation error is a property of the feedback function, and
it is invisible to oracle-state evaluation.  It also refutes the tidier claim
that once the state is recovered any automaton-based feedback works.

The mechanism is livelock, not misplanning.  Under `automaton_potential` the
same observer's under-reads rise from 163 to 1034 while over-reads stay near
zero, its dominant confusion being a drawer opened then closed that latches
`cube_stage` at 2 while reading as 1; 97.9% of task-5 episodes then exhaust the
decision budget.  Because that feedback credits every milestone equally, the
planner re-runs the branch it under-reads instead of switching to the branch that
would still progress.  This corrects an earlier generalisation: over-reads are
fatal under both feedbacks, but the 60.51% recoverability of under-reads was a
property of `event_progress`, not of the error direction.

A preregistered feedback-robustness sweep
(`docs/SCENE_FEEDBACK_ROBUSTNESS_PROTOCOL.md`) then asked whether the surviving
feedback was a lucky point.  It first established that the two feedbacks of the
3x2 grid are not two families at all: both equal
`0.90 * (w * cube/5 + (1-w) * window/3)` at `w = 0.500` and `w = 0.625`, exactly,
on every event state.  The 87.50-point collapse is produced by moving one
branch-weight parameter by 0.125, so the sweep walks that parameter and adds two
designs from outside the family.

The verdict is `PARTIAL`, 4 of 8 feedbacks passing the strict gate, but the
substantive answer is that `event_progress` is **not** a knife edge: with
`obs_history_full`, `w = 0.300`, `0.400` and `0.500` give 92.71%, 92.71% and
89.58%, a plateau spanning at least `[0.30, 0.50]`, and `shaped_gamma09` (92.19%)
and `anti_livelock` (85.94%) hold up from outside the family.  The collapse is
one-sided and abrupt, falling between `w = 0.500` and `w = 0.5625`.

The methodological finding sharpened considerably.  The feedback that is best
under the simulator's own event state, `branch_w070` at 98.44%, is the second
worst under a learned one at 8.85%; oracle and learned rankings across the eight
feedbacks correlate at Spearman 0.357 (exploratory, n=8).  Oracle-state
evaluation does not merely add noise to a feedback comparison here - it inverts
the top of the ranking.

Two honest negatives: the deliberately designed `anti_livelock` feedback
preserves but does not beat plain `branch_w050` (85.94% against 89.58%), so
robustness came from the branch weighting rather than from an explicit
repetition penalty; and `branch_w040` reaches 92.71% yet misses the preserve gate
on CI width alone, which was not reclassified after the fact.