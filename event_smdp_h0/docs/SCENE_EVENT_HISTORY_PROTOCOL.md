# Scene event-observer factorial: coverage versus history

Locked 2026-09-04, before any arm of this factorial was trained.

## Why this replaces a history-only intervention

The posterior-support gate (`49317`) returned `HISTORY_REQUIRED`: on hard MAP
errors the single-frame observer gives the true event state ~1e-07 probability
at rank 20/48, so no Bayesian wrapper over that posterior can recover it.  That
verdict is about the *posterior*, not about the *cause*.

The coverage audit (`49320`) then localised the cause:

| Visited state | n | exact-q accuracy |
|---|---:|---:|
| inside canonical training support | 2138 | 95.23% |
| outside canonical training support | 311 | 0.00% |
| outside canonical but inside full support | 311 | 0.00% |
| outside full support | 0 | n/a |

Every deployment failure is a state the collected data already contains as a
counterfactual endpoint but that the observer never saw, because
`train_scene_event_observer.py` deduplicated to canonical milestone roots.  Task
5 has 16 canonical-root event states against 22 in the full data; 368 training
samples against 2624.  The uncovered states are `(cube=1, window=2)` with 301
visits and `(cube=1, window=1)` with 10.

So there are two competing explanations for the observed failure, and the
history intervention only addresses one of them:

- **coverage** - the classifier projects unseen event states onto the nearest
  seen state, and simply training on the states already collected fixes it;
- **history** - the current frame is genuinely insufficient and the event state
  needs the observation/action prefix.

## Design

A single GRU architecture, `HistoryEventObserver`, serves both input regimes.
`history_length=1` feeds only the current observation with a `NO_SKILL` token;
the full setting feeds the whole `(previous skill, observation)` prefix.
Parameter count, optimiser, data tensors and code path are otherwise identical,
so the input contrast is not confounded with capacity.

| Arm | Input | Training states |
|---|---|---|
| `frame_canonical` | current frame | canonical roots only |
| `frame_full` | current frame | roots + counterfactual endpoints |
| `history_canonical` | full prefix | canonical roots only |
| `history_full` | full prefix | roots + counterfactual endpoints |

Three model seeds per arm.  No new simulation data is collected: sequences are
reconstructed from the existing H1 shards, whose canonical paths are a
deterministic function of `task_id`, asserted against
`collect_scene_h1.canonical_paths` at training time.  Train/validation reset
splits are inherited unchanged, so no reset leaks across the split.

Everything downstream of the observer is frozen and shared: the H1b abstract
transition checkpoint (`scene_h1b/checkpoints/seed0/abstract_smdp.pt`),
Skill-UCT with budget 112, horizon 4, exploration 0.55, and the same search
seed formula.  `oracle_event` (simulator q) and `abstract_terminal` run from the
same restored snapshot inside the same job.

## Evaluation

Fresh reset band `86000 + 100*task + local`, disjoint from every seed used so
far (`81xxx` collection, `83xxx` H1/H1b/H2/pilot, `84xxx` replication, `85xxx`
belief filter).

- Pilot: 32 resets, 16 per task.  Exploratory; may not be reported as the claim.
- Confirmatory: 128 fresh resets, 64 per task, run only if the pilot passes.

Primary endpoint is physical stable success at K=112, paired by reset,
bootstrapped over reset clusters.

## Locked decision rules

Contrasts, each paired by reset and averaged over the three model seeds:

- `COVERAGE  = frame_full - frame_canonical`
- `HISTORY   = history_full - frame_full`
- `HISTORY_UNDER_CANONICAL = history_canonical - frame_canonical`

Verdicts:

- `COVERAGE_IS_THE_FIX`: `frame_full` mean success >= 85%, **every** model seed
  >= 75% (the bar the previous replication missed), and the `COVERAGE` paired
  95% CI excludes zero.
- `HISTORY_ADDS_VALUE`: additionally `HISTORY` paired 95% CI lower bound > 0 and
  point estimate >= 5 points.  Only then is history part of the contribution.
- `HISTORY_REQUIRED_CONFIRMED`: `frame_full` fails its gate while `history_full`
  passes it.
- `BOTH_FAIL`: neither reaches the `frame_full` bar; the observer is not the
  remaining bottleneck and the event-perception line stops here.

If `COVERAGE_IS_THE_FIX` holds and `HISTORY` does not, history is reported as a
tested-and-unnecessary ablation, not as a contribution.

## Declared limitations

- Training histories are a canonical prefix plus at most one deviating skill,
  while deployment histories are planner-generated and may deviate repeatedly.
  A history arm is therefore evaluated off its training sequence distribution;
  this is a stated weakness of the history arm, not a defect of the contrast.
- The automaton and the skill library remain hand-specified.
- `stable_count` is read from a sigmoid threshold at 0.5 in every arm.

## Outcome (jobs 49322-49345, 2026-09-04)

Verdict on the locked confirmatory run of 128 fresh resets (64 per task, seeds
`88400-88463` and `88500-88563`): **`HISTORY_REQUIRED_CONFIRMED`**, with
`history_adds_value = true` and `frame_full_passes = false`.

| Arm | success | 95% CI | seed 0 | seed 1 | seed 2 | task 4 | task 5 |
|---|---:|---|---:|---:|---:|---:|---:|
| `oracle_event` | 87.50% | [81.25, 92.97] | - | - | - | 98.44% | 76.56% |
| `abstract_terminal` | 2.34% | [0.00, 5.47] | - | - | - | 0.00% | 4.69% |
| `frame_canonical` | 74.22% | [66.67, 80.99] | 79.69% | 70.31% | 72.66% | 93.75% | 54.69% |
| `frame_full` | 69.01% | [61.20, 76.04] | 67.19% | 71.88% | 67.97% | 83.85% | 54.17% |
| `history_canonical` | 82.29% | [75.78, 88.02] | 84.38% | 78.91% | 83.59% | 98.44% | 66.15% |
| `history_full` | **90.62%** | [85.68, 95.05] | 88.28% | 89.84% | 93.75% | 97.92% | 83.33% |

| Contrast | points | 95% CI |
|---|---:|---|
| `COVERAGE` | -5.21 | [-12.24, +1.82] |
| `HISTORY` | **+21.61** | [+15.36, +28.39] |
| `HISTORY_UNDER_CANONICAL` | +8.07 | [+3.39, +13.28] |
| `COVERAGE_UNDER_HISTORY` | +8.33 | [+2.34, +14.58] |
| `HISTORY_FULL_VS_TERMINAL` | +88.28 | [+81.77, +93.75] |
| `HISTORY_FULL_VS_ORACLE` | +3.12 | [-1.82, +8.33] |

`history_full` is statistically indistinguishable from planning with the
simulator's own event state, and its point estimate is above it, driven by task
5 (83.33% against 76.56%).  That excess is not claimed: the contrast CI covers
zero.

Coverage does not behave additively.  On its own it is worthless or harmful,
but combined with history it is worth +8.33 points; the two axes interact.

### Mechanism

The pilot's static held-out accuracy pointed the other way - `frame_full`
reached 98% exact q against `frame_canonical`'s 48% - yet planned worse.  The
error-direction audit (`49345`) resolves this.  Milestones advance only through
causally ordered predicates, so the two error directions are not
interchangeable:

| Worst error in an episode | episodes | success |
|---|---:|---:|
| exact only | 1137 | 98.77% |
| under-reads only (`behind`) | 243 | 36.21% |
| any over-read (`ahead`/`mixed`) | 156 | **1.92%** |

An over-read makes the planner skip a prerequisite, and nothing downstream
supplies it; the episode is effectively lost.  Per-decision error direction:

| Arm | decisions | exact | behind | ahead + mixed |
|---|---:|---:|---:|---:|
| `frame_canonical` | 2521 | 75.0% | 606 | 25 |
| `frame_full` | 2555 | 76.5% | 353 | **247** |
| `history_canonical` | 2494 | 81.5% | 377 | 84 |
| `history_full` | 2408 | **94.3%** | 134 | **3** |

`frame_full` has the *higher* exact rate of the two frame arms and still plans
worse, because training on off-canonical endpoints buys accuracy by trading
harmless under-reads for fatal over-reads (25 -> 247).  Conditioning on history
removes over-reading almost entirely.  This makes exact-q accuracy a misleading
model-selection metric here and error direction the quantity that matters.

**Correction (input ablation, `docs/SCENE_EVENT_ABLATION_PROTOCOL.md`).**  An
earlier version of this section attributed the removal of over-reads to the
action prefix, reasoning that an observer seeing the executed skills cannot
credit progress no skill could have produced.  The ablation refutes that: the
*observation* prefix does the work (`OBS_HISTORY_VS_FRAME` +24.48), while the
action prefix on its own is dead reckoning and produces the **most** over-reads
of any arm (368 in 2458 decisions).  Action information tempts the model to
assume an attempted skill succeeded.  `history_full` is therefore the
observation-history result plus a small, non-significant cost from the action
tokens (`ACTIONS_GIVEN_VISION` -2.86, CI [-6.25, +0.26]).

### Notes on the declared limitations

`beyond_trained_history` fired on 0 of 9978 learned decisions, so no deployment
sequence ran past the trained history length.  The distribution concern that
remains is the *shape* of deployment prefixes, not their length.  The automaton,
skill library and labels are still hand specified, and `oracle_event` still
reads simulator q, so this result is about the observer, not about the
automaton's provenance.

---

# Addendum: input-ablation gate (locked 2026-09-04, before any ablation arm was evaluated)

## The threat this addresses

The automaton advances almost deterministically with the executed skill
sequence whenever skills succeed.  A history observer could therefore reach
94.3% exact q by *dead reckoning* - replaying the transition dynamics from the
action prefix and never really looking at the image.  If so, `history_full` is a
restatement of the H1b transition model, not a perception result, and the honest
baseline would be open-loop tracking with a model we already have.

Training-time evidence sharpens the worry rather than settling it: on the
validation distribution `action_only_full` reaches 99.70% exact q with the
visual features zeroed, statistically the same as `obs_history_full` (99.70%)
and `history_full` (99.85%).  The static metric is saturated and cannot
separate these arms, because the validation states are a canonical prefix plus
at most one deviation, where the action sequence very nearly determines `q`.
Only closed-loop deployment, where skills sometimes fail, breaks that
determinism.

## Arms

Run on the confirmatory reset band `88400-88463` and `88500-88563`, from the
same restored snapshot, same frozen H1b transition checkpoint, same Skill-UCT
budget 112, horizon 4, exploration 0.55, same search-seed formula.

| Arm | observation input | action input | trained? |
|---|---|---|---|
| `openloop_transition` | none | executed skill applied to the H1b model | **no** |
| `action_only_full` | zeroed after standardisation | full prefix | 3 seeds |
| `obs_history_full` | full prefix | all tokens forced to `NO_SKILL` | 3 seeds |
| `frame_full` | current frame only | none | reused |
| `history_full` | full prefix | full prefix | reused |

`frame_full` and `history_full` are re-run rather than copied so every contrast
is paired inside one job, and so their agreement with the confirmatory
artifacts becomes a reproduction check.

Each ablated observer must demonstrably ignore the input it is supposed to
ignore.  The evaluator probes this at startup and refuses to run otherwise.

## Locked decision rules

Contrasts, paired by reset, seeds averaged, bootstrapped over reset clusters:

- `VISION_GIVEN_ACTIONS = history_full - action_only_full`
- `ACTIONS_GIVEN_VISION = history_full - obs_history_full`
- `LEARNED_VS_OPENLOOP  = history_full - openloop_transition`
- `ACTION_ONLY_VS_OPENLOOP = action_only_full - openloop_transition`
- `OBS_HISTORY_VS_FRAME = obs_history_full - frame_full`

Verdicts:

- `DEAD_RECKONING_REFUTED`: both `VISION_GIVEN_ACTIONS` and `LEARNED_VS_OPENLOOP`
  have a 95% CI lower bound above zero.  The history observer is doing sensor
  fusion, and the perception framing stands.
- `DEAD_RECKONING_SUFFICIENT`: `action_only_full` or `openloop_transition` is
  within 5 points of `history_full` with a CI covering zero.  The perception
  claim collapses and the paper must be reframed around the transition model.
- `PARTIAL`: anything else, reported as such with no reframing either way.
- `NONDETERMINISTIC_EVAL` overrides all of the above if the re-run `frame_full`
  or `history_full` results disagree with the confirmatory artifacts on any
  reset; in that case pairing across runs is unsafe and nothing else is read.

Prediction registered before running: `openloop_transition` should fail badly
and specifically by *over-reading*, because the H1b rollout clamps event stages
monotonically and therefore credits every skill it believes succeeded, with no
mechanism to walk that back when a skill actually failed.
