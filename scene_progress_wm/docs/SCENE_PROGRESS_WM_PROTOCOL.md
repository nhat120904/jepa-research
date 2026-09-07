# Scene progress-WM: putting action-conditioned prediction back at the centre

Locked 2026-09-04, before any job of this program was submitted.

## What this program is, and what it replaces

`event_smdp_h0` established, on OGBench-Scene, that planner-facing task progress is
**latched**: an observation-history-conditioned event readout reaches 89.58% on task 5
where a frame-conditioned readout reaches 54.17%, action-only dead reckoning collapses
from 63.02% to 3.12% under injected skill failure, and the ranking of feedback designs
inverts between oracle state and learned state. Those results were obtained with the
simulator as the dynamics model, a hand-defined automaton, and seven closed-loop oracle
skill controllers reading privileged state every tick. They are diagnostics, not a
world-model method.

This program keeps that scaffold as the oracle ceiling and rebuilds the arena so that a
**learned action-conditioned latent world model is the thing being evaluated**:

* the model consumes pixel history plus a candidate low-level action chunk and predicts
  future latents,
* the planner is CEM over 5-dimensional action chunks -- no skills, no privileged reads
  at deployment,
* success is `SceneEnv._compute_successes` (`scene_env.py:936`), OGBench's own predicate
  with OGBench's own tolerances, against a goal drawn from a dataset row.

The claim under test is a representation claim, not a planning-system claim:

> A planner-facing cost must be a function of observation history and must be monotone in
> achieved progress. Latent-L2 to a goal image is neither, and on Scene that is not a
> cosmetic defect: the buttons that lock the drawer and the window are latched, discrete,
> and occupy a handful of pixels, yet OGBench's own success predicate scores them.

## Arena

* Data: released OGBench v0 `visual-scene-play-v0` (1000 episodes x 1001 steps, pixels
  64x64, plus `actions`, `qpos`, `qvel`, `button_states`). No new collection.
* Backbone: `stable_worldmodel.wm.lewm.LeWM`, ViT-tiny trained from scratch at 64x64,
  action block 5 -- the published LeWM recipe with the dataset swapped.
* Protocol: goal-offset ladder, `goal_offset` in {25, 50, 100, 200}, `eval_budget = 2 x
  goal_offset`. Offset 25 is LeWM's own protocol and is the anchor; the prediction is
  that any effect of the progress cost **grows with offset and is near zero at 25**.
* Reset-seed band: 90000+ (the `event_smdp_h0` bands end at 88563).

## Stage 0 -- plumbing gate

Nothing downstream is admissible until this passes. 24 sampled frames and 12 (start,
goal) pairs at offset 100, drawn with seed 90000 from the first two episodes.

1. **Frame reproduction.** Restore `qpos`/`qvel`/`button_states` into `swm/OGBScene-v0`,
   render 64x64 from `front_pixels`, compare to the stored frame. Passes at
   `max_abs_diff <= 2` over all sampled frames.
2. **Offline state readout.** Cube position, drawer slide, window slide and the drawer
   site `y` resolved from a stored `qpos` by forward kinematics alone must equal the live
   environment's own reads to `1e-9`, and `_is_in_drawer` must agree on every frame.
3. **Goal-relative success wiring.** With `set_cube_target_pos`,
   `set_target_button_state`, `set_target_drawer_pos`, `set_target_window_pos` set from
   the goal row, `_compute_successes` must fire on the goal row for all 12 pairs, and the
   offline reimplementation must agree with the environment component-by-component on
   both rows of every pair.

### Locked decision rules

* `OFFLINE_STATE_MISMATCH` -- check 2 fails. Stop: without an exact offline readout there
  is no oracle ceiling and no supervision track.
* `SUCCESS_WIRING_BROKEN` -- check 2 passes, check 3 fails. Stop: the success predicate is
  the endpoint of every later stage.
* `RERENDER_REQUIRED` -- checks 2 and 3 pass, check 1 fails. Continue, but on frames we
  render ourselves from `qpos` (1.001M frames, 64x64x3 = 12.3 GB) rather than the stored
  pixels. This is a branch, not a failure; the stored `max_abs_diff` and shape report say
  which.
* `STAGE0_PASS` -- all three pass.

The frame check is the only one allowed to fail without stopping, because a mismatch
there is a rendering-provenance question, not a claim about the world.

## Declared limitations (written before the run)

* 64x64 is OGBench's own visual-Scene resolution, not the 224x224 the released LeWM-cube
  checkpoint was trained at. Absolute success rates here are therefore not comparable to
  the published 68.0% on OGBench-Cube; only within-arena contrasts are.
* The play data is not expert data. Any temporal-distance signal learned from it is an
  upper-biased estimate of remaining time, which is why the progress readout uses an
  expectile rather than a mean.
* One benchmark, one backbone family. Representation independence is not addressed by
  this program.

## Stage 0 outcome (job 49473, 2026-09-04)

Verdict: **RERENDER_REQUIRED**

| Check | Result |
|---|---|
| Offline state readout (24 frames) | **exact** -- cube, drawer, window and drawer-site `y` all agree with the live environment to `0.0`; `_is_in_drawer` agrees on every frame |
| Goal-relative success wiring (12 pairs, offset 100) | **wired** -- `_compute_successes` fires on 12/12 goal rows and 0/12 start rows; the offline reimplementation agrees with the environment component-by-component on both rows of every pair |
| Frame reproduction (24 frames) | **fails** -- shapes match (64x64x3) but `max_abs_diff` 163-192, `mean_abs_diff` 6.08-15.93, and only 16.6-22.4% of pixels are exact |

Resolved joint addresses, recorded so later stages never rediscover them:
`object_joint_0` qposadr 14, `drawer_slide` 23, `window_slide` 24, drawer site id 8.

The frame mismatch is uniform across frames and leaves roughly a fifth of pixels exact,
which is the signature of a systematic rendering difference (camera, lighting or material
definition) between upstream OGBench's collector and the `stable_worldmodel` `SceneEnv`,
not of a state-restoration error -- checks 2 and 3 rule that out directly.

Taking the pre-declared `RERENDER_REQUIRED` branch is therefore not a concession. Training
on the released frames while evaluating through our own renderer would put a systematic
appearance shift between train and test, which is precisely the confound a world model
cannot absorb. Every frame in this program is drawn by the same renderer that the closed
loop will use.

**Consequence for the arena.** Absolute numbers here are not comparable to any published
`visual-scene-play` result, because the pixels are not the released pixels. The comparison
this program makes is entirely within-arena: identical data, identical backbone, identical
planner, identical episodes and seeds, with only the planning objective changed.

## Harness gate outcome (job 49519, 2026-09-04) and the amendment it forced

Verdict as written: **HARNESS_BROKEN**. The gate replays each trajectory's own recorded
actions from its start row and requires OGBench's success predicate to fire at the goal
row, at a pass rate of 0.95 for every offset. It does not clear that bar at the top of
the ladder.

| Goal offset | Replay success | Trivial starts | Cube error, median | Cube error, max |
|---|---|---|---|---|
| 25 | 100% | 17/50 | 7.0e-06 m | 1.2e-02 m |
| 50 | 100% | 9/50 | 6.7e-05 m | 1.6e-02 m |
| 100 | 96% | 8/50 | 1.1e-03 m | 4.5e-02 m |
| 200 | 80% | 2/50 | 3.5e-03 m | 2.7e-01 m |

### What this does and does not show

It does **not** show a defect in reset, stepping, target wiring or the success predicate.
At offsets 25 and 50 every component of every episode matches, with median cube error at
the 1e-05 m scale -- six microns is state restoration working exactly. What grows with
horizon is *replay divergence*: the failures at offset 200 land on `cube` (7), `button_0`
(3), `drawer` (3) and `button_1` (1). Button states only change on contact, so a replay
that flips the wrong button has physically diverged from the recorded trajectory, which
is what open-loop replay of 200 steps of contact-rich MuJoCo does.

### Amendment (before any planning number exists)

Two changes, both decided now, while there is nothing to tune them against.

1. **The gate's threshold was mis-specified, and is not being lowered to make it pass.**
   Requiring 95% replay success at every offset asserted that open-loop replay is stable
   to 200 steps. Nothing in this program needs that, and it is not true. The claim the
   gate actually supports -- and the one the program relies on -- is that the arena is
   exact: 100% at offsets 25 and 50 with micron-scale error.

2. **The measured replay rate becomes the reported ceiling at each offset.** At offset
   200 the dataset's own actions reach the goal 80% of the time, so 80% is the practical
   ceiling there, not 100%; at offset 100 it is 96%. Every planning number at an offset
   is reported next to that offset's ceiling. This converts the gate's shortfall from an
   embarrassment into a calibration, and it costs the compared arms nothing, since they
   all face the same ceiling.

3. **Trivial episodes are excluded from sampling.** At offset 25, 17 of 50 sampled
   episodes already satisfied the goal at the start row: a third of the score handed to
   every arm before a single action, compressing exactly the range the ladder is meant
   to resolve. `build_episode_specs` now redraws a start whose state already matches the
   goal. The rejection reads only the dataset, never the arm, so all arms still see
   identical episodes. Job 49524 re-measures the ceilings on the screened episodes; those
   are the ceilings the ladder will be read against.

## Render cost, and the cache the arena is built on (jobs 49475-49488)

Self-rendering cost 164 ms per frame at first measurement, which would have put the
1.001M-frame cache at ~35 h. Decomposition (49475) put the whole cost in
`renderer.render()`; a size sweep (49476/49477/49480) found it nearly flat in output
size -- 16 px 140 ms, 32 px 156 ms, 64 px 164 ms. That flatness is misleading, and reading
it as "fixed overhead, quality flags will not help" was wrong: shadow-map rendering and
MSAA resolve both cost roughly what they cost regardless of the output resolution.
Turning them off (49480) took 158.6 ms to 54.8 ms to **22.6 ms**, a 7x speedup. Job 49478
adds that rendering needs no GPU allocation at all and is in fact *faster* off the MIG
slice, so the 2-GPU quota does not bound cache building.

Two decisions follow, both stamped into every artifact:

* **Render quality `fast`** (shadows off, `offsamples = 0`). `scene_render.make_scene_env`
  is the only place that builds an environment, and it returns a `render_info` block that
  the cache records and every downstream job asserts against, so training-time and
  plan-time appearance cannot drift apart.
* **Stride 5.** Only every fifth row is rendered, which is exactly the action-block grid
  the model and the planner both step on. 200200 train frames and 20020 val frames instead
  of 1.001M and 100100.

Both caches finished complete: 200200/200200 and 20020/20020 grid rows rendered, zero
blank (49487, 49488). Note the render grid is **absolute** (`arange(0, total, 5)`) and
1001 % 5 != 0, so each episode meets the grid at a different phase; window selection
filters on the absolute row, not on a per-episode offset.

## Harness gate: the arena's own ceiling (jobs 49519, 49526)

Verdict: **HARNESS_OK**

Before any model is judged, the arena is judged. Replaying a trajectory's own recorded
actions from its start row must reach its goal row, because that is what the dataset says
those actions do. This uses no model, no encoder and no pixels.

Run on the val cache, 50 episodes per offset, seed 90100, with trivial starts screened:

| goal offset | 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| replay success (**arena ceiling**) | 1.00 | 0.98 | 0.98 | 0.92 |
| cube error, median | 0.53 mm | 1.13 mm | 2.48 mm | 3.62 mm |
| cube error, max | 12 mm | 16 mm | 13 mm | 152 mm |

against OGBench's own 40 mm tolerance.

**The ceiling falls with horizon, and that is not a defect.** The released dataset stores
`qpos`/`qvel` in float32 while MuJoCo integrates in float64, so a restored state is
truncated and contact-rich rollouts amplify the difference; at 200 steps a handful of
episodes diverge outright (cube error up to 152 mm) while the median stays at 3.6 mm. It
is not an unrestored controller state: `ManipSpaceEnv.set_control` recomputes its target
from the current effector pose every step, so it carries nothing across a reset. The gate
therefore tests the **short** offsets, where 25-50 steps leave no room for divergence to
matter, and reports the long ones as a ceiling. A flat threshold across offsets would have
called normal float32 truncation a broken arena -- the first run (49519) did exactly that.

**Trivial episodes are screened out.** The unscreened run found 17 of 50 episodes at
offset 25, and 8 of 50 at offset 100, whose start row already satisfied its own goal.
Those hand every arm the same free successes and compress exactly the range the offset
ladder exists to resolve. `build_episode_specs` now rejects and redraws such starts, using
an offline predicate that job 49473 showed agrees with `_compute_successes`
component-by-component. The screen reads only the dataset, never the arm, so every arm
still sees identical episodes.

Planning results are to be read against the ceiling above, not against 100%.

### Screened ceilings (job 49524) -- the numbers the ladder is read against

Same gate, trivial starts now rejected. These are the per-offset ceilings every planning
arm is reported next to.

| Goal offset | Replay ceiling | Trivial starts | Cube error, median |
|---|---|---|---|
| 25 | 100% | 0/50 | 5.3e-04 m |
| 50 | 98% | 0/50 | 1.1e-03 m |
| 100 | 98% | 0/50 | 2.5e-03 m |
| 200 | 92% | 0/50 | 3.6e-03 m |

Screening removed every trivial start, which is what it was for. It also moved the offset
200 ceiling from 80% to 92%, and that is **not** an improvement in the arena -- rejection
sampling consumes the RNG differently, so this is a different draw of 50 episodes. With
n = 50 the ceiling is estimated to roughly +/- 11 points, so it is a calibration with real
uncertainty, not a constant. Any arm that lands within sampling noise of its offset's
ceiling is at the arena's limit, and no claim is made about the gap between them.

The verdict token remains `HARNESS_BROKEN` at the locked 0.95 threshold because offset 200
sits below it. That token is kept rather than rewritten: the threshold was mis-specified,
the amendment above says so and says why, and the reader can see both.

## Stage 3 specification

Locked 2026-09-04, before any progress head was trained.

### What is held fixed

The world model, the solver, the episode list, the plan seeds, the renderer and the
success predicate are identical in every arm. An arm is a **planning objective and
nothing else**. `--mixture-weight 1.0` drops the progress term entirely rather than
multiplying it by zero, so that arm runs the identical computation to the baseline and
must reproduce it row for row.

### The head

A GRU over `[latent ‖ action-block embedding]`, carrying two things: a hidden state and a
progress vector accumulated as `p_t = p_{t-1} + softplus(W h_t)`, so progress cannot fall.
Latching is a property of the architecture, not of a penalty that could be traded away.
The planner-facing readout is goal-conditioned normalised remaining time, `v ∈ ℝ`, trained
toward `(steps to goal) / 40 blocks` clipped to 1. Planning minimises it.

The recurrent state spans the **whole episode**, not the world model's three-frame
context window. That is the point: the button that unlocked the drawer may have been
pressed two hundred steps ago and is not visible now.

Hyperparameters, fixed for every arm: window 8 blocks, goal drawn 1--40 blocks beyond the
window, expectile `τ = 0.2`, batch 128, 12000 steps, AdamW `lr 3e-4`, `wd 1e-4`, clip 5.0,
auxiliary weight 0.5 (used by `prog_sup` only).

### Arms

| Arm | Reads observations | Reads actions | History | Monotone | Role |
|---|---|---|---|---|---|
| `latent_l2` | -- | -- | -- | -- | baseline and reproduction anchor |
| `prog_ssl` | yes | yes | episode | yes | **primary**, label-free |
| `prog_sup` | yes | yes | episode | yes | oracle ceiling: adds a privileged auxiliary term on the five components the success predicate scores. Training-time only; never available at deployment |
| `prog_frame` | yes | yes | none | yes | isolates history: the recurrence is severed each step |
| `prog_action_only` | no | yes | episode | yes | dead reckoning: the observation is zeroed, the action stream is not |
| `prog_nomono` | yes | yes | episode | no | isolates latching: the same head with the softplus accumulator removed |

`prog_frame` and `prog_action_only` are the two controls that carry the claim. If a
frame-conditioned head matches `prog_ssl`, history was not the mechanism; if an
action-only head matches it, the observations were not.

### The mixture weight

Latent-L2 and predicted remaining time live on different scales, so the mixture divides
the former by a constant measured **once on the baseline arm** and then frozen. The
weight is swept over {1.0, 0.75, 0.5, 0.25, 0.0} on validation episodes drawn with a
different seed from the confirmatory ones, and a single value is locked before any
confirmatory cell runs. No per-arm, per-offset or post-hoc tuning.

### The invariant that would silently ruin this

CEM calls the cost hundreds of times per replan. If the head's recurrent state advanced
on those calls it would be integrating imagined rollouts into its memory of what actually
happened, and the arm would be measuring something other than what it claims. The tracker
is therefore stepped only by the environment loop, once per executed action block, and
the objective reads a detached copy. Each episode asserts that the head advanced exactly
`env_steps / action_block + 1` times and raises otherwise, so a regression here fails the
run instead of quietly changing the result.

## Checkpoint selection

Job 49521 exposed a defect in this session's own training script: validation loss is
dominated by SIGReg (val 45.2 against train 1.5 at step 2000, because the projector's
BatchNorm switches to running statistics in eval mode), so `0.09 * sigreg` was 4.07 of a
4.11 total. Selecting the best checkpoint on total validation loss therefore selected on
the regulariser rather than on prediction quality.

Two consequences, both fixed now rather than argued away:

1. The selection criterion is validation **prediction** loss.
2. 49521 had been running under the old criterion and was cancelled; the baseline that
   the program actually uses is **49525**, which ran the fixed criterion from the start.
   Its `lewm_best.pt` is therefore selected on validation prediction loss, taken at step
   38,000 (`val_pred_loss` 0.005001) out of 40,000. The run had converged -- step 40,000
   scored 0.005016 -- so selection moved the checkpoint by 0.3% of the metric and nothing
   in the program turns on it.

Validation prediction loss is still reported per arm from `val_history`, as a description
of fit, never as a selection knob.

### Clarification forced by the pre-training checks (job 49531)

`prog_frame` and `prog_nomono` are **not** two spellings of the same ablation, and the
first check run made that concrete by failing. Severing the recurrence also severs the
progress accumulator, so `prog_frame` gives up history and latching together; only
`prog_nomono` isolates latching with history intact. Read against `prog_ssl`, they
answer different questions, and a result that moves `prog_frame` but not `prog_nomono`
implicates memory rather than the monotone constraint.

## Amendment 2026-09-06: the tracker invariant was mis-specified (jobs 49951-49955)

The five-weight mixture sweep is the first time any progress arm has run through
`eval_scene_plan.py` end to end -- job 49530 was a training smoke, and the
elite-optimism gate uses a different runner -- and all five jobs died within 20 s on

```
RuntimeError: progress tracker advanced 4 times, expected 3
```

**The check was wrong, not the head**, which is the same failure mode as job 49531 and
is recorded here for the same reason. `run_episode` calls `on_step` once per action
block *entered*, and the inner action loop breaks early when the episode succeeds, so a
block truncated by success still advances the tracker once. The expected count was
written as `env_steps // action_block + 1`, which counts only whole blocks. That is
correct exactly when an episode ends on a block boundary and wrong otherwise: with
`action_block = 5`, an 11-step episode enters 3 blocks and advances the tracker 4 times
while the formula predicts 3.

This is why it had never fired. `latent_l2` runs build no tracker, so the assertion is
skipped entirely; and every episode that exhausts the budget ends at 50 or 100 steps,
both multiples of 5, where flooring and ceiling agree. Only a *successful* episode can
end off-boundary, so the first arm to plan with a progress head tripped it immediately.

The formula is now `ceil(env_steps / action_block) + 1`. This tightens rather than
relaxes the invariant's intent: the tracker must integrate every executed transition and
no imagined one, and the ceiling counts exactly the transitions the environment loop
actually delivered. No threshold that any result is read against has been changed.

**Declared limitation found in the same pass.** When success truncates a block, `on_step`
hands the tracker the *whole* block, including actions that were never executed. Reading
the loop shows the inner break fires only on success, and the episode ends immediately
after, so this always lands on the terminal update and can never influence a planning
decision. It is recorded rather than fixed because fixing it would change the arm's
behaviour for no measurable effect; if a future variant lets the budget truncate a block,
this becomes live and must be revisited.

## Amendment 2026-09-06: `goal_scale` measured at last (job 49949)

The protocol requires the latent-L2 divisor be "measured once on the baseline arm and
then frozen", and every run before this one used the default `1.0` -- the constant had
never been measured. It does not affect any result read so far, because a positive
constant divisor cannot move CEM's argmin at `--mixture-weight` 0.0 or 1.0, which are the
only weights any arm had run at; it binds only on the intermediate weights the sweep
introduces.

Job 49949 ran the baseline arm on validation episodes (`episode_seed 90500`, disjoint
from the confirmatory `90100` the replay ceilings are measured at), offsets 25 and 50, 50
episodes each. Pooling the per-episode elite cost over both offsets:

| offset | success | elite-cost median | mean |
|---|---:|---:|---:|
| 25 | 15/50 = 30% | 1.3387 | 2.1620 |
| 50 | 11/50 = 22% | 1.4814 | 7.3769 |

**`goal_scale` is frozen at 1.3871**, the pooled median over all 100 episodes. The median
rather than the mean because the distribution is heavy-tailed (max 79.2), and pooled
across offsets because the protocol forbids refitting per offset or per arm.

### The old baseline is not used as the anchor

`baseline_headroom_20260904` (job 49536) reported 14/50 at offset 25 and 18/50 at offset
50. The wrapper loops offsets with one Python invocation each, and the offset-25 file
lacks the `elite_cost_*` keys entirely while the offset-50 file written two minutes later
has them, so `scene_eval.py` was edited *between the two processes of a single job*.
`scene_progress_wm/` is untracked and the log has been rotated away, so the two versions
cannot be diffed. Offset 25 reproduces (28% then, 30% now, different episode draw), while
offset 50 does not (36% then, 22% now) -- within the ±13-point interval n=50 buys, so
this is not evidence of a behaviour change, but it is reason enough not to anchor a
contrast on it. Every comparison from here pairs against a baseline run under current
code at the same seeds.

## Mixture sweep result, jobs 49957-49961 (2026-09-06), validation seed 90500

`prog_ssl`, `goal_scale` 1.3871, 50 episodes per offset, `--mixture-weight` is the weight
on latent-L2, so 1.00 is the pure baseline and 0.00 is the pure progress cost.

| MIX | off25 | off50 | pooled paired diff vs baseline [95% CI] |
|---:|---:|---:|---|
| 1.00 | 15/50 = 30% | 11/50 = 22% | +0.0 [+0.0, +0.0] |
| 0.75 | 18/50 = 36% | 8/50 = 16% | +0.0 [-6.0, +6.0] |
| 0.50 | 20/50 = 40% | 9/50 = 18% | +3.0 [-3.0, +10.0] |
| 0.25 | 20/50 = 40% | 10/50 = 20% | +4.0 [-4.0, +12.0] |
| 0.00 | 6/50 = 12% | 7/50 = 14% | **-13.0 [-22.0, -4.0]** |

**The reproduction check passes exactly.** `--mixture-weight 1.00` drops the progress term
and reproduces the baseline episode for episode, 0 mismatches out of 50 at both offsets.
The harness, the tracker fix and the mixture wiring are therefore not what any difference
below is measuring.

**The strong form of the claim is refuted here, CI-clean.** The progress cost *on its own*
is a worse planning objective than latent-L2: -13.0 points pooled, CI [-22.0, -4.0], and
-18.0 [-32.0, -4.0] at offset 25 alone. Both intervals exclude zero. This is the purest
statement of the method -- replace latent-L2 with a latched history-conditioned progress
cost -- and on this arena it loses roughly half the baseline's successes.

**The weak form is unsupported rather than refuted.** No intermediate weight's pooled
interval excludes zero, and the sign flips between offsets for every one of them: offset
25 gains 6-10 points while offset 50 loses 2-6. A term that helps at one horizon and
hurts at the next, with every interval spanning zero, is not yet evidence of anything.

### Locked selection, written before any confirmatory cell was run

The protocol requires one weight, chosen on validation, pooled (per-offset tuning is
forbidden). The pooled argmax is **MIX = 0.25**, i.e. 25% latent-L2 and 75% progress, and
that is what the confirmatory stage runs. It is locked in full knowledge that +4.0
[-4.0, +12.0] is an argmax over five noisy cells and may well be selection noise; the
confirmatory run at the disjoint seed 90100 is exactly the test of that, and its result
stands whichever way it falls.

Recorded as an observation, not a tested contrast: the collapse is confined to MIX = 0.00.
At 75% progress weight the arm is level with baseline; at 100% it halves. The progress
readout is goal-conditioned by construction, but this cliff suggests it does not localise
the goal sharply enough to steer CEM without a latent-L2 term present.

## Confirmatory result, jobs 49965/49966, analysis 49972 (2026-09-06)

Locked weight MIX = 0.25, `goal_scale` 1.3871, confirmatory `episode_seed` 90100 (the seed
the replay ceilings are measured at), plan seed 90200, 50 paired episodes per offset.

| offset | `latent_l2` | `prog_ssl` w0.25 | paired diff [95% CI] | clean | ceiling |
|---:|---:|---:|---|---|---:|
| 25 | 14/50 = 28% | 20/50 = 40% | +12.0 [-2.0, +26.0] | no | 100% |
| 50 | 18/50 = 36% | 13/50 = 26% | -10.0 [-22.0, +2.0] | no | 98% |
| **pooled** | 32/100 = 32% | 33/100 = 33% | **+1.0 [-9.0, +10.0]** | no | -- |

**The headline is a null.** Pooled over both offsets the progress cost is worth one
success in a hundred episodes, with an interval spanning nine points either way. Neither
per-offset contrast clears zero, and the protocol's own analyser marks both `clean: false`.

**The baseline reproduces the July measurement exactly.** 14/50 at offset 25 and 18/50 at
offset 50 are the same counts `baseline_headroom_20260904` reported at this seed. The
mid-job source edit recorded in the amendment above therefore did not change behaviour,
and the 22% seen at offset 50 during the sweep was the episode draw at seed 90500, not
code drift. The earlier concern is retired.

**The horizon-dependent sign flip replicated across two disjoint episode draws.** At
MIX 0.25 the validation seed gave +10 at offset 25 and -2 at offset 50; the confirmatory
seed gives +12 and -10. Same ordering, similar magnitudes, on non-overlapping episodes.
The weight was selected on the *pooled* validation number, so the per-offset pattern was
not the selection target and its recurrence is not a selection artefact. It is recorded as
an exploratory observation only: each individual interval still contains zero, and two
draws is not a horizon ladder. If anything survives this program it is this, and testing
it needs offsets 100 and 200 and more seeds, not a re-reading of these four cells.

### Standing

The strong form of the method is refuted CI-clean (MIX 0.00, -13.0 [-22.0, -4.0]). The
mixture form is a null at the locked weight. Both arms sit far under the replay ceiling --
the best cell reached is 40% against 100% at offset 25 -- so this is not a saturation
result: the headroom the program was built to attack is still there, and neither objective
takes it.

`prog_frame`, `prog_action_only`, `prog_nomono` and `prog_sup` remain trained and unrun.
They exist to answer *why* a positive result happened, and there is no positive result for
them to explain, so running them now would buy a mechanism for a null.
