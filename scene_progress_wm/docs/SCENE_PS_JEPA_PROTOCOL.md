# Predictive-state JEPA: arms, checks, and what the training metric can and cannot say

Date locked: 2026-09-05 UTC, written after the four-arm run because the run was a
plumbing check, not a contrast. The discriminating protocol is the downstream one
below, and its bars are locked here before it is executed.

## What was built

`H_phi` filters observation latents into a belief; `P_theta` rolls that belief
forward one action block at a time under a candidate chunk; the target is the EMA
filter's belief over the *real* future history, so two visually similar futures
reached through different pasts can carry different targets. Four arms, identical
architecture, 877440 trainable parameters each.

Seven pre-training checks pass (job 49582), including the three that hold the
prose to the code: candidate actions move the prediction in `ps_jepa` and not in
`ps_noaction_pred`; a different past moves the belief in `ps_jepa` and not in
`ps_frame`; the action *prefix* is ignored by `ps_jepa` and read by
`ps_action_history`.

## Four-arm result, jobs 49595-49598, seed 0, 12000 steps

| arm | pred | normalised error | effective rank | k=1 -> k=5 |
|---|---:|---:|---:|---|
| `ps_jepa` | 0.0558 | 0.300 | 18.4 | 0.022 -> 0.086 |
| `ps_frame` | 0.0629 | 0.296 | 8.1 | 0.029 -> 0.090 |
| `ps_action_history` | 0.0568 | 0.263 | 28.2 | 0.023 -> 0.086 |
| `ps_noaction_pred` | 0.2376 | 0.614 | 30.0 | 0.084 -> 0.385 |

**The control is load-bearing and passes.** Removing the candidate chunk from the
predictor costs 4.26x in prediction error and 2.04x normalised. The
action-conditioned premise is not decorative: without it the model cannot tell
candidates apart, and the metric says so loudly.

**Nothing collapsed.** Every arm sits well under a normalised error of 1, which is
what a constant predictor would score, and per-horizon error grows 3.1-4.6x from
one block to five, so the rollout is being used rather than short-circuited.

**Effective rank is a collapse detector, not a quality score.**
`ps_noaction_pred` has the highest rank (30.0) and the worst prediction. Rank is
read only to catch a degenerate belief, never to rank arms.

## Two things the metric cannot decide, and one it must not be used for

**History versus frame is a null here, and that is expected.** Normalised error
differs by 1%. Raw prediction error favours `ps_jepa` by 13%, but that is the
belief-variance artefact: each arm predicts *its own* EMA target, so an arm whose
belief occupies less space has an easier problem. Normalising by belief variance
corrects part of it and leaves nothing. A belief can be perfectly predictable and
useless; self-prediction cannot separate the two.

**`ps_action_history` must not be scored on this objective at all.** Its target is
`H_phi_bar(o_<=t+k, a_<=t+k)` -- a belief that encodes the action blocks between
`t` and `t+k` -- while the predictor is handed exactly those blocks. Part of the
target is therefore computable from the input with no world knowledge. Its 0.263
is a shortcut, not a result, and the arm needs either a downstream metric or a
target that excludes the actions it was given before it can be compared. This is
a defect in the arm's design, found before the number was used.

Together these mean the training objective has said everything it can. It
verified the plumbing and is structurally unable to test the claim.

## The discriminating protocol, locked before execution

Both stages share one frozen set of checkpoints and one candidate set, so an arm
difference is an information difference.

**Stage A -- history-swap audit.** Pairs of states whose current frame matches
closely but whose histories differ and whose optimal continuations differ. For
each pair, score a shared set of candidate chunks under each arm's belief and
compare the induced ranking against the true continuation cost, reusing the
oracle `c_true` harness from the elite-optimism gate. Primary metric: pairwise
ranking accuracy on swap pairs. `ps_jepa` must beat `ps_frame` with a
pair-bootstrap 95% CI excluding zero; if it does not, the belief carries no
planning-relevant information that the frame lacks and the central claim fails
here, whatever the rank numbers say.

**Stage B -- closed loop.** `ps_jepa`, `ps_frame`, and a recurrent LeWM at equal
capacity, same planner, same seeds, same episodes, reported against the replay
ceiling and the offset ladder.

Stage A is the cheap gate and runs first. A failure there stops Stage B: a belief
that cannot rank candidates on constructed pairs will not rank them in the loop.

## Stage 0 result, job 49618 (2026-09-05): the premise is refuted

Balanced accuracy, linear probe, split by episode, 15220 val positions:

| features | dim | button_0 | button_1 | drawer_open | window_open |
|---|---:|---:|---:|---:|---:|
| `frame_latent` | 192 | 0.9983 | 0.9979 | 0.9887 | 0.9848 |
| `ps_jepa` | 256 | 0.9980 | 0.9982 | 0.9864 | 0.9839 |
| `ps_frame` | 256 | 0.9987 | 0.9992 | 0.9884 | 0.9843 |

Matched contrast, `ps_jepa` minus `ps_frame`: -0.0007, -0.0010, -0.0020, -0.0003.

**A single frozen LeWM latent already recovers every latched variable at 98-99%.**
The buttons that lock the drawer and the window are visible in the frame. There
is no hidden persistent state on this arena for a filter to reconstruct, so the
claim that the current-frame latent is not predictively sufficient is false here,
and the recurrence has nothing to add. It adds nothing: all four gaps are
negative and within noise.

The harness check passes -- `ps_frame` lands on `frame_latent` to three decimals
-- so the belief extraction is correct and the null is a property of the arena,
not of the measurement.

### What this does and does not overturn

It does **not** contradict `event_smdp_h0`, and reading it that way would be the
easy mistake. There, conditioning on observation history was worth +21.61 points
of planning success, CI [+15.36, +28.39]. This probe says the information was
never missing from the frame. Both are true, and together they relocate the
mechanism: history did not supply *information the frame lacked*, it made the
observer *stop over-reading* -- 3 over-reads in 2408 decisions against 247 for
the frame observer trained on the wider support. That is a robustness and
error-direction effect in a learned readout, not partial observability.

So the surviving fact from the sibling program is about how a readout fails, not
about what an image contains, and any method built on "the frame is not enough on
Scene" is built on a premise this table refutes.

### Consequence

The predictive-state claim, in this form and on this arena, is closed. Stage A's
ranking test and Stage B's closed loop are not run: they ask whether a belief
that recovers nothing ranks candidates better, and the answer cannot be
interesting. The four trained arms and the audit stand as the record.

## Standing limits

`H_phi` reads frozen LeWM latents from the cache rather than pixels, which keeps
the comparison against a recurrent LeWM clean but caps the belief at whatever the
frozen encoder kept. If the aliasing lives in the encoder, no filter above it can
recover the difference and every arm fails together -- that is an answer, not a
bug, and it must be reported as one rather than as a failure of the method. One
seed so far; Stage A is to be repeated across seeds before any claim is made.
