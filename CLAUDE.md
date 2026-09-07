# CLAUDE.md

## What this repo is

A single research programme on **JEPA-style latent world models for robot planning**:
an encoder maps images to a latent, an action-conditioned predictor rolls that latent
forward, a cost scores the imagined outcome, and a sampling planner (CEM and friends)
picks actions. The question the whole repo circles is:

> When this stack fails at contact-rich manipulation, *which part* is actually broken?

It is a falsification-first repo. Most directories are **pilots that were designed to
kill an idea cheaply**, and most of them succeeded at exactly that. The honest summary
of the programme so far is a well-localized negative result plus one live positive lead.
Treat the negatives as load-bearing knowledge, not as failed work to be quietly retried.

## The through-line (read this before proposing anything)

1. **The original hypothesis was action-blindness.** Released action-conditioned world
   models seemed to predict nearly the same future for very different actions, worst
   right at the moment of contact. The diagnostic confirmed something real there, and
   confirmed it does not go away with model scale.

2. **The oracle ladder relocated the problem.** Replacing the learned predictor with the
   simulator itself — perfect dynamics, same encoder, same planner, same budget — did
   *not* rescue contact tasks, while a physically-grounded reference cost solved them
   easily. So the wall is not prediction. It is **the cost the planner descends**, i.e.
   the representation used to say "this imagined outcome is closer to the goal".

3. **The mechanism is optimizer-conditioned misranking (Goodhart / reward hacking).**
   A cost can be accurate on trajectories drawn from data and still be badly wrong on
   exactly the candidates a strong search invents. Search is an adversary against its own
   objective: it *finds* the pockets where the cost is mistaken. Agreement between the
   proxy cost and physical outcome is weak on the initial proposal population and gets
   *worse* after the planner refits toward its own optimum.

4. **Everything tried on the cost's inputs has failed to fix that.** Better grounding,
   better predictors, better latent geometry, extra sensing, extra information — each was
   measurable and real as a metric improvement, and none of them converted into planning
   success. "Grounding up does not imply planning up" is the most reproduced result here.

5. **The live lead changes what the cost *measures*, not how accurately it measures a
   goal image.** Task progress in multi-stage manipulation is **latched**: milestones are
   irreversible, so a single frame cannot express how far along you are, and a
   distance-to-goal-image cost is structurally the wrong quantity. That is the current
   programme (`scene_progress_wm/`).

## Repo map (each directory is one programme, with its verdict)

Closed / negative:

- `diagnosis/` — the original CAI-JEPA diagnostic plus the oracle ladder and all
  post-hoc cost interventions. Largest and most-cited directory; source of items 1–3
  above. Its `docs/CURRENT_STATUS.md` and `docs/CLAIMS_EVIDENCE.md` are the claim
  discipline for the paper.
- `contactworld_h0/` — does tactile sensing add the missing object state? **No.**
- `hys_h0/` — contact-aware gating of a temporal-straightening loss. **Refuted** in both
  frozen and fine-tuned forms; a matched *random* gating control did as well or better.
- `action_curvature_h0/` — action-space curvature mismatch. Real diagnostic, failed
  intervention (see below). `TECHNICAL_NOTE.md` and the root `AUGUST_2026_TECHNICAL_NOTE.md`
  are the readable write-ups.
- `counterfactual_flow/`, `crod_h0/`, `gfpr_h0/`, `physical_search_distillation/`,
  `rollout_repair_gate/` — the "use physical supervision / disagreement / reranking to
  repair the planner's choice" family. All four locked verdicts are STOP.
- `moment_wm_h0/` — hidden-physics (episode drag) context world model with a
  conditional-moment regularizer. **STOP** at the final gate; see its `DECISION_REPORT.md`,
  which is a model of how to write one.
- `belief_compression/` — decision-equivalent belief compression. Paused: its own novelty
  gate found the bound is likely a corollary of published work.
- `event_smdp_h0/` — the simulator-as-world-model gate that established the **latching**
  finding and the observer/feedback lessons below. Mostly positive, but privileged.

Live / active:

- `scene_progress_wm/` — the successor to `event_smdp_h0`, with the privilege removed:
  a learned world model, pixels plus action chunks, CEM, OGBench's own success predicate.
  Asks whether a latched, history-conditioned **progress** cost beats latent-L2 to a goal
  image. This is where new work should go unless told otherwise.

Writing:

- `paper/main.tex` — the paper of record (TMLR), a mechanistic audit of terminal-cost
  misranking, not a new method. `proposal/` is historical.
- `refine_jepa_h0/` is an empty shell; ignore it.

## What was tried, and why each thing failed (intuitive)

**Cost side, frozen encoder.** Latent distance to a goal image is not a measure of task
progress. It works where the task is "move the arm somewhere" and collapses where the
task is "move an object", because the object occupies a tiny, badly-conditioned part of a
representation trained for prediction, not for control.

**Decode the state from the latent and plan on that.** Any post-hoc readout has residual
error, and the planner spends its whole budget hunting for that error. Making the readout
robust off-policy fixed the readout and not the planning — proof that the failure is
exploitation of residual error, not missing information.

**Relearn a grounded adapter.** Grounding became excellent; planning did not move. This is
the cleanest statement of the recurring lesson.

**Ensemble / disagreement penalty** (the textbook fix for model exploitation). Failed
because every ensemble member sits on the same frozen backbone, so they share a blind
spot: they *agree* precisely in the pockets where they are all wrong, so disagreement is
flat exactly where a penalty was needed.

**Fine-tune the encoder (LoRA).** No crossing. One seed looked like a breakthrough and did
not replicate across the sweep — a reminder that single-seed wins in this repo are noise
until a sweep says otherwise.

**Fix the predictor with a counterfactual objective.** The one clear positive: the model
gets much better at telling apart what different actions do, on offline ranking metrics
and on real-robot action metrics. It still did not deliver closed-loop contact success,
which is consistent with the oracle ladder — the predictor was never the binding constraint.

**Straighten the action-to-outcome geometry.** Angular curvature of the model's
action→outcome map strongly predicts *false valleys*: minima the model believes in that
the simulator does not. Multi-step training reduces both curvature and false valleys, and
changes planning by nothing. Worse, the metric is gameable — a flatter map deletes true
minima along with false ones, so "fewer false valleys" is not automatically progress.

**Gate straightening on contact.** The physical premise held (motion really does bend more
at contact transitions), the mechanism did not: a control that dropped the same *number* of
terms at random did as well. When your idea does not beat matched randomness, the
information you were proud of extracting was not what was doing the work.

**Rerank, distill, or acquire with physical supervision.** Scoring a frozen planner's final
population with a learned physical-regret scorer, distilling the physical elite-refit
operator into a zero-query cost, and using cross-representation disagreement to pick what
to verify — all stopped, each against a cheap matched baseline (usually just proposing more
diverse actions). Physical-outcome oracles *do* help a lot, which pins the gap on
recovering that signal without querying physics.

**Remove the search.** Amortizing the controller instead of planning does not cross the
wall either, so it is not simply that CEM is too strong.

**Add the hidden variable.** In the hidden-drag study, the hidden physics was easy to
recover from frozen features and recovering it did not lower true planned cost: the same
proxy-versus-truth gap under optimizer selection, in a completely different arena.

**Add memory / belief machinery.** The scene-aliasing premise was refuted in minutes — a
single frozen latent already probes the latched scene variables nearly perfectly, and
recurrence adds nothing. Belief compression stalled on prior art. Run the cheap probe
before building the machinery.

## Compute policy (mandatory)

This checkout is on a Slurm **login node**. Never run MuJoCo, rendering, model loading,
training, encoding, or bulk result scans here — `sbatch` them to a compute node, including
CPU-only analysis (several programmes enforce this in code by raising unless
`SLURM_JOB_ID` is set). Login-node work is `rg`/`sed`/`git`, small metadata reads, syntax
checks, and `squeue`/`sacct`. Verify claimed job state with **both** `squeue` and `sacct`
before acting on it — peer sessions submit into the same queue, so check for duplicate
work before launching a long array.
