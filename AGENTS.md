# AGENTS.md

## Compute policy (mandatory)

This checkout is on a Slurm **login node**. Never run MuJoCo, rendering, model loading,
training, encoding, or bulk result scans here — `sbatch` them to a compute node, including
CPU-only analysis (several programmes enforce this in code by raising unless
`SLURM_JOB_ID` is set). Login-node work is `rg`/`sed`/`git`, small metadata reads,
`python -m py_compile`, and `squeue`/`sacct`. Verify claimed job state with **both**
`squeue` and `sacct` before acting on it — peer sessions submit into the same queue, so
check for duplicate work before launching a long array, and never overwrite another
session's dirty files.

## What the research is about

**JEPA-style latent world models for robot planning.** An encoder maps images to a latent,
an action-conditioned predictor rolls it forward, a cost scores the imagined outcome, and
a sampling planner (usually CEM) selects actions. The repo exists to answer one question:
when this stack fails on contact-rich manipulation, which component is actually broken?

The programme is falsification-first. Most directories are pilots built to kill an idea
cheaply, and most of them did. The current state is a well-localized negative result plus
one live positive lead.

## Established arc

1. The starting hypothesis was **action-blindness**: released world models predict nearly
   the same future for different actions, worst around contact. That effect is real and
   does not disappear with model scale.
2. The **oracle ladder relocated the problem**. Give the planner perfect dynamics (the
   simulator itself) with the same encoder, planner and budget, and contact tasks still
   fail — while a physically grounded reference cost solves them. The wall is the **cost**
   the planner descends, not the prediction.
3. The mechanism is **optimizer-conditioned misranking**: a cost accurate on data can be
   badly wrong on the candidates a strong search invents. Search is an adversary against
   its own objective, and proxy-versus-physical agreement degrades further after the
   planner refits toward its optimum. This is Goodhart, not a missing-information problem.
4. **Every attempt to fix the cost's inputs has failed to fix planning.** Better grounding,
   better predictors, better geometry, extra sensing — all improved their own metric and
   none improved control.
5. The **live lead changes what the cost measures**. Progress in multi-stage manipulation
   is *latched* (milestones are irreversible), so a single frame cannot express it and
   distance-to-a-goal-image is structurally the wrong quantity. That is `scene_progress_wm/`.

## Directory map and verdicts

| Directory | Question | Outcome |
|---|---|---|
| `diagnosis/` | the original diagnostic, oracle ladder, all post-hoc cost fixes | source of the arc above; claim discipline in `docs/CURRENT_STATUS.md`, `docs/CLAIMS_EVIDENCE.md` |
| `contactworld_h0/` | does tactile add the missing object state? | no |
| `hys_h0/` | gate a straightening loss on contact | refuted; matched random gating does as well |
| `action_curvature_h0/` | is the action→outcome map distorted? | real diagnostic, failed intervention |
| `counterfactual_flow/`, `crod_h0/`, `gfpr_h0/`, `physical_search_distillation/`, `rollout_repair_gate/` | repair the planner's choice with physical supervision, disagreement, reranking or rollout-matched training | all STOP |
| `moment_wm_h0/` | recover hidden physics with a moment regularizer | STOP; see `docs/DECISION_REPORT.md` |
| `belief_compression/` | decision-equivalent belief compression | paused on prior art |
| `event_smdp_h0/` | does backing up intermediate event state beat terminal success? | established the latching and observer findings, under privileged dynamics |
| `scene_progress_wm/` | latched, history-conditioned progress cost vs latent-L2, with a *learned* world model | **active** |
| `paper/` | TMLR paper of record: a mechanistic audit of terminal-cost misranking | writing |

`proposal/` is historical. `refine_jepa_h0/` is an empty shell.

## Why each direction failed (intuitive, no numbers)

- **Latent distance to a goal image** is not task progress. It works when the task is
  moving the arm and collapses when the task is moving an object, because the object is a
  small, badly-conditioned part of a representation trained for prediction, not control.
- **Decoding state from the latent and planning on it** fails because any readout has
  residual error and the planner spends its budget finding it. Hardening the readout fixed
  the readout, not the planning — so this is exploitation, not missing information.
- **Relearning a grounded adapter** produced excellent grounding and unchanged planning.
- **Ensembles / disagreement penalties**, the standard fix for model exploitation, fail
  because members share a frozen backbone: they agree precisely where they are all wrong,
  so disagreement is flat exactly where the penalty was needed.
- **Encoder fine-tuning (LoRA)** did not cross the wall; the one promising seed did not
  replicate.
- **A counterfactual predictor objective** is the clearest positive — the model gets much
  better at distinguishing what different actions do — but it does not deliver closed-loop
  contact success, exactly as the oracle ladder predicts.
- **Straightening the action→outcome geometry**: curvature genuinely predicts false valleys
  (minima the model believes and the simulator denies), and reducing it changes planning by
  nothing. The metric is also gameable, since flattening deletes true minima along with
  false ones.
- **Contact-gating that straightening**: the physical premise held, the mechanism did not —
  dropping the same number of terms at random did as well. Losing to matched randomness
  means the information you extracted was not what was doing the work.
- **Reranking / distilling / acquiring with physical supervision**: all stopped against
  cheap matched controls, usually just proposing more diverse actions. Physical-outcome
  oracles help a great deal, which pins the gap on getting that signal without querying
  physics.
- **Removing the search** (amortized control) does not cross the wall either, so the
  problem is not merely that CEM is too strong.
- **Supplying the hidden variable**: in a hidden-physics arena the latent variable was easy
  to recover and recovering it did not lower true planned cost — the same proxy-versus-truth
  gap under optimizer selection, in a different setting.
- **Adding memory/belief machinery**: the scene-aliasing premise was refuted almost
  immediately, because a single frozen latent already probes the latched variables and
  recurrence adds nothing. Run the cheap probe before building the machinery.
