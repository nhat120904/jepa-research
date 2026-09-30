# Overlap review — 2026-09-27

The user identified a substantial overlap with the earlier CAI-JEPA programme.
This review compares the actual implementations and saved results, rather than
the names of the methods.

| Question | Earlier CAI-JEPA | This prototype |
| --- | --- | --- |
| Core hypothesis | A world model can predict logged transitions yet respond incorrectly to alternative actions from the same state. | Same hypothesis, restricted to planning-scale LeWM action chunks. |
| Same-state intervention | `diagnosis/scripts/49_same_state_intervention.py` restores exact MuJoCo integration state and runs local action fans at horizons 1/2/4/8. | `pipeline.py` resets PushT to the same saved state, replays the same prefix, and executes two perturbed 25-step action sequences; uses block horizons 1/3/5. |
| Training target | Original `40_train_predictor_cf.py` uses factual-vs-shuffled-action InfoNCE. Later `25_train_cf_predictor.py`/`26_train_predictor_lora_cf.py` use object-response targets from a learned dynamics teacher, with factual latent/object loss. | Uses both simulator-observed branch endpoints and directly penalizes the error in their latent difference; frozen encoder, fine-tuned LeWM predictor/action encoder. |
| Control evidence | CAI predictor changes moved action-response proxy metrics, while reported MetaWorld contact success stayed 0/16 for baseline and CF-LoRA. | No result yet; LeWM PushT is a different task, model and evaluation protocol. |

The new paired target is a real implementation difference: it does not borrow a
counterfactual outcome from a learned teacher, and it supervises multi-step
latent response magnitude directly. It shares a research theme with earlier
work, but those internal attempts were not a publication claiming this method.
The earlier design had proposed multi-step sensitivity supervision, so the
present loss is not a proven novel contribution either. That question depends
on external prior work and strong closed-loop evidence, not simply on whether
we tried a related idea before.

There is an additional attribution risk. The old CAI programme found
representation-derived terminal costs could fail even when candidate dynamics
were supplied by the simulator (`diagnosis/docs/CURRENT_STATUS.md`). A better
action-conditioned predictor therefore need not improve selection. This result
is scoped to its MetaWorld setup and does not establish what happens on LeWM
PushT, but it is a concrete reason not to assume dynamics is the bottleneck.
The published LeWM PushT success is already 96.0 ± 2.83% on 50 tasks, leaving
little room for a reliable success gain in the proposed evaluation.

Decision after user correction: continue the independent LeWM PushT experiment
with matched branch-data prediction control and native closed-loop evaluation.
The MetaWorld null is a risk indicator, not a result for LeWM PushT. This
experiment is separate from CTA. No action-response training or control result
is claimed before the jobs finish.

## Evidence update — 2026-09-28

The corrected endpoint-response experiment is complete. Across three
independent *fine-tuning* seeds on 50 common PushT roots, the matched
prediction-only control scored [46, 46, 47]/50 and action-response scored
[48, 48, 45]/50. The mean paired gain was +1.33 percentage points, with a
two-level bootstrap 95% interval [-6, 8]. It does not establish an advantage
and does not exceed LeWM's published 96% mean. These seeds all start from
one released LeWM checkpoint, unlike the paper's independent pretraining
seeds. See `JOB_LEDGER.md` and `three_seed_endpoint_summary.json` in the run
root for exact artifacts. The seed-29 result is a concrete example where
lower held-out response loss did not mean better closed-loop control.

External prior work has narrowed the claim space since the original review.
AD-WM (arXiv:2609.30264, 24 September 2026) explicitly frames factual
prediction versus counterfactual action discrimination for LeWM-style MPC,
regularizes predicted transitions with action recovery, and reports a PushT
decrease from 94% to 92% despite gains elsewhere. CoCo
(arXiv:2608.04653) studies multi-step counterfactual consistency and action
response. Our exact same-state, simulator-observed *paired latent difference*
target is not the same objective as either paper, nor are unpublished
CAI-JEPA attempts external novelty claims. The general problem framing and
claim that action sensitivity helps planning are nevertheless already
occupied. A standalone loss paper would now need a sharper decision-level
method and reliable closed-loop evidence beyond saturated PushT.
