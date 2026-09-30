# Same-state action response on OGBench-Cube

Current method test: `response_followup.py` fine-tunes LeWM on differences
between the observed endpoints of action chunks branched from the **same
state**. Its objective is endpoint prediction plus the mean squared error of
all within-bank predicted versus observed action effects. The action-response
term is twice the variance of endpoint prediction residuals across actions;
thus it emphasizes action-dependent errors without physical-distance labels.
At deployment, the planner retains released LeWM GoalMSE and CEM settings.
The matched prediction-only/native-cost arm from job 55413 supplies the same
50 evaluation roots. See `JOB_LEDGER.md` for submitted job IDs and results.
That direct response run completed at 35/50 versus prediction-only 36/50;
the paired interval was [-8,+4] percentage points. The current focused
iteration, `response_mixed.py`, broadens candidate action effects by mixing
initial and final CEM populations and trains both arms on the same cache.
It completed with response 34/50 versus matched prediction-only 37/50,
despite a roughly 2.5-fold increase in endpoint-response spread. The direct
loss has not demonstrated a closed-loop benefit on Cube.

The scorer/ranking branch below is retained as a diagnostic record. It did
not improve closed-loop control: ranking+scorer 28/50 versus
prediction-only+native-cost 36/50. Its pending follow-ups were cancelled
before starting. AD-WM is a preprint relevant only to checking prior work;
it is not an accepted-paper benchmark or a reason to redirect this method.

This experiment trains on exactly restored simulator branches from CEM
candidate populations. A task-supervised goal-progress scorer sees encoded
endpoint and goal images; a world-model variant additionally learns to rank
near-elite candidates by the outcome of those branches. At deployment, both
parts read only current/goal images and candidate actions through predicted
latents. No simulator state, future image, or physical label reaches the planner.

The matched comparison is prediction-only versus prediction plus ranking, both
with the **same scorer and CEM**. Additional arms evaluate the scorer on the
released dynamics and the prediction-only model with its native latent-L2
cost, isolating cost calibration. All use 300 CEM candidates, 30 iterations,
30 elites, a five-block horizon and 50 environment steps on 50 held-out roots.
Collection excludes every evaluation episode and keeps one training root per
episode. The first two training roots require exact repeat of a branched
outcome. Simulator distance labels make this *task-supervised and
simulator-assisted training*, unlike self-supervised LeWM. At evaluation,
the corrected full MuJoCo reset from the existing Cube audit is used.

The original LeWM checkpoint is loaded as an initial model, not retrained.

The current follow-up runs with `sbatch slurm_response_followup.sh` using
`UPDATES`, `EVAL_ROOTS`, `SEED`, `LR`, and `RESPONSE_WEIGHT` overrides.
The original scorer/ranking pipeline runs with `sbatch slurm_pipeline.sh`.
Its environment overrides are `COLLECT`,
`SCORER_UPDATES`, `UPDATES`, `EVAL_ROOTS`, `SEED`, `LR`, `RANK_WEIGHT`, and
`POPULATION` (`final` or `mixed`).
Every run has a unique directory under
`/mnt/data/nhatnc129/jepa/action_response_cube/`.

The first full run (`55413`) has completed on the original 50-root protocol:
prediction-only + native latent-L2 36/50, prediction-only + learned scorer
26/50, ranking + learned scorer 28/50. The scorer itself did not beat native
latent-L2 on held-out true CEM endpoints. Twenty to twenty-two successes per
arm happened within three steps. `native_followup.py` was an unrun
task-supervised ranking variant; `response_followup.py` is the current
self-supervised same-state action-response continuation.
The `mixed` population broadens training to initial and final CEM
neighborhoods; it is prepared but has no result yet.
