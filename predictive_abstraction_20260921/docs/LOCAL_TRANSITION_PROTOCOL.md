# Local spatial transition feasibility — locked before submission, 2026-09-22

Motivation: 53700 did not rescue either compact or matched frame forecasts with a
new reader. Before another abstraction design, test whether local dynamics can be
learned and whether errors appear locally or only during open-loop rollout.

## Training

- Same 53630 frozen DINOv3 4x4 grid, 24 train / 12 reused val prefixes. No test reads,
  new trajectories, new encoder, physical state/reward labels or policy training.
- Retain 16 spatial tokens, each with the same frozen random 1024->32 projection
  already used by the successful observed-future metric. Train-only normalization.
- Residual patch transition: per-patch projection + spatial position + broadcast
  action embedding -> two width-64 Transformer layers -> per-patch feature increment.
  Initialize residual output zero (persistence); action/history only at prediction.
- Training target is next image feature, NOT metric embedding, query answers or RGB
  affinity. Frozen metric/readout is evaluation only. Feature normalization reuses
  its TRAIN-only empirical input moments, not the learned metric mapping.
- Scale increment by TRAIN per-coordinate one-step RMS change, floor .01. Loss is
  squared feature error divided by that scale, averaged over every unroll step.
  This fixed weighting emphasizes learning changes; report unweighted error too.
- Tiny test: top eight TRAIN one-step transitions by scaled motion, 400 updates;
  proceed only if final error < 50% of persistence error. This tests fitting, not
  generalization. A failure stops this job before the three full arms.
- Full arms: 1-step training; 4-step open-loop training; matched 4-step no-action.
  Same initialization, sampler seed 20260927, 1,200 updates, batch 32, AdamW 3e-4.
  Four-step costs more compute; not an equal-compute abstraction comparison.
- Uniform branch sampling and valid start times. H32/H48 banks contain different
  action sequences: retain both; do not deduplicate them as trajectory prefixes.

## Evaluation

- Teacher forcing (TF): every next state predicted from TRUE previous frame. It reads
  realized future history and is not a deployable forecasting/planning baseline.
- Open loop: initial observed frame + actions only, without future observations.
- Persistence controls: repeat initial state (open); repeat true previous state (TF).
- Action control: same trained step4 model, action sequences cyclically reassigned
  across candidates within each prefix. Separate trained no-action control also runs.
- Evaluate native banks H32/48/64. On H48 bank additionally truncate to H4/8/16/32;
  keep bank horizon explicit, because shorter-bank proposals are different actions.
- Report TRAIN/VAL feature MSE (weighted/unweighted), eight query MSEs, missed event
  counts, within-prefix selection with first/uniform ties and actual default regret.
- Reproduce 53700 observed-future native-horizon MSE/regret to within 1e-4.

## Predeclared developmental gate (not a paper claim)

For step4 on validation bank H48, require all three:

1. Teacher-forced feature MSE >=10% lower than teacher-forced persistence.
2. Open-loop H8 feature MSE >=10% lower than initial-state persistence.
3. Open-loop H8 feature MSE >=10% lower than the trained no-action arm.

These are short-dynamics checks only. A pass does not establish query retention,
long-horizon ranking, compression benefit or control. Tiny fit failure, finite-value
errors and reference mismatch are reported distinctly, not treated as scientific nulls.

If TF is weak: inspect training/generalization and local state assumptions before
adding long-horizon machinery. If TF is good but open-loop weak: investigate rollout
training and off-manifold states. If both work but queries/ranking fail: inspect
task-relevant feature errors and objective, not automatically another readout sweep.
Only after a capable frame reference should a new compact abstraction be tested.

One seed, three shared layouts, reused validation; not DINO-WM reproduction, memory
task or new-query generalization. No automatic follow-on job. One GPU, 20-minute cap;
tests execute on compute before training. Source and input hashes snapshotted.
