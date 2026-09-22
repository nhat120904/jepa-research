# Metric-space bridge: observed codec to action-conditioned prediction

Locked before launch. One bounded 30-minute GPU job, seed 20260924. Development
only, no physical-success/closed-loop planning claim, no subsequent jobs auto-submitted.

## Evidence motivating this change (53659)

53659 completed in 1m15s, exit 0, 18 tests passed. Both tiny fits passed; the general
learning code can fit at least one informative train prefix. On validation A-before-B:

| Observed-future arm | H32 MSE | H48 MSE | H64 MSE |
|---|---:|---:|---:|
| Learned frame affinity + fixed temporal algebra | .000417 | .002839 | .000878 |
| Uncompressed sequence, MSE | .044613 | .030603 | .028773 |
| Uncompressed sequence, pair loss | .054584 | .020924 | .024081 |
| Four-token summary, MSE | .045162 | .031375 | .029316 |
| Four-token summary, pair loss | .047924 | .032298 | .031733 |

Frame metric regret .007451/.002465/0.0. Summary pair loss did not consistently help.
The successful reference has dense RGB-derived affinity supervision and a known
aggregator; these are confounded changes, not proof of a unique failure mechanism.
It demonstrates recoverable signal in these projected DINO features on this small
development set, not that arbitrary robot-task information is retained.

## Frozen target, readout and data

Freeze the 53659 train-only frame metric, including its train normalization. Its
64-dimensional per-frame embeddings define a new SSL prediction target. Reuse 53630
features/53629 data and projection. All frame targets, affinity traces and query answers
come from RGB, not coordinates/state/rewards. No new labels/data/encoding. Training
still uses a HAND-DESIGNED visual-affinity/query grammar, not an unsupervised discovery
of arbitrary task specifications. Test split remains unread.

Metric embeddings standardized per channel using TRAIN frames only. Retain original
spatial DINO history as predictor input so a metric emphasizing the moving dot does
not remove wall/context information before action-conditioned prediction.

All embedding-output arms use the SAME frozen distance-to-anchor and SAME exact
reach/occupancy/endpoint/strict-order readout. Goals/queries do not enter summary or
trajectory predictor; they enter training losses and inference readout only.

## Observed trajectory compression

- Full observed metric trajectory: numerical reference, should reproduce 53659.
- Fixed 4/16-bin temporal averaging + interpolation: cheap compression baselines.
- Learned 4/16-token temporal codecs: width128 temporal Transformer, learned token
  pooling; time-query decoder reconstructs H×64 metric embeddings.
- Each learned codec gets 1,600 steps, AdamW 3e-4, batch two prefixes × eight candidates,
  H32/48, same minibatch RNG. Loss = normalized embedding MSE + native RGB-affinity
  trace MSE + 0.1 query-answer MSE. No pair loss, based on its inconsistent prior result.

Decoding requires H LIGHTWEIGHT latent tokens. This is not constant-time query reading
and not yet proof of a computation advantage. Dense trace supervision and fixed algebra
change the implementation from the previous learned query reader; comparisons MUST be
made against the new matched frame reference, not advertised as a gain over the old one.

## Predeclared development gate and forecasting

A codec is eligible for student training only if at EACH H32/48/64 on development val:
ordered MSE <= max(0.01, 2×observed-reference MSE), and regret <= reference regret+0.01.
This is a DEVELOPMENT routing gate, not held-out confirmation. Report all codecs and
failed gates. No tuning checkpoint or threshold after reading the result. Frozen codec
and decoder must transmit gradients to the forecasting student without being updated.

Always train action-conditioned full metric-frame prediction and its separately trained
zero-action control. These test predictability even if neither codec qualifies. Also
train direct-query prediction with the same metric-anchor inputs to its readout.
Only qualifying summary sizes receive an action-conditioned summary student. All
predictors receive the same grouped minibatches and 1,800 updates, no future inputs.
Frame and summary losses match the codec dense target losses; summary adds 0.1 frozen
codec-token alignment. Direct predicts eight answers with MSE (different intermediate
supervision; diagnostic, not an exactly matched training-compute control).

Frame/summary forecasting observes original history features + proposed actions only.
All arms are evaluated on train/validation; within-prefix action permutation diagnoses
dependence. Save per-prefix ordered selection, all query MSEs, codec gates and checkpoints.
Small-bank oracle-minus-default headroom remains .01853/.02369/.00615; no success
percentage or control-improvement claim can be substituted for these RGB scores.

## Interpretation

- Codec passes but student fails: now investigate forecastability/data, not readout alone.
- Full-frame forecast works but summary does not: no abstraction advantage yet.
- Fixed temporal pooling matches learned codec: learned compression not justified yet.
- Neither codec retains queries: do not spend more on its student in this run.
- Any positive result remains 1 seed on 12 reused development prefixes. It requires
  independent confirmation and a credible selection/control test before a method claim.
