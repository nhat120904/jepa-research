# Fixed-summary forecasting feasibility — bounded follow-up to 53674

Locked before submission. One seed 20260925, one GPU job capped at 20 minutes,
same data and feature cache; no subsequent job automatically scheduled.

## Evidence and interpretation

53674 completed in 1m39s, exit 0, 22 tests passed; frozen reference reproduced.
Learned 4-token codec failed every development retention gate. Learned 16-token
codec passed only H32, not H48/H64; neither summary student was trained.

| Observed-future representation | H32 order MSE | H48 | H64 |
|---|---:|---:|---:|
| Fixed 16 bins | .001135 | .003821 | .002774 |
| Learned 16 tokens | .007763 | .012818 | .027943 |

Fixed-bin regrets .012303/.002465/0.0. Thus temporal compression to this resolution
can preserve these queries on this small sample; learned pooling/decoding is not
justified over the simpler baseline yet. This does NOT establish arbitrary-horizon,
arbitrary-task sufficiency or a learned-summary contribution.

Full metric-frame forecast MSE .046395/.032088/.030536 is close to no-action
.045615/.031754/.029673 on validation A-before-B. Action shuffling has little effect
on these MSEs. Both predict poorly relative to true-future readout; neither improves
selection over default. Dense latent train loss falls with actions, but this has
not translated into held-out query prediction/selection.

## Concrete next change

Stop spending on learned codecs in this round. Forecast 16 fixed time-bin metric
means directly from original spatial DINO history plus proposed actions. Expand by
fixed linear interpolation and apply the SAME frozen affinity/temporal query algebra.
The target is query-independent; goals only enter readout/loss, never the predictor.
Still RGB-derived supervision, no simulator states/coordinates/rewards or policy labels.

Every NEW arm uses the SAME context encoder and time-query decoder. Decoder receives
normalized time, squared normalized time, known H/48 duration and their product, then
cross-attends history/actions. Full-frame queries are at observed times 1/H,...,1;
compact queries at equal-bin centers. Thus no learned table/new positional indices
are required for output slots at H64. Context action positions remain as before.
Phase/duration coding is an additional shared architectural change: any comparison
to 53674 cannot attribute a gain to compression alone.

## Gates and controls

Reproduce 53674 fixed-bin observed reference before any optimization (MSE/regret
tolerance 1e-4). Pick the H32 TRAIN prefix with largest ordered-target variance for
a tiny-fit test only. Train compact and full-frame/coarse-loss models for 1,000 steps
each. Tiny pass requires ordered-query MSE <.01 AND standardized 16-bin embedding
MSE <.05. All models consume only history/actions, not future inputs.

If NEITHER tiny fit passes, skip larger training and stop with the failure report.
If at least one passes, initialize FRESH models and run the following fixed four-arm
comparison (1,800 steps each, AdamW 3e-4, two prefixes × eight candidates per batch):

1. Full frames, dense embedding MSE.
2. Full frames, MSE on their 16-bin means (matched coarse-target control).
3. Direct 16-bin forecast, MSE on 16-bin targets.
4. Direct 16-bin forecast, separately trained with zero actions.

All arms additionally receive the SAME dense RGB affinity-trace loss and 0.1 times
query-answer MSE. Same seed, optimizer and grouped batch stream. Model parameter
counts match; output token count differs. No inference-speed benefit is assumed.
Compact arm is not an EMA JEPA/learned composer; it is a predictive-abstraction
feasibility baseline with a fixed summary definition.

## Measurements and next-step limits

Train/validation query MSE, bin/full latent MSE, candidate selection regret, within-prefix
action shuffling. H32/48 train, H64 development extrapolation. Same 24/12 prefixes,
eight candidates and limited oracle-default headroom. No test data or simulator work.
No physical-success, closed-loop, significance, novelty or paper-readiness claim.

- Tiny fits pass, broad TRAIN fit fails: optimization/capacity/objective still unresolved.
- TRAIN fits but held-out forecasts fail: generalization/data support issue remains;
  no claim that compression itself is the cause.
- Compact matches no-action or has poor selection: no control/planning expansion.
- Compact works but full-frame/coarse-loss control matches it: coarse supervision may
  explain the gain; compression needs a measured compute/accuracy benefit to be useful.
- A favorable result needs independent validation; no threshold retuning or broader
  arena search automatically follows this run. Do not keep extending this loop without
  an explicit review of whether there is any method evidence.
