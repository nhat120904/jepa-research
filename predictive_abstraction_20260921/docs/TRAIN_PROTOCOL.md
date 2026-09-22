# Offline codec and forecasting pilot — locked before training

**SUPERSEDED 2026-09-22:** use [TRAIN_PROTOCOL_V2.md](TRAIN_PROTOCOL_V2.md) and
`train_branches.py`. The old query construction and spatial baselines below are
historical, not approved for a new full run. No training was run under this draft.

Review update (2026-09-21): **requires revision before full training**. See
[protocol/code findings](REVIEW_AND_POLICY_GUIDANCE_20260921.md) on spatial baseline
targets, hindsight-query shortcuts, proposal-matched data, and the launcher's cache path.
No training run under this draft has been submitted.

Date: 2026-09-21. One-seed development protocol. Test split remains sealed. Offline
accuracy alone cannot validate planning or a method-paper claim.

## Data and query construction

Use only the 64 train and 16 validation episodes collected before headroom results.
Each example contains four observed feature frames, an ordered action chunk, and future
features at one native frame per action. Train horizons are 16 and 32; horizon 48 is an
unseen-length validation test. Episode split precedes every window and query.

Every window receives eight queries: two each of reach, occupancy, endpoint and strict
A-before-B. One query per family draws anchors from the realized future and one draws
unrelated anchors from the **training-frame anchor pool**. Validation negatives still use
the training pool; positive goal images may come from the validation episode, as a test-
time task query normally may. Query vectors use frozen global DINOv3 anchor features plus
an operator one-hot. Targets are still computed from the separately locked RGB-chroma
kernel at native cadence. No state, reward, wall/door coordinate or success label enters
training. The summary predictor never receives a query.

This balanced generator is a development distribution, not proof of arbitrary-query
generalization. Later confirmation must hold out anchor combinations and use independently
generated counterfactual branches; random windows from the same validation episode are
correlated and cannot be reported as independent episodes.

## Arms

All learned arms use width 128 and the same frozen spatial inputs, history, actions,
query tuples, optimizer family and stochastic batch stream where objectives permit.

1. **Query summary (method):** observed future -> four summary tokens -> reader, trained
   on answer loss. Freeze codec/reader; predict the four tokens from history+actions with
   primary answer loss and a 0.1 latent-alignment auxiliary.
2. **Generic summary:** same four-token bottleneck, trained to reconstruct ordered global
   future DINO features; then an equal-capacity query reader. Its action predictor gets
   the same answer and latent-alignment losses.
3. **Direct query:** query attends directly to history+all proposed action tokens and
   predicts the answer. Strong single-query baseline, no reusable predicted summary.
4. **Frame sequence WM:** predict one token per future time step, supervise it with a fixed
   slice of the global DINO feature and the same answer loss. It has H output tokens and is
   an information/compute reference, not matched compression.
5. **Endpoint WM:** predict one terminal token with the same latent and query losses.

Reader widths are matched. Total parameter counts and eventually inference costs must be
reported; the full frame model is intentionally allowed more output tokens. Exact query
statistics on observed paths are a target-computation ceiling, not an action-conditioned
baseline. A candidate-selection experiment using realized branches follows this pilot.

## Optimization and evaluation

Development full run: 1,200 codec steps; 1,200 generic reconstruction steps; 600 generic
reader steps; 1,800 steps for each predictor; batch 32; AdamW at 3e-4. A short profile at
2% of these steps must first exercise every arm end to end. It is not interpretable as an
accuracy result. Report MSE and Pearson r per query family at horizons 16/32/48.

Primary mechanism gate for investment, evaluated at unseen H=48:

- observed query codec reduces ordered-query MSE by at least 10% versus generic codec;
- forecast query summary reduces ordered-query MSE by at least 10% versus generic summary;
- at H=32 it is not more than 5% worse than generic summary.

Frame and direct-query arms are stronger threats rather than straw men. If either is more
accurate, the summary can proceed only with a measured accuracy/compute or multi-query
Pareto advantage, followed by improved candidate selection. If direct query dominates at
one and many queries, reusable summary machinery is not justified. If the observed codec
passes but its predictor does not, abstraction is learnable but not forecastable from the
available data. If only endpoint queries improve, the path-summary claim fails.

Passing this one-seed gate authorizes three-seed confirmation and counterfactual ranking;
it is not itself a result. Failing it stops this architecture/configuration, not every
possible predictive abstraction. Do not tune against the sealed test split.
