# GFPR H0-A locked protocol

Locked before GFPR model training or held-out prediction is run: 2026-08-19
UTC.

## Scope

- Native stack: released `quentinll/lewm-cube` checkpoint on OGBench-Cube.
- Data: the 32 Phase-0d snapshots whose complete final CEM populations were
  already executed from exact same simulator states.
- Candidates: the exact 300 final CEM samples plus the separately persisted
  exact CEM-returned mean anchor.
- Physical label: anchor physical distance minus candidate physical distance,
  in centimetres.  A corrective candidate improves the anchor by at least
  2 cm.
- No new outcome is queried for H0-A.  The existing outcomes are exposed only
  to the training folds and to post-selection evaluation of the held-out fold.

This is a learnability pilot, not final paper evidence.  There are only 32
independent snapshots, so a positive result licenses a fresh endpoint and a
negative result stops this formulation.

### Pre-training extraction-gate amendment

The first feature smoke (`42945`) measured a maximum absolute LeWM cost replay
difference of 0.001464 and stopped at the original `1e-4` numeric tolerance,
before writing features or training any model.  The gate was amended before
any GFPR outcome to recognize device-level floating drift while requiring the
decision-relevant ordering to remain unchanged: max absolute cost difference
at most `2e-3`, candidate-rank Spearman at least 0.9999, identical argmin, and
an identical top-10 set.  A second hard-snapshot extraction in array `42947`
stopped at max absolute drift 0.007792, while again preserving Spearman 1.0,
argmin, top-10, and every rank exactly; its persisted cost range extends to
297.93.  Before any GFPR training, the final extraction gate was therefore
locked to both max absolute drift at most `2e-2` and max relative drift at most
`1e-4`, while retaining all three ordering checks.  In that fresh v2 array,
snapshot 18 reached relative drift `1.0629e-4` (absolute drift `0.009614`) while
again preserving Spearman 1.0, identical argmin and top-10, and zero rank shift
across all 300 candidates.  Before any GFPR training or held-out prediction, the
final replay-only numeric gate was therefore amended to relative drift at most
`2e-4`; the absolute and exact ordering gates remain unchanged.  All partial v2
features are excluded and a fresh v3 array uses one code hash and a new output
root.  These amendments cannot hide candidate selection changes because no
learned scorer or held-out GFPR prediction existed, and the decision-relevant
ordering remained exactly invariant in every stopped run.

## Leakage-safe split

Snapshot is the indivisible group.  Outer test fold is `snapshot_index mod 4`:
eight held-out snapshots per fold.  Every snapshot comes from a distinct
episode in the locked manifest.  Models, feature normalization, class weights,
and uncertainty estimates use only the other 24 snapshots.  Candidate-level
random splits are forbidden.

Each arm is trained as a five-seed ensemble with the same architecture and
optimization budget:

1. `action_only`: normalized candidate action chunk and its delta from the
   returned anchor.
2. `proxy_action`: action features plus native and auxiliary scalar/rank
   features.
3. `latent_context` (primary): proxy-action features plus frozen LeWM current,
   goal, anchor-endpoint, and candidate-endpoint residual features.

The target is the physical gain over the exact returned anchor.  The loss is
Huber regression in centimetres plus a binary corrective-margin term.  No
simulator state, object position, success flag, or physical distance is an
input feature.

## Frozen-pool deployment rule

For a held-out snapshot, ensemble models score all 300 candidates once.  The
ungated arm selects the largest predicted physical gain if it is positive,
otherwise it retains the anchor.  The gated arm uses

`LCB_i = ensemble_mean_i - ensemble_std_i`

and switches only when the largest LCB exceeds the locked 2 cm corrective
margin.  The scorer never changes the CEM mean, variance, samples, or refit.

## Comparators

- exact native returned anchor;
- DINO-best single-candidate selector (outcome-blind);
- full-population physical oracle (evaluation-only upper bound);
- best of eight action-diverse candidates (physical-query upper bound, not a
  deployable zero-query baseline);
- action-only and proxy-action learned controls;
- ungated versus gated variants for every learned feature family.

## Readouts and decision rule

Primary arm: `latent_context_gated`.

Per-snapshot outcomes are selected physical distance, strict success, physical
gain over native, corrective selection, switch, and harmful switch (the anchor
was successful but the selected candidate was not).  Confidence intervals are
10,000-draw snapshot bootstraps over the 32 outer-fold predictions.

- `STRONG_GO_FRESH_H0`: lower 95% CI for mean physical gain and success gain
  over native are both above zero, harmful-switch rate is at most 5%, and at
  least four snapshots switch.
- `GO_FRESH_H0`: lower 95% CI for mean physical gain is above zero, point
  success gain is positive, harmful-switch rate is at most 10%, and at least
  four snapshots switch.
- `STOP_GFPR_FORMULATION`: mean physical gain is non-positive, point success
  gain is negative, or harmful-switch rate exceeds 20%.
- otherwise `HOLD_DIAGNOSE`.

No fresh 128-state run is submitted automatically unless H0-A reaches a GO
verdict.  This prevents adapting the final endpoint after observing pilot
failures.
