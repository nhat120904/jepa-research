# Runtime precision repair and bounded WM diagnosis

2026-09-26. This is an inference-contract repair, not a new training/debug
round or a change to progress labels. No checkpoint weights are changed.

## Blocker found before closed-loop Round 2

54976_0 failed the unchanged CODE8 preflight after 10m27s: median score
relative difference .00273, clear-bank argmax agreement .75 (12 clear banks),
within-bank Spearman .838. The required argmax agreement is .85. This is a
failed validation, not a control result. Its dependent jobs 54977/54978 were
cancelled before running. The remaining old preflight 54976_1 was cancelled
after identifying the precision mismatch, to avoid further obsolete compute.

Static source inspection identifies a real execution difference:

- `cta_encode.Tokens.__call__`: DINO + PCA in float32, then store fp16 tokens.
- `Planner.context` and goal initialization: token extraction outside autocast.
- `Planner.scores` CODE8/FULL8: `future()` calls token extraction *inside*
  the scorer's bfloat16 autocast region. `inference_mode` does not disable it.

The repair explicitly disables autocast around DINO/PCA in `Planner.tokens`,
then casts to fp16 as the cache does. This restores the intended precision
contract for all frame types. A regression test enters an outer CPU bfloat16
region and requires identical tokens and float32 feature execution.

This difference can affect quantization and selected actions, but we have
not yet measured how much of the failed preflight it explains. Rerun the
same preflight and thresholds; do not relax them. Batch-size-dependent
numeric differences may still remain and would require a separate diagnosis.

Existing offline ladders used cached features and are unchanged. Actual-future
closed-loop CODE8/FULL8 diagnostics may be affected; do not attribute a new
result to improved representation learning when only precision changed.
P0/PHYS8 do not depend on the CTA visual encoder. CTA8/CTA8E/DIRECT8 encode
current context outside scorer autocast already; weights and their intended
inference inputs remain unchanged.

## Why PushT remains useful, within a limited claim

W2 on 400 roots established repeated coverage-oracle headroom: P0 .625,
PHYS8 .765, gain .140 [.0875,.1925]. Coverage incompleteness therefore does
not explain away all current loss to a useful selector. PushT can test
whether predicted abstraction recovers that existing signal. It cannot
alone validate task-general or long-horizon progress reasoning.

Keep one final bounded training-debug opportunity, per the existing CTA
protocol. Before choosing it, finish runtime verification and the frozen
WM diagnostic below. Do not sweep lambda, add manual geometry rewards,
or launch a new terminal-continuation campaign on the current evidence.

## Frozen WM evidence audit, no new training

Use control checkpoint `cta_r2_54933_0/cta.pt` and the same fixed first 600
offline dev decisions as the existing privileged-prefix panel. Report the
number of independent roots; this panel is diagnostic, not population
confirmation. All modules frozen, actual-future code targets unchanged.

Score through the same reader:

1. Actual source code.
2. WM free-running prediction.
3. Context-only prior free-running prediction.
4. WM prediction with true source prefixes.
5. Context-only prior with the same true prefixes.
6. WM with sibling actions cyclically permuted, retaining original prefixes.

Report within-bank ranking and paired differences, CE, per-position token
error and collisions on label-distinct pairs. True-prefix variants are
privileged and never deployed. A high score shared by the action-free prior
would undermine attributing the previous high true-prefix result to WM action
understanding. Action-sensitive signal under true prefixes would motivate
testing a predictor that does not require true prefixes at train time.

The leading candidate for that next intervention is a parallel categorical
predictor with the good source encoder frozen, compared to an equal-budget
autoregressive predictor. Reader adaptation is a separate possibility if
CTA8E's closed-loop result supports its usefulness. Neither is automatically
selected or submitted by this audit; no third-round training is queued.

## Submission plan

New immutable release; CPU tests/audit -> corrected preflight (2 serial
20-minute MIG tasks) and WM audit (one 45-minute MIG task) -> original
closed-loop design (20 x 10-root tasks, concurrency 2, 90 minutes each) ->
CPU paired aggregation. Closed loop depends on both preflight and the WM
audit, so at most two MIGs are used by this chain simultaneously. Invalid
dependencies cancel. No foreground job polling after submission.

Submitted: CPU check **54980**; corrected preflight **54981** and WM audit
**54982** after that check; closed-loop array **54983** after both; paired
aggregation **54984**. Immutable release:
`cta_precision_release_20260926T024651Z` under the shared results directory.
The new jobs' outputs are pending; compilation and shell checks passed.
