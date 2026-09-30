# Endpoint versus trajectory codec: fixed offline scope experiment

Locked 2026-09-26 before training. User explicitly approved this comparison.
**Execution stopped later on 2026-09-26:** user challenged its priority for
the central CTA claim and requested avoiding uninformative compute. Two seed-0
tasks were cancelled after 9m50s each; seed-1 tasks and aggregation never ran.
No final scientific result. Original predeclared design below is preserved.
This is a representation scope test, not a new claim of improved planning.
It is not an extra WM debug round or a search over the final CTA controller.
No controller configuration will be selected from this experiment automatically.

## Question and two arms

Does access to intermediate future images improve a fixed-bit code's ability
to retain endpoint-coverage ranking information, compared to a codec trained
only on the endpoint? Both source encoders observe actual futures for this
test. No WM, actions, new labels, simulator branching or closed loop.

| Setting | Endpoint | Trajectory |
|---|---|---|
| Future image evidence | End frame | Frames 2, 4, 6 plus end frame |
| Future proprioception | Same end agent positions now/previous step | Same |
| Context and goal frames | Existing cached contract | Same |
| SourceEncoder | path=False | path=True |
| Source queries / FSQ | M=16, levels 8x8x4, 128 nominal bits | Same |
| Source/reader/decoder widths and depths | 256; 3/4/2 layers; 8 heads | Same |
| Main query | cov8 pairwise ranking, margin 1e-3 | Same |
| Auxiliary reconstruction | Endpoint tokens only | Same |
| Loss weights | rank 1, endpoint reconstruction 1, FSQ saturation 1 | Same |
| Co-design / reader adaptation | None | None |
| Initialization | Fresh; shared weights identical within each seed pair | Same |
| Updates | 10,000; fixed last checkpoint | Same |
| Batch | 16 banks (128 candidates), uniform over all decisions | Same exact batches/goals |
| AdamW | LR 3e-4, WD .05, 500 warmup, cosine decay, clip global norm 1 | Same |
| Paired seeds | 0, 1 | 0, 1 |

The path encoder naturally has additional intermediate-frame projection and
position parameters. All common initial weights are copied from the same
template and verified by SHA256; reader and endpoint decoder initialization
are exactly equal. Report parameter counts and elapsed training time. Equal
updates/examples/bit budget do not imply equal FLOPs: trajectory attention
sees more input tokens. Do not claim an exactly matched FLOP experiment.

## Avoid the unfair reconstruction target

The endpoint arm is not asked to reconstruct unseen intermediate frames.
Both decoders reconstruct only endpoint features from context and code,
normalized by the same train-derived copy-context MSE (same fixed sample).
This differs from earlier rounds' full-segment reconstruction in **both**
arms. Therefore compare these two new arms; do not attribute a difference
against an old Round-1/2 checkpoint solely to path information.

Both arms sample every phase, including coverage-flat banks. Ranking loss
masks noninformative pairs; reconstruction/saturation still train there.
This corrects exposure symmetrically but is not a separate estimate of the
effect of changing sampling. The endpoint input dictionary physically omits
intermediate frames. Source encoders and readers do not receive actions.

## Data and fixed evaluation

Existing `cta_feat_54490` cache only; no re-encoding or collection:

- Train roots 30250–31049 (800); dev roots 2000–2099 (100).
- All 22,680 training and 3,011 dev decisions; sealed roots untouched.
- Existing PCA and 16 goal images, unchanged.
- Development evaluation only once, after saving the fixed final checkpoint.
- Same cached-feature/bfloat16 scoring and goal-averaging contract in both arms.

Primary outcome: trajectory minus endpoint **retained coverage gap**, average
over the two paired training seeds. Bootstrap source roots, keeping all
decisions and seed pairs within each root cluster. Also report each seed's
paired interval. These intervals are conditional on two seeds, not a complete
characterization of training-seed uncertainty or a sealed confirmation result.

Secondary: within-bank Spearman, endpoint reconstruction R2, code perplexity,
bank distinctness, flat-bank override frequency, parameter counts and training
time. Overrides in coverage-flat banks are descriptive, not causal harm/benefit.
Evaluation duration includes transfer/reconstruction and is not deployment latency.

Predeclared practical margin: .05 in **retained-gap units**, not 5 percentage
points of native success. Interpret:

1. Mean advantage >=.05, lower 95% CI >0, and positive in both seeds:
   trajectory advantage supported **on this endpoint target**.
2. Reverse conditions: endpoint advantage supported.
3. Entire interval inside [-.05,+.05]: difference within the practical margin
   on tested seeds. Do not claim universal equivalence.
4. Otherwise: inconclusive. An interval crossing zero is not evidence of equality.

Even a positive result does not establish temporal-order retention or a need
for whole-path representation in native PushT. A null/negative result does
not refute trajectory abstraction for other query families. Conditional
per-frame coding and genuinely temporal queries remain separate missing controls.

## Compute and reproducibility

New immutable release and job-specific snapshots. No peer dirty files replaced.

1. CPU checks: all unit suites plus four tiny actual-cache training/evaluation
   runs and the paired aggregation (2 updates each, data caps 16 train/8 dev,
   one goal). 4 CPU, 24 GB, 30 minutes. Smoke results are not research evidence.
2. Training array 0–3, concurrency 2, afterok checks. Task mapping:
   0 endpoint seed 0; 1 trajectory seed 0; 2 endpoint seed 1; 3 trajectory seed 1.
   Each: one 3g.40gb MIG, 8 CPU, 64 GB, 90-minute hard limit.
   Total allocation ceiling: **6 MIG GPU-hours**. No speculative repeats.
3. CPU paired comparison afterok the complete array: 4 CPU, 24 GB, 15 minutes.
   Assert matching configs, root/label arrays, shared initialization hashes,
   sampled-batch/goal hashes and reconstruction normalizers before comparing.

No lambda sweep, early stopping, best-of-dev checkpoints, extra epochs or
automatic downstream training. Failures leave dependent jobs cancelled.
Check squeue and sacct before submission, record job IDs, then return without
foreground polling. A failed or timed-out run is incomplete, never a zero score.

Submitted: CPU check **54988** -> four-task training array **54989** -> CPU
comparison **54990**. Immutable release:
`/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_scope_release_20260926T040509Z/`.
Runtime results are pending. Compile/shell/whitespace checks passed before
submission. See JOB_LEDGER.md for exact output paths and task mapping.
