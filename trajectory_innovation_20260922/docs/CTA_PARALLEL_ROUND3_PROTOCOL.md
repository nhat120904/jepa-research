# Round 3: parallel FSQ world model with decision losses

2026-09-26. User's revised plan authorizes WM changes and matched frame-WM.
Keep PushT, existing geometry labels, K=8, execute8 and M=16. No source/reader
co-design or capacity sweep. Existing reader-only jobs 55066–55068 stay intact.

## Architecture and training

Start from geometry `cta_geometry_e2e_55018/train/cta.pt` (SHA256
`b777f1588e9969daab2fc80eebe3ce12a44e36759bddade51fc20645e1f29f5f`).
Freeze source encoder, FSQ, code reader and FULL reader. Keep the old AR-WM and
DIRECT as references. No privileged future is available to deployment selectors.

Parallel predictor: context/action encoder and unmasked Transformer decoder with
16 learned queries; output three categorical distributions of sizes 8/8/4 per
token. Reader input is the expectation of each normalized FSQ coordinate.
Coordinate likelihoods factorize conditional on context/actions; the decoder
features can interact across all token positions. This is not an exact joint
posterior and is not a calibrated entropy/rate measurement. Coordinate categorical
NLL itself does not impose an ordinal distance penalty; expectations and task
gradients make use of the numeric FSQ grid. No greedy prefix is used.

Source code has 128 nominal bits; prediction has 48 continuous scalars (768 bits
if stored at FP16; current expected-value computation returns FP32, 1536 bits).
Endpoint baseline output is 256x128 scalars plus four proprio values.
This roughly 683x output-size ratio is NOT a latency/compute speedup claim.

Train four networks in one job, on identical batches/goal samples, seed 0,
**6000 AdamW updates, effective batch 32 banks**, LR 1e-4, weight decay .05.
This budget amendment was authorized before jobs 55072–55074 started; those jobs
were cancelled at zero elapsed time. Sample all training banks uniformly from
roots 30250–31049. Offline dev: 2000–2099. Use the final checkpoint, not dev-best
selection. Total additional exposure is 192,000 banks per network (roughly 8.5
dataset passes with replacement); this does not itself prove convergence.

Memory-safe effective batch: four microbatches of eight banks, one optimizer
update and one gradient clip after all four. Coordinate NLL/MSE/consistency means
are divided by four. Pair ranking uses the total eligible-pair count over all
32 banks, not separate microbatch means. All four networks share the same full
batch and goal samples. Existing models use zero dropout and LayerNorm.

Run the full dev ladder every 1000 updates, including prediction NLL and normalized
frame/proprio errors. Save the five intermediate reports in `dev_step_*/` and the
final report at the run root; `dev_curve.json` tracks all six points. These are
reporting checkpoints, not gates or opportunities to choose a winning model.
Log wall time per full optimizer update, cumulative mean, estimated remaining
training time (excluding future eval overhead), and time spent per network.
The rough 0.5 seconds/update estimate is unverified and must not be promised.

Atomically save networks, optimizers and RNG states every 250 updates. An early
Slurm time-limit signal requests a final checkpoint after a complete update and
exits nonzero, preserving an explicit partial result; dependencies then cancel.
`--resume-from` can continue that checkpoint into a fresh run with unchanged
configuration. No speculative continuation job is submitted. If actual timings
show FRAME dominates the budget, split that training in a subsequent recorded
execution change instead of reporting an undertrained frame baseline as a win.

1. **NLL8:** parallel predictor, mean coordinate categorical NLL only.
2. **CTA3:** identical initialization/architecture, same NLL plus bank-centered
   score consistency and weighted geometry ranking, with unit loss coefficients.
   NLL anchors predictions to source codes; both additional losses backpropagate
   through the frozen reader into WM. No `no_grad` around predicted-code reading.
3. **FRAME8:** endpoint latent WM predicts residuals of the current 256 PCA-128
   tokens and current four-dimensional proprio. Same encoder/decoder width/depth;
   257 parallel queries, image/proprio output heads. Loss: average normalized
   image/proprio MSE + consistency with the frozen FULL reader on true futures +
   the same weighted geometry rank loss through FULL on predicted futures.
4. **DIRECT3:** continue the parent's direct scorer on the same weighted rank
   objective and number of updates. No new future representation or auxiliary
   code-prediction objective. DIRECT8 remains as the unmodified reference.

WM context/action encoder and decoder weights warm-start from the old AR-WM;
parallel query initialization reuses AR positional weights when dimensions match.
The NLL/task pair is bitwise identically initialized. FRAME shares the same warm
backbone initialization but has a different query count and new zero-initialized
residual output heads. This is a matched latent endpoint baseline, not a literal
reproduction of published DINO-WM or the image diffusion model in GPC.

Weighted pair loss: existing positive-label margin .001, multiplied by
`min(delta_geometry/.01, 1)`, averaged over all eligible pairs in the batch, not
normalized per bank. This is soft task-gap weighting, NOT a claim that .01 is a
proven pixel visibility cutoff. Consistency is smooth-L1 between bank-centered
predicted/teacher scores divided by a fixed teacher-score RMS estimated on 256
uniform TRAIN banks. Frame/proprio MSE normalization also uses training only.
CODE and FULL are imperfect teachers; task-label ranking and prediction anchors
are retained. FRAME uses its FULL teacher, CTA its CODE teacher; teacher quality
therefore differs and a performance gap is not pure causal attribution to compression.

## End-to-end evaluation

One training job includes tests, training, offline ladder and runtime consistency;
low offline scores do not block closed-loop evaluation. Frozen-state equality,
finite gradients and cache/runtime consistency are correctness checks only.
If both cached/runtime scores are flat in every bank, undefined Spearman does
not block control when the existing score-error checks pass; this is logged as
uninformative matching outputs. Non-flat rank thresholds are unchanged. Earlier
experiments retain the previous default behavior through an opt-in argument.

All 100 development roots 2100–2199 run paired across eight arms:
P0, CTA3, NLL8, FRAME8, DIRECT3, old CTA8, old CTA8E, old DIRECT8.
Only the selected branch is simulated; no oracle arm and no future-proprio leak.
Reader-only adaptation results remain a separate matched experiment, not a source
of cherry-picked initialization. No sealed roots 3000–3399 are opened.

**Primary:** CTA3 minus P0 mean maximum native reward over executed steps:
`mean_episode(max_t clip(coverage_t/.95,0,1))`, explicitly EXCLUDING reset.
**Secondary:** native success, CTA3 versus NLL8/DIRECT3/FRAME8/old AR, raw maximum
coverage, scorer latency and full episode/decision time. Report root-paired CIs
and McNemar on success. Same metric definition does not imply matched comparison
with GPC's published numbers; environments/policies/budgets differ.

Offline ladder: all banks, center displacement >= or < 512/96 units, and max
corresponding-vertex displacement >= or < that cutoff. Vertex displacement includes
rotation that center distance misses. Grouping uses true outcomes for diagnostic
stratification only and is not available to the deployed selector. Report bank
counts, oracle gain share, ranking retention and root-cluster intervals. Grouping
by small oracle gap preferentially selects states where candidate 0 is already
near-best, so overrides have little upside and can have substantial downside.
Negative retention there can reflect this selection, and small denominators can
amplify it; do not interpret it as evidence that the labels contain no information.
Such groups remain conditional descriptive analyses, not unbiased task-wide tests.

## Budget and claim discipline

Train: one MIG H100 3g.40gb, 8 CPU, 96 GB, 3-hour cap.
Closed loop: ten 10-root shards, concurrency 2, one MIG GPU, 4 CPU, 48 GB,
75-minute cap each. CPU aggregation: 2 CPU, 8 GB, 10 minutes.
Total allocation ceiling: 15.5 MIG GPU-hours; no speculative duplicate arrays.
Slurm dependencies cancel on upstream failure. Record immutable releases/job IDs.
Running reader-only jobs use their earlier source snapshot and are unaffected.
Its remaining array 55067 is throttled to concurrency 1 so this training can
start when either currently running reader shard finishes; neither active shard
is cancelled. This scheduling change does not alter the reader experiment.

This explicitly authorized Round-3 iteration amends the historical three-debug-round
cap for this bounded development work. It does not authorize unlimited sweeps.
The normalized-score primary applies prospectively to this round; do not rename
the primary of reader adaptation 55066–55068 after seeing its results. Any final
sealed-test protocol must be locked and documented before opening that set.

M=32, K=16/32, regression ablation and a second temporal arena remain subsequent
work. Parallel/continuous predictions may still collapse or average incompatible
futures. A smaller output does not establish computational advantage; a positive
offline score does not establish control improvement or CVPR readiness.
