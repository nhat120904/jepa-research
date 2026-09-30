# Round 2: predictability-aware conditional trajectory abstraction

Locked 2026-09-25 before submission. Development experiment, seed 0 only.
This is debug round 2 of the maximum three rounds in CTA_E2E_PROTOCOL.md.

## Decision and hypothesis

Continue checkpoint `cta_train_54717/cta.pt`, not a new stage-1/2 run.
Round 1 retains gap 0.709 in actual-future codes but only 0.323 in greedy
predictions and 0.454 in expected codewords. Source bank distinctness is 0.79;
predicted distinctness is 0.29. The hypothesis is that a code optimized jointly
for retention and categorical predictability transfers more of its information
through the action-conditioned WM.

Reader adaptation remains a valid Round-3 option. It addresses distribution
mismatch, but cannot distinguish two candidates with identical C and code.
Do not combine it with co-design in this experiment. Do not assume the entire
code-to-prediction gap is caused by inadequate WM capacity: reader mismatch,
autoregressive exposure mismatch, unavailable state and discrete collisions
can contribute. The new diagnostics only help distinguish these possibilities.

## One treatment, one matched control

| Setting | Control | Method |
|---|---|---|
| Parent | Round 1, job 54717, seed 0 | same exact checkpoint |
| Source predictability weight | 0 | 0.1 |
| Source/reader/feature decoder updates | 3,000 | 3,000 |
| FULL and DIRECT updates | 3,000 each | 3,000 each |
| Alternating WM/prior updates | 6,000 | 6,000 |
| Final fixed-source WM/prior alignment | 2,000 | 2,000 |
| Source / WM block | 200 / 400 | 200 / 400 |
| AdamW LR / weight decay / warmup | 1e-4 / .05 / 200 | same |
| FSQ saturation / feature reconstruction / sibling contrast weights | 1 / 1 / 1 | same |
| M / vocabulary / seed / EMA | 16 / 256 / 0 / .99 | same |
| Reader adaptation | none | none |

Batch sizes, training pool, action interface, PCA, goal images and all other
architecture settings are inherited. Ranking updates use the existing ranking
pool; WM updates sample uniformly over all training decisions and receive no
reward labels or goals. The supervised reader's gradient still shapes the
source code, as in the existing task-aware §5a amendment: this is not a fully
self-supervised representation.

Both arms start from identical parameters and RNG streams and use the same
batches, sampled goals and update counts. Different GPU kernels can still
produce numeric drift, so compare FULL/DIRECT score arrays explicitly.
Extra FULL/DIRECT training prevents crediting CTA for extra supervised updates
against a stale baseline. The decisive attribution contrast is lambda=.1 vs
lambda=0 continuation; a win over the old checkpoint alone is insufficient.

## Categorical objective and estimator

The unused old stage-3 implementation pulls a code toward the expected FSQ
codeword. This is a squared-error objective, not categorical cross-entropy;
under a multimodal distribution the expected vector can be an unlikely code.
Round 2 instead uses `fsq_predictability_nll` in `ti_wm/cta.py`:

1. Freeze a WM copy for each source block.
2. Compute categorical logits with the current source's hard, stopped-gradient
   prefix. The WM sees only context and candidate actions, plus this training
   prefix; no task label enters these logits.
3. Interpolate the frozen negative log probabilities on the neighboring FSQ
   grid vertices (8 neighbors for the three-dimensional FSQ).
4. At a quantized code, the forward value is exactly its categorical NLL,
   averaged per token. Backpropagate the interpolation through the existing
   FSQ straight-through estimator into the source encoder only.

Source objective: ranking + feature reconstruction + saturation + .1 * NLL.
The interpolation is a **local biased gradient surrogate**, not an exact
gradient through discrete choices, and it stops gradients through the prefix.
It avoids the particular attraction to a low-probability codeword mean; it
does not guarantee noncollapse or solve autoregressive exposure mismatch.
The upper edge uses the last interior grid cell. Unit tests cover equality
with categorical CE at all 256 codewords, encoder gradients at the boundary,
frozen-logit gradient isolation, and a bimodal counterexample to mean-code MSE.

The context-only prior is trained and logged, with no new beta/rate loss.
Finite code storage stays 128 nominal bits per candidate. The CE difference
between learned prior and WM is a predictive diagnostic, not a calibrated
measurement of true conditional mutual information.

## Stable final checkpoint

EMA supplies target codes during alternating WM updates. The ONLINE source
encoder, on which the reader was trained, is exported. Freeze that encoder,
regenerate the training codes, then run the final 2,000 WM/prior updates.
This avoids replacing the reader's encoder with a different EMA model at export.
Both arms follow the same rule. No reader adaptation occurs at this stage.

Fixed last checkpoint: no best-of-dev checkpoint selection, no lambda sweep,
no adaptive extra epochs. Store checkpoint before full offline evaluation and
mark the run failed if evaluation fails. Config, metrics JSONL, parent checkpoint
hash, source snapshot and state-load compatibility checks accompany outputs.

## Evaluation and interpretation

Training roots 30250–31049; offline dev 2000–2099. Root ranges are asserted.
Sealed roots 3000–3399 are untouched. Reuse the full 3,011-decision offline dev
ladder, including greedy, expected and four-sample readouts. Greedy remains the
primary method variant; expected and sampled variants remain secondary.

Paired root-bootstrap comparisons:

- method minus matched continuation, all ladder tiers;
- each continuation minus its exact Round-1 parent;
- predicted-code readers minus their contemporaneously trained DIRECT.

Keep diagnostics: code retention, feature reconstruction R², saturation loss,
perplexity, within-bank distinctness, prior/WM CE, and default-choice frequency.
Add before/after diagnostics on the fixed first 600 dev decisions:

- greedy code collisions on label-distinct sibling pairs;
- reader retention when each WM token uses the true source prefix.

The latter is explicitly privileged, mixes token predictions conditioned on
different true prefixes, and is neither a deployable arm nor a mathematical
upper bound. It is a directional exposure-mismatch diagnostic.

A fall of >0.05 in the method's source-code retained-gap point estimate vs
Round 1 is flagged as a retention concern, even if categorical CE improves.
Also inspect reconstruction and uncertainty in paired changes. No retrospective
definition of a win from whichever metric happens to improve.

After training, the next control experiment is both fixed arms on the same
100 closed-loop dev roots 2100–2199 (P0, CODE8, CTA8, DIRECT8; CTA8E secondary).
The training-only submission does not enqueue this larger rollout array.
No offline result counts as closed-loop success. The existing configuration
selection rule remains dev CTA8−P0, with offline greedy gap breaking ties.
The result must still be tested on sealed roots with three training seeds.

## Compute and reproducibility

Submit an immutable release under the shared results directory, then:

1. CPU unit tests and tiny continuations at both lambdas from the real parent:
   4 CPU, 48 GB, 30 min, no GPU. Tiny results are code-path checks only.
2. `afterok` training array tasks 0/1, max concurrency 1, one 3g.40gb MIG each,
   8 CPU, 128 GB, **3 h hard limit per arm** (at most 6 MIG GPU-h allocated).
3. `afterok` CPU paired comparison: 2 CPU, 8 GB, 15 min.

Dependencies are cancelled if they become invalid. Check both squeue and sacct
for duplicates before submission. No interactive allocation, foreground polling,
or idle GPU job. Record IDs in JOB_LEDGER.md and return after submission.

## Sources and novelty boundaries

- [FSQ](https://arxiv.org/abs/2309.15505): fixed scalar levels and implicit product
  codebook. Neither discrete quantization nor the straight-through estimator
  is CTA's novelty.
- [GCQ](https://arxiv.org/abs/2510.16039): action-conditioned spatiotemporal
  quantization already exists; segment coding alone is insufficient novelty.
- [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html): a relevant spatial
  feature world-model reference, not interchangeable with this policy/runtime.

The proposed Round-2 estimator is an implementation choice for this experiment,
not an attributed result from those papers. Conditional reader access, reusable
segment codes, query retention and predictability must earn their contribution
through matched controls. Conditional per-frame coding, path-vs-endpoint,
held-out goals/queries and total planning latency are still missing evidence.
