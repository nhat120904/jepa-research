# Reader refinement: train on deployed codes, evaluate against policy

2026-09-26. User requests continued practical refinement until CTA improves
learned control. This is one bounded intervention, not a guaranteed positive
result or a new qualification gate. Oracle selectors do not enter evaluation.

## Fixed experiment

Parent: geometry checkpoint `cta_geometry_e2e_55018/train/cta.pt`, SHA256
`b777f1588e9969daab2fc80eebe3ce12a44e36759bddade51fc20645e1f29f5f`.
Frozen components: source encoder/codebook, WM, prior, decoder, FULL and DIRECT.
Policy, K=8, execute8, simulator and registration target remain unchanged.
The reader still receives only context, code and query; no raw action bypass.

The implementation is `scripts/cta_reader_refine.py`. Train roots 30250–31049,
offline dev 2000–2099; closed-loop dev 2100–2199. These development roots have
been reused; they are not a final held-out test. No sealed roots are opened.

Two readers start from identical parent weights:

- **Control:** continue reader ranking training on source codes only.
- **Method:** mix source / WM greedy / WM expected codes with probabilities
  .50 / .25 / .25. Each entire bank uses one evidence type so score comparisons
  never mix source quality across siblings within a bank.

Both readers get 3000 AdamW updates, LR 3e-5, weight decay from parent, batch
16 banks, identical bank indices, goal samples and labels. Train banks have
geometry label spread above the existing 1e-3 margin. This differs from the
previous all-phase codec training but is identical between the two reader arms.
The matched control tests predicted-code exposure versus additional training.
Frozen source/greedy/expected codes are cached once on the training banks.
Expected code uses the current WM's greedy prefixes, as in existing deployment;
it is not the exact posterior expectation over all possible code sequences.

Only reader parameters receive gradients. Frozen-state equality is checked.
Checkpoints are saved every 1000 steps; final step is used, not dev-best selection.
Offline evaluation follows training and never gates closed-loop on scientific
scores. Existing runtime consistency checks cover greedy and now expected-code
for base/control/method before dependent evaluation. Numerical/nonfinite errors,
bad identity or inference mismatches are correctness failures, not method verdicts.

## Closed-loop comparison

Run eight arms on all 100 roots, ten independent 10-root shards:

| Output name | Selector |
|---|---|
| P0 | Original policy candidate 0 |
| DIRECT8 | Unchanged learned direct scorer |
| BASE_G / BASE_E | Original reader, greedy / expected code |
| CTRL_G / CTRL_E | Source-only continuation reader |
| ADAPT_G / ADAPT_E | Mixed-code adaptation reader |

All learned selectors score from context and proposed actions before simulation.
Only the selected branch is simulated (`log_candidates=False`); no unchosen
future is accessed to decide. This saves diagnostic simulation cost. Reset roots
and candidate RNG schedule are matched; visited states diverge after different
choices, so banks are not claimed identical after policies diverge.

Primary outcome: ADAPT_G minus P0 native episode success. Secondary: ADAPT_E
minus P0. Also report same-decoder adaptation minus control/base and both adapted
arms versus DIRECT. Report every arm, root-paired bootstrap intervals, exact
McNemar, mean maximum coverage and its paired differences. Do not pick a winning
decoder post hoc and call that the primary result. 100 roots are development
evidence, not guaranteed power for a 4–5 pp gain.

Root records are flushed after each complete eight-arm root. Aggregation rejects
missing/duplicate roots, missing arms and checkpoint drift. Log choices, ties and
override frequency on each arm's own visited states. Immediate unchosen physical
gains are unavailable because unchosen candidates are intentionally not simulated.

## Bounded compute and implementation checks

One training batch job: MIG H100 3g.40gb, 8 CPU, 96 GB, 2-hour cap; unit suites,
cache, matched training, offline eval and runtime checks in the same job.
Dependent closed-loop array: ten tasks, maximum concurrency 2, one MIG GPU,
4 CPU, 48 GB, 1-hour cap each. CPU aggregation: 2 CPU, 8 GB, 10-minute cap.
Allocation ceiling is **12 MIG GPU-hours**; expected usage is lower based on
55052, which ran eight arms including expensive oracle diagnostics in 36m30s
per ten roots. There are no speculative arrays or scientific score gates.
Dependencies cancel if training/evaluation fails. All source and outputs have
unique snapshots; no old checkpoint or collection is overwritten.

Before submission: syntax/whitespace checks on login. Numerical tests execute
inside the training batch. They cover whole-bank code mixing, paired aggregation
identity and NumPy report serialization as well as existing CTA/runtime suites.

## Interpretation and subsequent repair

The concrete hypothesis is a mismatch between source codes used to fit the
reader and predicted/soft codes encountered at deployment. A gain must beat the
source-only continuation control to support adaptation specifically. If it helps
offline but not control, that remains a failed control intervention.

The reader cannot separate actions mapped to identical codes. Adaptation does
not solve greedy collisions. If this remains the binding failure, the next
intervention should change WM prediction/decoding while anchoring the source
code, rather than raising co-design lambda that already collapsed code usage.
Task labels still supervise the reader; the whole pipeline is not claimed fully
self-supervised. No promise of a CVPR acceptance or statistically positive result.
