# CROD: Mining Cross-Representation Ordinal Disagreement for Physical Policy Distillation

**Status (2026-08-18):** falsification-first H0 is running. This document is
the idea-of-record for later sessions. It supersedes the failed
proxy-internal/flow-first direction as the candidate new-paper method; it does
not supersede the separate current ICLR cost-exploitation paper.

## One-sentence idea

When a deployed latent planner selects an action because its native
representation-induced cost ranks it best, use a genuinely different
representation to identify **rejected candidate actions that the native model
likely mis-ranked**, physically verify only those candidates, then distill the
verified corrective data into a simple search-free policy.

The proposed name is **Cross-Representation Ordinal Disagreement (CROD)**.
Do not call it “cross-view”: in world-model work that normally means camera
viewpoint transfer.

## The scientific question

Let a frozen native world model and planner produce an exact returned action
chunk `a0` and a planner-induced candidate population `A = {a1, ..., aN}` at a
single restored simulator state. Let `c*(a)` be the physical terminal task cost
under the same-state rollout. The native model can be locally wrong in the
decision-relevant sense when it rejects `ai` relative to `a0`, yet physics says
`ai` is materially better:

```text
native LeWM:     a0 ≻L ai
physical world:  ai ≻* a0
```

The question is not whether two models disagree in general. It is:

> Can a second, independently formed representation enrich these **native
> planner-proxy inversions** more efficiently than action-space diversity under
> a fixed physical-query budget?

This is a concrete acquisition hypothesis. It is falsifiable before training a
policy, calibrator, or flow model.

## Method

### Native and auxiliary views

- **Native/deployed view L:** frozen LeWM plus its original CEM planner and
  terminal latent goal cost. `a0` is the exact `CEMSolver.solve(...)["actions"]`
  returned action chunk, not the lowest-cost persisted CEM sample.
- **Auxiliary view D:** frozen DINOv2-Small visual encoder plus an action-only
  DINO-WM predictor, trained on the same OGBench-Cube demonstrations. It never
  proposes actions in H0: it only scores the native planner's candidate set.

LeWM is learned end-to-end from pixels; DINO-WM uses a frozen external visual
representation. This makes the pair a cleaner representation-diversity test
than two world models that share DINOv2. It does **not** imply that their errors
are independent: error complementarity must be measured.

### Directional ordinal score

For each native CEM candidate `ai`, compute the lower-is-better ranks jointly
with the returned anchor under both costs. CROD is

```text
S(i) = [ rL(ai) - rL(a0) ]+ × [ rD(a0) - rD(ai) ]+ .
```

It is positive exactly when:

```text
LeWM rejects ai relative to a0, and DINO-WM prefers ai relative to a0.
```

This is deliberately directional. Symmetric disagreement such as
`|rL - rD|` also becomes large when the auxiliary is wrong and LeWM is right;
it does not target errors of the deployed proxy.

### Physical verification and distillation

At each state, lock all selections before looking at simulator outcomes. Query
the returned anchor plus a small number of alternatives. A candidate is a
**corrective inversion** if LeWM rejected it and it improves physical terminal
distance by at least `δ = 2 cm` relative to the exact anchor.

If CROD passes H0, collect equal-budget verified corrective transitions and
first train a simple behavior-cloning policy:

```text
BC
BC + random counterfactual data
BC + action-diverse counterfactual data
BC + CROD-selected counterfactual data
```

All arms must match snapshots, physical interactions, selected examples,
updates, and architecture. Flow/OFP is explicitly deferred until CROD beats
diversity with this simple policy.

## Why this could be a paper contribution

The contribution is **not** “use an ensemble/disagreement for active data
selection.” That is established territory in uncertainty, active learning,
reward-model ensembles, and active policy fine-tuning.

The potentially novel claim is narrower:

> Directional ordinal disagreement on a deployed planner-induced candidate set
> can identify and repair physical misranking of one particular representation
> proxy with a fixed, small verification budget.

The full paper story, if supported, is:

```text
native planner ranking
        ≠
independent predictive ranking
        ↓
directional candidate acquisition
        ↓
same-state physical verification of native-proxy inversion
        ↓
more useful corrective data than diversity
        ↓
search-free policy distillation
```

Use the careful literature claim: **“we did not find a direct match for this
full pipeline.”** Do not claim absolute first work without a dedicated final
literature audit.

## Evidence already available

The prior LeWM OGBench-Cube diagnosis establishes that the substrate is worth
testing, but it is not CROD evidence:

- Phase 0d evaluated the exact CEM-returned plan under complete same-state
  resets. Its mean physical selection regret was 7.05 cm (95% CI
  `[4.24, 10.12]` cm), and 50% of snapshots had a proxy-rejected corrective
  alternative at the locked 2 cm margin.
- Phase 1a and the fresh Phase 1a-v2 replication found no robust
  proxy-internal signal beyond action diversity. This rules out the claim that
  LeWM confidence/instability alone tells us when LeWM is wrong.

These results motivate an external representation signal. They do not show
that DINO-WM is complementary or that CROD will work.

## H0: preregistered falsification test

Full protocol: `PROTOCOL.md`.

### Fresh sample-efficiency endpoint

- 128 OGBench-Cube snapshots, episode-disjoint from all Phase 0d, Phase 1a,
  and Phase 1a-v2 episodes.
- Native CEM: 300 samples, 30 updates, top-30, horizon 5, action block 5.
- Each arm is charged `1 + 8` physical branches: exact returned anchor plus
  eight alternatives. Overlap across arms may reuse a deterministic rollout
  computationally, but never changes logical budget.
- Primary arm: `crod_directional`—top directional scores among
  native-rejected candidates.
- Primary baseline: `action_diverse` over the native final population, because
  it was the strongest existing selector.
- Other controls: rejected-support diversity, DINO-best among rejected
  candidates, prior native instability, and support-matched random.
- Primary endpoint: **corrective hit rate**, i.e. whether at least one of eight
  alternatives is a 2 cm physical correction of the returned anchor.
- Secondary endpoint: best physical improvement per query.

### H0 gate

Authorize only the simple H1 BC pilot if both are true:

1. paired snapshot-bootstrap 95% CI lower bound of
   `CROD hit rate − action-diverse hit rate` is above zero; and
2. the point contrast in best corrective physical improvement is positive.

If the point estimate is positive but the CI crosses zero, report `HOLD` and
replicate/diagnose; do not start flow. If the point estimate is non-positive,
report `STOP_CROD_NO_GAIN_OVER_ACTION_DIVERSITY` and abandon this method
direction rather than adding more views or losses.

### Separate complementarity mechanism audit

The 32 Phase-0d populations were already fully physics-labelled. Rescore their
existing final candidates with DINO-WM, without using these states for the
fresh endpoint. At a 2 cm physical tie band, report:

```text
P(LeWM ranking wrong)
P(LeWM ranking wrong | LeWM and DINO-WM disagree)
P(LeWM wrong and DINO-WM wrong)
P(corrective | native rejected)
P(corrective | native rejected and DINO-WM prefers candidate)
```

This asks whether disagreement is a measured error-complementarity signal, not
just a different latent geometry.

## Deliberate exclusions from H0

- No calibrator, auxiliary ranking loss, or policy training.
- No flow/OFP/diffusion policy.
- No third view (depth, object geometry, or privileged state); it would confound
  representation disagreement with additional privileged information.
- No claimed homogeneous LeWM ensemble baseline: a matched independent LeWM
  seed checkpoint is unavailable. Native jitter/instability is only a
  single-model control and must not be called an ensemble.
- No reused fresh-endpoint episodes from earlier phases.

## Current implementation and execution state

All CROD work is isolated in `crod_h0/`.

- `core.py`: score, rank, and outcome-blind selectors.
- `scripts/run_h0.py`: native CEM, auxiliary rescoring, locked selection, and
  same-state physics execution.
- `scripts/rescore_phase0d_audit.py`: post-hoc complementarity audit on the
  existing fully-labelled populations.
- `scripts/analyze_h0.py` and `scripts/analyze_complementarity.py`: clustered
  bootstrap gates and reports.
- `JOB_LEDGER.md`: exact commands, dependencies, outputs, and job IDs.

At document creation, manifest job `42385` has completed, DINO-WM training job
`42386` is running, and the smoke, complementarity, fresh-H0 array, and
aggregation jobs are dependency-chained. Do not interpret any result until the
checkpoint smoke and exact-anchor replay gate both pass.

## Related-work positioning to preserve

- DINO-WM: frozen pretrained visual representation plus predictive world model;
  CROD uses it as an auxiliary ranker, not as a new planner.
- Active policy fine-tuning and generic disagreement: relevant novelty threats,
  but they do not by themselves establish native planner-proxy inversion under
  same-state physical verification.
- Prediction-error mining / representation repair / planning-cost ranking:
  closest conceptual neighbors, but CROD targets selection from the native
  planner's actual candidate set before any retraining.
- Execution-time verifier methods: distinguish clearly; CROD acts before
  executing the selected action and spends a limited physical verification
  budget on alternatives.

Primary-source links and the precise review corrections are in
`REVIEW_ASSESSMENT.md`.
