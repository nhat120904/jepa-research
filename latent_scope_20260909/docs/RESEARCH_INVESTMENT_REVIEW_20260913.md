# Research investment review after job 52399

Date: 13 September 2026. Scope: review, literature positioning, static implementation audit, and resource arithmetic. No simulator/model execution, training, new Slurm submission, or implementation change was performed for this review. Existing dirty files were preserved.

## Decision

Retain a narrower research question, but downgrade the current method from a selected investment to a conditional hypothesis. A bounded qualification effort is justified; a full six-arm training campaign, further threshold-chasing, or an 80-prefix oracle campaign is not justified yet.

The defensible question is:

> At a fixed observation, data and inference budget, can an action-conditioned visual segment representation preserve accumulated interaction effects under concatenation, and improve held-out candidate selection beyond ordered frame prediction, unstructured segment prediction, simple event monitoring, and additive transition representations?

For a composition-specific contribution, require a benefit on held-out segmentations or durations, as well as useful action selection. A benefit confined to in-distribution auxiliary prediction is a narrower regularization finding. Neither positive headroom nor a better progress probe demonstrates composition by itself.

The existing research question is coherent and falsifiable. Novelty is conditional, evidence for the method is absent, the arena is a weak mechanism test as currently sampled, and Stage C still has serious comparison defects. This is worth one focused qualification tranche, not an open-ended commitment based on sunk engineering effort.

## What the experiments establish

Slurm was checked through both squeue and sacct. Jobs 52208, 52226, 52284 and 52399 completed; 52279, 52282, 52394 and 52397 failed before candidate evaluation. There were no active user jobs at inspection.

The latest [52399 artifact](/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/stage_b_horizon_52399/stage_b_profile_result.json) contains two independent prefixes, with the same eight candidate banks evaluated at two horizons:

| Task | H=8 success | H=16 success | Baseline candidate 0 | Oracle improvement over candidate 0 |
|---|---:|---:|---|---:|
| ScrubCuttingBoard | 7/8 | 8/8 | succeeds at both horizons | 0 pp |
| RinseSinkBasin | 8/8 | 8/8 | succeeds at both horizons | 0 pp |

All branches attained the maximum recorded progress score. Thus the failed Scrub branch demonstrates that this scalar progress score is not sufficient for eventual completion: it omits the final release/separation requirement. That is an observed limitation of this score, not evidence that a richer learned segment target is necessary.

There is observed binary outcome variation at one Scrub prefix. Calling this definitive causal variation is premature until same-action suffix repeatability is checked for the new direct-restore path. It is also not positive gain against the locked candidate-0 baseline.

As a post-hoc descriptive calculation only, uniform choice among the eight H=8 candidates has success 87.5% on Scrub and 100% on Rinse; their pooled oracle-minus-uniform gap is 6.25 pp. H=16 has zero such gap. Do not promote this alternative baseline into the primary result, and do not treat 32 branches as 32 independent contexts.

The earlier B0 progress difference corresponds to one Rinse candidate reaching two washed regions instead of one. The previously quoted pooled +8.3 pp is an average difference in a normalized milestone score across four prefixes. It is not a task-success improvement, a replicated estimate, or evidence that composition helped.

## Why the current headroom test is weak

### Retrospective prefix selection

[run_stage_b_profile.py:276](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/scripts/run_stage_b_profile.py:276) requires eventual source success, then chooses `events[-1]`. This preferentially samples states immediately before the final progress event on successful trajectories. It is a useful engineering anchor but has a ceiling bias for evaluating recovery or selection. The latest prefixes already have 4 accepted Scrub contacts plus sweep, or 2/3 washed regions.

Select the next panel using only past/current observations or native diagnostic history at a predeclared time: e.g. first incomplete contact/coverage milestone before a deadline. Include rollouts whether their later outcome is success or failure. Record non-reached strata and sampling weights; do not discard difficult prefixes after seeing branch outcomes. Native history may stratify a diagnostic panel, but it must not become an undeclared model input or a deployable oracle trigger.

### State equality is not transition equivalence

[reset_to_event:155](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/scripts/run_stage_b_profile.py:155) reloads XML, a flattened simulator state and selected task-history fields, resets the environment RNG from its initial seed, and adjusts the wrapper horizon. It does not explicitly restore the source controller state, warm starts, buffers, current RNG state or task clock. The older [capture_snapshot:199](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/scripts/run_stage_a_preflight.py:199) recognizes these additional fields.

This does not prove that any omitted field changes the current result. It means the result has passed initial-state alignment, not the stronger transition-equivalence test. The old A4 repeat test rebuilt and replayed a prefix; it does not validate the new direct restore automatically.

Same source seed also produced different collection outcomes across retries: Scrub seed 63000 failed in 52394/52397 but succeeded in 52399. Seed declarations do not establish reproducibility. Persist selected source snapshots and exact candidate arrays; compare repeated identical suffixes on those artifacts. Local renderer differences have not been causally explained, so 'harmless rendering noise' is not an established diagnosis.

### One continuation seed and one decision

Single-rollout maximum labels estimate an optimistic availability ceiling. For positive evidence, select on one continuation seed set and evaluate the chosen candidate and candidate 0 on disjoint common seeds. Identical RNG seeds across branches are a useful coupling but do not prove simulator determinism.

This gate is specific to one intervention followed by frozen GR00T. A null result does not refute repeated MPC or world models generally. A positive result does not yet demonstrate closed-loop improvement.

## Novelty audit

The following are primary sources checked during this review. Their reported results were not reproduced. This is a targeted mechanism audit, not exhaustive novelty clearance; no acceptance claim is inferred from an arXiv listing.

| Work | Overlap | Remaining distinction to demonstrate |
|---|---|---|
| [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | Predicts spatial visual features for planning | A segment target must beat a strong full temporal readout, not merely goal-image distance |
| [TD-JEPA](https://arxiv.org/abs/2510.00739) | Policy-conditioned long-term latent prediction using TD learning | Explicit supplied-action sequence effects and path summaries are a different interface |
| [CompPlan](https://arxiv.org/html/2602.19634v1) | Multi-timescale occupancy prediction, horizon consistency, composition of policies | Path-effect summaries are distinct from policy occupancy models; generic multi-horizon consistency is not new |
| [AC-LAM](https://arxiv.org/html/2604.03340v1) | Short-horizon additive latent actions, identity/inverse/cycle structure | Accumulated effects that cannot be represented by net displacement or additive cancellation |
| [ALAM](https://arxiv.org/abs/2605.10819) | Reconstruction-grounded latent transitions with composition/reversal, transferred to VLA training | Supplied-action counterfactual planning and a representation of the interior path must earn the difference |
| [DLAM](https://arxiv.org/html/2607.27138v1) | Distributional transitions with composition/reversal constraints | Its frame-pair transition targets differ from full-path summaries; adding composition or variance alone is occupied territory |
| [hint²](https://arxiv.org/abs/2608.13678) | Action-induced proposition prediction for temporal-logic policy guidance | Generic event prediction for guiding a chunked policy is not an exclusive claim |
| [Signature control, L4DC 2024](https://proceedings.mlr.press/v242/ohnishi24a.html) | Composable path representations and control | Learned visual sufficiency and empirical control gains must supply the contribution |

CompPlan's Section 4.4 is particularly relevant: its authors report improved long-horizon prediction but relatively modest planning improvement from consistency at their evaluated horizons. This directly motivates checking the causal contribution of consistency rather than assuming prediction improvement transfers to control.

A useful distinguishing hypothesis concerns irreversibility: returning a faucet to its starting pose need not undo which regions were washed. However, a three-bit mask and union already express this for Rinse. Therefore irreversibility alone is not a reason for a learned Transformer composer. Demonstrate why the simple sufficient-statistic/monitor baseline is inadequate under permitted visual inputs or a constrained representation budget. Scrub's distinct-contact filtering also requires overlap handling; summing segment counts is not an exact native monitor. Do not rely on evaluator quirks as the paper's contribution.

## Stage C cannot currently settle the hypothesis

Static findings, not numerical tests:

| Issue | Evidence | Required correction before a method comparison |
|---|---|---|
| Candidate evaluation may use later windows | [data.py:73](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/data.py:73), [run_stage_c_train.py:121](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/scripts/run_stage_c_train.py:121) | Exactly one decision window per candidate; same pre-action context; reject duplicate/missing candidates and cross-split sources |
| Direct value ignores action order | [models.py:341](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/models.py:341) | Temporal positions or ordered action encoder; current position-free Transformer plus mean is permutation-invariant in evaluation |
| Frame baseline reduces sequence to endpoint and mean | [models.py:205](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/models.py:205) | Ordered full-sequence readout with matched capacity and history |
| Composer does not identify left/right | [models.py:148](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/models.py:148) | Specify whether the intended operator is ordered or a commutative coverage merge; a boundary token without segment positions does not label the sides |
| Additional endpoint supervision is confounded with composition | [models.py:328](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/models.py:328) | Train all relevant controls on the same full/subsegment targets; vary the composition mechanism alone |
| Target encoder is not directly trained by composition | [models.py:292](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/models.py:292) | Target outputs are detached; online target is trained to reconstruct mean future features. State this limitation and validate target sufficiency; a constant/mean shortcut must be excluded |
| Scoring uses direct summary, not composed summary | [models.py:300](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/models.py:300) | Test composed inference explicitly if claiming compositional planning; current interface tests a regularizer |
| Only three history frames; invented memory differs from observed memory | [models.py:68](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/models.py:68), [models.py:312](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/stage_c/models.py:312) | Shared defined causal-memory update and equal history access; standalone GRUCell for imagination is not the same as the observed GRU |
| CLS-only encoder replaces proposed patch features | [encode_stage_c_offline.py:69](/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/scripts/encode_stage_c_offline.py:69) | Probe spatial feature adequacy before large encoding/training; do not call CLS an exact DINO-WM reproduction |

The six-arm smoke test establishes finite forward/backward plumbing only. A three-arm initial comparison can be more decisive: matched unstructured multi-horizon segment model, composition variant, and ordered direct-value control. An ordered frame-sequence readout is also required to qualify the trajectory target. At least one comparison must separate composition from the extra multi-horizon targets; identical nominal parameter counts alone are insufficient.

## Resource calculations

All figures are allocated single-GPU walltime, not GPU utilization or monetary cost. The eight Stage-B jobs listed above used **7.7689 GPU-hours** in total. This excludes Stage A, baseline setup/evaluation and Stage C smoke jobs.

For job 52399, the recorded branch timers sum to:

- H=8: 2,826.43 seconds across 16 branches (Scrub mean 161.46 s; Rinse mean 191.84 s).
- H=16: 2,911.84 seconds across 16 branches.
- Remaining profile time: 1,676.39 seconds, including source generation, reconstruction outside timers, startup and bookkeeping.

The artifact's `82.3851 h = 40 × 7414.66 / 3600` extrapolates **both horizons on both tasks**, with one continuation seed. It must not be described as the exact cost of the original H=8-only B1.

Removing just the measured H=16 timers while retaining all remaining overhead gives an H=8-only proxy of **50.03 GPU-hours** for 40 prefixes/task. The measured branch-only part is **31.40 GPU-hours**. Both estimates are conditional on these late prefixes; earlier anchors have longer tails, and multiple continuation seeds add work. Simply halving 82.39 ignores fixed overhead.

For eight Scrub prefixes at H=8, measured branch-only cost is **2.87 h** for one seed or **5.74 h** for two. Pooled per-prefix overhead suggests about **5 h** for a one-seed eight-prefix panel, with substantial uncertainty and likely extra cost for earlier anchors. These are budgeting proxies, not promises.

Running the same simulations on four GPUs approximately quarters walltime but does not automatically reduce allocated GPU-hours. Shared inference batching, simulator concurrency on one allocation, fewer unnecessary full tails, and stopping after genuinely latched success can improve throughput; each needs measurement and semantic validation. Maximum-progress completion is not sufficient to stop Scrub.

## Statistical limits

Let a recoverable prefix mean candidate 0 fails and another candidate succeeds under a fixed deterministic continuation. With zero such prefixes among n independent samples, the one-sided exact 95% binomial upper bound is `1 - 0.05^(1/n)`:

| n | Upper bound after zero recoverable prefixes |
|---:|---:|
| 2 | 77.6% |
| 12 | 22.1% |
| 29 | 9.81% |
| 40 | 7.22% |

These formulas illustrate screening precision; they do not validate the retrospectively selected current sample or account for noisy oracle selection. Twelve new contexts cannot establish a tight negative result. Conversely, 32 candidates at two contexts would not solve the sample-size problem. For noisy continuation values use independent scoring/evaluation seeds and paired episode-level inference.

The existing 200-episode confirmation suggestion is not a universal power guarantee. For a 5 pp paired method gain with discordance probability 0.20, an illustrative normal approximation gives roughly 620 pairs for 80% power at two-sided alpha 0.05; the appropriate count depends on measured discordance and model-seed variation. Three training seeds remain separate replications.

## Concrete next investment

Recommend one additional qualification tranche with a **hard cap of 8 allocated GPU-hours**, plus bounded CPU diagnostics, and roughly 2-3 focused engineering days as a planning allowance. This is a proposed new cap, not a submitted job or an inferred remaining allocation. Stop within the cap even if sample targets are incomplete, and report the achieved panel.

1. **Make the experiment replayable and the comparison interpretable.** Persist source/controller/RNG state and action banks; test source versus restore on an identical 8/16-step suffix, plus repeated identical actions across restored carriers. Verify native progress updates and success calls do not mutate history extra times. Freeze the protocol version. Repair Stage-C decision-window and matched-baseline invariants before any learned comparison.
2. **Use Scrub as the primary diagnostic; retain Rinse as a simple-monitor control.** At H=8 and N=8, target eight new independent prefixes sampled before terminal saturation, with causal predeclared sampling. Use one continuation scoring seed initially. Budget up to about 5 GPU-hours based on the proxy above, and stop if early-prefix costs prevent fitting. Record immediate native events and separation as well as terminal success.
3. **Spend the remaining budget on confirmation of what was learned.** If scoring yields useful selection, freeze those selections and evaluate chosen versus candidate 0 under two independent common seeds per prefix. This evaluates conditional pilot selections only; any deployment claim still needs a new unbiased panel. Reserve a small random candidate/seed repeat set to quantify stochasticity even if no gain appears. Do not switch candidate 0 after observing labels.
4. **Qualify the target before a world-model campaign.** On separate training/held-out windows, compare readouts from causal history+endpoint, ordered true sequence, simple native-event predictions, and compressed target. Query native progress dimensions and eventual success separately. If useful visual signal is present but the compressed target loses it, allow one documented target correction. Probe selection under candidate grouping, not just global classification accuracy.
5. **Require the composition-specific experiment to earn its next budget.** On a qualified dataset, compare composition against identical multi-horizon supervision without composition, with ordered direct-value/full-sequence controls. Evaluate direct and composed predictions on held-out partitions or durations; keep task success as the later control endpoint. No six-arm sweep until these comparisons are valid.

A positive pilot is permission to investigate, not certification of the +10 pp gate. If the unbiased pilot remains saturated or hopeless and the mechanism/readout checks reveal no trajectory-specific advantage, pause this RoboCasa policy-task pairing. If uncertainty is merely large, call it inconclusive; the budget decision can still be to stop investing.

Only then consider qualifying one cheaper native task. [ManiSkill DrawTriangle](https://raw.githubusercontent.com/mani-skill/ManiSkill/main/mani_skill/envs/tasks/drawing/draw_triangle.py) has a 300-step horizon and a visible accumulated trace. It is a possible compression/partition pilot, but a strong endpoint image can already expose the path and it is not detailed contact physics. Verify render compatibility, snapshot/evaluator behavior, data availability and actual throughput before installing a new training campaign. A shorter nominal horizon is not evidence of lower engineering cost.

The present investment case rests on acquiring a discriminating mechanism result cheaply. It does not rest on proving a positive headroom number at any cost, repeating failed gates until they pass, or treating an infrastructure repair as evidence for the method.
