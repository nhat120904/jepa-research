# Experiment contract: qualify policy, scorer, data, then abstraction

Design date 2026-09-22. No measurements or resource availability are claimed below.
Numeric thresholds are proposed predeclared operational gates, not known results.

Implementation addendum: `IMPLEMENTATION_PLAN.md` records native action alignment,
nested-bank sensitivity, policy-history controls and current execution boundaries.
It takes precedence where this original design-only contract differs.

## 1. Arena choice

Primary candidate: **native gym-pusht with frozen lerobot/diffusion_pusht**. The task
is standard PushT goal overlap. This has a released action-chunk policy, native
environment, configuration and public evaluation reference. It avoids training a VLA.

The [model card](https://huggingface.co/lerobot/diffusion_pusht/raw/main/README.md)
reports 65.4% success across 500 episodes and .955 average maximum overlap, with
success defined at >=95% overlap. It identifies the 175k-step checkpoint and the
LeRobot training commit. These are author-reported baseline numbers, not a result
in this workspace and not evidence that picking among samples can improve success.

This contract differs from the previous original DINO-WM PushT audit: policy proposals,
overlap success, action units, controller/runtime, cadence and horizons differ.
Document every difference. Never compare its 65.4% to original DINO-WM's goal-reaching
success as if the benchmark were identical. The old audit's negative result remains valid.

Second environment is not selected to rescue a negative primary result. After a
primary method signal, independently reproduce one released Diffusion Policy
Robomimic manipulation task for external validity. The exact task/config must be
locked based on available releases and feasibility, before viewing method gains.
Source release: [Diffusion Policy, RSS 2023](https://github.com/real-stanford/diffusion_policy).

## 2. Action generation and scoring

Freeze a single released policy and its normalization/denoising settings. K=8
independent draws at the native sampling temperature; candidate 0 is the untouched
policy's first draw. Keep a manifest of random seeds, policy revision and action arrays.
Do not degrade the policy or broaden sampling until an oracle starts winning.

Use the released action horizon and executed-prefix cadence, read from the pinned
config. Do not assume the 16/8 values commonly used by diffusion policies until the
actual release is inspected. WM horizon initially matches that candidate horizon.
Longer horizons are a separate extrapolation panel, not an initial control requirement.

At each control decision:

1. Observe history and compute shared C once.
2. Draw the same eight chunks from the frozen proposal policy for all scorers.
3. For each chunk, predict its conditional segment code; read the task's image-query
   score from C,S,q. Average answers if using multiple code samples.
4. Select the highest-scoring candidate; exact ties select candidate 0. Also report
   uniform-tie expectation for offline analysis.
5. Execute the SAME native prefix length as policy-only baseline and replan.

No CEM refitting over raw action space in the first experiment. The action distribution
is still sampled, but by a competent fixed policy. The WM is a planner component
evaluating consequences; this is not actor-critic training. Policy success and proposal
support are separately measured, because policy competence alone does not imply oracle
selection headroom. The proposed WM does not improve actions absent from the bank.

Task score: a fixed image-goal query over predicted future feature evidence at the
policy's native horizon. Its exact kernel and temporal aggregation are pinned BEFORE
qualification. Simulator overlap/reward is used only for evaluation/oracles and is
not fed to a learned scorer. If this visual score cannot choose physically useful
chunks when supplied actual future images, halt this control contract.

## 3. Four qualification checks before method training

### A. Runtime/policy reproduction

Pin policy/environment/library versions; inspect cameras, action units, observation
history, normalizer, termination and executed cadence. Reproduce baseline on 100
root seeds, extending to 200 only under a predeclared uncertainty rule. The reported
65.4% should be statistically compatible, with a practical deviation bound set before
the run (propose <=10 pp at the initial screen). No tuning the policy on these seeds.
Verify full physics/controller/RNG snapshot restore with repeated action chunks;
pose-only reset is insufficient. A discrepancy is a measurement blocker.

### B. Genuine selection support

For each root, sample a scheduled anchor across the trajectory independently of
success/failure; include both failed and successful roots. Candidate futures are true
branches from the exact snapshot. Keep all branches from a root in one split.
Compare policy default, physical-score best in bank, and actual-future visual scorer.
No feasible witness is inserted into the ordinary bank.

For prospective deployment headroom, run a separate closed-loop reference where
the physical scorer selects every native chunk, using the same K/horizon/cadence.
Target oracle-over-policy improvement >=10 pp with paired episode CI excluding zero
on up to 200 qualification roots. This is an empirical oracle under a given score,
not a global upper bound; its failure may implicate score/horizon/proposals.
One failed qualification stops this arena contract. No sweep across temperatures,
checkpoints or tasks until headroom appears.

### C. Goal/query alignment

On the fixed branches, actual-future visual ranking should retain at least 80% of
the physical oracle's improvement over default, with uncertainty reported. Then
check its repeated-selection success. An offline retained-gap ratio is unstable when
the physical gap is tiny: require B first and report raw differences as well.
If visual scoring fails, improving latent prediction is not an adequate remedy.

### D. Data and adequate model baseline

Train/reference data must match this policy's action support and observation interface.
Use >=2,000 independent TRAIN root trajectories (one main anchor/root, K=8 branches),
500 separate development roots and 500 sealed confirmation roots as the target dataset.
These are design targets, not a sample-size guarantee. All starts/warm-ups and branches
belong to their root split. No descendant-frame split or random overlapping windows.
Qualification roots are never recycled as untouched test roots.

There are two levels of baseline evidence:

- Original DINO-WM source/checkpoint on its native `pusht_noise` contract is an
  implementation reference; existing order-jepa provenance can be read/reused.
- A DINO-WM-style full spatial predictor must be trained/evaluated on the NEW gym-pusht
  interface with identical data and budgets to the abstraction. This is an adaptation,
  not a direct checkpoint reproduction. Do not silently convert absolute actions to
  a relative-action checkpoint with incompatible controller state or normalization.

If no action-aware frame/direct baseline generalizes native-horizon query ranking,
the method test remains unqualified. Allow one predeclared learning curve at 500,
1,000 and 2,000 independent roots, then stop the data/model contract if inadequate.
The user should not be left in an unlimited sequence of 'baseline repair' jobs.

Data-scale rationale, not a transferred result: the original
[DINO-WM data ablation](https://arxiv.org/html/2411.04983v2#A4.SS1) reports PushT
success .08/.48/.72/.88/.92 for 200/1,000/5,000/10,000/18,500 trajectories. This
demonstrates sensitivity to data in that recipe; it does not predict our new curve.
Twenty-four roots were a pipeline pilot, not a credible capability ceiling.

## 4. Method experiment after qualification

Use three fixed nominal code budgets M={8,16,32}, V=256, and report expanded tensor
memory/context/decoder costs. Initial source codec and predictor are modest Transformers.
Choose schedule/learning-rate using development once, then lock it for confirmation.

Required comparisons in two controlled groups:

1. Same representation/decoder budget: conditional segment code with predictability
   term; lambda=0; no history side information at the reader; conditional compact
   per-frame code with equal TOTAL bits; uniform temporal pooling.
2. System alternatives: capable full-frame spatial WM; cached direct-query predictor;
   additive/multi-bin successor-feature prediction; proposal policy alone; no-action.

GCQ is additional sequence-quantization prior art. Compare its applicable released
sequence-coding mechanism or document an interface mismatch; do not present a bespoke
weak approximation as an official reproduction. The matched per-frame/context and
predictability ablations remain mandatory even if a full GCQ port is impractical.

Do not call standard Mamba/GRU an adequate full-frame WM without training/provenance.
Do not add learned composition until a segment method effect survives these controls.
Future query evaluation includes unseen image anchors, pairs and time windows,
native and longer horizons, and endpoint-matched paths where interior order differs.
Report natural prevalence plus a separate balanced mechanism panel. The latter does
not replace standard task success and must not become an engineered primary arena.

## 5. Statistical and compute decisions

Initial 48-hour deliverable is qualification and a resource profile, not a promise
of a paper result. Proposed initial cap: 8 GPU-hours and 32 CPU-hours total; every
job uses a smaller explicit Slurm limit and exits cleanly. Data/model scale is gated
by measured throughput and disk usage; don't silently pool spatial grids to fit disk.
After qualification, plan a 10--14-day method pilot with a separate declared compute
budget based on the profile. No huge model/array is submitted by this design document.

Gate sequence:

- A--C fail: stop control contract before abstraction training. Do not change the
  kernel or policy using failed qualification outcomes and label it a confirmation.
- D fails after the fixed learning curve: stop this arena/data contract.
- Observed conditional codec cannot improve the rate/query-distortion frontier over
  conditional per-frame coding: stop the segment representation claim.
- Codec works but action predictor fails at native horizon: stop the forecasting
  claim after the predeclared schedule, without an open-ended target redesign.
- Method works but conditional per-frame/direct controls match the system frontier:
  report no method advantage and stop this hypothesis.
- A development advantage triggers fresh-root, 3-training-seed confirmation; no
  re-selection of budget/primary metric after opening the confirmation set.

For an efficiency primary, predeclare >=2x predictor+reader speed and success
noninferiority margin 2 pp versus frame WM; also require measured end-to-end savings
after policy inference and encoding. A 500-episode test may be insufficient to resolve
2 pp: estimate paired discordance on development and size confirmation prospectively.
If required sample size exceeds budget, call the result inconclusive rather than
equivalence. For improvement over policy, propose >=5 pp absolute success with a
paired episode CI excluding zero, and show the same-bank frame/direct comparisons.
All intervals cluster by root episode; repeated candidates are not independent.

## 6. Current action boundary

This turn produces the design and qualification contract. It does not claim that
PushT+policy already has headroom, that checkpoints/interfaces are compatible, or that
conditional trajectory codes have been implemented. The next implementation is A--C
in an isolated environment with pinned versions and immutable manifests. Prior closed
Wall/ORDER/Scrub results and another session's files remain unchanged.
