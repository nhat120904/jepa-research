# First bounded pilot: protocol v1

Locked before the first job, 2026-09-21. Exploratory infrastructure/development run,
not confirmation data or proof of method novelty. No automatic GPU continuation.

## Scope and decisions

Use original DINO-WM `env/wall/envs/wall.py` physics and RGB renderer through an adapter.
Source: `/mnt/data/nhatnc129/jepa/dino_wm_original`. Import only its `wall` package to avoid
unrelated MuJoCo/PointMaze registration. Snapshot upstream Wall source and hash it per run.
No rewritten collision model. Native 65x65 RGB, 2-D actions, transition `pos += 2*action`
before collision handling. Candidate action norm is capped at 1.8: a declared proposal
restriction, not a Gym action-space bound. Native steps, no frame skipping.

The adapter strips proprio/state/reward from model observations. Privileged positions
may generate valid reset/goal specifications and evaluate success, never enter predictor,
proposal, or image-query scoring. Goal images are permitted task inputs. Check exact
snapshot/replay of both images and states, including RNG states. Nonfinite upstream
transitions cause an error, not filtering. Fixed layouts are diagnostic variants.

## First CPU job (not a learned-WM result)

1. Unit tests: strict temporal order, reverse order, occupancy, endpoint, native-step
   indexing; bounded reproducible proposals; model gradients/interfaces; RGB-only
   contract; exact branch replay.
2. 24 independently sampled development prefixes across three layouts. Sample A/B goals
   before candidates, at least 9 pixels from each other and start. No goals selected from
   candidate outcomes, no retries based on candidate success.
3. 64 smooth open-loop candidates of 32 native steps. Candidate zero holds still;
   others interpolate random velocity knots. No goal/state input to proposal.
4. Report random-candidate, best-in-bank, image-query-selected success and image-score
   regret, separately for reach-B and A-before-B. Compare a sequential waypoint surrogate
   (rank reach-A until A actually observed) on the same bank. The latter is NOT closed-loop
   waypoint MPC performance. Report failures and prevalence, not only positive cases.
5. Save per-prefix rows, two sample banks, provenance, and 64/16/16 independent
   train/val/test RGB-action episodes of 128 steps. Split before windows/anchor selection.
   Test data are not used for tuning; initial encoding should exclude them.

Oracle feasibility alone does not establish ranking or closed-loop headroom. This small
screen has no confirmatory significance threshold; failures diagnose setup/proposal/
objective, not an untrained method. Candidates sharing a prefix are not independent
episodes; uncertainty must be computed over prefixes. No automatic GPU continuation.

## Query contract

`obs[t] --action[t]--> obs[t+1]`; future segment is `obs[1:H+1]`, excluding initial frame.
Ordered visits require `i < j`, never the same frame. A at the current observation is
handled by an actual-observation progress monitor. B-before-A is allowed if B is revisited
after A. All targets use native cadence. Goal anchors are independent of candidate
outcomes. Training anchors/normalizers must use training episodes only.

CPU screen kernel: **fixed spatial chroma cosine on RGB**, suppressing achromatic
background. This is a task-favorable hand-designed observation kernel, NOT DINO features
or evidence for a learned method. It needs no simulator labels. Scores are not success
probabilities. Ordered: `max_{i<j} min(k(obs_i,A),k(obs_j,B))`; reach: max; occupancy: mean.
Physical evaluation separately uses upstream's 4.5-pixel radius. No test tuning. Passing
the RGB screen would not validate a future DINO-feature kernel; changing query targets
requires an explicitly versioned condition.

## Next: actual method pilot

Encode RGB with frozen spatial SSL features, initially keeping the same query target
definition. Codec: future features -> learned summary tokens -> query reader. Predictor:
causal history + ordered proposed actions -> summary -> same reader. Train observed codec
first; freeze for initial forecasting. Answer-space loss primary, latent loss optional.
The summary must be query-independent. Frame-WM, generic compression and direct-query
baselines receive identical data/query supervision. Include fixed query-statistic banks
and sequential-waypoint MPC. Modules are scaffolding until matched training is run.

Planning: fixed-bank MPC, execute four native steps then observe/replan. Shared proposal
and query semantics, never update task phase from predictions. Evaluate on real branches
as well as logged data. Separate same-candidate and equal-wall-clock comparisons. CEM
refitting is later. First report separates codec, forecasting, selection and closed-loop.
One seed is development only; fresh episodes/seeds are needed for confirmation.

Composition, actor-critic, VLA training and learned task rewards are out of scope.
PushT is the preselected next manipulation candidate after a useful navigation signal;
native PushT scores do not transfer to a new ordered-pushing task.

## Amendment v2 after job 53521 (locked before rerun)

Job 53521 falsified the unguided proposal for the ordered task, not the learned method:
reach-B had an oracle candidate in 12/24 prefixes and the fixed RGB query selected all
12, but A-before-B had zero oracle candidate in 24 x 64 branches. Training a ranker on
that bank would be uninterpretable.

Replace only the proposal contract for the headroom rerun. Generate candidate chunks
from the **RGB start/A/B images** using red-chroma centroids and the documented native
action scale. Include exact and perturbed A-then-B chunks, direct-B chunks, unguided
chunks, and hold. It receives no privileged state, collision map, future observation,
reward or candidate outcome. This is a hand-designed inverse proposal, not a learned
method contribution. All future WM arms must receive the identical saved candidates.

Run 48 independently sampled prefixes, 128 candidates and horizon 48. Goals are still
sampled independently before candidate generation and never selected from outcomes.
The purpose is to establish nonzero, non-saturated ordered candidate support and that the
observation query discriminates it. Do not recollect/replace the v1 train/val/test data.
If the exact kinematic candidate hits a wall, it may fail normally; no rejection or retry.
This amendment changes proposal competence, so v1 and v2 rates are reported separately.

## Amendment v3: frozen encoder selected before encoding

The DINOv2-S source exists locally but its pretrained weights are not cached. A licensed
local DINOv3 ViT-L/16 checkpoint and source are already available and have previously
been configured for backbone-only loading. Use that frozen encoder rather than adding a
network download to the experiment. This is an implementation choice, not a method claim.

Resize native 65x65 RGB to 224x224 with bicubic antialiasing and ImageNet normalization.
Read normalized patch tokens, spatially adaptive-pool the 14x14 patch grid to 4x4, and
store 16 x 1024 fp16 features per frame. Do not collapse to CLS. Encode **train and val
only** in development; test remains unread. Preserve actions and the exact temporal
contract. Hash the source episodes, checkpoint and encoded files. Query targets remain
the fixed RGB-chroma targets locked above, so changing the input encoder does not silently
change the question the summary must answer.
