# Grounded shared-teacher composition prototype (new hypothesis)

Locked 2026-09-15, before any training or result. This is a **new method hypothesis**, not a
continuation of the closed self-supervised version (`COMP_PILOT_TARGET_FIX_RESULT_52644.md`).
It is a trajectory JEPA **with physical supervision**. It is no longer purely
self-supervised, and the claim must say so.

## Question

> Once a target summary is trained to keep contact effects, does a learned composer improve
> prediction of the history-conditioned contribution of long or unseen segment splits, beyond
> matched controls that use the same teacher, inputs and supervision?

## Positioning (targeted check, not novelty clearance)

- Privileged teacher → student latent distillation exists:
  - [TWIST](https://arxiv.org/abs/2311.03622);
  - [PTLD](https://arxiv.org/pdf/2603.04531);
  - [Privileged Foresight Distillation](https://arxiv.org/pdf/2604.25859).
- Composition or semigroup consistency in latent dynamics exists:
  - [World Models as Group Actions](https://arxiv.org/abs/2605.24578);
  - [Semigroup-JEPA](https://arxiv.org/abs/2609.10464) (recursive latent rollout consistency, 09/2026).
- Multi-timescale composition for planning exists: [CompPlan](https://arxiv.org/abs/2602.19634).
- The remaining distinction to test is composing **interior-effect summaries queried under
  history**. A gain explained by supervision alone is engineering progress, not evidence for
  the composition claim.

## Design

**Data.**
- Same 504 Scrub demos, DINOv2-S 4×4 features at 20 Hz, playback-monitor labels.
- Test episodes are dropped before loading.
- Train 345 episodes. Validation is split by fixed hash into `val_select` (38: monitoring only;
  final checkpoints are used) and `val_decide` (38: all decisions).
- The earlier pilots already read `val_decide`, so this is a development result. Final
  confirmation needs evaluation not used for design.

**Stage 1 — teacher (trained once, then frozen).** Inputs are observed history memory plus
observed window frames (fused DINO + proprio). A 4-token summary S is trained with:

- history-free effect heads: accepted-contact count, disc-union area, 12×12 board-frame
  contact map, normalized first-contact time per occupied cell;
- history-conditioned contribution heads on (memory, S): count increment, area increment,
  newly covered cells;
- a memory update U(m, S) matching the observed-memory continuation;
- the spatial camera×patch + proprio reconstruction anchor (weight 0.25 each).

Map losses use positive weight 5. The teacher uses no future information beyond the window it
summarizes, and its heads never see simulator state.

**Stage 2 — students.** Inputs are the observed prefix (through the frozen teacher's fuser and
memory) and proposed actions only. All students read out through the same frozen teacher heads.

| Student | Prediction | Long or split inference paths |
|---|---|---|
| `segment_nocomp` | Summary S and endpoint latent | sequential (sum of contributions, advancing memory with U) and direct whole |
| `segment_comp` | Same, plus a learned composer C | composed, i.e. inc(m, C(Ŝ₁, Ŝ₂, …)); also sequential and direct whole |
| `frame_teacher` | Teacher-space frame latents, S = teacher encoder(frames) | composed (encoder over concatenated rolled-out frames) and sequential |

Shared student losses:
- summary and endpoint distillation for the full segment, the left half, the teacher-forced
  right half and the rolled right half;
- head supervision on the full segment, the left half and the rolled right half.

`segment_comp` adds only composition terms:
- C(S_left, S_right) ≈ S_full on teacher summaries;
- C(Ŝ_left, Ŝ_right_rolled) ≈ S_full;
- head supervision on C(Ŝ…);
- U(m, C(Ŝ…)) ≈ observed memory at the segment end.

`frame_teacher` adds frame-latent distillation. All students train 32/64-step windows for
12k steps, with the teacher trained 8k steps, one seed (20260915).

**Evaluation (`val_decide` only, contact windows).**
- Splits: 64 = 32+32, 64 = 24+40, 128 = 64+64, 128 = 32+32+64. The 128-step totals and the
  24/40 split are unseen in training.
- Targets: history-conditioned count increment (primary), area increment, and newly covered
  cells (micro average precision).
- The teacher reading the true future is reported only to diagnose information loss. It is
  not a method result.

## Locked decision rule

For each primary split (128 = 64+64 and 64 = 24+40):

- **Best matched control:** the lowest count-increment MAE among `segment_nocomp` sequential,
  `segment_nocomp` direct whole, `frame_teacher` composed and `frame_teacher` sequential.
- **Composition "clearly better"** requires both:
  - `segment_comp` composed reduces MAE by **≥ 10% relative to the best control on both
    primary splits**;
  - the paired episode-bootstrap MAE interval versus that control excludes zero on at
    least one of them.

Outcomes:

| Outcome | Decision |
|---|---|
| Rule passes | Confirm with 3 seeds and evaluation not used for design; only then short simulator decisions (64–128 steps from shared prefixes), never 8-step interventions followed by a 1,000-step policy tail |
| Rule fails and the controls improve over the closed version | Grounding helps; do not defend the learned composer on this design |
| Rule fails and nothing improves | Stop this hypothesis on Scrub |

Also reported but not part of the rule:
- `segment_comp` composed versus its own sequential path;
- Pearson r;
- area and map metrics.

## Budget

- One GPU job: profile first, then the full run.
- 3 h walltime cap, no chained jobs, no additional seeds before the rule is read.
