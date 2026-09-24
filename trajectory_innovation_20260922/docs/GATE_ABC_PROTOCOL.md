# PushT gates A–C: pre-registered protocol

Pinned 2026-09-22, before any policy rollout, bank outcome or visual score was observed.
This implements checks A–C of `EXPERIMENT_CONTRACT.md` §3 for frozen
`lerobot/diffusion_pusht` (HF revision `84a7c23`, LeRobot `3c0a209`). No world model is
trained here. PushT is a pipeline/headroom rehearsal, not the paper arena.

## Roots

- Qualification roots: env seeds 1000–1099. These are the first 100 seeds of the model
  card's own 500-episode evaluation (`eval_info.json`: 70/100 successes on these seeds,
  65.4% on all 500). They must never be used as untouched test roots later.
- Development roots, used only to build the visual goal map: seeds 5000–5031.
- Extension roots, used only under the rule in gate B: seeds 1100–1199.

## Runtime (shared by every arm except OFFICIAL)

- Environment: `gym_pusht/PushT-v0`, `pixels_agent_pos`, 96×96, 300-step limit, success
  when coverage > 0.95 at any step (the episode ends there).
- Decisions happen every 8 environment steps. The policy sees the last two observations,
  duplicated at reset exactly as `populate_queues` does, and executes the native slice
  `[1:9]` of the 16-step denoised tensor.
- Candidates: one batch of 32 DDPM samples per decision. Candidate k uses its own CPU
  generator seeded `candidate_seed(root, decision, k)` for the initial noise and for all
  100 per-step noises. Arms that use K=8 read candidates 0–7 of the same batch, so
  candidate 0 is identical across arms given the same state.
- Branching: every candidate is executed on a clone of the environment. The chosen
  candidate's clone becomes the real environment ("adopt the branch"), so the scored
  future and the executed future are the same object. The clone method is fixed from the
  smoke job (exact-repeat test required); pose-only reset is forbidden.
- Candidate tie rule: highest score, and the lowest index among exact ties (candidate 0
  wins any tie it is part of).

## Arms (closed loop, all 100 qualification roots)

| Arm | Selection per decision |
|---|---|
| OFFICIAL | Unmodified `policy.select_action` loop, `torch.manual_seed(root)`; no cloning |
| P0 | Candidate 0 (policy-only through our sampler) |
| PHYS8 | Argmax physical score over candidates 0–7 |
| VIS8 | Argmax visual score over candidates 0–7, computed on the ACTUAL future frame |
| MEDOID8 | Candidate 0–7 minimizing summed L2 distance of its 8 executed actions to the others (no future information; exploratory baseline) |

Physical score: execute the 8-action prefix and stop early at success. The score is the
coverage at the last executed step.

Visual score (pinned kernel): DINOv2 ViT-S/14 (`dinov2_vits14`, torch-hub LVD-142M
weights), frame at the last executed step of the prefix, resized 96→224 bilinearly and
ImageNet-normalized, `x_norm_patchtokens` (256×384). Score = −mean squared difference to the goal map G.
G = mean patch features of the terminal (success) frames of the first 16 successful P0
episodes on development roots, in root order. If fewer than 8 dev episodes succeed, stop:
this is a blocker, not a gate result.

## Gate definitions

**A: runtime reproduction.**
- PASS if OFFICIAL success on the 100 roots is within 10 pp of 65.4%.
- Also reported:
  - Wilson 95% CI;
  - exact binomial p vs 0.654;
  - paired agreement with the published per-seed outcomes;
  - P0 − OFFICIAL paired difference.
- A runtime blocker is declared if P0 − OFFICIAL has |diff| ≥ 10 pp and its CI excludes zero.

**B: selection headroom.** Δ_B = success(PHYS8) − success(P0), paired by root.
- PASS: Δ_B ≥ 10 pp and paired-bootstrap 95% CI lower bound > 0.
- EXTEND (rerun all arms on extension roots, pool 200, then apply the same PASS rule once):
  5 ≤ Δ_B < 10 pp, or Δ_B ≥ 10 pp with CI lower bound ≤ 0.
- FAIL: otherwise.

**C: visual-score alignment.** Only interpreted if B passes.
- Primary: retention = (success(VIS8) − success(P0)) / Δ_B ≥ 0.80.
- Reported alongside: raw differences, and a root-clustered bootstrap CI for retention.
- Offline secondary on P0 decisions: retained gap = Σ[phys(argmax vis) − phys(0)] / Σ[phys(best) − phys(0)] over candidates 0–7, plus the mean within-bank Spearman(visual, physical). Neither of these can rescue a failed primary.

## Diagnostics (not gates, interpretation fixed now)

**H_rep (repertoire headroom, continuation oracle).**
- Anchor: one per root, at decision ⌊u·n⌋, where n is the number of P0 decisions and u is a
  hash-uniform number from the root (independent of outcome).
- From the P0 snapshot at the anchor, each of candidates 0–7 is executed, and then the policy
  continues (candidate-0-style draws, dedicated seeds) until success or step 300.
- There are 4 continuation seeds per candidate: seeds {0,1} select, seeds {2,3} evaluate.
- H_rep_split = mean over anchors of [eval success of the selected candidate − eval success of candidate 0].
- The naive max-over-all-seeds estimate is reported only to show winner's-curse bias.

**Other diagnostics.**
- Offline myopic headroom for K = 8/16/32 on P0 decisions: phys(best of first K) − phys(0).
- Per-arm average max coverage.

**Reading the diagnostics together.**
- B FAIL with H_rep_split ≥ 10 pp and CI > 0: the repertoire exists, but an 8-step physical score cannot take it. That is a horizon/scorer problem, not a proposal problem.
- B FAIL with H_rep_split CI including 0: the proposals do not carry decision-relevant diversity at this cadence.

## Statistics

- Paired bootstrap over roots, 10,000 resamples, seed 0, percentile 95% CI.
- Exact McNemar for paired success.
- Decision-level quantities are bootstrapped by root cluster.
- No arm, kernel, threshold, root set or seed rule may change after qualification outputs
  are read. Any such change would be a new, labelled protocol.

## Compute

- Smoke: 1 GPU, ≤30 min. It covers strict load, sampler equivalence, nested banks, clone
  exactness, timing, bank diversity, and the dev goal map.
- Main: array of 4 GPU shards × 25 roots, time limit set from smoke timing.
- Aggregation: CPU job.
- The whole programme cap (8 GPU-h) applies.

## Amendment 1 (2026-09-22, after smoke 53787, before any qualification root was run)

Smoke 53787 found that plain `deepcopy` cloning repeated exactly, but deviated up to 112 state units from stepping the live environment
in place (20/20 checks), while the manual clone deviated at most 0.002. The dev P0 episodes
under deepcopy succeeded 0/32 and all eight candidates always had identical coverage: a
clone artifact, not a policy result. "Exact repeat" alone cannot detect a clone that is
wrong in the same way every time, so the selection rule changes to:

- candidates, in order: `deepcopy_fixed` (deepcopy with body kinematics and centre of
  gravity re-asserted from the source), `manual` (fresh space + copied kinematics), `deepcopy`;
- the first candidate with exact repeat AND live-continuation deviation <= 0.01 is used;
- post-clone integrity is recorded as well: pre-step state difference, bodies owned by the copied space, and centre of gravity.

Nothing else changes: roots, arms, scores, goal-map rule, thresholds. The dev results of
53787 are discarded, because they were produced under the broken clone.
