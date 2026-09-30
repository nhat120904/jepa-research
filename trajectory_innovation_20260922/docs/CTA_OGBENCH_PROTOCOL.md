# CTA on OGBench visual-cube: pre-registered protocol (2026-09-28)

Fixed before any CTA training on OGBench. Changes after this point are recorded as amendments with the reason.

## Headroom already measured (dev episodes 0-19, 5 tasks, 20 each; jobs 55416/55418)
| Task | P0 (goal-conditioned flow chunk policy) | ORACLE8 | ORACLE16 | Published context |
|---|---|---|---|---|
| visual-cube-single | 7% | 37% | 34% | GCBC 5%, HIQL 89% |
| visual-cube-double | 22% | 52% | (running) | GCBC 1%, HIQL 39% |
ORACLE = privileged greedy choice among the policy's own seeded samples by simulated cube-target progress after the
chunk. Bank size K = 8 (the design default; K = 16 adds nothing for the greedy oracle on cube-single).

## Question and claims
Q: with the same frozen policy and the same seeded bank of 8 chunks, does selecting by CTA (one-pass code S of the
chunk's future, read against the goal image) raise success over the policy, and over matched rerankers?
- C1 (main): CTA > P0 and CTA ≥ every matched deployable reranker in closed-loop success.
- C2: CTA's quality per decision-time is better than the per-frame latent world model's (same data, same reader).
- Not claimed here: goal generalization (the 5 evaluation goals are shared by training and test; only initial states
  differ), trajectory information beyond the endpoint (the label is end-of-chunk progress).

## Data (identical for every learned reranker)
- Branched collection on TRAIN roots: episodes 1000-1059 (cube-single) / 1000-1039 (cube-double) of each task, seeds
  root_id(task, ep); disjoint from dev (0-19) and sealed test (20-69) episodes.
- Each decision: the policy's 8 seeded chunks (same seeds rule as deployment); each chunk is simulated from an exact
  restore; stored: decision frame, previous frame, goal image, chunks, 5 future frames per chunk, progress label after
  every step, success, cube positions. Executed chunk: candidate 0 on even episodes; on odd episodes the oracle-best
  candidate on 50% of decisions (seeded), candidate 0 otherwise (state coverage along better trajectories, as PushT
  Round 4). Budget reported as simulator steps and compared with the 1M-transition offline dataset.
- Held-out collection episodes (last 10% of train episodes per task) are used only for checkpoint selection.
- Labels = simulator progress (reader / ranking training only; allowed by RESEARCH_DESIGN §5a). No label at test time.

## Representations
Frozen DINOv2 ViT-S/14 on the 64x64 frames upsampled to 224, 16x16 patch tokens, PCA-128 fitted on train decision
frames. C = decision frame (256 tokens) + previous frame (8x8 pooled). Future tau = end frame (256) + frames after
steps 1-4 (8x8 pooled each). Goal = goal image (256 tokens). Actions = the 5 x 5 chunk.

## Arms (closed loop, same policy, same seeded banks, tie -> candidate 0)
Deployable: P0; **CTA** (parallel FSQ world model p(S | C, A), 16 tokens, read by D(C, S, g)); DIRECT (D(C, A, g));
ENDPOINT (one-pass predictor of the end-frame tokens, read by the FULL reader); FRAME (per-frame latent world model:
autoregressive over the 5 steps, 256 tokens per step, read by the FULL reader); DISTILL (the policy fine-tuned by BC on
the oracle-best chunk of each collected bank: the data-matched control for "more data, no world model").
Privileged ceilings (not deployable): ORACLE8; FULL (reader on the actual future); CODE (reader on the actual future's
code).
All learned modules: same data, same optimizer budget, same selection rule (own objective on held-out collection
episodes). CTA training = Round-4 recipe (NLL + score consistency + ranking through the reader) on 8-banks.

## Evaluation
- Development: episodes 0-19 of each task (100 episodes), used for debugging and at most 3 recorded repairs
  (docs/CTA_DEBUG_LOG.md style: evidence, one change, expected effect).
- Sealed test: episodes 20-69 of each task (250 episodes = OGBench's 50 per task), opened once for the final
  configuration, 3 training seeds for CTA and the matched rerankers.
- Metrics: success rate with paired bootstrap CIs by root and McNemar; per-decision wall time split into policy
  sampling, world model, reader; offline ladder (FULL, CODE, CTA, DIRECT) on held-out collection banks.
- Primary contrast: CTA − P0 and CTA − best deployable reranker on the sealed test.
