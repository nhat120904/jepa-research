# Selection-aware encoder sprint (pre-registered 2026-07-21)

## Decision being purchased

This is a bounded feasibility sprint, not a commitment to a new full project.
It asks whether changing the encoder with a loss that matches test-time candidate
selection yields a contact-planning gain beyond ordinary supervised encoder
fine-tuning.  The primary comparison is **last-tail vs last-regression**.  A gain
from last-block capacity alone is not evidence for the proposed method.

The diagnostic paper remains the fallback if this gate fails.

## Fixed scope

- Checkpoint/task: `dino_wm_metaworld`, `mw-push` only for training and primary gate.
- Dynamics: simulator-perfect latent oracle.  Learned dynamics are out of scope
  until this cost/encoder gate passes.
- Mining seeds: 62000--62007; split by episode, 62000--62005 train and
  62006--62007 validation.  Evaluation seeds: 63000--63015.
- CEM: horizon 6, 100 candidates, 6 iterations, elite fraction 0.1, variance 1,
  execute 3 model steps, strict 100 environment steps.
- Registered subset per same-snapshot population: union of 10 lowest proxy-cost,
  10 lowest simulator cost, and 10 uniform random candidates.  The proxy-driven
  CEM update is unchanged.  Every row retains `(seed,replan,iteration,group_id)`.
- True push cost: object-to-goal distance + 0.5 hand-to-object distance, identical
  to the state-oracle shaping already verified in the project.

## Four arms (same buffer, head, steps and evaluation)

1. `lora_tail`: encoder LoRA + softmin expected-selection-regret.
2. `last_regression`: last four encoder blocks + direct Huber regression to true cost.
3. `last_pairwise`: last four encoder blocks + uniform within-population pairwise ranking.
4. `last_tail`: last four encoder blocks + softmin expected-selection-regret.

Three training seeds are run per arm.  `last_regression` is mandatory: without it,
the experiment cannot distinguish the selection objective from generic encoder
capacity.  All arms use the same goal-conditioned attention-pooling scalar head
and weight anchoring to their initial trainable encoder parameters.

Locked optimization: training seeds 0/1/2, four epochs of 400 sampled-population
updates, encoder LR `1e-5`, head LR `3e-4`, anchor coefficient `1e-4`, softmin
temperature `0.05 m`, regression Huber beta `0.05 m`, and pairwise minimum true
gap `0.005 m`.  Best checkpoint selection uses validation hard-selection regret
for every arm, which is favorable rather than restrictive to the regression baseline.

For candidate costs `chat_i` and simulator costs `c*_i` in one group:

```
p_i = softmax(-chat_i / tau)
L_tail = sum_i p_i (c*_i - min_j c*_j)
```

The deployed planner still uses the hard argmin.  Validation reports hard
selection regret; the relaxation is used only to make the training signal
differentiable.

## Locked readouts and gate

Primary: strict end success on 16 held-out push seeds and hard selected true-cost
regret on held-out populations.  Preservation control: reach under latent L2 with
the adapted encoder on the same 16 evaluation seeds.

Continue only if all are true:

1. `last_tail` beats `last_regression` in mean held-out selection regret and push
   success; the effect is not merely “full fine-tuning helps”.
2. All three `last_tail` training seeds reach at least 5/16 push success, crossing
   the prior encoder-LoRA envelope (mean 1.8/16, bootstrap upper bound 3.7/16).
3. Reach is preserved at >=13/16 for every continued seed.
4. Training is numerically stable and the result is not carried by one seed.

Stop the method direction if tail and regression are indistinguishable, push stays
0--2/16 despite lower regret, only one training seed moves, or all gains are
explained by last-block capacity.  The later top-tier bar (at least 8/16 push plus
cross-checkpoint/task and learned-dynamics replication) is deliberately not
confused with this pilot gate.

## Novelty boundary

Planner-aware objectives, differentiable MPC losses, and search-time robustness
exist separately in prior work.  The candidate contribution, if the gate passes,
is their intersection here: adaptive CEM populations as training groups, direct
selection-regret optimization, encoder updates, and contact-rich JEPA planning.
The paper must not claim novelty for ranking or differentiable planning alone.
