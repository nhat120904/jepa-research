# Signature composition versus the implemented learned composer

Date: 18 September 2026. Static source/code assessment; no new experiment.

## Corrected decision

The current results do **not** test or refute Chen-based signature composition. They test whether a Transformer can learn to merge freely learned trajectory summaries. Keep the negative verdict for that implementation. Before retiring the algebraic idea, add one small comparison using explicitly constructed signatures and a fixed product. Do not interpret this as evidence that signatures will improve planning.

The user's source is [Ohnishi et al., Signatures Meet Dynamic Programming, L4DC 2024](https://proceedings.mlr.press/v242/ohnishi24a.html). Its signature dynamic programming uses a known algebraic product, conditional future states and policy expectations. Its MPC combines past and imagined future path information. It is not an experiment showing that a learned visual summary merger improves a GR00T success critic. [Method, sections 4–5](https://arxiv.org/html/2312.05547).

This also corrects my previous explanation: a single-chunk planner can use composition meaningfully when scoring the concatenation of its actual prefix and candidate continuation. The absence of multiple imagined chunks does not, by itself, make composition irrelevant.

## What was actually implemented

The inspected [Composer class](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/comp_pilot/models.py:144) adds segment-side and duration embeddings, concatenates two four-token summaries with learned queries, applies a Transformer, and normalizes its output. It does not compute iterated integrals or a truncated tensor-algebra product.

The [grounded Student](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/comp_pilot/grounded.py:236) trains this merger to match the frozen teacher's whole-segment summary, with additional task-head and memory losses. A teacher trained to reconstruct/read useful information need not produce summaries closed under a small binary merger. The frozen-teacher experiment rules out target drift during student training as a complete explanation, but not a representation/merge mismatch.

The older [signature proposal](/Users/nhatcuong/code_project/vin-research/latent_scope_20260909/RESEARCH_VI.md:63) described an explicit signature/logsignature target and conditional joint endpoint–summary rollout. The later proposal deliberately replaced that with learned summaries. That was a substantive hypothesis change, not just an implementation detail.

## The precise distinction

For a consistently represented path X split at a shared boundary, truncated signatures obey

    S_<=M(X_left * X_right) = S_<=M(X_left) tensor_M S_<=M(X_right).

This is exact within the retained tensor degrees, apart from numerical error. Truncation removes information about the path, but does not make this product law an approximate learned relation. Associativity follows from the algebra. See the [signature primer](https://arxiv.org/abs/1603.03788).

In contrast, the implemented objective is

    C_phi(G(X_left), G(X_right)) approximately equals G(X_whole).

Neither a well-defined merge from these compressed inputs nor associativity is built in. A consistency loss can encourage the relation where observed, without providing the algebraic structure globally. A zero or uninformative summary can also be perfectly consistent, so consistency and decision usefulness need separate checks.

For intuition, at degree two let A and B be the two signatures, with degree-zero coefficient one. Their composition has

    C^(1) = A^(1) + B^(1)
    C^(2) = A^(2) + A^(1) outer B^(1) + B^(2).

As a simple constructed example, moving along x then y and moving along y then x have the same displacement, but different cross terms. An arbitrary token merger must learn which such interactions to preserve; this particular representation defines them in advance. That does not establish that these interactions are the ones Scrub's evaluator needs.

## Why the practical result can still be negative

An exact product removes learned-merge error, not all errors:

- The image encoder/projection may omit contact, object-relative geometry or other task-relevant channels.
- Low-degree signatures can alias trajectories. Their usual invariances may be inappropriate for absolute-position, dwell-time or retracing-sensitive tasks. Initial position, physical time and other relevant channels must be deliberately represented; augmenting channels changes which invariances remain.
- Action-conditioned predictions can still be inaccurate. Applying an exact product to inaccurate predicted tensors does not make their combined result correct or ensure the tensors correspond to realizable paths.
- The task readout can still misrank planner-selected candidates.
- A recurrent model, explicit map, or direct chunk critic may solve the same problem more efficiently.

The Scrub result therefore supports a decision against the tested learned merger. It does not identify its failure mechanism, nor prove that replacing it with fixed algebra would reverse the result.

## A bounded test that answers the user's objection

First test representation adequacy using observed trajectories, before a new world-model campaign. Keep data, source episodes, task readout supervision and tuning budgets matched. Compare endpoint features, a temporal encoder, a simple map/event representation, and degree-2/3 signatures of the same projected feature path. Existing Scrub data can expose obvious inadequacy, but repeatedly read splits remain development data.

Use one fixed projection and one coordinate/time convention throughout each comparison. Include the boundary sample in both adjacent subpaths; do not silently omit the connecting increment. If using an integrated feature lift, preserve accumulated coordinates across partitions. Independent per-chunk coordinate transforms or time normalization generally change the path being composed. Validate direct versus split computation numerically before measuring readouts.

A frozen projection is appropriate for this first attribution test. Learning the visual representation is a later controlled comparison; joint learning solely to satisfy composition can discard the very distinctions under investigation.

Only if the target retains useful information should the predictor learn

    F(m_t, z_t, actions, duration) -> (predicted endpoint, predicted signature).

Use a fixed Chen product, not another Transformer composer. If predicting logsignatures, convert through the appropriate exponential/product/logarithm or use the corresponding truncated BCH composition; adding logsignatures is not generally correct.

For multiple stochastic segments, condition later predictions on the sampled earlier endpoint/history and preserve their dependence. Products of independent marginal means generally do not equal the expected product. For a fixed observed prefix, composing it with an expected future signature is linear in the future argument; the remaining issue is whether a nonlinear readout of that expectation represents the desired objective. In particular, expected native success is not automatically recovered from a cost applied to a mean signature.

Give composition a concrete planning role: compare whole-path scoring with observed-prefix plus predicted-segment signatures against a matched history-aware baseline. For DrawTriangle, the visible trace and simple coverage aggregation remain strong controls. A prescribed direction or reference trajectory must not be added to its native task merely to favor signatures. A separate trajectory-tracking experiment is a different declared objective.

The decisive ablations are fixed algebra versus learned merger, direct whole-segment versus conditional split prediction, and signatures versus a dimension/compute-matched generic target. Matching the signature identity numerically is an implementation check; better selected actions or a meaningful quality–cost advantage is the research result.

## Implication for the research programme

The revised recommendation is to preserve **structured algebraic trajectory prediction as an untested candidate**, while retiring the current learned summary-merger implementation. The next expenditure should establish whether this specific representation is suitable, rather than simply train the same network longer. No claim of novelty or likely conference acceptance follows from inheriting a valid algebraic identity.
