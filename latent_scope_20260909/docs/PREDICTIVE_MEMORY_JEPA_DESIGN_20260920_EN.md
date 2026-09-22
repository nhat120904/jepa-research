# Predictive-Memory JEPA — design note (pre-qualification draft)

**Date:** 2026-09-20
**Status:** DESIGN ONLY. Nothing here has been implemented or measured. No qualification job
has been run in the new arena. Every number quoted from earlier work is from the closed
Scrub branch and does not transfer.
**Supersedes (as a research direction):** the compositional trajectory JEPA line documented in
`COMP_PILOT_*`, `COMP_GROUNDED_*`, `SEGMENT_VS_FRAME_RESULT_52662.md`,
`TRAJECTORY_SSL_STEP3_RESULT_53275.md`.
**Provenance note:** if qualification passes, this work should move to its own programme
directory. It is filed here because it descends directly from `latent_scope_20260909`.

---

## 1. Is this a different method from compositional trajectory JEPA?

No — not completely. It is the **same family with three substitutions**. Stating this
precisely matters, both to avoid over-claiming novelty against our own prior work and to
avoid re-running dead ends.

### Carried over unchanged

- Action-conditioned prediction: input is observation history plus a *proposed* action chunk.
- Compact per-segment summary rather than frame-by-frame rollout.
- An explicit composition rule across segments.
- Self-supervised training against future latents from an EMA teacher; no reward, success,
  contact, or coverage labels in the training objective.
- Evaluation must include unseen horizons and unseen partitions.

### Substituted

| Axis | Previous | Now |
|---|---|---|
| Object being composed | Learned summary tokens → path signature → (proposed) discounted occupancy | **Memory-update operator** `(D, b)` |
| Composition rule | Learned Transformer composer → Chen product → affine mixture | Operator product (parallel-scan form) |
| Carried state | Summary vector, memory advanced by a learned `U` | Persistent memory vector updated by the operator itself |
| Target semantics | Effect *inside* a segment (contacts, coverage) | Predictive memory sufficient for future-latent prediction |
| Arena | RoboCasa `ScrubCuttingBoard` | MIKASA-Robo memory tasks |
| Claimed capability | Better effect prediction than frame rollout | **Memory prediction for action chunks that were never executed** |

### Killed by prior results — do not resurrect

- **Learned neural composer.** Four evaluations, never beat sequential memory advance.
- **Chen / path-signature composition.** Bilinear cross terms amplify prediction error;
  at `128 = 64+64` the composed MSE was 2.511 vs 0.179 for ordered-frame rollout.
- **Coverage/contact-count as the headline target on Scrub.** Job 52662 closed this branch
  against pre-registered thresholds.
- **"Composition is the bottleneck" as a premise.** Job 52662 measured the learned memory
  update `U` at MAE 1.092 versus 1.093 for continuing the *observed* memory. Aggregation
  was already near-lossless; the residual error was in predicting summaries from actions.

The third and fourth points are why the new design does **not** sell composition stability
as its contribution.

---

## 2. Research question

> Can a self-supervised, action-conditioned world model learn a **predictive memory operator**
> for an action chunk — such that the memory state induced by a chunk can be predicted
> *before the chunk is executed*, remains consistent under unseen temporal partitions and
> horizons, and improves memory-intensive robot control beyond matched recurrent and
> full-history baselines given the same data and budget?

Two sub-questions, to be reported separately:

- **Q1 (representation).** Does a chunk-level ("jumpy") operator predicted in one shot agree
  with the composition of its sub-chunk operators, and does it hold up at horizons and
  partitions not seen in training?
- **Q2 (control).** Does having that operator improve action selection, relative to a matched
  recurrent memory encoder that can only encode *observed* history?

Q1 without Q2 is not a result. The repo's most reproduced finding is that representation
quality improvements do not transfer to control.

---

## 3. Method

### 3.1 Observation encoder

Frozen or EMA-updated spatial encoder over RGB patches plus proprioception. Spatial tokens
must be preserved, not globally pooled — the previous pilot compressed a 3-camera 4x4 patch
grid to a 16-d code before forming targets, which is a plausible cause of its weak targets.

### 3.2 Segment operator

A segment (action chunk) `A` of length `H` maps to an affine memory operator

```
T_A = (D_A, b_A),     m_out = D_A · m_in + b_A
```

with `D_A` diagonal (or block-diagonal) and entries constrained to `[0, 1]`.

Two ways to obtain `T_A`:

- **Composed:** product of per-step operators predicted along the chunk.
- **Jumpy:** predicted in one shot by `F_θ(m_t, z_t, a_{t:t+H-1}, H) → (D̂_A, b̂_A)`.

### 3.3 Composition

```
T_B ∘ T_A = (D_B · D_A,  D_B · b_A + b_B)
```

**This is the standard parallel-scan operator for a linear recurrence, and associativity is
true by construction.** It is a property of the parameterization, not a finding. Concretely:

- A "random-partition consistency" loss over *composed* operators is vacuous — it penalizes a
  quantity that is identically zero.
- The only non-trivial constraint is **jumpy vs. composed**: forcing the one-shot chunk
  operator to match the product of its sub-chunk operators.

That constraint is a cross-timescale consistency objective, which is conceptually the same
idea as the consistency objective in CompPlan (occupancy form). The differences we can claim
are the parameterization (bounded gates, explicit forget) and the conditioning (raw candidate
action chunks rather than pre-trained policies). Whether that is enough is an open question,
not an assumption.

### 3.4 Why bounded gates

- `‖D‖ ≤ 1` means composition is non-expansive, so operator error does not blow up with the
  number of segments — unlike the Chen product.
- A dimension can be latched open and held (checklist / irreversible milestones).
- A dimension can be selectively forgotten (object tracking, re-localization).

### 3.5 Training objective (self-supervised)

No task labels enter the loss.

1. **Future-latent JEPA.** From `(m_t, z_t, a_{t:t+H-1})`, predict EMA-teacher latents at
   future timesteps. Stop-gradient on the teacher.
2. **Jumpy-vs-composed operator consistency.** As in 3.3. Sample partitions at train time.
3. **Masked-segment prediction.** Hold out a middle segment; predict its operator from context
   and actions.
4. **Endpoint latent prediction.** Auxiliary, keeps the memory grounded in observable state.
5. **Variance/covariance regularization.** Anti-collapse.

Horizons are randomized during training. A set of horizons and partition patterns is held out
entirely for evaluation.

### 3.6 How the model is used at decision time

Given a fixed proposal policy producing `K` candidate chunks from the same state:

```
m^(j) = T̂_j ∘ m_t        for j = 1..K,  computed without executing anything
```

then score `m^(j)` and pick one. **The scoring function is the weakest and least resolved part
of this design.** It is the same gap that was never closed on Scrub. Options, none validated:

- Goal/reference-conditioned distance in memory space (changes the problem to reference
  matching, and must be reported as such).
- A separately-trained value head (breaks the fully-self-supervised claim for the *system*,
  though not for the world model).

This must be decided before any method training, not after.

---

## 4. Relationship to prior art

The transfer being made is: **associative-scan operator algebra from structured state-space
models → segment composition in an action-conditioned world model.**

| Work | Overlap | Remaining difference |
|---|---|---|
| CompPlan, jumpy world models | Multi-timescale prediction, cross-timescale consistency, compositional planning, manipulation/navigation | Conditions on pre-trained policies; ours on raw candidate action chunks; ours composes an *observed* prefix with an imagined continuation |
| Mamba / S5 / gated linear attention | The `(D, b)` operator and its scan composition | They encode observed sequences; they cannot produce the operator for an unexecuted chunk |
| TD-JEPA | JEPA latent prediction framed as successor features, self-supervised | Not chunk-conditioned, no operator composition |
| Memory VLAs (e.g. HAMLET, μVLA, MemoryVLA) | Memory for manipulation policies | Policy-side memory only; no counterfactual memory prediction |

**Honest statement of the delta:** the composition algebra is borrowed and is not new; the
multi-timescale consistency idea is not new; the arena is not ours. What is defensible, if it
works, is predicting the memory-update operator for an unexecuted action chunk and showing it
buys control over a matched recurrent baseline.

---

## 5. Arena and data

**Primary: MIKASA-Robo** (ICLR 2026; 32 tasks in the paper, expanded registry with more
memory types; ManiSkill/Gymnasium; released trajectory datasets).

Chosen because current-frame observations are genuinely insufficient by construction, it spans
several distinct memory types, it offers multiple horizons for length generalization, and it
has serious memory baselines already published. It was not built by us, which removes the risk
of designing a task that flatters the method.

Four representative tasks for the first pass:

| Task | Memory type |
|---|---|
| `BatteriesChecker` | Checklist / latching |
| `ShellGameShuffle` | Object tracking |
| `TraceShapeSeq` | Procedural |
| `GatherAndRecall` | Prospective |

**Secondary (only after a positive primary result):** ManiSkill `DrawTriangle`/`DrawSVG` as a
trajectory-accumulation and partition-generalization mechanism arena.

**Negative control:** OGBench Scene. Fully observable — a single frozen latent already probes
the latched scene variables at 98–99%. Used to show no regression, never to claim memory.

**Not an arena any more:** RoboCasa `ScrubCuttingBoard`. Appendix/transfer only.

---

## 6. Evaluation

### 6.1 Headroom ladder — run before any method training

| Rung | Question |
|---|---|
| 1. Current frame only | Does the task actually require memory? |
| 2. Fixed-window Transformer | Is a short window already enough? |
| 3. GRU / Mamba over full history | Has the recurrent baseline saturated? |
| 4. Oracle memory (privileged) | What is the ceiling from memory alone? |
| 5. **Oracle chunk selection** | Does picking the best candidate chunk raise success? |

Rung 5 is mandatory. Measuring rungs 1–4 and inferring rung 5 is exactly the inference that
failed on Scrub.

### 6.2 Mechanism metrics (Q1)

- Jumpy-vs-composed operator agreement, on held-out horizons and partitions.
- Future-latent prediction error at unseen total lengths.
- Predicted-vs-realized memory state for chunks that were executed, as a calibration check.

### 6.3 Control metrics (Q2)

- Selected-vs-default success under a fixed proposal policy, paired by initial state and seed.
- Candidate selection evaluated with a selection seed set and confirmed on a disjoint seed set.
- Bootstrap CIs over episodes, not over windows.
- At least 3 training seeds before any win is claimed.

---

## 7. Gates and stop rules

**Gate 0 — renderer (blocking, run first).** ManiSkill3/SAPIEN renders through Vulkan and needs
an NVIDIA ICD plus render-device access. On this cluster, jobs 51833 and 51834 failed with
`libEGL warning: failed to open /dev/dri/renderD135: Permission denied` on both MIG and full
GPU, and the login node carries no `nvidia_icd.json` (only intel/lvp/radeon/virtio). SAPIEN has
no OSMesa fallback; the fallback is lavapipe, i.e. CPU software Vulkan.

| Outcome | Action |
|---|---|
| NVIDIA ICD present, render node accessible | Proceed as planned |
| lavapipe only | Run the ladder on **state observations**; defer all visual claims |
| Renderer will not initialize | MIKASA is unusable here; stop before investing weeks |

**Gate 1 — headroom.** Rung 1 must lose clearly to rung 3; rung 3 must not have reached rung 4;
rung 5 must show a gain. If rung 5 is flat, do not build the method — that is the Scrub failure
repeating.

**Gate 2 — mechanism.** Jumpy operators must agree with composed operators on held-out
partitions, and future-latent error at unseen horizons must not exceed an ordered-frame
baseline by a pre-registered margin.

**Gate 3 — control.** Selection with the predictive memory must beat a matched recurrent
baseline given the same data, supervision, and parameter budget, on held-out seeds, across
3 training seeds.

Fail any gate → stop, write it up, do not re-parameterize and retry in place.

---

## 8. Known risks

1. **Rendering.** See Gate 0. Highest-probability blocker; the RoboCasa Stage A history on this
   cluster is directly relevant.
2. **Baseline collapse.** With the same `(D, b)` operator family, a plain Mamba history encoder
   computes the identical memory step by step. We cannot win on representational power — only
   on counterfactual capability, compute, or horizon extrapolation.
3. **Arena/claim mismatch.** MIKASA-Robo scores policy success on memory tasks; it does not by
   itself test counterfactual chunk evaluation. If the headline metric is plain task success,
   the distinctive capability is never measured and the comparison degenerates into
   "another memory encoder". The selection-based evaluation in 6.3 exists to prevent this.
4. **Scope of "self-supervised".** The world model can be trained without task labels. If the
   downstream policy is trained on demonstrations, the honest claim is *self-supervised memory
   pretraining with an imitation policy*, not a fully self-supervised system.
5. **Schedule.** Target is ICML 2027 (~January). A brand-new arena plus a new method in roughly
   four months is tight; RoboCasa Stage A alone took about ten job iterations to get replay
   working. One arena only until the mechanism shows a signal.

---

## 9. Open decisions (must be resolved before implementation)

1. The scoring function used at decision time (Section 3.6).
2. Whether the primary evaluation runs on RGB or state observations (depends on Gate 0).
3. Memory dimensionality and whether `D` is diagonal or block-diagonal.
4. The proposal policy for candidate chunks in the MIKASA setting.
5. Which pre-registered margins are used in Gates 2 and 3.
