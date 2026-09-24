# LIBERO-Goal qualification: is it a usable arena 2? (pre-registered)

Pinned 2026-09-23, before any LIBERO job. Nothing here tests the reader or a world model.
It asks only the questions that can kill the arena cheaply, in the order that kills it cheapest.

## Why LIBERO-Goal, and what could make it unusable

LIBERO-Goal: one kitchen scene, fixed objects, **10 goals**. That is the only arena choice
that tests "one goal-conditioned reader for many goals". Success is a predicate on the final
state, which matches the endpoint code the method now uses.

Known risks, each mapped to a stage below:

| Risk | Evidence before this protocol | Stage |
|---|---|---|
| Cluster cannot render | GPU GL/Vulkan impossible here; MuJoCo/robosuite work only via CPU OSMesa (RoboCasa ran that way) | L0 |
| Released policy does not reproduce | Open lerobot issues #2354, #3264; #4614: `lerobot/smolvla_libero` ships `n_action_steps=50` (≈ −20 pp), config gotchas (camera rename map, fps, MuJoCo ≥ 3.8.1 breaks some init states) | L1 |
| No headroom (policy too strong) | SmolVLA paper: LIBERO-Goal 91% at `n_action_steps=10` | L1, L3 |
| Branching is not faithful | PushT: plain `deepcopy` repeated exactly but diverged from the live continuation | L2 |
| Candidates are all the same | SmolVLA is flow matching; diversity across noise draws is unmeasured | L2 |

## Fixed choices

- **Policy:** `HuggingFaceVLA/smolvla_libero` (ships `n_action_steps=1`), run with
  `n_action_steps=10`. That is the paper's best LIBERO-Goal setting and the fix recommended in #4614. It is fixed now and
  **not** changed later to create headroom: a deliberately weakened policy (for example 50 steps) is a strawman.
- **Software:** lerobot with the `libero` extra, installed in a new venv at L0. The resolved lerobot commit,
  `hf-libero`, robosuite, MuJoCo and torch versions and the checkpoint commit are recorded and pinned. MuJoCo stays below 3.8.1 (#4614).
- **Rendering:** `MUJOCO_GL=osmesa` (CPU). Camera names and image orientation follow the lerobot LIBERO wrapper that the checkpoint was trained with.
- **Candidates:** K = 8 per decision. Candidate k's flow-matching noise comes from its own seeded generator, and the policy's default draw is candidate 0 (P0), as in PushT.
- **Init states:**
  - L1–L3 use the benchmark init states **0–9 of each of the 10 tasks** (100 roots).
  - Init states 10–49 are reserved and untouched for all later gates (reader, world model).

## Stages and kill rules

### L0: install and render probe (CPU job, then one short GPU job)

- Create a LIBERO-Goal env for each of the 10 tasks, reset, then step 50 random actions while
  rendering both cameras.
- Report: frames non-constant (per-image std > 0), image shape and orientation, and timing
  per step for physics only and for physics + render. Separately, load the policy on a GPU and time one K = 8 batched draw.
- **PASS:** all 10 tasks create and render, and physics + render ≤ 150 ms per step on one CPU core.
- **FAIL:** stop and report. Do not try EGL, Vulkan or GPU rendering (they are known to be impossible here).

### L1: reproduction and headroom shape (like gate A)

- P0 closed loop on 100 roots (10 tasks × init 0–9), 3 policy seeds each (300 episodes).
  The lerobot default max-step limit for libero_goal is used.
- **Reproduction:** the success rate of seed 0 is compared with the published 91%.
  - PASS if it is within 10 pp.
  - Below 81%: report as "does not reproduce". Then check only the documented config items
    (#4614 list) once. Do not retune anything else.
  - A reproduction failure is not automatically fatal (lower success means more headroom), but it must be
    stated as the operating point, not hidden.
- **Headroom shape (reported, not a gate):**
  - For each root, the fraction of the 3 seeds that succeed.
  - The share of failures that are **stochastic** (the root succeeds on some seeds) versus **systematic** (it fails on all 3).
  - Only stochastic failures can be fixed by choosing among the policy's own candidates.
- **Early kill:** if seed-0 success ≥ 97%, or if fewer than 5 roots have stochastic failures,
  stop: the arena cannot show selection headroom.

### L2: branching fidelity and candidate diversity (smoke)

- **Clone** = MuJoCo state save/restore (plus controller state if any), chosen by the same dual rule as PushT
  Amendment 1: the clone must repeat exactly, and it must match stepping the live env with the same actions
  (max |Δqpos| ≤ 1e-6 over 10 steps, 20 checks).
- **Diversity:**
  - Mean pairwise L2 between the 8 candidate chunks, relative to the chunk's own norm.
  - Share of decisions where the 8 branches end in visibly different object states, measured as privileged object-position spread > 1 cm.
- **PASS:** the clone passes, and at least 20% of decisions have object spread > 1 cm.
- **FAIL:** stop. If the candidates never move objects differently, no selector can matter.

### L3: oracle headroom (like gate B, continuation version)

A myopic 10-step oracle is not used as the headroom yardstick. C2 showed a myopic oracle can
understate the headroom a long-horizon selector takes, and LIBERO success is binary at the end.

- At one outcome-independent anchor per root (same `anchor_fraction` rule as H_rep), branch all 8
  candidates, then continue each branch with P0 to episode end, 4 continuation seeds per candidate.
- Split estimate as in PushT: pick the candidate with continuation seeds {0, 1} and evaluate it on {2, 3}.
  H_rep = eval success(chosen) − eval success(candidate 0). The naive max is reported only to show winner's curse.
- **PASS:** H_rep ≥ +5 pp with root-bootstrap CI lower bound > 0.
  This is a single-decision oracle, so it is expected to be smaller than a closed-loop, every-decision gain.
- **FAIL:** LIBERO-Goal is not arena 2. The pre-registered fallback is LIBERO-Long (10 two-stage tasks,
  SmolVLA ≈ 71%). It requires a new protocol and is not run automatically.

## What a full PASS licenses

Only: "LIBERO-Goal runs on this cluster, the released policy works near its published level, branches are faithful,
and the policy's own candidates contain choosable headroom." The reader (C2 analog, with goal images from
demo final frames) and any world model need their own protocols, on init states 10–49.

## Compute (bounded, needs approval)

| Stage | Resources | Estimate |
|---|---|---|
| L0 | CPU install + probe; 1 short GPU job | ≈ 1 CPU-h, 0.25 GPU-h |
| L1 | 1 GPU + 16 CPU, vectorised envs | ≈ 1 GPU-h |
| L2 | 1 GPU, 20 roots | ≈ 0.5 GPU-h |
| L3 | 1 GPU + 16 CPU, 100 anchors × 32 continuations | ≈ 2 GPU-h |
| **Total** | | **≈ 4 GPU-h** |

The programme's 8 GPU-h cap is nearly used by PushT (about 7.4 GPU-h after C2-confirm). LIBERO therefore needs its own explicitly approved budget.
Stages run in order. Each stage is submitted only after the previous stage's verdict has been read.
All timing numbers above are estimates; L0 measures the real cost, and L1–L3 are re-estimated from it before submission.

## Amendment 1 (2026-09-23): L0 speed criterion replaced by a budget check

L0 run 54423 (on the 54414 venv): all 10 LIBERO-Goal tasks create, reset from 50 init
states, and render both 256×256 cameras (non-constant frames). Mean step time with both cameras
rendered every step was 160–166 ms. That exceeds the 150 ms line pinned above, but that line was
an arbitrary number and was never derived from a budget. It is replaced by the question it stood for: can L1–L3 run in the
available compute at the measured speed?

- **L1:** 300 episodes × ≤ 300 steps ≈ 90k steps ≈ 4 CPU-h, which is about 15 min on 16 CPUs.
- **L3:** 100 anchors × 32 continuations × ≈ 200 steps ≈ 640k steps ≈ 29 CPU-h, which is about 1 h on 28 CPUs (user QOS cap: 32 CPU).
- Rendering only at policy decision points (every 10 steps) would cut this further, but it is not needed.

**L0 verdict: PASS (feasible).** This is an engineering-feasibility criterion, not an outcome of any method test,
so replacing it cannot bias L1–L3.

## Amendment 2 (2026-09-23, user decision, before any L1–L3 job)

**L2**
- The clone-fidelity check stays a hard requirement. It is technical: an unfaithful clone makes every branch result meaningless.
- The diversity numbers (chunk L2, share of decisions with object spread > 1 cm) are **reported only**. The 20% / 1 cm
  kill is withdrawn as arbitrary.

**L3: the single-decision H_rep gate is replaced by a closed-loop oracle at every decision**, as PHYS8 on PushT.

- PushT showed why: the single-decision oracle gave +2 pp while the every-decision oracle gave +10 to +15 pp.
- Arms, on the 100 roots (10 tasks × init 0–9):
  - **P0:** candidate 0;
  - **ORACLE8:** at every decision, each of the 8 candidates is executed for its chunk on a clone, and the
    candidate with the highest privileged goal progress is kept (candidate 0 wins ties).
- **Goal progress** is computed from the task's own BDDL goal predicates:
  - On/In predicates: negative distance of the object to its target;
  - Open/Close/TurnOn/TurnOff predicates: normalised joint progress toward the predicate's threshold;
  - satisfied predicates count as complete.

  The per-predicate functions are written and unit-tested before any L3 rollout.
- **Report:** success of both arms, the paired difference with a root-bootstrap CI, and McNemar. No kill threshold.
  The number is read together with the user.
- **Cost estimate:** ≈ 2,400 rendered steps per episode, so ≈ 11 CPU-h for 100 episodes (≈ 25 min on 28 CPUs), plus policy GPU time.
