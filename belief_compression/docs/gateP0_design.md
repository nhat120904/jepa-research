# Gate P0 — the diversity smoke test

**Purpose:** a cheap, decisive, pre-registered gate that must pass before the ~10-12 week / ~1000 A100-hour Gate C1 program (`gateC1_design.md` §2.6) is committed to.
**Date:** 2026-07-30
**Status:** design + verified feasibility + tested scaffolding. **No training run launched.**
**Budget:** ~25 GPU-hours, 7 wall-clock days (§6). Cap: 50 GPU-h / 1 week.
**Code:** `belief_compression/p0/`. **Tests:** `belief_compression/tests/test_p0.py` (40 new; the 46 pre-existing tests still pass — 85 passed + 1 skipped in `diagnosis/.venv`, 40/40 passed in a venv with the substrate installed).
**Entry point:** `diagnosis/.venv/bin/python -m belief_compression.p0.run_p0 --dry-run`

---

## 0. The question P0 must answer

Gate C0 established that the compute win requires `K >> |G|*(|A|-1)+1`, measured on **oracle** particle filters at K up to 1536. The load-bearing unknown is whether that regime exists for a **learned** belief model on **pixels**. Three ways it can fail, all fatal, all cheap to detect:

| | Failure | Gate C1 STOP |
|---|---|---|
| **VACUOUS** | the K learned hypotheses are near-identical, so M = 1. Compression is trivially "lossless" and meaningless; nothing to plan over, no VOI. | S5 |
| **NO-COMPRESSION** | the decision structure is rich relative to the filter, so M = K. No advantage. | S1 |
| **FILTER EATS IT** | the belief model's own forward cost is >= 80% of total compute, so saving planning compute is irrelevant end-to-end. | S3 |

P0 measures, on a real learned belief over a real partially-observed pixel task: **(a)** what K is achievable/meaningful, **(b)** the decision-diversity of the hypotheses => M, **(c)** M/K across goal-family richness, **(d)** the belief-model cost share.

---

## 1. Environment findings

All of the following is real command output from this machine on 2026-07-30, not documentation.

### 1.1 Compute

| Fact | Finding |
|---|---|
| **Login node GPU** | **NONE.** `nvidia-smi` produces no output at all. Nothing GPU- or Vulkan-dependent can be verified on the submit host. |
| **Slurm** | **Present and live.** Partition `main` (default): `worker-0..3`, each `gpu:nvidia_h100_80gb_hbm3:8`, 128 CPU, ~2 TB RAM. Partition `mig`: `worker-mig-3g40gb-0` with `gpu:nvidia_h100_80gb_hbm3_3g.40gb:16`. |
| **Currently usable** | `worker-0` (22/128 CPU alloc) and `worker-2` (68/128) are `MIXED`; `worker-1` is fully `ALLOCATED`; `worker-3` and `worker-mig-3g40gb-1` are `DOWN+DRAIN+NOT_RESPONDING`. `squeue` empty for this user. Account `normal`, no `GrpTRES` cap. |
| **Python** | system `3.10.12`; `uv` at `~/.local/bin/uv`. The **only** venv in the repo is `diagnosis/.venv` — `torch 2.7.0+cu126`, `cuda.is_available() == False` on the login node, **no jax**. |
| **Network** | PyPI reachable (`HTTP/2 200` on `pypi.org/simple/popgym-arcade/`). |

H100-80GB x8 nodes are far more than P0 needs. P0 fits on **one MIG 3g.40gb slice**.

### 1.2 Substrate 1 — POPGym Arcade: **WORKS, fully verified**

`uv pip install popgym-arcade` into a fresh Python 3.11 venv: **clean, no build step, ~40 s.** Installs `popgym-arcade==0.0.7`, `jax`/`jaxlib==0.10.2` (CPU wheels), `stable-gymnax==0.0.1`, `gymnasium==1.3.0`. **30 environments registered.**

Verified by running, not by reading:

```
popgym_arcade.make(env_id, partial_obs=...)  ->  (env, params)      [signature: (env_id: str, **env_kwargs)]
env.reset / env.step                          jax-jittable and vmappable
observation                                   (128, 128, 3) uint8, range [0, 255]
action space                                  Discrete(5)
```

`reset` + `step` confirmed for `BattleShip{Easy,Medium,Hard}`, `MineSweeper{Easy,Medium,Hard}`, `CountRecallEasy`, `NavigatorEasy`, at **both** `partial_obs=True` and `partial_obs=False`.

**The oracle hidden state is exposed on the returned `EnvState`** — this is what makes P0 possible at all:

| Task | hidden field | grid | positives | \|Z\| (exact) | enumerable? |
|---|---|---|---|---|---|
| MineSweeperEasy | `mine_grid` | 4x4 | 2 mines | **C(16,2) = 120** | **YES** |
| MineSweeperMedium | `mine_grid` | 6x6 | 6 | 1,947,792 | no |
| MineSweeperHard | `mine_grid` | 8x8 | 10 | 1.51e11 | no |
| BattleShipEasy | `board` | 8x8 | 12 cells | C(64,12) = 1.42e12 | no |
| BattleShipMedium | `board` | 10x10 | 12 | 1.05e15 | no |
| BattleShipHard | `board` | 12x12 | 12 | 1.13e17 | no |

**Confirmed empirically:** 2000 independent resets of `MineSweeperEasy` produced **exactly 120 distinct `mine_grid`s**, matching C(16,2) — the hidden space is genuinely, exhaustively enumerable. 2000 resets of `BattleShipEasy` produced **2000 distinct boards**.

**Throughput (CPU only, this login node, with rendering):** `jax.jit(jax.vmap(lax.scan))` over 128 parallel envs x 200 steps ran at **2.77e5 steps/s** after a 2.0 s compile. `jax[cuda12]==0.10.2` **resolves cleanly** (`uv pip install --dry-run` pulls the full `nvidia-*-cu12` wheel set), so the GPU path is available on the cluster; the paper's ~1e7 FPS claim was not verified here and is not relied on anywhere in this design.

**One honest negative.** The paired MDP/POMDP variants demonstrably differ on **BattleShipEasy** — identical at t=0..2, then `mean|diff| = 0.280` from t=3 onward. On **MineSweeperEasy** a 6-step random-action probe found **no pixel difference at all** between `partial_obs=True` and `False`. Reading `minesweeper.py` shows `partial_obs` only gates `_render_partial` vs `_render_full` and only once `state.timestep > 0`; the probe may simply not have revealed a cell. **This is a day-0 check, flagged, not assumed** (`envs.make` docstring records it). It does not affect P0's core measurement: the *mine positions* are unobservable in both variants either way, so the belief is genuinely uncertain regardless. It only affects the free "fully-observed upper bound" that Gate C1 §2.1 wants later.

### 1.3 Substrate 2 — MIKASA-Robo: **installs and registers; NOT verified to step here**

- `uv pip install mikasa-robo-suite` **fails**: it pins `mani-skill==3.0.0b15`, a pre-release, so resolution needs `--prerelease=allow`. With that flag it installs (uv resolves `mani_skill 3.0.0b14`, i.e. **the pinned version is not the one installed** — a latent version-skew risk).
- **Correction to `gateC1_design.md` §2.1.** That document names `BatteriesChecker{Easy,Hard}-{3,6}` and `ShellGameColorLampTouch` as if they were ordinary MIKASA-Robo tasks. They are **not in the RL family**. `mikasa_robo_suite.rl.memory_envs` contains 15 modules (ShellGame{Touch,Push,Pick}, RememberColor/Shape, RotateStrict/Lenient, Intercept, TakeItBack, SeqOfColors, BunchOfColors, ChainOfColors) and **no BatteriesChecker at all**. Importing `mikasa_robo_suite.vla.memory_envs` registers **164** envs and only there do these appear:
  `BatteriesChecker{Easy,Hard}-{3,6,9,12,15}-VLA-v0`, `ShellGameColorLampTouch-VLA-v0`, `ShellGameShuffleColorLampTouch{,-Long}-VLA-v0`.
  Gate C1's primary substrate is therefore the **VLA** family, which carries a different observation/action interface than the RL family. Budget accordingly.
- **Instantiation is blocked on this host** by two things, both expected and both GPU-node issues: an **interactive asset-download prompt** (`EOFError: EOF when reading a line` inside `mani_skill.utils.download_asset.prompt_yes_no`) and a **missing Vulkan ICD** (`sapien`: *"Failed to find Vulkan ICD file... incorrect or partial installation of the NVIDIA driver"*) because there is no driver on the submit node.

**Verdict:** plausible for Gate C1 P1/P2 on a GPU node after a non-interactive asset download, but **not verifiable today and far too heavy for a 1-week P0**. Not used.

---

## 2. Chosen substrate and task

### PRIMARY — POPGym Arcade `MineSweeperEasy` (4x4, 2 mines), pixels, `partial_obs=True`

Justified by what actually works:

1. **It installs and steps, today, with no fight** (§1.2). MIKASA-Robo does not.
2. **The hidden space is exactly enumerable (120 states, empirically confirmed).** This is the decisive property. It means the **exact Bayes posterior is computable in closed form with zero GPU-hours**, so every learned-belief number has a ground-truth reference on identical evidence. A smoke test whose only number is "the learned model gave M = 7" is uninterpretable; "the learned model gave M = 7 where the exact posterior gives M = 9" is a finding. Implemented as `belief.ExactEnumerationBelief`; needs no training and no substrate import.
3. **It is a genuine probe-and-reveal POMDP** — property (P) of `gateC1_design.md` §1.2 — with the hidden state unobservable in pixels.
4. It is fast enough that data collection is not a line item (2.8e5 steps/s on **CPU**).

### SECONDARY — `BattleShipEasy` (8x8, 12 ship cells)

Run in the same sweep. Its hidden space is 1.4e12, i.e. **not** enumerable, so K there is purely a knob of the *filter*, not of the task. `gateC1_design.md` §1.2 explicitly demands that **both readings be tested and both reported** ("at visual scale K is a property of the filter, not of the task"). MineSweeperEasy gives the task reading; BattleShipEasy gives the filter reading. It is also the task whose MDP/POMDP pairing is verified.

The two tasks are deliberately opposite in a second way: MineSweeper asks for a **safe** cell (2/16 positives — the decision is usually near-determined, so *lots* to compress) and BattleShip for a **ship** cell (12/64 — the decision genuinely depends on hidden bits, so *more* diversity). Between them they bracket the failure modes.

### The goal family (G) — added on the decision side, not by modifying the env

`gateC1_design.md` §1.2's load-bearing finding is that **no released benchmark provides (P) probing and (G) a goal family over shared hidden state simultaneously.** POPGym Arcade gives (P). P0 adds (G) **without touching the simulator** — same dynamics, same rewards, same observations; only the read-out question asked of the belief changes:

> hidden parameter `z` = the board's binary vector; a **goal** `g` = a **region** of the board; the terminal action = "name one cell in `g`"; `reward(z,a,g) = ±1` for matching the target bit, plus a fixed per-cell utility `u(a)` that ranks the matching cells.

This is exactly the move Gate B made on the numpy tasks and it is what makes the multi-goal pillar testable on an established suite. The utility term is not cosmetic: without it every hypothesis containing *any* matching cell ties, and `Task.preferred_action`'s first-wins tie-break makes the signature depend on cell **order** rather than on the hidden bits.

**Analytic bound.** For one goal the preferred action is one of `|g|` cells, so across a family
```
M  <=  min( K,  prod_{g in G} |g| )          [decision.RegionCommit.bound]
```
This is the P0 analogue of Gate C0's `min(K, |G|(|A|-1)+1)`. The **product** form (rather than C0's sum) follows from the hidden space being a bit vector with no 1-D geometry, and it is the **harder** test: with `|g| = 4` the bound runs `4, 16, 64, 256, 4096` for `|G| = 1..6`, so it **crosses K_REF = 128 between |G| = 3 and |G| = 4**. The goal-richness sweep therefore brackets, by construction, the point at which compression must stop working. That crossing is deliverable (c).

### FALLBACK

If MineSweeperEasy's oracle-signature control fails on day 0 (§5, gate A3), the ordered fallback is: **(i)** re-tune `region_size` / `target_bit` on the same task (free, no retraining); **(ii)** switch primary to `BattleShipEasy` (already in the sweep, richer decision structure by design); **(iii)** `MineSweeperMedium` (6x6, 6 mines) — loses exact enumerability but keeps everything else. MIKASA-Robo is **not** a P0 fallback; it is a C1 P1 concern.

---

## 3. The belief model

**Chosen: a small amortized posterior head + factored decode. Explicitly NOT DreamerV3.**

```
  frames (T, 64, 64, 3) --> 4-layer CNN --> GRU --> Linear --> n_cells logits
                                                                    |
                                            per-cell Bernoulli marginals
                                                                    |
                            K hypotheses  =  K ancestral draws (or exact top-K beam)
```

Trained supervised (per-cell BCE) against the **logged oracle hidden state**, which POPGym Arcade hands us for free (§1.2). ~3M parameters.

**Why this, and not the DreamerV3 that Gate C1 §2.2 borrows:**

1. **P0 asks whether decision-diversity EXISTS, not whether the model is good.** An amortized posterior trained directly on the oracle label is the **upper bound** on the hypothesis diversity any learned belief over these pixels can have — it is handed the answer at training time. That makes P0 a clean two-sided test: if even the upper bound is vacuous or incompressible, Gate C1 is dead and no RSSM will save it. This is the correct role for a smoke test and it is why the cheap model is the *right* model here, not a compromise.
2. It removes the dominant confound. A DreamerV3 that yields M = 1 is uninterpretable — undertrained, or genuinely no diversity? An amortized posterior at 95% per-cell accuracy has no such excuse.
3. It is ~2 GPU-hours per task instead of ~150.
4. `gateC1_design.md` §2.2 is explicit that **the belief filter is not the novelty** and that reinventing one is the fastest way to burn the budget.

**Three belief models are measured, all sharing one decode path (`belief.FactoredBernoulliBelief`):**

| Model | Role | Cost |
|---|---|---|
| `ExactEnumerationBelief` | **Reference.** Exact Bayes posterior over the 120 MineSweeperEasy states, conditioned on the same revealed cells the network saw. | **0 GPU-h** |
| amortized head (1 seed) | The learned-belief number. | ~2 GPU-h/task |
| `EnsembleBelief` (4 seeds) | Robustness: is diversity an artefact of one net's calibration? (Phase G in this repo's memory is the cautionary case — ensembled LoRA seeds over a frozen base shared their blind spot and the disagreement signal stayed flat.) | ~8 GPU-h/task |

Two decode modes, both shipped: `sample` (K ancestral draws, duplicates **kept** — deduplicating before measuring would hide the VACUOUS failure) and `beam` (the exact top-K configurations, deterministic, isolating "K is a filter knob" from sampling noise).

---

## 4. Measurement protocol

### 4.1 Hypothesis extraction

`BeliefModel.hypotheses(K, rng) -> HypothesisSet(params (K, n_cells) int8, weights (K,))`. Weights are `w_k ∝ p(z_k)` under the model (pre-registered) with uniform weighting reported alongside.

### 4.2 Decision signatures — **the existing `compress()`, verbatim**

This is the whole point of the module layout. `hypotheses.HypothesisTask` is a `belief_compression.core.Task` **whose `hidden_states` ARE the K decoded hypotheses** and whose `prior` is the K weights. Then

```python
comp = compress(Belief.prior(task), goal_family, tol=tol, counter=ComputeCounter())
M    = comp.M
```

is literally `belief_compression.compression.compress` — the same signature extraction (`_preferred_signature` / `_q_signature`), the same exact/tolerance merge, the same `pick_rep` and `rep_rule`s, the same `ComputeCounter`. **Nothing is reimplemented.** So M means exactly what it means in `gateC0_scaling_results.md` and every P0 number is directly comparable to the oracle-particle-filter numbers there.

Verified by test: the modes partition `range(K)` exactly, weights sum to 1, every representative is a member, and the counter charges exactly `K*|G|*|A|` reward evals (`test_hypothesis_task_satisfies_the_task_contract`, `test_compress_charges_the_signature_build`).

`param_embedding()` returns `None` (a bit vector has no 1-D ordering), so `centroid` falls back to `maxweight`. This costs P0 nothing: Gate C0 §S7 proved the mode **partition** — and therefore M — is identical under every `rep_rule`. The representative choice only affects *value* fidelity, a C1 P3/P4 question.

### 4.3 Sweep grid (pre-registered, frozen in `run_p0.py`)

| Axis | Values |
|---|---|
| K | 8, 16, 32, 64, 128, **128 = K_REF**, 256 |
| goal richness \|G\| | 1, 2, 3, 4 (MineSweeperEasy, 16 cells); 1, 2, 3, 4, 6 (BattleShipEasy, 64 cells) — capped at `n_cells // region_size` so regions stay **disjoint** |
| region size \|g\| | 4 |
| episode phase | 0.10, 0.25, 0.50, 0.75 |
| seeds | 5 |
| eval episodes | 200 per cell |

**Regions are drawn once from a fixed seed and reused across every cell**, so richness is the only thing varying between goal conditions. **Disjoint** by default — overlapping regions inflate signature agreement for free.

**Episode phase is part of the cell identity, not averaged over.** A belief model is diffuse early and nearly resolved late, and M moves with it by an order of magnitude; a single-point measurement would silently pick an arbitrary operating point. All phases are reported. The PASS gate needs a witness at **any one** phase — the honest reading, since the planner runs at every step and the question is whether the regime exists at *some* operating point.

### 4.4 Reported per cell

`M`, `M/K`, `bound`, `K/bound`, `ESS`, `n_distinct`, `C_comp` ops, and the **calibration error** `|sum(marginals) − n_positive|`.

**Two disclosed corrections found while building the scaffolding** (both caught by unit tests before any GPU time — this is why the scaffolding exists):

1. **`ESS` is the effective number of DISTINCT hypotheses, not the raw weight ESS.** Gate C1 STOP S5 states the diversity stop on "ESS of the particle weights". For an ancestral-sampled hypothesis set that statistic is **backwards**: a filter collapsed onto one configuration draws the *same* hypothesis K times, every draw has the *same* model probability, the weights come out uniform, and the raw weight ESS reads its **maximum (K) exactly when diversity is at its minimum (1)**. Measured: a collapsed belief at K=128 gives `ess_weights = 128.0`, `n_distinct = 1`. The statistic that means what S5 intends merges duplicates first: `ESS = 1 / Σ_u (Σ_{k: z_k=u} w_k)²`. Every P0 threshold is stated on that. For an exhaustive deduplicated support (the exact posterior) the two agree, so no Gate B / C0 number is affected. Pinned by `test_ess_detects_collapse`.
2. **Calibration error is a mandatory co-reported number.** A factored head whose marginals do not sum to the true positive count produces a degenerate hypothesis set — nearly every ancestral draw comes back all-zero — and reads as **spuriously VACUOUS**. A large calibration error means "the head is broken", not "the belief has no decision diversity", and those demand completely different responses. **P0 must never report a VACUOUS verdict without this number next to it.**

### 4.5 Oracle-side control (runs FIRST, costs nothing)

`measure.oracle_signature_count(true_params, decision, family)` — the number of distinct decision signatures produced by the **true** hidden state across the evaluation set. It needs **no belief model at all**, so it runs on day 0. If the oracle hidden state only ever yields one or two signatures, the *task* carries no decision diversity and no filter could represent any; that is a configuration bug, and it is fixed by re-tuning `region_size` / `target_bit` before a single GPU-hour is spent. This is the first half of Gate C1 STOP S5.

### 4.6 Belief-model cost share (STOP S3)

Measured, per decision, at K ∈ sweep and H ∈ {1, 3}, batched identically on one H100, in both registers (`C_prim` FLOPs and `C_wall` ms):

- `C_belief` = encoder forward + filter update. `measure.belief_flops(K, encoder_flops, per_hypothesis_flops, amortized)`.
- `C_search` = the `K*|G|*|A|` signature/Q build (Gate C0 §A6's `C_comp`, the honest headline — never the cached column) **plus** the expectimax tree, `expectimax_nodes(H, obs_branch) * carried * |A|` Q evaluations. `measure.search_flops(...)`.

**P0 does not need a trained Q head to do this.** A forward pass costs the same whatever the weights are, so an **untrained** head of the size Gate C1 §2.2 specifies gives the exact FLOP and wall-clock number. ~1 GPU-h.

**A-priori arithmetic (computed, in the repo, reproducible).** With a 50-MFLOP CNN encoder, K=128, M=16, |G|=2, |A|=4, branching 5, the belief share is:

| `q_flops` (one `Q(z,a,g)` eval) | H | amortized head | K-particle filter |
|---|---|---|---|
| 1e1 (analytic reward) | 1 / 3 | 0.9997 / 0.9978 | 1.0000 / 1.0000 |
| 1e4 (tiny MLP) | 1 / 3 | 0.7807 / 0.3129 | 0.9978 / 0.9831 |
| 1e6 (Gate C1 §2.2 UVFA head) | 1 / 3 | 0.0344 / 0.0045 | 0.8197 / 0.3677 |

**This is the single most useful thing P0 produces before it runs.** S3's outcome is not a fact about the world — it is determined almost entirely by two design choices: whether `Q(z,a,g)` is a table lookup or a network forward, and whether the belief is an amortized head (one forward, K cheap decodes; `C_belief` nearly **flat** in K) or a particle filter with a learned likelihood (K forwards; `C_belief` **linear** in K, exactly like the search it competes with). **P0 therefore measures both belief architectures**, because they land on opposite sides of the threshold. Pinned by `test_s3_verdict_is_driven_by_the_q_head_cost`.

---

## 5. Pre-registered PASS / FAIL thresholds

**Frozen in `measure.PREREG`, enforced by `test_prereg_thresholds_are_frozen` — changing any number breaks the build.** All are medians over 5 seeds x 200 episodes at **K_REF = 128** (the smallest K at which Gate C1's G1 is stated).

```python
K_REF = 128
MIN_MODES = 3          MIN_ESS = 8.0          MIN_ORACLE_SIGS = 3
MAX_RATIO = 0.25
MIN_K_OVER_BOUND = 4.0                        BOUND_SLACK = 1.25
MAX_BELIEF_SHARE = 0.80
STOP_VACUOUS_MODES = 2   STOP_VACUOUS_ESS = 2.0   STOP_NOCOMPRESS_RATIO = 0.5
```

### PASS requires a single WITNESS CELL — one (task, goal family, phase) at K_REF where A, B and C **all** hold — plus D globally

- **A — the hypotheses are decision-diverse (not vacuous).** median `M >= 3` **and** median distinct-hypothesis `ESS >= 8` **and** `oracle_sigs >= 3`.
- **B — something is actually compressed.** median `M/K <= 0.25`. (Gate C1 G1's number, at Gate C1 G1's K.)
- **C — the `K >> bound` regime Gate C0 requires exists.** at least one cell with `K / min(K, prod_g|g|) >= 4` **and** `M <= 1.25 * bound`, i.e. M tracks the *decision structure*, not K.
- **D — the filter does not eat the win.** `C_belief / (C_belief + C_search) < 0.80`.

**Why one witness cell and not "A somewhere, B somewhere else".** A filter whose hypotheses are diverse only under a rich goal family and which compresses only under a poor one has demonstrated nothing. It is satisfiable by construction — any family with `3 <= prod_g|g| <= 0.25*K` meets both, e.g. `|G|=2, |g|=4` gives bound 16 with K=128 — so this is a real constraint, not an impossible one. Enforced by `test_verdict_amber_when_gates_split_across_cells`.

### STOP if any (these fire independently of the PASS gates)

- **STOP-VACUOUS (Gate C1 S5).** max median `M <= 2` across every task/family/phase, **or** max median `ESS < 2`. *Never reported without the calibration error alongside (§4.4).* This is a **"fix the filter first"** stop, not automatically a program kill — but it is a hard stop on reporting any Gate C1 number.
- **STOP-NO-COMPRESSION (Gate C1 S1).** best median `M/K > 0.5` across everything. The decision structure is rich relative to the filter; M ~ K; the method buys nothing.
- **STOP-S3.** `C_belief` share `>= 0.80` at the operating point **and still** `>= 0.80` at max H. (Stated as a conjunction deliberately: a share that recovers with depth is Gate C0's amortization working, not a stop. `test_s3_does_not_fire_if_the_deep_horizon_recovers`.)

### AMBER

Not PASS, no STOP fired. **This buys exactly ONE targeted one-week extension, not the 10-12 week Gate C1 commitment.** Named and returned by the verdict function so it cannot be quietly reinterpreted later.

### The thresholds are verified to discriminate, offline, today

`run_p0.py --dry-run` runs the **same** `sweep` / `verdict` code with synthetic stand-ins:

| stand-in | verdict | why |
|---|---|---|
| `collapsed` (marginals pinned to 0/1) | **STOP-VACUOUS** | M = 1 at every K and every family |
| `diffuse` (every cell at p=0.5) | **PASS** at `\|G\|=1`, and the richness gradient is clean: M/K = **0.031 -> 0.109 -> 0.266 -> 0.484** for \|G\| = 1,2,3,4 (5 seeds, phase 0.5), crossing `MAX_RATIO` between \|G\| = 2 and 3 exactly where the analytic bound predicts |
| `partial` (information accrues with phase) | **PASS** at `\|G\|=3, phase=0.75` | the realistic case; near-misses at other cells |

---

## 6. Cost estimate

| Step | Work | Wall | GPU-h |
|---|---|---|---|
| 0 | Substrate check on a GPU node (`--check-substrate`); confirm the MineSweeper MDP/POMDP pairing (§1.2); run the **oracle-signature control** — needs no model, and can kill/retune the config for free | day 0-1 | **0** |
| 1 | Data collection: `jit(vmap(scan))`, 1024 envs, ~4M transitions/task. **Store `EnvState`, re-render on the fly** — 4M x 128x128x3 would be 196 GB of pixels, and the renderer is jittable, so this is near-zero storage | day 1 | 0.5 |
| 2 | Train the amortized posterior head: 2 tasks x (1 main + 4 ensemble seeds), ~2 GPU-h each, run concurrently across 8 GPUs on one node | day 2-4 | **20** |
| 3 | The measurement sweep (§4.3). `compress()` is pure CPU numpy; only the belief forwards touch the GPU | day 4-5 | 2 |
| 4 | Cost split: time the real belief forward and an **untrained** Q head of the C1-specified size, batched, K x H (§4.6) | day 5 | 1 |
| 5 | Analysis, `gateP0_results.md`, verdict | day 6-7 | 0 |
| | **Total** | **7 days** | **~24 GPU-h** |

Comfortably inside the 50 GPU-h / 1 week cap, with the whole thing fitting on **one MIG 3g.40gb slice** if the `main` partition is busy.

### What was cut to fit — and where it went

- **DreamerV3 / any RSSM.** -> Gate C1 P2. P0 uses the amortized upper bound instead (§3). *This is the biggest cut and §7 states its limitation plainly.*
- **All planning, regret and closed-loop return.** -> C1 P4. P0 measures M, not quality. Consequently `HypothesisTask.obs_space` / `obs_prob` **raise** rather than returning something plausible, so an accidental planner run cannot produce a meaningless number.
- **A trained `Q(z,a,g)` head.** -> C1 P3. The cost split uses an untrained head of identical size (identical FLOPs).
- **MIKASA-Robo / all manipulation.** -> C1 P1-P2. P0 is POPGym-only.
- **Medium/Hard difficulties.** Secondary, only if days 6-7 come in early.
- **Trajectory-clustered bootstrap CIs.** P0 reports medians + IQR over 5 seeds. CIs are a Gate C1 requirement; a smoke test that spends its budget on error bars around a number that is either 1 or 128 has misallocated.
- **The `tol > 0` tolerance-merge axis.** `compress(tol=...)` is wired through and tested, but the sweep runs `tol = 0` (exact decision-equivalence) only.

---

## 7. Honest read: can P0 really answer the question in ~1 week?

**For VACUOUS and NO-COMPRESSION — yes, decisively.** Those are the two questions this design is actually built for. The measurement is cheap, the ground truth is free (exact Bayes on 120 states), the substrate is verified working, the analytic bound tells us in advance where compression must break, and the thresholds are demonstrated to discriminate on synthetic stand-ins **before** any GPU time. If P0 returns STOP-VACUOUS or STOP-NO-COMPRESSION, that result is real and Gate C1 should not be funded.

**For STOP-S3 — only partially, and this should be stated up front.** §4.6's table shows the belief share is driven almost entirely by design choices (Q-head cost, amortized vs particle filter) rather than by anything P0 discovers about the world. P0 will produce a **real, measured** cost split for *this* belief model and *that* head size — genuinely useful, and enough to rule out the catastrophic corner — but it cannot settle S3 for a DreamerV3 RSSM plus a trained UVFA head that do not exist yet. P0 should report the split **parametrically in `q_flops`**, not as a single number, and Gate C1 P1 must re-measure it on the real components.

**The load-bearing limitation.** P0 tests an **amortized supervised posterior over the true hidden parameter**. That is an *upper bound* on the decision diversity a learned belief over these pixels can carry — it is handed the answer at training time. It is **not** the same object as a generative latent world model's posterior. So:

- P0 **STOP** => Gate C1 is dead. The upper bound has no diversity or no compressibility; no RSSM recovers that. **This inference is sound.**
- P0 **PASS** => the regime *exists in principle* on this substrate. Gate C1's actual belief model may still collapse. **This inference is one-sided**, and Gate C1 P2 must re-run the same measurement on the real filter — which the scaffolding supports for free, since `sweep` only needs a `BeliefModel` returning a `HypothesisSet`.

That asymmetry is the correct shape for a kill-gate: cheap to falsify, expensive to confirm. It is exactly what `gateC1_design.md` §2.6 R1 asked for ("this risk is cheap to falsify and expensive to discover late").

**Two things that could still make P0 take longer than a week.** (i) The MineSweeper MDP/POMDP pairing question (§1.2) — bounded, it is a day-0 check with a working fallback (BattleShipEasy). (ii) `main` partition contention: `worker-1` is fully allocated and two nodes are `DOWN+DRAIN`, so if `worker-0`/`worker-2` fill up, step 2 serialises from ~3 h to ~20 h wall-clock. Mitigation: the `mig` partition, where the whole of P0 fits on one 3g.40gb slice.

---

## 8. Scaffolding and how to run it

```
belief_compression/p0/
  decision.py    RegionCommit: the goal family / reward / signature layer, and
                 the analytic bound min(K, prod_g|g|)
  hypotheses.py  HypothesisSet + HypothesisTask -- the adapter that lets the
                 EXISTING compress() run verbatim on learned hypotheses
  belief.py      FactoredBernoulliBelief (production decode head AND synthetic
                 stand-in), EnsembleBelief, ExactEnumerationBelief,
                 calibration_error, and the collapsed/diffuse/partial constructors
  measure.py     PREREG thresholds, measure_cell, sweep, oracle_signature_count,
                 belief_flops / search_flops / expectimax_nodes, verdict
  envs.py        POPGym Arcade wrapper (jax imported lazily; TASKS carries the
                 verified substrate facts from §1.2)
  run_p0.py      one-command entry point
belief_compression/tests/test_p0.py   40 tests
```

```bash
# offline: full measurement + verdict path, no jax, no GPU, no trained model
diagnosis/.venv/bin/python -m belief_compression.p0.run_p0 --dry-run

# tests -- 85 passed, 1 skipped (the substrate test skips without popgym_arcade)
diagnosis/.venv/bin/python -m pytest belief_compression/tests/ -q

# on a GPU node, after `uv pip install popgym-arcade "jax[cuda12]"`
<p0venv>/bin/python -m belief_compression.p0.run_p0 --check-substrate
```

**Test status:** `85 passed, 1 skipped` in `diagnosis/.venv` (46 pre-existing + 40 new; the skip is the live-substrate test). In a venv with `popgym-arcade` installed the full P0 file is **`40 passed`** — the substrate test really does make, reset and step `MineSweeperEasy` and read its oracle hidden state. The tests cover: the `Task` contract and mode-partition invariants on the shared `compress()` path; `C_comp` charging; both fatal failure modes reachable by construction; `M <= bound` over a 48-cell grid; M/K falling in K and rising in |G|; the ESS correction and its trap; calibration error; exact-enumeration cardinality (`C(16,2) = 120`) and conditioning; the frozen `PREREG` dict; all five verdict outcomes including the split-cell AMBER; the expectimax node count against `planners.expectimax`'s recursion; and the S3 arithmetic.

**Not built (deliberately):** the CNN+GRU trainer and the POPGym data pipeline. Both need the GPU node, neither can be tested here, and writing untested training code was not the ask. The interface they must satisfy is one method — `hypotheses(K, rng) -> HypothesisSet` — and everything downstream of it is written and tested.
