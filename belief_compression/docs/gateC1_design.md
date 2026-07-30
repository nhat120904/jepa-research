# Gate C1 — Learned-Visual Stage: Substrate Verification + Design

**Method:** Decision-Equivalent Belief Compression + amortized decision-regret VOI
**Date:** 2026-07-30
**Status:** design + verification document only. **No training code written.**
**Prior gates:** `gateA_novelty_matrix.md` (GO, narrowly scoped), `gateB_oracle_results.md` (GO, exact-POMDP oracle), `gateC0_scaling_results.md` (**STRONG-BUT-NARROW**: scaling measurement passes, fidelity lossy-on-probing and only empirically patched, and the bound is elementary with an enabling regime -- `K >> |G|*(|A|-1)+1` -- that is unverified at visual scale; see its §3)

**Standing constraint:** the researcher has no real robot. Everything must stand on simulation, and the benchmark must not look bespoke-to-win.

**What Gate C1 has to buy.** Gate A left exactly two defensible novelties: (1) multi-goal invariance of the compression, with a bound holding *jointly* across a goal family; (2) an amortized flip-prediction VOI operating on the compressed M-mode representation. Gate C0 returned **STRONG-BUT-NARROW**: M saturates along *resolution-refinement* axes (a control axis where every latent dimension is decision-relevant gives M = K, so this is not a tautology) and the commit is exactly lossless, but the compute ratio flattens in K at fixed horizon (capped by an unavoidable O(K) belief read) and only approaches K/M as H grows; the saturation itself follows from the elementary arrangement bound `M <= min(K, |G|*(|A|-1)+1)`, whose enabling regime `K >> |G|*(|A|-1)+1` is exactly what P0 must verify for a learned belief. S6-S8 then sharpened the fidelity picture: compression preserves the *argmax* but not the *value*, which is what probing decisions depend on. Carrying a representative particle is value-inconsistent (`maxweight` 44.74% worst closed-loop regret, `centroid` 0.56% at identical compute -- `centroid` is now the default); a value-consistent per-mode summary reaches 0.00% regret, but the variant that is exact *under probing* costs ~1.0x of full belief, i.e. it removes the advantage by construction, and the cheap frozen variant is a net loss at H<=1. **This weakened novelty (2)**: the compressed representation is precisely what mis-values probes, because an observation reweights particles *within* a mode and that conditional is discarded by construction. Gate C1 must show those properties survive a learned belief over pixels — or kill the program.

---

# PART 1 — Substrate verification (Task 1)

## 1.1 IMBench — VERDICT

> **Headline: IMBench is REAL. "Mass Sort" and "Occluder Push" EXIST, verbatim, as tasks T25/T26 in category P5 "Hidden State & Active Discovery". But the cited arXiv ID is wrong — there is no arXiv posting at all — and the benchmark CODE / SIMULATOR IS NOT RELEASED. Only 10 demonstration episodes per task are public. It cannot be the substrate.**

### How it was verified

`imbench.org` is a JS-rendered React SPA; a plain HTML fetch returns only a 1.3 KB shell (`<div id="root">`), which is why the earlier attempt read nothing. The route that worked was to download the compiled bundle and read the embedded data structures directly:

- Site shell: <https://imbench.org/>
- **JS bundle (contains the full task table inline): <https://imbench.org/assets/app.mRqi76AQ.js>** (386 KB)
- Paper PDF: <https://imbench.org/paper.pdf> (verified `HTTP 200`, `application/pdf`, 11,789,400 bytes; downloaded and text-extracted in full)
- HuggingFace org: <https://huggingface.co/imbench> — enumerated via `https://huggingface.co/api/datasets?author=imbench&limit=100` -> **36 repos** (35 task repos + 1 index repo `imbench/imbench-dataset`)
- Example task card: <https://huggingface.co/datasets/imbench/pb-pr-mass-sort-v1>

Everything asserted below was read from the bundle, the PDF text, or the HF API — not inferred.

### Identity and publication status

| Field | Finding |
|---|---|
| Real title | **IMBench: A Benchmark for Intuitive Robotic Manipulation** |
| **arXiv ID** | **NONE.** The cited `arXiv:2607.15641` does not resolve to IMBench (the nearest search hit is `2601.15641`, an unrelated quantum-ML paper). There is **no arXiv link anywhere in the site bundle or in the paper PDF** — the only `arxiv.org/abs/` string in the whole PDF is reference [4] to ManiSkill (`2107.14483`). Treat `2607.15641` as fabricated / mis-transcribed. |
| Actual venue | **Anonymous submission to CoRL 2026.** PDF header line: *"Submitted to the 10th Conference on Robot Learning (CoRL 2026). Do not distribute."* |
| Authors | Literally `Anonymous` in the site's own BibTeX: `@article{imbench2025, title={IMBench: A Benchmark for Intuitive Robotic Manipulation}, author={Anonymous}, year={2025}}` |
| **Code / GitHub** | **No repository exists.** The only `github.com` string in the entire 386 KB bundle is the chart.js `kurkle/color` dependency. No code link on the site, none in the paper. A targeted GitHub search returns nothing. |
| **Simulator** | **robosuite** (paper: *"IMBENCH consists of 35 tasks built on robosuite [37]"*). Software table (Table 14): **robosuite 1.5, MuJoCo 3.1.2**, Python 3.12, PyTorch 2.7.1+cu126. Assets authored in MJCF. Robot = Panda; 4 cameras. |
| Data released | **Partially.** 35 HF dataset repos, Apache-2.0, LeRobot v3.0, public, last modified 2026-05-07. |

### The release gap — the decisive fact

Appendix B of the PDF states it explicitly:

> "We release a subset of the dataset (**10 episodes**) via Hugging Face **for review purposes**. ... **The complete benchmark and full dataset will be released upon acceptance.**"

Cross-checked against HF: `pb-pr-mass-sort-v1` reports **Episodes 10**, 37,899 frames, 100 FPS. The headline "14K filtered trajectories" is the *unreleased* set (the paper targets 200 teleop + 200 scripted per task; T25 and T26 both list `200 / 200`).

So the public artifact today is:

- YES — Paper PDF (readable; complete task specs incl. randomization ranges and success predicates)
- YES — 350 demonstration episodes total (35 x 10), LeRobot v3.0
- NO — **No environment code, no MJCF task assets, no reset/randomization implementation, no success predicates as code, no scripted-oracle generator, no evaluation harness**

**This is disqualifying as a Gate C1 primary substrate.** Gate C1's primary metric is a *planning-compute vs. quality frontier*, which requires stepping an interactive simulator on the order of 1e6-1e8 times. Ten offline demo episodes cannot support a planner at all.

### The 7 categories (exact, from bundle constant `f0` and the paper)

| ID | Category | # tasks | Description (verbatim) |
|---|---|---|---|
| P1 | Geometry-Constrained Grasping | 9 | Affordances, occlusion, handover reorientation |
| P2 | Dynamics & Trajectory Prediction | 7 | Projectile motion, pendulum timing, friction-dependent paths |
| P3 | Causal & Indirect Action | 6 | Dominoes, seesaw balance, force chaining |
| P4 | Tool Use & Augmented Reach | 2 | Hook-and-drag, cup as ladle |
| **P5** | **Hidden State & Active Discovery** | **2** | **Mass probing, occlusion removal** |
| P6 | Reactive Replanning | 3 | Slip recovery, collapse recovery, teleporting goals |
| P7 | Stability & Equilibrium | 4 | Shape stacking, rod balancing, packing |
| Misc | Diagnostic Tasks | 2 | Mirror observation inversion, bimanual typing |

(9+7+6+2+2+3+4+2 = 35. The site task grid shows 33 because the two Misc tasks are filed separately; both exist on HF as `pb-pr-mirror-pick-place-v1` and `pb-hobi-keyboard-typing-v1`.)

### The P5 tasks — verbatim from the paper (Appendix B.2.5)

> **T25 mass-sort.** *Setup:* Three visually identical cubes have hidden masses of **50 g, 100 g, and 200 g in randomized order**. Three labeled zones on the table accept the cubes in light-to-heavy order from left to right. *Goal:* Place each cube in its correct zone according to mass. *Intuition:* **Mass cannot be determined visually. It is observable only through wrist force and torque during a probing lift.** Each cube must be probed before placement. Any wrong-zone placement is an immediate failure.

> **T26 occluder-push.** *Setup:* One red target cube and two distractor cubes are each covered by a lightweight lid. A green goal patch sits elsewhere on the table. *Goal:* Place the red target cube on the goal patch. *Intuition:* **The red cube cannot be identified until its lid is lifted or pushed aside.** The agent must uncover each cube to locate the target.

These are genuine information-gathering POMDPs: `mass-sort` has K = 3! = 6 permutation hypotheses (or a continuous 3-D mass belief under noisy F/T); `occluder-push` has K = 3 identity hypotheses; in both cases information is obtainable **only by a physical probing action**. They are near-exact visual analogues of the Gate B numpy tasks — which, in hindsight, were evidently modelled on them.

Sensing is present in the released data: every episode carries `observation.force_torque` (6), `observation.force_torque_world` (3), and `observation.robot0_tactile_left` / `_right` (8x8 each). The paper's limitations section concedes: *"While we release force-torque and tactile signals, the evaluated baselines do not use tactile or force-feedback modalities."*

### Headroom on P5 — and a correction

The paper's per-task policy table (Table 15; columns pi0.5-ZS, pi0.5-FT, GR00T1.5-ZS, GR00T1.5-FT, DP-full-training):

| Task | pi0.5 ZS | pi0.5 FT | GR00T ZS | GR00T FT | DP |
|---|---|---|---|---|---|
| T25 mass-sort | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| T26 occluder-push | 0.00 | 0.15 | 0.00 | 0.00 | **0.50** |

So: **mass-sort is 0/5 across every baseline; occluder-push is *not* — Diffusion Policy reaches 50%.** (An earlier draft of this document claimed both were 0.00 everywhere; that was wrong and is corrected here.) The paper's own analysis: *"Diffusion Policy significantly outperforms VLAs on occluder-push (pi0.5 15%, GR00T 0%, DP 50%). VLA baselines frequently attempt to directly pick the occluding plates instead of performing clearing motions for active discovery."*

Meanwhile Stage-1 "understanding" accuracy on both P5 tasks is 90-100% for frontier VLMs, and Stage-3 action success is 0.0% even *with privileged object poses*. That is the same shape as this repo's CAI-JEPA finding (grounding up does not imply planning up), and it is large headroom. It is also a **warning**: mass-sort at 0/5 may be hard for reasons orthogonal to belief compression — contact-rich low-level control, an irreversible-failure success predicate (any wrong-zone placement ends the episode), and a 24.6 s mean episode. A planning-level contribution can be invisible under a control-level floor.

### Honest verdict on IMBench

- **Exists:** YES. Real CoRL 2026 anonymous submission with a publicly downloadable paper.
- **Real arXiv ID:** **NONE.** `2607.15641` is wrong; there is no arXiv version.
- **Mass Sort / Occluder Push exist:** **YES**, verbatim, as T25/T26 under P5 "Hidden State & Active Discovery". The prior analysis was substantively right about the tasks and wrong about the citation.
- **Released:** **NO, not usably.** Demos only (10/task). Environment code gated on acceptance; CoRL 2026 decisions land after our timeline.
- **Simulator:** **robosuite 1.5 / MuJoCo 3.1.2** — and robosuite itself *is* fully open (<https://github.com/ARISE-Initiative/robosuite>, v1.5.2 latest, actively maintained, ~2.5k stars).

**Conclusion: IMBench cannot be the substrate, but it is a usable *specification*.** The two P5 tasks are fully specified in the public PDF (object counts, mass values, randomization ranges, success predicates, sensing modality, canonical plan). Should we need a probe-and-commit manipulation task, reimplementing to their published spec is a materially stronger position than inventing our own — but it is still *our* code, and section 2.1 treats it as a fallback, not the primary.

---

## 1.2 Alternative substrates — verification

Every row below was checked against a live GitHub API call or a fetched page on 2026-07-30.

| Suite | Real? | Released? | arXiv / venue | Repo (status) | Pixels? | Genuine info-gathering **actions**? |
|---|---|---|---|---|---|---|
| **MIKASA-Robo / MIKASA-Robo-VLA** | Yes | **Yes, fully** | [2502.10550](https://arxiv.org/abs/2502.10550), **ICLR 2026** | `CognitiveAISystems/MIKASA-Robo` (MIT, 124 stars, pushed 2026-06-30), PyPI `mikasa-robo-suite` | Yes (ManiSkill3 RGB) | **Yes — BatteriesChecker family only** (see below) |
| **ManiSkill3** (base sim) | Yes | Yes | [2410.00425](https://arxiv.org/abs/2410.00425) | `mani-skill/ManiSkill` (Apache-2.0, 3174 stars, pushed 2026-07-29) | Yes, GPU-parallel rendering | No native POMDP tasks; **supports** mass/friction randomization and 1000s of per-env cameras with different extrinsics (mounted or fixed) |
| **POPGym Arcade** | Yes | **Yes, fully** | [2503.01450](https://arxiv.org/abs/2503.01450) | `bolt-research/popgym-arcade` (MIT, 34 stars, pushed 2026-07-23), PyPI | **Yes** (Atari-style pixels, JAX, ~10M FPS on a 4090) | **Yes — BattleShip, MineSweeper** are probe-and-reveal POMDPs. Ships **paired fully-observable / partially-observable variants** of every env. |
| **POPGym** (original) | Yes | Yes | [2303.01859](https://arxiv.org/abs/2303.01859), ICLR 2023 | `proroklab/popgym` (+ `popjaxrl`, `popjym`) | No (vector obs) | Yes (Concentration, MineSweeper, BattleShip) but non-visual |
| **Tactile MNIST** | Yes | **Yes, fully** | [2506.06361](https://arxiv.org/abs/2506.06361) | `TimSchneider42/tactile-mnist` (MIT, 15 stars, pushed 2026-07-28) | Tactile images (vision-based sensor) | **Yes — this is explicitly an *active tactile perception* benchmark**: localization, classification, pose and volume estimation by choosing where to touch. 13.5k synthetic 3D digits + 153.6k real tactile samples. Gymnasium API. |
| **APPLE** (active perception RL) | Yes | Code on project page | [2505.06182](https://arxiv.org/abs/2505.06182), **ICLR 2026** | <https://timschneider42.github.io/apple> | Yes | Method, not a benchmark — evaluated **on Tactile MNIST**. Best available *active-perception baseline*. |
| **robosuite / RoboCasa** | Yes | Yes | robosuite v1.5.2 (Dec 2024) | `ARISE-Initiative/robosuite` (2538 stars, pushed 2026-07-11) | Yes | **No** native partial-observability tasks. It is a *substrate*, not a POMDP benchmark. Already in this repo's orbit (RoboCasa is robosuite-based). |
| **Habitat** | Yes | Yes | — | `facebookresearch/habitat-lab` | Yes | Yes in the navigation sense (ObjectNav = exploration under occlusion), but the hidden state is *map geometry*, not a small hypothesis set with a decision signature, and the goal family is weakly structured. Poor fit for the compression pillar. |

### The critical finding in 1.2

Among established, released benchmarks, **exactly one manipulation task family requires a physical probing action to reveal a hidden property, then a commit**:

> **MIKASA-Robo `BatteriesChecker{Easy,Hard}-{3,6}-VLA-v0`** (memory type *Checklist*; episode lengths 540 / 1080 / 1080 / 2160):
> *"Find all working batteries by **inserting each one into the socket, observing the lamp result**, and then pressing the button to confirm."* (Hard variants additionally require returning each battery to its slot.)

This is structurally isomorphic to Gate B's `mass_sort`: N objects with a hidden binary property, observable only by an action; a terminal commit; irreversible cost for a wrong commit. It is published (ICLR 2026), MIT-licensed, GPU-parallel, ships PPO / motion-planning oracle trajectories on HF, and **we did not build it**.

The rest of MIKASA-Robo (verified against the shipped manifest `mikasa_robo_vla_envs.csv`, 90 rows, 10 memory types) is **memory**, not active discovery: `RememberColor*`, `RememberShape*`, `ShellGame*`, `FindImposter*`, `BunchOfColors*`, `ChainOfColors*`, `TraceShape*`, `BlinkCount*`, `TimedTransfer*`, `GatherAndRecall*` all follow "observe -> wait -> recall". The agent never chooses *what to find out*. **Do not misrepresent MIKASA-Robo as an information-gathering benchmark; it is one only in the BatteriesChecker family.**

### The gap no released benchmark closes

Our two surviving novelties need two properties in the same task:

- **(P) probing** — an action that reveals hidden state, so decision-regret VOI has something to value;
- **(G) a goal family** — several goals sharing the *same* hidden state, so multi-goal invariance of the compression is testable.

What actually exists:

| | has (P) | has (G) |
|---|---|---|
| MIKASA-Robo BatteriesChecker | YES | NO (single fixed goal) |
| POPGym Arcade BattleShip / MineSweeper | YES | NO |
| Tactile MNIST (classification / pose / volume) | YES | PARTIAL (three *tasks* over the same object, not a parameterized goal family within one episode) |
| MIKASA-Robo `ShellGameColorLampTouch` | NO (colors are shown, not probed) | YES — *"touch the cup matching the lamp color"*: **the lamp colour is a goal variable over a shared hidden state.** A textbook goal family. |
| POPGym Arcade `Navigator` | PARTIAL (exploration) | YES (different goal locations) |
| IMBench `mass-sort` (spec only, unreleased) | YES | YES (light/heavy zone assignments are a natural goal family) |

**No released benchmark gives (P) and (G) simultaneously.** That is an honest, load-bearing finding and it shapes Part 2: the two pillars get evaluated on *different established tasks*, and only the clearly-labelled reimplemented task carries both.

A second load-bearing warning from Gate C0. `BatteriesChecker`'s hidden state is an N-bit vector in which **every bit is decision-relevant** to the native goal ("report *all* working batteries"). That is precisely Gate C0's *control axis* (`mass_sort(N, all-relevant)`: log-log slope of M vs K = 1.000, M/K = 1.0). If our compression is measured against the *true* hidden state on that task, M = K and the method buys nothing — **by design, not by failure**. The learned-visual reframing that rescues it is that at visual scale **K is a property of the filter, not of the task**: K = the number of latent particles the belief model carries, which is a resolution knob exactly like `grid_param`'s. Gate C1 must test both readings and report both.

---

# PART 2 — Gate C1 plan (Task 2)

## 2.1 Substrate: primary, secondary, fallback

**PRIMARY — MIKASA-Robo (ManiSkill3): `BatteriesChecker{Easy,Hard}-{3,6}` + `ShellGameColorLampTouch` + `ShellGameShuffleColorLampTouch`.**

Justification, strictly from 1.2 evidence: it is the only *established, fully released, peer-reviewed (ICLR 2026)* pixel manipulation suite that contains a genuine probe-then-commit task (BatteriesChecker) **and**, separately, a genuine shared-hidden-state goal family (ShellGame-ColorLamp: same cup contents, goal = whichever colour the lamp shows). It runs on ManiSkill3, which is GPU-parallel — mandatory, because the primary metric is a compute frontier requiring ~1e7 env steps. It ships oracle trajectories, so the belief/world model can be trained from released data rather than from-scratch RL. MIT + Apache-2.0.

**SECONDARY (the scale axis) — POPGym Arcade: `BattleShip`, `MineSweeper`, `Navigator`, `CountRecall`, all three difficulties, POMDP *and* MDP variants.**

Why this is not merely a nice-to-have: Gate C0's headline is asymptotic (the ratio approaches K/M only as H grows, at K up to 1536). At robot-simulation cost we cannot afford K = 1536 x H = 4 x 6 methods x 5 seeds. POPGym Arcade is JAX-jitted at ~10M FPS, so it is the only place we can sweep K over two orders of magnitude and H to depth 5 **and** still get trajectory-clustered CIs. Its paired fully-observable variants give a free upper bound (the `fully_observed` row of Gate B) at zero extra engineering. It is pixel-based, so the belief is genuinely learned from images.

**FALLBACK / the one bespoke task — `MassSortProbe-v0`: IMBench T25 reimplemented to its published spec, in ManiSkill3.**

Used **only** to test the multi-goal pillar in manipulation, and only if the ShellGame-ColorLamp goal family proves too weak (risk R7). Why it may be unavoidable: 1.2 shows no released benchmark has (P) and (G) together, and (G) is one of Gate A's two surviving novelties. Mitigations against "designed to win", all pre-committed:

1. **Implement to someone else's published spec, not ours** — masses 50/100/200 g, three labelled zones, wrist F/T as the only mass channel, wrong-zone placement = immediate failure. Cite IMBench (Anonymous, CoRL 2026 submission, imbench.org) and state plainly that we reimplemented because their code is gated on acceptance.
2. **Swap-in clause**: if IMBench releases before submission, rerun on their environment and report both. Keep the observation/action interface deliberately compatible.
3. **Never report it alone.** Every headline claim must also hold on at least one established suite; the bespoke task appears in the same table, flagged, never as sole evidence.
4. **Ship an adversarial control**: `BatteriesChecker` (all-bits-relevant) is the pre-registered case where our method *should not* win. Publishing a task on which we predict failure, and then failing there, is the strongest available answer to "you built it to win". This mirrors Gate C0's control axis, which is what made C0 credible.
5. Release the environment code with the paper.

**Explicitly rejected:** IMBench itself (no code — 1.1); robosuite/RoboCasa alone (no POMDP tasks); Habitat (hidden state is map geometry, no compact hypothesis set or goal family); the original POPGym (no pixels).

**Considered, held in reserve:** Tactile MNIST + APPLE. Tactile MNIST is a real, released active-perception benchmark and APPLE (ICLR 2026) is the strongest published active-perception baseline on it. It is the natural *third* domain if reviewers push on generality, and the cheapest way to show the method is not manipulation-specific. Not primary because its hidden state (digit identity / pose / volume) has no natural multi-goal family, and its action space is a touch location, which makes the "planner compute" axis less comparable to robot MPC.

## 2.2 What must be LEARNED vs. what is BORROWED

> **The belief filter is NOT the novelty.** Gate A row 8 is explicit: RSSM/PlaNet/Dreamer, flow-based (FORBES), Stein-variational (ESCORT) and Wasserstein-Believer filters are commodity. Reinventing one is the single most likely way to burn the budget and lose the paper.

**BORROWED, unmodified, off the shelf:**

| Component | Source | Note |
|---|---|---|
| Belief model producing the K hypotheses | **DreamerV3** (`danijar/dreamerv3`, MIT, 3608 stars, pushed 2026-05-25) | Its latent is **categorical (32x32 discrete)**, not Gaussian — this matters: categorical posteriors are genuinely multimodal, which is what makes K diverse hypotheses possible at all. K hypotheses = K samples from the posterior/prior categorical, or K particles in a bootstrap filter using the RSSM as the proposal. |
| Alternative filter (robustness check only) | `NM512/dreamerv3-torch`, or `glambrechts/informed-dreamer` | One alternative filter is enough to show the result is not a DreamerV3 artefact. |
| Simulators | ManiSkill3 (`mani-skill/ManiSkill`), MIKASA-Robo, POPGym Arcade | Unmodified. |
| Demonstration data | MIKASA-Robo HF release (22.5k PPO / motion-planning oracle trajectories, RLDS + LeRobot v3, 6M+ transitions) | Removes the need to solve these tasks with RL from scratch. |
| Discrete active-inference reference | `pymdp` (`infer-actively/pymdp`, MIT, 723 stars, pushed 2026-07-24) | Sanity-check the EFE baseline on a tabular version of each task before trusting the deep version. |
| Compute accounting | this repo's `belief_compression` `ComputeCounter` | Extend to NN FLOPs; keep the Gate B/C0 discipline (never estimate a cell you cannot afford to measure — print `n/a`). |

**LEARNED / BUILT BY US (the contribution surface, and nothing else):**

1. **Goal-conditioned action-value head** `Q(z, a, g)` over the frozen belief latent, trained on the goal family. Needed to *define* a decision signature at all. Small; a UVFA-style head on the RSSM latent.
2. **Decision-signature extractor + clustering**: `sigma(z_k) = (rank of Q(z_k, ., g))_{g in G}` -> M modes, with the epsilon-tolerance sweep from Gate B. This is pillar 1, ported to a learned latent.
3. **Mode-representative / mode-value estimator.** Gate C0 S6 proved max-weight under-values and centroid over-values; neither is value-consistent. A *calibrated* within-mode value estimator (weight-respecting mixture, or a learned correction) is a required new component, not an optional refinement — probing decisions depend on value, not argmax.
4. **Amortized flip-predictor** `f(mode-structure, candidate probe) -> P(argmax action changes)`. Pillar 2. A small MLP/transformer over the M-mode features plus the candidate probing action, trained by supervision from the exact (expensive) VOI computed offline on a subsample.
5. **The compute-accounting harness** for the frontier (2.4).

## 2.3 Full baseline set

Every entry names a concrete implementation. Six baselines + six ablations; the ablations are where this gate is actually won or lost.

**Baselines**

| # | Baseline | Concrete implementation | What it tests |
|---|---|---|---|
| B1 | **RSSM / Dreamer reference agent** | `danijar/dreamerv3` actor-critic, unchanged | Can a strong commodity agent just solve it without any explicit belief-hypothesis machinery? If yes on all tasks, the whole framing is decoration. |
| B2 | **Active inference / expected free energy planner** | EFE objective (pragmatic + epistemic terms) over the *same* RSSM, CEM search; tabular cross-check with `pymdp` | Gate A flagged Ran Wei ([2408.06542](https://arxiv.org/abs/2408.06542)) as the strongest threat to pillar 2 — EFE's epistemic term is *already* a closed-form non-nested VOI surrogate. This is the head-to-head that decides whether pillar 2 survives. **Non-negotiable.** |
| B3 | **Full-particle-belief planner** | Expectimax/CEM over all K particles (Gate C0's `C_full`); plus POMCPOW / PFT-DPW from `POMDPs.jl` on a tabularized version as a sanity anchor | The quality ceiling and the compute denominator. Every frontier claim is stated relative to this. |
| B4 | **Entropy-seeking / information-gain probing** | (a) greedy max expected posterior-entropy-reduction; (b) **amortized** info-gain in the style of Deep Adaptive Design ([2103.02438](https://arxiv.org/abs/2103.02438)) — a design network trained on a contrastive info-gain bound | Gate B showed decision-VOI beats entropy-seeking on toy tasks; DAD is the *amortized* version and is the prior-art threat to "VOI without nested planning". Both variants required. |
| B5 | **Multimodal predictor without an explicit hypothesis set** | A diffusion/flow or categorical-mixture next-latent predictor + CEM; no particles, no clustering | Tests whether explicit K hypotheses are needed at all, or whether an implicitly multimodal predictor already captures it. |
| B6 | **Published active-perception method** | **APPLE** ([2505.06182](https://arxiv.org/abs/2505.06182), ICLR 2026) on Tactile MNIST; MIKASA-Robo's own reported PPO+memory baselines on MIKASA tasks | Anchors against a real published number rather than only against our own re-implementations (repo convention: *always report next to the published baseline*). |

**Ablations (each isolates one claimed mechanism)**

| # | Ablation | Isolates |
|---|---|---|
| A1 | **Compression without active probing** — plan over M modes, certainty-equivalent, no VOI | Does compression alone buy anything? |
| A2 | **Probing without compression** — exact decision-regret VOI over all K particles | The quality ceiling for the VOI pillar; the compute we claim to remove. |
| A3 | **Random subsample K -> M** — matched M, matched compute, no decision signature | **The single most important control.** If random M-particle subsampling matches decision-signature clustering, the criterion contributes nothing and the paper is "use fewer particles". Pre-registered as STOP condition S4. |
| A4 | **State-similarity clustering (k-means on z) -> M** | Direct head-to-head against Gate A's biggest prior-art threat (bisimulation / Ferns / DeepMDP / ANPL). Decision-signature clustering must beat latent k-means at matched M, or pillar 1 reduces to known state abstraction. |
| A5 | **Per-goal recomputation vs one shared partition** | Pillar 3 / multi-goal invariance (feeds G4). |
| A6 | **Signature cost accounted vs cached** | Gate C0 reported `C_comp` (signatures built at decision time, O(K x |G| x |A|)) and `C_cached` (precomputed once) as separate columns. At visual scale caching may be impossible. Both columns must appear; the honest headline uses `C_comp`. |

## 2.4 PRIMARY METRIC — the compute-vs-quality frontier

**Not** a success-rate table. The claim is scalability, so the deliverable is a Pareto front.

**x-axis — planning compute per decision.** Reported in three registers, all three always:

1. `C_prim` — hardware-independent primitive-op count, extending this repo's `ComputeCounter` (reward/Q evals, belief touches, expectimax nodes, VOI inner calls) **plus NN cost measured in FLOPs** (encoder forwards x FLOPs/forward, predictor rollouts x FLOPs/rollout).
2. `C_wall` — median wall-clock ms/decision on one fixed GPU, batched identically across methods.
3. **`C_split` — the decomposition `C_belief` (encoder + filter update) vs `C_search` (planning).** Mandatory, not optional: the sharpest way this whole program dies is that at visual scale the belief model dominates and the search saving is irrelevant. See STOP criterion S3.

**y-axis — quality.** Two curves per method:

1. Normalized task return (benchmark-native success predicate where one exists — MIKASA-Robo's own protocol; POPGym Arcade's standardized [0,1] / [-1,1] returns).
2. **Decision regret** vs. the B3 full-particle-belief planner run at maximum affordable compute, plus (POPGym Arcade only, free) regret vs. the *fully observable* MDP variant as an upper bound.

**Sweep grid.** `K in {8,16,32,64,128,256,512}` (POPGym Arcade extends to 2048); planner budget (CEM iterations x population) at 4 settings; `H in {1,2,3,4,5}`; 5 seeds; >= 100 evaluation episodes per cell; CIs are **trajectory-clustered bootstrap** (repo convention).

**Headline statistics** (these are what go in the abstract):

- **Compute at iso-quality**: total `C_prim` needed to reach 95% of B3's best return. Report the ratio `C_prim(B3) / C_prim(ours)`.
- **Quality at iso-compute**: return at matched `C_prim`.
- **Frontier slope**: log-log slope of the iso-quality compute ratio vs K. **A flat slope means a constant factor, which Gate C0 already called WEAK.** The claim is only alive if the ratio *grows*.
- **Horizon behaviour**: the (K, H) ratio grid, exactly as in Gate C0 S3b, since that table is where the asymptotic story lives.

## 2.5 Pre-registered GO / STOP criteria

Fixed before any run, in the style of Gates A/B/C0. All thresholds are on trajectory-clustered bootstrap 95% CIs.

### GO requires **all four**

- **G1 — compression is real, and it is *our* criterion doing the work.** On >= 2 of the 3 task families: `M/K <= 0.25` at `K >= 128`, with closed-loop return within **5%** of B3 (CI excluding a 10% drop). **And** decision-signature clustering must beat both A3 (random subsample -> M) and A4 (latent k-means -> M) at matched M with **non-overlapping CIs**. Without the A3/A4 margin, G1 fails even if M/K is tiny.
- **G2 — the compute win is asymptotic, not a constant.** At the largest feasible K: >= **5x** reduction in **total** `C_prim` (belief + search, using `C_comp` not `C_cached`) at iso-quality (>= 95% of B3's best return), **and** log-log slope of the iso-quality compute ratio vs K >= **0.3**.
- **G3 — the amortized VOI pillar holds.** Amortized flip-predictor VOI within **3%** of exact decision-regret VOI (A2) return, at >= **5x** lower compute; **and** beating B4 (both entropy variants, including the DAD-style amortized info-gain) *and* B2 (EFE) by >= **15% relative return** on the probing tasks, non-overlapping CIs. B2 is the discriminating comparison — Gate A says EFE already provides a non-nested VOI surrogate.
- **G4 — multi-goal invariance (the headline novelty).** One M-mode partition computed once for a goal family must hold on **held-out goals from that family** with <= **10%** regret inflation vs. per-goal recomputation (A5). Measured on ShellGame-ColorLamp (established) and, if used, `MassSortProbe-v0`.

### STOP if **any**

- **S1 — nothing to compress.** `M/K > 0.5` at `K >= 128` on all three families. Either the learned hypotheses are all decision-relevant, or (worse, see R1) they are not diverse at all.
- **S2 — constant factor only.** Iso-quality compute ratio `< 2x`, or its slope vs K `<= 0.1`. This is Gate C0's WEAK verdict recurring at visual scale, and it kills the scalability claim outright.
- **S3 — the filter eats the win.** `C_belief / (C_belief + C_search) >= 0.8` at the operating point *and* still >= 0.8 at the largest H. Then the search saving is irrelevant to end-to-end cost. (Pivot option, not a free pass: this would become a paper about belief-model cost, which is not the paper Gate A cleared.)
- **S4 — the criterion adds nothing.** A3 (random subsample -> M) matches decision-signature clustering within CI on >= 2 families.
- **S5 — mode collapse in the learned belief.** Effective hypothesis diversity (ESS of the particle weights, *and* the number of distinct decision signatures computed under the **oracle** hidden state) `< 2` on the probing tasks. Then the belief never represents the ambiguity and the premise is untestable with this filter. This is a **"fix the filter first"** stop, not necessarily a program kill — but it is a hard stop on reporting any Gate C1 number.
- **S6 — control task inverted.** If the method *wins* on `BatteriesChecker` under the all-bits-relevant reading, treat it as a bug, not a result: Gate C0's control axis predicts M = K there. An unexplained win means the compute accounting or the signature extraction is wrong.

### Conditional outcome (state it plainly; do not let it drift)

If G1-G3 pass but **G4 fails**, the paper loses one of its two surviving novelties. Gate A already established that *single-goal* regret-preserving belief simplification is covered by the Technion ANPL program ([2311.07745](https://arxiv.org/abs/2311.07745), [2310.01791](https://arxiv.org/abs/2310.01791)) and by bisimulation metrics. A G4 failure therefore downgrades the work to pillar 2 alone (amortized flip-VOI on compressed modes) — a workshop-scale contribution, not the paper. Decide explicitly at that point.

## 2.6 Engineering effort, GPU cost, and risks

### Effort (one person, honest)

| Phase | Work | Wall time | GPU |
|---|---|---|---|
| **P0 — diversity smoke test** (do this **first**; it can kill the gate for ~5 GPU-days) | Train/borrow one DreamerV3 on `BatteriesCheckerEasy-3` + `BattleShipEasy`; sample K latents; measure ESS and the number of distinct oracle-labelled decision signatures. This is STOP criterion S5, run before anything else is built. | 1 week | ~50 h |
| P1 | Env plumbing (ManiSkill3 + MIKASA-Robo + POPGym Arcade), oracle hidden-state logging, `ComputeCounter` extension to FLOPs, `C_split` accounting | 2 weeks | ~20 h |
| P2 | Belief/world models: DreamerV3 on 4-6 MIKASA tasks (from released oracle data) + 4 POPGym Arcade envs (cheap, JAX) | 3-4 weeks | 300-600 h |
| P3 | `Q(z,a,g)` head, signature extraction, calibrated mode-value estimator, flip-predictor training | 2 weeks | ~150 h |
| P4 | **The frontier sweep** — 6 baselines + 6 ablations x 7 K x 5 H x 5 seeds x 4 tasks x 100 episodes | 2-3 weeks | 300-500 h |
| P5 | Analysis, CIs, figures, `gateC1_results.md` | 1 week | ~20 h |
| **Total** | | **~10-12 weeks** | **~850-1350 GPU-hours** (A100-class); **budget 1500 h with buffer** |

Add `MassSortProbe-v0` (the IMBench-spec reimplementation) only if triggered: **+2 weeks, +150 GPU-h**, and it must include the scripted probing oracle and the success predicate exactly as specified.

### Biggest technical risks, worst first

- **R1 (BIGGEST) — the learned hypotheses may not be decision-diverse.** If the K particles from the belief model are near-identical, then M = 1, "compression" is trivially lossless and completely vacuous, and the frontier is meaningless. Symptom: regret vs B3 near 0 *and* regret vs the fully-observed upper bound enormous. Mitigations: DreamerV3's categorical latent (genuinely multimodal, unlike a Gaussian RSSM); an explicit bootstrap particle filter with a learned likelihood on the probing observation; and above all **P0 runs first** — this risk is cheap to falsify and expensive to discover late.
- **R2 — the O(K) belief read caps the win.** Gate C0 S3b already showed the ratio flattening in K at fixed H, because the compressed planner still reads all K particles once per decision. At visual scale the encoder forward is orders of magnitude more expensive per particle than a toy reward eval, so the cap binds much earlier. This is exactly STOP S3, and it is the most likely honest way this gate fails.
- **R3 — computing the signatures may cost more than they save.** A decision signature needs `Q(z_k, a, g)` for all K x |G| x |A|. Gate C0's flattering `C_cached` column assumed signatures precomputed once per task; at visual scale the belief is new every step, so caching may not apply. Mitigation: an amortized signature *predictor* (one forward pass per particle instead of |G|*|A| rollouts). Reporting rule: `C_comp` is the headline, `C_cached` is a footnote (A6).
- **R4 — decision fidelity is not value fidelity (Gate C0 S6, already proven on the toy).** Compression preserves the argmax, not the value; probing compares "commit now" against "commit after observing", which is a *value* comparison. Max-weight under-values, centroid over-values, structurally. With a *learned* cost this bias can dominate — and this repo has a long documented history of planners exploiting learned-cost error (the Phase-A / E0 reward-hacking findings). Mitigation: calibrated within-mode value estimator; always report both representatives.
- **R5 — "designed to win".** Mitigated by (i) making established suites primary, (ii) shipping the adversarial `BatteriesChecker` control where we *predict failure*, (iii) building any bespoke task to IMBench's published spec with a swap-in clause, (iv) the A3/A4 controls, which are the substantive version of the criticism.
- **R6 — MIKASA-Robo tasks may be too hard end-to-end.** BatteriesCheckerHard-6 has a 2160-step horizon. If no method — ours or B3 — gets off the floor, the frontier is measured in noise. Mitigation: use the Easy-3 / Easy-6 variants for the frontier; check against MIKASA-Robo's own reported baselines *before* committing; use their motion-planning oracle as a scripted low-level controller so the comparison is at the planning layer, not the control layer. (The IMBench mass-sort 0/5 result is the cautionary example: a planning contribution is invisible under a control-level floor.)
- **R7 — the goal family may be too thin.** ShellGame-ColorLamp has 3 cups, so |G| <= 3, and Gate B's `bound = |G|(|A|-1)+1` leaves little room to show *interesting* multi-goal invariance. If |G| cannot be made >= 4-8 without modifying the env, G4 is untestable on established tasks and the bespoke fallback is triggered. Check this in P1, not P4.
- **R8 — IMBench may release mid-project.** Upside, not downside. Keep `MassSortProbe-v0`'s interface aligned to their spec so their environment can be dropped in; if CoRL 2026 accepts, rerun and report both.

---

## Appendix — verification trail (all fetched 2026-07-30)

- IMBench site: <https://imbench.org/> ; bundle <https://imbench.org/assets/app.mRqi76AQ.js> ; paper <https://imbench.org/paper.pdf> (11,789,400 B, `application/pdf`) ; HF org <https://huggingface.co/imbench> (36 repos via `api/datasets?author=imbench`) ; example card <https://huggingface.co/datasets/imbench/pb-pr-mass-sort-v1>
- arXiv `2607.15641`: does **not** resolve to IMBench; no arXiv version of IMBench exists (no `arxiv.org` string in the site bundle or the paper other than ref [4] -> `2107.14483`, ManiSkill).
- MIKASA-Robo: <https://arxiv.org/abs/2502.10550> (ICLR 2026) ; <https://github.com/CognitiveAISystems/MIKASA-Robo> (MIT, 124 stars, pushed 2026-06-30) ; manifest `mikasa_robo_vla_envs.csv` (90 rows, read in full) ; docs <https://mikasarobo.github.io/>
- ManiSkill3: <https://arxiv.org/abs/2410.00425> ; <https://github.com/mani-skill/ManiSkill> (Apache-2.0, 3174 stars, pushed 2026-07-29)
- POPGym Arcade: <https://arxiv.org/abs/2503.01450> ; <https://github.com/bolt-research/popgym-arcade> (MIT, 34 stars, pushed 2026-07-23)
- POPGym: <https://arxiv.org/abs/2303.01859> (ICLR 2023)
- Tactile MNIST: <https://arxiv.org/abs/2506.06361> ; <https://github.com/TimSchneider42/tactile-mnist> (MIT, pushed 2026-07-28)
- APPLE: <https://arxiv.org/abs/2505.06182> (ICLR 2026) ; <https://timschneider42.github.io/apple>
- robosuite: <https://github.com/ARISE-Initiative/robosuite> (v1.5.2, latest release 2024-12-24, 2538 stars, pushed 2026-07-11)
- DreamerV3: <https://github.com/danijar/dreamerv3> (MIT, 3608 stars, pushed 2026-05-25)
- pymdp: <https://github.com/infer-actively/pymdp> (MIT, 723 stars, pushed 2026-07-24)
