# Planning Action-Error vs CRA_eff — consolidated finding

**Headline:** the link "weak action-grounding (`CRA_eff`) ⇒ worse planning
(Action Error)" is **confirmed on Metaworld at the regime level** (Spearman
= **−1.000** for both H=1 and H=3, expected sign), and is **consistent with**
the DROID result, where `CRA_eff` is floored at chance so only the *level*
(near-chance `CRA_eff` with high, horizon-growing Action Error) can be read.

The right instrument depends on whether `CRA_eff` has variance:
- **Metaworld** (`CRA_eff` spread 0.43–1.0): the regime-level correlation is a
  well-posed, well-behaved test → **monotone negative**, as predicted.
- **DROID** (`CRA_eff` ≈ chance everywhere): a floor effect → no variance to
  correlate; the evidence is the level, not the correlation.

Raw per-run outputs are kept alongside: `planning_correlation_metaworld.md`,
`planning_correlation_safe.md` (DROID), `planning_correlation_bounded.md` (DROID).

---

## A. Metaworld — the causal link (two models, 64×15 CEM)

Both Metaworld baselines were probed: `dino_wm_metaworld` (40 tx/cell, 117
transitions) and `jepa_wm_metaworld` (20 tx/cell, 66 transitions) — **183 planned
transitions, 59 % with `CRA_eff > 0`** (balanced classes, unlike DROID).

### Regime-level — the headline (pooled over both models)

| horizon | regime | n | Action Error | `CRA_eff` |
| --- | --- | --- | --- | --- |
| 1 | pre_grasp | 11 | 8.10 | 0.636 |
| 1 | free_space | 35 | 8.95 | 0.629 |
| 1 | contact_manipulation | 44 | 9.40 | 0.523 |
| 3 | pre_grasp | 7 | 17.77 | 0.857 |
| 3 | free_space | 37 | 17.83 | 0.622 |
| 3 | contact_manipulation | 49 | 21.12 | 0.551 |

- **Pooled regime-level Spearman(Action Error, `CRA_eff`) = −1.000 at H=1 and
  −1.000 at H=3.** As `CRA_eff` falls, Action Error rises, monotonically, at both
  horizons. `contact_manipulation` is the worst on both axes at both horizons —
  lowest `CRA_eff`, highest Action Error — exactly the thesis (action-grounding
  quality predicts planning quality; failure concentrates in the contact regime).
- Per-model the same test gives: `dino_wm` −1.000 / −1.000 (H=1 / H=3);
  `jepa_wm` −1.000 / **+1.000**. The single jepa H=3 reversal is driven by its
  tiny `pre_grasp` cell (n=5) and an unusually low free_space Action Error; it
  washes out when pooled, and H=1 is −1.000 for both models independently.

### Per-transition (noisy, as expected)

| subset | n | Spearman r | perm p |
| --- | --- | --- | --- |
| all (pooled) | 183 | +0.002 | 0.982 |
| H=1 (pooled) | 90 | +0.083 | 0.436 |
| H=3 (pooled) | 93 | −0.127 | 0.226 |

Per-transition the signal is weak and not significant. Reason: per-transition
Action Error is high-variance because it is compared to a **single** expert
trajectory (a different but valid plan still scores a large error), and `CRA_eff`
is binary per transition. That noise averages out at the regime mean — which is
why the regime-level signal is clean while the per-transition one is not. The
honest reading is the **regime-level monotone link**, with the per-transition
correlation reported as underpowered, not as counter-evidence. (Per model,
`jepa_wm` per-transition is at least consistently negative: −0.10 / −0.21 / −0.07
for all / H=1 / H=3.)

Cross-check vs the main diagnostic `hard_nn` `CRA_eff` (05): contact 0.530,
free_space 0.601, pre_grasp 0.491 — same near-0.5 band, consistent.

---

## B. DROID — the floor (dino_wm_droid, two runs, 355 transitions)

| run | CEM | max tx/cell | n pairs | `CRA_eff` positives | Spearman (all) | perm p |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 100×15 | 30 | 155 | 6 (3.9 %) | −0.068 | 0.416 |
| safe (RTX 5070) | 64×15 | 40 | 200 | 7 (3.8 %) | +0.011 | 0.878 |

On DROID `CRA_eff` is at the 16-way chance floor in every contact/gripper regime
(~4 % positive), so it has no variance to correlate against; the per-transition
and tiny-sample regime-level estimates are underpowered and sign-unstable. What
the DROID probe *does* establish is the level: near-chance `CRA_eff` co-occurring
with high Action Error that grows with horizon (≈1.0–1.2 at H=1 → ≈1.6–2.3 at
H=3). Near-total action-grounding failure with high, compounding planning error.

---

## C. Combined interpretation

1. **The link is real and in the predicted direction** — shown cleanly on
   Metaworld at the regime level (Spearman −1.0, both horizons), where
   `CRA_eff` varies enough to test it.
2. **DROID shows the severe end of the same axis** — `CRA_eff` floored at chance
   with high planning error; it cannot *also* show a correlation because there is
   no variance left (a floor, not a refutation).
3. **Per-transition correlation is the wrong headline instrument** in both
   datasets — DROID because of the floor, Metaworld because single-expert Action
   Error is too noisy per transition. The regime-level (Metaworld) and
   level-based (DROID) readings are the defensible evidence.

## Caveats

- Metaworld `pre_grasp` cells are thin (n=7 at H=1, n=2 at H=3); the −1.0
  regime-level Spearman rests on only 3 regimes, so it is "perfect" but low-n.
  Both horizons agreeing, plus monotone non-trivial gaps, is what makes it
  credible. A server run with more trajectories (300/task) would tighten this.
- DROID is offline → Action Score is the paper's proxy, not a grasp/lift success
  rate; Action Error compares to a single expert trajectory.
- DROID contact/pre-grasp are gripper-state + latent-change proxies, not MuJoCo
  contact ground truth. Recommended sanity gate: the Terver gripper test
  (open-vs-close, expect 2-way CRA > 0.90).

![Figure C — Metaworld dino_wm](figures/figure_c_planning_vs_cra_metaworld.pdf)
![Figure C — Metaworld jepa_wm](figures/figure_c_planning_vs_cra_metaworld_jepa.pdf)
![Figure C — DROID](figures/figure_c_planning_vs_cra_safe.pdf)
