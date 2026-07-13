# RoboCasa experiments — results (2026-07-07)

Third dataset (PnPCounterTop / Pick-and-Place, custom teleop, **10 trajectories ≈ 4180
transitions**). Unblocked by setting `JEPAWM_DSET` (see `robocasa-jepawm-dset-fix` memory).
Model: `dino_wm_droid` (shares the 7-dim droid action format; 12→7 via
`rcasa_to_droid_action_format`). **Tertiary / robustness only** — small set, and the shared
DROID regime heuristic populated **only free_space + pre_grasp** (0 gripper/contact transitions),
so the contact-regime cell is `pre_grasp`.

## C1 — Diagnostic (CRA), `results/robocasa_diagnostic.csv`

The hard-NN grounding collapse **replicates** the MetaWorld/DROID pattern: near-perfect on
opposite/random distractors, at/near the chance floor (0.059) on nearby-action (hard_nn)
distractors.

| distractor | free_space CRA | pre_grasp CRA |
|---|---|---|
| opposite (easy) | 1.000 | 0.989 |
| random | 0.997 | 0.783 |
| **hard_nn (strict)** | **0.026** [0.005, 0.057] | **0.159** [0.132, 0.195] |

n = 2089 (free_space), 2091 (pre_grasp). Instrument sanity holds (opposite≈1.0); the drop to
hard_nn is the action-grounding failure. Only 2 regimes populated — honest limit vs the 4-regime
MetaWorld/DROID diagnostic.

## C6 — Counterfactual predictor A/B (second beat-baseline arena)

Frozen `dino_wm_droid` vs CF-LoRA, same CEM 64×15 / 40tx, pre_grasp H∈{1,3} (the only regime with
enough transitions to plan). Frozen baseline is **already strong** here (Action-Error 0.29 vs
DROID's 1.47 — PnP is in-distribution for the droid-trained WM), leaving little headroom.

| recipe | cf_rank_acc | recon | Action-Error ↓ | effect-CRA ↑ | Action-Score ↑ |
|---|---|---|---|---|---|
| frozen | — | — | 0.293 | 0.100 | 0.533 |
| **CF rank8/8ep (default)** | 0.24→0.29 | 0.86× | **0.281** (−4%) | **0.169** | **0.558** |
| CF rank16/24ep (stronger) | 0.24→0.32 | 1.41× (overfit) | 0.338 (WORSE) | 0.194 | 0.466 (WORSE) |

**Reading (honest):** at the DROID recipe (rank8) all three metrics improve, but **marginally** —
the large DROID gain (AE −26%, CRA→0.61) does **not** transfer to a small in-distribution set with
a strong frozen baseline. Pushing the objective harder (rank16/24ep) **overfits** the ~4k-transition
set (recon 1.41×): grounding (CRA) rises but planning (Action-Error, Action-Score) degrades — the
paper's own grounding↑↛planning gap, re-appearing. Two principled recipes tried; not chased further.

**Artifact note:** job 25016 (rank16) overwrote `results/robocasa_planning_cf_dino_wm_droid_lora.csv`
+ `checkpoints/predictor_cf_dino_wm_droid_robocasa.pt` with the *stronger/overfit* version. The
rank8 (better) numbers above are from the job-25013 log; re-run `slurm_phaseH_cf_robocasa.sh`
(default env) to regenerate the canonical rank8 CSV if needed for archival.

## Decision (2026-07-07)

Per user: **not folded into `paper/main.tex`** — results saved to `results/` + memory only. C1 is a
clean robustness add if wanted later; C6 is an honest weak-positive / scope limit. Jobs:
25012 (diagnostic), 25013 (CF rank8), 25016 (CF rank16).
