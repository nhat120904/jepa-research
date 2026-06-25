# Closed-loop Metaworld evaluation — L2 vs grounded cost (final report)

**Date:** 2026-06-12 · **Model:** `dino_wm_metaworld` (frozen public checkpoint)
· **Harness:** `scripts/18_closed_loop_eval.py` via `scripts/run_sweep_resume.ps1`
· **Data:** `results/metaworld_closed_loop.csv` (96 arm-episodes, 16 paired
episodes × 2 arms × 3 tasks, seeds 10000–10015, same env/rand_vec and CEM
noise per pair)

## Protocol

Replicates the upstream JEPA-WMs Metaworld evaluation read off the shipped
config (`base_configs/mw/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2.yaml`):
CEM-L2, horizon 6, 300 samples, 15 iterations, var_scale 1.0, 10 elites,
3 model-steps (15 raw actions) per replan, max 100 raw steps, goal frame =
scripted expert's **final** frame, one zero-action warmup step, horizon
shrink near episode end, `alpha=0` (no proprio term in the cost; proprio
stays in the unroll context — the predictor requires it), success = the
simulator's flag. Arms:

- **l2** — upstream planning objective (latent MSE to goal).
- **hdyn** — + β·grounded object term (β=0.1): integrate `h(z,a)→Δobject`
  (`checkpoints/object_dynamics_dino_wm_metaworld.pt`, cf-corr +0.682) along
  the unrolled plan, penalize distance to the goal's probed object position.

## Results

| task | arm | success | mean final state-dist | mean final ee-dist |
|---|---|---|---|---|
| mw-reach | l2 | any-step 15/16 (94%); **episode-end 6/16 (37.5%)** [18.5–61.4%] | 0.332 | 0.022 |
| mw-reach | hdyn | any-step 16/16 (100%); **episode-end 8/16 (50.0%)** [28.0–72.0%] | 0.412 | 0.031 |
| mw-push | l2 | 0/16 | 0.624 | 0.038 |
| mw-push | hdyn | 0/16 | **0.527** | 0.028 |
| mw-pick-place | l2 | 0/16 | 0.578 | 0.035 |
| mw-pick-place | hdyn | 0/16 | **0.496** | 0.021 |

Paired per-episode delta (l2 − hdyn on final state-dist; positive = grounded
ends closer; 10k-resample bootstrap over pairs):

| task | mean Δ | 95% CI |
|---|---|---|
| mw-push | +0.097 | [−0.014, +0.221] |
| mw-pick-place | +0.081 | [+0.007, +0.160] |
| **contact pooled (n=32)** | **+0.089** | **[+0.022, +0.162]** |

## Reading

1. **Reproduction: PASS, matches the published number (D.2 strict re-score,
   2026-06-23).** The headline table below uses an **any-step** success latch
   (TD-MPC2 convention: success if the env flag *ever* fires), which gives
   L2 15/16 = 94% — but the upstream/paper protocol judges at **episode end**.
   Re-scored strictly (`scripts/18 --strict-success`, same seeds 10000–10015,
   cluster EGL; `results/metaworld_reach_strict.csv`), the **episode-end**
   success is **L2 6/16 = 37.5% [Wilson 18.5–61.4%]**, grounded **hdyn 8/16 =
   50.0% [28.0–72.0%]**. Paper Table 1 (CEM-L2): DINO-WM MW-Reach **44.8 ±8.9
   → [35.9, 53.7%]**. Our strict L2 lands **inside the paper's CI** — the
   harness *reproduces* the published baseline, it does **not** beat it. The
   earlier "94% > 44.8%" was an artifact of the lenient any-step convention and
   has been retracted. (We evaluate the released checkpoint at 16 episodes; the
   paper averages 3 training seeds × 96 episodes — same ballpark, not a
   seed-matched replication.) This *strengthens* point 3 below: a harness that
   reproduces the published Reach number, then sees contact collapse to 0, is
   measuring a model failure, not a harness artifact.
2. **Contact tasks are 0% for BOTH arms — the closed-loop face of Boundary
   Blindness.** The arm reliably reaches the goal's end-effector pose
   (final ee 2–4 cm) while the object never moves to its target (state-dist
   ~0.5–0.6). The upstream paper saw the same thing qualitatively and said
   so (§"Object manipulation on Metaworld": the model "*hallucinates
   grasping the object*"; decoded plans show "*a gap between the imagined
   consequences of the actions and their consequences in the simulator*") —
   which is why its closed-loop Metaworld tables stop at Reach/Reach-Wall.
   Our BB gate measured that gap offline (pre-grasp bb_boundary 1.32 vs
   free-space 0.28); this sweep shows it closed-loop.
3. **The grounded cost helps, measurably but not success-flippingly:**
   no-harm on reach (100%), and on pooled contact tasks the paired
   improvement in final state distance is +0.089 [+0.022, +0.162] — the CI
   excludes zero. It does not produce successes: CEM rarely *samples* an
   action sequence that creates the contact the grounded term could then
   score. The remaining bottleneck is exploration/imagination at the contact
   boundary (planner-side), not the scoring metric — consistent with the fix
   ladder's conclusion that the predictor's counterfactual action→object
   channel is the broken piece.

## Reproduction pitfalls fixed on the way (all environment-side)

1. **Renderer used the default free camera at 480px** — metaworld builds its
   offscreen renderer in the constructor; assigning `env.camera_name`/`width`
   afterwards is ignored. Mirror upstream's `init_renderer()` (re-create
   `MujocoRenderer` at 224 with corner2).
2. **The training data is vertically flipped** relative to today's corner2
   render: pixel calibration vs dataset init frames scores `flipud(render)`
   at MSE 71.6 vs ≥3000 for every unflipped candidate
   (`results/logs/camera_calib7`). The upstream wrapper's `[::-1]` flip is
   part of the data pipeline — keep it.
3. **Goal frame must be the expert's final frame, not its first-success
   frame** — the flag fires on entering the 5 cm radius; the expert keeps
   refining to ~1 cm. Goal-at-first-success systematically parked the
   planner just outside the success radius (ee 2–4 cm, success=0).

Verification after fixes: one-step prediction error on rendered transitions
1.4× the dataset level (was 8.5×); rendered latents land on the dataset
manifold (NN ratio 0.97); `scripts/_baseline_probe.py`,
`scripts/_camera_calib.py`, `scripts/_replay_check.py` reproduce these
checks. Physics is identical (action replay: median ee error 1.5 mm).

Ops note: the sweep python occasionally dies natively (MuJoCo/driver on
Windows, no traceback). `18` resumes from its own CSV (whole pairs only —
rand_vec is per-env, so half-pairs are redone) and `run_sweep_resume.ps1`
relaunches until complete.
