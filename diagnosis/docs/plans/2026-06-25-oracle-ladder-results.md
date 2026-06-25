# Oracle ladder — localizing the contact-task failure (2026-06-25)

Deconfounding experiment that asks *where* the contact-task closed-loop failure
comes from: the planner, the predictor `F`, or the cost/representation. All arms
share the same env (`scripts/18` `make_env`, frozen rand_vec), the same expert
goal, the same CEM / horizon 6 / `num_act_stepped` 3 / `max_episode_steps` 100,
and the same strict end-of-episode success judging. Only the *forward model* and
the *cost* change between rungs.

## Results (16 episodes/task, strict success)

| Task | WM arms (predictor `F`, L2-latent cost) | State-oracle (`scripts/29`) | Latent-oracle (`scripts/30`) |
|---|---|---|---|
| mw-push | **0/16** (l2 / l2inv / l2lora / l2lorainv) | **16/16** | **0/16** |
| mw-pick-place | **0/16** (all 4 arms) | **11/16** | **0/16** |
| mw-reach | 13–16/16 | 0/16 *(cost artifact†)* | 16/16 |

- **State-oracle** = perfect sim dynamics + perfect object-state cost (`‖obj−goal_obj‖`
  + hand→object approach). Removes both `F` and the encoder.
- **Latent-oracle** = perfect latent dynamics *through the real frozen encoder*
  (sim-step → render → encode) + the upstream **L2-in-DINO-latent** cost to `z_goal`.
  `F` is never called; only the cost/representation is the WM's.

CSVs: `results/metaworld_matrix_closed_loop.csv` (WM arms),
`results/metaworld_oracle_ceiling.csv` (state), `results/metaworld_latent_oracle.csv`
(latent).

## Causal reading

1. **Planner ruled out.** State-oracle solves push 16/16, pick-place 11/16 → the CEM
   budget, horizon, and success radius are adequate for contact under this protocol.
2. **Predictor not the sole bottleneck.** Latent-oracle has *perfect* latent dynamics
   (F removed) yet is 0/16 on contact → fixing `F` alone cannot, under the L2-latent
   cost, beat the baseline.
3. **The wall is the L2-in-DINO-latent cost geometry / encoder representation.** The
   latent distance to the goal frame has no usable minimum at task success for contact
   (latent-oracle `obj_goal_dist` stalls ~0.2–0.3 m); an object-grounded cost does
   (state-oracle). reach 16/16 on the latent-oracle confirms the protocol is sound.

† State-oracle reach = 0/16 is an artifact: its object-centric cost optimizes the
wrong target on a no-object task (reach success = hand→goal_pos). Not a real failure;
WM arms already solve reach. The latent-oracle (L2-latent cost) handles reach correctly.

## Comparison to the published baselines (required framing)

- **JEPA-WMs** (`world_model/jepa-success.pdf`, the model under diagnosis): Metaworld
  eval is **Reach + Reach-Wall only**; RoboCasa they "focus on Place & Reach" and
  **admit low success on the Pick task**. → there is **no published MW push/pick-place
  number** → MW contact is the controlled *microscope*, not a beat-baseline arena.
- **V-JEPA 2-AC** (`world_model/v-jepa2.pdf`, real Franka, 10 trials): reach 100%,
  grasp cup 65% / box ~25-30%, pick-&-place degraded — contact is the field-wide weak spot.
- **Beat-baseline arenas** = RoboCasa Pick (sim, object GT) and DROID Action-Score
  (`results/droid_planning_safe.csv`: `dino_wm_droid` contact action_score ≈0.60, cra_eff 0).

## Implied method + next step

The oracles imply the method = **grounded object-centric planning cost + action-responsive
dynamics**. Untested bridge cell: `hdynlora` (grounded object cost on the LoRA predictor),
running on MW contact as the method keystone. If it solves contact in the microscope,
port to RoboCasa Pick / DROID for the beat-baseline claim.
