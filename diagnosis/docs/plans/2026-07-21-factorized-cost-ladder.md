# Factorized object/hand cost ladder

Date: 2026-07-21. Status: **cancelled after a five-seed smoke audit**.

The run was stopped before the remaining array cells launched. A review against
the repository's existing Phase 3/C/G evidence showed that privileged channel
substitution would add only thin localization, while the proposed frozen-latent
ranker follow-on repeated a research direction already strongly predicted to
fail by robust-readout, adversarial-$\phi$, encoder-LoRA, and ensemble nulls. The
partial outputs are retained only as provenance and are not paper evidence.

## Question

Under simulator-perfect candidate dynamics, which decoded channel in the
stateprobe cost causes contact-planning failure: object xyz, end-effector xyz,
or their interaction?

The existing shaped cost is
`||object-goal|| + 0.5 ||hand-object||`. Here `hand` means end-effector xyz;
gripper aperture and contact state are not represented in this cost.

## Registered arms

| Arm | Endpoint object / goal | Endpoint hand |
|---|---|---|
| `decoded_both` | decoded / decoded | decoded |
| `true_object` | simulator / simulator | decoded |
| `true_hand` | decoded / decoded | simulator |
| `true_both` | simulator / simulator | simulator |

Changing the object channel also changes the goal-object coordinate so the
intervention replaces the complete object-information channel.

## Protocol and endpoints

- Two released checkpoints: DINO-WM and JEPA-WM MetaWorld.
- Tasks: `mw-push`, `mw-pick-place`.
- Fresh pilot seeds: `61000..61015`, paired across all arms.
- Oracle candidate dynamics, CEM `100 x 6`, horizon 6, execute 3 model steps,
  maximum 100 environment steps, strict endpoint success.
- Primary endpoint: `success_end`.
- Secondary endpoints: final object-goal distance and simulator-state shaped cost.
- Protocol gate: `true_both` outcomes must be identical across model-labelled
  cells because neither decoded channel enters that score; analysis aborts if not.
- Paired contrasts: each single-channel correction versus `decoded_both`, full
  correction versus `decoded_both`, and each missing channel conditional on the
  other channel being true.

The pilot is directional. A paper claim requires a fresh locked 64-seed
confirmation. The experiment localizes the representation--probe--cost
composition; it does not by itself distinguish information absent from the
encoder from information inaccessible to the probes.

## Execution record

- Code commit: `3ecd529` (`Add factorized object-hand oracle cost ladder`).
- GPU array: `28901`, submitted from `diagnosis/` with
  `sbatch scripts/slurm_factorized_cost_ladder.sh`.
- Analysis dependency: `28904`, submitted with
  `sbatch --dependency=afterok:28901 scripts/slurm_factorized_cost_analysis.sh`.
- Array configuration: tasks `0-15%2`, one GPU, 8 CPUs, 48 GB RAM, 24-hour
  limit per cell. Default pilot configuration is 16 episodes from seed 61000.
- Logs: `/mnt/data/nhatnc129/jepa_runs/logs/factorized_cost_28901_<task>.out`
  and `factorized_cost_analysis_28904.out`.
- Final state checked with both `squeue` and `sacct`: array tasks 0--1 were
  cancelled after about 29 minutes; tasks 2--15 never started; analysis `28904`
  was cancelled before execution.
- Partial smoke rows: seeds 61000--61004 only. DINO push `decoded_both` was 0/5;
  `true_object` was 2/5. These incomplete, adaptively inspected rows must not be
  interpreted inferentially or promoted into the paper.
- Outputs: `diagnosis/results/factorized_cost_*_seed61000_n16.csv` and
  `diagnosis/results/factorized_cost_ladder_pilot.md`.
