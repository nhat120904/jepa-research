# Factorized object/hand cost ladder

Date: 2026-07-21. Status: implementation complete; awaiting Slurm execution.

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

- Code commit: pending.
- GPU array: pending.
- Analysis dependency: pending.
- Outputs: `diagnosis/results/factorized_cost_*_seed61000_n16.csv` and
  `diagnosis/results/factorized_cost_ladder_pilot.md`.
