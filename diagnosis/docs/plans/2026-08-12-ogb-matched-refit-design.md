# OGBench-Cube matched-refit: cost-only CEM intervention under exact dynamics

Date: 2026-08-12. Status: **locked protocol, execution in progress**.

## Decision being purchased

The paper's causal step — *using the latent cost to refit CEM makes later
candidates physically worse* — currently exists only on MetaWorld, where
DINO-WM and JEPA-WM share one frozen `dinov2_vits14` encoder. The OGBench-Cube
evidence (`2026-08-11-ogb-corrected-true-endpoint-design.md`) is a **fixed
population** audit: it shows residual misranking on a second, independently
trained encoder, but it never lets the misranking feed back into the search.

This experiment closes that gap:

> On the released LeWM OGBench-Cube stack, with the learned predictor bypassed
> and every candidate executed in the true simulator, does refitting CEM on the
> latent terminal cost produce physically worse candidates than refitting on a
> physical cost, holding the initial population and the sampling noise fixed?

If yes, the evidence chain (exact dynamics → same-candidate misranking →
matched-refit degrades proposals) holds on two independent representations
rather than one. If no, the causal claim stays MetaWorld-scoped and the paper
must say so.

## Relationship to the MetaWorld experiment

Mirrors `scripts/54_shared_population_branch.py` and its estimator in
`scripts/55_analyze_shared_population_branch.py`, with three deliberate
differences:

1. **Open loop, one snapshot.** MetaWorld forks branches at every MPC replan on
   a shared snapshot sequence. Here each evaluation snapshot is planned once,
   which is what the released LeWM evaluator's first solve does, and it removes
   the carrier-branch choice entirely.
2. **Released CEM update rule.** The refit is copied from
   `external/stable-worldmodel/.../solver/cem.py`: candidates are
   `mean + eps * var`, `var` is the elite **standard deviation** (not variance),
   candidate 0 is forced to the current mean, elites are `topk=30` of `300`.
   No variance floor, no clipping in normalized action space — the released
   stack has neither.
3. **Physical cost uses the gripper term.** See below.

## Locked protocol

### Snapshots

The 32 rows of `results/ogb_stage0/audit_locked/manifest.json`, unchanged, so
every number here is paired with the corrected true-endpoint audit on the same
starts. Unit of uncertainty is the snapshot.

### Costs

Both branches score exactly the same executed endpoints; only the scalar used
to pick elites differs.

- **latent** (the deployable-shaped proxy under test):
  `‖enc(render(o_T)) − enc(render(goal_state))‖²`, i.e. squared L2 to the
  **same-renderer** goal embedding. This is the primary arm of the corrected
  audit; the dataset-goal variant is recorded per candidate as a secondary
  column but never drives a refit.
- **physical** (positive control, not deployable):
  `‖cube_T − target‖ + w·‖pinch_T − cube_T‖` with `w = 0.5`, matching the
  MetaWorld shaped cost `‖o_T−o_g‖ + 0.5‖h_T−o_T‖`. `cube_T` is
  `joint("object_joint_0").qpos[:3]`, `pinch_T` is
  `site_xpos[_pinch_site_id]` — the same effector site OGBench reports as
  `proprio/effector_pos`.

The gripper term is included because a pure endpoint-distance control gives the
physical branch no gradient toward contact, which would understate the control
and weaken the comparison. Its weight is fixed at the MetaWorld value and is
not tuned.

### Reported quantities

Primary, per snapshot, at the final iteration, paired latent minus physical:

- `delta_best_task_distance` — difference in the branch's **best cube-to-target
  distance** available in its final population (the MetaWorld headline
  estimator, Table `tab:branch`, "final Δ best task distance");
- `delta_best_shaped_cost` — same for the shaped physical cost.

Secondary: difference in argmin-selected distance, in population mean distance,
and in successful-candidate availability; plus per-iteration Spearman and
top-10% recall of the latent cost against the physical cost, for both branches.

Aggregation is a 10,000-draw snapshot-clustered bootstrap of the paired
differences, plus the sign count across snapshots.

### Gates (a shard that fails any gate exits non-zero)

1. **Shared start.** Iteration-0 populations of the two branches must be
   bitwise identical arrays.
2. **Executor provenance.** Before planning, the shard re-executes the locked
   *final* CEM population for that snapshot and must reproduce, exactly, the
   corrected audit's `physical_distance_m`, `success`, and `executed_steps`,
   and reproduce `true_rendered_goal` cost within `1e-5`. This proves the new
   executor is the corrected one, not the retracted Stage-0 path.
3. **Repeat determinism.** The latent branch's final population is executed a
   second time; cube distance, hand distance, success, executed steps, and
   latent cost must match exactly.
4. Reset protocol is `76_ogb_true_endpoint_corrected.restore_complete`
   (`mj_resetData` → `set_state` → `set_target_pos` → `mj_forward` →
   finite-difference alignment → `post_step`), with solver warm-start disabled,
   imported rather than reimplemented.

### Budget

`horizon=5`, `action_block=5`, `num_samples=300`, `topk=30`, `var_scale=1.0`,
`cem_iterations=30` — the released LeWM planning budget. Per snapshot this is
`300 + 29 × 2 × 300 = 17,700` simulator rollouts plus gate rollouts. Smoke runs
use `--cem-iterations 6` on three snapshots to measure throughput before the
locked array is submitted.

## Scope and claim boundary

This tests one OGBench-Cube task with the released `quentinll/lewm-cube`
checkpoint and squared-L2 terminal scoring. It does not isolate encoder from
cost form, does not test other planners, and does not make simulator-assisted
selection a deployable method. A positive result licenses exactly one added
sentence: the matched-refit degradation replicates on a second, independently
trained representation.

## Execution ledger

Filled in as jobs complete.

| stage | script | job | status |
|---|---|---|---|
| smoke (snapshot 0, 3 iters) | `84_ogb_matched_refit.py` | `39156` | PASS, all 10 gates, 3.5 min |
| locked array (32 snapshots, 30 iters) | `84_ogb_matched_refit.py` | `39157_[0-31]` | submitted |
| aggregate | `85_ogb_matched_refit_aggregate.py` | | |

Measured throughput on the smoke: 0.6 min for the shared iteration-0
population, ~1.4 min per subsequent full iteration (two branches x 300
rollouts), so ~45 min per snapshot at 30 iterations. The smoke's provenance
gate reproduced the corrected audit's physical, success, executed-step, and
same-renderer latent arrays with max absolute difference exactly `0.0`.
