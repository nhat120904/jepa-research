# Current research status (updated 2026-08-11)

This is the documentation entry point for the current project. The paper of
record is `../../paper/main.tex`; the claim-to-artifact ledger is
`CLAIMS_EVIDENCE.md`; and the Slurm record is
`JOB_LEDGER_2026-07-13.md`. Older ICLR reviews, dated plans, handoffs, and the
CAI-JEPA proposal are retained as provenance, not as the current submission
framing.

## Submission target and current thesis

The active target is **TMLR**. The paper is a mechanistic audit, not a new
planning method:

> Under exact simulator candidate dynamics, representation-derived terminal
> costs can remain near failure even when the same finite CEM budget succeeds
> with a simulator-state reference cost. For stateprobe, identical-population,
> matched-null, and cost-only refitting interventions establish harmful
> optimizer-conditioned misranking. The matched-null excess is specific to the
> operational shaped cost; the refitting intervention also worsens direct
> object-to-goal task distance. Exact-success candidates are sparse in
> final proxy-guided populations, but that observation does not independently
> identify a proposal-generation failure.

The claim is scoped to two released MetaWorld checkpoints, push and pick-place,
the tested costs, and the reported planner protocol.

## Submission evidence

### Terminal-cost comparison under oracle dynamics

Fresh paired environment seeds `30000--30063`, strict episode-end success:

| cost | DINO push | DINO pick | JEPA push | JEPA pick |
|---|---:|---:|---:|---:|
| simulator-state reference | 64/64 | 49/64 | 64/64 | 49/64 |
| latent L2 | 0/64 | 0/64 | 0/64 | 0/64 |
| stateprobe | 5/64 | 0/64 | 1/64 | 1/64 |

The reference row is shared across representations. It is a privileged
positive control for the search budget, not a deployable method. This
comparison shows that learned-dynamics error is not necessary for failure in
this harness; it does not isolate an encoder defect.

### Stateprobe mechanism

- On identical iteration-0 populations, stateprobe selection incurs
  **2.05--2.48 cm** within-population reference-cost regret.
- A residual-permutation null with the same marginal errors incurs
  **1.07--1.15 cm**; actual-minus-null is **0.91--1.36 cm**, with all four
  seed-clustered confidence intervals above zero.
- Changing only the CEM refitting cost makes the stateprobe branch's best
  available candidate **0.84--1.55 cm worse by direct object-to-goal distance**
  and **1.51--2.01 cm worse by shaped reference cost** after five refits.
  Task-distance seed means point in the expected direction in 7/8, 7/8, 8/8,
  and 7/8 seeds; the exact sign test is below 0.05 only for JEPA push, while all
  seed-bootstrap intervals exclude zero.
- Re-evaluating the residual-shuffle null by direct task distance gives positive
  actual physical regret in every cell but only **−0.06 to +0.08 cm**
  actual-minus-null; only JEPA push excludes zero. Do not generalize the
  shaped-cost structured-residual claim to task return.

The detailed cross-task selection and refitting claims apply to **stateprobe**.
Latent L2 is supported by the end-to-end terminal-cost comparison, the
same-population ranking audit below, and one targeted DINO-push branch causal
audit; the latter is not a cross-task or independent-encoder replication.

### Latent L2 same-population ranking audit

Folded into the paper (§4.4/Table 4) from `results/cem_preselection_audit.md`
(`*_l2` arms, script 53). L2 fails by a different mechanism than stateprobe:

| cell | rho init | rho final | recall init | recall final | R_sel init |
|---|---:|---:|---:|---:|---:|
| DINO push | 0.25 | −0.08 | 0.16 | 0.02 | 2.98 cm |
| DINO pick | 0.02 | −0.09 | 0.07 | 0.01 | 3.74 cm |
| JEPA push | 0.27 | −0.08 | 0.16 | 0.02 | 3.01 cm |
| JEPA pick | −0.02 | −0.12 | 0.06 | 0.01 | 3.96 cm |

Chance recall is 0.10. Initial rho covers zero on both pick cells; final rho is
CI-clean negative in all four. Initial R_sel is **larger** than stateprobe's
2.05--2.48 cm.

Claim discipline for this row:

- Say **"weak, task-dependent initial ordering that becomes anti-aligned under
  search,"** not "absent ordering." Push retains rho=0.25/0.27 and recall 0.16,
  which is weak but not absent.
- On pick, failure is evident **before** refitting. On DINO push, the targeted
  L2-versus-reference branch makes the final best candidate **2.04 cm
  [1.58, 2.51]** worse in task distance and **3.64 cm [2.83, 4.36]** worse in
  shaped cost, with all 8 seed means positive (exact sign and sign-flip
  p=0.0078). This is one representative cell, not a cross-task claim.
- Report **only rho and R_sel** for L2. Script 53 line 59 reads
  `stateprobe_optimism_m` and `obj_decode_error_cm` in *every* arm, so
  `proxy_elite_optimism_enrichment_m` / `proxy_elite_enrichment_cm` on L2 rows
  are stateprobe quantities, not L2 ones; optimism `c* − chat` is also not
  dimensionally meaningful when chat is not in metres.

### Matched-snapshot coverage control

Paper §4.6/Table 7, from `results/shared_population_branch_audit.md`
(`coverage_success_end`, iter 5). Stateprobe-refit 42.5/16.2/34.4/8.3% vs
reference-refit 47.5/18.7/47.5/15.1%; paired within-run difference −2.5 to
−13.1 points, CI-clean only for JEPA push.

- The difference column is a **paired within-run** contrast and is valid.
- Do **not** call the two 47.5% push entries a protocol check. The
  reference-refit branch is not identical across model cells: the pick cells
  differ at **iteration 0** already (0.0595 vs 0.0417), so cross-cell equality
  is not a demonstrated harness invariant. Separate runs do not share seeds or
  base noise, so the reference-refit column is not comparable across rows.
- Do **not** say the cross-experiment gap (42.5% vs Table 6's 8.0%) attributes
  "most" of the deficit to the trajectory. Disjoint seeds (42000--42007 vs
  40000--40015) and different snapshot-generating processes mean the gap cannot
  be decomposed. Permitted wording: limited within-snapshot refitting effect,
  plus an additional state-distribution effect of unidentified magnitude.

### Held-out scalar-cost validation

The complete deployed stateprobe scalar composition is now evaluated on the
immutable expert-trajectory split. On the 11 held-out push/pick-place
trajectories (209 transitions), DINO/JEPA scalar MAE is **2.43/2.84 cm** and
trajectory-mean Spearman is **0.800/0.826**; the all-manipulation Spearman
aggregate over 67 trajectories is **0.832/0.822**. See
`results/stateprobe_scalar_validation_{dino,jepa}_*`, script 79, jobs
`38668/38671`. This is expert-distribution validation for one fixed probe
training seed, not a substitute for the CEM candidate audit.

### Stateprobe validation

The exact probe checkpoints used by planning have held-out expert-trajectory
object coordinate RMSE **1.64--1.71 cm** and hand coordinate RMSE
**3.75--4.31 cm**. On the optimizer-induced candidate populations, the initial
stateprobe/reference shaped-cost Spearman is **0.43--0.55** with reference
top-10 recall **0.19--0.26**. In final proxy-guided populations, Spearman is
**0.11--0.16** and recall **0.07--0.10**. Each cell/stage contains 112
populations and 11,200 candidates clustered over 16 episode seeds. See
`results/stateprobe_cem_validation.md` and script 63.

This validates and characterizes the fixed probe rather than claiming it is an
optimal decoder. The hand readout is weaker than the object readout, and no
architecture or probe-training-seed sensitivity has been established.

### Candidate availability

In final stateprobe-guided CEM populations, exact-success availability is
**8.0%** for DINO push, **3.6%** for DINO pick, and **0%** in both JEPA cells.
Positive physical selection regret remains on the same populations.

Call this **sparse exact-success candidate availability under proxy-guided
search**, not “proposal-coverage failure.” The metric is measured after prior
proxy refits, with horizon `H=6`, and at encountered simulator snapshots.
Therefore it cannot separately assign the absence to snapshot feasibility,
horizon, initial proposals, or earlier proxy selection.

## Claim discipline

- Say **“grounding alone is insufficient,”** not “grounding is necessary.”
- Say **“the tested representation-derived costs,”** not all latent costs.
- Say **“learned-dynamics error is not necessary for failure in this
  harness,”** not that the planner or model is an adversary.
- Do not claim that CEM “exploits model error”: candidate dynamics are exact in
  the headline audit. The supported mechanism is stateprobe cost misranking
  under selection and adaptive refitting.
- Do not use “encoder exploitation” as a unique attribution. Stateprobe
  measures the representation--readout--cost composition.
- Do not call final-population availability an independent proposal failure.
- Do not generalize across planners from endpoint controls: the
  identical-population and refitting mechanisms were tested only for CEM.
- Uncertainty over environment seeds is not uncertainty over checkpoint
  training seeds.

## TRM-style adaptation

The TRM-style replacement and hybrid artifacts remain in `results/` for
research provenance. `scripts/52_analyze_trm.py` summarizes strict endpoint
field `success_end`; the prior report used the any-step `success` latch and was
inconsistent for DINO stateprobe pick (`1/64` any-step versus `0/64` endpoint).

The empirical TRM adaptation is **excluded from the TMLR paper**. The current
artifacts do not include enough adaptation-specific ranking/training and
positive-control validation to interpret a null planning result as a method
boundary rather than an implementation/adaptation failure. TRM remains in
Related Work.

## OGBench-Cube cross-substrate audit (supporting evidence in the paper)

A fresh renderer-controlled audit now provides independent OGBench support for
the cost-misranking phenomenon. On one released `quentinll/lewm-cube`
checkpoint, the exact same 300 final CEM candidates at 32 snapshots give:

- learned predicted-endpoint latent L2: 16/32 success, 7.80 cm regret;
- reproducibly rendered true endpoint plus same-renderer exact goal: 21/32
  success, **3.29 cm [1.49, 5.67]** residual regret;
- physical best candidate: 25/32 available.

All 32 shards pass independent-world exact pixel/physics/encoder-repeat gates;
the worst same-state encoder domain ratio is `6.59e-4` against a precommitted
0.25 maximum. This supports a scoped claim about the released encoder **plus
terminal squared-L2**, not an encoder-only or universal representation claim.
The historical Stage-0 candidate artifacts remain retracted; use only
`results/ogb_true_endpoint_corrected/`. The PFCG mitigation remains a locked
no-go. The scoped result is included in `paper/main.tex` as a cross-substrate
same-population replication and is explicitly distinguished from the causal
MetaWorld stateprobe refitting intervention.

## Historical and supporting work

The earlier CAI-JEPA/action-identifiability/Boundary-Blindness program remains
useful provenance but is not the submission spine. `hard_nn` CRA uses nearby
cross-state actions and is observational unless validated by exact same-state
interventions. DROID scaling, Phase-H, mitigation grids, ACID-style
approximations, and selection-aware method experiments remain supporting or
historical artifacts and are excluded from the TMLR empirical narrative.

The planner endpoint control is in the appendix and was strengthened
2026-08-11 (job `38698`) with a push physical-reference arm per optimizer,
closing the "maybe MPPI/shooting just can't solve push" gap: on DINO push
under exact dynamics, stateprobe succeeds with CEM `1/16`, MPPI `0/16`, and
random shooting `2/16`; latent $L_2$ is `0/16` under MPPI and `0/16` under
shooting; the privileged physical-reference cost, run under the same
optimizers/seeds/budget, is `16/16` (CEM), `16/16` (MPPI), and `11/16`
(shooting). Their free-space Reach controls are `16/16`, `13/16`, and `6/16`.
Because the physical reference converts the same search into success under
every tested optimizer, the contact-task failure is not explained by planner
capacity alone. This still does not establish the full candidate-level
ranking/refitting mechanism generality (scripts 53-55 remain CEM-only).

## Generality extension (task breadth complete and included)

Post-submission review argued the paper is an existence-and-mechanism case
study (two released stacks x two tasks), not a prevalence/generality claim, and asked
for staged extension. Full design and pre-registration:
`docs/plans/2026-08-04-generality-extension-design.md`. The earlier instruction
not to edit the paper during that pass was superseded by the explicit
2026-08-11 paper-update request. The shared-encoder disclosure, completed task-
breadth ladder, and independent OGBench substrate are now in `paper/main.tex`.

### Shared-encoder finding (changes how "n=2 checkpoints" should be read)

`dino_wm_metaworld` and `jepa_wm_metaworld` both use a **frozen
`dinov2_vits14`** visual encoder (`external/jepa-wms/configs/evals/
simu_env_planning/mw/{dino-wm,jepa-wm}/*.yaml`, identical `enc_version`,
`pretrain_enc_path:` empty in both; frozen at
`external/jepa-wms/app/vjepa_wm/utils.py:685-692`). Checked every upstream
environment family (metaworld, wall, pusht, point-maze, droid, robocasa) --
**all** `dino-wm`/`jepa-wm` arms share this same frozen encoder; no exception.
Under exact dynamics the predictor is never called
(`scripts/30_latent_oracle.py:12`), so the two checkpoints' latent-$L_2$ arm
differs only by an affine input rescale. Confirmed empirically: `dino_wm` and
`jepa_wm` L2 mean object-goal distance on mw-pick-place both equal
26.6314 cm to 4 decimal places (`results/confirmatory_{dino,jepa}
_wm_metaworld_l2_mw-pick-place_seed20000_n64.csv`).

**On MetaWorld the paper has n=1 representation, not n=2.** This is now
disclosed in the main Experimental Design and Limitations sections. The
corrected OGBench LeWM audit supplies one independent end-to-end encoder
substrate, with one-checkpoint/one-task scope.

The one genuinely independent encoder anywhere in the upstream registry is
`vjepa2_ac_droid`/`vjepa2_ac_oss` (V-JEPA-2 ViT-Giant, `embed_dim: 1408`,
`enc_type: vjepa` -- confirmed distinct by `results/droid_scaling_curve.md`),
but it only runs on DROID (real-robot video, no simulator), so it cannot
support exact-dynamics snapshot/restore. Pairing it with a real simulator
(RoboCasa is the only in-repo candidate that loads it, per
`configs/diagnostic_robocasa.yaml:15,17`) would require building a new oracle
harness from scratch -- `snapshot()`/`restore()`
(`scripts/29_oracle_ceiling.py:62-90`) and `make_env`/expert-policy lookup
are all MetaWorld/Sawyer-specific, not generic MuJoCo. **Track B
(representation independence) is paused by user decision** pending a scoped
harness-effort estimate; not attempted in this pass.

### Task-breadth extension (Track A) -- complete and in the paper

Gate 1 (scripted-expert competence, `scripts/70_task_breadth_expert_check.py`,
job 35500): 5/6 candidate tasks ELIGIBLE (door-open, drawer-close,
button-press, window-close, assembly); `mw-peg-insert-side` INELIGIBLE-BUDGET
(69% success within the production 100-step cap vs the pre-registered 75%
threshold, confirming the earlier signal in
`results/metaworld_precision_ladder.csv`) and excluded from this pass.
`results/task_breadth_expert_check.{csv,md}`.

Gate 2 (positive control) + Stage 1 terminal-cost ladder (job `35508_[0-9]`,
all ten tasks completed exit 0; 5 tasks x {oracle, l2}, 16 episodes,
seed0=70000) gives:

- button-press: reference 16/16, latent L2 0/16;
- drawer-close: reference 16/16, latent L2 **16/16 boundary**;
- window-close: reference 16/16, latent L2 0/16;
- door-open: reference 2/16, latent L2 0/16, excluded from cost
  interpretation because the positive control is weak;
- assembly: reference 0/16, latent L2 0/16, excluded because the positive
  control fails.

Only `dino_wm_metaworld` runs the L2 arm because the second released stack
shares the encoder. All rows, including the exclusions, are now reported in
the paper's task-breadth table.

The preregistered reporting rule is satisfied: drawer-close is reported as a
boundary condition rather than discarded.

## Second review-response pass (2026-08-11, later same day)

An independent re-read of the 2026-08-11 revision flagged: (1) an inaccurate
Abstract/Introduction claim that CEM branch-intervention candidates are "held
fixed" past the initial population (they are not -- only the initial
population, snapshots, and base sampling noise are shared); (2) the OGBench
sentence reading as if the full degradation-under-search mechanism was
reproduced on a second encoder, when only fixed-candidate reranking was; (3)
Appendix E's planner-robustness control lacking a push physical-reference arm
for MPPI/shooting, so a reviewer could attribute the near-failure to planner
capacity rather than cost; (4) the latent-$L_2$ branch intervention still
being n=8, push-only; (5) "preregistered" implying an external, hash/timestamp
registry that does not exist; (6) residual reviewer-defensive phrasing
("intentionally scoped", "we report rather than resolve").

Text fixes (1/2/5/6) were applied directly. Items (3) and (4) were closed with
fresh Slurm jobs the same day:

- Job `38698` (`scripts/slurm_planner_generality_push_reference.sh`) added a
  `--planner {mppi,shooting}` flag to `scripts/29_oracle_ceiling.py` (mirroring
  the switch already in `scripts/30_latent_oracle.py`) and ran the push
  physical-reference and latent-$L_2$ arms under MPPI/shooting at the existing
  seed0=10000/n=16 convention. Result folded into Appendix~E (see above).
- Jobs `38699` (array: push n=16 at seed0=42000 superset of the original 8;
  pick-place n=16 at fresh seed0=44000) and `38746` (`scripts/78_analyze_
  task_aligned_cost_audit.py` re-run with the new candidate files,
  `results/task_aligned_cost_audit_n16*`) extended the latent-$L_2$ cost-only
  refitting intervention. Result folded into paper Table 6; see `CLAIMS_
  EVIDENCE.md` row 5.12 (updated) for the full numbers.

## Submission readiness

The 2026-08-11 source revision adds the shared-encoder disclosure, completed
task breadth, corrected OGBench replication, the second review-response pass
above, the strengthened planner-robustness control, and the extended
latent-$L_2$ branch intervention (push n=16, new pick-place n=16 cell). Final
paper-build job `38391` (pre-dates this pass) completed exit 0: 17 US-Letter
pages, no undefined citations/references, overfull boxes, underfull boxes, or
final-pass LaTeX warnings; the paper has been rebuilt locally since with no
new warnings (15 pages under the current TMLR template).

Remaining author-side submission checks:

1. Package an anonymized supplement containing the relevant analysis scripts,
   manifests, per-seed summaries, and exact commands.
2. Perform the final author-side OpenReview anonymity and metadata check.

All GPU, simulator, model, large-data analysis, and paper builds must run
through Slurm. The login node is limited to lightweight inspection, editing,
syntax checks, and job monitoring.
