# Current research status (2026-07-30)

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
> optimizer-conditioned misranking. Exact-success candidates are also sparse in
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
  available reference cost **1.51--2.01 cm** worse after five refits.
  The seed-level mean direction is positive in **8/8** seeds in every cell
  (two-sided exact sign test `p=0.0078` per cell).

These detailed selection and refitting claims apply to **stateprobe only**.
Latent L2 is supported by the end-to-end terminal-cost comparison **and by the
same-population ranking audit below**, but not by the branch causal audit.

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
- On pick, failure is evident **before** refitting. On push, whether refitting
  causally contributes is **unresolved** — there is no L2 refit branch.
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

### Open validation item

The deployed stateprobe **scalar** cost has never been evaluated end-to-end on
the held-out expert split; only its two channels have. The 2.1 cm object-channel
figure neither estimates nor bounds the scalar cost's error, since the channels
enter through two norms and their errors may cancel or compound. The paper title
is scoped to **per-channel** decoding accuracy for this reason.

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

## Historical and supporting work

The earlier CAI-JEPA/action-identifiability/Boundary-Blindness program remains
useful provenance but is not the submission spine. `hard_nn` CRA uses nearby
cross-state actions and is observational unless validated by exact same-state
interventions. DROID scaling, Phase-H, mitigation grids, ACID-style
approximations, and selection-aware method experiments remain supporting or
historical artifacts and are excluded from the TMLR empirical narrative.

The planner endpoint control remains in the appendix: on DINO push under exact
dynamics, stateprobe succeeds with CEM `1/16`, MPPI `0/16`, and random shooting
`2/16`; their Reach controls are `16/16`, `13/16`, and `6/16`. This shows the
contact outcome is not observed only with CEM, but does not establish
mechanism generality.

## Submission readiness

Review-driven source fixes are complete. Job `33069` regenerated the TRM
report/JSON with strict endpoint success. Job `33071` built the revised
TMLR PDF: 9 US-Letter pages, embedded fonts, no undefined citation/reference,
and no overfull box. The audit figure and protocol table were visually checked.

Remaining author-side submission checks:

1. Verify metadata for the 2025--2026 citations and the current TMLR LLM-use
   wording.
2. Package an anonymized supplement containing the relevant analysis scripts,
   manifests, per-seed summaries, and exact commands.

All GPU, simulator, model, large-data analysis, and paper builds must run
through Slurm. The login node is limited to lightweight inspection, editing,
syntax checks, and job monitoring.
