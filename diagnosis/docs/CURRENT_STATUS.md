# Current research status (2026-07-16)

This is the documentation entry point for the current project. The paper of
record is `../../paper/main.tex`; the claim-to-artifact ledger is
`CLAIMS_EVIDENCE.md`; the current Slurm submission record is
`JOB_LEDGER_2026-07-13.md`; the venue-level review is
`ICLR_READINESS_REVIEW_2026-07-13.md`; the current primary-source novelty audit
is `plans/2026-07-13-iclr-literature-positioning-audit.md`. Dated plans,
handoffs, and the original CAI-JEPA proposal
are retained as research provenance, not as current claims or execution plans.

## Current thesis

The strongest defensible result is about the **optimizer--cost interface** in
latent world-model planning:

> Under oracle-perfect dynamics, true simulator-state cost solves contact tasks
> while every tested representation-derived cost remains near failure. On
> identical candidate populations, the learned state cost misranks physical
> progress beyond a matched noisy-score null, and changing only the refitting
> cost changes the physical quality of later populations. Proposal coverage is
> also sparse, so the supported mechanism is a joint coverage-and-ranking
> bottleneck rather than a single encoder defect.

The locked oracle ladder uses 64 unseen paired seeds: privileged state cost
solves push **63/64** and pick-place **41/64**; DINO-WM stateprobe solves
**4/64, 0/64**, JEPA-WM stateprobe **1/64, 0/64**, and latent L2 **0/64** in all
four contact cells. On the first identical population, proxy selection incurs
**2.05--2.47 cm** physical regret. Its matched-residual permutation null incurs
only **1.07--1.15 cm**; actual-minus-null is **0.91--1.36 cm**, with all four
seed-clustered CIs above zero. The shared-population branch then shows
**1.51--2.01 cm** worse best physical cost after five proxy-based refits. These
simulator-state results are the paper's current empirical core.

The earlier **CAI-JEPA / action-identifiability / Boundary Blindness** program
was productive as a diagnostic precursor, but it is no longer the primary
novelty claim. CRA with `hard_nn` negatives is an **observational action-
discriminability diagnostic**: negatives come from other observed states, not
from interventions executed from the exact same simulator state. It therefore
does not by itself identify a causal action effect, and its nominal `1/17`
chance interpretation requires care because factual and negative candidates are
not exchangeable.

## Claim discipline

- Say **"grounding alone is insufficient"**, not "grounding is necessary and
  insufficient." No current intervention establishes necessity.
- Say **"every frozen-encoder cost tested"**, not every possible latent cost.
- Say **"no scale trend under the current model-native diagnostic"**, not
  "scale is not the fix." Current scaling cells use model-specific latent
  effect masks and nearest neighbours.
- Treat the same-population regret, permutation null, and shared-branch
  intervention as the primary evidence. Treat the Goodhart budget curve as
  supporting evidence of proxy/true-objective
  decoupling. Within-search decode-error growth is a point-estimate trend with
  overlapping confidence intervals, not a monotonic law already established.
- Keep Push-T and PointMaze as sanity checks only. Metaworld provides simulator
  verification; DROID is external-scale/real-robot diagnostic evidence, not
  causal ground truth.
- Separate an **informative privileged control** (tracks truth and supports
  optimization), a **no-signal** cost (small proxy--truth gap can still be
  useless), and a **misranking** cost (selects worse physical candidates than
  available alternatives). Do not use “encoder exploitation” as a unique
  causal attribution: the measured object is the representation--readout--cost
  composition under search.

## Phase H: held-out result completed, supporting only

The previously reported predictor-LoRA / counterfactual-objective numbers were
not a held-out planning result: the planning probe could read the complete
latent cache. They remain exploratory and are excluded from current claims.

The corrected implementation uses an immutable 70/15/15 trajectory manifest,
validation-only checkpoint selection, and test-only planning anchors and hard
negatives. Jobs `26493--26495` completed. Across four seeds, DINO-WM improves
offline AE **1.492→1.085**, AS **0.471→0.545**, and CRA **0.036→0.233**.
JEPA-WM improves CRA **0.030→0.085** but not AE (**1.337→1.344**) or AS
(**0.501→0.501**). This mixed offline result is supporting evidence only, not
task success and not a paper method contribution.

## Active confirmatory work

Do not duplicate or overwrite the following running jobs/agent work:

| Work | Current owner / job | Purpose |
|---|---|---|
| Second-checkpoint oracle/planning pipeline | **26481 completed** (`jepa_wm_metaworld`) | Incorporated in locked two-checkpoint oracle ladder |
| Planner generality | Original `26400` cancelled; resumed as **26482**, completed | Push success: CEM `1/16`, MPPI `0/16`, shooting `2/16`; reach L2 controls `16/16`, `13/16`, `6/16`; interpret with control-strength caveat |
| Encoder-information upper bound | **26485 completed** | Object is decodable off-policy, but end-effector and relative geometry degrade strongly; does not isolate encoder vs readout |
| Held-out Phase-H | **26493** DINO-WM, **26494** JEPA-WM; aggregation **26495**, completed | DINO: all four seeds improve AE/AS/CRA; JEPA: CRA improves but AE/AS do not; no closed-loop success claim |
| Locked confirmatory ladder | **26491/26492 completed** | Headline 64-seed results incorporated in the paper |
| Exact same-state intervention | **26497**, completed | Snapshot/restore causal action fan; summary artifacts available, aggregate analysis still required |
| Shared physical scaling | `26498` OOM; retry **26610** | Compare four checkpoints with one effect mask and fixed negative IDs |
| Paper compile/LaTeX check | **26499** completed; post-literature rebuild **26504** | Hardened draft builds on compute; generic article layout remains over ICLR budget |
| Instrumented exploitation audit | **26502/26746 completed** | Replaced by clearer same-population preselection audit for headline use |
| Coverage vs selection | **26505 completed**; `26506` failed but corrected artifacts are present | Final exact-success coverage: DINO push 8.0%, pick 3.6%; JEPA contact 0% |
| Same-population preselection + branch | audits complete; permutation-null **28322 completed** | Immediate regret, matched-noise null, and adaptive refitting consequence |
| TRM-style closest baseline | train **26507**; eval **26508**; analysis **26509** | Replacement/hybrid horizon-matched metric on two checkpoints and held-out oracle seeds |
| ACID-style closest baseline | train **26510**; learned/oracle eval **26511**; paired analysis **26512** | Test inverse-dynamics consistency; MLP verifier approximation because official code/checkpoint is unavailable |

TRM, ACID, and shared scaling remain pending. Do not infer their outcomes from
partial logs. Completed rows above may be cited only through their final
artifacts.

## Highest-priority remaining experiments

1. Complete the TRM-style comparison (`26508/26509`) and the repaired ACID
   smoke/full workflow (`27982`, then replacement full eval if the smoke passes).
2. Aggregate and interpret the completed exact-same-state action fan (`26497`)
   before using it to upgrade the observational CRA/BB claim.
3. Complete DROID shared-scaling retry `26610` with a physical/proprioceptive
   effect mask and fixed negative indices across models.
4. Convert the current generic article into the official ICLR template and cut
   the main paper to the venue page budget; move historical Boundary Blindness,
   mitigation, and DROID details to the appendix.
5. Add a learned non-privileged positive control if TRM/ACID supplies one;
   otherwise state plainly that the current positive control is privileged.

All GPU, simulator, model, or data-heavy work must run through Slurm on a
compute node. The login node is only for lightweight inspection, editing,
small text/CSV aggregation, and job submission/monitoring.

## Document status

| Document | Role now |
|---|---|
| `../../paper/main.tex` | Paper of record; still a draft and subject to the caveats above |
| `../../paper/ICLR_CUT_PLAN.md` | Provisional nine-page contraction map; apply after pending evidence gates |
| `CLAIMS_EVIDENCE.md` | Claim/artifact ledger; rows marked blocked must not enter the paper |
| `../../cai_jepa_paper_proposal.md` | Historical March/June proposal; superseded framing |
| `../../diagnostic_implementation_plan_v2.md` | Historical implementation specification; useful for provenance/API decisions |
| `PAPER_IDEA.md`, `PROGRESS_REPORT.md`, `DIAGNOSIS_PLAN.md` | Historical snapshots of the Boundary-Blindness phase |
| `METHODOLOGY.md` | Reference for the original observational diagnostic implementation, not a causal-identification protocol |
| `HANDOFF*.md` | Historical operational records; use `../RUNBOOK.md` and current Slurm scripts for new runs |
| `plans/YYYY-MM-DD-*.md` | Dated design/result records; status is local to their date |
