# Current research status (2026-07-13)

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

> Test-time search can select residual errors in learned latent costs on
> contact-rich manipulation. Under oracle-perfect dynamics, true simulator-state
> cost solves push, while every tested cost built on the frozen encoder remains
> near failure; the optimizer's own elites are much less accurate under the
> learned readout than off-policy frames.

The oracle ladder separates cost failure from learned-dynamics error, and direct
simulator-state inspection verifies that selected elites can look successful to
the proxy while the object remains far from the goal. This is the paper's
current empirical core.

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
- Treat the Goodhart budget curve as evidence of proxy/true-objective
  decoupling. Within-search decode-error growth is a point-estimate trend with
  overlapping confidence intervals, not a monotonic law already established.
- Keep Push-T and PointMaze as sanity checks only. Metaworld provides simulator
  verification; DROID is external-scale/real-robot diagnostic evidence, not
  causal ground truth.
- Separate an **honest** cost (tracks truth and supports optimization), a
  **no-signal** cost (small proxy--truth gap can still be useless), and an
  **exploitable** cost (search improves the proxy while selecting poor true
  outcomes). A small exploitation gap alone is not proof of honesty.

## Phase H: validity fix implemented, result pending

The previously reported predictor-LoRA / counterfactual-objective numbers were
not a held-out planning result: the planning probe could read the complete
latent cache. They remain exploratory and are excluded from current claims.

The implementation now creates an immutable deterministic 70/15/15
trajectory manifest, embeds its hash and membership in the checkpoint, selects
the checkpoint on validation only, and restricts both planning anchors and the
hard-negative pool to the untouched test split. Four-seed held-out reruns are
queued as jobs `26493` and `26494`, with aggregation job `26495`.

Until those jobs produce their final artifacts, Phase H is **exploratory only**
and must not support a main-paper claim. If the held-out gain disappears,
remove the method contribution rather than weakening the split standard.

## Active confirmatory work

Do not duplicate or overwrite the following running jobs/agent work:

| Work | Current owner / job | Purpose |
|---|---|---|
| Second-checkpoint oracle/planning pipeline | Original `26166` cancelled; resumed as **26481** (`jepa_wm_metaworld`) | Test checkpoint generality |
| Planner generality | Original `26400` cancelled; resumed as **26482** (MPPI / shooting) | Test whether cost exploitation is CEM-specific |
| Encoder-information upper bound | Agent PROBE / **26485** | Test whether task-relevant state is recoverable from the frozen encoder |
| Held-out Phase-H | **26493** DINO-WM, **26494** JEPA-WM; aggregation **26495** | Replace leaked exploratory tables with test-only evaluation |
| Locked confirmatory ladder | **26491** array; paired analysis **26492** | Increase headline cells to 64 unseen paired seeds |
| Exact same-state intervention | **26497**, completed | Snapshot/restore causal action fan; summary artifacts available, aggregate analysis still required |
| Shared physical scaling | `26498` OOM; retry **26610** | Compare four checkpoints with one effect mask and fixed negative IDs |
| Paper compile/LaTeX check | **26499** completed; post-literature rebuild **26504** | Hardened draft builds on compute; generic article layout remains over ICLR budget |
| Instrumented exploitation audit | **26502** array; component analysis **26503** | Separate elite readout shift, true outcome/opportunity regret, and proxy--truth corruption |
| Coverage vs selection | **26505** array; analysis **26506** | Test whether good physical candidates are absent or present-but-misranked |
| TRM-style closest baseline | train **26507**; eval **26508**; analysis **26509** | Replacement/hybrid horizon-matched metric on two checkpoints and held-out oracle seeds |
| ACID-style closest baseline | train **26510**; learned/oracle eval **26511**; paired analysis **26512** | Test inverse-dynamics consistency; MLP verifier approximation because official code/checkpoint is unavailable |

Results from these jobs are pending. Do not write them as completed or infer
their outcome from logs before the owning workflow has produced a final
artifact.

## Highest-priority remaining experiments

1. Complete and inspect the queued held-out Phase-H reruns.
2. Complete queued exact-same-state job `26497` in Metaworld: snapshot one state,
   execute a controlled local action fan, and record successor object pose,
   contacts, and impulses. Use this to calibrate observational CRA/BB against a
   genuinely causal action-effect measurement.
3. Complete the queued locked, paired, unseen-seed confirmatory evaluation for the central
   oracle ladder. The current `n=8/16` cells support a mechanism study but are
   too small for strong null/general claims.
4. Complete queued DROID shared-scaling job `26498` with a physical/proprioceptive
   effect mask and fixed negative indices across models.
5. Validate proxy--truth divergence on all headline cost arms and include a
   successful learned-cost or other honest positive control where possible.
6. Compare against the closest concurrent methods on cost/reachability and
   action semantics; the literature/positioning notes in
   `plans/2026-07-05-novel-methods-survey.md` are a dated starting point, not a
   complete July literature review.

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
