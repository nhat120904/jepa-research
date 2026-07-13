# ICLR main-text contraction plan (2026-07-13)

The current compute-node build is 19 pages total in generic `article` format;
the appendix starts on page 15. The latest available official ICLR guide allows
9 main-text pages. The 2027 style was not yet published on the audit date, so
page budgeting should use the 2026 style provisionally and switch once the
official 2027 package appears.

Do not mechanically shrink fonts, margins, captions, or tables. The paper needs
one mechanism spine and must move secondary research history to the appendix.

## Proposed nine-page allocation

| Main-text component | Target |
|---|---:|
| Abstract + Introduction | 1.25 pages |
| Related work and exact novelty wedge | 0.75 |
| Setting, factorial oracle design, estimands | 1.0 |
| Core oracle ladder | 1.25 |
| Coverage-versus-selection and elite simulator audit | 1.5 |
| Search-pressure, checkpoint, and optimizer generality | 1.5 |
| TRM/ACID comparisons and scoped mitigation result | 1.0 |
| Limitations + conclusion | 0.75 |

## Keep in the main paper

- One compact failure-surface taxonomy: prediction, proposal coverage,
  transition realizability, and terminal cost selection.
- One oracle-factorial table with true-state positive control.
- One figure combining candidate coverage/selection, proxy/physical behavior,
  and powered seed confidence intervals.
- One generality/comparison table: second checkpoint, CEM/MPPI/shooting, TRM,
  and ACID under learned/oracle dynamics.
- Exact paired seed counts, confidence intervals, and pre-registered endpoint
  definitions.

## Move to appendix or remove

- Full CRA/AUG/ECS/BB definitions, per-regime tables, threshold robustness, and
  the model-native scaling table. Retain at most one motivating sentence in
  main; restore a scaling claim only if shared job `26498` supports it.
- Reproduction bug ladder details. Keep only upstream parity and link the full
  integrity checklist to appendix.
- Individual readout/adapter/LoRA/ensemble mitigation rows. Main text should
  summarize their scoped null only after closest baselines are included.
- Phase-H objective and formulas unless held-out jobs `26493--26495` produce a
  clear replicated gain and a closed-loop endpoint. Otherwise remove the
  section entirely, not merely move its invalid historical numbers.
- The amortized-controller E1 null, detailed probe architectures, all per-task
  tables, and historical Boundary-Blindness derivation.
- Repeated Goodhart exposition. Use “selection into residual cost-error
  pockets” unless powered jobs `26502/26503` establish the stronger curve.

## Editing sequence after results land

1. Freeze paper-ready rows in `diagnosis/docs/CLAIMS_EVIDENCE.md`.
2. Choose the scope from the go/no-go rule in
   `diagnosis/docs/ICLR_READINESS_REVIEW_2026-07-13.md`.
3. Replace current Sections 3--8 with the seven-component allocation above;
   do not try to preserve every completed experiment in main text.
4. Apply the official ICLR 2027 style when released; until then use the latest
   official style only as a page-budget proxy.
5. Build through Slurm, inspect main-text page boundary from the `.aux`, and
   repeat until conclusion ends by page 9 without style violations.
