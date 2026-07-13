# ICLR readiness review — 2026-07-13

## Verdict

**Not submission-ready yet, but the project has a credible ICLR paper if the
pending generality and confirmatory gates land.** The strongest paper is no
longer CAI-JEPA as a proposed training method. It is a mechanism paper about
test-time search selecting errors in learned latent costs for contact-rich
planning, isolated with simulator-perfect dynamics and independently verified
with simulator state.

The central oracle experiment is unusually useful: it changes dynamics and cost
separately, includes a privileged positive control, and audits the candidates
selected by the optimizer rather than only average held-out frames. The current
weakness is breadth and confirmatory power, not lack of an interesting core.

## What is currently convincing

- The oracle ladder removes learned dynamics while preserving the frozen visual
  representation and planner. Failure under perfect rollouts therefore cannot
  be assigned to predictor error in that harness.
- Privileged true-state cost is a necessary positive control and demonstrates
  that the task and tested planning budget can succeed.
- CEM elite inspection uses simulator object state as an independent verifier;
  proxy improvement with poor true outcomes is stronger evidence than another
  latent-space metric.
- The budget sweep and positive/no-signal/exploitable arms give a coherent
  mechanism story, with uncertainty caveats now stated in the draft.
- The reproduction fixes and upstream-parity checks make the result more
  auditable than an end-to-end success table alone.

## What is not yet convincing enough for ICLR

### Critical empirical gates

1. **Checkpoint generality.** The central mechanism currently rests primarily
   on DINO-WM/MetaWorld. Job `26481` must establish whether the ladder and elite
   failure reproduce on JEPA-WM rather than merely adding another diagnostic.
2. **Planner generality.** A CEM-only failure invites the objection that the
   optimizer or its implementation is pathological. Job `26482` must compare
   MPPI and shooting with matched rollout budgets and simulator-verified
   endpoints.
3. **Power and locked seeds.** Many headline cells use 8 or 16 episodes. Array
   `26491` uses 64 unseen paired seeds, and `26492` performs paired analysis.
   Strong null/general claims must wait for these artifacts.
4. **Causal action-effect calibration.** CRA/BB currently use observational
   cross-state negatives. Job `26497` snapshot/restores identical MuJoCo states
   and executes a controlled local action fan; this is required before causal
   action-identifiability language can return.
5. **Controlled scaling.** The existing 22M--1B comparison uses model-native
   effect masks and neighbours. Job `26498` fixes the transition universe,
   physical effect mask, and negative IDs across all checkpoints. Until then the
   correct claim is only “no upward trend in the model-native diagnostic.”
6. **Information vs readout failure.** Encoder upper-bound job `26485` is needed
   to distinguish information absence from failure of the tested probes.
7. **Closest-method comparison.** TRM already audits CEM candidate selection,
   true-state cost, task-relevant subspaces, and high-budget search; IMWM already
   uses literal environment dynamics and diagnoses proposal coverage. A credible
   submission needs a same-protocol TRM baseline, ACID under learned/oracle
   dynamics, and an IMWM-style coverage-versus-selection decomposition. Oracle
   dynamics or optimizer-conditioned auditing alone is not novel.

### Validity issue already fixed in code

The old Phase-H method tables were not held-out planning results. The corrected
pipeline now persists an immutable trajectory-level 70/15/15 manifest, selects
on validation, and evaluates planning anchors and hard-negative pools only on
test. Jobs `26493/26494` rerun four seeds for both model families; `26495`
aggregates them. No old Phase-H number should appear as evidence. A held-out gain
would be supporting predictor-side evidence, not a solution to the cost failure;
a null should remove the method section from the submission.

## Methodology corrections now reflected in the draft

- CRA/BB are described as observational diagnostics, not causal interventions.
- The nominal `1/17` CRA baseline is conditioned on candidate exchangeability;
  nearest-neighbour candidates need not satisfy it.
- Cross-model latent gaps and native effect masks are not treated as a
  controlled scale law.
- “Grounding is necessary” has been removed; current evidence only says the
  tested grounding metrics/objectives are insufficient success certificates.
- Finite mitigation nulls no longer imply that every post-hoc latent cost is
  exploitable.
- Elite readout shift, simulator-verified true outcome, and within-search
  proxy--truth corruption are separate quantities. A low readout shift can be a
  no-signal cost and is not proof of honesty.
- The amortized-controller null no longer identifies an “insufficient
  representation”; controller optimization, coverage, and architecture remain
  alternatives.

## Paper/positioning decision

Recommended main-paper spine:

1. Problem: average latent metrics do not certify a cost under test-time search.
2. Upstream-parity contact failure and observational diagnostic as motivation.
3. Oracle ladder localizing the tested failure away from learned dynamics.
4. Elite-conditioned simulator audit and search-budget intervention.
5. Checkpoint/planner/seed generality.
6. Scoped mitigation nulls and limitations.

Boundary Blindness and CAI-JEPA should move to supporting motivation or the
appendix unless same-state and held-out results are unusually strong. The
submission should not market action identifiability itself as novel given the
concurrent action-aware world-model literature.

## Format and writing blocker

The paper still uses generic `article` format. Compute-node build `26499`
completed successfully: the hardened PDF has 19 total pages, with the
conclusion beginning on page 13 and appendix on page 15 (roughly 14 pages of
main paper in this provisional layout). The latest
available official guide, [ICLR 2026 Author
Guide](https://iclr.cc/Conferences/2026/AuthorGuide), caps submission main text
at 9 pages and requires the official style. The ICLR 2027 template/guide was not
available at this review date, so use the 2026 style only for provisional page
budgeting and switch to the 2027 release when official. Reaching nine pages will
require moving most diagnostic detail, mitigation grids, Phase-H protocol, and
per-task tables to the appendix; prose polishing alone will not be enough. The
exact allocation and move/remove list is in `../../paper/ICLR_CUT_PLAN.md`.

## Go/no-go rule after queued jobs

- **GO with the optimizer--cost paper:** mechanism reproduces on the second
  checkpoint or another planner, the 64-seed oracle contrast remains large, and
  simulator audits confirm selected poor outcomes.
- **GO, narrower case study:** only DINO-WM+CEM reproduces, but the powered
  contrast and elite corruption remain strong. Title and abstract must name the
  exact scope; do not imply JEPA-wide generality.
- **PIVOT:** true-state advantage disappears at 64 seeds, or elite proxy--truth
  divergence does not survive locked evaluation.
- **CAI-JEPA method contribution only if:** held-out Phase-H gains replicate
  across seeds/models and a closed-loop endpoint moves. Planning-probe metric
  gains alone are not enough for a headline method claim.
