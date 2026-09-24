# Conditional trajectory abstraction — research design

Status: QUALIFICATION IMPLEMENTATION, 2026-09-22. No new model trained and no
qualification/control result yet; see JOB_LEDGER.md for submissions. Closed Wall and Scrub runs remain closed. This is a new experimental
contract, not permission to repeat their tuning sweeps.

**Question:** can a compact, action-predictable code for a future trajectory preserve
new information relative to observed history, answer independent temporal queries,
and improve the accuracy/compute trade-off of policy-guided planning?

Borrowed principle: conditional coding with information already available to the
decoder, from learned video compression. The proposed adaptation is a query-trained
code for the *whole future segment*, plus action-conditioned prediction of that code.
The potential contribution must be demonstrated against conditional frame compression
and a cached direct-query model, not merely against frame-by-frame reconstruction.

Flow:

    history -> shared cached context C ---------------------------> query reader
       |                                                             ^
       + frozen proposal policy -> K candidate chunks                |
       + C + each action chunk -> WM -> predicted innovation code ---+
                                                          queries -> scores
                                                select chunk -> execute prefix

The future-observation encoder is used only for training the code. The planner never
observes future frames. Decoder receives C, the predicted code and queries; it never
receives the candidate actions directly. Direct-query baselines may receive actions.

Documents:

- [Current execution plan and CompPlan interpretation](IMPLEMENTATION_PLAN.md)
- [Job ledger](JOB_LEDGER.md)
- [Method, equations and novelty boundaries](RESEARCH_DESIGN.md)
- [Arena, action proposals, qualification and stopping rules](EXPERIMENT_CONTRACT.md)

Recommendation: qualify frozen Diffusion Policy on its native gym-pusht setup first.
Its published model card is a feasibility reference, not our achieved baseline or
proof of selection headroom. An original DINO-WM reproduction is a separate control;
its action/runtime interface must not be silently mixed with gym-pusht.

All compute follows the root AGENTS.md: sbatch only, explicit limits, unique snapshots,
queue/accounting checks, no duplicate jobs or idle GPU allocations.
