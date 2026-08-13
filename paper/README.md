# TMLR submission source

`main.tex` is the paper of record. It uses the official TMLR submission style
in anonymous mode. The vendored style files were copied without modification
from `JmlrOrg/tmlr-style-file`, commit
`7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`.

Build on a compute node from `diagnosis/`:

```bash
sbatch scripts/slurm_paper_build.sh
```

Last verified build: Slurm job `38953` (2026-08-12), 13 US-Letter pages, clean
final LaTeX pass with no undefined citations/references or box warnings. The
exact command was `sbatch diagnosis/scripts/slurm_paper_build.sh`; its log is
`/mnt/data/nhatnc129/jepa_runs/logs/paper_build_38953.out`.

The submission narrative is intentionally limited to the evidence-backed
mechanistic audit:

- terminal-cost comparison under oracle dynamics on fresh seeds 30000--30063;
- preregistered task-breadth ladder on seeds 70000--70015, including the
  drawer-close success boundary and all failed positive controls;
- identical-population stateprobe selection audit and residual-permutation
  null on seeds 41000--41015;
- a task-aligned reanalysis of that null using direct object-to-goal distance,
  which explicitly limits the structured-residual claim;
- held-out validation of the complete scalar stateprobe composition on the
  immutable trajectory split;
- stateprobe readout/rank validation on the initial and final CEM populations
  from those same audit seeds;
- an optimizer-conditioned rank/elite-recall figure generated from the
  persisted candidate populations;
- within-snapshot stateprobe refitting intervention on seeds 42000--42007;
- targeted within-snapshot latent-L2 refitting interventions for DINO push and
  pick-place on 16 seeds each;
- exact-success candidate-availability audit on seeds 40000--40015; and
- corrected OGBench-Cube true-endpoint same-population audit on the locked
  32-snapshot, 300-candidate cohort; and
- separate search-budget sensitivity and planner-level endpoint controls in
  the appendix.

The detailed cross-task misranking and refitting mechanism is claimed only for
stateprobe. Its refitting effect is reported both in shaped-cost units and in
direct physical task distance. The matched-residual excess is claimed only for
the operational shaped cost because it does not consistently transfer to task
distance. Latent L2 is included in the exact-dynamics comparison, a
same-population ranking audit, and targeted DINO push and pick-place causal
refitting interventions. These establish the intervention only for the shared
MetaWorld DINOv2 representation. The
OGBench extension is an independent same-population replication of residual
cost misranking; it is not a closed-loop refitting intervention. A TRM-style
adaptation remains in the research artifacts but is excluded from the
submission because its adaptation-specific ranking and positive-control
validation are not yet sufficient to interpret a null planning result.

The two released MetaWorld stacks share the same frozen `dinov2_vits14`
encoder, and exact dynamics bypass their different predictors. The manuscript
therefore treats them as two released-stack checks, not two independent
representation replications. The LeWM Cube checkpoint supplies the independent
encoder/model substrate, with one-task/one-checkpoint scope stated explicitly.
The MetaWorld breadth result is deliberately non-uniform: button-press and
window-close add failures, while drawer-close is a successful boundary. This is
part of the submission argument, not an exception to hide.

Historical Boundary-Blindness, DROID scaling, Phase-H, mitigation-grid,
selection-aware method sprint, and CAI-JEPA proposal narratives are not part of
the submission. Their research artifacts remain under `diagnosis/` and in git
history.

Before upload, the authors still need to:

1. create an anonymized supplementary archive with analysis scripts,
   manifests, per-seed summaries, and exact commands;
2. ensure the OpenReview submission and supplementary archive are anonymized;
3. confirm that no overlapping version is published or simultaneously under
   review at another archival venue; and
4. replace author, month, year, and OpenReview placeholders only for the
   accepted or preprint version.

The 2025--2026 citation metadata and first-page assistive-LLM disclosure were
checked against the corresponding primary sources and current TMLR author
guidance during the 2026-08-11 revision.
