# TMLR submission source

`main.tex` is the paper of record. It uses the official TMLR submission style
in anonymous mode. The vendored style files were copied without modification
from `JmlrOrg/tmlr-style-file`, commit
`7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`.

Build on a compute node from `diagnosis/`:

```bash
sbatch scripts/slurm_paper_build.sh
```

The submission narrative is intentionally limited to the evidence-backed
mechanistic audit:

- terminal-cost comparison under oracle dynamics on fresh seeds 30000--30063;
- identical-population stateprobe selection audit and residual-permutation
  null on seeds 41000--41015;
- stateprobe readout/rank validation on the initial and final CEM populations
  from those same audit seeds;
- an optimizer-conditioned rank/elite-recall figure generated from the
  persisted candidate populations;
- within-snapshot refitting intervention on seeds 42000--42007;
- exact-success candidate-availability audit on seeds 40000--40015; and
- separate search-budget sensitivity and planner-level endpoint controls in
  the appendix.

The detailed misranking and refitting mechanism is claimed only for stateprobe.
Latent L2 is included in the end-to-end terminal-cost comparison, not in the
same-population causal audit. A TRM-style adaptation remains in the research
artifacts but is excluded from the submission because its adaptation-specific
ranking and positive-control validation are not yet sufficient to interpret a
null planning result.

Historical Boundary-Blindness, DROID scaling, Phase-H, mitigation-grid,
selection-aware method sprint, and CAI-JEPA proposal narratives are not part of
the submission. Their research artifacts remain under `diagnosis/` and in git
history.

Before upload, the authors only need to:

1. verify the metadata of the 2025--2026 preprints in `refs.bib`;
2. confirm the title-page LLM-use disclosure matches the current TMLR policy;
3. create an anonymized supplementary archive with analysis scripts,
   manifests, per-seed summaries, and exact commands;
4. ensure the OpenReview submission and supplementary archive are anonymized;
   and
5. replace author, month, year, and OpenReview placeholders only for the
   accepted or preprint version.
