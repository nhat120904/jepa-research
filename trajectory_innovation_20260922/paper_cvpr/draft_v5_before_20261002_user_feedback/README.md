# CTA research manuscript — revision 2026-10-02

Title: **Conditional Trajectory Abstraction for Visual Action Ranking**.

`main.tex` includes the English main paper; `supplement.tex` is a standalone supplementary document. The project uses the existing CVPR 2026 author kit in page-numbered development mode. Before an actual submission, check the target conference/year's rules, anonymity requirements and assigned submission ID. No submission or acceptance is implied.

The revision replaces projected result tables with archived development evidence, differentiates ranking from historical closed-loop performance, updates the closest literature, and corrects the source/deployment information contract. It includes three vector figures: the training/deployment overview, L8 matched-bank ranking intervals, and the L15 selection-data prediction diagnostic. `fig/evidence_data.json` records quantitative plot provenance; `EVIDENCE_PROVENANCE.json` records archived evidence hashes; `REVISION_MANIFEST.json` records the final source and PDF hashes.

## Build

On a local workstation with TeX Live/MacTeX:

```sh
bash build.sh
```

On the H100 cluster, run document compilation on a compute node:

```sh
sbatch build.sh
```

Record the returned job ID and verify both `squeue` and `sacct`; do not use an interactive allocation or a blocking `sbatch --wait`. This revision was compiled and visually checked on the local Mac; no cluster job was submitted while DNS was unavailable.

## Evidence and revision history

The main findings are development-set results with one training seed. Historical control runs use the old batched proposal sampler, and canonical learned-arm evaluation remains pending. The prepared native-hit continuation has no new training or control result in this manuscript. Sixteen goal images depict one target pose and do not demonstrate semantic-goal transfer. Component latency excludes different components in different rows and does not establish a whole-planner speedup.

`draft_v4_before_20261002_review/` preserves the active source/assets/PDF before rewriting; its `SHA256.json` records that snapshot. Earlier historical snapshots are left intact. The Vietnamese reviewer report is at `../docs/CTA_PAPER_REVIEW_20261002_VI.md`.
