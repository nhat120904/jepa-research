# Survey source

`main.tex` is a survey of latent world models for robot planning: the design
space (representation objectives × planning algorithms), a taxonomy of failure
modes, evaluation practice, and open problems.

**This is not the submission.** The paper of record is `paper/main.tex` (TMLR),
a narrow mechanistic audit. The survey covers the field; the project's own
results appear only as §6, one controlled case study among the surveyed work.

Build on a compute node from `diagnosis/`:

```bash
sbatch scripts/slurm_proposal_build.sh
```

## Conventions

1. **It must not contradict `paper/main.tex`.** Where both report a number, the
   submission wins. Success counts are strict episode-end, never any-step latched.
2. **Unrefereed sources are marked.** There is no longer a standalone literature
   index; preprint status is instead flagged inline at the point of use (see
   `refs.bib`'s `[VERIFIED]`/`[UNVERIFIED]`/`[VERIFIED, no local PDF]` notes for
   the underlying source-verification record). Findings from unrefereed sources
   are reported as claims, not settled results. A large share of the closest
   work is 2026 preprints, so this distinction is load-bearing.
3. **§6 stays proportionate.** It is a case study illustrating the methodology
   argued for in §5, not the centre of the document. If it grows past ~3 pages
   the document has drifted back toward being a paper.
4. **Figures are native TikZ only, no raster images.** The two figures
   (`fig:pipeline`, `fig:collapse`) are drawn with `tikz`/`graphicx` but embed
   no external image files; the build script asserts the PDF has zero
   `pdfimages`-listed bitmaps. The document stays self-contained and compiles
   anywhere with a standard TeX Live install.
5. **Blocked evidence stays out.** Rows marked ⛔ in
   `diagnosis/docs/CLAIMS_EVIDENCE.md` are excluded, lacking the positive-control
   validation needed to distinguish a method boundary from an implementation
   failure.

## Bibliography

`refs.bib` is self-contained. Its first 16 entries are copied verbatim from
`paper/refs.bib` with keys unchanged, so text is portable in both directions.
Do **not** point this document at `../paper/refs.bib` — the survey cites work the
submission deliberately omits, and polluting the submission bibliography is a
real risk.

Entries carry `[VERIFIED]` when the arXiv ID and author list were extracted from
the PDF in `world_model/` with `pdftotext`, and `[UNVERIFIED]` otherwise. **Every
`[UNVERIFIED]` entry must be checked against the primary source before external
use.** This currently includes the classical world-model block (World Models,
PlaNet, DreamerV3, MPPI, DINOv2), which has no local copies.
