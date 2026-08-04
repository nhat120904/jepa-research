# Research proposal source

`main.tex` is an English, literature-weighted research proposal covering the
whole research direction: what to read, what the completed diagnostic work
established, and which directions the evidence leaves open.

**This is not the submission.** The paper of record is `paper/main.tex`
(TMLR). This document is deliberately broader: it carries the Boundary-Blindness,
DROID-scaling, counterfactual-predictor, mitigation-grid and selection-sprint
material that `paper/README.md` explicitly excludes from the submission.

Build on a compute node from `diagnosis/`:

```bash
sbatch scripts/slurm_proposal_build.sh
```

## Rules this document is written under

1. **It must not contradict `paper/main.tex`.** Where both report a number, the
   submission wins. Success counts are strict episode-end, never any-step latched.
2. **Every results table carries a status**: `paper` (carried by the submission),
   `supporting` (measured and reproducible, caveat carried), or `historical`
   (superseded framing, retained for provenance). Supporting and historical
   evidence may motivate a direction in §7; neither may carry a headline claim.
3. **Claim discipline is enforced by macros**, not by memory. `\testedcosts`,
   `\pocketsel`, `\groundinsuff` and `\scopedscale` are defined in the preamble
   and expand to the guarded phrasing. Do not inline those strings by hand — the
   macros exist so the wording survives editing. The full rule list is Table 13,
   sourced from `diagnosis/docs/CURRENT_STATUS.md`.
4. **No figures.** `graphicx` is deliberately not loaded and the build script
   asserts that the PDF embeds no images. Keep it that way; the document is
   self-contained and compiles anywhere.
5. **Blocked evidence stays out.** Rows marked ⛔ in
   `diagnosis/docs/CLAIMS_EVIDENCE.md` — currently the ACID-style adaptive
   inverse-consistency cost and the TRM-style adaptation null — are excluded,
   because neither has the positive-control validation needed to distinguish a
   method boundary from an implementation failure.

## Bibliography

`refs.bib` is self-contained. Its first 16 entries are copied verbatim from
`paper/refs.bib` with keys unchanged, so text is portable in both directions;
everything after the divider exists only here. Do **not** point this document at
`../paper/refs.bib` — the proposal cites work the submission deliberately omits,
and polluting the submission bibliography is a real risk.

Entries carry `[VERIFIED]` when the arXiv ID and author list were extracted from
the PDF in `world_model/` with `pdftotext`, and `[UNVERIFIED]` when they came
from the project's literature ledger or from recall. **Every `[UNVERIFIED]` entry
must be checked against the primary source before any external use.**
