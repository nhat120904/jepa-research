# Assessment of the CROD review

Assessment date: 2026-08-18 UTC.

## Verdict

The review's central recommendation is correct: generic model disagreement is
not a sufficient novelty claim, while **directional ordinal disagreement on a
deployed planner's own candidate set** is a precise and falsifiable H0. The
small H0 should precede any calibrator, flow model, or policy post-training.

The strongest parts are the directional rather than symmetric score, the exact
CEM-returned anchor, action diversity as the primary baseline, the matched
physical-query budget, and explicit measurement of error complementarity.
LeWM versus action-only DINO-WM on OGBench-Cube is also substantially cleaner
than comparing two DINOv2-based MetaWorld models because the visual
representations are formed differently.

## Corrections and limits

1. The review overstates immediate checkpoint availability. The LeWM project
   reports a DINO-WM Cube baseline and links a baseline checkpoint folder, but
   that Google Drive folder currently returns 404 and has open public issues.
   H0 therefore first trains the official `stable-worldmodel` DINO-WM
   reproduction.
2. “Independent representation” means a different representation-formation
   process, not statistically independent errors. Both predictors still see
   the same Cube demonstrations, so complementarity is measured rather than
   assumed.
3. No absolute priority claim is justified. The defensible statement is that
   the checked literature contains no direct match for the complete pipeline:
   native planner-induced candidates, directional cross-representation ranks,
   budgeted physical verification, and subsequent distillation.
4. A proposition about disagreement enrichment needs explicit accuracy and
   error-correlation assumptions. It should be written only after the empirical
   complementarity audit establishes that those assumptions are plausible.
5. H0 can validate an acquisition signal, not a full paper method. A pass
   authorizes a simple matched-data BC pilot first; a fail stops CROD without
   spending compute on flow training.

## Primary sources checked

- DINO-WM: https://arxiv.org/abs/2411.04983
- LeWorldModel and Cube baseline: https://le-wm.github.io/
- stable-worldmodel reference implementation:
  https://github.com/galilai-group/stable-worldmodel
- Active Fine-Tuning of Multi-Task Policies:
  https://arxiv.org/abs/2410.05026
- Cross-model disagreement UQ: https://arxiv.org/abs/2604.17112
- VIScore: https://arxiv.org/abs/2608.11174
- CheckVLA: https://arxiv.org/abs/2607.26789
- Inaccessible baseline checkpoint issue:
  https://github.com/lucas-maes/le-wm/issues/93
