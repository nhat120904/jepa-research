# Latent-world-model discovery sprint — 30 September 2026

Completed screening outcome: six-region literature scan and three executed no-training probes. No candidate currently supports a SOTA-improvement claim or selection for immediate CVPR method development. Corrected tracking uses all 30 TAP-Vid-DAVIS videos; earlier mismatched-feature tracking runs are debug-only. See [Vietnamese report](REPORT_VI.md), [final results](results/summary_final.json), and [verified job ledger](JOB_LEDGER.md).

Authorized scope: scan six application regions beyond LeWM/CTA; run two or three bounded, no-training probes using released models and standard public benchmarks; then select a direction only if task-level evidence supports it. No requirement to force a winner or call an unrun proposal a failure.

Research is organized in `research/`. Experimental scripts are exploratory, not a proposed learned method. Third-party sources live in ignored `external/`, pinned SHAs are recorded in the ledger. Heavy work runs only in time-limited Slurm jobs. Data and checkpoints stay in `/mnt/data/nhatnc129/jepa/idea_discovery_20260930/`.

Initial probe candidates, subject to source/data checks:

1. IntPhys2: physical plausibility under nuisance, measuring paired possible/impossible classification with the released V-JEPA2 encoder/predictor. Plain surprise is existing prior art; probe adaptations must face it, not claim the benchmark as new.
2. TAP-Vid-DAVIS: causal latent prediction for tracking between observations, compared with repeat/copy and released CoTracker3. Accuracy and observation/compute budgets must be reported together. Future frames cannot reach encoder/predictor before their observation time.
3. Sentinel real PushChair: causal rollout-outcome monitoring with a frozen external WM, compared with released precomputed STAC/Sentinel results on the same episodes. Outcome-prefix scoring is not pre-onset warning unless onset labels are verified.

No “SOTA win” can be declared from a small subset or a different protocol. Broad existing applications are not automatically excluded; the question is whether a specific important gap remains and whether the available probe can measure it.
