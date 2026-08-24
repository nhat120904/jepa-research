# CROD H0

This new folder contains the falsification-first pilot for
**Cross-Representation Ordinal Disagreement** on OGBench-Cube.

Start with `PAPER_IDEA.md`: it is the cross-session statement of the proposed
paper, its permitted claims, H0 gate, and H1 continuation.

The implementation has three dependency-chained stages:

1. train the official action-only DINO-WM reproduction with a frozen
   DINOv2-Small encoder;
2. rescore the already fully-labelled Phase-0d populations for the independent
   complementarity audit;
3. run a fresh 128-state, matched 1+8-query H0 against action diversity.

See `PROTOCOL.md` for the locked estimand and gate and `JOB_LEDGER.md` for exact
Slurm provenance. No H1 policy, flow, calibrator, or privileged geometry is
implemented before H0 passes.
