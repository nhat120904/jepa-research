# Direction A readiness audit

Audit date: 2026-09-07.

## Verified locally

- The requested datasets exist at `/mnt/data/vhoangth2/datasets/ogbench_data` for both medium and large `navigate`.
- The installed environment package is OGBench 1.2.1. The environment names in `research/REPORT_VI.md` are valid.
- OGBench 1.2.1 exposes five fixed evaluation tasks, 29-dimensional Ant state, XY goal success with radius 0.5, and
  10 Hz Ant control. Its official repository includes standalone JAX implementations of GCBC, GCIVL, GCIQL, QRL, CRL,
  and HIQL, but does not distribute pretrained reference checkpoints.
- The public HSVL repository at commit `a24587f85e78badc2f601ab9eee64039985a1098` has OGBench configs and a pinned
  `uv.lock`; reproduction has not yet been run here.
- No exact-title GitHub repository for CompPlan was found, and the paper/project materials inspected do not identify an
  official implementation. Any local reconstruction must be labeled as such and matched to the paper pseudocode.

## Consequence for execution

The first cluster job trains and freezes a transparent GCBC skill solely to establish A0 and A1 mechanism feasibility.
It is not a substitute for the required CompPlan, HIQL, or HSVL system comparisons. A large method campaign stays blocked
until (i) A0/A1/A2 pass, (ii) the CompPlan reconstruction has a pseudocode-to-code audit, and (iii) HIQL/HSVL are run with
their own recommended capacity and hyperparameters.

## Narrow novelty delta to test

The mathematical object “option transition plus termination time” is old. CompPlan already composes policies across
geometric horizons, and SVL/HSVL already models censored time-to-goal. A publishable delta would need evidence that
conditioning the exit-state distribution on termination time/outcome changes downstream composition after strong
fixed-horizon, option-termination, CompPlan, HIQL, and HSVL comparisons. A duration head alone is explicitly out of scope.
