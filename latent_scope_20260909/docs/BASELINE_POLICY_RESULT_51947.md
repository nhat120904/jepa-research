# Official baseline-policy gate result — job 51947

Date: 11 September 2026.

## Verdict

`BASELINE_USABLE_FOR_A4` — **PASS**.

The split-runtime evaluation completed normally (`COMPLETED`, exit `0:0`) on one H100.
The official GR00T N1.5 target-posttrained checkpoint was evaluated for exactly ten
native target-split episodes per task:

| Task | Successes | Rate | Locked minimum |
|---|---:|---:|---:|
| `ScrubCuttingBoard` | 3 / 10 | 30% | 2 / 10 |
| `RinseSinkBasin` | 6 / 10 | 60% | 2 / 10 |

Both tasks pass the predeclared usefulness gate. This establishes that a released policy
can produce successful trajectories in the current RoboCasa `1.0.1` runtime. It does not
test the compositional-JEPA hypothesis and the ten-episode rates are not final benchmark
estimates.

Non-fatal warnings were emitted for the disabled home-directory Mesa shader cache and
Gymnasium observation-space validation. Rendering, inference and native success tracking
continued through all twenty episodes.

## Decision

Proceed to a bounded A4 design using successful current-runtime policy rollouts to build
the transition-calibration and progress-label bridge. Do not launch Stage B until A4's
determinism, action-diversity and progress-label conditions pass.

Machine-readable result:
`/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/eval_51947/baseline_policy_result.json`.
