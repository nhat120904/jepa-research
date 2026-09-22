# Job 53630: pipeline passes, not a method verdict

Checked both Slurm queue and accounting: COMPLETED, exit 0, 1m48s.
14/14 tests passed, including full spatial reconstruction/gradient flow, prospective
spatial queries, strict temporal order and exact simulator replay. All 672 branches
encoded (384 train, 288 validation), spatial shape 16×1024. Test not read.
Every codec/predictor arm finished 20 updates without nonfinite loss/gradient failure.
`profile/result.json`, `profile/predictions.jsonl` and `profile/checkpoint.pt` exist
as outputs of the completed runner; profile numbers are NOT a research comparison.

## Data-only warning: current selection headroom is small

These are already computed bank statistics, independent of any trained model:

| Horizon | Default RGB order score | Oracle RGB order score | Informative prefixes |
|---|---:|---:|---:|
| 32 | 0.171065 | 0.189597 | 5/12 |
| 48 | 0.176572 | 0.200266 | 4/12 |
| 64 | 0.197849 | 0.204003 | 3/12 |

The improvement ceilings are approximately 0.01853, 0.02369 and 0.00615 SCORE units,
NOT success percentage points. Informative means candidate score spread >0.05,
not necessarily a candidate better than the default. Do not import 19/48 success
from the older full 128-candidate screen into this eight-candidate dataset.

## Next authorized step

One bounded full pilot, not an array: seed 20260922, 1,200 updates each codec,
600 generic-reader updates, 1,800 updates each of seven forecasting/control arms.
Batch 16, same optimizer/loss/horizons/features as the passed profile. Reuse encoded
features; initialize fresh rather than extend profile checkpoints. Explicit 30-minute
Slurm GPU limit. No automatic subsequent job and no VLA/policy training.

Added reporting ONLY: train-fit MSE after training (no checkpoint selection), architecture
metadata and human-readable report. This helps distinguish insufficient fitting,
codec information loss, prediction failure and poor generalization. Optimization and
the existing validation protocol are unchanged.

This run can diagnose predictive abstraction and action dependence. Given the small
bank headroom and 12 development prefixes, neither a tie nor a positive ranking result
alone supports a general control verdict. No new arena or full MPC run is justified
before reviewing the observed codec, no-action/direct/frame controls and selection.
