# Runtime compatibility decision

Date: 10 September 2026.

## Finding

The failed open-loop replay cannot currently be repaired by installing a public
`env_version=0.5.1` release:

- the official RoboCasa repository and the local complete public history expose only the
  `v0.2` and `v1.0` tags; there is no public `0.5.1` tag or release;
- the current official dataset documentation explicitly asks users to update to the
  latest RoboCasa version (`1.0.1`) for evaluation;
- the official playback documentation presents state/video playback as the normal
  inspection path. Action playback is optional, and the source utility warns when it
  diverges instead of treating exact action replay as a dataset-validity condition;
- Stage A independently showed exact direct-state loading and valid camera observations.

Therefore the `0.5.1` value stored in the dataset metadata is historical provenance, not
a reproducible public installation target. Chasing an unknown commit would create a new,
unverifiable environment rather than repair the benchmark.

Primary sources:

- <https://github.com/robocasa/robocasa/tags>
- <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>
- <https://github.com/robocasa/robocasa/blob/main/robocasa/scripts/dataset_scripts/playback_dataset.py>
- <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos>

## Recollection feasibility

Fresh successful demonstrations cannot be produced safely and automatically from the
current checkout. The official collection workflow is human teleoperation. The lightest
official released policy, Diffusion Policy, reports only 0.2% average success on the
composite-seen split, so it is not a credible generator for a balanced successful contact
dataset. No compatible checkpoint is installed locally. Downloading and running a large
model speculatively would violate the bounded, falsification-first plan.

Primary sources:

- <https://robocasa.ai/docs/build/html/use_cases/creating_datasets.html>
- <https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html>

## Decision

The compositional-JEPA idea remains alive, but the original Stage-A requirement of exact
open-loop reproduction is not achievable from the public release and should not be
silently weakened.

Use the released trajectories only as internally coherent offline image/action data. Use
RoboCasa `1.0.1` as the evaluation runtime, as the official instructions require. Before
any Stage-B model comparison, add a small current-runtime transition-calibration set so
that model failure cannot be explained solely by the historical data/runtime gap.

Do not launch Stage B until both conditions hold:

1. the current dataset revision supplies usable per-frame subtask/progress annotations, or
   an equally explicit progress-label source is locked;
2. the A4 calibration gate shows deterministic current-runtime one-step branches with the
   same action schema and cameras used by every model arm.

The locally pinned five-episode samples do not contain the per-frame subtask fields
announced in the current release. Refreshing the two complete 500-episode archives would
be a material data/storage operation and is deliberately not started without first
locking the A4 sample and required fields.
