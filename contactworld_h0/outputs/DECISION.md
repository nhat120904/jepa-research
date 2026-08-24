# ContactWorld Phase 0 — decision report

Success tolerance the benchmark itself uses on `plug_pos`: **1 cm** (`eval_planner.py:313`). All errors below are held-out, split by episode, episode-clustered bootstrap CIs, 3 seeds.

## exploration_search (108 episodes)

### Raw-observation readout of object state (upper bound on available information)

| condition | mean err (cm) | 95% CI | hit@1cm |
|---|---|---|---|
| linear readout, proprio only (no history) | 1.84 | [1.69, 2.01] | 0.313 |
| pointcloud, current frame | 3.81 | [3.52, 4.11] | 0.008 |
| pointcloud + proprio, current frame | 1.48 | [1.34, 1.63] | 0.499 |
| pointcloud + proprio, 8-step history | 1.22 | [1.09, 1.38] | 0.604 |
| pointcloud + proprio + TACTILE, 8-step history | 1.23 | [1.11, 1.38] | 0.589 |
| proprio + TACTILE, 8-step history (no vision) | 1.26 | [1.13, 1.42] | 0.590 |
| proprio only, 8-step history | 1.23 | [1.10, 1.39] | 0.603 |

### Paired effect tests (same held-out windows)

| test | delta (cm) | 95% CI | CI excludes 0 |
|---|---|---|---|
| tactile added to vision+proprio history | +0.009 | [-0.04, 0.06] | no |
| tactile added to proprio history (no vision) | +0.031 | [-0.02, 0.08] | no |
| pointcloud added to proprio history | -0.009 | [-0.05, 0.03] | no |

### Frozen world-model latent readout (what the CEM planner actually scores)

| checkpoint | readout | mean err (cm) | 95% CI | hit@1cm |
|---|---|---|---|---|
| pointcloud only | linear | 3.95 | [3.68, 4.24] | 0.025 |
| pointcloud only | mlp | 4.58 | [4.33, 4.83] | 0.051 |
| pointcloud + TacFF | linear | 3.52 | [3.22, 3.82] | 0.101 |
| pointcloud + TacFF | mlp | 3.79 | [3.52, 4.06] | 0.117 |

## insertion_usb (201 episodes)

### Raw-observation readout of object state (upper bound on available information)

| condition | mean err (cm) | 95% CI | hit@1cm |
|---|---|---|---|
| linear readout, proprio only (no history) | 0.44 | [0.42, 0.47] | 0.974 |
| pointcloud, current frame | 0.31 | [0.30, 0.32] | 0.996 |
| pointcloud + proprio, current frame | 0.23 | [0.22, 0.24] | 1.000 |
| pointcloud + proprio, 8-step history | 0.25 | [0.24, 0.26] | 0.999 |
| pointcloud + proprio + TACTILE, 8-step history | 0.24 | [0.23, 0.26] | 0.995 |
| proprio + TACTILE, 8-step history (no vision) | 0.24 | [0.23, 0.25] | 0.997 |
| proprio only, 8-step history | 0.49 | [0.46, 0.52] | 0.929 |

### Paired effect tests (same held-out windows)

| test | delta (cm) | 95% CI | CI excludes 0 |
|---|---|---|---|
| tactile added to vision+proprio history | -0.006 | [-0.02, 0.01] | no |
| tactile added to proprio history (no vision) | -0.251 | [-0.28, -0.22] | **yes** |
| pointcloud added to proprio history | -0.238 | [-0.27, -0.21] | **yes** |

### Frozen world-model latent readout (what the CEM planner actually scores)

| checkpoint | readout | mean err (cm) | 95% CI | hit@1cm |
|---|---|---|---|---|
| pointcloud only | linear | 0.85 | [0.80, 0.91] | 0.718 |
| pointcloud only | mlp | 0.58 | [0.54, 0.62] | 0.876 |
| pointcloud + TacFF | linear | 0.80 | [0.75, 0.86] | 0.767 |
| pointcloud + TacFF | mlp | 0.64 | [0.59, 0.69] | 0.840 |

## Gate verdict

Gate condition (pre-registered): *GO only if tactile history measurably reduces task-relevant uncertainty.*

- **exploration_search**: tactile delta +0.009 cm, CI [-0.04, 0.06] -> **NO EFFECT**
- **insertion_usb**: tactile delta -0.006 cm, CI [-0.02, 0.01] -> **NO EFFECT**


## Interpretation

Two tasks, two different bottlenecks. Tactile is not the fix in either.

| task | object state in raw obs | object state in WM latent | implied bottleneck |
|---|---|---|---|
| insertion_usb | present (0.23 cm) | mostly present (0.58 cm, 88% under 1 cm) | downstream of perception: cost / planner |
| exploration_search | present only via proprio (1.23 cm) | absent (3.5-4.0 cm) | representation / input modality |

**1. Tactile carries object-state information, but only where vision already works.**
On insertion_usb, adding tactile to proprio-only history is a real gain
(-0.25 cm, CI [-0.28, -0.22]) -- so TacFF is not noise. But adding it on top of the
pointcloud is a null (-0.006 cm, CI [-0.02, 0.01]). It is redundant with vision.
On exploration_search, where the pointcloud readout is 3.81 cm and effectively useless,
tactile does *not* step in: every tactile contrast there is a tight null.
So tactile does not compensate for vision exactly where compensation would be needed.

**2. The information the benchmark's own metric needs is often not in the latent the planner scores.**
ContactWorld's success test thresholds plug_pos at 1 cm. On exploration_search the frozen
latent reads out plug_pos at 3.5-4.0 cm, i.e. the CEM cost is computed in a space that cannot
resolve the object to within the tolerance it is being scored against.

**3. The world model never sees proprioception.**
`train.py:73` loads only `["action", vision_key]` (+ tactile); the planner's history buffers
(`planner_utils.py:285-300`) carry only vision (+tactile). Yet on exploration_search a readout
from proprioceptive history alone reaches 1.23 cm, three times better than the latent.

**4. On insertion_usb, perception is NOT the bottleneck.**
The latent already resolves plug_pos to 0.58 cm with 88% of frames under the 1 cm threshold,
while published planning success at this modality/horizon is far below 88%. Whatever fails
there fails after perception -- consistent with a cost/planner-side failure rather than a
missing-information failure.

## Caveat that bounds all of the above

Every number here is measured on **demonstration** trajectories. Part of the proprio readout's
accuracy plausibly comes from the demonstrator's behaviour correlating with object pose (the arm
goes where the object is), which need not survive under CEM-generated motion. Testing that
requires executing off-policy actions in the simulator, which is blocked by the missing NVIDIA
graphics stack (see docs/GATE_0_INFRA.md). Until that is unblocked, "add proprioception to the
world model" is the best-supported lead, not an established fix.

