# Phase 0 / Gate 0 — Infrastructure feasibility (ContactWorld on this cluster)

Date: 2026-08-20
Status: **PARTIAL BLOCK — closed-loop planning eval blocked, offline track open**

## What was checked

Cluster: SLURM, partition `main`, nodes `worker-0..3` (H100 80GB), plus `mig` partition.
Nodes are Kubernetes pods (`cri-containerd` cgroup), driver `570.133.20`.

| Check | Result |
|---|---|
| GPU present on compute node | YES — NVIDIA H100 80GB HBM3 |
| Network from compute node | YES (huggingface.co reachable) |
| CUDA userspace (`libcuda`, `libnvidia-ml`, `nvvm`, `ptxjitcompiler`) | present |
| NVIDIA **graphics** userspace (`libnvidia-glcore`, `libnvidia-eglcore`, `libnvidia-glvkspirv`, `libGLX_nvidia`) | **ABSENT** |
| `nvidia_icd.json` (Vulkan ICD) in `/usr/share/vulkan/icd.d`, `/etc/vulkan/icd.d` | **ABSENT** |
| Vulkan ICDs actually present | intel, radeon, virtio, `lvp` (lavapipe = CPU software only) |
| GLVND EGL vendor | `50_mesa.json` only |

Conclusion: the GPU pods are provisioned with `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
The **graphics/display capability is not injected**, so Vulkan cannot enumerate the NVIDIA GPU.

## Why this matters for ContactWorld

ContactWorld visual observations are produced by Isaac Gym camera sensors:

- `isaacgymenvs/tacsl_sensors/tacsl_sensors.py:214` — `gym.create_camera_sensor(...)`
- `isaacgymenvs/tacsl_sensors/tacsl_sensors.py:248` — `gym.render_all_camera_sensors(self.sim)`
- `:177-182` — `get_camera_image_gpu_tensor(..., gymapi.IMAGE_COLOR / IMAGE_DEPTH)`

`pointcloud`, `front`, `wrist`, `*_rgb`, `*_depth` all derive from this path, and it requires the
Isaac Gym Vulkan pipeline backed by the NVIDIA ICD. Without it, camera sensors cannot render on GPU.
Falling back to `lvp` (lavapipe, CPU) is nominally possible but is orders of magnitude too slow for
`--num-envs 100 --candidates 100 --iterations 4` CEM evaluation.

Tactile force field (`TacFF`) is the exception: it is SDF/physics-based
(`tacsl_sensors.py:379 setup_tactile_force_field`, `:1014 get_tactile_shear_force_fields`),
so it does not require rendering.

## What is blocked vs. open

**BLOCKED (needs sim + rendering):**
- `eval_planner.py` closed-loop planning success — i.e. reproducing the published cells,
  and any closed-loop measurement of a new method.
- The "physical probing candidates -> realized reduction in estimation error" part of the
  observability diagnostic (needs to execute probe actions in sim).

**OPEN (dataset-only, no sim, no rendering):**
- `train.py` — world model training on the released zarr datasets.
- `eval_rollout.py` — multi-step rollout prediction error.
- The **observability diagnostic** (probe hidden state from current obs / history / history+tactile),
  because the released zarr datasets already contain rendered `pointcloud`, tactile fields and
  privileged `state` per timestep.

This means the scientifically decisive part of Phase 0 — *does tactile history actually reduce
task-relevant uncertainty, and is that reduction predictable from deployable observations* — can
proceed now. Only the closed-loop numbers need the unblock.

## Required unblock (cluster admin request)

The GPU pod spec / NVIDIA device plugin must request graphics capability:

    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

so that the container receives `libnvidia-glcore`, `libnvidia-eglcore`, `libnvidia-glvkspirv`,
`libGLX_nvidia` and `/usr/share/vulkan/icd.d/nvidia_icd.json`.

Verification command once applied (must list the H100, not just `llvmpipe`):

    srun --partition=main --gres=gpu:1 bash -c 'ls /usr/share/vulkan/icd.d/ && vulkaninfo --summary | head -30'
