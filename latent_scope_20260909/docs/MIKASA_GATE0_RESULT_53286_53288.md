# MIKASA Gate 0 — renderer and throughput: FAILED

**Date:** 2026-09-20
**Jobs:** `53279`–`53288`
**Verdict:** `GATE0_SIM_UNAVAILABLE` — ManiSkill3/SAPIEN cannot run on this cluster.
**Consequence:** MIKASA-Robo is not usable here. Per the decision table in
`PREDICTIVE_MEMORY_JEPA_DESIGN_20260920_EN.md`, stop before investing further.

## Root cause: compute-only NVIDIA driver

`worker-0`, H100 80GB, driver 570.133.20. The compute half of the driver is installed and
the graphics half is not.

| Evidence | Value |
|---|---|
| `libnvidia-gpucomp.so.570.133.20` | present |
| `libGLX_nvidia.so.0` | **absent** |
| `nvidia_icd.json` in `/usr/share/vulkan/icd.d` or `/etc/vulkan/icd.d` | **absent** (only intel, lvp, radeon, virtio) |
| `/dev/dri` nodes (9 `card*`, 8 `renderD*`) | present, **all unreadable and unwritable** from inside a job |
| `vulkaninfo` | not installed |

Vulkan loader trace (job `53287`, `VK_LOADER_DEBUG=all`):

```
DEBUG: Searching for ICD drivers named libGLX_nvidia.so.0
ERROR: libGLX_nvidia.so.0: cannot open shared object file: No such file or directory
ERROR | DRIVER: loader_icd_scan: Failed to add ICD JSON libGLX_nvidia.so.0. Skipping ICD JSON.
```

SAPIEN states it directly: `Your GPU driver does not support Vulkan. You may not use the
renderer for rendering.`

## What this rules out

- Every ManiSkill3 probe failed with `vk::createInstanceUnique: ErrorIncompatibleDriver`.
- **`obs_mode="state"` does not avoid it.** Env creation initializes the render system
  regardless of observation mode. This invalidates the "run the ladder on state
  observations" fallback that the design note assumed was safe.
- **`sim_backend="physx_cpu"` does not avoid it either.**
- Supplying our own `nvidia_icd.json` does not help — it also points at the missing
  `libGLX_nvidia.so.0`.

## The lavapipe path: partial, then blocked

`sapien/_vulkan_tricks.py::_ensure_vulkan_icd()` returns early when `VK_ICD_FILENAMES` is
already set, so the system lavapipe ICD can be forced. Under
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` (job `53288`):

- `sapien.Scene()` → **`SAPIEN_SCENE_OK`**. Vulkan instance creation is solved.
- All four ManiSkill probes → `RuntimeError: Failed to find a supported physical device "cuda:0"`.
  SAPIEN binds the render device to `cuda:0`; lavapipe is a CPU device.

One knob remains untried: forcing SAPIEN's render device to CPU. It was deliberately not
attempted, because the best possible outcome is CPU physics plus CPU software rendering,
which removes the GPU-parallel throughput that motivated choosing MIKASA in the first place.
Reopening it is a decision about whether a slow CPU-only MIKASA is still worth having, not
an infrastructure question.

## Two setup faults of ours, fixed (not capability findings)

Recorded so they are not mistaken for results:

1. **Job `53280`** reported `GATE0_SIM_UNAVAILABLE`, but every probe died at
   `torch._C._cuda_init()`: unpinned resolution installed `torch 2.14.0+cu130` against a
   CUDA 12.8 driver. The driver now classifies this pattern as `GATE0_SETUP_FAULT`.
2. **Job `53283`** skipped installation because the readiness marker had been written
   *before* verification in job `53281`. The marker is now written only after torch,
   `sapien` and `mani_skill` all verify.

Working environment after the fix: `torch 2.8.0+cu128`, `sapien 3.0.3`, `mani_skill 3.0.1`.

## Job ledger

| Job | Role | Outcome |
|---|---|---|
| `53279` | setup | COMPLETED 3m14s; torch cu130 (wrong) |
| `53280` | probe | COMPLETED 21s; **setup fault**, not a capability finding |
| `53281` | setup retry | FAILED; `--extra-index-url` re-picked cu130; guard caught it |
| `53282` | probe | CANCELLED (dead dependency), no GPU held |
| `53283` | setup retry | FAILED 3s; stale readiness marker |
| `53284` | probe | CANCELLED |
| `53285` | setup retry | COMPLETED 4m03s; torch 2.8.0+cu128 verified |
| `53286` | probe | COMPLETED 18s; **`ErrorIncompatibleDriver` on all four probes** |
| `53287` | vulkan diag | COMPLETED 7s; missing `libGLX_nvidia.so.0` identified |
| `53288` | lavapipe | COMPLETED 20s; Scene OK, ManiSkill blocked on `cuda:0` |

No job held a GPU idle. Total GPU time across all probes is under two minutes.

## What was missed

The repo's own memory already recorded that this cluster cannot render Isaac Gym due to
compute-only driver caps (ContactWorld Phase 0). That fact predicted this outcome and was
not checked before MIKASA-Robo was proposed as the primary arena. A renderer probe must be
the first job of any new-arena qualification. Recorded as
`cluster-no-gpu-vulkan-rendering` in memory.

## Implications for the design note

Section 5 (arena) and Gate 0 of `PREDICTIVE_MEMORY_JEPA_DESIGN_20260920_EN.md` need
revision:

- MIKASA-Robo is unavailable, and so is ManiSkill `DrawTriangle`/`DrawSVG` — same engine.
- The remaining viable simulators on this cluster are MuJoCo/robosuite family under CPU
  OSMesa: RoboCasa (closed as a method arena), OGBench, MetaWorld, and the existing
  `stable-worldmodel`/LeWM infrastructure.
- Any replacement memory arena must be chosen against the renderer constraint first and the
  memory-type coverage second, which is the reverse of how MIKASA was selected.
