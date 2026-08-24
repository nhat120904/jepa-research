# Request to cluster admin — enable NVIDIA graphics capability on GPU pods

## Summary

The GPU worker nodes expose only the NVIDIA **compute** userspace. Any workload that needs GPU
rendering (OpenGL / EGL / Vulkan) cannot run. This blocks Isaac Gym camera sensors, which we need
for closed-loop robot-manipulation evaluation.

## Evidence (from `srun --partition=main --gres=gpu:1` on worker-1)

Present:

    libcuda.so.1, libnvidia-ml.so.1, libnvidia-nvvm.so.4,
    libnvidia-ptxjitcompiler.so.1, libnvidia-opencl.so.1, libnvidia-allocator.so.1

Absent:

    libnvidia-glcore.so.*      libnvidia-eglcore.so.*
    libnvidia-glvkspirv.so.*   libGLX_nvidia.so.*        libEGL_nvidia.so.*

Vulkan ICD directory contains no NVIDIA entry:

    $ ls /usr/share/vulkan/icd.d/
    intel_hasvk_icd.x86_64.json  intel_icd.x86_64.json  lvp_icd.x86_64.json
    radeon_icd.x86_64.json       virtio_icd.x86_64.json
    # /etc/vulkan/icd.d/ is empty; no nvidia_icd.json anywhere

    $ ls /usr/share/glvnd/egl_vendor.d/
    50_mesa.json          # mesa only, no 10_nvidia.json

`nvidia-smi` works and reports driver 570.133.20, so the kernel driver is fine — only the
graphics userspace is not being injected into the pod.

## Cause

The nodes are Kubernetes pods (`cri-containerd` cgroup). The NVIDIA container runtime injects
userspace libraries according to `NVIDIA_DRIVER_CAPABILITIES`, which is currently
`compute,utility`.

## Requested change

Set on the GPU pod spec (or the device-plugin / runtime class default):

    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

This adds the GL/EGL/Vulkan libraries and `nvidia_icd.json`. It does not change CUDA behaviour
and does not require a driver upgrade.

## Verification after the change

    srun --partition=main --gres=gpu:1 bash -c \
      'ls /usr/share/vulkan/icd.d/ && ls /usr/share/glvnd/egl_vendor.d/'

Expected: `nvidia_icd.json` present, and `10_nvidia.json` in the EGL vendor directory.

If `vulkaninfo` is installed, `vulkaninfo --summary` should list `NVIDIA H100 80GB HBM3`
rather than only `llvmpipe`.

## Why it matters here

We evaluate vision-based robot manipulation policies in Isaac Gym. Observations are produced by
`gym.create_camera_sensor` + `gym.render_all_camera_sensors`, which require the Vulkan pipeline
on the NVIDIA device. The software fallback (`lvp` / lavapipe) is CPU rasterisation and is orders
of magnitude too slow for the evaluation workload (100 parallel envs x 100 CEM candidates x 4
iterations per planning step).
