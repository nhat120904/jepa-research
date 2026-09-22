"""Gate 0 driver: does ManiSkill/SAPIEN render on this cluster, and how fast?

This answers the blocking question in PREDICTIVE_MEMORY_JEPA_DESIGN_20260920_EN.md
before any MIKASA-Robo work is scheduled. It uses base ManiSkill (PickCube-v1), not
MIKASA itself, because the decisive question is Vulkan availability, not task logic.

Each probe runs in its own subprocess (Vulkan failures frequently abort the process).
No project code, no MuJoCo, no model loading.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "mikasa_gate0_worker.py")

# Heuristic only, recorded as such in the output: GPU-backed ManiSkill RGB at 16 envs
# should be far above this; a CPU software rasterizer (lavapipe) sits far below it.
SOFTWARE_RENDER_FPS_SUSPECT = 50.0

NVIDIA_ICD_BODY = {
    "file_format_version": "1.0.0",
    "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.277"},
}


def run(cmd: list[str], timeout: int = 60) -> dict:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except FileNotFoundError:
        return {"cmd": " ".join(cmd), "error": "not_found"}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "error": f"timeout_{timeout}s"}


def environment_report() -> dict:
    rep: dict = {
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vk_icd_filenames_env": os.environ.get("VK_ICD_FILENAMES"),
    }

    dri: dict = {"exists": os.path.isdir("/dev/dri"), "nodes": []}
    if dri["exists"]:
        for name in sorted(os.listdir("/dev/dri")):
            path = os.path.join("/dev/dri", name)
            dri["nodes"].append(
                {
                    "name": name,
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK),
                }
            )
    dri["any_render_node_accessible"] = any(
        n["name"].startswith("renderD") and n["readable"] and n["writable"]
        for n in dri["nodes"]
    )
    rep["dev_dri"] = dri

    icds: dict = {}
    for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        icds[d] = sorted(os.listdir(d)) if os.path.isdir(d) else None
    rep["vulkan_icd_dirs"] = icds
    rep["nvidia_icd_present"] = any(
        any("nvidia" in f.lower() for f in (files or [])) for files in icds.values()
    )

    rep["nvidia_smi"] = run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]
    )
    if shutil.which("vulkaninfo"):
        rep["vulkaninfo_summary"] = run(["vulkaninfo", "--summary"], timeout=60)
    else:
        rep["vulkaninfo_summary"] = {"error": "vulkaninfo_not_installed"}
    return rep


def probe(
    python: str,
    out_dir: str,
    name: str,
    obs_mode: str,
    num_envs: int,
    extra_env: dict | None = None,
    timeout: int = 420,
) -> dict:
    out_path = os.path.join(out_dir, f"probe_{name}.json")
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    cmd = [
        python,
        WORKER,
        "--obs-mode",
        obs_mode,
        "--num-envs",
        str(num_envs),
        "--out",
        out_path,
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False
        )
        returncode = proc.returncode
        stdout, stderr = proc.stdout[-6000:], proc.stderr[-6000:]
        timed_out = False
    except subprocess.TimeoutExpired:
        returncode, stdout, stderr, timed_out = None, "", "", True

    record = {
        "name": name,
        "obs_mode": obs_mode,
        "num_envs": num_envs,
        "extra_env": extra_env or {},
        "subprocess_returncode": returncode,
        "subprocess_timed_out": timed_out,
        "wall_seconds": round(time.time() - started, 2),
        "stdout_tail": stdout,
        "stderr_tail": stderr,
    }
    if os.path.exists(out_path):
        with open(out_path) as fh:
            record["worker"] = json.load(fh)
    else:
        record["worker"] = None
        record["note"] = "worker produced no JSON (likely hard abort)"
    return record


def write_user_nvidia_icd(out_dir: str) -> str:
    path = os.path.join(out_dir, "nvidia_icd.json")
    with open(path, "w") as fh:
        json.dump(NVIDIA_ICD_BODY, fh, indent=2)
    return path


def ok(rec: dict | None) -> bool:
    return bool(rec and rec.get("worker") and rec["worker"].get("ok"))


# Failures that mean "our environment is built wrong", not "this cluster cannot do it".
# Job 53280 reported GATE0_SIM_UNAVAILABLE when the real cause was a torch cu130 wheel
# against a CUDA 12.8 driver; that must never be recorded as a capability finding.
SETUP_FAULT_SIGNATURES = (
    "NVIDIA driver on your system is too old",
    "_cuda_init",
    "No module named",
    "ImportError",
    "undefined symbol",
    "version `GLIBC",
)


def setup_fault(rec: dict | None) -> str | None:
    if not rec or not rec.get("worker"):
        return None
    blob = f"{rec['worker'].get('error')}\n{rec['worker'].get('traceback')}"
    for sig in SETUP_FAULT_SIGNATURES:
        if sig in blob:
            return sig
    return None


def fps(rec: dict | None) -> float | None:
    if ok(rec):
        return rec["worker"].get("fps_env_steps")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    report: dict = {
        "gate": "gate0_renderer_and_throughput",
        "design_doc": "docs/PREDICTIVE_MEMORY_JEPA_DESIGN_20260920_EN.md",
        "software_render_fps_suspect_threshold": SOFTWARE_RENDER_FPS_SUSPECT,
        "threshold_is_heuristic": True,
        "environment": environment_report(),
        "probes": {},
    }

    def add(name: str, **kw) -> dict:
        rec = probe(args.python, args.out_dir, name, **kw)
        report["probes"][name] = rec
        print(f"[gate0] {name}: ok={ok(rec)} fps={fps(rec)}", flush=True)
        return rec

    # 1) Does physics simulation work at all, without any rendering?
    state1 = add("state_n1", obs_mode="state", num_envs=1)
    state16 = add("state_n16", obs_mode="state", num_envs=16)

    # 2) Does rendering work with the cluster's default Vulkan configuration?
    rgb1 = add("rgb_n1", obs_mode="rgb", num_envs=1)
    rgb16 = add("rgb_n16", obs_mode="rgb", num_envs=16) if ok(rgb1) else None

    # 3) Documented workaround: supply an NVIDIA ICD ourselves and retry once.
    rgb1_icd = None
    if not ok(rgb1):
        icd_path = write_user_nvidia_icd(args.out_dir)
        report["user_nvidia_icd_path"] = icd_path
        rgb1_icd = add(
            "rgb_n1_user_icd",
            obs_mode="rgb",
            num_envs=1,
            extra_env={"VK_ICD_FILENAMES": icd_path},
        )
        if ok(rgb1_icd):
            rgb16 = add(
                "rgb_n16_user_icd",
                obs_mode="rgb",
                num_envs=16,
                extra_env={"VK_ICD_FILENAMES": icd_path},
            )

    rgb_works = ok(rgb1) or ok(rgb1_icd)
    rgb_fps16 = fps(rgb16)
    software_suspect = (
        rgb_works
        and rgb_fps16 is not None
        and rgb_fps16 < SOFTWARE_RENDER_FPS_SUSPECT
    )
    hard_evidence_no_gpu_render = not (
        report["environment"]["nvidia_icd_present"]
        or report["environment"]["dev_dri"]["any_render_node_accessible"]
    )

    fault = setup_fault(state1) or setup_fault(rgb1)
    if fault and not ok(state1):
        verdict = "GATE0_SETUP_FAULT"
        action = (
            f"Our environment is built wrong ({fault}); this says nothing about the "
            "cluster's rendering capability. Fix the environment and re-run."
        )
        report["setup_fault_signature"] = fault
    elif not ok(state1):
        verdict = "GATE0_SIM_UNAVAILABLE"
        action = "ManiSkill does not run here at all; stop before investing in MIKASA."
    elif not rgb_works:
        verdict = "GATE0_RGB_UNAVAILABLE_STATE_OK"
        action = (
            "Run the headroom ladder on state observations; defer all visual claims."
        )
    elif software_suspect or hard_evidence_no_gpu_render:
        verdict = "GATE0_RGB_SOFTWARE_SUSPECT"
        action = (
            "RGB initializes but looks CPU-backed; treat visual throughput as "
            "unavailable and prefer state observations for the ladder."
        )
    else:
        verdict = "GATE0_RGB_GPU_OK"
        action = "Proceed with MIKASA-Robo qualification as planned."

    report["summary"] = {
        "state_sim_ok": ok(state1),
        "state_fps_n1": fps(state1),
        "state_fps_n16": fps(state16),
        "rgb_ok": rgb_works,
        "rgb_needed_user_icd": bool(ok(rgb1_icd) and not ok(rgb1)),
        "rgb_fps_n16": rgb_fps16,
        "nvidia_icd_present": report["environment"]["nvidia_icd_present"],
        "render_node_accessible": report["environment"]["dev_dri"][
            "any_render_node_accessible"
        ],
        "software_render_suspect": software_suspect,
    }
    report["verdict"] = verdict
    report["recommended_action"] = action

    out = os.path.join(args.out_dir, "gate0_result.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\n[gate0] VERDICT: {verdict}")
    print(f"[gate0] {action}")
    print(f"[gate0] result: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
