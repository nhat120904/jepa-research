"""CPU-only preparation. Download weights but NEVER instantiate a policy here."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import subprocess
import sys
import traceback
from pathlib import Path

from ti_wm.contract import (
    LEROBOT_REVISION, POLICY_REPO, compatible_config, native_action_slice, require_compute,
)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(run):
    require_compute()
    report = {"status": "PREPARING", "model_instantiated": False, "source_revision": LEROBOT_REVISION}
    try:
        # Importing classes is a dependency check, not model loading.
        import draccus
        import gymnasium as gym
        import gym_pusht
        import numpy as np
        from huggingface_hub import HfApi, snapshot_download
        from lerobot.common.envs.configs import PushtEnv
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        source = run / "upstream"
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        assert revision == LEROBOT_REVISION, revision
        sha = HfApi().model_info(POLICY_REPO).sha
        checkpoint = Path(snapshot_download(
            POLICY_REPO, revision=sha, local_dir=run / "checkpoint",
            allow_patterns=["config.json", "model.safetensors", "README.md", "eval_info.json"],
        ))
        report["checkpoint_revision"] = sha
        report["checkpoint_hashes"] = {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}
        raw = json.loads((checkpoint / "config.json").read_text())
        config, removed = compatible_config(raw, {f.name for f in dataclasses.fields(DiffusionConfig)})
        decoded = draccus.decode(DiffusionConfig, config)
        decoded.validate_features()
        (run / "policy_config_compatible.json").write_text(json.dumps(config, indent=2))
        report["runtime_fields_removed"] = removed
        report["native_action_slice"] = native_action_slice(raw)
        report["policy_config"] = raw
        report["policy_class"] = f"{DiffusionPolicy.__module__}.{DiffusionPolicy.__name__}"
        env_config = PushtEnv()
        report["env_kwargs"] = env_config.gym_kwargs
        report["env_id"] = f"gym_pusht/{env_config.task}"

        def replay(seed):
            env = gym.make(report["env_id"], **env_config.gym_kwargs)
            try:
                obs, info = env.reset(seed=seed)
                traces = []
                images = []
                # Fixed non-policy actions test deterministic reset/action replay only.
                for t in range(32):
                    images.append(obs["pixels"].copy())
                    traces.append(np.asarray(obs["agent_pos"]).copy())
                    obs, reward, terminated, truncated, info = env.step(
                        np.asarray([220.0 + t * 2, 256.0], dtype=np.float32)
                    )
                    traces.append(np.asarray([reward, terminated, truncated], dtype=float))
                    if terminated or truncated:
                        break
                images.append(obs["pixels"].copy())
                traces.append(np.asarray(obs["agent_pos"]).copy())
                return images, traces, str(inspect.getfile(type(env.unwrapped)))
            finally:
                env.close()

        checks = []
        for seed in (900000, 900001, 900002, 900003):
            first, states, env_path = replay(seed)
            second, repeated, _ = replay(seed)
            equal = len(first) == len(second) and len(states) == len(repeated)
            equal = equal and all(np.array_equal(a, b) for a, b in zip(first, second))
            equal = equal and all(np.array_equal(a, b) for a, b in zip(states, repeated))
            checks.append({"seed": seed, "frames": len(first), "exact_replay": equal,
                           "pixel_shape": list(first[0].shape)})
            assert equal, f"Action replay mismatch seed {seed}"
        report["replay_checks"] = checks
        report["env_source"] = {"path": env_path, "sha256": digest(env_path)}
        report["versions"] = {
            name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "numpy", "gymnasium", "gym-pusht", "pymunk", "pygame",
             "diffusers", "draccus", "huggingface-hub", "safetensors")
        }
        report["status"] = "READY_FOR_GPU_SMOKE_NOT_QUALIFIED"
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        report["status"] = "PREPARATION_FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (run / "preparation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run)
