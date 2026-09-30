"""Run OGBench impls/main.py unchanged, but with Weights & Biases offline.

impls/utils/log_utils.setup_wandb passes mode='online' explicitly, which overrides WANDB_MODE, so the smoke run
(55324) synced to the user's W&B account. This wrapper only changes that default before main.py imports it.
Usage (from impls/): python run_offline.py <main.py flags>
"""
import runpy
import sys

sys.path.insert(0, ".")
from utils import log_utils  # noqa: E402

defaults = list(log_utils.setup_wandb.__defaults__)
defaults[-1] = "offline"                    # (entity, project, group, name, mode)
log_utils.setup_wandb.__defaults__ = tuple(defaults)
sys.argv = ["main.py", *sys.argv[1:]]
runpy.run_path("main.py", run_name="__main__")
