"""Safety wrapper: cap PyTorch's CUDA allocator so it can NEVER spill into
shared system RAM (the cause of the full-system freeze on this 16 GB box).

Sets a hard per-process VRAM fraction, then execs the requested script's main.
Usage:
    python scripts/_safe_launch.py <fraction> <script.py> [script args...]
"""
import runpy
import sys

import torch

frac = float(sys.argv[1])
script = sys.argv[2]
script_args = sys.argv[3:]

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[safe_launch] VRAM cap = {frac:.2f} x {total_gb:.1f}GB "
          f"= {frac*total_gb:.1f}GB; allocations beyond this OOM cleanly "
          f"instead of spilling to system RAM.", flush=True)

# Hand control to the target script as if invoked directly.
sys.argv = [script] + script_args
runpy.run_path(script, run_name="__main__")
