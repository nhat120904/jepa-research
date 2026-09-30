"""Run the unmodified DINO-WM planner (gaoyuezhou/dino_wm, ICML 2025) on its official PushT checkpoint.

The only intervention: `env/__init__.py` imports the PointMaze package (d4rl, mujoco_py) just to read the constant
U_MAZE for a registration kwarg. PushT never uses it, so a stub `env.pointmaze` module with U_MAZE = None is placed in
sys.modules before the planner's imports. All planner, model, dataset and evaluator code runs as released.
Usage: plan_official.py <dino_wm_root> <hydra overrides...>  (compute node only)
"""
import os
import runpy
import sys
import types

assert os.environ.get("SLURM_JOB_ID"), "Run through sbatch"
root = os.path.abspath(sys.argv[1])
stub = types.ModuleType("env.pointmaze")
stub.U_MAZE = None
sys.modules["env.pointmaze"] = stub
sys.path.insert(0, root)
sys.argv = [os.path.join(root, "plan.py"), *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
