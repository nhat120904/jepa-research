"""Compute-node guard shared by experiment entry points."""
import os


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use sbatch: simulation, tests and models belong on compute nodes.")
