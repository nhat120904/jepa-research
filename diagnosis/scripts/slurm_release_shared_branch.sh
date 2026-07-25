#!/usr/bin/env bash
#SBATCH --job-name=jepa_release_sb
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/release_shared_branch_%j.out
# Control-plane job: Stage B is held only while Stage A has GPU priority.
set -euo pipefail
scontrol release 27994
echo "Released shared-branch array 27994 after Stage-A completion: $(date)"
