"""Server-side DROID completion runner.

Runs the remaining DROID public-baseline path in the right order:

1. encode missing `vjepa2_ac_droid` latents,
2. classify regimes,
3. score metrics into `results/droid_diagnostic.csv`,
4. regenerate the decision report,
5. run the coverage audit.

This script is intended for the 24 GB GPU server. It is dry-runnable locally
and performs lightweight prerequisite checks before any model can load.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _resource_guard import _cuda_total_memory_gib  # noqa: E402


MIN_GPU_GIB = 22.0
VJEPA_BASE_REL = Path("vjepa2_opensource") / "vjepa2_vit_giant.pth"


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]


def build_steps(*, allow_known_blockers: bool, include_planning: bool,
                audit_all: bool) -> list[Step]:
    py = sys.executable
    steps = [
        Step("encode_droid_latents", (py, "scripts/03_extract_latents.py", "--config", "configs/diagnostic_droid.yaml")),
        Step("classify_droid_regimes", (py, "scripts/04_classify_regimes.py", "--config", "configs/diagnostic_droid.yaml")),
        Step("score_droid_metrics", (py, "scripts/05_run_diagnostic.py", "--config", "configs/diagnostic_droid.yaml")),
        Step(
            "regenerate_decision_report",
            (
                py,
                "scripts/06_analyze_results.py",
                "--metaworld_csv",
                "results/metaworld_diagnostic.csv",
                "--droid_csv",
                "results/droid_diagnostic.csv",
            ),
        ),
    ]
    if include_planning:
        steps.extend([
            Step("run_droid_planning_probe", (py, "scripts/08_planning_probe.py", "--config", "configs/diagnostic_droid.yaml")),
            Step(
                "correlate_droid_planning",
                (
                    py,
                    "scripts/09_correlate_planning.py",
                    "--planning_csv",
                    "results/droid_planning.csv",
                    "--pertrans",
                    "results/droid_planning_pertrans.npz",
                    "--diagnostic_csv",
                    "results/droid_diagnostic.csv",
                ),
            ),
        ])
    audit = [py, "scripts/10_audit_coverage.py"]
    if not audit_all:
        audit.extend(["--dataset", "droid"])
    if allow_known_blockers:
        audit.append("--allow-known-blockers")
    steps.append(Step("audit_requested_coverage", tuple(audit)))
    return steps


def check_prerequisites(root: Path) -> list[str]:
    errors: list[str] = []
    gpu_gib = _cuda_total_memory_gib()
    if gpu_gib is None:
        errors.append("Could not read GPU memory via nvidia-smi.")
    elif gpu_gib < MIN_GPU_GIB:
        errors.append(f"GPU has {gpu_gib:.1f} GiB total memory; expected >= {MIN_GPU_GIB:.1f} GiB.")

    oss = os.environ.get("JEPAWM_OSSCKPT")
    if not oss:
        errors.append("JEPAWM_OSSCKPT is not set.")
    else:
        base = Path(oss) / VJEPA_BASE_REL
        if not base.exists():
            errors.append(f"Missing V-JEPA-2 base checkpoint: {base}")

    if not (root / "results/metaworld_diagnostic.csv").exists():
        errors.append("Missing results/metaworld_diagnostic.csv; run Metaworld diagnostic first.")
    if not (root / "data/precomputed_latents/droid__dino_wm_droid.h5").exists():
        errors.append("Missing DROID DINO cache; run DROID encoding for dino_wm_droid first.")

    return errors


def run_step(step: Step, root: Path, env: dict[str, str]) -> None:
    print(f"\n=== {step.name} ===", flush=True)
    print(" ".join(step.argv), flush=True)
    subprocess.run(step.argv, cwd=root, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="diagnosis repo root")
    parser.add_argument("--dry-run", action="store_true", help="print steps and exit")
    parser.add_argument("--skip-prereq-check", action="store_true")
    parser.add_argument(
        "--allow-known-blockers",
        action="store_true",
        help="waive documented external blockers in the final coverage audit",
    )
    parser.add_argument(
        "--include-planning",
        action="store_true",
        help="also run the DROID planning probe/correlation after metric scoring",
    )
    parser.add_argument(
        "--audit-all",
        action="store_true",
        help="run the final coverage audit for all requested datasets instead of DROID only",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    steps = build_steps(
        allow_known_blockers=args.allow_known_blockers,
        include_planning=args.include_planning,
        audit_all=args.audit_all,
    )

    print(f"Root: {root}")
    print("Planned steps:")
    for step in steps:
        print(f"  - {step.name}: {' '.join(step.argv)}")

    if args.dry_run:
        return 0

    if not args.skip_prereq_check:
        errors = check_prerequisites(root)
        if errors:
            print("\nPrerequisite check FAILED:")
            for err in errors:
                print(f"  - {err}")
            return 1

    env = os.environ.copy()
    env["CAI_JEPA_ALLOW_HEAVY_MODEL"] = "1"
    for step in steps:
        run_step(step, root, env)
    print("\nDROID completion runner finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
