"""Run the boundary diagnostic (Boundary Blindness per regime) on frozen baselines.

The gate from design 2026-06-09 §3 / HANDOFF_BOUNDARY_FIX §3: prove the
contact-boundary gap is real and measurable before training any fix. No training —
same cost profile as 05_run_diagnostic.py. Writes ``results/{dataset}_boundary.csv``.

    python scripts/12_boundary_diagnostic.py --config configs/diagnostic_metaworld.yaml

All logic lives in the importable ``scripts._boundary_diagnostic`` (unit-tested in
tests/test_boundary_diagnostic.py with synthetic adapters).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._boundary_diagnostic import main  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    sys.exit(main(args.config))
