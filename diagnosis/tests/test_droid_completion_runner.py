import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "droid_completion_runner", ROOT / "scripts" / "11_run_droid_completion.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_build_steps_includes_core_droid_order():
    steps = runner.build_steps(allow_known_blockers=True, include_planning=False, audit_all=False)
    names = [s.name for s in steps]
    assert names == [
        "encode_droid_latents",
        "classify_droid_regimes",
        "score_droid_metrics",
        "regenerate_decision_report",
        "audit_requested_coverage",
    ]
    assert steps[-1].argv[-3:] == ("--dataset", "droid", "--allow-known-blockers")


def test_build_steps_can_include_planning_probe():
    steps = runner.build_steps(allow_known_blockers=False, include_planning=True, audit_all=False)
    names = [s.name for s in steps]
    assert "run_droid_planning_probe" in names
    assert "correlate_droid_planning" in names
    assert "--allow-known-blockers" not in steps[-1].argv


def test_build_steps_can_audit_all_requested_datasets():
    steps = runner.build_steps(allow_known_blockers=False, include_planning=False, audit_all=True)
    assert "--dataset" not in steps[-1].argv
