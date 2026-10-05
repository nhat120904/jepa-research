"""Actual shard loading/aggregation regressions; run through CPU sbatch."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.cta_hit_aggregate import aggregate, load_runs
from scripts.cta_hit_closed import compare_logs
from ti_wm.contract import require_compute


def setUpModule():
    require_compute()


class AggregateContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def shard(self, name="shard", first=2200):
        path = self.directory / name
        path.mkdir()
        roots = [first, first + 1]
        report = {"schema": "cta_hit_closed_v1", "status": "DONE", "roots": roots,
                  "arms": ["P0", "CTA_HIT"], "log_scorers": ["CTA_HIT"],
                  "checkpoint_specs": {"CTA_HIT": ["HIT", "cta"]},
                  "checkpoints": {"HIT": {"sha256": "fixture"}}, "n_exec": 15,
                  "draw": 8, "bank": "policy", "proposal_mode": "canonical",
                  "proposal_microbatch": 8, "score_root_microbatch": 1, "goal_images": 16,
                  "hashes": {"smoke": "same-smoke", "visual_model_state": "same-visual"},
                  "backend": {"dependencies": {"pymunk": "6.6.0"}},
                  "max_decisions": None, "learned_input_contract": "C,A before simulation; reader C,S,g",
                  "clone_method": "deepcopy", "qualification": {"status": "PASS"},
                  "timing": {}}
        episodes = []
        for arm in report["arms"]:
            episodes += [{"arm": arm, "root": root, "success": False, "steps": 300,
                          "score": 0., "max_coverage": 0., "reset_excluded": True,
                          "terminated": True, "truncated_by_max_decisions": False} for root in roots]
            np.savez(path / f"log_{arm}.npz", root=np.repeat(roots, 20),
                     decision=np.tile(np.arange(20), 2), t=np.tile(np.arange(20) * 15, 2),
                     chosen=np.zeros(40, dtype=np.int64), cov=np.zeros((40, 8)),
                     geom=np.zeros((40, 8)), native_hit=np.zeros((40, 8), dtype=bool),
                     score_CTA_HIT=np.zeros((40, 8), dtype=np.float32))
            report["timing"][arm] = {"proposal_seconds": 1., "simulator_seconds": 1.,
                                      "shared_context_seconds": 1., "diagnostic_wall_seconds": 4.,
                                      "proposal_root_banks": 40, "context_calls": 40,
                                      "scorer_seconds": {"CTA_HIT": 1.}, "scorer_calls": {"CTA_HIT": 40}}
        (path / "closed_report.json").write_text(json.dumps(report))
        (path / "episodes.jsonl").write_text("\n".join(json.dumps(ep) for ep in episodes) + "\n")
        return path

    def mutate_log(self, path, key, value):
        log_path = path / "log_P0.npz"
        with np.load(log_path) as blob:
            data = {field: blob[field] for field in blob.files}
        data[key] = np.full(data[key].shape, value)
        np.savez(log_path, **data)

    def test_qualification_compares_boolean_native_hits_without_subtraction_errors(self):
        left = {"root": np.array([2200]), "native_hit": np.array([[True, False]], dtype=bool)}
        right = {key: value.copy() for key, value in left.items()}
        result = compare_logs(left, right, 2200)
        self.assertTrue(result["native_hit"]["contract_pass"])
        self.assertEqual(result["native_hit"]["max_abs"], 0.)
        right["native_hit"][0, 0] = False
        result = compare_logs(left, right, 2200)
        self.assertFalse(result["native_hit"]["contract_pass"])
        self.assertEqual(result["native_hit"]["max_abs"], 1.)

    def test_zero_headroom_produces_null_and_readable_report(self):
        path = self.shard()
        out = self.directory / "aggregate"
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate(SimpleNamespace(closed_run=[path], out=out, expect_roots=[2200, 2201],
                                      pairs="CTA_HIT-P0"))
        def reject_constant(value):
            self.fail(f"Invalid JSON constant: {value}")
        summary = json.loads((out / "summary.json").read_text(), parse_constant=reject_constant)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["arms"]["CTA_HIT"]["successes"], 0)
        self.assertIsNone(summary["onpolicy"]["P0"]["geom"]["own_choices"]["ratio"])
        self.assertIsNone(summary["onpolicy"]["P0"]["native_hit"]["scorers"]["CTA_HIT"]["capture"]["ratio"])
        self.assertIn("n/a", (out / "summary.md").read_text())

    def test_invalid_candidate_indices_and_native_labels_are_rejected(self):
        for index, (key, value) in enumerate((("chosen", -1), ("chosen", 1.5), ("native_hit", .5))):
            with self.subTest(key=key, value=value):
                path = self.shard(f"bad_{index}")
                self.mutate_log(path, key, value)
                with self.assertRaisesRegex(ValueError, "chosen candidate|native_hit"):
                    load_runs([path])

    def test_nonboolean_episode_success_is_rejected(self):
        path = self.shard()
        file = path / "episodes.jsonl"
        episodes = [json.loads(line) for line in file.read_text().splitlines()]
        episodes[0]["success"] = "false"
        file.write_text("\n".join(json.dumps(ep) for ep in episodes) + "\n")
        with self.assertRaisesRegex(ValueError, "Nonboolean episode"):
            load_runs([path])

    def test_different_clone_or_dependency_identity_cannot_be_pooled(self):
        left = self.shard("left", 2200)
        for key in ("clone_method", "backend", "hashes"):
            with self.subTest(key=key):
                right = self.shard(f"right_{key}", 2202)
                file = right / "closed_report.json"
                report = json.loads(file.read_text())
                report[key] = "different" if key == "clone_method" else {"changed": True}
                file.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    load_runs([left, right])

    def test_partial_and_truncated_runs_cannot_enter_success_results(self):
        for key, value in (("status", "RUNNING"), ("max_decisions", 2)):
            with self.subTest(key=key):
                path = self.shard(key)
                file = path / "closed_report.json"
                report = json.loads(file.read_text())
                report[key] = value
                file.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    load_runs([path])


if __name__ == "__main__":
    unittest.main()
