"""Archived-output fallback contracts; run through CPU sbatch."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.cta_hit_fallback_check import array_check, check
from ti_wm.contract import require_compute


def setUpModule():
    require_compute()


class FallbackChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def shard(self, *, fallback=True):
        path = self.directory / "results"
        path.mkdir()
        roots = [2200, 2201]
        kinds = {"CTA": "cta", "END": "endpoint", "DIR": "direct"}
        specs = {f"{prefix}_{label}": [label, kind] for label in ("BASE", "CTRL", "HIT")
                 for prefix, kind in kinds.items()}
        arms, names = ["P0", "GEOM8", *specs], list(specs)
        selected = {f"{kind}_{suffix}": 0 if fallback and suffix == "hit" else 1200
                    for kind in kinds.values() for suffix in ("control", "hit")}
        checkpoints = {"BASE": {"sha256": "base", "config": {"chunk": 15}}}
        for label in ("CTRL", "HIT"):
            checkpoints[label] = {"sha256": f"different-whole-file-{label}",
                                  "config": {"continuation": {"base_sha256": "base", "source_reader_frozen": True,
                                                               "selected_steps": selected}}}
        report = {"schema": "cta_hit_closed_v1", "status": "DONE", "roots": roots,
                  "arms": arms, "log_scorers": names, "checkpoint_specs": specs, "checkpoints": checkpoints,
                  "n_exec": 15, "draw": 8, "bank": "policy", "proposal_mode": "canonical",
                  "proposal_microbatch": 8, "score_root_microbatch": 1, "goal_images": 16,
                  "hashes": {"same": "release"}, "backend": {"same": "backend"}, "max_decisions": None,
                  "learned_input_contract": "C,A before simulation; reader C,S,g", "clone_method": "deepcopy",
                  "qualification": {"status": "PASS"}, "timing": {}}
        episodes = []
        for arm in arms:
            fields = {"root": np.repeat(roots, 20), "decision": np.tile(np.arange(20), 2),
                      "t": np.tile(np.arange(20) * 15, 2), "chosen": np.full(40, 0 if arm == "P0" else 7, np.int64),
                      "cov": np.tile(np.arange(8, dtype=np.float64)[None] / 20, (40, 1)),
                      "geom": np.tile(np.arange(8, dtype=np.float64)[None] / 20, (40, 1)),
                      "native_hit": np.zeros((40, 8), bool)}
            fields.update({f"score_{name}": np.tile(np.arange(8, dtype=np.float32)[None], (40, 1)) for name in names})
            np.savez(path / f"log_{arm}.npz", **fields)
            episodes.extend({"arm": arm, "root": root, "success": False, "steps": 300, "score": .4 / .95,
                             "max_coverage": .4, "reset_excluded": True, "terminated": True,
                             "truncated_by_max_decisions": False} for root in roots)
        (path / "closed_report.json").write_text(json.dumps(report))
        (path / "episodes.jsonl").write_text("\n".join(json.dumps(item) for item in episodes) + "\n")
        return path

    def mutate(self, path, arm, field, value):
        filename = path / f"log_{arm}.npz"
        with np.load(filename) as archive:
            log = {name: archive[name] for name in archive.files}
        if log[field].ndim == 2:
            log[field][0, 0] = value
        else:
            log[field][0] = value
        np.savez(filename, **log)

    def run_check(self, path):
        with contextlib.redirect_stdout(io.StringIO()):
            return check([path], self.directory / "fallback.json")

    def test_all_three_fallbacks_pass_with_different_whole_file_hashes(self):
        report = self.run_check(self.shard())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual({item["alias"] for item in report["relationships"]}, {"CTA_HIT", "END_HIT", "DIR_HIT"})
        self.assertFalse(report["weight_equality_independently_attested"])
        for item in report["relationships"]:
            self.assertTrue(item["fallback_reused_base"])
            self.assertNotEqual(item["alias_whole_checkpoint_sha256"], item["baseline_whole_checkpoint_sha256"])
            self.assertEqual(len(item["cross_scores"]), 11)
            self.assertTrue(item["episodes"]["pass"])

    def test_no_step_zero_states_are_not_applicable(self):
        report = self.run_check(self.shard(fallback=False))
        self.assertEqual(report["status"], "NOT_APPLICABLE")
        self.assertEqual(report["relationships"], [])

    def test_control_step_zero_states_are_checked_too(self):
        path = self.shard()
        file = path / "closed_report.json"
        report = json.loads(file.read_text())
        selected = report["checkpoints"]["CTRL"]["config"]["continuation"]["selected_steps"]
        for kind in ("cta", "endpoint", "direct"):
            selected[f"{kind}_control"] = 0
        file.write_text(json.dumps(report))
        saved = self.run_check(path)
        self.assertEqual(saved["status"], "PASS")
        self.assertEqual(len(saved["relationships"]), 6)

    def test_cross_scores_must_match_on_other_arms_too(self):
        path = self.shard()
        self.mutate(path, "DIR_CTRL", "score_CTA_HIT", .25)
        with self.assertRaisesRegex(RuntimeError, "equivalence failed"):
            self.run_check(path)
        result = json.loads((self.directory / "fallback.json").read_text())
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["relationships"][0]["cross_scores"]["DIR_CTRL"]["pass"])

    def test_coverage_ulps_pass_but_larger_trajectory_discrepancies_fail(self):
        self.assertTrue(array_check([.3], [.3 + 3e-16], tolerance=True)["pass"])
        self.assertFalse(array_check([.3], [.3 + 1e-8], tolerance=True)["pass"])
        path = self.shard()
        self.mutate(path, "CTA_HIT", "cov", 1e-8)
        with self.assertRaisesRegex(RuntimeError, "equivalence failed"):
            self.run_check(path)

    def test_native_hit_boolean_changes_fail_without_subtraction_errors(self):
        path = self.shard()
        self.mutate(path, "END_HIT", "native_hit", True)
        with self.assertRaisesRegex(RuntimeError, "equivalence failed"):
            self.run_check(path)
        result = json.loads((self.directory / "fallback.json").read_text())
        endpoint = next(item for item in result["relationships"] if item["family"] == "endpoint")
        self.assertFalse(endpoint["trajectory"]["native_hit"]["pass"])
        self.assertEqual(endpoint["trajectory"]["native_hit"]["max_absolute_difference"], 1.)

    def test_wrong_base_hash_is_a_provenance_failure(self):
        path = self.shard()
        file = path / "closed_report.json"
        report = json.loads(file.read_text())
        report["checkpoints"]["HIT"]["config"]["continuation"]["base_sha256"] = "wrong-base"
        file.write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, "equivalence failed"):
            self.run_check(path)
        saved = json.loads((self.directory / "fallback.json").read_text())
        self.assertFalse(saved["relationships"][0]["provenance"]["base_sha256_matches"])

    def test_episode_change_fails_even_when_candidate_logs_agree(self):
        path = self.shard()
        file = path / "episodes.jsonl"
        episodes = [json.loads(line) for line in file.read_text().splitlines()]
        next(item for item in episodes if item["arm"] == "DIR_HIT")["steps"] = 299
        file.write_text("\n".join(json.dumps(item) for item in episodes) + "\n")
        with self.assertRaisesRegex(RuntimeError, "equivalence failed"):
            self.run_check(path)
        saved = json.loads((self.directory / "fallback.json").read_text())
        direct = next(item for item in saved["relationships"] if item["family"] == "direct")
        self.assertEqual(direct["episodes"]["failed_roots"], [2200])

    def test_bitwise_scores_distinguish_signed_zero(self):
        result = array_check(np.array([0.], np.float32), np.array([-0.], np.float32))
        self.assertFalse(result["pass"])
        self.assertEqual(result["max_absolute_difference"], 0.)


if __name__ == "__main__":
    unittest.main()
