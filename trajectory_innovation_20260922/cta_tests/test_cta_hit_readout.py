"""Artifact-provenance and selected-state tests; CPU Slurm compute node only."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import cta_hit_training_readout as readout
from ti_wm.contract import require_compute


def setUpModule():
    require_compute()


class TrainingReadout(unittest.TestCase):
    def test_paired_ratio_uses_common_root_denominator(self):
        samples = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        denominator = np.array([1., 9.])
        a = np.array([1., 3.])
        b = np.array([0., 3.])
        result = readout.ratio_stat((a, denominator), samples, (b, denominator))
        self.assertAlmostEqual(result["estimate"], .1)
        self.assertGreater(result["hi"], .1)
        self.assertEqual(result["undefined_resamples"], 0)
        with self.assertRaises(ValueError):
            readout.ratio_stat((a, denominator), samples, (b, denominator + 1))

    def test_source_literal_parser_does_not_execute_code(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            marker = Path(directory) / "must_not_exist"
            source.write_text(f"from pathlib import Path\nTI = Path({directory!r})\nGOALS = TI / 'goals.npy'\nPath({str(marker)!r}).write_text('wrong')\n")
            self.assertEqual(readout.assignment_path(source, "GOALS"), Path(directory) / "goals.npy")
            self.assertFalse(marker.exists())

    def fixture(self, directory):
        directory = Path(directory)
        train = directory / "new/train"
        base = directory / "old/train"
        train.mkdir(parents=True)
        base.mkdir(parents=True)
        selection = directory / "features/selection"
        selection.mkdir(parents=True)
        geom = np.array([[0., 1.], [1., 0.], [0., 1.]], np.float32)
        hits = np.array([[False, True], [True, False], [False, False]])
        np.savez(selection / "meta.npz", root=np.array([10, 11, 12]), decision=np.zeros(3, np.int64), cov8=geom, done8=hits)
        sources = {}
        for name in ("r4std", "r4pert"):
            path = directory / "features" / name
            path.mkdir()
            np.savez(path / "meta.npz", root=np.array([100, 101]), decision=np.zeros(2, np.int64),
                     cov8=np.array([[0., 1.], [0., 1.]], np.float32),
                     done8=np.array([[False, True], [False, False]]))
            sources[name] = str(path)
        config = {"selection": {"r4dev": str(selection)}, "sources": sources,
                  "spread_frac": .75, "rank_margin": .001,
                  "continuation": {"steps": 2, "decisions": 2, "selection_limit": None, "selection_goal_images": 16,
                                   "hit_temperatures": {f"{family}_hit": 1. for family in readout.FAMILIES},
                                   "sampling": {"banks": {name: {"banks": 2} for name in sources},
                                                "expected_standard_exposure": .8, "expected_mixed_exposure": .625}}}
        (train / "config.json").write_text(json.dumps(config))
        (base / "config.json").write_text(json.dumps(config))
        goals = directory / "goals.npy"
        np.save(goals, np.zeros((16, 1), np.float32))
        v2 = f"from pathlib import Path\nTI = Path({str(directory)!r})\nGOALS = TI / 'goals.npy'\nsc = score_banks(sel, tiers, mods, wms, goals, device, amp)\n"
        for folder in (train.parent, base.parent):
            code = folder / "code/scripts"
            code.mkdir(parents=True)
            (code / "cta_train_v2.py").write_text(v2)
        (train.parent / "code/scripts/cta_hit_train.py").write_text("sc = select_scores(selection, models, mods, goals, device, amp)\n")
        initial = np.zeros((3, 2), np.float32)
        improved = np.array([[0., 2.], [2., 0.], [0., 1.]], np.float32)
        selected = {arm: (1 if arm.endswith("_hit") else 0) for arm in readout.ARMS}
        events = []
        for step in (0, 1, 2):
            values = {arm: improved if step == 1 and arm.endswith("_hit") else initial for arm in readout.ARMS}
            np.savez(train / f"selection_scores_{step}.npz", **{f"{arm}__r4dev": value for arm, value in values.items()})
            metric = {arm: {"geometry": {"retained_gap": {"ratio": 1. if step == 1 and arm.endswith("_hit") else 0.}},
                            "native": {"mixed_capture": 1. if step == 1 and arm.endswith("_hit") else .5},
                            "selection_criterion": 1.25 if step == 1 and arm.endswith("_hit") else .125}
                      for arm in readout.ARMS}
            events.append({"stage": "selection", "step": step, "metrics": metric,
                           "selected_steps": selected if step else {arm: 0 for arm in readout.ARMS}})
        np.savez(base / "selection_scores.npz", code__r4dev=initial, full__r4dev=improved)
        exposure = {"control": {"r4std": {"banks": 2, "mixed": 1}, "r4pert": {"banks": 2, "mixed": 1}},
                    "hit": {"r4std": {"banks": 3, "mixed": 2}, "r4pert": {"banks": 1, "mixed": 1}}}
        for step in (1, 2):
            event = {"stage": "train", "step": step, "exposure": exposure}
            for arm in readout.ARMS:
                event[arm] = {"loss": .6 if arm.endswith("_hit") else .5, "gradient_norm_before_clip": .1,
                              "parts": {"rank": .5} | ({"hit": .1} if arm.endswith("_hit") else {})}
            events.append(event)
        events.append({"status": "DONE", "steps": 2, "selected_steps": selected,
                       "exposure": exposure, "frozen_source_readers_unchanged": True})
        (train / "metrics.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
        (train / "selection.json").write_text(json.dumps({"step": 2, "selected_steps": selected}))
        return argparse.Namespace(train_run=train, base_run=base, out=directory / "readout", seed=0, resamples=100)

    def test_uses_selected_state_instead_of_last_state(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            readout.main(args)
            report = json.loads((args.out / "training_readout.json").read_text())
            self.assertEqual(report["selected"]["cta_hit"]["selected_step"], 1)
            self.assertEqual(report["selected"]["cta_hit"]["metrics"]["retained_geometry_gap"]["estimate"], 1.)
            self.assertEqual(report["state_curve"]["2"]["cta_hit"]["retained_geometry_gap"], 0.)
            self.assertGreater(report["source_agreement_curve"]["1"]["cta_hit"]["bank_centered_rmse"],
                               report["source_agreement_curve"]["2"]["cta_hit"]["bank_centered_rmse"])
            self.assertGreater(report["score_scale_curve"]["1"]["cta_hit"]["bank_centered_rms_over_temperature"],
                               report["score_scale_curve"]["2"]["cta_hit"]["bank_centered_rms_over_temperature"])
            self.assertGreater(report["score_scale_curve"]["1"]["cta_hit"]["mean_mixed_hit_softmax_mass"],
                               report["score_scale_curve"]["2"]["cta_hit"]["mean_mixed_hit_softmax_mass"])
            self.assertEqual(report["contrasts"]["cta_hit - cta_control"]["mixed_hit_capture"]["estimate"], .5)
            self.assertTrue(report["goal_contract"]["compatible_for_supplemental_agreement"])
            self.assertEqual(report["exposure"]["actual"]["hit"]["standard_fraction"], .75)
            self.assertIn("không xác định", (args.out / "training_readout_vi.md").read_text())
            with self.assertRaises(FileExistsError):
                readout.main(args)

    def test_stale_selected_state_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            path = args.train_run / "selection.json"
            value = json.loads(path.read_text())
            value["selected_steps"]["cta_hit"] = 2
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "selected-step records disagree"):
                readout.main(args)

    def test_wrong_score_row_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            path = args.train_run / "selection_scores_1.npz"
            np.savez(path, **{f"{arm}__r4dev": np.zeros((2, 2)) for arm in readout.ARMS})
            with self.assertRaisesRegex(ValueError, "wrong-shape scores"):
                readout.main(args)


if __name__ == "__main__":
    unittest.main()
