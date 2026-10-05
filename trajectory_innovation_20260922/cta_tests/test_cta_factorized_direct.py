"""Task-only attribution baseline checks; run on a Slurm compute node."""
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.cta_factorized_direct_train import TaskOnlyBank
from ti_wm.contract import require_compute
from ti_wm.cta import LEVELS, Scorer
from ti_wm.cta_factorized_direct import FactorizedDirect
from ti_wm.cta_hit_training import HitBank, StratifiedHitPool, native_hit_loss
from ti_wm.cta_parallel import ParallelFSQWM
from ti_wm.sibling import PCA_DIM, rank_loss


def setUpModule():
    require_compute()


def small_model():
    return FactorizedDirect(m=16, width=16, predictor_layers=1, reader_layers=1, heads=2, dim=6, chunk=15)


def observed_context(rows=2):
    return {"cur": torch.randn(1, 256, 6).expand(rows, -1, -1).clone(),
            "prev": torch.randn(1, 64, 6).expand(rows, -1, -1).clone(),
            "prop": torch.randn(1, 4).expand(rows, -1).clone()}


class FactorizedControl(unittest.TestCase):
    def test_channel_matches_expected_code_and_has_no_goal_or_future_argument(self):
        torch.manual_seed(3)
        model = small_model().eval()
        ctx, actions = observed_context(), torch.randn(2, 15, 4)
        self.assertEqual(list(inspect.signature(model.encode).parameters), ["ctx", "actions"])
        self.assertEqual(list(inspect.signature(model.read).parameters), ["ctx", "code", "goal"])
        with patch.object(model.predictor, "nll", side_effect=AssertionError("NLL must not be used")):
            code = model.encode(ctx, actions)
            scores = model.read(ctx, code, torch.randn(2, 256, 6))
        self.assertEqual(tuple(code.shape), (2, 16, 3))
        self.assertEqual(tuple(scores.shape), (2,))
        for coordinate, count in enumerate(LEVELS):
            self.assertTrue((code[..., coordinate] >= -1).all())
            self.assertTrue((code[..., coordinate] <= (count - 1 - count // 2) / (count // 2)).all())
        goal1, goal2 = torch.randn(2, 256, 6), torch.randn(2, 256, 6)
        model.read(ctx, code, goal1)
        model.read(ctx, code, goal2)
        torch.testing.assert_close(model.encode(ctx, actions), code, rtol=0, atol=0)
        with self.assertRaises(TypeError):
            model.encode(ctx, actions, goal1)

    def test_action_gradient_is_cut_by_detaching_the_only_bottleneck(self):
        torch.manual_seed(5)
        model = small_model().eval()
        ctx, actions = observed_context(), torch.randn(2, 15, 4, requires_grad=True)
        code = model.encode(ctx, actions)
        model.read(ctx, code.detach(), torch.randn(2, 256, 6)).sum().backward()
        self.assertIsNone(actions.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.predictor.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.reader.parameters()))

    def test_task_losses_train_both_modules_without_code_supervision(self):
        torch.manual_seed(7)
        model = small_model().eval()
        ctx = observed_context()
        actions = torch.randn(2, 15, 4, requires_grad=True)
        goals = torch.randn(1, 256, 6).expand(2, -1, -1).clone()
        with patch.object(model.predictor, "nll", side_effect=AssertionError("Task-only model cannot use NLL")):
            scores = model(ctx, actions, goals).view(1, 2)
            loss = rank_loss(scores, torch.tensor([[0., 1.]])) + native_hit_loss(scores, torch.tensor([[False, True]]))
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(float(actions.grad.abs().sum()), 0.)
        for module in (model.predictor, model.reader):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
            self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.)

    def test_fresh_initialization_and_minimal_runtime_checkpoint(self):
        torch.manual_seed(9)
        first = small_model()
        torch.manual_seed(10)
        second = small_model()
        self.assertFalse(torch.equal(first.predictor.head.weight, second.predictor.head.weight))
        self.assertFalse(torch.equal(first.reader.head.weight, second.reader.head.weight))
        with self.assertRaisesRegex(ValueError, "standard CTA"):
            first.deployable_checkpoint({})
        model = FactorizedDirect(chunk=15)
        checkpoint = model.deployable_checkpoint({"steps": 16000, "future_supervision": False})
        self.assertEqual(set(checkpoint["stage1"]), {"reader"})
        self.assertEqual(set(checkpoint["wms"]), {"cta"})
        self.assertEqual(checkpoint["config"]["method"], "task_only_factorized_direct")
        self.assertEqual(checkpoint["config"]["m"], 16)
        self.assertEqual(checkpoint["config"]["chunk"], 15)
        predictor, reader = ParallelFSQWM(m=16, chunk=15), Scorer("code", m=16)
        predictor.load_state_dict(checkpoint["wms"]["cta"], strict=True)
        reader.load_state_dict(checkpoint["stage1"]["reader"], strict=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in predictor.parameters()) + sum(p.numel() for p in reader.parameters()))

    def test_training_bank_works_when_future_feature_files_do_not_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "cur.npy", np.zeros((2, 256, PCA_DIM), np.float32))
            np.save(path / "prev.npy", np.zeros((2, 64, PCA_DIM), np.float32))
            hits = np.zeros((2, 8), np.bool_)
            hits[0, 1] = True
            np.savez(path / "meta.npz", root=np.array([100, 101]), decision=np.zeros(2, np.int64),
                     cov8=np.tile(np.arange(8, dtype=np.float32), (2, 1)), done8=hits,
                     ctx_pos=np.zeros((2, 2, 2), np.float32), chunk=np.ones((2, 8, 15, 2), np.float32))
            bank = HitBank(TaskOnlyBank(path, "standard"), path, standard=True)
            self.assertEqual(set(bank.mm), {"cur", "prev"})
            pool = StratifiedHitPool([bank], standard_mass=1.)
            ctx, future, actions, labels, flags = pool.sample(np.random.default_rng(11), 4, "cpu")
            self.assertEqual(future, {})
            self.assertEqual(set(ctx), {"cur", "prev", "prop"})
            self.assertEqual(tuple(actions.shape), (32, 15, 4))
            self.assertEqual(tuple(flags.shape), tuple(labels.shape))
            self.assertFalse((path / "end.npy").exists())
            self.assertFalse((path / "seg.npy").exists())


if __name__ == "__main__":
    unittest.main()
