"""Matched reader objectives/frozen-code checks; CPU Slurm compute node only."""
import copy
import unittest
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from scripts.cta_reader_calibrate import (check_frozen, frozen_codes, predicted_selection,
                                           random_state, reader_objective, restore_random_state)
from ti_wm.contract import require_compute


def setUpModule():
    require_compute()


class TinyReader(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., .3, -.2]))
        self.drop = nn.Dropout(dropout)

    def forward(self, ctx, code, goal):
        return (self.drop(code[:, 0]) * self.weight).sum(-1)


class TinySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.register_buffer("marker", torch.tensor(7.))

    def forward(self, ctx, future):
        return self.scale * future["code"]


class TinyWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.5))

    def forward(self, ctx, actions):
        return self.scale * actions.mean(1)[:, None], ()


def codes():
    return (torch.tensor([[[0., 0., 0.]], [[1., 0., 0.]]]),
            torch.tensor([[[.25, 0., 0.]], [[.5, 0., 0.]]]))


class ReaderCalibration(unittest.TestCase):
    def test_native_objective_gives_gradients_in_both_domains_at_flat_geometry(self):
        reader = TinyReader()
        source, predicted = (value.requires_grad_() for value in codes())
        labels, hits = torch.zeros(1, 2), torch.tensor([[False, True]])
        control, _, _, _ = reader_objective(reader, {}, source, predicted, None, labels, hits, 1., 0., nullcontext)
        self.assertEqual(float(control), 0.)
        hit, parts, _, _ = reader_objective(reader, {}, source, predicted, None, labels, hits, 1., 1., nullcontext)
        self.assertGreater(float(hit), 0.)
        hit.backward()
        for value in (source, predicted):
            self.assertGreater(float(value.grad[0, 0, 0]), 0.)
            self.assertLess(float(value.grad[1, 0, 0]), 0.)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(parts["source_hit"]), 0.)
        self.assertGreater(float(parts["predicted_hit"]), 0.)

    def test_all_success_or_failure_does_not_add_hit_preference(self):
        reader = TinyReader()
        source, predicted = codes()
        labels = torch.tensor([[0., 1.]])
        for hits in (torch.tensor([[True, True]]), torch.tensor([[False, False]])):
            control, _, _, _ = reader_objective(reader, {}, source, predicted, None, labels, hits, 1., 0., nullcontext)
            hit, parts, _, _ = reader_objective(reader, {}, source, predicted, None, labels, hits, 1., 1., nullcontext)
            torch.testing.assert_close(control, hit)
            self.assertEqual(float(parts["weighted_hit_average"]), 0.)

    def test_no_grad_codes_allow_reader_backward_and_leave_frozen_modules_exact(self):
        encoder, wm = TinySource().eval().requires_grad_(False), TinyWM().eval().requires_grad_(False)
        base = {"stage1": {"enc": {key: value.clone() for key, value in encoder.state_dict().items()}},
                "wms": {"cta": {key: value.clone() for key, value in wm.state_dict().items()}}}
        hard, predicted = frozen_codes(encoder, wm, {}, {"code": codes()[0]}, torch.randn(2, 3, 3), nullcontext)
        self.assertFalse(hard.requires_grad or predicted.requires_grad)
        self.assertFalse(hard.is_inference() or predicted.is_inference())
        reader = TinyReader()
        optimizer = torch.optim.SGD(reader.parameters(), lr=.1)
        loss, _, _, _ = reader_objective(reader, {}, hard, predicted, None, torch.tensor([[0., 1.]]),
                                         torch.tensor([[False, True]]), 1., 1., nullcontext)
        loss.backward()
        optimizer.step()
        self.assertTrue(check_frozen(encoder, wm, base)["encoder_wm_have_no_gradients"])
        with torch.no_grad():
            wm.scale.add_(.1)
        with self.assertRaisesRegex(RuntimeError, "wm:weights_or_buffers"):
            check_frozen(encoder, wm, base)

    def test_matching_dropout_state_preserves_identical_dense_branches(self):
        torch.manual_seed(17)
        control = TinyReader(dropout=.5).train()
        hit = copy.deepcopy(control)
        source, predicted = codes()
        labels, hits = torch.tensor([[0., 1.]]), torch.tensor([[False, True]])
        before = random_state()
        _, control_parts, control_source, control_predicted = reader_objective(control, {}, source, predicted, None,
                                                                             labels, hits, 1., 0., nullcontext)
        after = random_state()
        expected_next = torch.rand(5)
        restore_random_state(before)
        _, hit_parts, hit_source, hit_predicted = reader_objective(hit, {}, source, predicted, None,
                                                                 labels, hits, 1., 1., nullcontext)
        torch.testing.assert_close(control_source, hit_source, rtol=0, atol=0)
        torch.testing.assert_close(control_predicted, hit_predicted, rtol=0, atol=0)
        torch.testing.assert_close(control_parts["dense_average"], hit_parts["dense_average"], rtol=0, atol=0)
        restore_random_state(after)
        torch.testing.assert_close(torch.rand(5), expected_next, rtol=0, atol=0)

    def test_selection_has_only_predicted_code_and_rejects_future_features(self):
        class ObservedBank:
            name, n = "selection", 2
            forbidden_future = False
            def batch(self, indices, device):
                actions = torch.zeros(8, 1, 3)
                actions[:, 0, 0] = torch.arange(8) + int(indices[0])
                return {"cur": torch.zeros(8, 1, 3)}, ({"future": 1} if self.forbidden_future else {}), actions, torch.zeros(1, 8)
        bank = ObservedBank()
        wm = TinyWM().eval().requires_grad_(False)
        readers = {"control": TinyReader(), "hit": TinyReader()}
        result = predicted_selection(readers, wm, [bank], torch.zeros(16, 1, 3), "cpu", nullcontext)
        np.testing.assert_allclose(result["control"]["selection"][0], np.arange(8) * .5)
        np.testing.assert_array_equal(result["control"]["selection"], result["hit"]["selection"])
        bank.forbidden_future = True
        with self.assertRaisesRegex(RuntimeError, "future features"):
            predicted_selection(readers, wm, [bank], torch.zeros(16, 1, 3), "cpu", nullcontext)


if __name__ == "__main__":
    unittest.main()
