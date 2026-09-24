import unittest
from pa_wm.runtime import require_slurm
require_slurm()
import torch
from pa_wm.local_transition import PatchTransition, unroll, teacher_forced, window_batch


class AddAction(torch.nn.Module):
    def forward(self, state, action):
        return state + action[:, None, :]


class LocalTransitionTests(unittest.TestCase):
    def test_rollout_does_not_read_future(self):
        initial = torch.zeros(2, 3, 2)
        actions = torch.ones(2, 4, 2)
        pred = unroll(AddAction(), initial, actions)
        expected = torch.arange(1, 5).float()[None, :, None, None].expand_as(pred)
        torch.testing.assert_close(pred, expected)
        altered = actions.clone()
        altered[:, 2:] = 100
        torch.testing.assert_close(pred[:, :2], unroll(AddAction(), initial, altered)[:, :2])

    def test_teacher_forcing_alignment_and_shape_guard(self):
        states = torch.arange(5).float()[None, :, None, None].expand(2, 5, 3, 2)*10
        pred = teacher_forced(AddAction(), states, torch.ones(2, 4, 2))
        torch.testing.assert_close(pred, states[:, :-1]+1)
        with self.assertRaises(ValueError):
            teacher_forced(AddAction(), states[:, :-1], torch.ones(2, 4, 2))

    def test_residual_initialization_and_action_gradient(self):
        model = PatchTransition(4, 8, 32)
        state = torch.randn(2, 4, 8)
        action = torch.randn(2, 2, requires_grad=True)
        torch.testing.assert_close(model(state, action), state)
        torch.nn.init.normal_(model.output.weight, std=.02)
        out = model(state, action)
        self.assertEqual(out.shape, state.shape)
        out.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(action.grad).all()))
        self.assertGreater(action.grad.abs().sum().item(), 0.)

    def test_windows_never_enter_padding(self):
        lengths = torch.tensor([5, 8])
        states = torch.full((2, 9, 1, 2), -999.)
        actions = torch.full((2, 8, 2), -999.)
        for row, h in enumerate(lengths):
            states[row, :h+1] = torch.arange(h+1)[:, None, None]
            actions[row, :h] = 1
        generator = torch.Generator().manual_seed(2)
        initial, act, future = window_batch(states, actions, lengths, generator, 4, batch=100)
        self.assertTrue(bool((act == 1).all()))
        expected = initial[:, None] + torch.arange(1, 5)[None, :, None, None]
        torch.testing.assert_close(future, expected)


if __name__ == '__main__':
    unittest.main()
