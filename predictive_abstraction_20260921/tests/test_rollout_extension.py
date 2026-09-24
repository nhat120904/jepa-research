import unittest
from pa_wm.runtime import require_slurm
require_slurm()
import torch
from pa_wm.local_transition import unroll, teacher_forced
from pa_wm.rollout_extension import refreshed, continuation_schedule, paired_change


class AddAction(torch.nn.Module):
    def forward(self, state, action):
        return state + action[:, None]


class RolloutExtensionTests(unittest.TestCase):
    def test_refresh_endpoints_and_alignment(self):
        states = torch.arange(6).float()[None, :, None, None].expand(2, 6, 3, 2)*10
        actions = torch.ones(2, 5, 2)
        model = AddAction()
        torch.testing.assert_close(refreshed(model, states, actions, 1), teacher_forced(model, states, actions))
        torch.testing.assert_close(refreshed(model, states, actions, 5), unroll(model, states[:, 0], actions))
        expected = torch.tensor([1, 2, 21, 22, 41.])[None, :, None, None].expand(2, 5, 3, 2)
        torch.testing.assert_close(refreshed(model, states, actions, 2), expected)

    def test_persistence_refresh_is_explicit_future_access(self):
        states = torch.arange(6).float()[None, :, None, None].expand(2, 6, 3, 2)*10
        actions = torch.ones(2, 5, 2)
        expected = torch.tensor([0, 0, 20, 20, 40.])[None, :, None, None].expand(2, 5, 3, 2)
        torch.testing.assert_close(refreshed(None, states, actions, 2), expected)
        with self.assertRaises(ValueError):
            refreshed(None, states, actions, 0)

    def test_matched_budget_axes(self):
        short, long, budget = continuation_schedule()
        self.assertEqual(short[2], long[2])
        self.assertEqual(long[1]*long[2], budget[1]*budget[2])
        self.assertEqual(short[1], budget[1])

    def test_paired_bootstrap_direction_and_identity(self):
        before = [{'prefix': str(i), 'state_mse': 2.} for i in range(4)]
        after = [{'prefix': str(i), 'state_mse': 1.} for i in range(4)]
        result = paired_change(before, after, 'state_mse')
        self.assertEqual(result['reduction'], 1.)
        self.assertEqual(result['prefix_ci'], [1., 1.])
        with self.assertRaises(ValueError):
            paired_change(before, after[::-1], 'state_mse')


if __name__ == '__main__':
    unittest.main()
